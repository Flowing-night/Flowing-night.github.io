# -*- coding: utf-8 -*-
"""
第六讲课堂示例（IC 加权多因子改动版）
====================================
在原始流程的基础上，加入 Alpha#001 / Alpha#101 / Alpha#006 三个技术类因子，
并按各因子的历史 Rank-IC 均值做权重，合成一个多因子总分，桥接到 lab05 选股与归因。

多因子打分的权重规则（逐月滚动，只用当期及之前的信息，无未来函数）：
    w_f = sign(平均Rank-IC) * |平均Rank-IC| / Σ|平均Rank-IC|
  - 用符号保证因子方向正确（IC 为负的因子自动反向）；
  - 只用 |Rank-IC| 排名靠前的因子（由 KEEP_TOP 控制，默认取前 5 个）。

与原始版本的差异（按 # %% 标记）：
  A. %%2 ：raw 新增 Alpha001 / Alpha101 / Alpha006；
  B. %%4 ：summary 增加 Rank-IC 均值，并据此计算 IC 权重向量；
  C. %%7 ：新增 build_multifactor_score()，生成多因子合成分数矩阵；
  D. %%8 ：把多因子权重、合成分数、各因子原始矩阵、summary 存成 pkl 桥接文件。

按 # %% 在 Spyder 中逐段运行，或在项目根目录运行：
    python code/lab06/lab06_multifactor.py
"""
# %% 1. 数据载入：行是交易日，列是股票
from pathlib import Path
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from data_io import load_data, load_benchmark
from utils.FactorTest import factor_test
from utils.data_standardize import preprocess
from utils.utils import industry_neutral, calc_nav, plot_equity
from utils.stockselection import stock_selection
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

daily = {}
for name in ['close', 'adj_factor', 'pb', 'total_mv', 'vol', 'amount', 'is_st', 'open', 'high', 'low']:
    try:
        daily[name] = load_data(name, '2019-01-01', '2025-12-31')
    except Exception as e:
        print(f'[跳过] {name} 不可载入：{e}')
        daily[name] = pd.DataFrame()


# %% 2. 构建基础因子与 Alpha 类因子，取月末观测
price = daily['close'].mul(daily['adj_factor'], fill_value=np.nan)

raw = {
    'BP': 1 / daily['pb'].where(daily['pb'] > 0),
    'MV': np.log(daily['total_mv'].where(daily['total_mv'] > 0)),
    'Momentum': price / price.shift(21) - 1,
}

# —— 新增：Alpha 类技术因子 ——
if not daily['open'].empty and not daily['high'].empty and not daily['low'].empty:
    raw['Alpha101'] = (daily['close'] - daily['open']) / (daily['high'] - daily['low'] + 1e-4)
else:
    print('[跳过] Alpha101：open/high/low 数据缺失')

def _ts_rank_corr(a, b, d):
    out = pd.DataFrame(np.nan, index=a.index, columns=a.columns)
    ra, rb = a.rank(axis=1, pct=True), b.rank(axis=1, pct=True)
    for t in range(d, len(a)):
        wa, wb = ra.iloc[t - d + 1:t + 1], rb.iloc[t - d + 1:t + 1]
        ma, mb = wa.mean(axis=0), wb.mean(axis=0)
        num = ((wa - ma) * (wb - mb)).sum(axis=0)
        den = np.sqrt(((wa - ma) ** 2).sum(axis=0) * ((wb - mb) ** 2).sum(axis=0))
        out.iloc[t] = num / den.replace(0, np.nan)
    return out
if not daily['open'].empty:
    raw['Alpha006'] = -_ts_rank_corr(daily['open'], daily['vol'], 10)
else:
    print('[跳过] Alpha006：open 数据缺失')
raw['Alpha001'] = -(price - price.shift(5)) / (price.shift(5) + 1e-6)

# 月末交易日（研究区间）
dates = pd.DatetimeIndex(pd.Series(price.index).groupby(price.index.to_period('M')).max())
dates = dates[(dates >= '2019-12-01') & (dates <= '2025-12-31')]

# 建立月末股票池（沿用原逻辑）
stock_info = pd.read_csv(HERE.parent.parent / 'Existing directory' / 'data' / 'sse_stock_basic.csv')
stock_info['code'] = stock_info['code'].astype(str).str.extract(r'(\d+)')[0].str.zfill(6) + '.SH'
ipo_date = pd.to_datetime(stock_info.set_index('code')['ipoDate'])
panel = pd.concat({name: daily[name].reindex(dates).stack(future_stack=True)
                   for name in ['is_st', 'amount', 'vol']}, axis=1)
panel = panel.rename_axis(['date', 'code']).reset_index()
panel['listed_days'] = (panel['date'] - panel['code'].map(ipo_date)).dt.days
panel['tradable'] = (panel['vol'] > 0).astype(int)

filtered_pool = panel[
    (panel['is_st'] == 0)
    & (panel['listed_days'] >= 120)
    & (panel['tradable'] == 1)
    & (panel['amount'] >= 10_000)   # 千元，至少1000万元
].copy()
valid = (filtered_pool.assign(in_pool=True)
         .pivot(index='date', columns='code', values='in_pool')
         .reindex(index=dates, columns=price.columns).eq(True))

raw = {name: x.reindex(dates).where(valid).replace([np.inf, -np.inf], np.nan)
       for name, x in raw.items() if (not x.empty) and x.notna().sum().sum() > 0}

# 下一期收益：factor_test 用第 t 期因子配第 t+1 期收益，此处不再提前 shift
monthly_price = price.reindex(dates)
returns = monthly_price.pct_change(fill_method=None)


# %% 3. 因子预处理：MAD 缩尾后作 Z-score
factors = {name: preprocess(x) for name, x in raw.items()}


# %% 4. IC、五档与多空检验，并据此计算 IC 权重
results = {}
summary_rows = []
ic_series_store = {}      # 保存每个因子的逐期 Rank-IC，用于滚动权重

for name, x in factors.items():
    result = factor_test(x, returns, NumOfGroup=5,
                         DispFlag=False, PlotFlag=False, comsn=0, name=name)
    results[name] = result
    ic = pd.Series(result['Rank_IC'][1:], index=dates[1:]).dropna()
    ic_series_store[name] = ic
    for metric, key in [('Pearson IC', 'Normal_IC'), ('Rank IC', 'Rank_IC')]:
        s = pd.Series(result[key][1:], index=dates[1:]).dropna()
        summary_rows.append({'因子': name, '指标': metric,
            '平均IC': s.mean(), 'IC标准差': s.std(ddof=1),
            'ICIR': s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan,
            'p+': (s > 0).mean()})

summary = pd.DataFrame(summary_rows)
rank_ic = (summary[summary['指标'] == 'Rank IC']
           .set_index('因子')['平均IC'].reindex(list(factors)))
print('各因子 Rank-IC 表现：')
print(summary.round(4).to_string(index=False))


# %% 5. 比较 BP 行业中性化前后的表现（沿用原逻辑）
neutralizer = industry_neutral('l1_code')
bp_src = factors.get('BP')
if bp_src is not None and bp_src.notna().sum().sum() > 0:
    bp_neutral = neutralizer(bp_src, standardize=False)
    bp_before = factors['BP'].where(bp_neutral.notna())
else:
    print('[跳过] BP 行业中性化：BP 因子为空')
    bp_neutral = None
    bp_before = None

industry_rows = []
for name, x in {'BP_before': bp_before, 'BP_neutral': bp_neutral}.items():
    if x is None:
        continue
    result = factor_test(x, returns, NumOfGroup=5,
                         DispFlag=False, PlotFlag=False, comsn=0, name=name)
    results[name] = result
    group_returns = pd.DataFrame(result['GroupRtn'][1:], index=dates[1:])
    long_short = group_returns.iloc[:, -1] - group_returns.iloc[:, 0]
    for metric, key in [('Pearson IC', 'Normal_IC'), ('Rank IC', 'Rank_IC')]:
        s = pd.Series(result[key][1:]).dropna()
        industry_rows.append({'因子': name, '指标': metric,
            '平均IC': s.mean(), 'IC标准差': s.std(ddof=1),
            'ICIR': s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan,
            'p+': (s > 0).mean(),
            '多空月均收益': long_short.mean(),
            '多空月收益标准差': long_short.std(ddof=1),
            '多空期末净值': (1 + long_short).prod()})
if industry_rows:
    industry_summary = pd.DataFrame(industry_rows)
    print('\n行业中性化对比：')
    print(industry_summary.round(4).to_string(index=False))


# %% 6. 从行业中性化 BP 的高值组选股，等权计算组合净值
if bp_neutral is not None:
    # 使用 G10（最高组），安全截取前30
    codes = results['BP_neutral']['Code_G10']
    holdings = codes[:, :min(30, codes.shape[1])]
    weights_mat = stock_selection(returns.columns, holdings, dates)
    outputs = calc_nav(returns.fillna(0), weights_mat)
    # 取 outputs['nav'] 的索引对齐基准
    benchmark = load_benchmark('2019-12-01', '2025-12-31')['close'].reindex(outputs['nav'].index).ffill()
    plot_equity(outputs['nav'], benchmark)
    plt.show()
else:
    print('[跳过] 中性化 BP 组合净值（BP 因子为空）')

# %% 7. 新增：IC 加权合成多因子分数（滚动，无未来函数）
# —— 权重：w_f = sign(meanRankIC) * |meanRankIC| / Σ|meanRankIC|，仅保留 |Rank-IC| 最大的 KEEP_TOP 个 ——
# 权重 = sign(平均Rank-IC) * |平均Rank-IC| / Σ|平均Rank-IC|
# 用绝对 IC 归一化：保证权重和为 1，同时用符号保留每个因子的正确方向。
KEEP_TOP = 5
ic_abs = rank_ic.abs().sort_values(ascending=False)
keep = ic_abs.index[ic_abs.notna()][:KEEP_TOP]

abs_sum = ic_abs.reindex(keep).sum()
signed = rank_ic.reindex(keep).map(np.sign) * ic_abs.reindex(keep)
weights = signed / abs_sum if abs_sum > 0 else signed   # 绝对IC归一化，保证 Σw=1
weights = weights.dropna()
print('\nIC 加权多因子权重：')
print(weights.round(4).to_string())

def build_multifactor_score(factors, weights, dates):
    """按当期及之前的信息滚动计算 IC 权重，合成多因子分数。返回 T×N DataFrame。"""
    cols = list(factors.values())[0].columns
    score = pd.DataFrame(0.0, index=dates, columns=cols)
    cum_ic = {}
    for t in dates:
        for f in weights.index:
            s = ic_series_store[f]
            s = s[s.index <= t]
            cum_ic[f] = s.mean() if len(s) > 0 else 0.0
        w = pd.Series(cum_ic)
        # 与最终权重同口径：sign(IC) * |IC| / Σ|IC|，保证每月权重和为 1
        w = w.map(np.sign) * w.abs()
        if w.abs().sum() > 0:
            w = w / w.abs().sum()
        for f in w.index:
            score.loc[t] += w[f] * factors[f].loc[t]
    return score

multifactor = build_multifactor_score(factors, weights, dates)
print(f'多因子合成分数矩阵：{multifactor.shape}，每月非空均值 '
      f'{multifactor.notna().sum(axis=1).mean():.0f} 只')


# %% 8. 新增：把多因子结果桥接到 lab05 —— 存成 pkl
import pickle

BRIDGE_DIR = HERE.parent.parent / 'output' / 'lab06'
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
BRIDGE_PATH = BRIDGE_DIR / 'multifactor.pkl'

with open(BRIDGE_PATH, 'wb') as f:
    pickle.dump({
        'weights': weights.to_dict(),          # 各因子 IC 权重（final）
        'multifactor': multifactor,            # 多因子合成分数 T×N
        'factors': factors,                   # 各因子预处理后的截面矩阵
        'returns': returns,                   # 下一期收益，供 lab05 对齐归因
        'dates': dates,                       # 研究区间月末
        'summary': summary,                   # IC 汇总，供 lab05 打印对照
    }, f)
print(f'\n已保存桥接文件：{BRIDGE_PATH}')
print(f'  因子 {list(weights.index)}，覆盖月末 {len(dates)} 个')
