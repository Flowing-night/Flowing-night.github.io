# -*- coding: utf-8 -*-
"""
实践二 example（IC 加权多因子改动版）
====================================
用 lab06 产出的 IC 加权多因子分数做组合：按合成分数排名选股，再用全因子
截面回归对组合收益做归因。与单因子版本的区别：选股与归因都同时涉及所有因子。

与原始版本的差异（按 # %% 标记）：
  A. %%2.5 ：从 lab06 的 multifactor.pkl 加载多因子分数与各因子矩阵；
  B. %%3   ：score 改为多因子合成分数（IC 加权），不再用单因子排名；
  C. %%5   ：样本标准化扩展为全部入选因子；
  D. %%6/7 ：横截面回归 X 包含全部入选因子，factor / stock 中带出对应列；
  E. %%9   ：持仓归因新增每个入选因子的贡献项，并画图。

在项目根目录运行：python code/lab05/lab05_multifactor.py
也可在 IDE 中按 # %% 顺序分段运行。

输入：code/data/ 中的日度宽表 + output/lab06/multifactor.pkl；
输出：output/lab05/multifactor/ 下的图片。
收益使用小数，乘 100 后单位为百分点。
"""
import sys
import pickle
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 让脚本能 import 到 code/data_io 下的载入函数
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# %% 0. 准备运行环境与结果目录
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_io import load_data

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'output' / 'lab05' / 'multifactor'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# %% 1. 数据载入：读入全部日度宽表
START, END = '2019-12-01', '2025-12-31'
DAILY_FIELDS = ['close', 'adj_factor', 'pb', 'total_mv', 'vol',
                'amount', 'is_st', 'turnover_rate']

daily = {}
for name in DAILY_FIELDS:
    df = load_data(name, START, END)
    df.index.name = 'date'
    df.columns.name = 'code'
    daily[name] = df

# %% 2. 生成月末面板
calendar = daily['close'].index
dates = pd.DatetimeIndex(pd.Series(calendar).groupby(calendar.to_period('M')).max())

tradable = ((daily['close'] > 0) & (daily['vol'] > 0)
            & (daily['amount'] > 0) & (daily['is_st'] == 0)
            & (daily['turnover_rate'] > 0))
tradable_end = tradable.reindex(dates)

monthly = {}
for name in DAILY_FIELDS:
    monthly[name] = (daily[name].reindex(dates).where(tradable_end)
                     .stack(future_stack=True).rename_axis(['date', 'code']))
panel = pd.DataFrame(monthly).reset_index().sort_values(['code', 'date'])

panel['adj_close'] = panel['close'] * panel['adj_factor']
panel['bp'] = 1 / panel['pb']
panel['size'] = np.log(panel['total_mv'])

g = panel.groupby('code')
panel['return_end'] = g['date'].shift(-1)
panel['next_ret'] = g['adj_close'].shift(-1) / panel['adj_close'] - 1
panel['next_ret'] = panel['next_ret'].replace([np.inf, -np.inf], np.nan)

# %% 2.5 新增：加载 lab06 的 IC 加权多因子结果
BRIDGE_PATH = (Path(__file__).resolve().parents[2] / 'output' / 'lab06'
               / 'multifactor.pkl')
if not BRIDGE_PATH.exists():
    raise FileNotFoundError(
        f'未找到桥接文件 {BRIDGE_PATH}，请先运行 lab06_multifactor.py')

with open(BRIDGE_PATH, 'rb') as f:
    bridge = pickle.load(f)

weights = bridge['weights']                    # 因子名 -> IC 权重
multifactor = bridge['multifactor']            # T×N 多因子合成分数
factors_in = bridge['factors']                 # 各因子预处理后矩阵
bridge_dates = bridge['dates']                 # lab06 研究区间月末

SCORE_NAME = 'multifactor_score'
score_long = (multifactor.reindex(bridge_dates)
              .stack(future_stack=True)
              .rename_axis(['date', 'code'])
              .reset_index(name=SCORE_NAME))
panel = panel.merge(score_long, on=['date', 'code'], how='left')

# 把入选因子也并入 panel，供后续标准化与回归使用
factor_names = list(weights.keys())
for fn in factor_names:
    fl = (factors_in[fn].reindex(bridge_dates)
          .stack(future_stack=True)
          .rename_axis(['date', 'code'])
          .reset_index(name=fn))
    panel = panel.merge(fl, on=['date', 'code'], how='left')

print('>> IC 加权多因子权重：')
for k, v in weights.items():
    print(f'    {k}: {v:+.4f}')
print(f'   面板行数 {len(panel)}，合成分数列 {SCORE_NAME}，入选因子 {factor_names}')

# %% 3. 逐月调仓：按 IC 加权多因子分数排名选股（多因子打分法）
# —— 与单因子版本的核心区别：score 是多因子合成值，而非单因子排名 ——
TOPN = 20
holding_rows, return_rows = [], []
for date, g in panel.groupby('date'):
    g = g.dropna(subset=[SCORE_NAME, 'next_ret'])
    if g.empty:
        continue
    g = g.copy()
    chosen = g.nlargest(TOPN, SCORE_NAME)      # 多因子分数越高越靠前
    chosen['weight'] = 1 / len(chosen)
    holding_rows.append(chosen)
    return_rows.append({
        'date': date,
        'return_end': chosen['return_end'].iloc[0],
        '组合收益': chosen['next_ret'].mean(),
    })

holdings = pd.concat(holding_rows, ignore_index=True)
strategy_returns = pd.DataFrame(return_rows)

# %% 4. 回测：净值、收益与回撤
nav = (1 + strategy_returns.set_index('return_end')['组合收益']).cumprod().sort_index()
nav = pd.concat([pd.Series([1.0], index=[strategy_returns['date'].min()]), nav])
drawdown = nav / nav.cummax() - 1

performance = pd.DataFrame({
    '累计收益': nav.iloc[-1] - 1,
    '年化收益': nav.iloc[-1] ** (12 / len(strategy_returns)) - 1,
    '年化波动率': strategy_returns['组合收益'].std() * np.sqrt(12),
    '最大回撤': drawdown.min(),
}, index=['组合'])
print('\n组合表现：')
print(performance)

fig, ax = plt.subplots(figsize=(10, 4.5))
nav.plot(ax=ax, linewidth=1.3)
ax.set(xlabel='收益实现日期', ylabel='净值（初始值为1）')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'multifactor_nav.png', dpi=160)

# %% 5. 构造全部入选因子的暴露并按月标准化
cols_needed = factor_names + [SCORE_NAME, 'next_ret']
sample = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)
for col in factor_names:
    grp = sample.groupby('date')[col]
    sample[col + '_z'] = (sample[col] - grp.transform('mean')) / grp.transform('std')

# %% 6. 单月横截面回归：解释系数（全部入选因子）
# 观测是同一个月的不同股票
month = sample[sample['date'] == '2021-12-31']
Xcols = [c + '_z' for c in factor_names]
X = sm.add_constant(month[Xcols])
res = sm.OLS(month['next_ret'], X).fit()
print('\n2022 年 1 月全因子回归系数（乘 100 后单位为百分点）：')
print((100 * res.params).round(3))
print('调整后 R²：', round(res.rsquared_adj, 4))

# %% 7. 逐月估计：保存全部因子收益与个股残差
factor_rows, stock_rows = [], []
for date, grp in sample.groupby('date'):
    X = sm.add_constant(grp[Xcols])
    y = grp['next_ret']
    fit = sm.OLS(y, X).fit()
    factor_rows.append({'date': date, 'return_end': grp['return_end'].iloc[0],
                        **fit.params.to_dict(), 'n': int(fit.nobs),
                        'adjusted_r_squared': fit.rsquared_adj})
    stock_rows.append(grp.assign(fitted=fit.fittedvalues, resid=fit.resid))

factor = pd.DataFrame(factor_rows).set_index('date')
stock = pd.concat(stock_rows, ignore_index=True)

# %% 8. 观察各因子收益的跨月变化
fig, axes = plt.subplots(len(Xcols), 1, figsize=(10, 2.4 * len(Xcols)), sharex=True)
for col, label, ax in zip(Xcols, factor_names, axes):
    ax.plot(factor['return_end'], 100 * factor[col], linewidth=1.2)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set_ylabel(label + '（百分点）')
    ax.grid(alpha=.2)
axes[-1].set_xlabel('收益实现月份')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'monthly_factor_returns.png', dpi=160)

# %% 9. 利用原持仓分解多因子组合的月收益
# 持仓的归因：每个因子贡献 = 持仓加权平均暴露 × 当月该因子系数
merge_cols = ['date', 'code'] + Xcols + ['next_ret', 'resid']
positions = holdings[['date', 'code', 'weight']].merge(
    stock[[c for c in merge_cols if c in stock]], on=['date', 'code'], how='left')
for col in Xcols + ['next_ret', 'resid']:
    if col in positions:
        positions['weighted_' + col] = positions['weight'] * positions[col]

agg_cols = ['weighted_' + c for c in Xcols] + ['weighted_next_ret', 'weighted_resid']
portfolio = positions.groupby('date')[agg_cols].sum().join(
    factor[['return_end', 'const'] + Xcols])

portfolio['截距项'] = portfolio['const']
for rawc, zc in zip(factor_names, Xcols):
    portfolio[rawc + '贡献'] = portfolio['weighted_' + zc] * portfolio[zc]
portfolio['残差部分'] = portfolio['weighted_resid']
portfolio['实际收益'] = portfolio['weighted_next_ret']

parts = ['截距项'] + [c + '贡献' for c in factor_names] + ['残差部分']
fig, ax = plt.subplots(figsize=(9, 4))
(100 * portfolio[parts].mean()).plot.bar(ax=ax, rot=25, color='#225896')
ax.axhline(0, color='gray', linewidth=.7)
ax.set(ylabel='平均月收益贡献（百分点）', xlabel='')
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'portfolio_attribution.png', dpi=160)
print('\n平均月收益分解（百分点）：')
print((100 * portfolio[parts + ['实际收益']].mean()).round(3))
print('结果目录：', OUTPUT_DIR)
