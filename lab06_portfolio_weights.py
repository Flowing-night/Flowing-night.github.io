# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:14:39 2026

@author: 30261
"""

# -*- coding: utf-8 -*-
"""
第六讲实践：多权重组合构建与风险配置回测
============================================
比较 EW、IV、IVar、GMVP、ERC、RB、MDP、MDCP 八种权重在**同一股票池**下的
收益与风险表现。仅在月末调仓日按目标权重调整，各策略使用相同调仓时间、
相同样本区间与相同费用假设（默认零成本）。

数据接口沿用仓库 lab05/lab06：
    from data_io import load_data
    daily[name] = load_data(name, START, END)   # 行=交易日，列=股票代码

用法：
    1) 将本文件放入 code/lab06/ 目录；
    2) 在项目根目录运行：
        python code/lab06/lab06_portfolio_weights.py
    3) 图片与绩效表输出到 code/output/lab06/portfolio_weights/

若无法直接 import data_io（例如本地只有 CSV），仅需修改 SECTION 1 的
_load_all() 函数，把 load_data 替换为 pd.read_csv 读取日度宽表即可。
"""

# %% 0. 环境与输出目录
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# 让脚本能 import 到 data_io
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

try:
    from data_io import load_data
    HAVE_DATA_IO = True
except Exception:
    HAVE_DATA_IO = False

OUTPUT_DIR = HERE.parent / 'output' / 'lab06' / 'portfolio_weights'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% 1. 数据载入
START, END = '2019-12-01', '2025-12-31'
DAILY_FIELDS = ['close', 'adj_factor', 'pb', 'total_mv', 'vol',
                'amount', 'is_st', 'turnover_rate']


def _load_all():
    """读取全部日度宽表。行=交易日，列=股票，index.name='date'，columns.name='code'。"""
    if HAVE_DATA_IO:
        daily = {}
        for name in DAILY_FIELDS:
            df = load_data(name, START, END)
            df.index.name = 'date'
            df.columns.name = 'code'
            daily[name] = df
        return daily

    # ---- 本地 CSV 兜底实现（按需要修改路径与代码格式）----
    base = Path('/data/inputs')  # 改成你的日度宽表所在目录
    daily = {}
    for name in DAILY_FIELDS:
        # 约定：每个因子一个 csv，第一列 date，其余列为各股票代码
        f = base / f'{name}.csv'
        if not f.exists():
            raise FileNotFoundError(f'未找到 {f}，请确认日度宽表路径或改用 data_io 接口')
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.index.name = 'date'
        df.columns.name = 'code'
        daily[name] = df
    return daily


daily = _load_all()

# %% 2. 构造月末面板与复权价
calendar = daily['close'].index
dates = pd.DatetimeIndex(
    pd.Series(calendar).groupby(calendar.to_period('M')).max())
dates = dates[(dates >= START) & (dates <= END)]

price_raw = daily['close']
adj_price = price_raw * daily['adj_factor']          # 复权收盘价
ret_daily = adj_price.pct_change(fill_method=None)   # 日收益率

# 股票基本信息：上市日期（用于次新股过滤）
_stock_info_path = HERE.parent /'Existing directory' / 'data' / 'sse_stock_basic.csv'
if not _stock_info_path.exists():
    _stock_info_path = Path('/Existing directory/data/sse_stock_basic.csv')
stock_info = pd.read_csv(_stock_info_path)
stock_info['code_simple'] = stock_info['code'].str.replace('sh.', '', case=False)
ipo_map = pd.to_datetime(stock_info.dropna(subset=['ipoDate']).set_index('code_simple')['ipoDate'])
# 兼容代码格式：尝试 'sh.600000' 与 '600000.SH'
ipo_date = {}
for c, d in ipo_map.items():
    ipo_date['sh.' + c] = d
    ipo_date[c + '.SH'] = d

# 月末截面可交易判断（沿用 lab05：正价/量/额、非ST、有换手）
tradable = ((daily['close'] > 0) & (daily['vol'] > 0)
            & (daily['amount'] > 0)
            & (daily['is_st'] == 0)
            & (daily['turnover_rate'] > 0))


def _monthly_field(name):
    """取某字段在月末交易日的值，不可交易的格子置 NaN。"""
    return daily[name].reindex(dates).where(tradable.reindex(dates))


_monthly = {
    'close': _monthly_field('close'),
    'adj_close': _monthly_field('close') * _monthly_field('adj_factor'),
    'pb': _monthly_field('pb'),
    'total_mv': _monthly_field('total_mv'),
    'vol': _monthly_field('vol'),
    'amount': _monthly_field('amount'),
    'is_st': _monthly_field('is_st'),
}
panel_dict = {k: v.stack(future_stack=True) for k, v in _monthly.items()}
panel = pd.DataFrame(panel_dict).rename_axis(['date', 'code']).reset_index()

# 上市天数（向量化映射）
ipo_series = panel['code'].map(ipo_date)
panel['listed_days'] = (panel['date'] - ipo_series).dt.days

# 因子：BP 与 对数市值（用于选股打分，与 lab05/lab06 一致）
panel['bp'] = 1.0 / panel['pb'].where(panel['pb'] > 0)
panel['size'] = np.log(panel['total_mv'].where(panel['total_mv'] > 0))

print(f'[数据] 月末日期数={len(dates)}，面板样本行数={len(panel)}')


# %% 3. 选股：动态股票池 + 多因子打分（核心 select_universe）
def select_universe(panel, date, n_stocks=30, min_listed_days=120,
                    min_amount=10_000, use_zscore=True):
    """
    每月末选股，返回 code 列表。规则合并 lab05 + lab06：
      1) 基础过滤：pb>0、total_mv>0、有成交、有成交额、非 ST、上市满 min_listed_days；
      2) 因子构造：BP=1/PB（价值，正向）、size=log(total_mv)（市值，负向/小市值偏好）；
      3) 横截面 Z-score 标准化（可选，默认开启）；
      4) 综合打分 score = BP_z - size_z，取前 n_stocks 只。
    所有信号均只使用月末已知量，无未来函数。
    """
    df = panel[panel['date'] == date].copy()
    df = df.dropna(subset=['pb', 'total_mv', 'vol', 'amount'])
    df = df[(df['pb'] > 0) & (df['total_mv'] > 0)
            & (df['vol'] > 0) & (df['amount'] > 0)]
    if 'is_st' in df.columns:
        df = df[df['is_st'] == 0]
    if 'listed_days' in df.columns:
        df = df[df['listed_days'] >= min_listed_days]
    df = df[df['amount'] >= min_amount]

    if len(df) < n_stocks:
        return df['code'].tolist()

    df['bp'] = 1.0 / df['pb']
    df['size'] = np.log(df['total_mv'])

    if use_zscore:
        for col in ['bp', 'size']:
            mu, sd = df[col].mean(), df[col].std(ddof=0)
            df[col + '_z'] = (df[col] - mu) / (sd + 1e-12)
        bp_s, size_s = df['bp_z'], df['size_z']
    else:
        bp_s = df['bp'].rank(pct=True)
        size_s = df['size'].rank(pct=True)

    df['score'] = bp_s - size_s
    chosen = df.nlargest(n_stocks, 'score')
    return chosen['code'].tolist()


# %% 4. 协方差 / 波动率估计（滚动窗口）
def get_cov_and_vol(ret, codes, end_date, window=60):
    """用 end_date 之前的 window 个交易日估计年化协方差与波动率。"""
    data = ret.loc[:end_date, codes].dropna(how='any')
    if len(data) > window:
        data = data.tail(window)
    if len(data) < 10:
        return None, None
    cov = data.cov() * 252
    vol = data.std() * np.sqrt(252)
    cov = np.clip(cov.values, 1e-8, None)
    vol = np.clip(vol.values, 1e-8, None)
    if np.any(~np.isfinite(cov)) or np.any(~np.isfinite(vol)):
        return None, None
    return cov, vol


# %% 5. 八种权重算法
def weight_ew(n):
    return np.ones(n) / n


def weight_iv(vol):
    w = 1.0 / vol
    return w / w.sum()


def weight_ivar(vol):
    w = 1.0 / (vol ** 2)
    return w / w.sum()


def weight_gmvp(cov):
    """全局最小方差：w = Sigma^-1 1 / 1' Sigma^-1 1（允许小负权重）。"""
    inv = np.linalg.pinv(cov)
    w = inv @ np.ones(cov.shape[0])
    w = w / w.sum()
    return w


def _rc_objective(w, cov):
    """ERC 目标：最小化各资产风险贡献之间的离散度。"""
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    rc = w * (cov @ w)
    rc = rc / (rc.sum() + 1e-18)
    return float(np.sum((rc - rc.mean()) ** 2))


def weight_erc(cov):
    n = cov.shape[0]
    x0 = np.ones(n) / n
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bnds = [(0, 1)] * n
    res = minimize(_rc_objective, x0, args=(cov,), method='SLSQP',
                   bounds=bnds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-9})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def weight_rb(mv_array):
    """非等预算 RB：按总市值加权（风险预算的市值代理）。"""
    mv = np.maximum(mv_array, 0)
    if mv.sum() <= 0:
        return np.ones(len(mv)) / len(mv)
    return mv / mv.sum()


def _mdp_objective(w, cov, vol):
    """MDP 目标：最小化 组合波动 / 加权平均个券波动（即最大化分散度）。"""
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    port_vol = np.sqrt(w @ cov @ w)
    wvol = w @ vol
    if wvol <= 0:
        return 1e6
    return port_vol / wvol


def weight_mdp(cov, vol):
    n = len(vol)
    x0 = np.ones(n) / n
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bnds = [(0, 1)] * n
    res = minimize(_mdp_objective, x0, args=(cov, vol), method='SLSQP',
                   bounds=bnds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def weight_mdcp(cov, vol, max_weight=0.05):
    """MDCP：带单股权重上限的最大分散度组合。"""
    n = len(vol)
    x0 = np.ones(n) / n
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bnds = [(0, max_weight)] * n
    res = minimize(_mdp_objective, x0, args=(cov, vol), method='SLSQP',
                   bounds=bnds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-10})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


WEIGHT_METHODS = {
    'EW':   weight_ew,
    'IV':   weight_iv,
    'IVar': weight_ivar,
    'GMVP': weight_gmvp,
    'ERC':  weight_erc,
    'RB':   weight_rb,
    'MDP':  weight_mdp,
    'MDCP': weight_mdcp,
}

STRATEGY_ORDER = ['EW', 'IV', 'IVar', 'GMVP', 'ERC', 'RB', 'MDP', 'MDCP']


# %% 6. 月度调仓回测
def run_backtest(ret_daily, adj_price, panel, dates,
                 strategies=None, n_stocks=30, est_window=60,
                 cost_rate=0.0, mdcp_max_weight=0.05):
    """
    月末调仓：
      - 用 select_universe 确定当月股票池（所有策略共用同一股票池）；
      - 用过去 est_window 个交易日的日收益估计协方差/波动率；
      - 在当月内按目标权重持有，下月末再平衡；
      - cost_rate：双边调仓成本比例（默认 0 = 零成本）。
    返回：日频策略收益 DataFrame（index=交易日）。
    """
    if strategies is None:
        strategies = STRATEGY_ORDER

    daily_idx = adj_price.index
    strat_ret = pd.DataFrame(index=daily_idx, columns=strategies, dtype=float)
    prev_weights = {s: None for s in strategies}
    prev_date = None

    for i, d in enumerate(dates):
        # 6.1 选股
        codes = select_universe(panel, d, n_stocks=n_stocks)
        if len(codes) < 5:
            prev_date = d
            continue

        cov, vol = get_cov_and_vol(ret_daily, codes, d, window=est_window)
        if cov is None:
            prev_date = d
            continue

        # 调仓日市值（RB 用）
        mv_row = panel[(panel['date'] == d) & (panel['code'].isin(codes))]
        mv_dict = dict(zip(mv_row['code'], mv_row['total_mv']))
        mv_arr = np.array([mv_dict.get(c, 0) for c in codes], dtype=float)

        # 6.2 计算各策略权重
        weights = {}
        n = len(codes)
        for s in strategies:
            try:
                if s == 'EW':
                    weights[s] = WEIGHT_METHODS[s](n)
                elif s in ('IV', 'IVar'):
                    weights[s] = WEIGHT_METHODS[s](vol)
                elif s == 'GMVP':
                    weights[s] = WEIGHT_METHODS[s](cov)
                elif s == 'ERC':
                    weights[s] = WEIGHT_METHODS[s](cov)
                elif s == 'RB':
                    weights[s] = WEIGHT_METHODS[s](mv_arr)
                elif s == 'MDP':
                    weights[s] = WEIGHT_METHODS[s](cov, vol)
                elif s == 'MDCP':
                    weights[s] = WEIGHT_METHODS[s](cov, vol, mdcp_max_weight)
                else:
                    weights[s] = WEIGHT_METHODS[s](n)
                weights[s] = np.clip(weights[s], 0, None)
                weights[s] = weights[s] / (weights[s].sum() + 1e-18)
            except Exception:
                weights[s] = np.ones(n) / n

        # 6.3 持有区间：上一次调仓日 ~ 本次调仓日之间的交易日
        if prev_date is None:
            hold_slice = daily_idx[daily_idx <= d]
        else:
            hold_slice = daily_idx[(daily_idx > prev_date) & (daily_idx <= d)]
        if len(hold_slice) == 0:
            prev_date = d
            prev_weights = {s: pd.Series(w, index=codes) for s, w in weights.items()}
            continue

        # 6.4 区间内按权重持有，日收益加权求和
        for s in strategies:
            w = pd.Series(weights[s], index=codes)
            valid = w.index.intersection(ret_daily.columns)
            wv = w.loc[valid]
            if wv.sum() <= 0:
                continue
            wv = wv / wv.sum()
            r = ret_daily.loc[hold_slice, valid].fillna(0).values @ wv.values

            # 6.5 交易成本：仅在调仓日按换手近似扣除
            if cost_rate > 0 and prev_weights[s] is not None:
                old = prev_weights[s].reindex(codes).fillna(0)
                old = old / (old.sum() + 1e-18)
                turnover = float(np.abs(np.asarray(weights[s]) - old.values).sum())
                r = r - cost_rate * turnover
            strat_ret.loc[hold_slice, s] = r

        prev_date = d
        prev_weights = {s: pd.Series(w, index=codes) for s, w in weights.items()}

    return strat_ret.astype(float)


strategy_returns = run_backtest(
    ret_daily, adj_price, panel, dates,
    strategies=STRATEGY_ORDER,
    n_stocks=30,
    est_window=60,
    cost_rate=0.0,        # 零成本；如需手续费改为例如 0.0003
    mdcp_max_weight=0.05,
)
strategy_returns = strategy_returns.dropna(how='all')

# %% 7. 净值与绩效指标
nav = (1 + strategy_returns.fillna(0)).cumprod()
nav = nav[nav.index >= dates[0]]          # 从首个调仓月开始


def compute_performance(strat_ret, nav_df, rf=0.0):
    rows = []
    for col in strat_ret.columns:
        r = strat_ret[col].dropna()
        nv = nav_df[col].dropna()
        if len(r) < 2 or nv.iloc[-1] <= 0:
            continue
        T = len(r)
        total_ret = nv.iloc[-1] - 1
        ann_ret = nv.iloc[-1] ** (252.0 / T) - 1
        ann_vol = r.std(ddof=1) * np.sqrt(252)
        excess = r.mean() - rf / 252
        sharpe = (excess * 252) / ann_vol if ann_vol > 0 else np.nan
        mdd = (nv / nv.cummax() - 1).min()
        rows.append({
            '策略': col,
            '累计收益': total_ret,
            '年化收益': ann_ret,
            '年化波动率': ann_vol,
            '夏普率': sharpe,
            '最大回撤': mdd,
        })
    return pd.DataFrame(rows).set_index('策略')


performance = compute_performance(strategy_returns, nav, rf=0.0)
print('\n========== 绩效表 ==========')
print(performance.apply(lambda c: c.round(4) if c.dtype == float else c).to_string())
print('=============================')

best_ret = performance['累计收益'].idxmax()
best_sr = performance['夏普率'].idxmax()
print(f'\n累计收益最高：{best_ret}（{performance.loc[best_ret, "累计收益"]:.2%}）')
print(f'夏普率最高：{best_sr}（{performance.loc[best_sr, "夏普率"]:.3f}）')
print(f'两者是否相同：{best_ret == best_sr}')

# %% 8. 可视化：净值曲线 + 年化收益/波动率/夏普率柱状图
COLOR_MAP = {
    'EW': '#1f77b4', 'IV': '#ff7f0e', 'IVar': '#2ca02c', 'GMVP': '#d62728',
    'ERC': '#9467bd', 'RB': '#8c564b', 'MDP': '#e377c2', 'MDCP': '#7f7f7f',
}

# 图1：净值曲线
fig, ax = plt.subplots(figsize=(11, 5.5))
for s in performance.index:
    ax.plot(nav.index, nav[s], label=s, color=COLOR_MAP.get(s, '#333'),
            linewidth=1.3, alpha=0.9)
ax.set_title('净值曲线（月末调仓，零成本）', fontsize=13)
ax.set_ylabel('净值 NAV（初始=1）')
ax.set_xlabel('日期')
ax.legend(loc='upper left', ncol=4, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'nav_curve.png', dpi=160)
plt.close(fig)

# 图2：年化收益、年化波动率、夏普率三组柱状图
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
strategies = list(performance.index)
bar_colors = [COLOR_MAP.get(s, '#4472C4') for s in strategies]

axes[0].bar(strategies, performance['年化收益'], color=bar_colors)
axes[0].set_title('年化收益')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(strategies, performance['年化波动率'], color=bar_colors)
axes[1].set_title('年化波动率')
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(strategies, performance['夏普率'], color=bar_colors)
axes[2].set_title('夏普率（rf=0）')
axes[2].grid(axis='y', alpha=0.3)

for ax in axes:
    ax.tick_params(axis='x', rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'performance_bars.png', dpi=160)
plt.close(fig)

# %% 9. 结果说明
print('\n========== 结果解读 ==========')
print(f'从净值曲线看，{best_ret} 在样本期内累计收益最高，但高收益通常伴随较高波动；')
print(f'{best_sr} 夏普率最高，代表单位风险获得的超额收益更优。')
print('累计收益最高与夏普率最高通常并不相同：收益高的组合若波动/回撤过大，夏普率反而可能落后。')
print('EW/IV/IVar 依赖波动率缩放，实现简单但风险分散有限；')
print('GMVP/ERC/MDP/MDCP 通过协方差结构主动控制风险，波动与回撤通常更低。')
print(f'RB 按市值加权，偏向大市值股票，收益风险特征受市值因子影响明显。')
print('===========================')
print(f'\n图片已保存至：{OUTPUT_DIR}')
print(f'  - {OUTPUT_DIR / "nav_curve.png"}')
print(f'  - {OUTPUT_DIR / "performance_bars.png"}')
