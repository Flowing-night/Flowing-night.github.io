# -*- coding: utf-8 -*-
"""
lab07_compare.py
三项比较（同一股票池 / 同一基准 / 同一测试期）
  任务一：lab05 vs lab06 组合与基准的行业权重（超配/低配）
  任务二：周度 vs 月度再平衡净值
  任务三：不同费率下的净值
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 0. 路径 =====================
BASE_DIR = os.getcwd() # ← 改成你的项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)

# ===================== 1. 导入 =====================
from data_io import load_data, load_benchmark
import utils.utils as U

_get_industries  = U._get_industries
industry_neutral = U.industry_neutral
calc_nav         = U.calc_nav

# ===================== 2. 参数 =====================
START, END = '2022-12-30', '2025-12-31'
TOPN  = 20
COSTS = [0.0, 0.002, 0.010]

# ===================== 3. 数据加载 =====================
print('加载数据 ...')
close  = load_data('close',  START, END)
vol    = load_data('vol', START, END)
amount = load_data('amount', START, END)
pb     = load_data('pb',     START, END)
bp     = 1.0 / pb.replace(0, np.nan)   # BP = 1 / PB

# ===================== 4. 列去重（关键） =====================
def _dedup_cols(df, tag=''):
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        print(f'[{tag}] 重复列 {len(dup)} 个，已保留首次出现，例如 {dup[:3]}')
        df = df.loc[:, ~df.columns.duplicated()]
    return df

close = close[~close.index.duplicated(keep='last')]
dates = close.index          # 重新取一次

close  = _dedup_cols(close, 'close')
vol    = _dedup_cols(vol,   'volume').reindex(columns=close.columns)
amount = _dedup_cols(amount,'amount').reindex(columns=close.columns)
bp     = _dedup_cols(bp,    'bp').reindex(columns=close.columns)

dates = close.index
ret_d = close.pct_change().fillna(0)

# ===================== 5. 上市满 120 天 =====================
# ---- 上市满 120 天：纯位置对齐，完全不用 reindex ----
raw = U.get_listdate()
if isinstance(raw, pd.DataFrame):
    raw = raw.iloc[:, 0]
if isinstance(raw, pd.Series):
    raw = raw[~raw.index.duplicated(keep='first')]   # 先去掉源数据自身的重复代码
    ld_arr = raw.reindex(close.columns).values        # 此时 index 已唯一，安全
else:
    ld_map = pd.Series(raw)
    ld_map = ld_map[~ld_map.index.duplicated(keep='first')]
    ld_arr = ld_map.reindex(close.columns).values

ld_dt = pd.to_datetime(pd.Series(ld_arr), errors='coerce').values
age_ok = pd.DataFrame(
    close.index.values[:, None] >= (ld_dt[None, :] + np.timedelta64(120, 'D')),
    index=dates, columns=close.columns)

# ===================== 6. 统一股票池 =====================
valid = (close > 0) & (vol > 0) & (amount > 0) & age_ok
valid = valid.astype(float)
print(f'股票池：平均 {valid.sum(axis=1).mean():.0f} 只/日')

# ===================== 7. 因子与选股规则 =====================
def zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1, ddof=1).replace(0, np.nan), axis=0)

bp_z = zscore(bp)
in_neu = industry_neutral(ind='l1_code')
bp_neutral = in_neu(bp, standardize=True)

def select_lab05(d, pool):
    try:
        row = bp_z.loc[d]
    except KeyError:
        return {}
    s = row.where(pool.reindex(bp_z.columns).fillna(0) > 0).dropna()
    s = s[s != 0]
    c = list(s.sort_values(ascending=False).index[:TOPN])
    return {k: 1.0/len(c) for k in c} if c else {}

def select_lab06(d, pool):
    try:
        row = bp_neutral.loc[d]
    except KeyError:
        return {}
    z = row.where(pool.reindex(bp_neutral.columns).fillna(0) > 0).dropna()
    z = z[z != 0]
    if len(z) < 5:
        return {}
    try:
        g = pd.qcut(z.rank(method='first'), 5, labels=False)
        z = z[g == g.max()]
    except ValueError:
        z = z.sort_values(ascending=False).head(max(int(len(z)*0.2), TOPN))
    c = list(z.sort_values(ascending=False).index[:TOPN])
    return {k: 1.0/len(c) for k in c} if c else {}

# ===================== 8. 回测引擎 =====================
def run_bt(sel, reb_pos, comsn=0.0):
    w_mat = pd.DataFrame(0.0, index=dates, columns=close.columns)
    w, tots = {}, []
    for i, d in enumerate(dates):
        if i in reb_pos:                       # ← 位置判断，永不出错
            new = sel(d, valid.loc[d])
            if new:                            # 选不出票就维持原持仓
                tots.append(sum(abs(new.get(c, 0) - w.get(c, 0)) for c in set(w) | set(new)))
                w = new
        elif w:
            r = ret_d.loc[d].reindex(list(w.keys())).fillna(0)
            w = {k: v * (1 + r[k]) for k, v in w.items()}
            s = sum(w.values())
            if s > 0:
                w = {k: v / s for k, v in w.items()}
        if w:
            w_mat.loc[d, list(w.keys())] = pd.Series(w)
    nav = calc_nav(ret_d, w_mat.fillna(0), comsn=comsn)['nav']
    return nav / nav.iloc[0], float(np.sum(tots))

# ===================== 9. 基准 =====================
bench = load_benchmark(START, END)
bench = bench.iloc[:, 0] if isinstance(bench, pd.DataFrame) else bench
bench_nav = (1 + bench.pct_change().fillna(0)).cumprod()
bench_nav = bench_nav / bench_nav.iloc[0]
bench = bench[~bench.index.duplicated(keep='last')]     # ← 日期去重

# ---- 调仓日：用位置索引，避免 DatetimeIndex 的 in 判断失效 ----
dates = pd.DatetimeIndex(dates)
start_ts = pd.Timestamp(START)

reb_W = dates[dates.is_month_start | True]                       # 占位，下面覆盖
_wk_end = pd.Series(dates, index=dates).resample('W-SUN').last().dropna()
reb_W = pd.DatetimeIndex(_wk_end.values)
_mn_end = pd.Series(dates, index=dates).resample('ME').last().dropna()
reb_M = pd.DatetimeIndex(_mn_end.values)

# 只保留测试期内的调仓日，并转成位置集合供回测循环使用
pos_W = set(np.where(dates >= start_ts)[0][np.isin(dates[dates >= start_ts], reb_W)])
pos_M = set(np.where(dates >= start_ts)[0][np.isin(dates[dates >= start_ts], reb_M)])
print(f'周度调仓 {len(pos_W)} 次，月度调仓 {len(pos_M)} 次')


# ===================== 10. 任务一 =====================
def task1():
    for d in dates[sorted(pos_M)]:             # ← 用位置反查日期
        ind = _get_industries('l1_code', date=d).reindex(close.columns)
        
    rows05, rows06 = [], []
    for d in reb_M:
        ind = _get_industries('l1_code', date=d).reindex(close.columns)
        for sel, store in ((select_lab05, rows05), (select_lab06, rows06)):
            w = sel(d, valid.loc[d])
            if not w: continue
            s = pd.Series(0.0, dtype=float)
            for k, v in w.items():
                s[ind.get(k, '未知')] = s.get(ind.get(k, '未知'), 0) + v
            store.append(s)
    a05 = pd.concat(rows05, axis=1).mean(axis=1).sort_values(ascending=False)
    a06 = pd.concat(rows06, axis=1).mean(axis=1).sort_values(ascending=False)
    bench_w = ind.value_counts(normalize=True)
    top = a05.head(10).index
    df = pd.DataFrame({'基准': bench_w.reindex(top).fillna(0.0051),
                       'lab05': a05.reindex(top), 'lab06': a06.reindex(top)})
    df['lab05超配'] = df['lab05'] - df['基准']
    df['lab06超配'] = df['lab06'] - df['基准']
    print('\n【任务一】行业权重与超配/低配\n', df.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(top))
    for j, (key, col) in enumerate((('lab05','#2E5A88'), ('lab06','#D98C3A'))):
        ax[j].bar(x-0.27, df['基准'], 0.27, label='基准', color='#9AA5B1')
        ax[j].bar(x,      df[key],   0.27, label=f'{key}组合', color=col)
        ax[j].bar(x+0.27, df[key+'超配'], 0.27, label='超配/低配', color='#C0392B')
        ax[j].axhline(0, color='k', lw=.6)
        ax[j].set_xticks(x); ax[j].set_xticklabels(top, rotation=30, fontsize=8)
        ax[j].set_title(f'{key} vs 基准 行业配置'); ax[j].legend(fontsize=8)
    plt.tight_layout(); plt.savefig('task1_industry.png', dpi=120); plt.show()

# ===================== 11. 任务二 =====================
def task2():
    navW, tW = run_bt(select_lab05, pos_W, 0.002)
    navM, tM = run_bt(select_lab05, pos_M, 0.002)
    navW, tW = run_bt(select_lab05, reb_W, 0.002)
    navM, tM = run_bt(select_lab05, reb_M, 0.002)
    plt.figure(figsize=(12, 5))
    plt.plot(bench_nav, label='基准', color='gray', ls='--')
    plt.plot(navW, label=f'周度再平衡', color='#2E5A88')
    plt.plot(navM, label=f'月度再平衡', color='#D98C3A')
    plt.axhline(1, color='k', lw=.5, ls=':')
    plt.legend(); plt.grid(alpha=.3)
    plt.title('任务二：周度 vs 月度再平衡（lab05 规则，c=0.002）')
    plt.tight_layout(); plt.savefig('task2_rebalance.png', dpi=120); plt.show()
    print(f'\n【任务二】周度：{len(reb_W)}次，换手 {tW:.1%}，净值 {navW.iloc[-1]:.4f}')
    print(f'         月度：{len(reb_M)}次，换手 {tM:.1%}，净值 {navM.iloc[-1]:.4f}')

# ===================== 12. 任务三 =====================
def task3():
    for i, c in enumerate(COSTS):
        nav, t = run_bt(select_lab05, pos_W, comsn=c)
        
    plt.figure(figsize=(12, 5))
    plt.plot(bench_nav, label='基准', color='gray', ls='--')
    cols = ['#2E5A88', '#D98C3A', '#4F8A5B']
    print('\n【任务三】固定周度再平衡：')
    for i, c in enumerate(COSTS):
        nav, t = run_bt(select_lab05, reb_W, comsn=c)
        plt.plot(nav, label=f'c={c}（净值 {nav.iloc[-1]:.3f}）', color=cols[i])
        print(f'  c={c}: 净值 {nav.iloc[-1]:.4f}')
    plt.axhline(1, color='k', lw=.5, ls=':')
    plt.legend(); plt.grid(alpha=.3)
    plt.title('任务三：不同交易费率（lab05 周度再平衡）')
    plt.tight_layout(); plt.savefig('task3_cost.png', dpi=120); plt.show()

def _diag():
    if len(reb_W) == 0:
        print('[诊断] reb_W 为空，跳过'); return
    d = reb_W[0]
    print('\n[诊断] 调仓日 d =', d)
    print('  d 在 bp_z.index ?', d in bp_z.index)
    print('  valid 当日数量  ', int(valid.loc[d].sum()))
    try:
        w05 = select_lab05(d, valid.loc[d])
        print('  select_lab05 选出', len(w05), '只，样例', list(w05.items())[:2])
    except Exception as e:
        print('  select_lab05 报错:', type(e).__name__, e)
    try:
        w06 = select_lab06(d, valid.loc[d])
        print('  select_lab06 选出', len(w06), '只，样例', list(w06.items())[:2])
    except Exception as e:
        print('  select_lab06 报错:', type(e).__name__, e)
    mask = valid.loc[d].reindex(bp_z.columns).fillna(0) > 0
    print('  bp_z 当日非空        ', int(bp_z.loc[d].notna().sum()))
    print('  其中在股票池内       ', int((bp_z.loc[d].notna() & mask).sum()))
    print('  bp_neutral 当日非空  ', int(bp_neutral.loc[d].notna().sum()))
    print('  其中在股票池内       ', int((bp_neutral.loc[d].notna() & mask).sum()))
    print('  close/bp_z 列相同 ?  ', list(close.columns) == list(bp_z.columns))
    print('  列名交集数量         ', len(set(close.columns) & set(bp_z.columns)))
    print('  close 列样例         ', list(close.columns[:3]))
# ===================== 13. main =====================
if __name__ == '__main__':
    _diag()          # ← 临时加这一行，跑完删掉
    print('='*60)
    task1()
    task2()
    task3()
    print('\n完成 ✅ 三张图已保存')