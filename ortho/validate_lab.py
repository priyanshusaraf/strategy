"""Apparatus controls for ortho/lab.py (AMR constitution: confirm the tool can
detect a planted edge and scores noise at ~0 BEFORE trusting any kills)."""
import numpy as np
import pandas as pd
import lab

rng = np.random.default_rng(42)
N = 60_000

def synth(drift_after_events=0.0, n_events=800, h_inject=8, seed=7):
    r = np.random.default_rng(seed)
    ret = r.normal(0, 1.0, N)
    ev = np.zeros(N, bool)
    ev[r.choice(np.arange(200, N - 100), size=n_events, replace=False)] = True
    if drift_after_events:
        for p in np.flatnonzero(ev):
            ret[p + 1:p + 1 + h_inject] += drift_after_events
    px = 100 + np.cumsum(ret) * 0.05
    idx = pd.date_range('2015-01-01', periods=N, freq='h', tz='UTC')
    df = pd.DataFrame({'o': px, 'h': px + 0.05, 'l': px - 0.05,
                       'c': px + r.normal(0, 0.01, N), 'v': 1.0}, index=idx)
    df['o'] = df.c.shift(1).fillna(df.c)   # next-bar open = prior close (honest chain)
    return df, ev

print('=== NEGATIVE control: random walk, random events ===')
df, ev = synth(0.0)
es = lab.event_study(df, ev, horizons=(1, 4, 8, 24), n_placebo=200)
print(es)
assert (es['t'].abs() < 3).all(), 'FALSE POSITIVE: apparatus finds edge in noise'
pnl0, n = lab.backtest(df, ev, h=8, side=1, cost=0.0)
m0 = lab.metrics(pnl0)
print('backtest cost=0:', m0, 'trades', n)
assert abs(m0['sharpe']) < 1.0, 'noise should be ~flat at zero cost'
pnl3, n = lab.backtest(df, ev, h=8, side=1, cost=3.0)
drag = pnl0.sum() - pnl3.sum()
exp = n * 3e-4 * 100 / 0.1   # n trades x 3bps x price~100 / ATR~0.1
print(f'cost drag {drag:.0f} vu vs expected ~{exp:.0f} vu')
assert 0.5 * exp < drag < 1.5 * exp, 'cost accounting wrong'

print('\n=== POSITIVE control: +0.05 sigma/bar drift for 8 bars after events ===')
df, ev = synth(0.05)
es = lab.event_study(df, ev, horizons=(1, 4, 8, 24), n_placebo=200)
print(es)
assert es.loc[8, 't'] > 4 and es.loc[8, 'p_placebo'] < 0.02, 'MISSED PLANTED EDGE'
print('(note: 0.05/bar edge = 0.2 vu < 0.3 vu cost -> correctly NOT monetizable)')

print('\n=== POSITIVE control 2: cost-clearing edge (0.15 sigma/bar x 8) ===')
df, ev = synth(0.15)
pnl, n = lab.backtest(df, ev, h=8, side=1, cost=3.0)
m = lab.metrics(pnl)
print('backtest:', m, 'trades', n)
assert m['sharpe'] > 1.0, 'backtester failed to monetize planted edge'

print('\n=== SHUFFLE test on planted edge: real should beat ~all shuffles ===')
rt, p = lab.shuffle_test(df, ev, h=8, side=1, cost=3.0, n_shuffle=50)
print(f'real total {rt:.0f} vu, frac shuffles >= real: {p:.2f}')
assert p <= 0.05

print('\nALL CONTROLS PASS')
