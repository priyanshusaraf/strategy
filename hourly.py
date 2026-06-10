#!/usr/bin/env python3
"""Hourly spread MR test on CL-BRN (the one real intraday spread series available).
Honest: signal at bar t close -> execute bar t+1 open; costs in DOLLAR ticks per turn.
CL and BRN both $0.01 tick, deep books -> half-spread+slip ~ $0.015-0.03 per leg-pair turn.
Grid over z window (hours) and entry/exit, report net Sharpe (hourly-compounded, annualized
by sqrt(bars/yr)), maxDD, IS/OOS halves."""
import math, sys
import numpy as np, pandas as pd

F = pd.read_csv("/Users/priyanshusaraf/Desktop/internship-final-reports/data/raw/cl_brn_spread_60.csv")
F["time"] = pd.to_datetime(F["time"], utc=True)
F = F.set_index("time").sort_index()[["open", "close"]].dropna()
print(f"bars={len(F)}  {F.index.min().date()} -> {F.index.max().date()}")
dS = F["close"].diff()
sig_h = dS.ewm(span=24 * 21, min_periods=100).std()
print(f"hourly sigma ~ ${dS.std():.4f}, spread level {F['close'].iloc[-1]:.2f}")
BPY = len(F) / ((F.index.max() - F.index.min()).days / 365.25)   # bars per year

def run(z_n, ze, zx, cost_usd, tstop=24 * 10, detrend=24 * 21):
    mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    n = len(S)
    w = np.zeros(n); cur = 0; held = 0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= 4.0 or held >= tstop
                    or (cur > 0 and zz >= -zx) or (cur < 0 and zz <= zx)):
                cur, held = 0, 0
        if cur == 0 and zz == zz:
            if zz <= -ze: cur, held = 1, 0
            elif zz >= ze: cur, held = -1, 0
        w[i] = cur
    w = pd.Series(w, index=F.index)
    ua = w.shift(1)                     # held during bar t (entered at bar t open)
    du = ua.diff().abs().fillna(0.0)
    pnl = (ua * (F["close"] - F["open"])
           + ua.shift(1) * (F["open"] - F["close"].shift(1))
           - cost_usd * du)
    pnl = pnl.fillna(0.0)
    ann = math.sqrt(BPY)
    out = {}
    for tag, rr in [("full", pnl), ("IS", pnl.iloc[:len(pnl)//2]), ("OOS", pnl.iloc[len(pnl)//2:])]:
        sh = rr.mean() / rr.std() * ann if rr.std() > 0 else 0
        eq = rr.cumsum()
        dd = (eq - eq.cummax()).min()
        out[tag] = (sh, dd)
    trades = int(du.gt(0).sum())
    return out, trades

print(f"\n{'z_n':>5}{'ze':>5}{'zx':>5}{'cost$':>7}{'fullSh':>8}{'ISsh':>7}{'OOSsh':>7}{'maxDD$':>8}{'trades':>7}")
for z_n in [24, 72, 168, 336]:
    for ze, zx in [(2.0, 0.5), (2.5, 0.5), (3.0, 1.0)]:
        for cost in [0.0, 0.02, 0.04]:
            o, tr = run(z_n, ze, zx, cost)
            print(f"{z_n:>5}{ze:>5.1f}{zx:>5.1f}{cost:>7.2f}{o['full'][0]:>8.2f}"
                  f"{o['IS'][0]:>7.2f}{o['OOS'][0]:>7.2f}{o['full'][1]:>8.2f}{tr:>7}")
