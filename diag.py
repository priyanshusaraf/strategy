#!/usr/bin/env python3
"""Diagnostics: gross edge by horizon, autocorrelation profile, GO_crack artifact check."""
import math
import numpy as np, pandas as pd
import mrlab

START = "1990-01-01"

print("=" * 90)
print("1) AR(1) of spread changes + variance-ratio profile (is there MR at ANY horizon?)")
print("   VR(q) < 1 -> mean reversion at horizon q days; > 1 -> trending.")
print("=" * 90)
rows = []
for nm, fn in mrlab.UNIVERSE.items():
    dS = fn().dropna()
    dS = dS[dS.index >= START]
    if len(dS) < 1000:
        continue
    x = dS.values
    ac1 = pd.Series(x).autocorr(1)
    S = dS.cumsum()
    vr = {}
    for q in [5, 10, 20, 60]:
        dq = S.diff(q).dropna().values
        vr[q] = dq.var() / (q * x.var()) if x.var() > 0 else np.nan
    rows.append((nm, ac1, vr))
rows.sort(key=lambda r: r[2][20])
print(f"{'spread':<12}{'AC1':>7}{'VR5':>7}{'VR10':>7}{'VR20':>7}{'VR60':>7}")
for nm, ac1, vr in rows:
    print(f"{nm:<12}{ac1:>7.3f}{vr[5]:>7.2f}{vr[10]:>7.2f}{vr[20]:>7.2f}{vr[60]:>7.2f}")

print()
print("=" * 90)
print("2) GROSS edge (lam=0, band=0) of linear z40 detrended — signal problem vs cost problem")
print("=" * 90)
p = dict(z_n=40, z_cap=2.0, deadband=0.5, use_gate=False)
print(f"{'spread':<12}{'grossSh':>9}{'netSh(.05)':>11}")
for nm, fn in mrlab.UNIVERSE.items():
    dS = fn()
    g = mrlab.backtest(dS, "linear", p, lam=0.0, band=0.0, start=START, detrend=126)
    n = mrlab.backtest(dS, "linear", p, lam=0.05, band=0.25, start=START, detrend=126)
    mg, mn = mrlab.metrics(g), mrlab.metrics(n)
    if mg and mn:
        print(f"{nm:<12}{mg['full']['sharpe']:>9.2f}{mn['full']['sharpe']:>11.2f}")

print()
print("=" * 90)
print("3) GO_crack artifact check: 1-day-lagged execution (kill async-close artifact)")
print("   if Sharpe collapses with 1 extra day of execution delay -> artifact, not edge")
print("=" * 90)
dS = mrlab.UNIVERSE["GO_crack"]().dropna()
dS = dS[dS.index >= START]
for lag in [0, 1, 2]:
    mu = dS.shift(1).rolling(126, min_periods=63).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    z = mrlab.zscore(S, 10)
    w = (-(z / 2.0)).clip(-1, 1)
    w[z.abs() < 0.5] = 0.0
    w = w.fillna(0.0)
    sig = dS.ewm(span=63, min_periods=20).std().shift(1)
    tgt = 0.10 / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pnl = units.shift(1 + lag) * dS - 0.05 * sig * units.diff().abs().fillna(0.0)
    pnl = pnl.fillna(0.0)
    m = mrlab.metrics(pnl)
    print(f"  exec delay +{lag}d: Sharpe={m['full']['sharpe']:.2f}  OOS={m['OOS']['sharpe']:.2f}")

print()
print("4) GO_crack leg date alignment: % days where both legs present")
a, b = mrlab.leg("ULS1"), mrlab.leg("BRN1")
ja = a[a.index >= START]; jb = b[b.index >= START]
both = ja.index.intersection(jb.index)
print(f"  ULS days={len(ja)}, BRN days={len(jb)}, common={len(both)}")
