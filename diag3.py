#!/usr/bin/env python3
"""Last structural angles for spread MR:
  A) long-horizon fades k in {126, 252} (margin reversion over 0.5-1y)
  B) asymmetric fade: short-rich-only vs long-cheap-only (k=21,63,126)
  C) subperiod stability of the k=21 fade: 1990-2000, 2000-2013, 2013-2026
Next-open execution, lam=0.05."""
import math
import numpy as np, pandas as pd
import mrlab

LAM = 0.05

def fade(F, k, lam=LAM, side="both", start=None, end=None, cap=1.5, db=0.5):
    F = F.dropna()
    if start: F = F[F.index >= start]
    if end:   F = F[F.index < end]
    if len(F) < 800: return None
    dS = F["c"].diff()
    sig = dS.ewm(span=63, min_periods=20).std()
    move = (F["c"] - F["c"].shift(k)) / (sig * math.sqrt(k))
    w = (-move / cap).clip(-1, 1)
    w[move.abs() < db] = 0.0
    if side == "short": w = w.clip(upper=0.0)   # only short rich spreads
    if side == "long":  w = w.clip(lower=0.0)   # only long cheap spreads
    tgt = 0.10 / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ua = units.shift(1); du = ua.diff().abs().fillna(0.0)
    pnl = (ua * (F["c"] - F["o"]) + ua.shift(1) * (F["o"] - F["c"].shift(1))
           - lam * sig.shift(1) * du)
    return pnl.fillna(0.0)

def sh(r):
    if r is None: return np.nan
    r = r[r != 0]
    return r.mean() / r.std() * math.sqrt(252) if len(r) > 150 and r.std() > 0 else np.nan

names = list(mrlab.UNIVERSE)
print("A/B) long-horizon and asymmetric fades (2000-2026):")
print(f"{'spread':<12}{'f126':>7}{'f252':>7}{'s21':>7}{'s63':>7}{'s126':>7}{'l21':>7}{'l63':>7}{'l126':>7}")
for nm in names:
    F = mrlab.UNIVERSE[nm]()
    vals = [sh(fade(F, 126, start="2000-01-01")), sh(fade(F, 252, start="2000-01-01"))]
    for s_ in ["short", "long"]:
        for k in [21, 63, 126]:
            vals.append(sh(fade(F, k, side=s_, start="2000-01-01")))
    print(f"{nm:<12}" + "".join(f"{v:>7.2f}" if v == v else f"{'na':>7}" for v in vals))

print("\nC) subperiod k=21 fade:")
print(f"{'spread':<12}{'90-00':>8}{'00-13':>8}{'13-26':>8}")
for nm in names:
    F = mrlab.UNIVERSE[nm]()
    a = sh(fade(F, 21, start="1990-01-01", end="2000-01-01"))
    b = sh(fade(F, 21, start="2000-01-01", end="2013-01-01"))
    c = sh(fade(F, 21, start="2013-01-01"))
    row = "".join(f"{v:>8.2f}" if v == v else f"{'na':>8}" for v in [a, b, c])
    print(f"{nm:<12}{row}")
