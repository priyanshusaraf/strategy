#!/usr/bin/env python3
"""Post-2000, open-execution reality map per spread:
   - MR Sharpe of fading the past-k normalized move (k=5,10,21,42,63), gross & net
   - momentum Sharpe (sign of past 63d move), to see what we're fighting
All with next-open execution. This tells us WHERE reversion is monetizable."""
import math
import numpy as np, pandas as pd
import mrlab

START = "2000-01-01"
LAM = 0.05

def run_fade(F, k, lam=LAM, trend_filter=None):
    F = F.dropna()
    F = F[F.index >= START]
    dS = F["c"].diff()
    sig = dS.ewm(span=63, min_periods=20).std()
    move = (F["c"] - F["c"].shift(k)) / (sig * math.sqrt(k))
    w = (-move / 1.5).clip(-1, 1)
    w[move.abs() < 0.5] = 0.0
    if trend_filter is not None:
        tz = (F["c"] - F["c"].shift(126)) / (sig * math.sqrt(126))
        w[tz.abs() > trend_filter] = 0.0
    tgt = 0.10 / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ua = units.shift(1)
    du = ua.diff().abs().fillna(0.0)
    pnl = ua * (F["c"] - F["o"]) + ua.shift(1) * (F["o"] - F["c"].shift(1)) - lam * sig.shift(1) * du
    return pnl.fillna(0.0)

def sh(r):
    r = r[r != 0]
    return r.mean() / r.std() * math.sqrt(252) if len(r) > 200 and r.std() > 0 else np.nan

print(f"{'spread':<12}", end="")
for k in [5, 10, 21, 42, 63]:
    print(f"{'f'+str(k):>7}", end="")
print(f"{'f21tf':>7}{'mom63':>7}")
agg = {k: [] for k in [5, 10, 21, 42, 63]}
for nm, fn in mrlab.UNIVERSE.items():
    F = fn()
    out = f"{nm:<12}"
    for k in [5, 10, 21, 42, 63]:
        s = sh(run_fade(F, k))
        agg[k].append(s)
        out += f"{s:>7.2f}" if s == s else f"{'na':>7}"
    s = sh(run_fade(F, 21, trend_filter=1.0))
    out += f"{s:>7.2f}" if s == s else f"{'na':>7}"
    # momentum 63d (diagnostic only)
    Fd = fn().dropna(); Fd = Fd[Fd.index >= START]
    dS = Fd["c"].diff(); sig = dS.ewm(span=63, min_periods=20).std()
    mz = (Fd["c"] - Fd["c"].shift(63)) / (sig * math.sqrt(63))
    w = np.sign(mz).where(mz.abs() > 0.3, 0.0)
    tgt = 0.10 / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ua = units.shift(1); du = ua.diff().abs().fillna(0.0)
    pnl = (ua * (Fd["c"] - Fd["o"]) + ua.shift(1) * (Fd["o"] - Fd["c"].shift(1))
           - LAM * sig.shift(1) * du).fillna(0.0)
    s = sh(pnl)
    out += f"{s:>7.2f}" if s == s else f"{'na':>7}"
    print(out, flush=True)
print("\nmean fade Sharpe by k:", {k: round(np.nanmean(v), 2) for k, v in agg.items()})
