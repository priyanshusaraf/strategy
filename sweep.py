#!/usr/bin/env python3
"""Broad sweep under the honest standard (delay=1). Goal: find configs where MANY
spreads are OOS-positive, not one lucky spread. Reports per-config cross-spread stats."""
import itertools, math, sys
import numpy as np, pandas as pd
import mrlab

START = "1990-01-01"
LAM = 0.05

print("loading universe...", flush=True)
DS = {}
for nm, fn in mrlab.UNIVERSE.items():
    try:
        dS = fn().dropna()
        dS = dS[dS.index >= START]
        if len(dS) > 2000:
            DS[nm] = dS
    except Exception as e:
        print(f"  {nm}: {e}")
print(f"{len(DS)} spreads\n", flush=True)

grid = list(itertools.product(
    [20, 40, 60, 120],      # z_n
    [0, 126, 252],          # detrend
    [False, True],          # seasonal
    [0.25, 0.5],            # band
))

results = []
for z_n, dt, seas, band in grid:
    p = dict(z_n=z_n, z_cap=2.0, deadband=0.5, use_gate=False)
    oos_sh, full_sh, surv = [], [], 0
    for nm, dS in DS.items():
        r = mrlab.backtest(dS, "linear", p, lam=LAM, seasonal=seas, band=band,
                           start=None, detrend=dt, delay=1)
        m = mrlab.metrics(r)
        if m is None:
            continue
        oos_sh.append(m["OOS"]["sharpe"])
        full_sh.append(m["full"]["sharpe"])
        if m["IS"]["sharpe"] > 0.3 and m["OOS"]["sharpe"] > 0.3:
            surv += 1
    o = np.array(oos_sh)
    results.append((z_n, dt, seas, band, o.mean(), (o > 0).sum(), len(o), surv,
                    np.mean(full_sh)))
    print(f"z_n={z_n:>4} dt={dt:>4} seas={int(seas)} band={band:.2f} | "
          f"meanOOS={o.mean():+.2f}  OOS>0: {(o>0).sum():2d}/{len(o)}  surv={surv:2d}  "
          f"meanFull={np.mean(full_sh):+.2f}", flush=True)

results.sort(key=lambda r: -(r[4] + r[7] * 0.05))
print("\nTOP 8 by mean OOS Sharpe + survivors:")
for z_n, dt, seas, band, mo, np_, n, surv, mf in results[:8]:
    print(f"  z_n={z_n} dt={dt} seas={int(seas)} band={band} -> meanOOS={mo:+.2f} "
          f"OOS>0 {np_}/{n} surv={surv}")
