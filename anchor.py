#!/usr/bin/env python3
"""PRE-REGISTERED test of the anchored long-cheap fade (one config, all spreads, no tuning):

HYPOTHESIS (economic, stated before running):
  Storable-commodity spreads have a FLOOR: crack margins below refiners' variable cost
  trigger run cuts; calendar spreads at full carry are capped on the downside by
  cash-and-carry storage arbitrage. Therefore deep k-day DECLINES revert UP, while
  richness does not symmetrically revert down (no ceiling). Trade LONG-ONLY:
    move = (S_t - S_{t-k}) / (sig * sqrt(k)),  k = 63
    w    = clip((-move - DB) / CAP, 0, 1)      DB=0.75, CAP=1.5
  Execution: next open. Cost lam=0.05*sig per unit (stress 0.10). Vol-target 10%/yr.
  Start 2000 (real opens). IS/OOS split 60/40. Success = most spreads OOS-positive.
"""
import math, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrlab

START = "2000-01-01"
K, DB, CAP = 63, 0.75, 1.5
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

def anchor_fade(F, k=K, lam=LAM, db=DB, cap=CAP):
    F = F.dropna()
    F = F[F.index >= START]
    if len(F) < 1500:
        return None
    dS = F["c"].diff()
    sig = dS.ewm(span=63, min_periods=20).std()
    move = (F["c"] - F["c"].shift(k)) / (sig * math.sqrt(k))
    w = ((-move - db) / cap).clip(0, 1)
    tgt = 0.10 / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ua = units.shift(1)
    du = ua.diff().abs().fillna(0.0)
    pnl = (ua * (F["c"] - F["o"]) + ua.shift(1) * (F["o"] - F["c"].shift(1))
           - lam * sig.shift(1) * du)
    return pnl.fillna(0.0)

rows, rets = [], {}
for nm, fn in mrlab.UNIVERSE.items():
    try:
        r = anchor_fade(fn())
    except Exception as e:
        print(f"{nm}: ERROR {e}")
        continue
    if r is None:
        continue
    m = mrlab.metrics(r)
    if m is None:
        continue
    rows.append((nm, m))
    rets[nm] = r

rows.sort(key=lambda x: -x[1]["OOS"]["sharpe"])
pos_o = sum(1 for _, m in rows if m["OOS"]["sharpe"] > 0)
pos_b = sum(1 for _, m in rows if m["OOS"]["sharpe"] > 0 and m["IS"]["sharpe"] > 0)
print(f"ANCHORED LONG-CHEAP FADE  k={K} db={DB} cap={CAP} lam={LAM}  "
      f"OOS>0: {pos_o}/{len(rows)}   IS&OOS>0: {pos_b}/{len(rows)}")
print(f"{'spread':<12}{'fullSh':>8}{'CAGR':>7}{'maxDD':>8}{'eqR2':>6}{'ISsh':>7}{'OOSsh':>7}{'%active':>9}")
for nm, m in rows:
    f, i, o = m["full"], m["IS"], m["OOS"]
    act = rets[nm]
    pct = (act != 0).mean() * 100
    print(f"{nm:<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
          f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}{pct:>8.0f}%")

B = pd.DataFrame(rets)
act = B.notna().sum(axis=1)
port = B.fillna(0).sum(axis=1) / act.clip(lower=1)
port = port[act >= 5]
mb = mrlab.metrics(port)
if mb:
    f, i, o = mb["full"], mb["IS"], mb["OOS"]
    print(f"{'BOOK':<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
          f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}")
