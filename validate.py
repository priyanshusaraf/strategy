#!/usr/bin/env python3
"""Apparatus validation (the constitution's standing gate) + SYNC-universe evaluation.

1. POSITIVE control: synthetic OU spread (known half-life) -> engine must find it.
2. NEGATIVE control: pure random walk -> engine must NOT find an edge.
3. SYNC universe under settlement execution (close), post-2000, cost + stress.
"""
import math, sys
import numpy as np, pandas as pd
import mrlab

rng = np.random.default_rng(7)

def synth(kind, n=6000, hl=10.0, vol=1.0):
    """Synthetic spread open/close frame."""
    if kind == "ou":
        phi = math.exp(-math.log(2) / hl)
        eps = rng.normal(0, vol, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + eps[i]
    else:
        x = np.cumsum(rng.normal(0, vol, n))
    # opens: previous close + small overnight piece of the move
    o = np.empty(n); o[0] = x[0]
    o[1:] = x[:-1] + 0.3 * (x[1:] - x[:-1])
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"o": o, "c": x}, index=idx)

P_LIN = dict(z_n=20, z_cap=2.0, deadband=0.5)

print("=== apparatus validation ===")
for kind in ["ou", "rw"]:
    for em in ["close", "open"]:
        rs = []
        for trial in range(5):
            F = synth(kind)
            r = mrlab.backtest(F, "linear", P_LIN, lam=0.05, band=0.25, exec_mode=em)
            m = mrlab.metrics(r)
            rs.append(m["full"]["sharpe"] if m else 0)
        print(f"  {kind:<3} exec={em:<6} Sharpe (5 trials): "
              + " ".join(f"{s:+.2f}" for s in rs))

print("\n=== SYNC universe, exec=close (settlement orders), post-2000, lam=0.05 ===")
START = "2000-01-01"
CONFIGS = {
    "fast10": dict(strat="linear", detrend=126, band=0.33, p=dict(z_n=10, z_cap=2.0, deadband=0.5)),
    "lin20":  dict(strat="linear", detrend=126, band=0.33, p=dict(z_n=20, z_cap=2.0, deadband=0.5)),
    "lin40":  dict(strat="linear", detrend=126, band=0.33, p=dict(z_n=40, z_cap=2.0, deadband=0.5)),
    "tail60": dict(strat="binary", detrend=126, band=0.0,
                   p=dict(z_n=60, z_entry=2.0, z_exit=0.5, z_stop=4.0, t_stop=60, use_gate=False)),
}
for cname, cfg in CONFIGS.items():
    oos = []; surv = 0; n = 0; details = []
    for nm in sorted(mrlab.SYNC):
        if nm not in mrlab.UNIVERSE:
            continue
        try:
            F = mrlab.UNIVERSE[nm]()
            r = mrlab.backtest(F, cfg["strat"], cfg["p"], lam=0.05, band=cfg["band"],
                               start=START, detrend=cfg["detrend"], exec_mode="close")
            m = mrlab.metrics(r)
        except Exception as e:
            continue
        if m is None:
            continue
        n += 1
        oos.append(m["OOS"]["sharpe"])
        if m["IS"]["sharpe"] > 0.3 and m["OOS"]["sharpe"] > 0.3:
            surv += 1
            details.append((nm, m["IS"]["sharpe"], m["OOS"]["sharpe"]))
    o = np.array(oos)
    print(f"{cname:<8} meanOOS={o.mean():+.2f}  OOS>0: {(o>0).sum()}/{n}  surv={surv}  "
          + " ".join(f"{nm}({i:.1f}/{s:.1f})" for nm, i, s in details))
