#!/usr/bin/env python3
"""Round 2: patient tail-fading under honest execution (delay=1), with and without
the causal trailing-Sharpe meta-filter. Reports per-spread and book."""
import sys, math
import numpy as np, pandas as pd
import mrlab

START = "1990-01-01"
LAM = 0.05

CONFIGS = {
    # patient tail fade: rare entries at big dislocation, exit near mean, 60d patience
    "tail60":   dict(strat="binary", detrend=126, band=0.0,
                     p=dict(z_n=60, z_entry=2.0, z_exit=0.5, z_stop=4.0,
                            t_stop=60, use_gate=False)),
    "tail120":  dict(strat="binary", detrend=252, band=0.0,
                     p=dict(z_n=120, z_entry=2.0, z_exit=0.5, z_stop=4.0,
                            t_stop=90, use_gate=False)),
    # linear slow with weekly rebalance (Monday) + band
    "lin60w":   dict(strat="linear", detrend=126, band=0.25,
                     p=dict(z_n=60, z_cap=2.0, deadband=0.75, rebal_dow=0, use_gate=False)),
    "lin120w":  dict(strat="linear", detrend=252, band=0.25,
                     p=dict(z_n=120, z_cap=2.0, deadband=0.75, rebal_dow=0, use_gate=False)),
}

def show(name, cfg, meta=True):
    rows, rets = [], {}
    for nm, fn in mrlab.UNIVERSE.items():
        try:
            dS = fn()
            r = mrlab.backtest(dS, cfg["strat"], cfg["p"], lam=LAM, band=cfg["band"],
                               start=START, detrend=cfg["detrend"], delay=1)
            if meta:
                r, _ = mrlab.meta_filter(r)
            m = mrlab.metrics(r)
        except Exception as e:
            print(f"  {nm}: ERROR {e}"); continue
        if m is None:
            continue
        rows.append((nm, m)); rets[nm] = r
    rows.sort(key=lambda x: -x[1]["OOS"]["sharpe"])
    pos = sum(1 for _, m in rows if m["OOS"]["sharpe"] > 0)
    surv = sum(1 for _, m in rows if m["IS"]["sharpe"] > 0.3 and m["OOS"]["sharpe"] > 0.3)
    print(f"\n=== {name}{' +meta' if meta else ''} ===  OOS>0: {pos}/{len(rows)}  surv: {surv}")
    print(f"{'spread':<12}{'fullSh':>8}{'CAGR':>7}{'maxDD':>8}{'eqR2':>6}{'ISsh':>7}{'OOSsh':>7}{'OOSdd':>8}")
    for nm, m in rows[:14]:
        f, i, o = m["full"], m["IS"], m["OOS"]
        print(f"{nm:<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
              f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}{o['maxdd']*100:>7.1f}%")
    B = pd.DataFrame(rets)
    act = B.notna().sum(axis=1)
    port = B.fillna(0).sum(axis=1) / act.clip(lower=1)
    port = port[act >= 3]
    mb = mrlab.metrics(port)
    if mb:
        f, i, o = mb["full"], mb["IS"], mb["OOS"]
        print(f"{'BOOK':<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
              f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}{o['maxdd']*100:>7.1f}%")
    return rets

if __name__ == "__main__":
    sel = sys.argv[1:] or list(CONFIGS)
    for name in sel:
        show(name, CONFIGS[name], meta=False)
        show(name, CONFIGS[name], meta=True)
