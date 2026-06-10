#!/usr/bin/env python3
"""Experiment runner: compare signal configs across the full spread universe.
Usage:  /usr/bin/python3 run.py [config ...]   (default: all configs)
Prints per-spread full/IS/OOS metrics and a survivor count per config.
"""
import sys, math
import numpy as np, pandas as pd
import mrlab

CONFIGS = {
    # baseline replication of the old lab (binary OU-gated)
    "binary_base": dict(strat="binary", seasonal=False, band=0.0,
                        p=dict(z_n=40, z_entry=2.0, z_exit=0.5, z_stop=3.0,
                               hl_win=120, hl_lo=2, hl_hi=40, t_mult=3.0, use_gate=True)),
    # continuous sizing, no gate, trade band to cut costs
    "linear":      dict(strat="linear", seasonal=False, band=0.25,
                        p=dict(z_n=40, z_cap=2.0, deadband=0.5, use_gate=False)),
    # + seasonal adjustment
    "linear_seas": dict(strat="linear", seasonal=True, band=0.25,
                        p=dict(z_n=40, z_cap=2.0, deadband=0.5, use_gate=False)),
    # lookback ensemble
    "ens":         dict(strat="linear_ens", seasonal=False, band=0.25,
                        p=dict(z_ns=[20, 40, 80], z_cap=2.0, deadband=0.5, use_gate=False)),
    # ensemble + seasonal
    "ens_seas":    dict(strat="linear_ens", seasonal=True, band=0.25,
                        p=dict(z_ns=[20, 40, 80], z_cap=2.0, deadband=0.5, use_gate=False)),
    # ensemble + OU gate
    "ens_gate":    dict(strat="linear_ens", seasonal=False, band=0.25,
                        p=dict(z_ns=[20, 40, 80], z_cap=2.0, deadband=0.5,
                               use_gate=True, hl_win=120, hl_lo=2, hl_hi=60)),
    # drift-adjusted signal level (kills roll-carry bleed on calendars)
    "lin_dt":      dict(strat="linear", seasonal=False, band=0.25, detrend=126,
                        p=dict(z_n=40, z_cap=2.0, deadband=0.5, use_gate=False)),
    "lin_dt_seas": dict(strat="linear", seasonal=True, band=0.25, detrend=126,
                        p=dict(z_n=40, z_cap=2.0, deadband=0.5, use_gate=False)),
    # fast reversal on detrended level
    "fast_dt":     dict(strat="linear", seasonal=False, band=0.25, detrend=126,
                        p=dict(z_n=10, z_cap=2.0, deadband=0.5, use_gate=False)),
    "ens_dt":      dict(strat="linear_ens", seasonal=False, band=0.25, detrend=126,
                        p=dict(z_ns=[10, 20, 40], z_cap=2.0, deadband=0.5, use_gate=False)),
}

LAM, LAM_STRESS = 0.05, 0.10
START = "1990-01-01"

def run_config(name, cfg, spreads=None):
    rows = []
    rets = {}
    for nm, fn in mrlab.UNIVERSE.items():
        if spreads and nm not in spreads:
            continue
        try:
            dS = fn()
            r = mrlab.backtest(dS, cfg["strat"], cfg["p"], lam=LAM,
                               seasonal=cfg["seasonal"], band=cfg["band"], start=START,
                               detrend=cfg.get("detrend", 0))
            m = mrlab.metrics(r)
        except Exception as e:
            print(f"  {nm}: ERROR {e}")
            continue
        if m is None:
            continue
        surv = m["IS"]["sharpe"] > 0.3 and m["OOS"]["sharpe"] > 0.3
        rows.append((nm, m, surv))
        rets[nm] = r
    rows.sort(key=lambda x: -x[1]["OOS"]["sharpe"])
    print(f"\n=== {name} ===  (lam={LAM}, start={START})")
    print(f"{'spread':<12}{'trades':>7}{'fullSh':>8}{'CAGR':>7}{'maxDD':>8}{'eqR2':>6}"
          f"{'ISsh':>7}{'OOSsh':>7}{'OOSdd':>8}  survive")
    nsurv = 0
    for nm, m, surv in rows:
        f, i, o = m["full"], m["IS"], m["OOS"]
        nsurv += surv
        print(f"{nm:<12}{m['trades']:>7}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%"
              f"{f['maxdd']*100:>7.1f}%{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}"
              f"{o['maxdd']*100:>7.1f}%  {'YES' if surv else '.'}")
    # equal-weight book of all spreads
    B = pd.DataFrame(rets)
    active = B.notna().sum(axis=1)
    port = B.fillna(0).sum(axis=1) / active.clip(lower=1)
    port = port[active >= 3]
    mb = mrlab.metrics(port)
    if mb:
        f, i, o = mb["full"], mb["IS"], mb["OOS"]
        print(f"{'BOOK':<12}{'':>7}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
              f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}{o['maxdd']*100:>7.1f}%")
    print(f"survivors (IS&OOS Sharpe>0.3): {nsurv}/{len(rows)}")
    return rows, rets

if __name__ == "__main__":
    sel = sys.argv[1:] or list(CONFIGS)
    for name in sel:
        run_config(name, CONFIGS[name])
