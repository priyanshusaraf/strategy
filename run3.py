#!/usr/bin/env python3
"""Round 3: next-open execution (signal close t -> fill open t+1). Opens are real from
~2000, so START=2000. Tests fast->slow linear MR and tail fade, +/- meta filter."""
import sys, math
import numpy as np, pandas as pd
import mrlab

START = "2000-01-01"
LAM = 0.05

CONFIGS = {
    "fast10":  dict(strat="linear", detrend=126, band=0.33,
                    p=dict(z_n=10, z_cap=2.0, deadband=0.5)),
    "lin20":   dict(strat="linear", detrend=126, band=0.33,
                    p=dict(z_n=20, z_cap=2.0, deadband=0.5)),
    "lin40":   dict(strat="linear", detrend=126, band=0.33,
                    p=dict(z_n=40, z_cap=2.0, deadband=0.5)),
    "lin60":   dict(strat="linear", detrend=126, band=0.33,
                    p=dict(z_n=60, z_cap=2.0, deadband=0.5)),
    "ens":     dict(strat="linear_ens", detrend=126, band=0.33,
                    p=dict(z_ns=[10, 20, 40, 60], z_cap=2.0, deadband=0.5)),
    "tail60":  dict(strat="binary", detrend=126, band=0.0,
                    p=dict(z_n=60, z_entry=2.0, z_exit=0.5, z_stop=4.0,
                           t_stop=60, use_gate=False)),
}

def show(name, cfg, meta=False, quiet=False):
    rows, rets = [], {}
    for nm, fn in mrlab.UNIVERSE.items():
        try:
            F = fn()
            r = mrlab.backtest(F, cfg["strat"], cfg["p"], lam=LAM, band=cfg["band"],
                               start=START, detrend=cfg["detrend"])
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
    oos = np.array([m["OOS"]["sharpe"] for _, m in rows])
    print(f"\n=== {name}{' +meta' if meta else ''} ===  meanOOS={oos.mean():+.2f}  "
          f"OOS>0: {pos}/{len(rows)}  surv: {surv}")
    if not quiet:
        print(f"{'spread':<12}{'fullSh':>8}{'CAGR':>7}{'maxDD':>8}{'eqR2':>6}{'ISsh':>7}{'OOSsh':>7}")
        for nm, m in rows:
            f, i, o = m["full"], m["IS"], m["OOS"]
            print(f"{nm:<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
                  f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}")
    B = pd.DataFrame(rets)
    act = B.notna().sum(axis=1)
    port = B.fillna(0).sum(axis=1) / act.clip(lower=1)
    port = port[act >= 3]
    mb = mrlab.metrics(port)
    if mb:
        f, i, o = mb["full"], mb["IS"], mb["OOS"]
        print(f"{'BOOK':<12}{f['sharpe']:>8.2f}{f['cagr']*100:>6.1f}%{f['maxdd']*100:>7.1f}%"
              f"{f['r2']:>6.2f}{i['sharpe']:>7.2f}{o['sharpe']:>7.2f}")
    return rets

if __name__ == "__main__":
    sel = sys.argv[1:] or list(CONFIGS)
    for name in sel:
        show(name, CONFIGS[name], meta=False, quiet=(name not in ("fast10", "ens")))
        show(name, CONFIGS[name], meta=True, quiet=True)
