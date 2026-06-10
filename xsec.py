#!/usr/bin/env python3
"""Cross-sectional MR within economically-coherent families.
Each spread is vol-normalized to a unit-risk return stream r_i = dS_i / sig_i.
Signal: z_i of each member's (detrended) level; cross-sectional demean -> xz_i;
weight w_i = -xz_i / cap (clipped), gross-normalized so sum|w| = 1.
Execution: next open. Cost lam * |du| (in unit-risk terms, sig=1 after normalization).
The family trend factor cancels in the demeaning — exactly what killed univariate MR."""
import math, sys
import numpy as np, pandas as pd
import mrlab

START = "2000-01-01"
LAM = 0.05

FAMILIES = {
    "energy_lvl":  ["RB_crack", "HO_crack", "GO_crack", "BRN_WTI", "RB_HO"],
    "energy_cal":  ["NG_cal", "RB_cal", "BRN_cal", "WBS_cal", "ULS_cal"],
    "ag_cal":      ["ZS_cal", "ZC_cal", "ZW_cal"],
    "ags":         ["ZW_KE", "ZC_ZW", "ZO_ZC", "crush"],
    "livestock":   ["LE_GF", "HE_LE", "LE_cal", "HE_cal"],
    "metal_ratio": ["GC_SI", "SI_HG", "GC_PL", "PL_PA"],
    "metal_cal":   ["GC_cal", "SI_cal"],
}

def load_family(names, z_n=60, detrend=126):
    Z, RO, RC = {}, {}, {}   # z, overnight unit-risk ret, day unit-risk ret
    for nm in names:
        F = mrlab.UNIVERSE[nm]().dropna()
        F = F[F.index >= START]
        dS = F["c"].diff()
        sig = dS.ewm(span=63, min_periods=20).std()
        if detrend:
            mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
            S = (dS - mu).cumsum()
        else:
            S = F["c"]
        Z[nm] = mrlab.zscore(S, z_n)
        RO[nm] = (F["o"] - F["c"].shift(1)) / sig.shift(1)   # overnight, normalized
        RC[nm] = (F["c"] - F["o"]) / sig.shift(1)            # day session, normalized
    return (pd.DataFrame(Z).dropna(how="all"), pd.DataFrame(RO), pd.DataFrame(RC))

def xsec_family(names, z_n=60, detrend=126, cap=1.5, lam=LAM):
    Z, RO, RC = load_family(names, z_n, detrend)
    Z = Z.dropna(thresh=2)                      # need >=2 members live
    xz = Z.sub(Z.mean(axis=1), axis=0)
    w = (-xz / cap).clip(-1, 1)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0.5, np.nan), axis=0).fillna(0.0)
    ua = w.shift(1)                             # held during day t (set at open t)
    du = ua.diff().abs()
    idx = ua.index
    pnl = (ua * RC.reindex(idx) + ua.shift(1) * RO.reindex(idx)).sum(axis=1) \
          - lam * du.sum(axis=1)
    return pnl.fillna(0.0)

def show(r, label):
    m = mrlab.metrics(r)
    if m is None:
        print(f"{label:<14} insufficient data"); return
    f, i, o = m["full"], m["IS"], m["OOS"]
    print(f"{label:<14} full={f['sharpe']:+.2f}  IS={i['sharpe']:+.2f}  OOS={o['sharpe']:+.2f}  "
          f"maxDD={f['maxdd']*100:.1f}%  eqR2={f['r2']:.2f}")

if __name__ == "__main__":
    for z_n in [20, 60, 120]:
        print(f"\n--- cross-sectional z_n={z_n} (detrend=126, cap=1.5, lam={LAM}) ---")
        tot = {}
        for fam, names in FAMILIES.items():
            r = xsec_family(names, z_n=z_n)
            show(r, fam)
            tot[fam] = r
        B = pd.DataFrame(tot)
        act = B.notna().sum(axis=1)
        port = B.fillna(0).sum(axis=1) / act.clip(lower=1)
        show(port[act >= 3], "ALL-FAMILY")
