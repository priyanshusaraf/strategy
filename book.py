#!/usr/bin/env python3
"""DIVERSIFIED BOOK — the smoothness engine.

Smoothness in stat-arb is not a property of any single spread; it is produced by
combining MANY low-correlation return streams. This script:

  1. runs the FROZEN hourly rule on the full economic spread universe (~35 spreads),
     ensemble over z-windows {9,12,18,24}td x entries {2.5,3.0};
  2. forms the book CAUSALLY (no lookahead): a spread is included at time t only while
     its OWN trailing 6-month strategy Sharpe (through t-1) is positive, and is weighted
     inverse to its trailing realized vol (risk parity across the active set);
  3. caps any single FAMILY's risk share (cracks are one factor — diversify across them);
  4. reports book smoothness honestly: Sharpe, eqR2 (linearity), maxDD, gain/pain,
     and how many individual spreads clear "smooth" (positive both halves, gain>=3x maxDD).

Two book variants are compared:
  - GATED   : per-spread vol gate ON  (higher Sharpe, lumpier — concentrates in dislocations)
  - CONTINUOUS: vol gate OFF (lower peak Sharpe, but trades all regimes -> smoother accrual)
The point of the comparison is to pick the variant that best matches the GOAL (smoothness).
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hourly_lab as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves")
os.makedirs(OUT, exist_ok=True)

ZDS = [9, 12, 18, 24]
ZES = [2.5, 3.0]
FAMILY = {  # for family risk capping
    "RB_crack": "energy", "HO_crack": "energy", "crack_321": "energy", "BZ_CL": "energy",
    "RB_HO": "energy", "RB_BZ": "energy", "HO_BZ": "energy", "crack321b": "energy",
    "NG_HO": "energy", "NG_CL": "energy", "NG_BZ": "energy", "RB_CL": "energy", "HO_CL": "energy",
    "BOHO": "ag", "crush": "ag", "ZW_KE": "ag", "ZC_ZW": "ag", "ZS_ZC": "ag", "ZM_ZL": "ag",
    "ZC_KE": "ag", "ZS_ZW": "ag", "ZW_ZL": "ag", "ZC_SB": "ag",
    "GC_SI": "metal", "SI_HG": "metal", "GC_PL": "metal", "PL_PA": "metal", "GC_HG": "metal",
    "SI_PL": "metal", "HG_PL": "metal", "GC_PA": "metal", "SI_PA": "metal",
    "KC_CC": "soft", "KC_SB": "soft", "CC_SB": "soft", "CT_SB": "soft", "KC_CT": "soft",
}
UNIVERSE = list(FAMILY)

def run_spread(nm, volgate):
    rets = []
    for zd in ZDS:
        for ze in ZES:
            p = dict(H.P); p["ze"] = ze; p["z_days"] = zd; p["volgate"] = volgate
            res = H.run(nm, p=p, slip_mult=1.0)
            if res is None:
                return None
            rets.append(res["ret"])
    R = pd.concat(rets, axis=1).fillna(0.0)
    return R.mean(axis=1), res["bars"]

def build_book(volgate, trail_months=6, sharpe_floor=0.0):
    streams, bars = {}, None
    for nm in UNIVERSE:
        out = run_spread(nm, volgate)
        if out is None:
            continue
        streams[nm], bars = out
    B = pd.DataFrame(streams).sort_index()
    bpy = len(B) / 730 * 365
    win = int(len(B) * (trail_months * 30.4) / 730)
    # causal inclusion mask + inverse-vol weight, each through t-1
    incl, ivol = {}, {}
    for nm in B.columns:
        r = B[nm]
        mu = r.shift(1).rolling(win, min_periods=win // 3).mean()
        sd = r.shift(1).rolling(win, min_periods=win // 3).std()
        tr_sh = (mu / sd * math.sqrt(bpy)).fillna(-9)
        incl[nm] = (tr_sh > sharpe_floor).astype(float)
        rv = r.shift(1).rolling(int(bpy / 12), min_periods=50).std()  # ~1m vol
        ivol[nm] = 1.0 / rv.replace(0, np.nan)
    I = pd.DataFrame(incl)[B.columns]
    V = pd.DataFrame(ivol)[B.columns].clip(upper=pd.DataFrame(ivol).quantile(0.99, axis=1).values[:, None] if False else None)
    V = pd.DataFrame(ivol)[B.columns].replace([np.inf], np.nan)
    raw_w = (I * V).fillna(0.0)
    # cap each family's gross weight share at 35% (cracks must not dominate)
    fam = pd.Series(FAMILY)
    for f in set(FAMILY.values()):
        cols = [c for c in B.columns if FAMILY[c] == f]
        fw = raw_w[cols].sum(axis=1)
        tot = raw_w.sum(axis=1).replace(0, np.nan)
        over = (fw / tot) > 0.35
        scale = np.where(over, 0.35 * tot / fw.replace(0, np.nan), 1.0)
        raw_w.loc[:, cols] = raw_w[cols].mul(pd.Series(scale, index=raw_w.index), axis=0)
    wsum = raw_w.sum(axis=1).replace(0, np.nan)
    W = raw_w.div(wsum, axis=0).fillna(0.0)
    book = (W * B).sum(axis=1)
    nactive = (W > 0).sum(axis=1)
    return book, B, nactive, bpy

def metrics(r, bpy, vol_target=0.10):
    nz = r[r != 0]
    if len(nz) < 200:
        return None
    r = r.loc[nz.index[0]:]
    ann = math.sqrt(bpy)
    if r.std() > 0:
        r = r * (vol_target / ann) / r.std()
    def block(rr):
        if len(rr) < 100 or rr.std() == 0:
            return dict(sh=0, dd=0, r2=0, gp=0)
        eq = rr.cumsum()
        dd = (eq - eq.cummax()).min()
        x = np.arange(len(eq)); b1, b0 = np.polyfit(x, eq.values, 1)
        ssr = ((eq.values - (b1 * x + b0)) ** 2).sum()
        sst = max(((eq.values - eq.values.mean()) ** 2).sum(), 1e-12)
        return dict(sh=rr.mean()/rr.std()*ann, dd=dd, r2=1-ssr/sst,
                    gp=eq.iloc[-1]/max(abs(dd), 1e-9))
    h = len(r) // 2
    return dict(full=block(r), IS=block(r.iloc[:h]), OOS=block(r.iloc[h:]), eq=r.cumsum(), r=r)

def per_spread_smooth(B, bpy):
    n = 0; names = []
    for nm in B.columns:
        m = metrics(B[nm], bpy)
        if m and m["full"]["sh"] > 0 and m["OOS"]["sh"] > 0 and m["full"]["gp"] >= 3.0 and m["full"]["dd"] >= -0.12:
            n += 1; names.append(nm)
    return n, names

if __name__ == "__main__":
    results = {}
    VARIANTS = [
        ("NAIVE_cont", False, -9.0),     # no perf selection, vol gate off
        ("NAIVE_gated", True, -9.0),     # no perf selection, vol gate on
        ("SELECT_cont", False, 0.0),     # trailing-Sharpe>0 selection, vol gate off
    ]
    for tag, vg, floor in VARIANTS:
        book, B, nact, bpy = build_book(vg, sharpe_floor=floor)
        m = metrics(book, bpy)
        ns, names = per_spread_smooth(B, bpy)
        npos = sum(1 for c in B.columns
                   if (mm := metrics(B[c], bpy)) and mm["full"]["sh"] > 0 and mm["OOS"]["sh"] > 0)
        results[tag] = (book, B, m, bpy, nact)
        f, i, o = m["full"], m["IS"], m["OOS"]
        print(f"\n=== {tag} book ===  spreads={len(B.columns)}  median active={int(nact.median())}")
        print(f"  full Sharpe={f['sh']:.2f}  IS={i['sh']:.2f}  OOS={o['sh']:.2f}")
        print(f"  eqR2(smoothness)={f['r2']:.3f}  maxDD={f['dd']*100:.1f}%  gain/pain={f['gp']:.1f}")
        print(f"  individual spreads positive both halves: {npos}/{len(B.columns)}")
        print(f"  individual spreads passing STRICT smooth screen: {ns} {names}")

    # plot book curves
    tags = [v[0] for v in VARIANTS]
    fig, ax = plt.subplots(1, len(tags), figsize=(7 * len(tags), 4.5))
    for k, tag in enumerate(tags):
        book, B, m, bpy, nact = results[tag]
        eq = m["eq"]; h = len(eq) // 2
        f = m["full"]
        ax[k].plot(eq.index, eq.values, lw=1.1, color="darkgreen")
        ax[k].axvline(eq.index[h], color="gray", ls="--", lw=0.7)
        ax[k].axhline(0, color="red", ls=":", lw=0.6)
        ax[k].set_title(f"{tag}: Sh={f['sh']:.2f} OOS={m['OOS']['sh']:.2f} eqR2={f['r2']:.2f} "
                        f"maxDD={f['dd']*100:.1f}% G/P={f['gp']:.1f}", fontsize=10)
    fig.suptitle("Diversified causally-selected book (10% ann vol) — smoothness comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/diversified_book.png", dpi=120)
    print(f"\n-> {OUT}/diversified_book.png")
