#!/usr/bin/env python3
"""SEASONAL CRACK-SPREAD mean reversion — on-mandate (cracks), forward-applicable, long history.

Economic thesis (forward-valid, not a decayed micro-inefficiency):
  Refined-product cracks carry a STRONG, REPEATING seasonal driven by demand + turnarounds:
    - gasoline (RB) crack peaks spring/summer (driving season + spring refinery maintenance),
    - distillate (HO/ULS) crack peaks autumn/winter (heating + harvest/diesel),
    - the RB-HO product spread therefore oscillates seasonally between the two.
  A crack that is rich/cheap RELATIVE TO ITS SEASONAL NORM mean-reverts as refiners adjust
  yields and run rates. This anchor is structural (physics of refining + weather), so unlike
  generic level-MR it does NOT arbitrage away — it is re-established every year.

Signal (causal):
  C   = crack level ($/bbl) from leg closes.
  Cs  = seasonal expectation = day-of-year mean of C over the PRIOR `yrs` years (>=3 yrs),
        which absorbs both the seasonal shape AND slow level drift (handles back-adjustment).
  z   = (C - Cs) / rolling_std(C - Cs, 60).   Fade |z|>=entry, exit near 0, stops.
Execution: same-venue cracks (RB/HO/CL all NYMEX) -> same-close fill legitimate; cost 0.05*sig.
Horizon ensemble {15,20,30,40} bars x entry {2.0,2.5}.  Smoothness: gain/pain + eqR2 + both halves.
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrlab

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves")
os.makedirs(OUT, exist_ok=True)

START = "1990-01-01"
LAM = 0.05
ZDS = [15, 20, 30, 40]
ZES = [2.0, 2.5]

# on-mandate crack / refined-product spreads, all same-venue where possible
CRACKS = {
    "RB_crack":  (["RB1", "CL1"], [42, -1]),       # gasoline crack ($/bbl)
    "HO_crack":  (["HO1", "CL1"], [42, -1]),       # heating-oil crack
    "crack_321": (["RB1", "HO1", "CL1"], [28, 14, -1]),
    "RB_HO":     (["RB1", "HO1"], [42, -42]),      # gasoline-distillate (strong seasonal)
    "RBb_crack": (["RB1", "WBS1"], [42, -1]),      # gasoline crack vs WTI(WBS)
    "RB_brent":  (["RB1", "BRN1"], [42, -1]),      # gasoline vs Brent
    "HO_brent":  (["HO1", "BRN1"], [42, -1]),      # distillate vs Brent
    "GO_crack":  (["ULS1", "BRN1"], [1/7.45, -1]), # gasoil crack
    "321_brent": (["RB1", "HO1", "BRN1"], [28, 14, -1]),
    "BRN_WTI":   (["BRN1", "WBS1"], [1, -1]),      # quality/location crack
    "UHO_crack": (["UHO1", "BRN1"], [42, -1]),     # ICE heating crack
}

def seas_exp(C, yrs=8, min_yrs=3, tol=10):
    idx = C.index; v = C.values; rows = []
    for k in range(1, yrs + 1):
        tgt = idx - pd.DateOffset(years=k)
        pos = np.clip(idx.searchsorted(tgt), 0, len(idx) - 1)
        good = np.abs((idx[pos] - tgt).days) <= tol
        rows.append(np.where(good, v[pos], np.nan))
    mat = np.vstack(rows); cnt = (~np.isnan(mat)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        s = np.where(cnt >= min_yrs, np.nanmean(mat, axis=0), np.nan)
    return pd.Series(s, index=idx)

def build(legs, w):
    try:
        O = pd.concat([mrlab.leg_df(l)["open"] for l in legs], axis=1, keys=range(len(legs))).dropna()
        Cc = pd.concat([mrlab.leg_df(l)["close"] for l in legs], axis=1, keys=range(len(legs))).dropna()
    except FileNotFoundError:
        return None
    idx = O.index.intersection(Cc.index)
    o = sum(wi * O.loc[idx, i] for i, wi in enumerate(w))
    c = sum(wi * Cc.loc[idx, i] for i, wi in enumerate(w))
    return pd.DataFrame({"o": o, "c": c})

def one(F, zd, ze, volgate=True):
    F = F.dropna(); F = F[F.index >= START]
    if len(F) < 2500: return None
    C = F["c"]
    dS = C.diff()
    Cs = seas_exp(C).shift(1)
    Cs = Cs.fillna(C.shift(1).rolling(252, min_periods=120).mean())   # fallback pre-3yr
    dev = C - Cs
    sd = dev.shift(1).rolling(zd).std()
    z = (dev / sd).values
    n = len(F)
    if volgate:
        sf = dS.ewm(span=5, min_periods=5).std(); ss = sf.rolling(126, min_periods=60).median()
        hot = (sf > ss).values
    else:
        hot = np.ones(n, bool)
    sig = dS.ewm(span=63, min_periods=20).std(); sig_safe = sig.replace(0, np.nan).ffill().bfill().values
    w = np.zeros(n); cur = 0; held = 0
    for i in range(max(zd, 252) + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= 4.0 or held >= 20
                    or (cur > 0 and zz >= -0.5) or (cur < 0 and zz <= 0.5)):
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:
            if zz <= -ze: cur, held = 1, 0
            elif zz >= ze: cur, held = -1, 0
        w[i] = cur
    u = np.zeros(n)
    for i in range(1, n):
        if w[i] == 0: u[i] = 0.0
        elif w[i-1] == 0 or np.sign(w[i]) != np.sign(w[i-1]): u[i] = w[i] / sig_safe[i]
        else: u[i] = u[i-1]
    u = pd.Series(u, index=F.index); du = u.diff().abs().fillna(0.0)
    pnl = u.shift(1) * dS - LAM * sig.shift(1) * du
    return pnl.fillna(0.0) * (0.10 / math.sqrt(252))

def ensemble(name, **kw):
    legs, w = CRACKS[name]; F = build(legs, w)
    if F is None: return None
    parts = []
    for zd in ZDS:
        for ze in ZES:
            r = one(F, zd, ze, **kw)
            if r is None: return None
            parts.append(r)
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)

def metr(r, lo=None, hi=None):
    if lo: r = r[r.index >= lo]
    if hi: r = r[r.index < hi]
    nz = r[r != 0]
    if len(nz) < 250: return None
    r = r.loc[nz.index[0]:]
    if r.std() > 0: r = r * (0.10/math.sqrt(252)) / r.std()
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    x = np.arange(len(eq)); b1, b0 = np.polyfit(x, eq.values, 1)
    ssr = ((eq.values-(b1*x+b0))**2).sum(); sst = max(((eq.values-eq.values.mean())**2).sum(), 1e-12)
    h = len(r)//2; sa, sb = r.iloc[:h], r.iloc[h:]
    return dict(eq=eq, sh=r.mean()/r.std()*math.sqrt(252), gp=eq.iloc[-1]/max(abs(dd), 1e-9),
                r2=1-ssr/sst, dd=dd,
                a=sa.mean()/sa.std()*math.sqrt(252) if sa.std()>0 else 0,
                b=sb.mean()/sb.std()*math.sqrt(252) if sb.std()>0 else 0, yrs=len(r)/252)

if __name__ == "__main__":
    vg = "novg" not in sys.argv
    rows = {}
    for nm in CRACKS:
        try: r = ensemble(nm, volgate=vg)
        except Exception as e: print(f"{nm}: ERR {e}"); continue
        if r is None: print(f"{nm}: insufficient data"); continue
        m = metr(r)
        if m: rows[nm] = (r, m)
    items = sorted(rows.items(), key=lambda kv: -kv[1][1]["gp"])
    print(f"SEASONAL CRACK MR | volgate={vg} | full history from {START} | sorted by gain/pain")
    print(f"{'crack':<11}{'yrs':>5}{'Sh':>6}{'H1':>6}{'H2':>6}{'eqR2':>6}{'G/P':>6}  smooth?")
    nsm = 0; smooth = []
    for nm, (r, m) in items:
        ok = m['a'] > 0 and m['b'] > 0 and m['gp'] >= 2.5 and m['r2'] >= 0.80
        nsm += ok; smooth += [nm] if ok else []
        print(f"{nm:<11}{m['yrs']:>5.0f}{m['sh']:>6.2f}{m['a']:>6.2f}{m['b']:>6.2f}{m['r2']:>6.2f}{m['gp']:>6.1f}"
              f"  {'SMOOTH' if ok else ('+' if m['a']>0 and m['b']>0 else '')}")
    print(f"\nboth-halves positive: {sum(1 for _,(_,m) in items if m['a']>0 and m['b']>0)}/{len(items)}")
    print(f"SMOOTH crack spreads (eqR2>=.80, gp>=2.5, both halves+): {nsm}: {smooth}")
