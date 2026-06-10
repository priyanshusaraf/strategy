#!/usr/bin/env python3
"""12-YEAR hourly economically-linked spread MR on Dukascopy data (data_duka/).
The frozen vol-gated ensemble, applied to refined-product cracks, crude differentials,
BTU spreads, and metals ratios over ~12 years — long enough for genuinely smooth curves
(BRN-WTI was already smooth at 3y; 12y should smooth the whole crack complex).
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_duka")
_c = {}
def leg(sym):
    if sym in _c: return _c[sym]
    df = pd.read_csv(f"{DIR}/{sym}.csv")
    df["t"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("h")
    s = df.dropna(subset=["t", "open", "close"]).set_index("t")[["open", "close"]].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s[(s["close"] > 0) & (s["open"] > 0)]
    _c[sym] = s
    return s

# economically-linked spreads; weights None -> vol-matched on first ~1/4 sample
SPREADS = {
    # refined-product & crude cracks (ON-MANDATE)
    "DIESEL_WTI":  (["DIESEL", "WTI"], [None, None], True),    # distillate crack
    "DIESEL_BRENT":(["DIESEL", "BRENT"], [None, None], True),  # gasoil crack
    "BRENT_WTI":   (["BRENT", "WTI"], [1, -1], True),          # crude quality/location crack
    "GAS_WTI":     (["GAS", "WTI"], [None, None], True),       # BTU / fuel switching
    "GAS_BRENT":   (["GAS", "BRENT"], [None, None], True),     # BTU
    "DIESEL_GAS":  (["DIESEL", "GAS"], [None, None], True),    # distillate vs gas
    # metals relative value
    "XAU_XAG":     (["XAU", "XAG"], [1, None], False),
    "XAU_COPPER":  (["XAU", "COPPER"], [1, None], False),
    "XAG_COPPER":  (["XAG", "COPPER"], [1, None], False),
    # softs
    "COCOA_SUGAR": (["COCOA", "SUGAR"], [1, None], False),
}
ZDS = [9, 12, 18, 24]; ZES = [2.5, 3.0]

def build(legs, w):
    try:
        dfs = [leg(l) for l in legs]
    except FileNotFoundError:
        return None
    idx = dfs[0].index
    for d in dfs[1:]: idx = idx.intersection(d.index)
    if len(idx) < 5000: return None
    O = np.column_stack([d.loc[idx, "open"].values for d in dfs])
    C = np.column_stack([d.loc[idx, "close"].values for d in dfs])
    w = list(w)
    n0 = min(8000, len(idx)//4)
    if w[0] is None:
        w[0] = 1.0/np.nanstd(np.diff(C[:n0,0])); w[1] = -1.0/np.nanstd(np.diff(C[:n0,1]))
    elif w[1] is None:
        w[1] = -np.nanstd(np.diff(C[:n0,0]))/np.nanstd(np.diff(C[:n0,1]))
    W = np.array(w, float)
    return pd.DataFrame({"o": O@W, "c": C@W}, index=idx)

def one(F, zd, ze, volgate=True):
    F = F.dropna(); dS = F["c"].diff()
    bd = len(F)/max(1, len(np.unique(F.index.date)))
    z_n = max(80, int(zd*bd)); dt = max(150, int(28*bd)); ts = int(10*bd)
    mu = dS.shift(1).rolling(dt, min_periods=dt//2).mean().fillna(0.0)
    S = (dS-mu).cumsum()
    m = S.shift(1).rolling(z_n).mean(); sd = S.shift(1).rolling(z_n).std(); z = ((S-m)/sd).values
    n = len(F)
    if volgate:
        sf = dS.ewm(span=int(5*bd), min_periods=50).std(); ss = sf.rolling(int(126*bd), min_periods=800).median()
        hot = (sf>ss).values
    else: hot = np.ones(n, bool)
    sig = dS.ewm(span=int(63*bd), min_periods=200).std(); ssafe = sig.replace(0, np.nan).ffill().bfill().values
    w = np.zeros(n); cur = 0; held = 0
    for i in range(z_n+2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz!=zz or abs(zz)>=4.0 or held>=ts or (cur>0 and zz>=-0.75) or (cur<0 and zz<=0.75)):
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:
            if zz<=-ze: cur, held = 1, 0
            elif zz>=ze: cur, held = -1, 0
        w[i] = cur
    u = np.zeros(n)
    for i in range(1, n):
        if w[i]==0: u[i]=0.0
        elif w[i-1]==0 or np.sign(w[i])!=np.sign(w[i-1]): u[i]=w[i]/ssafe[i]
        else: u[i]=u[i-1]
    u = pd.Series(u, index=F.index); du = u.diff().abs().fillna(0.0)
    pnl = u.shift(1)*dS - 0.05*sig.shift(1)*du
    bpy = len(F)/((F.index.max()-F.index.min()).days/365.25)
    return pnl.fillna(0.0)*(0.10/math.sqrt(bpy)), bpy

def ensemble(nm):
    legs, w, _ = SPREADS[nm]; F = build(legs, w)
    if F is None: return None
    parts = []; bpy = None
    for zd in ZDS:
        for ze in ZES:
            r, bpy = one(F, zd, ze); parts.append(r)
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1), bpy

def metr(r, bpy):
    nz = r[r!=0]
    if len(nz) < 500: return None
    r = r.loc[nz.index[0]:]
    if r.std()>0: r = r*(0.10/math.sqrt(bpy))/r.std()
    eq = r.cumsum(); dd = (eq-eq.cummax()).min()
    x = np.arange(len(eq)); b1, b0 = np.polyfit(x, eq.values, 1)
    ssr = ((eq.values-(b1*x+b0))**2).sum(); sst = max(((eq.values-eq.values.mean())**2).sum(), 1e-12)
    h = len(r)//2; sa, sb = r.iloc[:h], r.iloc[h:]
    return dict(eq=eq, sh=r.mean()/r.std()*math.sqrt(bpy), gp=eq.iloc[-1]/max(abs(dd), 1e-9),
                r2=1-ssr/sst, dd=dd, yrs=(r.index.max()-r.index.min()).days/365.25,
                a=sa.mean()/sa.std()*math.sqrt(bpy) if sa.std()>0 else 0,
                b=sb.mean()/sb.std()*math.sqrt(bpy) if sb.std()>0 else 0)

if __name__ == "__main__":
    print("12-YEAR Dukascopy hourly | frozen vol-gated ensemble | on-mandate cracks first")
    print(f"{'spread':<13}{'crack':>6}{'yrs':>5}{'Sh':>6}{'H1':>6}{'H2':>6}{'eqR2':>6}{'G/P':>6}  smooth?")
    rows = {}
    for nm in SPREADS:
        out = ensemble(nm)
        if out is None: print(f"{nm:<13} insufficient data"); continue
        r, bpy = out; m = metr(r, bpy)
        if m is None: continue
        rows[nm] = (r, m)
        ok = m['a']>0 and m['b']>0 and m['gp']>=2.5 and m['r2']>=0.75
        crk = 'yes' if SPREADS[nm][2] else ''
        print(f"{nm:<13}{crk:>6}{m['yrs']:>5.1f}{m['sh']:>6.2f}{m['a']:>6.2f}{m['b']:>6.2f}{m['r2']:>6.2f}{m['gp']:>6.1f}"
              f"  {'SMOOTH' if ok else ('+' if m['a']>0 and m['b']>0 else '')}")
    nsm = [nm for nm,(r,m) in rows.items() if m['a']>0 and m['b']>0 and m['gp']>=2.5 and m['r2']>=0.75]
    npos = sum(1 for _,(r,m) in rows.items() if m['a']>0 and m['b']>0)
    print(f"\npositive both halves: {npos}/{len(rows)}")
    print(f"SMOOTH (both halves+, gp>=2.5, eqR2>=0.75): {len(nsm)}: {nsm}")
    # figure
    items = sorted(rows.items(), key=lambda kv:-kv[1][1]['gp'])
    fig, ax = plt.subplots(4, 3, figsize=(15, 11)); ax = ax.flatten()
    for k,(nm,(r,m)) in enumerate(items):
        eq = m['eq']; h = len(eq)//2
        sm = m['a']>0 and m['b']>0 and m['gp']>=2.5 and m['r2']>=0.75
        col = 'darkgreen' if sm else 'navy'
        ax[k].plot(eq.index, eq.values, lw=1.0, color=col); ax[k].axhline(0, color='red', ls=':', lw=0.6)
        ax[k].axvline(eq.index[h], color='gray', ls='--', lw=0.6)
        ax[k].fill_between(eq.index, eq.values, 0, alpha=0.08, color=col)
        crk='[crack] ' if SPREADS[nm][2] else ''
        ax[k].set_title(f"{nm} {crk}{'SMOOTH' if sm else ''}\nSh={m['sh']:.2f}(H1 {m['a']:.1f}/H2 {m['b']:.1f}) eqR2={m['r2']:.2f} G/P={m['gp']:.1f}", fontsize=8.5)
        ax[k].tick_params(labelsize=7)
    for j in range(len(items), len(ax)): ax[j].axis('off')
    fig.suptitle(f"12-year hourly economically-linked spread MR (Dukascopy) — {len(nsm)} smooth, {npos} positive both halves", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "curves", "duka_cracks.png"), dpi=110)
    print("-> curves/duka_cracks.png")
