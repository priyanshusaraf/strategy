#!/usr/bin/env python3
"""4-YEAR hourly test — uses the longer TV exports (mean-reversion-data, 2022-2026) merged
with Yahoo (data_hourly, 2024-2026) to maximize per-security history. Tests whether more
hourly history pushes individual spreads over the smoothness bar.

Mandate note: BRN-WTI is an economically-linked CRUDE QUALITY/LOCATION crack (Brent-WTI
differential) — on-mandate. Metals ratios included for diversification context.
Bars are aligned to the hourly grid by flooring to the hour; legs intersected on timestamp.
"""
import math, os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TV = os.path.expanduser("~/Downloads/mean-reversion-data")
YH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_hourly")

def load_tv(fname):
    df = pd.read_csv(f"{TV}/{fname}")
    df.columns = [c.strip().lower() for c in df.columns]
    df["t"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor("h")
    return df.dropna(subset=["t", "open", "close"]).set_index("t")[["open", "close"]].sort_index()

def load_yh(sym):
    f = f"{YH}/{sym}.csv"
    if not os.path.exists(f): return None
    df = pd.read_csv(f); df["t"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("h")
    return df.dropna(subset=["t", "open", "close"]).set_index("t")[["open", "close"]].sort_index()

def merge_leg(tv_file, yh_sym):
    """Prefer TV (longer); extend with Yahoo only where TV missing (usually not needed)."""
    parts = []
    if tv_file: parts.append(load_tv(tv_file))
    if yh_sym:
        y = load_yh(yh_sym)
        if y is not None: parts.append(y)
    if not parts: return None
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="first")].sort_index()

# legs available at long (TV) history
LEG = {
    "BRN": merge_leg("ICEEUR_DLY_BRN1!, 60 (1).csv", "BZ"),
    "WTI": merge_leg("CFI_WTI, 60.csv", "CL"),
    "GC":  merge_leg("COMEX_DL_GC2!, 60.csv", None),
    "SI":  merge_leg("COMEX_DL_SI2!, 60.csv", None),
    "HG":  merge_leg("COMEX_DL_HG1!, 60.csv", "HG"),
    "PA":  merge_leg("NYMEX_DL_PA1!, 60.csv", "PA"),
    "PL":  merge_leg(None, "PL"),
    "ZC":  merge_leg("CBOT_DL_ZC1!, 60.csv", "ZC"),
}

SPREADS = {  # (legs, weights, is_crack)
    "BRN_WTI":  (["BRN", "WTI"], [1, -1], True),      # crude quality crack (on-mandate)
    "GC_SI":    (["GC", "SI"], [1, None], False),
    "SI_HG":    (["SI", "HG"], [1, None], False),
    "GC_HG":    (["GC", "HG"], [1, None], False),
    "GC_PA":    (["GC", "PA"], [1, None], False),
    "SI_PA":    (["SI", "PA"], [1, None], False),
    "HG_PA":    (["HG", "PA"], [1, None], False),
    "PL_PA":    (["PL", "PA"], [1, None], False),
    "GC_PL":    (["GC", "PL"], [1, None], False),
    "SI_PL":    (["SI", "PL"], [1, None], False),
}

ZDS = [9, 12, 18, 24]; ZES = [2.5, 3.0]

def build(legs, w):
    dfs = [LEG[l] for l in legs]
    if any(d is None for d in dfs): return None
    idx = dfs[0].index
    for d in dfs[1:]: idx = idx.intersection(d.index)
    if len(idx) < 3000: return None
    O = np.column_stack([d.loc[idx, "open"].values for d in dfs])
    C = np.column_stack([d.loc[idx, "close"].values for d in dfs])
    w = list(w)
    if w[1] is None:
        n0 = min(2000, len(idx)//3)
        w[1] = -np.nanstd(np.diff(C[:n0,0]))/np.nanstd(np.diff(C[:n0,1]))
    W = np.array(w, float)
    return pd.DataFrame({"o": O@W, "c": C@W}, index=idx)

def one(F, zd, ze, volgate=True):
    F = F.dropna(); dS = F["c"].diff()
    bars_day = len(F)/max(1,len(np.unique(F.index.date)))
    z_n = max(60, int(zd*bars_day)); dt = max(100, int(28*bars_day)); ts = int(10*bars_day)
    mu = dS.shift(1).rolling(dt, min_periods=dt//2).mean().fillna(0.0)
    S = (dS-mu).cumsum()
    m = S.shift(1).rolling(z_n).mean(); sd = S.shift(1).rolling(z_n).std(); z=((S-m)/sd).values
    n=len(F)
    if volgate:
        sf=dS.ewm(span=int(5*bars_day),min_periods=50).std(); ss=sf.rolling(int(126*bars_day),min_periods=500).median()
        hot=(sf>ss).values
    else: hot=np.ones(n,bool)
    sig=dS.ewm(span=int(63*bars_day),min_periods=100).std(); ssafe=sig.replace(0,np.nan).ffill().bfill().values
    w=np.zeros(n); cur=0; held=0
    for i in range(z_n+2,n):
        zz=z[i]
        if cur!=0:
            held+=1
            if (zz!=zz or abs(zz)>=4.0 or held>=ts or (cur>0 and zz>=-0.75) or (cur<0 and zz<=0.75)):
                cur,held=0,0
        if cur==0 and zz==zz and hot[i]:
            if zz<=-ze: cur,held=1,0
            elif zz>=ze: cur,held=-1,0
        w[i]=cur
    u=np.zeros(n)
    for i in range(1,n):
        if w[i]==0: u[i]=0.0
        elif w[i-1]==0 or np.sign(w[i])!=np.sign(w[i-1]): u[i]=w[i]/ssafe[i]
        else: u[i]=u[i-1]
    u=pd.Series(u,index=F.index); du=u.diff().abs().fillna(0.0)
    pnl=u.shift(1)*dS - 0.05*sig.shift(1)*du
    bpy=len(F)/((F.index.max()-F.index.min()).days/365.25)
    return pnl.fillna(0.0)*(0.10/math.sqrt(bpy)), bpy

def ensemble(nm):
    legs,w,_=SPREADS[nm]; F=build(legs,w)
    if F is None: return None
    parts=[]; bpy=None
    for zd in ZDS:
        for ze in ZES:
            r,bpy=one(F,zd,ze); parts.append(r)
    return pd.concat(parts,axis=1).fillna(0.0).mean(axis=1), bpy

def metr(r,bpy):
    nz=r[r!=0]
    if len(nz)<200: return None
    r=r.loc[nz.index[0]:]
    if r.std()>0: r=r*(0.10/math.sqrt(bpy))/r.std()
    eq=r.cumsum(); dd=(eq-eq.cummax()).min()
    x=np.arange(len(eq)); b1,b0=np.polyfit(x,eq.values,1)
    ssr=((eq.values-(b1*x+b0))**2).sum(); sst=max(((eq.values-eq.values.mean())**2).sum(),1e-12)
    h=len(r)//2; sa,sb=r.iloc[:h],r.iloc[h:]
    return dict(eq=eq, sh=r.mean()/r.std()*math.sqrt(bpy), gp=eq.iloc[-1]/max(abs(dd),1e-9),
                r2=1-ssr/sst, dd=dd, yrs=(r.index.max()-r.index.min()).days/365.25,
                a=sa.mean()/sa.std()*math.sqrt(bpy) if sa.std()>0 else 0,
                b=sb.mean()/sb.std()*math.sqrt(bpy) if sb.std()>0 else 0)

if __name__=="__main__":
    print("4-YEAR hourly (TV+Yahoo merged) | frozen vol-gated ensemble")
    print(f"{'spread':<9}{'crack':>6}{'yrs':>5}{'Sh':>6}{'H1':>6}{'H2':>6}{'eqR2':>6}{'G/P':>6}  smooth?")
    rows={}
    for nm in SPREADS:
        out=ensemble(nm)
        if out is None: print(f"{nm:<9} insufficient data"); continue
        r,bpy=out; m=metr(r,bpy)
        if m is None: continue
        rows[nm]=(r,m,bpy)
        ok=m['a']>0 and m['b']>0 and m['gp']>=2.5 and m['r2']>=0.6
        iscrk='yes' if SPREADS[nm][2] else ''
        print(f"{nm:<9}{iscrk:>6}{m['yrs']:>5.1f}{m['sh']:>6.2f}{m['a']:>6.2f}{m['b']:>6.2f}{m['r2']:>6.2f}{m['gp']:>6.1f}"
              f"  {'SMOOTH' if ok else ('+' if m['a']>0 and m['b']>0 else '')}")
    nsm=sum(1 for _,(r,m,_) in rows.items() if m['a']>0 and m['b']>0 and m['gp']>=2.5 and m['r2']>=0.6)
    print(f"\nsmooth (both halves+, gp>=2.5, eqR2>=0.6) over ~4y: {nsm}")
