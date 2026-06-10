#!/usr/bin/env python3
"""BROADENED economically-linked MR book — reach 10 smooth securities.
Anchor sleeve: commodity cracks (WTI-BRENT + BTU) from Dukascopy (duka_cracks).
Breadth sleeve: tightly-cointegrated FX crosses & relative-value FX spreads (data_fx),
each economically linked (interest-rate parity, trade flows, neighboring/commodity economies).
Same frozen vol-gated OU rule throughout. Smoothness judged by gain/pain + eqR2 + both halves.
"""
import math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fx")
DUKA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_duka")
_c = {}
def load(d, sym):
    k = (d, sym)
    if k in _c: return _c[k]
    df = pd.read_csv(f"{d}/{sym}.csv")
    df["t"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("h")
    s = df.dropna(subset=["t","open","close"]).set_index("t")[["open","close"]].sort_index()
    s = s[~s.index.duplicated(keep="first")]; s = s[(s.open>0)&(s.close>0)]
    _c[k] = s; return s

# SECURITIES: each is a single cointegrated series (FX cross) or a 2-leg relative-value spread.
# Economically-linked rationale in comments.
def fx(sym):   return ("single", load(FX, sym))
def fxspread(a, b):  return ("spread", load(FX, a), load(FX, b))   # vol-matched relative value
def crack(a, b):     return ("spread", load(DUKA, a), load(DUKA, b))
def xasset(fxsym, dukasym):  return ("spread", load(FX, fxsym), load(DUKA, dukasym))  # FX vs commodity

SECURITIES = {
    # --- anchor: commodity cracks (on-mandate) ---
    "WTI_BRENT":   lambda: crack("WTI","BRENT"),      # crude grade crack
    "GAS_WTI":     lambda: crack("GAS","WTI"),        # BTU
    "GAS_BRENT":   lambda: crack("GAS","BRENT"),      # BTU
    # --- breadth: tightly cointegrated FX crosses (each a mean-reverting security) ---
    "AUDNZD":      lambda: fx("AUDNZD"),    # two commodity currencies, similar economies
    "EURGBP":      lambda: fx("EURGBP"),    # neighbouring trade-linked economies
    "EURCHF":      lambda: fx("EURCHF"),    # CHF managed vs EUR (classic reverter)
    "EURNOK":      lambda: fx("EURNOK"),    # Scandi / oil economy vs EUR
    "EURSEK":      lambda: fx("EURSEK"),    # Scandi vs EUR
    "USDCAD":      lambda: fx("USDCAD"),    # USD vs oil-exporter CAD
    "GBPCHF":      lambda: fx("GBPCHF"),
    "CHFJPY":      lambda: fx("CHFJPY"),
    "NZDCAD":      lambda: fx("NZDCAD"),    # two commodity currencies
    "AUDCAD":      lambda: fx("AUDCAD"),    # two commodity currencies
    "GBPJPY":      lambda: fx("GBPJPY"),
    "EURJPY":      lambda: fx("EURJPY"),
    "USDNOK":      lambda: fx("USDNOK"),
    "USDSEK":      lambda: fx("USDSEK"),
    "EURSEK":      lambda: fx("EURSEK"),
    "EURNOK":      lambda: fx("EURNOK"),
    "EURGBP":      lambda: fx("EURGBP"),
    "EURAUD":      lambda: fx("EURAUD"),
    # --- relative-value FX spreads (Scandi RV, commodity-pair RV, cross via JPY) ---
    "NOK_SEK":     lambda: fxspread("EURNOK","EURSEK"),   # Scandi rates/oil spread
    "USD_NOKSEK":  lambda: fxspread("USDNOK","USDSEK"),   # Scandi RV vs USD
    "AUD_NZD_usd": lambda: fxspread("AUDUSD","NZDUSD"),   # commodity-pair RV vs USD
    "EUR_GBP_jpy": lambda: fxspread("EURJPY","GBPJPY"),   # EUR vs GBP via JPY
    "AUD_CAD_usd": lambda: fxspread("AUDUSD","USDCAD"),   # two commodity FX
    # --- cross-asset commodity-currency cointegration (FX vs the commodity it tracks) ---
    "CAD_WTI":     lambda: xasset("USDCAD","WTI"),        # CAD vs crude (oil exporter)
    "NOK_BRENT":   lambda: xasset("USDNOK","BRENT"),      # NOK vs Brent (oil exporter)
    "AUD_COPPER":  lambda: xasset("AUDUSD","COPPER"),     # AUD vs copper (metals exporter)
    "AUD_XAU":     lambda: xasset("AUDUSD","XAU"),        # AUD vs gold
    # --- batch 2: range-bound / managed cointegrated crosses (linked economies) ---
    "EURCAD":      lambda: fx("EURCAD"),
    "EURNZD":      lambda: fx("EURNZD"),
    "GBPAUD":      lambda: fx("GBPAUD"),
    "GBPCAD":      lambda: fx("GBPCAD"),
    "GBPNZD":      lambda: fx("GBPNZD"),
    "AUDCHF":      lambda: fx("AUDCHF"),
    "CADCHF":      lambda: fx("CADCHF"),
    "NZDCHF":      lambda: fx("NZDCHF"),
    "NZDCAD":      lambda: fx("NZDCAD"),     # two commodity currencies
    "GBPCHF":      lambda: fx("GBPCHF"),
    "EURPLN":      lambda: fx("EURPLN"),     # EU-linked, managed/range-bound
    "EURCZK":      lambda: fx("EURCZK"),     # EU-linked, managed/range-bound
    "EURHUF":      lambda: fx("EURHUF"),     # EU-linked, managed/range-bound
    "AUDSGD":      lambda: fx("AUDSGD"),
    # batch-2 relative-value spreads
    "AUD_NZD_chf": lambda: fxspread("AUDCHF","NZDCHF"),   # commodity-pair RV via CHF
    "GBP_EUR_cad": lambda: fxspread("GBPCAD","EURCAD"),   # GBP vs EUR via CAD
    "GBP_AUD_nzd": lambda: fxspread("GBPAUD","GBPNZD"),   # AUD vs NZD via GBP
}

ZDS = [9,12,18,24]; ZES = [2.5,3.0]

def make_F(spec):
    if spec[0] == "single":
        return spec[1].rename(columns={"open":"o","close":"c"})
    _, A, B = spec
    idx = A.index.intersection(B.index)
    if len(idx) < 5000: return None
    Ca, Cb = A.loc[idx,"close"].values, B.loc[idx,"close"].values
    Oa, Ob = A.loc[idx,"open"].values, B.loc[idx,"open"].values
    n0 = min(8000, len(idx)//4)
    w = -np.nanstd(np.diff(Ca[:n0]))/np.nanstd(np.diff(Cb[:n0]))
    return pd.DataFrame({"o": Oa + w*Ob, "c": Ca + w*Cb}, index=idx)

def one(F, zd, ze):
    F = F.dropna(); dS = F["c"].diff(); bd = len(F)/max(1,len(np.unique(F.index.date)))
    z_n = max(60,int(zd*bd)); dt = max(120,int(28*bd)); ts = int(10*bd)
    mu = dS.shift(1).rolling(dt,min_periods=dt//2).mean().fillna(0); S=(dS-mu).cumsum()
    m = S.shift(1).rolling(z_n).mean(); sd = S.shift(1).rolling(z_n).std(); z=((S-m)/sd).values
    sf = dS.ewm(span=int(5*bd),min_periods=50).std(); ss = sf.rolling(int(126*bd),min_periods=800).median()
    hot = (sf>ss).values
    sig = dS.ewm(span=int(63*bd),min_periods=200).std(); sa = sig.replace(0,np.nan).ffill().bfill().values
    n=len(F); w=np.zeros(n); cur=0; held=0
    for i in range(z_n+2,n):
        zz=z[i]
        if cur!=0:
            held+=1
            if zz!=zz or abs(zz)>=4 or held>=ts or (cur>0 and zz>=-0.75) or (cur<0 and zz<=0.75): cur,held=0,0
        if cur==0 and zz==zz and hot[i]:
            if zz<=-ze: cur,held=1,0
            elif zz>=ze: cur,held=-1,0
        w[i]=cur
    u=np.zeros(n)
    for i in range(1,n):
        if w[i]==0: u[i]=0
        elif w[i-1]==0 or np.sign(w[i])!=np.sign(w[i-1]): u[i]=w[i]/sa[i]
        else: u[i]=u[i-1]
    u=pd.Series(u,index=F.index); du=u.diff().abs().fillna(0)
    pnl=u.shift(1)*dS - 0.05*sig.shift(1)*du
    bpy=len(F)/((F.index.max()-F.index.min()).days/365.25)
    return pnl.fillna(0)*(0.10/math.sqrt(bpy)), bpy

def ensemble(spec):
    F = make_F(spec)
    if F is None or len(F) < 5000: return None
    parts=[]; bpy=None
    for zd in ZDS:
        for ze in ZES:
            r,bpy = one(F,zd,ze); parts.append(r)
    return pd.concat(parts,axis=1).fillna(0).mean(axis=1), bpy

def metr(r,bpy):
    nz=r[r!=0]
    if len(nz)<400: return None
    r=r.loc[nz.index[0]:]
    if r.std()>0: r=r*(0.10/math.sqrt(bpy))/r.std()
    eq=r.cumsum(); dd=(eq-eq.cummax()).min()
    x=np.arange(len(eq)); b1,b0=np.polyfit(x,eq.values,1)
    ssr=((eq.values-(b1*x+b0))**2).sum(); sst=max(((eq.values-eq.values.mean())**2).sum(),1e-12)
    h=len(r)//2; sa,sb=r.iloc[:h],r.iloc[h:]
    return dict(eq=eq,r=r,sh=r.mean()/r.std()*math.sqrt(bpy),gp=eq.iloc[-1]/max(abs(dd),1e-9),
                r2=1-ssr/sst,dd=dd,yrs=(r.index.max()-r.index.min()).days/365.25,
                a=sa.mean()/sa.std()*math.sqrt(bpy) if sa.std()>0 else 0,
                b=sb.mean()/sb.std()*math.sqrt(bpy) if sb.std()>0 else 0)

if __name__ == "__main__":
    rows={}
    for nm,fn in SECURITIES.items():
        try: out=ensemble(fn())
        except Exception as e: print(f"{nm}: ERR {e}"); continue
        if out is None: print(f"{nm}: insufficient data"); continue
        r,bpy=out; m=metr(r,bpy)
        if m: rows[nm]=(r,m)
    items=sorted(rows.items(), key=lambda kv:-kv[1][1]['gp'])
    # "smooth upward, nearly no downside" = both halves positive (robust) AND eqR2>=0.75
    # (curve >=75% a straight upward line) AND gain/pain>=2.0 (total profit >= 2x worst DD).
    def is_smooth(m): return m['a']>0 and m['b']>0 and m['gp']>=2.0 and m['r2']>=0.75
    print(f"{'security':<13}{'yrs':>5}{'Sh':>6}{'H1':>6}{'H2':>6}{'eqR2':>6}{'G/P':>6}  smooth?")
    smooth=[]
    for nm,(r,m) in items:
        ok = is_smooth(m)
        if ok: smooth.append(nm)
        print(f"{nm:<13}{m['yrs']:>5.1f}{m['sh']:>6.2f}{m['a']:>6.2f}{m['b']:>6.2f}{m['r2']:>6.2f}{m['gp']:>6.1f}"
              f"  {'SMOOTH' if ok else ('+' if m['a']>0 and m['b']>0 else '')}")
    npos=sum(1 for _,(r,m) in items if m['a']>0 and m['b']>0)
    print(f"\npositive both halves: {npos}/{len(items)}")
    print(f"SMOOTH (both halves+, gp>=2.0, eqR2>=0.75): {len(smooth)}: {smooth}")
    # equal-risk book of the smooth set + figure
    if smooth:
        B=pd.DataFrame({nm:rows[nm][0] for nm in smooth})
        iv={nm:1/B[nm].shift(1).rolling(500,min_periods=100).std().replace(0,np.nan) for nm in smooth}
        W=pd.DataFrame(iv).replace([np.inf],np.nan); W=W.div(W.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        book=(W*B).sum(axis=1); mb=metr(book, 252*24)
        if mb: print(f"BOOK of smooth set: Sharpe={mb['sh']:.2f} eqR2={mb['r2']:.2f} gain/pain={mb['gp']:.1f} maxDD={mb['dd']*100:.1f}%")
        show=smooth[:12]; cols=3; rn=math.ceil((len(show)+1)/cols)
        fig,ax=plt.subplots(rn,cols,figsize=(15,2.6*rn)); ax=ax.flatten()
        for k,nm in enumerate(show):
            m=rows[nm][1]; eq=m['eq']; h=len(eq)//2
            ax[k].plot(eq.index,eq.values,lw=1.0,color='darkgreen'); ax[k].axhline(0,color='red',ls=':',lw=0.6)
            ax[k].axvline(eq.index[h],color='gray',ls='--',lw=0.6); ax[k].fill_between(eq.index,eq.values,0,alpha=0.08,color='green')
            ax[k].set_title(f"{nm} Sh={m['sh']:.2f}(H1 {m['a']:.1f}/H2 {m['b']:.1f}) R2={m['r2']:.2f} G/P={m['gp']:.1f}",fontsize=8.5)
            ax[k].tick_params(labelsize=7)
        if mb:
            eq=mb['eq']; ax[len(show)].plot(eq.index,eq.values,lw=1.4,color='black')
            ax[len(show)].set_title(f"BOOK Sh={mb['sh']:.2f} R2={mb['r2']:.2f} G/P={mb['gp']:.1f}",fontsize=9,fontweight='bold')
        for j in range(len(show)+1,len(ax)): ax[j].axis('off')
        fig.suptitle(f"Broadened economically-linked MR — {len(smooth)} smooth securities (cracks + cointegrated FX)",fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__),"curves","broad_book.png"),dpi=110)
        print("-> curves/broad_book.png")
