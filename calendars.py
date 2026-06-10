#!/usr/bin/env python3
"""SEASONAL CALENDAR-SPREAD mean reversion — the most persistent, smoothest economic edge.

Why calendars: a calendar spread (front - next contract of the SAME commodity) is pinned by
storage economics — cash-and-carry arbitrage caps contango, stockout/convenience-yield caps
backwardation — and it carries a strong, repeating SEASONAL shape (injection/withdrawal for
NG, harvest for grains, placement cycles for livestock). That structural anchor is why the
36y daily test found calendars dominating the both-halves-positive, smooth set. This engine
adds the seasonal anchor the generic z-score lacks.

Signal:  S = detrended cumulative spread level.
         seasonal(t) = causal day-of-year mean of S over PRIOR years (>=3 yrs history).
         z = (S - seasonal) / rolling_std   -> deviation from the SEASONAL norm, not a flat mean.
Rest of the construction is the frozen rule (horizon ensemble {10,15,20,30}, entry 2.5/3.0,
exit 0.75, stops 4/15, per-trade sizing, vol gate, close exec for same-venue calendars).
Smoothness judged horizon-fairly by gain/pain (Calmar) + eqR2, plus BOTH-halves-positive.
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
ZDS = [10, 15, 20, 30]
ZES = [2.5, 3.0]
LAM = 0.05

# every storable-commodity calendar available with long history (economic: storage/carry)
CALENDARS = [
    "NG_cal", "CL_cal", "RB_cal", "HO_cal", "BRN_cal", "ULS_cal",      # energy storage
    "ZC_cal", "ZW_cal", "ZS_cal", "ZL_cal", "ZM_cal", "KE_cal", "ZO_cal",  # grain/oilseed harvest
    "LE_cal", "HE_cal", "GF_cal",                                       # livestock placement cycle
    "GC_cal", "SI_cal", "HG_cal", "PL_cal", "PA_cal",                   # metals carry
    "CC_cal", "CT_cal", "SB_cal", "KC_cal", "OJ_cal",                   # softs storage
]

def seasonal_anchor(S, years=8, min_years=3, tol=10):
    """Causal day-of-year mean of S over the prior `years` years (only past data)."""
    idx = S.index
    vals = S.values
    rows = []
    for k in range(1, years + 1):
        tgt = idx - pd.DateOffset(years=k)
        pos = np.clip(idx.searchsorted(tgt), 0, len(idx) - 1)
        good = np.abs((idx[pos] - tgt).days) <= tol
        rows.append(np.where(good, vals[pos], np.nan))
    mat = np.vstack(rows)
    cnt = (~np.isnan(mat)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        seas = np.where(cnt >= min_years, np.nanmean(mat, axis=0), np.nan)
    return pd.Series(seas, index=idx)

def one(F, zd, ze, seasonal=True, volgate=True):
    F = F.dropna(); F = F[F.index >= START]
    if len(F) < 2500:
        return None
    dS = F["c"].diff()
    detrend = 28
    mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    if seasonal:
        anc = seasonal_anchor(S).shift(1)            # causal
        anc = anc.fillna(S.shift(1).rolling(zd).mean())   # fallback before 3y history
        dev = S - anc
    else:
        dev = S - S.shift(1).rolling(zd).mean()
    sd = dev.shift(1).rolling(zd).std()
    z = (dev / sd).values
    n = len(F)
    if volgate:
        sf = dS.ewm(span=5, min_periods=5).std(); ss = sf.rolling(126, min_periods=60).median()
        hot = (sf > ss).values
    else:
        hot = np.ones(n, bool)
    sig = dS.ewm(span=63, min_periods=20).std()
    sig_safe = sig.replace(0, np.nan).ffill().bfill().values
    w = np.zeros(n); cur = 0; held = 0
    for i in range(max(zd, detrend) + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= 4.0 or held >= 15
                    or (cur > 0 and zz >= -0.75) or (cur < 0 and zz <= 0.75)):
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:
            if zz <= -ze: cur, held = 1, 0
            elif zz >= ze: cur, held = -1, 0
        w[i] = cur
    u = np.zeros(n)
    for i in range(1, n):
        if w[i] == 0: u[i] = 0.0
        elif w[i - 1] == 0 or np.sign(w[i]) != np.sign(w[i - 1]): u[i] = w[i] / sig_safe[i]
        else: u[i] = u[i - 1]
    u = pd.Series(u, index=F.index)
    du = u.diff().abs().fillna(0.0)
    pnl = u.shift(1) * dS - LAM * sig.shift(1) * du
    return pnl.fillna(0.0) * (0.10 / math.sqrt(252))

def ensemble(nm, **kw):
    F = mrlab.UNIVERSE[nm]()
    parts = []
    for zd in ZDS:
        for ze in ZES:
            r = one(F, zd, ze, **kw)
            if r is None: return None
            parts.append(r)
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)

def metrics(r, vt=0.10):
    nz = r[r != 0]
    if len(nz) < 250: return None
    r = r.loc[nz.index[0]:]
    if r.std() > 0: r = r * (vt / math.sqrt(252)) / r.std()
    def block(rr):
        if len(rr) < 100 or rr.std() == 0: return dict(sh=0, dd=0, r2=0, gp=0)
        eq = rr.cumsum(); dd = (eq - eq.cummax()).min()
        x = np.arange(len(eq)); b1, b0 = np.polyfit(x, eq.values, 1)
        ssr = ((eq.values - (b1 * x + b0)) ** 2).sum()
        sst = max(((eq.values - eq.values.mean()) ** 2).sum(), 1e-12)
        return dict(sh=rr.mean()/rr.std()*math.sqrt(252), dd=dd, r2=1-ssr/sst, gp=eq.iloc[-1]/max(abs(dd),1e-9))
    h = len(r)//2
    return dict(full=block(r), IS=block(r.iloc[:h]), OOS=block(r.iloc[h:]), eq=r.cumsum(), r=r, yrs=len(r)/252)

if __name__ == "__main__":
    seasonal = "noseas" not in sys.argv
    rows, rets = [], {}
    for nm in CALENDARS:
        try:
            r = ensemble(nm, seasonal=seasonal, volgate="novg" not in sys.argv)
        except Exception as e:
            print(f"{nm}: ERR {e}"); continue
        if r is None: continue
        m = metrics(r)
        if m: rows.append((nm, m)); rets[nm] = r
    rows.sort(key=lambda x: -x[1]["full"]["gp"])
    print(f"SEASONAL CALENDAR MR | seasonal={seasonal} | start={START} | sorted by gain/pain")
    print(f"{'cal':<9}{'yrs':>5}{'fullSh':>8}{'ISsh':>7}{'OOSsh':>7}{'eqR2':>6}{'G/P':>6}")
    smooth = []
    for nm, m in rows:
        f, i, o = m["full"], m["IS"], m["OOS"]
        ok = i["sh"] > 0 and o["sh"] > 0 and f["gp"] >= 2.0 and f["r2"] >= 0.80
        if ok: smooth.append(nm)
        print(f"{nm:<9}{m['yrs']:>5.0f}{f['sh']:>8.2f}{i['sh']:>7.2f}{o['sh']:>7.2f}{f['r2']:>6.2f}{f['gp']:>6.1f}"
              f"{'  SMOOTH' if ok else (' +' if i['sh']>0 and o['sh']>0 else '')}")
    print(f"\nboth-halves positive: {sum(1 for _,m in rows if m['IS']['sh']>0 and m['OOS']['sh']>0)}/{len(rows)}")
    print(f"SMOOTH (both halves>0, gain>=2x DD, eqR2>=0.80): {len(smooth)}: {smooth}")

    show = smooth if len(smooth) >= 10 else [nm for nm, _ in rows[:12]]
    cols = 3; rn = math.ceil(len(show)/cols)
    fig, ax = plt.subplots(rn, cols, figsize=(15, 2.6*rn)); ax = ax.flatten(); md = dict(rows)
    for k, nm in enumerate(show):
        m = md[nm]; eq = m["eq"]; h = len(eq)//2; f = m["full"]
        ax[k].plot(eq.index, eq.values, lw=0.9, color="navy")
        ax[k].axvline(eq.index[h], color="gray", ls="--", lw=0.7); ax[k].axhline(0, color="red", ls=":", lw=0.6)
        ax[k].set_title(f"{nm} Sh={f['sh']:.2f} G/P={f['gp']:.1f} R2={f['r2']:.2f}", fontsize=9)
        ax[k].tick_params(labelsize=7)
    for j in range(len(show), len(ax)): ax[j].axis("off")
    fig.suptitle(f"Seasonal calendar-spread MR (seasonal={seasonal}) — per-security, full history", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/calendars.png", dpi=110)
    print(f"-> {OUT}/calendars.png")

    # ---- CALENDAR BOOK: equal-risk across ALL calendars (no selection), 36y ----
    # Many persistent storage spreads + long sample = the smoothest honest curve.
    B = pd.DataFrame(rets).sort_index()
    iv = {nm: 1.0 / B[nm].shift(1).rolling(63, min_periods=20).std().replace(0, np.nan)
          for nm in B.columns}
    W = pd.DataFrame(iv)[B.columns].replace([np.inf], np.nan)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book = (W * B).sum(axis=1)
    mb = metrics(book)
    f, i, o = mb["full"], mb["IS"], mb["OOS"]
    print(f"\n=== CALENDAR BOOK (all {len(B.columns)} calendars, {mb['yrs']:.0f}y, equal-risk) ===")
    print(f"  full Sharpe={f['sh']:.2f}  IS={i['sh']:.2f}  OOS={o['sh']:.2f}")
    print(f"  eqR2(smoothness)={f['r2']:.3f}  maxDD={f['dd']*100:.1f}%  gain/pain={f['gp']:.1f}")
    fig2, a2 = plt.subplots(figsize=(12, 4.5)); eq = mb["eq"]; h = len(eq) // 2
    a2.plot(eq.index, eq.values, lw=1.2, color="darkgreen")
    a2.axvline(eq.index[h], color="gray", ls="--", lw=0.8); a2.axhline(0, color="red", ls=":", lw=0.7)
    a2.set_title(f"CALENDAR BOOK ({len(B.columns)} storage spreads, {mb['yrs']:.0f}y, 10% vol)  "
                 f"Sharpe={f['sh']:.2f} IS={i['sh']:.2f} OOS={o['sh']:.2f}  "
                 f"eqR2={f['r2']:.2f} G/P={f['gp']:.1f}", fontsize=10)
    fig2.tight_layout(); fig2.savefig(f"{OUT}/calendar_book.png", dpi=120)
    print(f"-> {OUT}/calendar_book.png")
