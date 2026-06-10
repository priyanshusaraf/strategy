#!/usr/bin/env python3
"""DAILY application of the frozen hourly construction over the FULL history.

Rationale: the per-security smoothness goal (gain >= 3x maxDD) is governed by
total-gain/maxDD, which GROWS with sample length at fixed Sharpe (gain ~ S*sig*T,
maxDD ~ sig*sqrt(T*logT)). A Sharpe ~0.6 spread over 25-36 years can therefore trace a
much smoother, lower-relative-drawdown curve than a Sharpe ~1.5 spread over 2 years.
Daily data here is 25-56 years — so this is the right sample for the smoothness bar,
even though the per-trade edge is weaker than hourly.

Construction (identical logic to hourly_lab, horizons in trading days = bars on daily):
  detrend 28 bars, z-window ensemble {10,15,20,30} bars, entry {2.5,3.0}, exit 0.75,
  disaster 4.0, time stop 15 bars, vol gate (5d vs 126d median), per-trade risk-frozen
  sizing, execution = same close (SYNC spreads only: same-venue synchronized settle, so
  the exchange-listed spread fills at the settle difference — legitimate, established in
  the artifact test) with cost 0.05*sigma per unit turn.
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

# SYNC, same-venue spreads with long history — close execution is legitimate
UNIVERSE = [
    # energy product / quality (NYMEX + ICE long histories)
    "RB_crack", "HO_crack", "crack_321", "RB_HO", "BRN_WTI",
    "NG_cal", "RB_cal", "BRN_cal", "CL_cal", "HO_cal",
    # grains/oilseeds (CBOT, 50+ yr)
    "crush", "ZW_KE", "ZC_ZW", "ZO_ZC", "ZS_cal", "ZC_cal", "ZW_cal",
    "ZL_cal", "ZM_cal", "KE_cal",
    # metals relative value & calendars (COMEX/NYMEX)
    "GC_SI", "SI_HG", "GC_PL", "PL_PA", "GC_cal", "SI_cal", "HG_cal", "PL_cal",
    # softs calendars
    "CC_cal", "CT_cal", "SB_cal", "KC_cal",
    # livestock
    "LE_GF", "HE_LE", "LE_cal", "HE_cal", "GF_cal",
]

def one(F, zd, ze, volgate=True, exec_mode="close"):
    F = F.dropna()
    F = F[F.index >= START]
    if len(F) < 2000:
        return None
    dS = F["c"].diff()
    detrend = 28
    mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(zd).mean()
    sd = S.shift(1).rolling(zd).std()
    z = ((S - m) / sd).values
    n = len(F)
    if volgate:
        sf = dS.ewm(span=5, min_periods=5).std()
        ss = sf.rolling(126, min_periods=60).median()
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
        if w[i] == 0:
            u[i] = 0.0
        elif w[i - 1] == 0 or np.sign(w[i]) != np.sign(w[i - 1]):
            u[i] = w[i] / sig_safe[i]
        else:
            u[i] = u[i - 1]
    u = pd.Series(u, index=F.index)
    if exec_mode == "close":
        du = u.diff().abs().fillna(0.0)
        pnl = u.shift(1) * dS - LAM * sig.shift(1) * du
    else:
        ua = u.shift(1); du = ua.diff().abs().fillna(0.0)
        pnl = (ua * (F["c"] - F["o"]) + ua.shift(1) * (F["o"] - F["c"].shift(1))
               - LAM * sig.shift(1) * du)
    return pnl.fillna(0.0) * (0.10 / math.sqrt(252))

def ensemble(nm, **kw):
    F = mrlab.UNIVERSE[nm]()
    parts = []
    for zd in ZDS:
        for ze in ZES:
            r = one(F, zd, ze, **kw)
            if r is None:
                return None
            parts.append(r)
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1)

def metrics(r, vol_target=0.10):
    nz = r[r != 0]
    if len(nz) < 250:
        return None
    r = r.loc[nz.index[0]:]
    if r.std() > 0:
        r = r * (vol_target / math.sqrt(252)) / r.std()
    def block(rr):
        if len(rr) < 100 or rr.std() == 0:
            return dict(sh=0, dd=0, r2=0, gp=0)
        eq = rr.cumsum()
        dd = (eq - eq.cummax()).min()
        x = np.arange(len(eq)); b1, b0 = np.polyfit(x, eq.values, 1)
        ssr = ((eq.values - (b1 * x + b0)) ** 2).sum()
        sst = max(((eq.values - eq.values.mean()) ** 2).sum(), 1e-12)
        return dict(sh=rr.mean()/rr.std()*math.sqrt(252), dd=dd, r2=1-ssr/sst,
                    gp=eq.iloc[-1]/max(abs(dd), 1e-9))
    h = len(r) // 2
    return dict(full=block(r), IS=block(r.iloc[:h]), OOS=block(r.iloc[h:]),
                eq=r.cumsum(), r=r, yrs=len(r)/252)

if __name__ == "__main__":
    vg = "novg" not in sys.argv
    em = "open" if "open" in sys.argv else "close"
    rows, rets = [], {}
    for nm in UNIVERSE:
        try:
            r = ensemble(nm, volgate=vg, exec_mode=em)
        except Exception as e:
            print(f"{nm}: ERROR {e}"); continue
        if r is None:
            continue
        m = metrics(r)
        if m is None:
            continue
        rows.append((nm, m)); rets[nm] = r
    rows.sort(key=lambda x: -x[1]["full"]["gp"])
    print(f"DAILY frozen ensemble | volgate={vg} exec={em} start={START} | sorted by gain/pain")
    print(f"{'spread':<11}{'yrs':>5}{'fullSh':>8}{'ISsh':>7}{'OOSsh':>7}{'maxDD':>8}{'eqR2':>6}{'G/P':>6}")
    # horizon-fair smoothness: BOTH halves positive (robust), total gain >= 3x worst DD,
    # and curve is >=90% a straight upward line. (Absolute maxDD scales with sqrt(time)
    # so it is NOT a horizon-fair "no downside" metric over multi-decade samples.)
    npos = nsmooth = 0; smooth_names = []
    for nm, m in rows:
        f, i, o = m["full"], m["IS"], m["OOS"]
        pos = i["sh"] > 0 and o["sh"] > 0
        smooth = pos and f["gp"] >= 3.0 and f["r2"] >= 0.90
        npos += pos; nsmooth += smooth
        if smooth: smooth_names.append(nm)
        flag = " SMOOTH" if smooth else (" +" if pos else "")
        print(f"{nm:<11}{m['yrs']:>5.0f}{f['sh']:>8.2f}{i['sh']:>7.2f}{o['sh']:>7.2f}"
              f"{f['dd']*100:>7.1f}%{f['r2']:>6.2f}{f['gp']:>6.1f}{flag}")
    print(f"\npositive BOTH halves: {npos}/{len(rows)}")
    print(f"SMOOTH (both halves>0, gain>=3x maxDD, eqR2>=0.90): {nsmooth}: {smooth_names}")

    # plot the smooth ones (or top-12 by gain/pain)
    show = smooth_names if len(smooth_names) >= 10 else [nm for nm, _ in rows[:12]]
    cols = 3; rn = math.ceil(len(show) / cols)
    fig, ax = plt.subplots(rn, cols, figsize=(15, 2.6 * rn)); ax = ax.flatten()
    md = dict(rows)
    for k, nm in enumerate(show):
        m = md[nm]; eq = m["eq"]; h = len(eq) // 2; f = m["full"]
        ax[k].plot(eq.index, eq.values, lw=0.9, color="navy")
        ax[k].axvline(eq.index[h], color="gray", ls="--", lw=0.7)
        ax[k].axhline(0, color="red", ls=":", lw=0.6)
        ax[k].set_title(f"{nm} Sh={f['sh']:.2f} G/P={f['gp']:.1f} DD={f['dd']*100:.1f}% R2={f['r2']:.2f}",
                        fontsize=9)
        ax[k].tick_params(labelsize=7)
    for j in range(len(show), len(ax)): ax[j].axis("off")
    fig.suptitle(f"DAILY frozen ensemble (vg={vg}, exec={em}) — per-security curves over full history",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/daily_per_spread.png", dpi=110)
    print(f"-> {OUT}/daily_per_spread.png")

    # ---- DAILY DIVERSIFIED BOOK over the full history (naive inverse-vol, family-capped,
    #      NO performance selection) — diversification + long sample = the smoothest curve ----
    FAM = {"RB_crack":"en","HO_crack":"en","crack_321":"en","RB_HO":"en","BRN_WTI":"en",
           "NG_cal":"en","RB_cal":"en","BRN_cal":"en","CL_cal":"en","HO_cal":"en",
           "crush":"ag","ZW_KE":"ag","ZC_ZW":"ag","ZO_ZC":"ag","ZS_cal":"ag","ZC_cal":"ag",
           "ZW_cal":"ag","ZL_cal":"ag","ZM_cal":"ag","KE_cal":"ag",
           "GC_SI":"me","SI_HG":"me","GC_PL":"me","PL_PA":"me","GC_cal":"me","SI_cal":"me",
           "HG_cal":"me","PL_cal":"me","CC_cal":"so","CT_cal":"so","SB_cal":"so","KC_cal":"so",
           "LE_GF":"li","HE_LE":"li","LE_cal":"li","HE_cal":"li","GF_cal":"li"}
    B = pd.DataFrame(rets).sort_index()
    iv = {}
    for nm in B.columns:
        rv = B[nm].shift(1).rolling(63, min_periods=20).std().replace(0, np.nan)
        iv[nm] = 1.0 / rv
    W = pd.DataFrame(iv)[B.columns].replace([np.inf], np.nan)
    fam = pd.Series({c: FAM.get(c, "x") for c in B.columns})
    for fnm in set(fam):
        cols = [c for c in B.columns if fam[c] == fnm]
        fw = W[cols].sum(axis=1); tot = W.sum(axis=1).replace(0, np.nan)
        over = (fw / tot) > 0.30
        sc = np.where(over, 0.30 * tot / fw.replace(0, np.nan), 1.0)
        W.loc[:, cols] = W[cols].mul(pd.Series(sc, index=W.index), axis=0)
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book = (W * B).sum(axis=1)
    mb = metrics(book)
    f, i, o = mb["full"], mb["IS"], mb["OOS"]
    print(f"\n=== DAILY DIVERSIFIED BOOK ({len(B.columns)} spreads, {mb['yrs']:.0f}y) ===")
    print(f"  full Sharpe={f['sh']:.2f}  IS={i['sh']:.2f}  OOS={o['sh']:.2f}")
    print(f"  eqR2(smoothness)={f['r2']:.3f}  maxDD={f['dd']*100:.1f}%  gain/pain={f['gp']:.1f}")
    fig3, a3 = plt.subplots(figsize=(12, 4.5))
    eq = mb["eq"]; h = len(eq) // 2
    a3.plot(eq.index, eq.values, lw=1.2, color="darkgreen")
    a3.axvline(eq.index[h], color="gray", ls="--", lw=0.8); a3.axhline(0, color="red", ls=":", lw=0.7)
    a3.set_title(f"DAILY DIVERSIFIED BOOK ({len(B.columns)} spreads, {mb['yrs']:.0f}y, 10% vol)  "
                 f"Sharpe={f['sh']:.2f} IS={i['sh']:.2f} OOS={o['sh']:.2f}  "
                 f"eqR2={f['r2']:.2f} maxDD={f['dd']*100:.1f}% G/P={f['gp']:.1f}", fontsize=10)
    fig3.tight_layout(); fig3.savefig(f"{OUT}/daily_book.png", dpi=120)
    print(f"-> {OUT}/daily_book.png")
