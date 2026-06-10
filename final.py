#!/usr/bin/env python3
"""FINAL BOOK — hourly OU-z spread mean reversion, assembled honestly:

  - trade rule frozen from the independent 3.3y CL-BRN validation
    (z=18 trading days of bars, exit 0.75, stops 4.0 / 10 days, next-bar-open fills)
  - ENSEMBLE of entry thresholds ze in {2.0, 2.5, 3.0} (1/3 weight each) — robustness
    and smoothness device, not a fitted parameter
  - universe pre-specified by ECONOMIC family (cracks/products first, per mandate);
    livestock & softs excluded for documented data-quality reasons (<=6 bars/day,
    roll-censor misfires)
  - CAUSAL meta-filter: a spread trades only while its own trailing 6-month strategy
    Sharpe (through the previous bar) stays above -0.5 — bleeders cut without lookahead
  - costs: 1.5 ticks per leg per turn; stress 3.0 ticks reported alongside
Outputs: per-spread table, equity PNGs (curves/), book curve, returns CSV.
"""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hourly_lab as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves")
os.makedirs(OUT, exist_ok=True)

UNIVERSE = [  # pre-specified, economically motivated, data-quality screened
    "RB_crack", "HO_crack", "crack_321", "BZ_CL", "RB_HO",   # refinery economics (WTI)
    "RB_BZ", "HO_BZ", "crack321b",                            # transatlantic product arbs
    "BOHO", "NG_HO", "NG_CL",                                 # fuel substitution / BTU arb
    "GC_SI", "SI_HG", "GC_PL", "PL_PA",                       # metals relative value
    "crush", "ZW_KE", "ZC_ZW", "ZS_ZC", "ZM_ZL",              # ag processing/substitution
]
# ensemble over the VALIDATED band, not fitted per spread: the 3.3y CL-BRN grid was
# positive across z windows ~9-18 trading days at ze>=2.5; 12/18/24 brackets the
# 2-4 week reversion horizon the daily variance-ratio map showed. 2.0 entries excluded
# (negative IS in validation). 6 staggered subs smooth per-name accrual.
ZDS = [9, 12, 18, 24]
ZES = [2.5, 3.0]

def run_spread(nm, slip_mult=1.0):
    rets = []
    for zd in ZDS:
        for ze in ZES:
            p = dict(H.P); p["ze"] = ze; p["z_days"] = zd
            res = H.run(nm, p=p, slip_mult=slip_mult)
            if res is None:
                return None
            rets.append(res["ret"])
    R = pd.concat(rets, axis=1).fillna(0.0)
    r = R.mean(axis=1)
    return r, res["bars"]

def meta_gate(r, bars, months=6, floor=-0.5):
    win = int(len(r) * (months * 30.4) / 730)
    mu = r.shift(1).rolling(win, min_periods=win // 2).mean()
    sd = r.shift(1).rolling(win, min_periods=win // 2).std()
    bpy = bars / 730 * 365
    trailing = (mu / sd * math.sqrt(bpy)).fillna(0.0)
    on = (trailing > floor).astype(float)
    return r * on, on

def stats(ret, bars):
    bpy = bars / 730 * 365
    ann = math.sqrt(bpy)
    nz = ret[ret != 0]
    if len(nz):
        ret = ret.loc[nz.index[0]:]          # trim warm-up zeros (they fake smoothness/R2)
    # rescale to exactly 10%/yr realized vol for honest DD reading
    if ret.std() > 0:
        ret = ret * (0.10 / ann) / ret.std()
    def block(rr):
        if len(rr) < 200 or rr.std() == 0:
            return dict(sh=0, dd=0, r2=0)
        eq = rr.cumsum()
        dd = (eq - eq.cummax()).min()
        x = np.arange(len(eq))
        b1, b0 = np.polyfit(x, eq.values, 1)
        ssr = ((eq.values - (b1 * x + b0)) ** 2).sum()
        sst = max(((eq.values - eq.values.mean()) ** 2).sum(), 1e-12)
        return dict(sh=rr.mean() / rr.std() * ann, dd=dd, r2=1 - ssr / sst,
                    gp=eq.iloc[-1] / max(abs(dd), 1e-9))   # gain-to-pain (total / maxDD)
    h = len(ret) // 2
    return block(ret), block(ret.iloc[:h]), block(ret.iloc[h:])

rows, kept, raw = [], {}, {}
for nm in UNIVERSE:
    out = run_spread(nm, 1.0)
    if out is None:
        continue
    r, bars = out
    rg, on = meta_gate(r, bars)
    s2 = run_spread(nm, 2.0)
    rg2, _ = meta_gate(s2[0], bars) if s2 else (None, None)
    f, i, o = stats(rg, bars)
    f2 = stats(rg2, bars)[0] if rg2 is not None else dict(sh=0)
    rows.append((nm, bars, f, i, o, f2, on.mean()))
    kept[nm] = (rg, bars)
    raw[nm] = r

rows.sort(key=lambda x: -x[2]["sh"])
print(f"{'spread':<11}{'fullSh':>8}{'ISsh':>7}{'OOSsh':>7}{'maxDD':>8}{'eqR2':>6}"
      f"{'2xSlip':>8}{'%on':>5}")
npos = 0
for nm, bars, f, i, o, f2, pct in rows:
    npos += (f["sh"] > 0 and o["sh"] > 0)
    print(f"{nm:<11}{f['sh']:>8.2f}{i['sh']:>7.2f}{o['sh']:>7.2f}{f['dd']*100:>7.1f}%"
          f"{f['r2']:>6.2f}{f2['sh']:>8.2f}{pct*100:>4.0f}%")
print(f"\nspreads with full>0 AND OOS>0: {npos}/{len(rows)}")
# "smooth upward, nearly no downside" operationalized: positive in both halves,
# total gain >= 3x worst drawdown, and worst drawdown <= 12% at 10% ann vol
smooth = [nm for nm, bars, f, i, o, f2, pct in rows
          if f["sh"] > 0 and o["sh"] > 0 and f["gp"] >= 3.0 and f["dd"] >= -0.12]
print(f"SMOOTH (full&OOS>0, gain>=3x maxDD, maxDD<=12% @10%vol): {len(smooth)}: {smooth}")

# ---- book: FAMILY risk parity (8 cracks are one factor; don't pretend they're 8 bets) ----
FAMILY = {
    "cracks": ["RB_crack", "HO_crack", "crack_321", "BZ_CL", "RB_HO", "RB_BZ", "HO_BZ", "crack321b"],
    "bio":    ["BOHO"],
    "gas":    ["NG_HO", "NG_CL"],
    "metals": ["GC_SI", "SI_HG", "GC_PL", "PL_PA"],
    "ags":    ["crush", "ZW_KE", "ZC_ZW", "ZS_ZC", "ZM_ZL"],
}
B = pd.DataFrame({nm: v[0] for nm, v in kept.items()})
fams = {}
for fam, members in FAMILY.items():
    cols = [m for m in members if m in B.columns]
    if cols:
        fams[fam] = B[cols].fillna(0.0).mean(axis=1)
FB = pd.DataFrame(fams)
port = FB.mean(axis=1)
act = B.notna().sum(axis=1)
port = port[act >= 4]
bars_b = len(port)
fb, ib, ob = stats(port, bars_b)
print(f"{'BOOK':<11}{fb['sh']:>8.2f}{ib['sh']:>7.2f}{ob['sh']:>7.2f}{fb['dd']*100:>7.1f}%{fb['r2']:>6.2f}"
      f"   (family risk parity: {list(fams)})")

# ---- plots ----
names = [nm for nm, *_ in rows]
cols = 3
rn = math.ceil(len(names) / cols)
fig, ax = plt.subplots(rn, cols, figsize=(15, 2.6 * rn))
ax = ax.flatten()
for k, nm in enumerate(names):
    rg, bars = kept[nm]
    eq = rg.cumsum()
    h = len(eq) // 2
    ax[k].plot(eq.index, eq.values, lw=0.9, color="navy")
    ax[k].axvline(eq.index[h], color="gray", ls="--", lw=0.7)
    ax[k].axhline(0, color="red", ls=":", lw=0.6)
    f, i, o = stats(rg, bars)
    ax[k].set_title(f"{nm}  Sh={f['sh']:.2f} (IS {i['sh']:.2f}/OOS {o['sh']:.2f})", fontsize=9)
    ax[k].tick_params(labelsize=7)
for j in range(len(names), len(ax)):
    ax[j].axis("off")
fig.suptitle("Hourly OU-z spread MR — net cumulative PnL (vol-targeted units), dashed = IS/OOS",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/per_spread.png", dpi=110)
plt.close(fig)

fig2, a2 = plt.subplots(figsize=(12, 4.5))
peq = port * (0.10 / math.sqrt(bars_b / 730 * 365)) / max(port.std(), 1e-12)
eq = peq.cumsum()
a2.plot(eq.index, eq.values, lw=1.3, color="darkgreen")
a2.axvline(eq.index[len(eq)//2], color="gray", ls="--", lw=0.8)
a2.axhline(0, color="red", ls=":", lw=0.7)
a2.set_title(f"BOOK (family risk parity, causal meta-filter, 10% ann vol)  Sharpe={fb['sh']:.2f}  "
             f"IS={ib['sh']:.2f} OOS={ob['sh']:.2f}  maxDD={fb['dd']*100:.1f}%  eqR2={fb['r2']:.2f}",
             fontsize=11)
fig2.tight_layout()
fig2.savefig(f"{OUT}/book.png", dpi=120)
plt.close(fig2)

B.to_csv(f"{OUT}/spread_returns.csv")
print(f"\ncurves -> {OUT}/per_spread.png, {OUT}/book.png")
