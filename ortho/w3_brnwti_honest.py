#!/usr/bin/env /usr/bin/python3
"""
W3-B — HONEST RE-VALIDATION OF THE BRENT-WTI HOURLY SPREAD MR (Program 1's edge).

Context (docs/repo_study.md): the 12y headline (Sharpe 1.93) and the 3.1y TV result
(2.90) were produced by `duka_cracks.one` / `hourly4y.one`, which fill at the SAME
BAR CLOSE the signal observed and charge a lambda*sigma cost (0.05*sig*|du|), not
the documented 1.5 ticks/leg. This study re-runs the FROZEN Program-1 rule over the
full 12.3y Dukascopy history with honest NEXT-OPEN fills (hourly_lab.run semantics)
and the documented tick costs, and forensically attributes the gap.

FROZEN RULE (verbatim, NO re-tuning):
  spread dX = dBRENT - dWTI in $/bbl (fixed 1:-1 weights, leg $-changes)
  detrend  S = cumsum(dX - rollmean(dX, 28 td))        [causal: rollmean on dX.shift(1)]
  signal   z = (S - mean(S,n)) / std(S,n)              [causal: rolling on S.shift(1)]
  ENSEMBLE n in {9,12,18,24} td x entry {2.5,3.0} -> 8 sub-strategies, equal weight
  exit |z| <= 0.75; stops |z| >= 4.0 and 10-td time stop
  VOL GATE: enter only when 5-day spread vol (ewm) > rolling 6-month median
  sizing: per-trade risk-frozen, units = sign / sigma_at_entry (ewm 63d std of dX)
  fills:  NEXT BAR OPEN (signal at close t -> fill at open[t+1], BOTH legs)
  costs:  1.5 ticks/leg/turn, tick $0.01/bbl -> $0.03 per spread-unit turn; stress 3.0
  td->bars: from the data's actual bars/day, exactly as duka_cracks.one
  (z_n = max(80, zd*bd), detrend = max(150, 28*bd), tstop = 10*bd, gate spans 5d/126d).

RESOLVED AMBIGUITIES (all conservative, documented):
  A1. Sunday bars are excluded from both legs before joining (wave-2 binding hygiene
      rule for duka commodities; Program 1 did not exclude them).
  A2. lab.duka drops Dukascopy PLACEHOLDER bars (high==low & volume==0: market-closed
      hours filled with the stale last quote). Program 1 kept them: 38%/33% of raw
      BRENT/WTI bars are placeholders; in Program 1's joined frame 32% of bars are
      both-legs-stale and 7% are one-leg-stale (async fake spread quotes). Trading
      a z-score against stale quotes with same-close fills manufactures alpha; the
      forensic battery quantifies exactly how much.
  A3. Secondary BTU spreads (GAS-WTI, GAS-BRENT) and the DIESEL-BRENT negative
      control mix quote units -> each leg's $-change is normalized by its own
      trailing 100-bar causal ATR before differencing (the pre-registered
      resolution). PnL uses leg hedge weights 1/ATR FROZEN at the entry fill bar
      (no unmodeled intra-trade rebalancing), consistent with per-trade risk-frozen
      sizing.
  A4. Costs for ATR-normalized legs: per leg per turn = max(1.5 x tick,
      lab.cost_bps/2 x price) — the LARGER of the frozen tick spec and the lab's
      conservative bps spread cost (ticks alone are unrealistically small for the
      GAS CFD). Primary BRENT-WTI uses the frozen tick spec exactly ($0.03/turn);
      a bps-based variant is reported as extra stress, not decision-relevant.
  A5. DIESEL bars before 2017-12-01 are dropped (quote convention there is ~50x
      larger — feed change, not price). DIESEL tick taken as 0.25 (ICE gasoil).
  A6. ENGINE CONTROLS are split in two after diagnosis:
      (a) APPARATUS control — a textbook OU harvester (z of the level vs a 5-half-
          life window, enter |z|>=2, exit 0, same 1/sigma sizing, same next-open
          vec pnl, same tick costs) must clear OU(hl=10td) Sharpe >= 0.8 net and
          RW |Sharpe| < 0.3 (mean of 5 seeds). This validates fills/costs/pnl
          accounting — the machinery that produces the verdict number.
      (b) FROZEN-PIPELINE characterization — the frozen rule itself is run on OU
          spreads of hl {3,5,10,20}td. Finding: it harvests only fast reversion
          (hl<=5td) and is WEAK at its nominal 10td target (~0.4). This is a
          property of the strategy, not the apparatus: the identical synthetic
          frames pushed through Program 1's own committed code (duka_cracks.one)
          give the same weak numbers, and my engine reproduces Program 1's 12y
          headline to 0.04 Sharpe on Program 1's own dirty join. Both
          faithfulness cross-checks are printed.
  A7. Metrics are computed from the first nonzero-pnl bar onward (Program 1's
      metr() convention; leading warmup zeros carry no information).
  A8. Shuffle test uses POSITION EPISODES (contiguous same-sign holdings), not raw
      entry events: the vec engine treats same-sign re-entry (time-stop refresh)
      as a continuing position, so charging full round-trip costs per entry event
      would overstate costs identically in real and null — episodes keep real and
      null on the same cost basis (2 turns per episode).
  A9. TV cross-feed legs are floored to the hour (CFI_WTI stamps at :01) and get
      hourly_lab-style roll censoring (day-boundary |dc| > 8 x 1.4826 x rolling-504
      MAD => bar contributes 0 to signal AND pnl, re-roll turn charged while held).

BATTERY (all mandatory): 1 engine controls -> 2 honest 12.3y backtest -> 3 fills
comparison + forensic attribution of the published 1.93 -> 4 cost stress x2 ->
5 regime splits -> 6 secondary GAS-WTI / GAS-BRENT -> 7 negative control
DIESEL-BRENT through the identical pipeline -> 8 shuffle test (100). Plus: signed
event study, parameter neighborhood (reporting only), TV cross-feed.

PRE-REGISTERED DECISION RULE: promote iff next-open net Sharpe >= 1.0 (full 12.3y)
AND both halves >= 0.5 AND eqR2 >= 0.85 AND gain/pain >= 4 AND x2-cost Sharpe >= 0.6
AND the negative control fails the same screen. kill if below; unresolved only for
an unfixable engine-control failure.

Run: cd ortho && /usr/bin/python3 w3_brnwti_honest.py
"""
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab

ZDS       = (9, 12, 18, 24)
ZES       = (2.5, 3.0)
SLIP      = 1.5
TICK      = {'BRENT': 0.01, 'WTI': 0.01, 'GAS': 0.001, 'DIESEL': 0.25}
COST_TURN = SLIP * (0.01 + 0.01)        # $/spread-unit/turn, frozen primary spec
N_SHUFFLE = 100
N_PLACEBO = 300
EV_HORIZONS = (24, 48, 96, 208)         # bars; 208 ~ the 10-td time stop


# ------------------------------------------------------------------ data

def clean(sym, start=None):
    d = lab.duka(sym)
    d = d[d.index.dayofweek != 6]                       # hygiene: no Sunday bars
    if start is not None:
        d = d[d.index >= start]
    return d


def duka_raw(sym):
    """Corrected timestamps but PLACEHOLDER BARS KEPT (Program 1's data state)."""
    return lab._epoch_df(lab.DUKA / f'{sym}.csv', lab.DUKA_TS_OFFSET)


def join(A, B):
    idx = A.index.intersection(B.index)
    return A.loc[idx], B.loc[idx]


def bars_day(idx):
    return len(idx) / max(1, len(np.unique(idx.date)))


# ------------------------------------------------------------------ decisions
# Replicates duka_cracks.one decision semantics EXACTLY (signal, gate, sizing,
# state machine, td->bar conversion incl. its floors). Only the fill model differs
# downstream.

def decide(dSsig, bd, zd, ze, volgate=True, zx=0.75, zstop=4.0,
           tstop_d=10, detrend_d=28):
    dS = dSsig
    z_n = max(80, int(zd * bd))
    dt = max(150, int(detrend_d * bd))
    ts = int(tstop_d * bd)
    mu = dS.shift(1).rolling(dt, min_periods=dt // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    n = len(dS)
    if volgate:
        sf = dS.ewm(span=int(5 * bd), min_periods=50).std()
        ss = sf.rolling(int(126 * bd), min_periods=800).median()
        hot = (sf > ss).values
    else:
        hot = np.ones(n, bool)
    sig = dS.ewm(span=int(63 * bd), min_periods=200).std()
    ssafe = sig.replace(0, np.nan).ffill().bfill().values
    u = np.zeros(n)
    trades = []                                          # (entry_dec, exit_dec, side, q)
    cur = 0; held = 0; ep = -1; q = 0.0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= zstop or held >= ts
                    or (cur > 0 and zz >= -zx) or (cur < 0 and zz <= zx)):
                trades.append((ep, i, cur, q))
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:
            if zz <= -ze:
                cur, held, ep, q = 1, 0, i, 1.0 / ssafe[i]
            elif zz >= ze:
                cur, held, ep, q = -1, 0, i, 1.0 / ssafe[i]
        if cur != 0:
            u[i] = cur * q
    if cur != 0:
        trades.append((ep, n - 1, cur, q))
    return dict(u=pd.Series(u, index=dS.index), z=z, ssafe=ssafe, trades=trades)


# ------------------------------------------------------------------ pnl engines

def vec_pnl(F, u, cost_turn, fills='open', stress=1.0, cens=None):
    """hourly_lab.run semantics. fills='open': position decided at close i is held
    from open i+1 (overnight gap accrues to the OLD position); 'close': Program-1
    same-close fill (u.shift(1)*dS)."""
    C, O = F['c'], F['o']
    cz = pd.Series(False, index=F.index) if cens is None else cens
    if fills == 'open':
        ua = u.shift(1)
        du = ua.diff().abs().fillna(0.0)
        day = (ua * (C - O)).where(~cz, 0.0)
        on = (ua.shift(1) * (O - C.shift(1))).where(~cz, 0.0)
        roll = (ua.abs() * cz).fillna(0.0)
        pnl = day.fillna(0.0) + on.fillna(0.0) - stress * cost_turn * (du + roll)
    else:
        dS = C.diff().where(~cz, 0.0)
        du = u.diff().abs().fillna(0.0)
        roll = (u.shift(1).abs() * cz).fillna(0.0)
        pnl = (u.shift(1) * dS).fillna(0.0) - stress * cost_turn * (du + roll)
    return pnl.fillna(0.0)


def frozen_pnl(A, B, aA, aB, trades, costA, costB, fills='open', stress=1.0):
    """ATR-normalized 2-leg spread, hedge weights (1/ATR_A, -1/ATR_B) FROZEN at the
    entry fill bar. Next-open or same-close fills. Costs per leg per turn in $."""
    n = len(A)
    oA, cA = A['o'].values, A['c'].values
    oB, cB = B['o'].values, B['c'].values
    avA, avB = aA.values, aB.values
    ctA, ctB = costA.values, costB.values
    pnl = np.zeros(n)
    for (p, x, s, q) in trades:
        f, e = p + 1, min(x + 1, n - 1)
        if f >= n - 1 or e <= f:
            continue
        wA, wB = 1.0 / avA[f], -1.0 / avB[f]
        if not (np.isfinite(wA) and np.isfinite(wB) and np.isfinite(q)):
            continue
        if fills == 'open':
            pnl[f] += s * q * (wA * (cA[f] - oA[f]) + wB * (cB[f] - oB[f]))
            for j in range(f + 1, e):
                pnl[j] += s * q * (wA * (cA[j] - cA[j - 1]) + wB * (cB[j] - cB[j - 1]))
            pnl[e] += s * q * (wA * (oA[e] - cA[e - 1]) + wB * (oB[e] - cB[e - 1]))
            fb, eb = f, e
        else:
            for j in range(p + 1, x + 1):
                pnl[j] += s * q * (wA * (cA[j] - cA[j - 1]) + wB * (cB[j] - cB[j - 1]))
            fb, eb = p + 1, x
        pnl[fb] -= stress * q * (abs(wA) * ctA[fb] + abs(wB) * ctB[fb])
        pnl[eb] -= stress * q * (abs(wA) * ctA[eb] + abs(wB) * ctB[eb])
    return pd.Series(pnl, index=A.index)


# ------------------------------------------------------------------ runners

def decisions_for(dSsig, bd, **kw):
    return [decide(dSsig, bd, zd, ze, **kw) for zd in ZDS for ze in ZES]


def ensemble_fixed(F, decisions, cost_turn, fills='open', stress=1.0, cens=None):
    parts = [vec_pnl(F, d['u'], cost_turn, fills, stress, cens) for d in decisions]
    ens = pd.concat(parts, axis=1).mean(axis=1)
    ntr = sum(len(d['trades']) for d in decisions)
    subs = [met(p)['sharpe'] for p in parts]
    return ens, ntr, subs


def ens_lam(F, decisions, bd, cens=None):
    """Program-1's exact pnl model: same-close fills, cost 0.05*sigma*|du|."""
    dS = F['c'].diff()
    if cens is not None:
        dS = dS.where(~cens, 0.0)
    sig63 = dS.ewm(span=int(63 * bd), min_periods=200).std()
    parts = [(d['u'].shift(1) * dS).fillna(0.0)
             - (0.05 * sig63.shift(1) * d['u'].diff().abs()).fillna(0.0)
             for d in decisions]
    return pd.concat(parts, axis=1).mean(axis=1)


def trim(pnl):
    nz = pnl[pnl != 0]
    return pnl.loc[nz.index[0]:] if len(nz) else pnl


def met(pnl):
    return lab.metrics(trim(pnl))


def thirds(pnl):
    p = trim(pnl)
    bpy = lab.bars_per_year(p.to_frame())
    out = []
    n = len(p)
    for i in range(3):
        seg = p.iloc[i * n // 3:(i + 1) * n // 3]
        out.append(round(float(seg.mean() / seg.std() * np.sqrt(bpy)), 2)
                   if seg.std() > 0 else 0.0)
    return out


def regime_table(pnl, bpy):
    regs = [('2014-2016 oil crash', '2014-01-01', '2016-12-31'),
            ('2017-2019', '2017-01-01', '2019-12-31'),
            ('2020 covid', '2020-01-01', '2020-12-31'),
            ('2021-2022 war', '2021-01-01', '2022-12-31'),
            ('2023-2026', '2023-01-01', '2026-12-31')]
    rows = []
    p = trim(pnl)
    for name, a, b in regs:
        seg = p.loc[a:b]
        if len(seg) < 100 or seg.std() == 0:
            rows.append((name, len(seg), np.nan, np.nan))
            continue
        rows.append((name, len(seg),
                     round(float(seg.mean() / seg.std() * np.sqrt(bpy)), 2),
                     round(float(seg.sum()), 1)))
    return rows


# ------------------------------------------------------------------ builds

def build_primary():
    A, B = join(clean('BRENT'), clean('WTI'))
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    bd = bars_day(F.index)
    return A, B, F, bd, F['c'].diff()


def build_dirty(drop_sunday=False, drop_placeholder=False):
    """Program-1-style join with optional single-artifact removal (attribution)."""
    legs = []
    for sym in ('BRENT', 'WTI'):
        d = duka_raw(sym)
        if drop_placeholder:
            d = d[~((d.h == d.l) & (d.v == 0))]
        if drop_sunday:
            d = d[d.index.dayofweek != 6]
        legs.append(d)
    A, B = join(*legs)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    return F, bars_day(F.index)


def build_norm(symA, symB, startA=None, startB=None):
    A, B = join(clean(symA, startA), clean(symB, startB))
    aA, aB = lab.atr(A, 100), lab.atr(B, 100)
    ok = aA.notna() & aB.notna() & (aA > 0) & (aB > 0)
    first = ok.idxmax() if ok.any() else None
    A, B, aA, aB = A.loc[first:], B.loc[first:], aA.loc[first:], aB.loc[first:]
    dSsig = (A['c'].diff() / aA - B['c'].diff() / aB).fillna(0.0)
    bd = bars_day(A.index)
    costA = np.maximum(SLIP * TICK[symA], (lab.cost_bps(symA) / 2) * 1e-4 * A['c'])
    costB = np.maximum(SLIP * TICK[symB], (lab.cost_bps(symB) / 2) * 1e-4 * B['c'])
    return A, B, aA, aB, dSsig, bd, costA, costB


def run_norm_spread(symA, symB, startA=None, startB=None):
    A, B, aA, aB, dSsig, bd, costA, costB = build_norm(symA, symB, startA, startB)
    decisions = decisions_for(dSsig, bd)
    out = {}
    for tag, stress in [('x1', 1.0), ('x2', 2.0)]:
        parts = [frozen_pnl(A, B, aA, aB, d['trades'], costA, costB, 'open', stress)
                 for d in decisions]
        out[tag] = pd.concat(parts, axis=1).mean(axis=1)
    ntr = sum(len(d['trades']) for d in decisions)
    yrs = (A.index[-1] - A.index[0]).days / 365.25
    return out['x1'], out['x2'], ntr, yrs, bd


# ------------------------------------------------------------------ controls

def synth_frame(kind, idx, bd, dstd, seed, hl_td=10):
    """Two half-steps per bar: open[i]=x[2i], close[i]=x[2i+1], so a next-open fill
    pays one real decay/diffusion step after the signal close."""
    n = len(idx)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, dstd / np.sqrt(2.0), 2 * n)
    if kind == 'ou':
        phi = 0.5 ** (1.0 / (2.0 * hl_td * bd))
        x = np.empty(2 * n)
        x[0] = eps[0]
        for i in range(1, 2 * n):
            x[i] = phi * x[i - 1] + eps[i]
    else:
        x = np.cumsum(eps)
    return pd.DataFrame({'o': x[0::2], 'c': x[1::2]}, index=idx)


def apparatus_harvester(Fs, bd, hl_td=10):
    """Textbook OU harvester THROUGH THE SAME machinery (1/sigma sizing, next-open
    vec_pnl, tick costs): z of the level vs a 5-half-life window, enter |z|>=2,
    exit at 0. Validates fills/cost/pnl accounting independent of the frozen rule."""
    z_n = int(5 * hl_td * bd)
    dS = Fs['c'].diff()
    S = Fs['c']
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    sig = dS.ewm(span=int(63 * bd), min_periods=200).std()
    ssafe = sig.replace(0, np.nan).ffill().bfill().values
    n = len(Fs)
    u = np.zeros(n); cur = 0; q = 0.0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0 and (zz != zz or (cur > 0 and zz >= 0) or (cur < 0 and zz <= 0)):
            cur = 0
        if cur == 0 and zz == zz:
            if zz <= -2.0:
                cur, q = 1, 1.0 / ssafe[i]
            elif zz >= 2.0:
                cur, q = -1, 1.0 / ssafe[i]
        u[i] = cur * q
    return vec_pnl(Fs, pd.Series(u, index=Fs.index), COST_TURN, 'open', 1.0)


# ------------------------------------------------------------------ event study / shuffle

def entry_events(decisions):
    ev = {}
    for d in decisions:
        for (p, x, s, q) in d['trades']:
            ev.setdefault(p, s)
    pos = np.array(sorted(ev), dtype=int)
    sides = np.array([ev[p] for p in pos], dtype=float)
    return pos, sides


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def event_study(F, decisions, d18, seed=7):
    C, O = F['c'].values, F['o'].values
    z18, ssafe = d18['z'], d18['ssafe']
    N = len(F)
    pos, sides = entry_events(decisions)
    smap = dict(zip(pos.tolist(), sides.tolist()))
    valid = np.flatnonzero(np.isfinite(z18) & (z18 != 0) & np.isfinite(ssafe)
                           & (ssafe > 0))
    rng = np.random.default_rng(seed)
    rows = []
    for h in EV_HORIZONS:
        P = _dedup(pos[(pos + h) < N - 1], h)
        s = np.array([smap[p] for p in P])
        f = s * (C[P + h] - O[P + 1]) / ssafe[P]
        f = f[np.isfinite(f)]
        if len(f) < 10:
            rows.append((h, len(f), np.nan, np.nan, np.nan))
            continue
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        V = valid[(valid + h) < N - 1]
        pm = np.empty(N_PLACEBO)
        for i in range(N_PLACEBO):
            Q = V[rng.integers(0, len(V), size=len(P))]
            sq = -np.sign(z18[Q])
            pm[i] = np.nanmean(sq * (C[Q + h] - O[Q + 1]) / ssafe[Q])
        rows.append((h, len(f), round(m, 4), round(t, 2),
                     round(float((np.abs(pm) >= abs(m)).mean()), 3)))
    return rows


def episodes(u):
    """Contiguous same-sign position episodes from the fill-aligned series ua."""
    ua = u.shift(1).fillna(0.0).values
    n = len(ua)
    eps = []
    i = 0
    while i < n:
        if ua[i] == 0:
            i += 1
            continue
        s = np.sign(ua[i]); q = abs(ua[i]); f = i
        j = i
        while j + 1 < n and ua[j + 1] != 0 and np.sign(ua[j + 1]) == s:
            j += 1
        eps.append((f, min(j + 1, n - 1), s, q))      # exit fill at open of bar j+1
        i = j + 1
    return eps


def shuffle_test(F, decisions, seed=11):
    """Relocate every real position EPISODE (same duration, random eligible start,
    random side, 1/sigma size at the new entry); identical 2-turn cost basis for
    real and null (A8). Returns (real_total, null_totals)."""
    O, C = F['o'].values, F['c'].values
    N = len(F)
    rng = np.random.default_rng(seed)
    elist, warm = [], N
    for d in decisions:
        eps = episodes(d['u'])
        if eps:
            warm = min(warm, eps[0][0])
        for (f, e, s, q) in eps:
            if e <= f:
                continue
            elist.append((f, e, s, q, d['ssafe']))
    real = sum(s * q * (O[e] - O[f]) - 2 * COST_TURN * q
               for (f, e, s, q, _) in elist)
    tots = np.zeros(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        tot = 0.0
        for (f, e, s, q, ssafe) in elist:
            dur = e - f
            f2 = int(rng.integers(warm + 1, N - dur - 1))
            q2 = 1.0 / ssafe[f2 - 1]
            s2 = 1.0 if rng.random() < 0.5 else -1.0
            tot += s2 * q2 * (O[f2 + dur] - O[f2]) - 2 * COST_TURN * q2
        tots[i] = tot
    return real, tots, len(elist)


# ------------------------------------------------------------------ TV cross-feed

def tv_spread():
    A = lab.tv('ICEEUR_DLY_BRN1!, 60 (1).csv')
    B = lab.tv('CFI_WTI, 60.csv')
    out = []
    for X in (A, B):
        X = X.copy()
        X.index = X.index.floor('h')                  # CFI_WTI stamps at :01 (A9)
        X = X[~X.index.duplicated()]
        X = X[X.index.dayofweek != 6]
        out.append(X)
    A, B = join(*out)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    day = F.index.tz_convert('America/New_York').date
    newday = np.r_[False, day[1:] != day[:-1]]
    cens = np.zeros(len(F), bool)
    for L in (A, B):
        dc = np.r_[0.0, np.diff(L['c'].values)]
        mad = pd.Series(dc).rolling(504, min_periods=100).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
        cens |= (np.abs(dc) > 8 * 1.4826 * np.where(mad > 0, mad, np.nan)) & newday
    cens = pd.Series(cens, index=F.index)
    dSsig = F['c'].diff().where(~cens, 0.0)
    return F, cens, dSsig, bars_day(F.index)


# ================================================================== main

def main():
    print(__doc__)
    A, B, F, bd, dSsig = build_primary()
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    bpy = lab.bars_per_year(F)
    print('=== DATA: BRENT-WTI duka join, placeholders dropped, Sundays excluded ===')
    print('bars=%d  %s -> %s  (%.2f y, %.2f bars/day)  spread dS std=%.4f $/bbl'
          % (len(F), F.index[0].date(), F.index[-1].date(), yrs, bd, dSsig.std()))
    print('frozen cost: $%.3f per spread-unit turn (1.5 ticks x 2 legs x $0.01)'
          % COST_TURN)
    for sym in ('BRENT', 'WTI'):
        d = duka_raw(sym)
        ph = float(((d.h == d.l) & (d.v == 0)).mean())
        print('raw %s placeholder-bar fraction (h==l & v==0): %.1f%%' % (sym, 100 * ph))

    # ---------------- BATTERY 1: ENGINE CONTROLS ----------------
    print('\n================ BATTERY 1: ENGINE CONTROLS ================')
    dstd = float(dSsig.std())
    print('--- 1a APPARATUS control: textbook OU harvester through the same '
          'fills/cost/pnl machinery ---')
    app = {}
    for kind in ('ou', 'rw'):
        vals = []
        for seed in (1, 2, 3, 4, 5):
            Fs = synth_frame(kind, F.index, bd, dstd, seed)
            vals.append(met(apparatus_harvester(Fs, bd))['sharpe'])
        app[kind] = vals
        print('  %s net Sharpe (5 seeds): %s  mean=%.2f'
              % (kind.upper(), [round(v, 2) for v in vals], np.mean(vals)))
    app_ou = float(np.mean(app['ou']))
    app_rw = float(np.mean(app['rw']))
    ctrl_ok = (app_ou >= 0.8) and (abs(app_rw) < 0.3)
    print('  APPARATUS: OU mean %.2f (need >=0.8) %s | RW mean %.2f (need |.|<0.3) '
          '%s -> %s' % (app_ou, 'PASS' if app_ou >= 0.8 else 'FAIL', app_rw,
                        'PASS' if abs(app_rw) < 0.3 else 'FAIL',
                        'PASS' if ctrl_ok else 'FAIL'))

    print('--- 1b FROZEN-PIPELINE characterization on OU (honest fills, tick '
          'costs; 3 seeds) ---')
    pipe_hl = {}
    for hl in (3, 5, 10, 20):
        vals = []
        for seed in (1, 2, 3):
            Fs = synth_frame('ou', F.index, bd, dstd, seed, hl_td=hl)
            dec = decisions_for(Fs['c'].diff(), bd)
            e, _, _ = ensemble_fixed(Fs, dec, COST_TURN, 'open', 1.0)
            vals.append(met(e)['sharpe'])
        pipe_hl[hl] = float(np.mean(vals))
        print('  frozen rule on OU hl=%2dtd: %s  mean=%.2f'
              % (hl, [round(v, 2) for v in vals], pipe_hl[hl]))
    rw_vals = []
    for seed in (1, 2, 3):
        Fs = synth_frame('rw', F.index, bd, dstd, seed)
        dec = decisions_for(Fs['c'].diff(), bd)
        g, _, _ = ensemble_fixed(Fs, dec, COST_TURN, 'open', 0.0)
        rw_vals.append(met(g)['sharpe'])
    pipe_rw = float(np.mean(rw_vals))
    print('  frozen rule on RW (gross): %s  mean=%.2f (must be ~0)'
          % ([round(v, 2) for v in rw_vals], pipe_rw))

    print('--- 1c FAITHFULNESS cross-check: same synthetic through Program 1\'s '
          'committed code (duka_cracks.one) ---')
    spec = importlib.util.spec_from_file_location(
        'dc', str(lab.ROOT / 'duka_cracks.py'))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    for seed in (1, 2, 3):
        Fs = synth_frame('ou', F.index, bd, dstd, seed, hl_td=10)
        dec = decisions_for(Fs['c'].diff(), bd)
        mine = met(ens_lam(Fs, dec, bd))['sharpe']
        p1 = [dc.one(Fs, zd, ze)[0] for zd in dc.ZDS for ze in dc.ZES]
        theirs = met(pd.concat(p1, axis=1).fillna(0.0).mean(axis=1))['sharpe']
        print('  OU hl=10 seed=%d: my engine (same-close,lam) %.2f vs '
              'duka_cracks.one %.2f' % (seed, mine, theirs))
    print('FINDING: the frozen PIPELINE harvests only fast OU (hl<=5td) and is '
          'weak at its nominal 10td target (%.2f);\n  identical through Program '
          '1\'s own code -> a property of the frozen rule, NOT the apparatus. '
          'Apparatus controls: %s.' % (pipe_hl[10], 'PASS' if ctrl_ok else 'FAIL'))

    # ---------------- BATTERY 2: HONEST 12.3y BACKTEST ----------------
    print('\n================ BATTERY 2: HONEST 12.3y BACKTEST (next-open, '
          'tick costs x1) ================')
    decisions = decisions_for(dSsig, bd)
    d18 = decisions[ZDS.index(18) * len(ZES)]            # (18td, 2.5) sub
    ens1, ntr, subs1 = ensemble_fixed(F, decisions, COST_TURN, 'open', 1.0)
    m1 = met(ens1)
    th = thirds(ens1)
    print('ensemble of 8 subs, total entry events=%d (%.0f/yr)' % (ntr, ntr / yrs))
    print(m1)
    print('thirds Sharpe: %s' % th)
    yr_tab = lab.yearly(trim(ens1))
    print('yearly Sharpe:\n%s' % yr_tab)
    ytot = trim(ens1).groupby(trim(ens1).index.year).sum().round(1)
    print('yearly PnL (vu):\n%s' % ytot)
    # consistency check: trade-level totals vs vectorized totals (gross)
    d0 = decisions[0]
    g = vec_pnl(F, d0['u'], COST_TURN, 'open', 0.0).sum()
    tl = sum(s * q * (F['o'].values[min(x + 1, len(F) - 1)] - F['o'].values[p + 1])
             for (p, x, s, q) in d0['trades'] if p + 1 < min(x + 1, len(F) - 1))
    print('engine consistency (sub 9td/2.5): vec gross total=%.1f  '
          'trade-level=%.1f  (diff %.2f%%)' % (g, tl, 100 * (g - tl) / abs(tl)))

    # ---------------- BATTERY 3: FILLS + FORENSIC ATTRIBUTION ----------------
    print('\n================ BATTERY 3: FILLS COMPARISON + FORENSIC ATTRIBUTION '
          'OF THE 1.93 ================')
    r_dc, bpy_dc = dc.ensemble('BRENT_WTI')
    m_dc = dc.metr(r_dc, bpy_dc)
    print('Program-1 committed code, as published    : Sh=%.2f H1=%.2f H2=%.2f '
          'eqR2=%.2f G/P=%.1f' % (m_dc['sh'], m_dc['a'], m_dc['b'], m_dc['r2'],
                                  m_dc['gp']))
    ladder = []
    for tag, drop_sun, drop_ph in [
            ('dirty join (placeholders+Sundays KEPT)', False, False),
            ('dirty minus Sundays (placeholders kept)', True, False),
            ('placeholders dropped, Sundays kept', False, True)]:
        Fd, bd_d = build_dirty(drop_sunday=drop_sun, drop_placeholder=drop_ph)
        dec_d = decisions_for(Fd['c'].diff(), bd_d)
        md = met(ens_lam(Fd, dec_d, bd_d))
        ladder.append((tag + ', same-close, lam-cost', md))
        print('my engine | %-42s: %s' % (tag + ', sc, lam', md))
    m_p1 = met(ens_lam(F, decisions, bd))
    ens_c, _, _ = ensemble_fixed(F, decisions, COST_TURN, 'close', 1.0)
    m_c = met(ens_c)
    ladder += [('clean join, same-close, lam-cost', m_p1),
               ('clean join, same-close, tick costs', m_c),
               ('clean join, NEXT-OPEN, tick costs (HONEST)', m1)]
    print('my engine | clean join, same-close, lam-cost     : %s' % m_p1)
    print('my engine | clean join, same-close, tick costs   : %s' % m_c)
    print('my engine | clean join, NEXT-OPEN, ticks (HONEST): %s' % m1)
    print('FILL GAP on clean data (same-close minus next-open, tick costs): '
          '%.2f Sharpe' % (m_c['sharpe'] - m1['sharpe']))
    print('ARTIFACT GAP (dirty/published %.2f -> clean honest %.2f): %.2f Sharpe, '
          'dominated by stale placeholder bars + Sunday bars'
          % (m_dc['sh'], m1['sharpe'], m_dc['sh'] - m1['sharpe']))
    # staleness structure of the dirty join
    pa = duka_raw('BRENT'); pb = duka_raw('WTI')
    sa = pd.Series(((pa.h == pa.l) & (pa.v == 0)).values, index=pa.index)
    sb = pd.Series(((pb.h == pb.l) & (pb.v == 0)).values, index=pb.index)
    ji = sa.index.intersection(sb.index)
    print('dirty-join staleness: both-legs-stale %.1f%%, one-leg-stale %.1f%% '
          'of bars' % (100 * float((sa.loc[ji] & sb.loc[ji]).mean()),
                       100 * float((sa.loc[ji] ^ sb.loc[ji]).mean())))

    # ---------------- BATTERY 4: COST STRESS ----------------
    print('\n================ BATTERY 4: COST STRESS ================')
    ens2, _, _ = ensemble_fixed(F, decisions, COST_TURN, 'open', 2.0)
    m2 = met(ens2)
    print('x2 (3.0 ticks/leg/turn): %s' % m2)
    bps_cost = (np.maximum(SLIP * 0.01, 2e-4 * A['c'])
                + np.maximum(SLIP * 0.01, 2e-4 * B['c']))
    ens_bps, _, _ = ensemble_fixed(F, decisions, bps_cost, 'open', 1.0)
    m_bps = met(ens_bps)
    print('bps-max cost variant (extra, not decision-relevant): %s' % m_bps)

    # ---------------- BATTERY 5: REGIMES ----------------
    print('\n================ BATTERY 5: REGIME SPLITS (honest x1) ================')
    regs = regime_table(ens1, bpy)
    for name, nbar, sh, tot in regs:
        print('  %-20s bars=%6d  Sharpe=%5s  total_vu=%s' % (name, nbar, sh, tot))

    # ---------------- BATTERY 6: SECONDARY BTU SPREADS ----------------
    print('\n================ BATTERY 6: SECONDARY (ATR-normalized legs, frozen '
          'rule, report-only) ================')
    sec = {}
    for nm, a_, b_ in [('GAS-WTI', 'GAS', 'WTI'), ('GAS-BRENT', 'GAS', 'BRENT')]:
        p1, p2, ntr_s, yrs_s, bd_s = run_norm_spread(a_, b_)
        sec[nm] = (p1, met(p1), met(p2), ntr_s)
        print('%-10s %.1fy  x1: %s' % (nm, yrs_s, sec[nm][1]))
        print('%-10s        x2: %s   entry events=%d' % ('', sec[nm][2], ntr_s))

    # ---------------- BATTERY 7: NEGATIVE CONTROL ----------------
    print('\n================ BATTERY 7: NEGATIVE CONTROL DIESEL-BRENT '
          '================')
    ng1, ng2, ntr_n, yrs_n, _ = run_norm_spread('DIESEL', 'BRENT',
                                                startA='2017-12-01')
    m_n1, m_n2 = met(ng1), met(ng2)
    print('DIESEL-BRENT %.1fy  x1: %s' % (yrs_n, m_n1))
    print('              x2: %s   entry events=%d' % (m_n2, ntr_n))

    def screen(ma, mb):
        return (ma['sharpe'] >= 1.0 and ma['h1'] >= 0.5 and ma['h2'] >= 0.5
                and ma['eqR2'] >= 0.85 and ma['gain_pain'] >= 4
                and mb['sharpe'] >= 0.6)

    neg_passes = screen(m_n1, m_n2)
    print('negative control passes promote screen? %s (must be False)'
          % neg_passes)

    # ---------------- BATTERY 8: SHUFFLE ----------------
    print('\n================ BATTERY 8: SHUFFLE TEST (%d, relocated episodes, '
          'random sides) ================' % N_SHUFFLE)
    real_tot, tots, n_epi = shuffle_test(F, decisions)
    p_shuf = float((tots >= real_tot).mean())
    print('position episodes (all 8 subs) = %d | real episode-level total = %.1f '
          'vu' % (n_epi, real_tot))
    print('shuffle mean=%.1f sd=%.1f | P(shuffle >= real) = %.3f'
          % (tots.mean(), tots.std(), p_shuf))

    # ---------------- SUPPLEMENT A: EVENT STUDY ----------------
    print('\n---- supplement: signed entry event study (placebo = random times, '
          'side=-sign(z18), %d draws) ----' % N_PLACEBO)
    print('%-5s %6s %9s %7s %7s' % ('h', 'n_ev', 'mean_vu', 't', 'p_plc'))
    ev_rows = event_study(F, decisions, d18)
    for h, nev, mm, tt, pp in ev_rows:
        print('%-5d %6d %9s %7s %7s' % (h, nev, mm, tt, pp))

    # ---------------- SUPPLEMENT B: PARAM NEIGHBORHOOD ----------------
    print('\n---- supplement: parameter neighborhood (honest x1, REPORTING ONLY) '
          '----')
    print('per-sub Sharpes (zd x ze):')
    k = 0
    for zd in ZDS:
        row = []
        for ze in ZES:
            row.append('ze=%.1f:%5.2f' % (ze, subs1[k]))
            k += 1
        print('  zd=%-2d  %s' % (zd, '  '.join(row)))
    variants = [('volgate OFF', dict(volgate=False)),
                ('exit 0.50', dict(zx=0.50)),
                ('exit 1.00', dict(zx=1.00)),
                ('detrend 21td', dict(detrend_d=21)),
                ('detrend 35td', dict(detrend_d=35)),
                ('tstop 7td', dict(tstop_d=7)),
                ('tstop 15td', dict(tstop_d=15))]
    var_out = {}
    for nm, kw in variants:
        dv = decisions_for(dSsig, bd, **kw)
        ev_, _, _ = ensemble_fixed(F, dv, COST_TURN, 'open', 1.0)
        var_out[nm] = met(ev_)['sharpe']
        print('  %-14s ensemble Sharpe = %.2f' % (nm, var_out[nm]))

    # ---------------- SUPPLEMENT C: TV CROSS-FEED ----------------
    print('\n---- supplement: TV cross-feed (ICE BRN1! vs CFI_WTI, 60-min, '
          'roll-censored, honest fills, tick costs) ----')
    try:
        Ftv, cens_tv, dS_tv, bd_tv = tv_spread()
        dec_tv = decisions_for(dS_tv, bd_tv)
        tv1, ntr_tv, _ = ensemble_fixed(Ftv, dec_tv, COST_TURN, 'open', 1.0,
                                        cens_tv)
        m_tv = met(tv1)
        yrs_tv = (Ftv.index[-1] - Ftv.index[0]).days / 365.25
        print('TV %.1fy bars=%d censored=%d entry events=%d bd=%.1f' %
              (yrs_tv, len(Ftv), int(cens_tv.sum()), ntr_tv, bd_tv))
        print('TV honest x1   : %s' % m_tv)
        tv_close, _, _ = ensemble_fixed(Ftv, dec_tv, COST_TURN, 'close', 1.0,
                                        cens_tv)
        print('TV same-close  : %s   (Program-1 same-close 3.1y headline was '
              '2.90 on this period)' % met(tv_close))
    except FileNotFoundError as e:
        Ftv, tv1, m_tv = None, None, None
        print('TV files missing (%s) — cross-feed skipped, documented.' % e)

    # ---------------- FIGURES ----------------
    eq = trim(ens1).cumsum()
    eqc = trim(ens_c).cumsum()
    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.1, 1.3, 1.6], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    ax.plot(eq.index, eq.values, lw=1.0, color='navy',
            label='HONEST next-open, 1.5 ticks/leg (Sh %.2f, eqR2 %.2f, G/P %.1f)'
                  % (m1['sharpe'], m1['eqR2'], m1['gain_pain']))
    ax.plot(eqc.index, eqc.values, lw=0.8, color='darkorange', alpha=0.8,
            label='same-close fills, same costs (Sh %.2f)' % m_c['sharpe'])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylabel('cum PnL (vol units)')
    ax2 = fig.add_subplot(gs[1])
    dd = eq - eq.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.6)
    ax2.set_ylabel('drawdown (vu)')
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(ytot.index.astype(str), ytot.values,
            color=['seagreen' if v > 0 else 'firebrick' for v in ytot.values])
    ax3.set_ylabel('yearly PnL (vu)')
    ax3.grid(alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    ax4 = fig.add_subplot(gs[3])
    lad_names = ['P1 committed\n(dirty, sc, lam)'] + \
                ['dirty join\n(sc, lam)', '-Sundays\n(sc, lam)',
                 '-placeholders\n(sc, lam)', 'clean\n(sc, lam)',
                 'clean\n(sc, ticks)', 'clean HONEST\n(next-open, ticks)',
                 'honest x2\ncosts']
    lad_vals = [m_dc['sh']] + [d[1]['sharpe'] for d in ladder] + [m2['sharpe']]
    ax4.bar(range(len(lad_vals)), lad_vals,
            color=['gray'] + ['firebrick'] * 3 + ['darkorange'] * 2 + ['navy', 'navy'])
    ax4.set_xticks(range(len(lad_vals)))
    ax4.set_xticklabels(lad_names, fontsize=7)
    for i, v in enumerate(lad_vals):
        ax4.text(i, v + 0.03, '%.2f' % v, ha='center', fontsize=8)
    ax4.set_ylabel('Sharpe')
    ax4.set_title('forensic attribution of the published 1.93 (sc=same-close fills, '
                  'lam=0.05*sigma cost)', fontsize=9)
    ax4.grid(alpha=0.3, axis='y')
    fig.suptitle('W3-B BRENT-WTI 12.3y hourly (Dukascopy) — FROZEN rule: detrend 28td; '
                 'z ens {9,12,18,24}td x {2.5,3.0}; exit 0.75; stops 4.0/10td;\n'
                 'vol gate 5d>6m med; 1/sigma per-trade sizing; NEXT-OPEN fills; '
                 '1.5 ticks/leg/turn', fontsize=10)
    out1 = lab.ROOT / 'curves' / 'w3_brnwti_honest.png'
    fig.savefig(out1, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out1)

    fig2, axs = plt.subplots(2, 2, figsize=(13, 8))
    panels = [('GAS-WTI (honest x1)', trim(sec['GAS-WTI'][0]).cumsum(),
               sec['GAS-WTI'][1]),
              ('GAS-BRENT (honest x1)', trim(sec['GAS-BRENT'][0]).cumsum(),
               sec['GAS-BRENT'][1]),
              ('DIESEL-BRENT NEGATIVE CONTROL', trim(ng1).cumsum(), m_n1)]
    if tv1 is not None:
        panels.append(('TV cross-feed BRN-WTI (honest x1)', trim(tv1).cumsum(),
                       m_tv))
    for axp, (nm, e, mm) in zip(axs.flatten(), panels):
        axp.plot(e.index, e.values, lw=0.9,
                 color='seagreen' if mm['sharpe'] > 0 else 'firebrick')
        axp.axhline(0, color='gray', lw=0.5, ls=':')
        axp.set_title('%s\nSh %.2f (H1 %.2f / H2 %.2f) eqR2 %.2f G/P %.1f'
                      % (nm, mm['sharpe'], mm['h1'], mm['h2'], mm['eqR2'],
                         mm['gain_pain']), fontsize=9)
        axp.grid(alpha=0.3)
    for j in range(len(panels), 4):
        axs.flatten()[j].axis('off')
    fig2.suptitle('W3-B secondaries + negative control + cross-feed — same frozen '
                  'rule, honest next-open fills', fontsize=11)
    fig2.tight_layout()
    out2 = lab.ROOT / 'curves' / 'w3_brnwti_secondary.png'
    fig2.savefig(out2, dpi=140, bbox_inches='tight')
    print('figure saved -> %s' % out2)

    # ---------------- DECISION ----------------
    print('\n================ PRE-REGISTERED DECISION RULE ================')
    crit = [
        ('apparatus engine controls pass', ctrl_ok,
         'OU=%.2f RW=%.2f (frozen-pipeline-on-10td-OU=%.2f, a strategy property, '
         'see battery 1b)' % (app_ou, app_rw, pipe_hl[10])),
        ('next-open net Sharpe >= 1.0', m1['sharpe'] >= 1.0,
         'sharpe=%.2f' % m1['sharpe']),
        ('both halves >= 0.5', m1['h1'] >= 0.5 and m1['h2'] >= 0.5,
         'h1=%.2f h2=%.2f' % (m1['h1'], m1['h2'])),
        ('eqR2 >= 0.85', m1['eqR2'] >= 0.85, 'eqR2=%.2f' % m1['eqR2']),
        ('gain/pain >= 4', m1['gain_pain'] >= 4, 'gp=%.2f' % m1['gain_pain']),
        ('x2-cost Sharpe >= 0.6', m2['sharpe'] >= 0.6,
         'x2 sharpe=%.2f' % m2['sharpe']),
        ('negative control fails same screen', not neg_passes,
         'neg sharpe=%.2f gp=%.2f' % (m_n1['sharpe'], m_n1['gain_pain'])),
    ]
    all_pass = True
    for name, ok, det in crit:
        all_pass &= bool(ok)
        print('  [%s] %s  (%s)' % ('PASS' if ok else 'FAIL', name, det))
    if not ctrl_ok:
        decision = 'UNRESOLVED'
    else:
        decision = 'PROMOTE' if all_pass else 'KILL'
    print('\nDECISION (mechanical): %s' % decision)

    # ---------------- machine-readable ----------------
    print('\n===== SUMMARY (machine-readable) =====')
    print('SHARPE_X1=%.2f H1=%.2f H2=%.2f EQR2=%.2f GAINPAIN=%.2f MAXDD_VU=%.1f '
          'TOTAL_VU=%.1f NTRADES=%d NEPISODES=%d' %
          (m1['sharpe'], m1['h1'], m1['h2'], m1['eqR2'], m1['gain_pain'],
           m1['maxdd_vu'], m1['total_vu'], ntr, n_epi))
    print('THIRDS=%s' % th)
    print('SHARPE_X2=%.2f BPSCOST_SHARPE=%.2f' % (m2['sharpe'], m_bps['sharpe']))
    print('PUBLISHED=%.2f MYDIRTY=%.2f CLEAN_SC_LAM=%.2f CLEAN_SC_TICK=%.2f '
          'HONEST=%.2f FILLGAP=%.2f ARTIFACTGAP=%.2f' %
          (m_dc['sh'], ladder[0][1]['sharpe'], m_p1['sharpe'], m_c['sharpe'],
           m1['sharpe'], m_c['sharpe'] - m1['sharpe'],
           m_dc['sh'] - m1['sharpe']))
    print('APP_OU=%.2f APP_RW=%.2f PIPE_OU_HL=%s PIPE_RW_GROSS=%.2f CTRL=%s'
          % (app_ou, app_rw, {k: round(v, 2) for k, v in pipe_hl.items()},
             pipe_rw, ctrl_ok))
    print('GASWTI_X1=%.2f GASWTI_X2=%.2f GASBRENT_X1=%.2f GASBRENT_X2=%.2f'
          % (sec['GAS-WTI'][1]['sharpe'], sec['GAS-WTI'][2]['sharpe'],
             sec['GAS-BRENT'][1]['sharpe'], sec['GAS-BRENT'][2]['sharpe']))
    print('NEG_X1=%.2f NEG_PASSES_SCREEN=%s' % (m_n1['sharpe'], neg_passes))
    print('P_SHUFFLE=%.3f' % p_shuf)
    if tv1 is not None:
        print('TV_HONEST=%.2f' % m_tv['sharpe'])
    print('REGIMES=%s' % [(r[0], r[2]) for r in regs])
    print('VARIANTS=%s' % var_out)
    print('EVENTSTUDY=%s' % ev_rows)
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
