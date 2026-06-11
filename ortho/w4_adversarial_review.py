#!/usr/bin/env /usr/bin/python3
"""
W4-B — ADVERSARIAL REVIEW OF THE W3-B RETRACTION (hostile referee).

W3-B retracted Program 1's 12y BRENT-WTI Sharpe 1.93, attributing it to 33-38%
stale placeholder bars (h==l & v==0) with ladder: dirty 1.98 -> drop-placeholders
0.60 -> tick costs 0.41 -> next-open 0.37. This study tries to OVERTURN it.

ATTACK BATTERY (pre-registered):
 S0 independent replication of the headline ladder endpoints
    (published ~1.93 via duka_cracks.py itself; dirty same-close lam ~1.98 via
    w3 machinery; clean next-open ticks ~0.37).
 S1 stale-bar forensics: fraction, time distribution (year / dow x hour),
    weekend vs scheduled-break vs residual freezes; ALTERNATIVE CLEANINGS:
      L1 drop ONLY the true weekend closure (Fri 22:00 -> Sun 22:00 UTC)
      L2 drop weekend + empirically-scheduled closed cells (per-leg (dow,hr)
         cells with >=90% stale frequency)
      L3 drop all h==l&v==0 placeholders (w3's cleaning, expect ~0.60)
    If L1/L2 recover the headline, w3's cleaning was over-aggressive — UNLESS
    R3b (below) shows the recovered pnl also lives on frozen/thaw bars.
 S1b (R3b closure test) for each of L1/L2: re-mask pnl on the frame's RESIDUAL
    stale + thaw bars. If the recovered Sharpe collapses, the "recovery" is the
    surviving async-artifact channel, not alpha.
 S2 DECISIVE inert-bar experiment: positions formed on the DIRTY z exactly as
    published; pnl masked on bar classes:
      V0 full dirty                      (baseline ~1.98)
      V1 pnl zeroed on ANY-leg-stale bars (the pre-registered decisive test)
      V2 pnl zeroed on BOTH-legs-stale only (pure dilution test - analytic
         prediction: Sharpe unchanged, zero-pnl bars are annualization-neutral)
      V3 pnl zeroed on ANY-stale + THAW bars (thaw = first both-fresh bar after
         a stale run; catches snap-back profits realized at re-open)
    plus a per-bar-class gross pnl attribution table.
 S3 z-mechanics: rolling-sd shrinkage over frozen windows, |z| at thaw bars,
    entry clustering by bar class (dirty) vs session-open clustering (clean).
 S4 OU half-life claim replication with FRESH seeds (101-103): frozen pipeline
    on OU hl=10td (claim: ~0.4 < 0.8), contrast hl=3,5.
 S5 TV cross-feed (0.93, 3.2y) construction audit: stamp alignment, censor
    coverage, 1-bar entry delay, deliberate 1h misalignment (should DEGRADE if
    construction is sound), session-open pnl concentration, halves/yearly.

PRE-REGISTERED VERDICT RULES (mechanical):
 R1 replication: published in [1.78,2.08]; dirty lam in [1.7,2.3]; clean honest
    in [0.20,0.55]. Fail -> ladder numbers wrong -> at least 'mixed'.
 R2 decisive: V3 net < 0.6 -> alpha lived on frozen-quote/thaw bars (CONFIRM);
    V1 net >= 1.5 AND V3 net >= 1.5 -> signal real off stale bars (OVERTURN).
 R3 cleanings: L1 net < 1.0 AND L2 net < 1.0 -> no recovery (CONFIRM);
    either >= 1.5 -> over-dropping candidate, adjudicated by R3b:
 R3b a cleaning's recovery is GENUINE only if its stale+thaw-masked net Sharpe
    stays >= 1.0; if it collapses < 0.6 the recovery is the artifact channel.
 R4 OU: mean(hl=10, fresh seeds) < 0.8 confirms the incoherence claim.
 R5 TV sound iff: delay-1 Sharpe > 0.3 AND misaligned Sharpe <= honest + 0.2
    AND no uncensored roll-sized jumps (>5 mid-day 8xMAD day-jumps unflagged).
 FINAL: retraction-confirmed iff R1 passes AND R2 confirms AND no cleaning
        shows GENUINE (R3b-validated) recovery >= 1.5.
        retraction-overturned iff R2 overturns OR a genuine recovery >= 1.5.
        mixed otherwise (specify which numbers move).

 AMENDMENT NOTE (transparency): the first run's FINAL clause was
 'overturned iff R2 overturns OR R3 overturns' without the R3b adjudication;
 it fired 'retraction-overturned' on L1=1.90/L2=1.59 even though S2 showed
 95.6% of the pnl on ONE_STALE+THAW bars (9.7% of bars) and V3=0.12. That
 clause misencoded the task's hierarchy — the task pre-registers S2 as THE
 decisive experiment — and its premise (milder-cleaning recovery = alpha) is
 testable, so R3b was added and the clause rewritten. Both versions' outputs
 are reported in the study log; nothing else changed.

Run: cd ortho && /usr/bin/python3 w4_adversarial_review.py
"""
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab
import w3_brnwti_honest as w3

pd.set_option('display.width', 250)

WKND = lambda idx: (((idx.dayofweek == 4) & (idx.hour >= 22)) |
                    (idx.dayofweek == 5) |
                    ((idx.dayofweek == 6) & (idx.hour < 22)))


def load_dc():
    spec = importlib.util.spec_from_file_location('dc', str(lab.ROOT / 'duka_cracks.py'))
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
    return dc


def build_dirty_legs(drop_fn=None):
    """Program-1-style join, returning legs too. drop_fn(df)->bool mask of rows to DROP."""
    legs = []
    for sym in ('BRENT', 'WTI'):
        d = w3.duka_raw(sym)
        if drop_fn is not None:
            d = d[~drop_fn(d)]
        legs.append(d)
    A, B = w3.join(*legs)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    return A, B, F, w3.bars_day(F.index)


def ens_lam_masked(F, decisions, bd, mask=None, gross=False):
    """Program-1 pnl model (same-close, lam cost) with dS zeroed where mask=True.
    Costs stay on the ORIGINAL dirty basis (du * 0.05*sig63) for comparability."""
    dS0 = F['c'].diff()
    dS = dS0 if mask is None else dS0.where(~mask, 0.0)
    sig63 = dS0.ewm(span=int(63 * bd), min_periods=200).std()
    parts = []
    for d in decisions:
        p = (d['u'].shift(1) * dS).fillna(0.0)
        if not gross:
            p = p - (0.05 * sig63.shift(1) * d['u'].diff().abs()).fillna(0.0)
        parts.append(p)
    return pd.concat(parts, axis=1).mean(axis=1)


def classify_bars(A, B):
    """Bar classes on the dirty join: BOTH_STALE / ONE_STALE / THAW / FRESH."""
    sa = ((A['h'] == A['l']) & (A['v'] == 0)).values
    sb = ((B['h'] == B['l']) & (B['v'] == 0)).values
    both = sa & sb
    one = sa ^ sb
    anyst = sa | sb
    prev_any = np.r_[False, anyst[:-1]]
    thaw = (~anyst) & prev_any
    cls = np.where(both, 'BOTH_STALE',
          np.where(one, 'ONE_STALE',
          np.where(thaw, 'THAW', 'FRESH')))
    return cls, sa, sb, anyst, thaw


def stale_runs(anyst):
    runs, n = [], 0
    for v in anyst:
        if v:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return np.array(runs)


def dilution_demo(pnl):
    """Interleave a zero bar after every real bar (same calendar span):
    analytic prediction Sharpe unchanged."""
    p = w3.trim(pnl)
    v = np.zeros(2 * len(p))
    v[0::2] = p.values
    idx = pd.date_range(p.index[0], p.index[-1], periods=2 * len(p))
    return lab.metrics(pd.Series(v, index=idx))['sharpe']


def z_components(dSsig, bd, zd=18, detrend_d=28):
    z_n = max(80, int(zd * bd))
    dt = max(150, int(detrend_d * bd))
    mu = dSsig.shift(1).rolling(dt, min_periods=dt // 2).mean().fillna(0.0)
    S = (dSsig - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = (S - m) / sd
    return z, sd, z_n


def tv_spread_shifted(shift_hours=0):
    """w3.tv_spread with optional deliberate misalignment of the WTI leg."""
    A = lab.tv('ICEEUR_DLY_BRN1!, 60 (1).csv')
    B = lab.tv('CFI_WTI, 60.csv')
    out = []
    for k, X in enumerate((A, B)):
        X = X.copy()
        X.index = X.index.floor('h')
        if k == 1 and shift_hours:
            X.index = X.index + pd.Timedelta(hours=shift_hours)
        X = X[~X.index.duplicated()]
        X = X[X.index.dayofweek != 6]
        out.append(X)
    A, B = w3.join(*out)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    day = F.index.tz_convert('America/New_York').date
    newday = np.r_[False, day[1:] != day[:-1]]
    cens = np.zeros(len(F), bool)
    for L in (A, B):
        dc_ = np.r_[0.0, np.diff(L['c'].values)]
        mad = pd.Series(dc_).rolling(504, min_periods=100).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
        cens |= (np.abs(dc_) > 8 * 1.4826 * np.where(mad > 0, mad, np.nan)) & newday
    cens = pd.Series(cens, index=F.index)
    dSsig = F['c'].diff().where(~cens, 0.0)
    return A, B, F, cens, dSsig, w3.bars_day(F.index), pd.Series(newday, index=F.index)


def main():
    print(__doc__)
    met = w3.met
    results = {}

    # ================= S0: HEADLINE REPLICATION =================
    print('================ S0: INDEPENDENT HEADLINE REPLICATION ================')
    dc = load_dc()
    r_dc, bpy_dc = dc.ensemble('BRENT_WTI')
    m_pub = dc.metr(r_dc, bpy_dc)
    print('published (duka_cracks.py itself)        : Sh=%.2f H1=%.2f H2=%.2f eqR2=%.2f G/P=%.1f'
          % (m_pub['sh'], m_pub['a'], m_pub['b'], m_pub['r2'], m_pub['gp']))

    A0, B0, F0, bd0 = build_dirty_legs(None)
    dec0 = w3.decisions_for(F0['c'].diff(), bd0)
    m_dirty = met(ens_lam_masked(F0, dec0, bd0))
    print('dirty join, same-close, lam (w3 claim 1.98): %s' % m_dirty)

    Ac, Bc, Fc, bdc, dSc = w3.build_primary()
    decc = w3.decisions_for(dSc, bdc)
    ens_h, _, _ = w3.ensemble_fixed(Fc, decc, w3.COST_TURN, 'open', 1.0)
    m_honest = met(ens_h)
    print('clean join, next-open, ticks (w3 claim 0.37): %s' % m_honest)
    ens_ct, _, _ = w3.ensemble_fixed(Fc, decc, w3.COST_TURN, 'close', 1.0)
    m_ct = met(ens_ct)
    print('clean join, same-close, ticks (w3 claim 0.41): %s' % m_ct)
    r1 = (1.78 <= m_pub['sh'] <= 2.08) and (1.7 <= m_dirty['sharpe'] <= 2.3) \
         and (0.20 <= m_honest['sharpe'] <= 0.55)
    print('R1 ladder replicates: %s' % ('PASS' if r1 else 'FAIL'))
    results.update(pub=m_pub['sh'], dirty=m_dirty['sharpe'],
                   honest=m_honest['sharpe'], clean_ct=m_ct['sharpe'], r1=r1)

    # ================= S1: STALE-BAR FORENSICS =================
    print('\n================ S1: STALE-BAR FORENSICS + ALTERNATIVE CLEANINGS '
          '================')
    cell_drop = {}
    for sym in ('BRENT', 'WTI'):
        d = w3.duka_raw(sym)
        st = pd.Series(((d.h == d.l) & (d.v == 0)).values, index=d.index)
        byyr = st.groupby(st.index.year).mean() * 100
        wk = WKND(st.index)
        cells = st.groupby([st.index.dayofweek, st.index.hour]).mean()
        sched = cells[cells >= 0.90]
        sched_set = set(sched.index)
        in_sched = pd.MultiIndex.from_arrays([st.index.dayofweek, st.index.hour]).isin(sched_set)
        cell_drop[sym] = sched_set
        resid = st[~np.asarray(in_sched)]
        print('%s: stale %.1f%% | by-year min %.1f max %.1f (flat) | weekend window '
              '100%% stale | scheduled(>=90%%) cells=%d cover %.1f%% of bars | '
              'residual stale OUTSIDE schedule: %.2f%% of non-scheduled bars'
              % (sym, 100 * st.mean(), byyr.min(), byyr.max(), len(sched_set),
                 100 * float(np.asarray(in_sched).mean()), 100 * resid.mean()))
        nw = st[~wk]
        top = (nw.groupby(nw.index.hour).mean() * 100).sort_values(ascending=False)
        print('  non-weekend stale by hour (top5): %s'
              % {h: round(v, 1) for h, v in top.head(5).items()})

    cls0, sa0, sb0, any0, thaw0 = classify_bars(A0, B0)
    runs = stale_runs(any0)
    print('joined dirty frame: BOTH_STALE %.1f%% ONE_STALE %.1f%% THAW %.1f%% FRESH %.1f%%'
          % tuple(100 * (cls0 == k).mean() for k in
                  ('BOTH_STALE', 'ONE_STALE', 'THAW', 'FRESH')))
    print('any-leg-stale run lengths: n=%d median=%d p90=%d max=%d (49 = full weekend)'
          % (len(runs), int(np.median(runs)), int(np.percentile(runs, 90)), runs.max()))

    print('--- alternative cleanings (same-close, lam cost — published basis) ---')

    def sched_drop(sym):
        def fn(d):
            key = pd.MultiIndex.from_arrays([d.index.dayofweek, d.index.hour])
            return np.asarray(key.isin(cell_drop[sym])) | np.asarray(WKND(d.index))
        return fn

    def build_named(nm):
        if nm.startswith('L1'):
            return build_dirty_legs(lambda d: np.asarray(WKND(d.index)))
        if nm.startswith('L2'):
            legs = []
            for sym in ('BRENT', 'WTI'):
                d = w3.duka_raw(sym)
                legs.append(d[~sched_drop(sym)(d)])
            Ax, Bx = w3.join(*legs)
            Fx = pd.DataFrame({'o': Ax['o'] - Bx['o'], 'c': Ax['c'] - Bx['c']},
                              index=Ax.index)
            return Ax, Bx, Fx, w3.bars_day(Fx.index)
        return build_dirty_legs(lambda d: (d.h == d.l) & (d.v == 0))

    ladders, frames = {}, {}
    for nm in ('L1 weekend-only drop', 'L2 weekend+scheduled-cells drop',
               'L3 all-placeholders drop (w3)'):
        Ax, Bx, Fx, bdx = build_named(nm)
        decx = w3.decisions_for(Fx['c'].diff(), bdx)
        mx = met(ens_lam_masked(Fx, decx, bdx))
        clsx, _, _, anyx, thawx = classify_bars(Ax, Bx)
        rs = float(anyx.mean())
        ladders[nm] = mx
        frames[nm[:2]] = (Fx, decx, bdx, anyx, thawx)
        print('%-34s bars=%6d bd=%4.1f residual-stale=%4.1f%% : %s'
              % (nm, len(Fx), bdx, 100 * rs, mx))
    l1, l2, l3 = (ladders[k]['sharpe'] for k in ladders)
    r3_confirm = (l1 < 1.0) and (l2 < 1.0)
    r3_candidate = (l1 >= 1.5) or (l2 >= 1.5)
    print('R3 cleanings: L1=%.2f L2=%.2f L3=%.2f -> %s'
          % (l1, l2, l3, 'CONFIRM (no recovery)' if r3_confirm else
             ('RECOVERY CANDIDATE -> adjudicate with R3b' if r3_candidate
              else 'INTERMEDIATE')))

    # ---- R3b closure test: is the L1/L2 "recovery" itself frozen-quote pnl? ----
    print('--- R3b: mask each recovered frame on its OWN residual stale+thaw bars ---')
    r3b = {}
    for key in ('L1', 'L2'):
        Fx, decx, bdx, anyx, thawx = frames[key]
        mk = pd.Series(anyx | thawx, index=Fx.index)
        mb = met(ens_lam_masked(Fx, decx, bdx, mk))
        r3b[key] = mb['sharpe']
        print('%s positions kept, pnl zeroed on residual stale+thaw '
              '(%4.1f%% of bars): net Sh=%5.2f  (unmasked %.2f)'
              % (key, 100 * float((anyx | thawx).mean()), mb['sharpe'],
                 ladders[[n for n in ladders if n.startswith(key)][0]]['sharpe']))
    genuine_recovery = r3_candidate and (max(r3b.values()) >= 1.0) and \
        ((l1 >= 1.5 and r3b['L1'] >= 1.0) or (l2 >= 1.5 and r3b['L2'] >= 1.0))
    print('R3b: recovered alpha survives off frozen/thaw bars? %s '
          '(L1b=%.2f L2b=%.2f; genuine iff >=1.0)'
          % (genuine_recovery, r3b['L1'], r3b['L2']))
    results.update(L1=l1, L2=l2, L3=l3, L1b=r3b['L1'], L2b=r3b['L2'],
                   r3_confirm=r3_confirm, genuine_recovery=genuine_recovery)

    # ================= S2: DECISIVE INERT-BAR EXPERIMENT =================
    print('\n================ S2: DECISIVE EXPERIMENT — dirty positions, pnl '
          'masked by bar class ================')
    idx0 = F0.index
    both_m = pd.Series(cls0 == 'BOTH_STALE', index=idx0)
    any_m = pd.Series(any0, index=idx0)
    thaw_m = pd.Series(thaw0, index=idx0)
    variants = [
        ('V0 full dirty (baseline)', None),
        ('V1 zero pnl on ANY-stale bars', any_m),
        ('V2 zero pnl on BOTH-stale only', both_m),
        ('V3 zero on ANY-stale + THAW bars', (any_m | thaw_m)),
    ]
    vshs = {}
    for nm, mask in variants:
        g = met(ens_lam_masked(F0, dec0, bd0, mask, gross=True))
        n = met(ens_lam_masked(F0, dec0, bd0, mask, gross=False))
        vshs[nm[:2]] = (g['sharpe'], n['sharpe'])
        print('%-36s gross Sh=%5.2f  net(lam) Sh=%5.2f  total_vu(net)=%s'
              % (nm, g['sharpe'], n['sharpe'], n['total_vu']))
    print('dilution sanity (analytic: zero-bar interleave leaves Sharpe ~unchanged): '
          'V0 net %.2f -> interleaved %.2f'
          % (vshs['V0'][1], dilution_demo(ens_lam_masked(F0, dec0, bd0))))

    # per-class gross pnl attribution of the FULL dirty ensemble
    eg = ens_lam_masked(F0, dec0, bd0, None, gross=True)
    print('--- gross pnl attribution by bar class (full dirty ensemble) ---')
    tot = eg.sum()
    attr = {}
    for k in ('BOTH_STALE', 'ONE_STALE', 'THAW', 'FRESH'):
        m = cls0 == k
        s = float(eg[m].sum())
        attr[k] = s
        print('  %-10s bars %5.1f%%  pnl %8.1f  (%5.1f%% of total)'
              % (k, 100 * m.mean(), s, 100 * s / tot))
    fab_share = (attr['ONE_STALE'] + attr['THAW']) / tot
    r2_confirm = vshs['V3'][1] < 0.6
    r2_overturn = (vshs['V1'][1] >= 1.5) and (vshs['V3'][1] >= 1.5)
    print('ONE_STALE+THAW pnl share = %.1f%% on %.1f%% of bars'
          % (100 * fab_share, 100 * float(np.isin(cls0, ['ONE_STALE', 'THAW']).mean())))
    print('R2 decisive: V1 net=%.2f V3 net=%.2f -> %s'
          % (vshs['V1'][1], vshs['V3'][1],
             'CONFIRM (alpha lived on frozen/thaw bars)' if r2_confirm else
             ('OVERTURN' if r2_overturn else 'INTERMEDIATE')))
    results.update(V0=vshs['V0'][1], V1=vshs['V1'][1], V2=vshs['V2'][1],
                   V3=vshs['V3'][1], fab_share=fab_share,
                   r2_confirm=r2_confirm, r2_overturn=r2_overturn)

    # ================= S3: Z-MECHANICS / ENTRY CLUSTERING =================
    print('\n================ S3: Z-SCORE MECHANICS + ENTRY CLUSTERING ================')
    z0, sd0, zn0 = z_components(F0['c'].diff(), bd0)
    stale_frac_win = pd.Series(any0, index=idx0).rolling(zn0).mean()
    ok = stale_frac_win.notna() & sd0.notna()
    qhi = stale_frac_win[ok].quantile(0.9)
    qlo = stale_frac_win[ok].quantile(0.1)
    hi = ok & (stale_frac_win >= qhi)
    lo = ok & (stale_frac_win <= qlo)
    shrink = float(sd0[hi].median() / sd0[lo].median())
    print('rolling sd (z denominator, 18td sub, %d-bar window): median sd in '
          'top-decile stale-fraction windows (>=%.2f) = %.4f vs bottom-decile '
          '(<=%.2f) = %.4f -> shrinkage ratio %.2f'
          % (zn0, qhi, sd0[hi].median(), qlo, sd0[lo].median(), shrink))
    za = z0.abs()
    print('|z| mean: THAW bars %.2f | ONE_STALE %.2f | FRESH %.2f | BOTH_STALE %.2f'
          % (za[thaw0].mean(), za[cls0 == 'ONE_STALE'].mean(),
             za[cls0 == 'FRESH'].mean(), za[cls0 == 'BOTH_STALE'].mean()))
    print('P(|z|>=2.5): THAW %.3f | ONE_STALE %.3f | FRESH %.3f'
          % ((za[thaw0] >= 2.5).mean(), (za[cls0 == 'ONE_STALE'] >= 2.5).mean(),
             (za[cls0 == 'FRESH'] >= 2.5).mean()))

    # entry clustering, all 8 dirty subs
    entries = np.concatenate([[t[0] for t in d['trades']] for d in dec0]).astype(int)
    near_thaw = thaw0 | np.r_[False, thaw0[:-1]] | np.r_[False, False, thaw0[:-2]]
    rows = []
    for k in ('BOTH_STALE', 'ONE_STALE', 'THAW', 'FRESH'):
        m = cls0 == k
        e = float(np.mean(cls0[entries] == k))
        rows.append((k, 100 * m.mean(), 100 * e, e / max(m.mean(), 1e-9)))
    print('--- entry-bar class distribution (all 8 dirty subs, %d entries) ---'
          % len(entries))
    print('%-11s %8s %9s %7s' % ('class', 'bars%', 'entries%', 'ratio'))
    for k, b, e, r in rows:
        print('%-11s %7.1f%% %8.1f%% %6.2fx' % (k, b, e, r))
    pe_near = float(np.mean(near_thaw[entries]))
    print('entries within 2 bars of a thaw: %.1f%% (base rate of such bars %.1f%%)'
          % (100 * pe_near, 100 * near_thaw.mean()))

    # clean-frame comparison: session-open clustering
    gap = np.r_[False, (Fc.index[1:] - Fc.index[:-1]) > pd.Timedelta(hours=1)]
    near_open = gap | np.r_[False, gap[:-1]] | np.r_[False, False, gap[:-2]]
    entc = np.concatenate([[t[0] for t in d['trades']] for d in decc]).astype(int)
    print('CLEAN frame: entries within 2 bars of a session open: %.1f%% '
          '(base %.1f%%) — vs dirty near-thaw %.1f%% (base %.1f%%)'
          % (100 * float(np.mean(near_open[entc])), 100 * near_open.mean(),
             100 * pe_near, 100 * near_thaw.mean()))
    results.update(entry_thaw_ratio=[r for k, b, e, r in rows if k == 'THAW'][0],
                   entry_onestale_ratio=[r for k, b, e, r in rows
                                         if k == 'ONE_STALE'][0],
                   sd_shrink=shrink)

    # ================= S4: OU HALF-LIFE CLAIM, FRESH SEEDS =================
    print('\n================ S4: OU HALF-LIFE REPLICATION (fresh seeds) ================')
    dstd = float(dSc.std())
    ou = {}
    for hl, seeds in [(3, (101, 102)), (5, (101, 102)), (10, (101, 102, 103))]:
        vals = []
        for seed in seeds:
            Fs = w3.synth_frame('ou', Fc.index, bdc, dstd, seed, hl_td=hl)
            dec = w3.decisions_for(Fs['c'].diff(), bdc)
            e, _, _ = w3.ensemble_fixed(Fs, dec, w3.COST_TURN, 'open', 1.0)
            vals.append(met(e)['sharpe'])
        ou[hl] = float(np.mean(vals))
        print('frozen pipeline on OU hl=%2dtd seeds %s: %s mean=%.2f'
              % (hl, seeds, [round(v, 2) for v in vals], ou[hl]))
    r4 = ou[10] < 0.8
    print('R4 (claim: cannot harvest hl=10td, <0.8): hl10 mean=%.2f -> %s'
          % (ou[10], 'CONFIRM' if r4 else 'OVERTURN'))
    results.update(ou3=ou[3], ou5=ou[5], ou10=ou[10], r4=r4)

    # ================= S5: TV CROSS-FEED AUDIT =================
    print('\n================ S5: TV CROSS-FEED (0.93) CONSTRUCTION AUDIT ================')
    rawA = lab.tv('ICEEUR_DLY_BRN1!, 60 (1).csv')
    rawB = lab.tv('CFI_WTI, 60.csv')
    print('raw stamp minutes: BRN %s | CFI_WTI %s'
          % (dict(pd.Series(rawA.index.minute).value_counts().head(3)),
             dict(pd.Series(rawB.index.minute).value_counts().head(3))))
    for nm, X in (('BRN1!', rawA), ('CFI_WTI', rawB)):
        hl_eq = float((X['h'] == X['l']).mean()) if 'h' in X.columns else float('nan')
        print('  %s bars=%d %s->%s  h==l frac %.3f%%'
              % (nm, len(X), X.index[0].date(), X.index[-1].date(), 100 * hl_eq))
    At, Bt, Ft, cens_t, dS_t, bd_t, newday_t = tv_spread_shifted(0)
    cov = len(Ft) / min(len(rawA), len(rawB))
    print('join: bars=%d (%.0f%% of smaller leg) bd=%.1f censored=%d'
          % (len(Ft), 100 * cov, bd_t, int(cens_t.sum())))
    # uncensored roll-sized jumps not at day boundary
    miss = 0
    for L in (At, Bt):
        dc_ = np.r_[0.0, np.diff(L['c'].values)]
        mad = pd.Series(dc_).rolling(504, min_periods=100).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
        big = (np.abs(dc_) > 8 * 1.4826 * np.where(mad > 0, mad, np.nan))
        miss += int((big & ~newday_t.values).sum())
    print('roll-sized (8xMAD) jumps NOT at a day boundary (uncensored): %d' % miss)

    dec_t = w3.decisions_for(dS_t, bd_t)
    tv1, _, _ = w3.ensemble_fixed(Ft, dec_t, w3.COST_TURN, 'open', 1.0, cens_t)
    m_tv = met(tv1)
    print('TV honest x1 (w3 claim 0.93): %s' % m_tv)
    print('TV yearly Sharpe: %s' % dict(lab.yearly(w3.trim(tv1))))
    # 1-bar entry delay
    parts_d = [w3.vec_pnl(Ft, d['u'].shift(1).fillna(0.0), w3.COST_TURN, 'open',
                          1.0, cens_t) for d in dec_t]
    m_tvd = met(pd.concat(parts_d, axis=1).mean(axis=1))
    print('TV honest, ENTRY DELAYED 1 bar: %s' % m_tvd)
    # zero pnl on first bar of each NY day (session-open concentration test)
    parts_o = [w3.vec_pnl(Ft, d['u'], w3.COST_TURN, 'open', 1.0,
                          (cens_t | newday_t)) for d in dec_t]
    m_tvo = met(pd.concat(parts_o, axis=1).mean(axis=1))
    print('TV honest, pnl CENSORED on first bar of each NY day: %s' % m_tvo)
    # deliberate 1h misalignment of the WTI leg
    Am, Bm, Fm, cens_m, dS_m, bd_m, _ = tv_spread_shifted(1)
    dec_m = w3.decisions_for(dS_m, bd_m)
    tvm, _, _ = w3.ensemble_fixed(Fm, dec_m, w3.COST_TURN, 'open', 1.0, cens_m)
    m_tvm = met(tvm)
    print('TV with WTI leg DELIBERATELY shifted +1h: %s' % m_tvm)
    r5 = (m_tvd['sharpe'] > 0.3) and (m_tvm['sharpe'] <= m_tv['sharpe'] + 0.2) \
         and (miss <= 5)
    print('R5 TV construction sound: delay1=%.2f (>0.3) misaligned=%.2f '
          '(<=honest+0.2) uncensored-jumps=%d (<=5) -> %s'
          % (m_tvd['sharpe'], m_tvm['sharpe'], miss, 'PASS' if r5 else 'FAIL'))
    results.update(tv=m_tv['sharpe'], tv_delay=m_tvd['sharpe'],
                   tv_open_cens=m_tvo['sharpe'], tv_mis=m_tvm['sharpe'], r5=r5)

    # ================= FIGURE =================
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax = axs[0, 0]
    names = ['published\n(P1 code)', 'dirty\n(sc,lam)', 'L1 wknd\nonly drop',
             'L2 sched\ncells drop', 'L3 placeholders\ndrop (w3)',
             'clean sc\nticks', 'clean HONEST\nnext-open']
    vals = [m_pub['sh'], m_dirty['sharpe'], l1, l2, l3, m_ct['sharpe'],
            m_honest['sharpe']]
    ax.bar(range(len(vals)), vals,
           color=['gray', 'firebrick', 'darkorange', 'darkorange', 'goldenrod',
                  'steelblue', 'navy'])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, '%.2f' % v, ha='center', fontsize=8)
    ax.bar([2, 3], [r3b['L1'], r3b['L2']], width=0.35, color='black', alpha=0.75,
           label='same frame, pnl zeroed on\nresidual stale+thaw bars (R3b)')
    for i, v in zip([2, 3], [r3b['L1'], r3b['L2']]):
        ax.text(i, v + 0.03, '%.2f' % v, ha='center', fontsize=7, color='black')
    ax.legend(fontsize=7)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('Sharpe'); ax.grid(alpha=0.3, axis='y')
    ax.set_title('attribution ladder re-run + alternative cleanings', fontsize=10)

    ax = axs[0, 1]
    vn = ['V0 full', 'V1 zero\nANY-stale', 'V2 zero\nBOTH-stale', 'V3 zero\nstale+THAW']
    gv = [vshs[k][0] for k in ('V0', 'V1', 'V2', 'V3')]
    nv = [vshs[k][1] for k in ('V0', 'V1', 'V2', 'V3')]
    x = np.arange(4)
    ax.bar(x - 0.18, gv, 0.36, label='gross', color='seagreen')
    ax.bar(x + 0.18, nv, 0.36, label='net (lam)', color='navy')
    for i in range(4):
        ax.text(i - 0.18, gv[i] + 0.03, '%.2f' % gv[i], ha='center', fontsize=8)
        ax.text(i + 0.18, nv[i] + 0.03, '%.2f' % nv[i], ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(vn, fontsize=8); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y'); ax.set_ylabel('Sharpe')
    ax.set_title('DECISIVE: dirty positions, pnl masked by bar class', fontsize=10)

    ax = axs[1, 0]
    for k, col in [('BOTH_STALE', 'gray'), ('ONE_STALE', 'firebrick'),
                   ('THAW', 'darkorange'), ('FRESH', 'navy')]:
        ax.plot(idx0, eg.where(pd.Series(cls0 == k, index=idx0), 0.0).cumsum(),
                lw=1.0, color=col,
                label='%s (%.0f%% of pnl, %.0f%% of bars)'
                      % (k, 100 * attr[k] / tot, 100 * (cls0 == k).mean()))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('dirty-ensemble GROSS pnl accrual by bar class', fontsize=10)
    ax.set_ylabel('cum pnl')

    ax = axs[1, 1]
    tn = ['honest\n(w3 0.93)', 'entry\ndelay 1', 'open-bar\ncensored',
          'WTI leg\n+1h misalign']
    tvv = [m_tv['sharpe'], m_tvd['sharpe'], m_tvo['sharpe'], m_tvm['sharpe']]
    ax.bar(range(4), tvv, color=['navy', 'steelblue', 'darkorange', 'firebrick'])
    for i, v in enumerate(tvv):
        ax.text(i, v + 0.02, '%.2f' % v, ha='center', fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(tn, fontsize=8)
    ax.grid(alpha=0.3, axis='y'); ax.set_ylabel('Sharpe')
    ax.set_title('TV cross-feed (3.2y) construction stress', fontsize=10)
    fig.suptitle('W4-B adversarial review of the W3-B retraction (BRENT-WTI 12.3y '
                 'Dukascopy)', fontsize=12)
    fig.tight_layout()
    out = lab.ROOT / 'curves' / 'w4_adversarial_review.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out)

    # ================= VERDICT =================
    print('\n================ PRE-REGISTERED VERDICT ================')
    print('R1 ladder replicates: %s' % ('PASS' if r1 else 'FAIL'))
    print('R2 decisive: %s' % ('CONFIRM' if r2_confirm else
                               ('OVERTURN' if r2_overturn else 'INTERMEDIATE')))
    print('R3/R3b cleanings: L1=%.2f L2=%.2f, masked L1b=%.2f L2b=%.2f -> %s'
          % (l1, l2, r3b['L1'], r3b['L2'],
             'GENUINE RECOVERY (overturn)' if results['genuine_recovery'] else
             ('recovery is the artifact channel (confirm)' if not r3_confirm
              else 'no recovery (confirm)')))
    print('R4 OU hl=10 claim: %s' % ('CONFIRM' if r4 else 'OVERTURN'))
    print('R5 TV construction: %s' % ('SOUND' if r5 else 'SUSPECT'))
    if r2_overturn or results['genuine_recovery']:
        verdict = 'retraction-overturned'
    elif r1 and r2_confirm and not results['genuine_recovery']:
        verdict = 'retraction-confirmed'
    else:
        verdict = 'mixed'
    print('\nVERDICT (mechanical): %s' % verdict)
    if not r5:
        print('ADDENDUM: W3-B\'s residual claim (TV cross-feed 0.93 "modest live '
              'edge on real ICE quotes") FAILS the construction stress battery '
              '(delay-1 %.2f, +1h misalignment %.2f, %d uncensored roll-sized '
              'jumps) — it belongs in the suspect column, strengthening the '
              'retraction, not weakening it.'
              % (results['tv_delay'], results['tv_mis'], miss))

    print('\n===== SUMMARY (machine-readable) =====')
    print('PUB=%.2f DIRTY=%.2f HONEST=%.2f CLEAN_SC_TICK=%.2f' %
          (results['pub'], results['dirty'], results['honest'], results['clean_ct']))
    print('L1=%.2f L2=%.2f L3=%.2f L1_MASKED=%.2f L2_MASKED=%.2f' %
          (l1, l2, l3, r3b['L1'], r3b['L2']))
    print('V0=%.2f V1=%.2f V2=%.2f V3=%.2f FABSHARE=%.2f' %
          (results['V0'], results['V1'], results['V2'], results['V3'],
           results['fab_share']))
    print('ENTRY_THAW_RATIO=%.2f ENTRY_ONESTALE_RATIO=%.2f SD_SHRINK=%.2f' %
          (results['entry_thaw_ratio'], results['entry_onestale_ratio'],
           results['sd_shrink']))
    print('OU3=%.2f OU5=%.2f OU10=%.2f' % (ou[3], ou[5], ou[10]))
    print('TV=%.2f TV_DELAY=%.2f TV_OPENCENS=%.2f TV_MISALIGN=%.2f' %
          (results['tv'], results['tv_delay'], results['tv_open_cens'],
           results['tv_mis']))
    print('VERDICT=%s' % verdict)


if __name__ == '__main__':
    main()
