#!/usr/bin/env /usr/bin/python3
"""
W5 — PRE-REGISTERED ADJUDICATION OF C2 (Yahoo BZ-CL, both legs NYMEX Globex
futures, same venue / same feed, 2.39y hourly, roll-censored). README "Wave 5".

CONTEXT: W4-B destroyed the C1 (TV cross-venue) construction with delay-1
(0.93 -> 0.15) and a deliberate +1h WTI-leg misalignment that SUPERCHARGED the
strategy (5.14) — proving the frozen pipeline harvests async-feed artifacts on
cross-venue data. The W4-A promote now rests SOLELY on C2 (x1 1.21 / x2 0.99).
This study applies the identical adversarial controls to C2. Machinery is
imported unchanged from w3_brnwti_honest / w4_brnwti_futures; the stress
patterns are w4_adversarial_review's (delay via u.shift(k) into vec_pnl,
misalignment via +1h leg index shift before the join, attribution masking).

PRE-REGISTERED BATTERY (README Wave 5, run in full):
 1. DELAY-1 (and delay-2): entries at open[t+2] (open[t+3]) via u.shift(k)
    into the same next-open vec engine. GATE: delay-1 x1 Sharpe >= 50% of the
    aligned x1 Sharpe. Also: per-trade gross tick edge under delay, halves
    under delay-1, x2 line.
 2. MISALIGNMENT CONTROL: shift the CL leg's series +1h (also -1h) and rerun
    the FULL frozen pipeline (rebuild spread, censor, decisions, pnl).
    GATE: misaligned x1 Sharpe < 1.5x aligned for BOTH shifts (kill-biased:
    if deliberate misalignment helps materially, the construction is
    async-artifact-prone and fails).
 3. LIQUID-HOURS TEST: hours classified by BOTH legs' volume.
      A (primary gate): hour-of-day is liquid iff each leg's per-hour median
        volume > that leg's own overall median volume (both legs).
      B (reported):     US session 13:00-20:00 UTC (bar-open hour 13..19).
    Entries restricted to liquid hours (entry decision bar's hour; exits/stops
    unchanged). GATE: liquid-A-only x1 Sharpe >= 50% of aligned. Converse
    (thin-hours-only entries) reported.
 4. BOUNCE/STALENESS DIAGNOSTICS (gate: no artifact dominance):
    (a) lag-1 autocorr of hourly spread changes (consecutive-hour, uncensored
        pairs), overall and by hour-of-day, vs the same for each leg;
    (b) per-leg within-bar staleness proxy (v==0 or h==l) by hour;
    (c) hour-of-day of the strategy's profitable exits — flag concentration
        in the thinnest hours;
    (d) pnl attribution by entry-hour liquidity tercile (T1 thin / T2 / T3
        liquid, 8 hours each, score = min over legs of per-hour median volume
        normalized by that leg's overall median).
    GATE: no single thin-hour class carries > 50% of total net pnl
    (operationalized: T1-tercile share <= 50%; the classification-A thin
    class share is reported for context).
 ADDITIONAL (report, not gates): x2 cost line for every variant; C1 numbers
 CITED from W4-B for side-by-side context (honest 0.93, delay-1 0.15,
 NY-open-bar pnl-censored 0.45, +1h misalign 5.14); sensitivity of
 conclusions to BZ thin overnight hours being excluded from event detection
 entirely (entry mask blocking hours with BZ per-hour median volume below
 BZ's own median that lie outside 13-19 UTC).

PRE-REGISTERED DECISION RULE (mechanical):
  PROMOTE-CONFIRMED iff gates 1-3 all pass AND gate 4 passes.
  KILL otherwise — and the program's final conclusion is the documented
  negative result.

RESOLVED AMBIGUITIES (conservative, documented):
  RA1. Liquid-hours gate uses classification A (the literal "both legs trade
       actively" definition); B is a 7-hour session and is reported, not
       gated — restricting 24h of entries to 7h mechanically sheds trades
       even for a genuine edge.
  RA2. The misalignment gate is applied to max(+1h, -1h) — kill-biased.
  RA3. Entry-hour restriction needs an entry mask inside the decision state
       machine; w3.decide takes none. decide_masked() below is a VERBATIM
       copy of w3.decide with the single added conjunct `and entry_ok[i]`;
       fidelity is asserted at runtime: an all-True mask must reproduce the
       frozen decisions' u series exactly on all 8 subs.
  RA4. Pnl attribution (4c/4d) is per position EPISODE on the A8 episode
       basis (gross s*q*(O[exit]-O[entry]) minus 2 turns x1 cost), summed
       across all 8 subs; approximate only for the 13 censored bars (R6).
  RA5. Delay-k = d['u'].shift(k) into vec_pnl (vec_pnl itself shifts once
       more, so k=1 fills at open[t+2], k=2 at open[t+3]) — exactly
       w4_adversarial_review's S5 pattern.
  RA6. Per-hour median volumes are computed over the full joined sample
       (a diagnostic classification, not a tradable signal — noted).
  RA7. build_c2_shifted(0) must equal w4.build_c2() exactly (asserted).

Run: cd ortho && /usr/bin/python3 w5_adjudication.py
Figure: ../curves/w5_adjudication.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab
import w3_brnwti_honest as w3
import w4_brnwti_futures as w4

US_HOURS = set(range(13, 20))                      # 13:00-20:00 UTC bar opens
C1_CITED = dict(honest=0.93, delay1=0.15, opencens=0.45, mis_p1=5.14)


# ------------------------------------------------------------------ builders

def build_c2_shifted(shift_hours=0):
    """w4.build_c2 with optional deliberate misalignment of the CL leg
    (the w4_adversarial_review.tv_spread_shifted pattern, applied to C2).
    shift_hours=0 reproduces w4.build_c2 exactly (asserted in main)."""
    legs = []
    for k, sym in enumerate(('BZ', 'CL')):
        d = lab.yahoo(sym)
        d.index = d.index.floor('h')
        if k == 1 and shift_hours:
            d.index = d.index + pd.Timedelta(hours=shift_hours)
        d = d[~d.index.duplicated()]
        d = d[d.index.dayofweek != 6]
        legs.append(d)
    A, B = w3.join(*legs)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    cens = (A['cens'].astype(bool) | B['cens'].astype(bool))
    dSsig = F['c'].diff().where(~cens, 0.0)
    return A, B, F, cens, dSsig, w3.bars_day(F.index)


# ------------------------------------------------------------------ masked decide
# VERBATIM copy of w3.decide with ONE added conjunct (`and entry_ok[i]`) in the
# entry condition (RA3). Fidelity asserted in main with an all-True mask.

def decide_masked(dSsig, bd, zd, ze, entry_ok, volgate=True, zx=0.75, zstop=4.0,
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
    trades = []
    cur = 0; held = 0; ep = -1; q = 0.0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= zstop or held >= ts
                    or (cur > 0 and zz >= -zx) or (cur < 0 and zz <= zx)):
                trades.append((ep, i, cur, q))
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i] and entry_ok[i]:
            if zz <= -ze:
                cur, held, ep, q = 1, 0, i, 1.0 / ssafe[i]
            elif zz >= ze:
                cur, held, ep, q = -1, 0, i, 1.0 / ssafe[i]
        if cur != 0:
            u[i] = cur * q
    if cur != 0:
        trades.append((ep, n - 1, cur, q))
    return dict(u=pd.Series(u, index=dS.index), z=z, ssafe=ssafe, trades=trades)


def decisions_masked(dSsig, bd, entry_ok, **kw):
    return [decide_masked(dSsig, bd, zd, ze, entry_ok, **kw)
            for zd in w3.ZDS for ze in w3.ZES]


# ------------------------------------------------------------------ helpers

def run_pnl(F, dec, cens, shift=0):
    """x1/x2 ensemble through the frozen vec engine; shift=k delays fills to
    open[t+1+k] (RA5)."""
    out = {}
    for tag, stress in (('x1', 1.0), ('x2', 2.0)):
        parts = [w3.vec_pnl(F, d['u'].shift(shift).fillna(0.0) if shift
                            else d['u'], w3.COST_TURN, 'open', stress, cens)
                 for d in dec]
        out[tag] = pd.concat(parts, axis=1).mean(axis=1)
    return out['x1'], out['x2'], w3.met(out['x1']), w3.met(out['x2'])


def episode_table(F, dec, cens, shift=0):
    """Episode-level table on the A8 basis (RA4): entry/exit fill bar, side,
    size, gross ticks, net x1 pnl (gross - 2 turns)."""
    O = F['o'].values
    rows = []
    for d in dec:
        u = d['u'].shift(shift).fillna(0.0) if shift else d['u']
        for (f, e, s, q) in w3.episodes(u):
            if e <= f:
                continue
            gross = s * (O[e] - O[f])
            rows.append(dict(f=f, e=e, s=s, q=q,
                             ticks=gross * 100.0,
                             pnl=q * gross - 2 * w3.COST_TURN * q,
                             eh=F.index[f].hour, xh=F.index[e].hour))
    return pd.DataFrame(rows)


def fmt(m):
    return ('Sh=%5.2f H1=%5.2f H2=%5.2f eqR2=%4.2f G/P=%5.2f maxDD=%6.1f '
            'tot=%6.1f' % (m['sharpe'], m['h1'], m['h2'], m['eqR2'],
                           m['gain_pain'], m['maxdd_vu'], m['total_vu']))


def lag1_ac(x, hours, ok):
    """lag-1 autocorr overall + by hour-of-day of t, on consecutive-hour
    uncensored pairs. x: pd.Series of changes; hours: hour of t; ok: pair
    validity mask aligned to t."""
    a, b = x.values, x.shift(1).values
    m = ok & np.isfinite(a) & np.isfinite(b)
    overall = float(np.corrcoef(a[m], b[m])[0, 1])
    byh = {}
    for h in range(24):
        mm = m & (hours == h)
        if mm.sum() >= 50:
            byh[h] = float(np.corrcoef(a[mm], b[mm])[0, 1])
    return overall, byh


# ================================================================== main

def main():
    print(__doc__)

    # ---------------- construction + fidelity assertions ----------------
    print('================ C2 CONSTRUCTION (frozen) ================')
    A, B, F, cens, dSsig, bd = build_c2_shifted(0)
    F0, cens0, dS0, bd0 = w4.build_c2()
    assert F.equals(F0) and cens.equals(cens0) and dS0.equals(dSsig) \
        and bd == bd0, 'build_c2_shifted(0) != w4.build_c2'
    print('RA7 PASS: build_c2_shifted(0) == w4.build_c2 (bit-identical).')
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    print('bars=%d  %s -> %s  (%.2fy, %.1f bars/day)  censored=%d (%.2f%%)'
          % (len(F), F.index[0].date(), F.index[-1].date(), yrs, bd,
             int(cens.sum()), 100 * cens.mean()))

    dec = w3.decisions_for(dSsig, bd)
    all_true = np.ones(len(F), bool)
    dec_chk = decisions_masked(dSsig, bd, all_true)
    for da, db in zip(dec, dec_chk):
        assert da['u'].equals(db['u']) and da['trades'] == db['trades']
    print('RA3 PASS: decide_masked(all-True) reproduces w3.decide on all 8 '
          'subs (u + trades bit-identical).')

    a1, a2, m1, m2 = run_pnl(F, dec, cens)
    print('ALIGNED  x1: %s' % fmt(m1))
    print('ALIGNED  x2: %s' % fmt(m2))
    print('(W4-A published C2: x1 1.21 / x2 0.99 — must match)')
    base = m1['sharpe']
    ep0 = episode_table(F, dec, cens)
    print('episodes=%d | gross edge mean=%.1f ticks med=%.1f ticks vs '
          'round-trip cost %.0f ticks'
          % (len(ep0), ep0.ticks.mean(), ep0.ticks.median(), w4.TICKS_RT))

    # ---------------- battery 1: DELAY ----------------
    print('\n================ 1. DELAY BATTERY (entries at open[t+2], '
          'open[t+3]) ================')
    delays = {}
    for k in (1, 2):
        d1, d2x, md1, md2 = run_pnl(F, dec, cens, shift=k)
        epk = episode_table(F, dec, cens, shift=k)
        delays[k] = dict(p1=d1, m1=md1, m2=md2, ep=epk)
        print('DELAY-%d x1: %s' % (k, fmt(md1)))
        print('        x2: %s' % fmt(md2))
        print('        per-trade gross edge: mean=%.1f ticks med=%.1f ticks '
              '(aligned: %.1f/%.1f) vs cost %.0f ticks'
              % (epk.ticks.mean(), epk.ticks.median(), ep0.ticks.mean(),
                 ep0.ticks.median(), w4.TICKS_RT))
    g1 = delays[1]['m1']['sharpe'] >= 0.5 * base
    print('GATE 1: delay-1 x1 = %.2f vs required >= %.2f (50%% of %.2f) -> %s'
          % (delays[1]['m1']['sharpe'], 0.5 * base, base,
             'PASS' if g1 else 'FAIL'))
    print('  halves under delay-1: H1=%.2f H2=%.2f'
          % (delays[1]['m1']['h1'], delays[1]['m1']['h2']))
    print('  [C1 cited, W4-B: honest %.2f -> delay-1 %.2f]'
          % (C1_CITED['honest'], C1_CITED['delay1']))

    # ---------------- battery 2: MISALIGNMENT ----------------
    print('\n================ 2. MISALIGNMENT CONTROL (CL leg shifted +1h / '
          '-1h, full pipeline rerun) ================')
    mis = {}
    for sh in (+1, -1):
        Am, Bm, Fm, cm, dSm, bdm = build_c2_shifted(sh)
        decm = w3.decisions_for(dSm, bdm)
        p1m, p2m, mm1, mm2 = run_pnl(Fm, decm, cm)
        mis[sh] = dict(p1=p1m, m1=mm1, m2=mm2, bars=len(Fm))
        print('CL %+dh x1: %s  (bars=%d bd=%.1f cens=%d)'
              % (sh, fmt(mm1), len(Fm), bdm, int(cm.sum())))
        print('       x2: %s' % fmt(mm2))
    worst = max(mis[+1]['m1']['sharpe'], mis[-1]['m1']['sharpe'])
    g2 = worst < 1.5 * base
    print('GATE 2: max(misaligned x1) = %.2f vs bound < %.2f (1.5 x %.2f) -> %s'
          % (worst, 1.5 * base, base, 'PASS' if g2 else 'FAIL'))
    print('  [C1 cited, W4-B: honest %.2f -> WTI +1h misaligned %.2f '
          '(SUPERCHARGED — the async-artifact signature)]'
          % (C1_CITED['honest'], C1_CITED['mis_p1']))

    # ---------------- battery 3: LIQUID HOURS ----------------
    print('\n================ 3. LIQUID-HOURS TEST ================')
    medv, allmed = {}, {}
    for nm, L in (('BZ', A), ('CL', B)):
        medv[nm] = L['v'].groupby(L.index.hour).median()
        allmed[nm] = float(L['v'].median())
        print('%s overall median volume = %.0f | per-hour median: %s'
              % (nm, allmed[nm],
                 {h: int(v) for h, v in medv[nm].items()}))
    liqA = sorted(h for h in medv['BZ'].index
                  if medv['BZ'][h] > allmed['BZ']
                  and h in medv['CL'].index and medv['CL'][h] > allmed['CL'])
    print('classification A (both legs above own median): liquid hours = %s'
          % liqA)
    print('classification B (US session): liquid hours = %s' % sorted(US_HOURS))

    hours_arr = F.index.hour.values
    runs3 = {}
    for nm, hset in (('liquid-A-only', set(liqA)),
                     ('liquid-B-only (US 13-20)', US_HOURS),
                     ('thin-A-only (converse)',
                      set(range(24)) - set(liqA))):
        mask = np.isin(hours_arr, sorted(hset))
        dech = decisions_masked(dSsig, bd, mask)
        p1h, p2h, mh1, mh2 = run_pnl(F, dech, cens)
        eph = episode_table(F, dech, cens)
        runs3[nm] = dict(p1=p1h, m1=mh1, m2=mh2, n=len(eph))
        print('%-26s x1: %s  (episodes=%d, retention %.0f%%)'
              % (nm, fmt(mh1), len(eph), 100 * mh1['sharpe'] / base))
        print('%-26s x2: %s' % ('', fmt(mh2)))
    g3 = runs3['liquid-A-only']['m1']['sharpe'] >= 0.5 * base
    print('GATE 3: liquid-A-only x1 = %.2f vs required >= %.2f -> %s '
          '(B reported: %.2f, retention %.0f%%)'
          % (runs3['liquid-A-only']['m1']['sharpe'], 0.5 * base,
             'PASS' if g3 else 'FAIL',
             runs3['liquid-B-only (US 13-20)']['m1']['sharpe'],
             100 * runs3['liquid-B-only (US 13-20)']['m1']['sharpe'] / base))

    # ---------------- battery 4: BOUNCE / STALENESS DIAGNOSTICS ----------------
    print('\n================ 4. BOUNCE / STALENESS DIAGNOSTICS ================')
    # (a) lag-1 autocorr of hourly changes, consecutive-hour uncensored pairs
    gap1 = np.r_[False, (F.index[1:] - F.index[:-1]) == pd.Timedelta(hours=1)]
    okpair = gap1 & ~cens.values & ~np.r_[False, cens.values[:-1]]
    ac_sp, ac_sp_h = lag1_ac(F['c'].diff(), hours_arr, okpair)
    ac_bz, ac_bz_h = lag1_ac(A['c'].diff(), hours_arr, okpair)
    ac_cl, ac_cl_h = lag1_ac(B['c'].diff(), hours_arr, okpair)
    print('(a) lag-1 autocorr of hourly changes (consecutive-hour, uncensored '
          'pairs): SPREAD %.3f | BZ %.3f | CL %.3f' % (ac_sp, ac_bz, ac_cl))
    print('    spread lag-1 AC by hour-of-day: %s'
          % {h: round(v, 2) for h, v in sorted(ac_sp_h.items())})
    print('    BZ     lag-1 AC by hour-of-day: %s'
          % {h: round(v, 2) for h, v in sorted(ac_bz_h.items())})
    print('    CL     lag-1 AC by hour-of-day: %s'
          % {h: round(v, 2) for h, v in sorted(ac_cl_h.items())})

    # (b) staleness proxy by hour
    stal = {}
    for nm, L in (('BZ', A), ('CL', B)):
        st = ((L['v'] == 0) | (L['h'] == L['l']))
        stal[nm] = st.groupby(L.index.hour).mean() * 100
        print('(b) %s staleness (v==0 or h==l) by hour: %s | overall %.1f%%'
              % (nm, {h: round(v, 1) for h, v in stal[nm].items()},
                 100 * st.mean()))

    # (c) profitable exits by hour
    prof = ep0[ep0.pnl > 0]
    pex = prof.groupby('xh').pnl.sum()
    pex_share = (pex / pex.sum()).sort_values(ascending=False)
    thinA = set(range(24)) - set(liqA)
    thin_exit_share = float(pex[pex.index.isin(thinA)].sum() / pex.sum())
    print('(c) profitable-exit pnl by exit hour (top 6 shares): %s'
          % {h: round(v, 3) for h, v in pex_share.head(6).items()})
    print('    share of profitable-exit pnl exiting in THIN-A hours (%s): '
          '%.1f%% (thin hours hold %.1f%% of bars)'
          % (sorted(thinA), 100 * thin_exit_share,
             100 * float(np.isin(hours_arr, sorted(thinA)).mean())))

    # (d) pnl attribution by entry-hour liquidity tercile
    score = {h: min(medv['BZ'].get(h, 0) / allmed['BZ'],
                    medv['CL'].get(h, 0) / allmed['CL']) for h in range(24)}
    order = sorted(range(24), key=lambda h: score[h])
    terc = {h: ('T1 thin' if i < 8 else 'T2 mid' if i < 16 else 'T3 liquid')
            for i, h in enumerate(order)}
    print('(d) hour liquidity terciles (score = min-leg normalized median '
          'volume):')
    for t in ('T1 thin', 'T2 mid', 'T3 liquid'):
        print('    %-9s hours %s' % (t, sorted(h for h in terc if terc[h] == t)))
    ep0['terc'] = ep0.eh.map(terc)
    attr = ep0.groupby('terc').pnl.sum()
    tot = float(ep0.pnl.sum())
    nattr = ep0.groupby('terc').size()
    shares = {}
    for t in ('T1 thin', 'T2 mid', 'T3 liquid'):
        s = float(attr.get(t, 0.0))
        shares[t] = s / tot
        print('    %-9s episodes=%4d  net pnl=%8.2f  (%5.1f%% of total)'
              % (t, int(nattr.get(t, 0)), s, 100 * s / tot))
    # context: classification-A thin class share
    ep0['liqA'] = ep0.eh.isin(set(liqA))
    thinA_share = float(ep0.loc[~ep0.liqA, 'pnl'].sum() / tot)
    print('    context: classification-A THIN class (entry) pnl share = %.1f%%'
          % (100 * thinA_share))
    g4 = shares['T1 thin'] <= 0.5
    print('GATE 4: T1-thin entry-tercile pnl share = %.1f%% vs bound <= 50%% '
          '-> %s' % (100 * shares['T1 thin'], 'PASS' if g4 else 'FAIL'))

    # ---------------- additional: BZ thin-overnight exclusion sensitivity ----
    print('\n================ ADDITIONAL (report, not gates) ================')
    bz_thin_on = sorted(h for h in range(24)
                        if medv['BZ'].get(h, 0) < allmed['BZ']
                        and h not in US_HOURS)
    keep = sorted(set(range(24)) - set(bz_thin_on))
    print('BZ thin overnight hours (BZ per-hour med vol < own median, outside '
          '13-19 UTC): %s -> entries kept in %s' % (bz_thin_on, keep))
    if set(keep) == set(liqA):
        print('NOTE: identical to the liquid-A entry set — sensitivity run '
              'coincides with the gate-3 liquid-A run:')
        mbz1, mbz2 = runs3['liquid-A-only']['m1'], runs3['liquid-A-only']['m2']
    else:
        decb = decisions_masked(dSsig, bd, np.isin(hours_arr, keep))
        _, _, mbz1, mbz2 = run_pnl(F, decb, cens)
    print('BZ-thin-overnight-excluded x1: %s' % fmt(mbz1))
    print('                           x2: %s' % fmt(mbz2))
    print('C1 side-by-side (CITED from W4-B, not re-run): honest %.2f | '
          'delay-1 %.2f | NY-open-bar pnl-censored %.2f | +1h misalign %.2f'
          % (C1_CITED['honest'], C1_CITED['delay1'], C1_CITED['opencens'],
             C1_CITED['mis_p1']))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    for p, lbl, col, lw in (
            (a1, 'aligned x1 (Sh %.2f)' % base, 'navy', 1.2),
            (delays[1]['p1'], 'delay-1 (Sh %.2f)'
             % delays[1]['m1']['sharpe'], 'steelblue', 1.0),
            (mis[+1]['p1'], 'CL +1h misaligned (Sh %.2f)'
             % mis[+1]['m1']['sharpe'], 'firebrick', 1.0),
            (runs3['liquid-A-only']['p1'], 'liquid-A-only entries (Sh %.2f)'
             % runs3['liquid-A-only']['m1']['sharpe'], 'seagreen', 1.0)):
        e = w3.trim(p).cumsum()
        ax.plot(e.index, e.values, lw=lw, color=col, label=lbl)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_title('C2 equity: aligned vs delay-1 vs misaligned vs liquid-only',
                 fontsize=10)
    ax.set_ylabel('cum pnl (vu)')

    ax = fig.add_subplot(gs[0, 1])
    names = ['aligned', 'delay-1', 'delay-2', 'CL +1h', 'CL -1h',
             'liqA only', 'US only', 'thinA only']
    v1 = [base, delays[1]['m1']['sharpe'], delays[2]['m1']['sharpe'],
          mis[+1]['m1']['sharpe'], mis[-1]['m1']['sharpe'],
          runs3['liquid-A-only']['m1']['sharpe'],
          runs3['liquid-B-only (US 13-20)']['m1']['sharpe'],
          runs3['thin-A-only (converse)']['m1']['sharpe']]
    v2 = [m2['sharpe'], delays[1]['m2']['sharpe'], delays[2]['m2']['sharpe'],
          mis[+1]['m2']['sharpe'], mis[-1]['m2']['sharpe'],
          runs3['liquid-A-only']['m2']['sharpe'],
          runs3['liquid-B-only (US 13-20)']['m2']['sharpe'],
          runs3['thin-A-only (converse)']['m2']['sharpe']]
    x = np.arange(len(names))
    ax.bar(x - 0.2, v1, 0.4, color='navy', label='x1')
    ax.bar(x + 0.2, v2, 0.4, color='firebrick', alpha=0.8, label='x2')
    for i in range(len(names)):
        ax.text(i - 0.2, v1[i] + 0.02, '%.2f' % v1[i], ha='center', fontsize=7)
        ax.text(i + 0.2, v2[i] + 0.02, '%.2f' % v2[i], ha='center', fontsize=7)
    ax.axhline(0.5 * base, color='gray', ls='--', lw=0.8,
               label='50%% of aligned (%.2f)' % (0.5 * base))
    ax.axhline(1.5 * base, color='black', ls=':', lw=0.8,
               label='1.5x aligned (%.2f)' % (1.5 * base))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('Sharpe')
    ax.set_title('battery Sharpes (C1 cited: honest 0.93 / delay-1 0.15 / '
                 '+1h 5.14)', fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    hs = sorted(ac_sp_h)
    ax.plot(hs, [ac_sp_h[h] for h in hs], 'o-', color='navy', lw=1.0,
            label='spread dS lag-1 AC')
    ax.plot(sorted(ac_bz_h), [ac_bz_h[h] for h in sorted(ac_bz_h)], 's--',
            color='seagreen', lw=0.8, ms=3, label='BZ leg')
    ax.plot(sorted(ac_cl_h), [ac_cl_h[h] for h in sorted(ac_cl_h)], '^--',
            color='goldenrod', lw=0.8, ms=3, label='CL leg')
    ax.axhline(0, color='gray', lw=0.5)
    axb = ax.twinx()
    axb.bar(stal['BZ'].index, stal['BZ'].values, color='firebrick', alpha=0.25,
            label='BZ stale% (right)')
    axb.set_ylabel('BZ stale %', fontsize=8)
    ax.set_xlabel('hour of day (UTC)'); ax.set_ylabel('lag-1 autocorr')
    ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)
    ax.set_title('bounce/staleness: lag-1 AC by hour + BZ staleness',
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    tlabels = ('T1 thin', 'T2 mid', 'T3 liquid')
    ax.bar(range(3), [100 * shares[t] for t in tlabels],
           color=['firebrick', 'goldenrod', 'seagreen'])
    for i, t in enumerate(tlabels):
        ax.text(i, 100 * shares[t] + 1, '%.0f%%' % (100 * shares[t]),
                ha='center', fontsize=9)
    ax.axhline(50, color='black', ls='--', lw=0.8, label='50% artifact bound')
    ax.set_xticks(range(3)); ax.set_xticklabels(tlabels, fontsize=8)
    ax.set_ylabel('% of total net pnl')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.set_title('pnl attribution by entry-hour liquidity tercile', fontsize=10)
    fig.suptitle('W5 adjudication of C2 (Yahoo BZ-CL, same-venue NYMEX legs, '
                 '%.2fy) — frozen rule, next-open fills, pre-registered '
                 'battery' % yrs, fontsize=11)
    out = lab.ROOT / 'curves' / 'w5_adjudication.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out)

    # ---------------- decision ----------------
    print('\n================ PRE-REGISTERED DECISION RULE (mechanical) '
          '================')
    crit = [
        ('1. delay-1 retains >= 50% of aligned x1', g1,
         'delay1=%.2f aligned=%.2f' % (delays[1]['m1']['sharpe'], base)),
        ('2. misaligned < 1.5x aligned (both +/-1h)', g2,
         '+1h=%.2f -1h=%.2f bound=%.2f' % (mis[+1]['m1']['sharpe'],
                                           mis[-1]['m1']['sharpe'],
                                           1.5 * base)),
        ('3. liquid-A-only retains >= 50%', g3,
         'liqA=%.2f required=%.2f' % (runs3['liquid-A-only']['m1']['sharpe'],
                                      0.5 * base)),
        ('4. no thin-hour class > 50% of pnl', g4,
         'T1=%.1f%% (thinA class %.1f%%)' % (100 * shares['T1 thin'],
                                             100 * thinA_share)),
    ]
    all_pass = True
    for name, ok, det in crit:
        all_pass &= bool(ok)
        print('  [%s] %s  (%s)' % ('PASS' if ok else 'FAIL', name, det))
    decision = 'PROMOTE-CONFIRMED' if all_pass else 'KILL'
    print('\nDECISION (mechanical): %s' % decision)

    # ---------------- machine-readable ----------------
    print('\n===== SUMMARY (machine-readable) =====')
    print('ALIGNED_X1=%.2f ALIGNED_X2=%.2f H1=%.2f H2=%.2f'
          % (base, m2['sharpe'], m1['h1'], m1['h2']))
    print('DELAY1_X1=%.2f DELAY1_X2=%.2f DELAY1_H1=%.2f DELAY1_H2=%.2f '
          'DELAY2_X1=%.2f DELAY2_X2=%.2f'
          % (delays[1]['m1']['sharpe'], delays[1]['m2']['sharpe'],
             delays[1]['m1']['h1'], delays[1]['m1']['h2'],
             delays[2]['m1']['sharpe'], delays[2]['m2']['sharpe']))
    print('TICKS_ALIGNED=%.1f/%.1f TICKS_D1=%.1f/%.1f TICKS_D2=%.1f/%.1f '
          '(mean/med, cost %.0f)'
          % (ep0.ticks.mean(), ep0.ticks.median(),
             delays[1]['ep'].ticks.mean(), delays[1]['ep'].ticks.median(),
             delays[2]['ep'].ticks.mean(), delays[2]['ep'].ticks.median(),
             w4.TICKS_RT))
    print('MIS_P1_X1=%.2f MIS_P1_X2=%.2f MIS_M1_X1=%.2f MIS_M1_X2=%.2f'
          % (mis[+1]['m1']['sharpe'], mis[+1]['m2']['sharpe'],
             mis[-1]['m1']['sharpe'], mis[-1]['m2']['sharpe']))
    print('LIQA_X1=%.2f LIQA_X2=%.2f LIQB_X1=%.2f LIQB_X2=%.2f THIN_X1=%.2f '
          'THIN_X2=%.2f LIQA_HOURS=%s'
          % (runs3['liquid-A-only']['m1']['sharpe'],
             runs3['liquid-A-only']['m2']['sharpe'],
             runs3['liquid-B-only (US 13-20)']['m1']['sharpe'],
             runs3['liquid-B-only (US 13-20)']['m2']['sharpe'],
             runs3['thin-A-only (converse)']['m1']['sharpe'],
             runs3['thin-A-only (converse)']['m2']['sharpe'], liqA))
    print('AC_SPREAD=%.3f AC_BZ=%.3f AC_CL=%.3f' % (ac_sp, ac_bz, ac_cl))
    print('AC_SPREAD_BYH=%s' % {h: round(v, 2) for h, v in sorted(ac_sp_h.items())})
    print('STALE_BZ_OVERALL=%.1f STALE_CL_OVERALL=%.1f'
          % (float(((A['v'] == 0) | (A['h'] == A['l'])).mean() * 100),
             float(((B['v'] == 0) | (B['h'] == B['l'])).mean() * 100)))
    print('PNL_SHARE_T1=%.3f PNL_SHARE_T2=%.3f PNL_SHARE_T3=%.3f '
          'THINA_CLASS_SHARE=%.3f THIN_EXIT_SHARE=%.3f'
          % (shares['T1 thin'], shares['T2 mid'], shares['T3 liquid'],
             thinA_share, thin_exit_share))
    print('BZ_EXCL_X1=%.2f BZ_EXCL_X2=%.2f' % (mbz1['sharpe'], mbz2['sharpe']))
    print('GATES=%s' % {'g1_delay': g1, 'g2_misalign': g2, 'g3_liquid': g3,
                        'g4_attribution': g4})
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
