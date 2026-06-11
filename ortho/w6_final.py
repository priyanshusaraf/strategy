#!/usr/bin/env /usr/bin/python3
"""
W6 — THE FINAL, DISCRIMINATING EXPERIMENT (pre-registered, README "Wave 6").
The program's last empirical act.

CONTEXT: W5 killed C2 by the pre-registered misalignment gate (CL +/-1h
supercharges the frozen rule to 4.56/4.27 vs bound 1.81) while recording
honestly that the gate, as designed, is unpassable in principle for a z-MR
rule under NO-delay fills: misaligning a leg injects S' = S + dCL[t] — large
stationary noise added to a level — and any z-MR rule harvests added
stationary noise by construction. The adjudicator's prescription: an edge
claim from this family is admissible only from an engine whose performance
DEGRADES under deliberate misalignment. Delay-1 execution is exactly such an
engine candidate: the injected noise dCL[t] renews each bar, so it leaves the
spread before a delayed fill; misalignment-under-delay should collapse while
a genuine multi-bar edge survives.

ENGINE (the strategy's DEFINITION, not a stress test): frozen Program-1 rule
(28td detrend, z ensemble {9,12,18,24}td x {2.5,3.0}, exit 0.75, stops
4.0/10td, vol gate, risk-frozen sizing, 1.5 ticks/leg/turn) with ALL fills —
entries AND exits (including stop and time-stop exits) — executed at
open[t+2] instead of open[t+1]. Implemented as u.shift(1) into the frozen
vec engine (w5 RA5 pattern: vec_pnl itself shifts once more, so the entire
position path, hence every transition, fills one bar later). Decisions are
untouched (frozen rule observes close[t] exactly as before); only fills move.

DATA HYGIENE (binding, from the W5 addendum): hour-22-UTC bars (Globex
break, 67-94% stale) are dropped AT INGESTION — per leg, after hour-flooring,
before deduplication / Sunday drop / any misalignment shift — for EVERY
construction, and all joins are recomputed after the drop.

CONSTRUCTIONS:
  A. C2 aligned (Yahoo BZ-CL, both legs NYMEX futures, roll-censored) — the
     candidate.
  B. C2 with the CL leg shifted +1h, and with CL shifted -1h — the
     certification controls (misalignment under delayed execution).
  C. C1 (TV BRN1! - CFI_WTI 60-min) aligned — consistency check (its edge was
     adjudicated artifact in W4-B/W5; expected to stay dead under this engine).

REPORT (each construction, delayed engine): x1 and x2 Sharpe, halves, eqR2,
gain/pain, maxDD, trades/yr, per-trade gross tick edge (mean AND median) vs
the 6-tick round-trip cost, yearly Sharpes. For A also: equity figure
(../curves/w6_final.png: A vs B+1h vs B-1h vs C, labeled), longest
underwater, param micro-neighborhood (exit 0.5/1.0, detrend 21/35td) as
REPORTING only.

MECHANISM CHECK (report): quantify how much of the no-delay misalignment
supercharge (4.56/4.27 in wave 5) survives delayed execution — both as the
raw ratio B_delay/B_nodelay and as the survival of the misalignment EXCESS
(B - A) from no-delay to delay-1, under wave-6 hygiene; the wave-5
no-hygiene numbers are reproduced exactly first (fidelity).

PRE-REGISTERED DECISION RULE (README Wave 6, mechanical):
  PROMOTE-AS-BOUNDED-CANDIDATE iff ALL of
    1. A x1 >= 0.6 AND A x2 >= 0.4 AND A halves both > 0;
    2. max(B+1h, B-1h) x1 < 1.5 * A x1 (the certification gate — if
       misalignment still supercharges under delay, kill irrevocably);
    3. C x1 < 0.3 (consistency: its edge was artifact).
  Else FINAL-KILL and the program concludes with the documented negative
  result. EITHER WAY: the 2.39y sample and single-construction evidence
  bound any positive claim to 'candidate requiring extended validation' —
  never 'proven robust edge'.

FIDELITY (asserted at runtime against the w5 machinery):
  F1. build_c2_w6(shift=0, drop_h22=False) == w4.build_c2() bit-identical
      (w5 RA7, re-asserted here through the new builder).
  F2. build_c1_w6(drop_h22=False) == w3.tv_spread() bit-identical.
  F3. no-drop no-delay C2 ensemble reproduces W4-A published x1 1.21 / x2
      0.99 exactly (2dp).
  F4. no-drop delay-1 reproduces W5 published 1.19 / 0.98 exactly (2dp).
  F5. no-drop no-delay misalignment reproduces W5 published +1h 4.56 /
      -1h 4.27 exactly (2dp).
  F6. w5.decide_masked(all-True) reproduces w3.decide on the wave-6 A data
      (u + trades bit-identical on all 8 subs) — w5 RA3 re-asserted on the
      hour-22-dropped frame.

RESOLVED AMBIGUITIES (conservative, documented):
  RB1. Hygiene-vs-misalignment order: hour-22 bars are dropped at ingestion
       (the data layer), THEN the control shift is applied to the cleaned CL
       leg. The misaligned joins therefore carry no hour-22 stamps either
       (the shifted orphans die in the inner join); bar counts reported.
  RB2. Leg roll-censor flags ('cens') come from lab.yahoo at ingestion,
       before the drop — identical treatment to w5 (only joins are
       recomputed, per the pre-registration).
  RB3. trades/yr on the A8 position-episode basis (w5.episode_table),
       totalled across the 8 ensemble subs and also shown per sub.
  RB4. The micro-neighborhood is run on A under the delayed engine
       (the engine is the strategy definition now) — reporting only.
  RB5. B's no-delay runs reuse B's own decisions (decisions are
       fill-independent); only the fill shift differs.

Run: cd ortho && /usr/bin/python3 w6_final.py
Figure: ../curves/w6_final.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab
import w3_brnwti_honest as w3
import w4_brnwti_futures as w4
import w5_adjudication as w5

DELAY = 1                                   # all fills at open[t+2] (RA5 pattern)
W5_PUB = dict(aligned_x1=1.21, aligned_x2=0.99,      # W4-A published C2
              delay1_x1=1.19, delay1_x2=0.98,        # W5 gate-1 published
              mis_p1=4.56, mis_m1=4.27)              # W5 gate-2 published


# ------------------------------------------------------------------ builders

def build_c2_w6(shift_hours=0, drop_h22=True):
    """w5.build_c2_shifted with hour-22-UTC bars dropped at ingestion (RB1).
    drop_h22=False, shift=0 must equal w4.build_c2() bit-identically (F1)."""
    legs = []
    for k, sym in enumerate(('BZ', 'CL')):
        d = lab.yahoo(sym)
        d.index = d.index.floor('h')
        if drop_h22:
            d = d[d.index.hour != 22]
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


def build_c1_w6(drop_h22=True):
    """w3.tv_spread with hour-22-UTC bars dropped at ingestion; MAD roll
    censor recomputed on the new join. drop_h22=False must equal
    w3.tv_spread() bit-identically (F2)."""
    out = []
    for f in ('ICEEUR_DLY_BRN1!, 60 (1).csv', 'CFI_WTI, 60.csv'):
        X = lab.tv(f)
        X = X.copy()
        X.index = X.index.floor('h')
        if drop_h22:
            X = X[X.index.hour != 22]
        X = X[~X.index.duplicated()]
        X = X[X.index.dayofweek != 6]
        out.append(X)
    A, B = w3.join(*out)
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
    return F, cens, dSsig, w3.bars_day(F.index)


# ------------------------------------------------------------------ reporting

def full_report(tag, F, dec, cens, shift=DELAY):
    """The pre-registered per-construction report (delayed engine)."""
    x1, x2, m1, m2 = w5.run_pnl(F, dec, cens, shift=shift)
    ep = w5.episode_table(F, dec, cens, shift=shift)
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    print('%s x1: %s' % (tag, w5.fmt(m1)))
    print('%s x2: %s' % (' ' * len(tag), w5.fmt(m2)))
    print('%s    trades: %d episodes (%.0f/yr all 8 subs; %.1f/yr/sub) | '
          'gross edge mean=%.1f ticks med=%.1f ticks vs round-trip cost '
          '%.0f ticks'
          % (' ' * len(tag), len(ep), len(ep) / yrs, len(ep) / yrs / len(dec),
             ep.ticks.mean(), ep.ticks.median(), w4.TICKS_RT))
    print('%s    yearly Sharpe (x1): %s'
          % (' ' * len(tag), lab.yearly(w3.trim(x1)).to_dict()))
    return dict(x1=x1, x2=x2, m1=m1, m2=m2, ep=ep, yrs=yrs)


# ================================================================== main

def main():
    print(__doc__)

    # ---------------- fidelity battery (F1-F5: machinery == w4/w5) ----------
    print('================ FIDELITY BATTERY (asserted) ================')
    A0, B0, F0, c0, dS0, bd0 = build_c2_w6(0, drop_h22=False)
    F0w, c0w, dS0w, bd0w = w4.build_c2()
    assert F0.equals(F0w) and c0.equals(c0w) and dS0.equals(dS0w) \
        and bd0 == bd0w, 'F1 FAIL: build_c2_w6(0, drop_h22=False) != w4.build_c2'
    print('F1 PASS: build_c2_w6(0, drop_h22=False) == w4.build_c2 '
          '(bit-identical).')

    F1c, c1c, dS1c, bd1c = build_c1_w6(drop_h22=False)
    F1w, c1w, dS1w, bd1w = w3.tv_spread()
    assert F1c.equals(F1w) and c1c.equals(c1w) and dS1c.equals(dS1w) \
        and bd1c == bd1w, 'F2 FAIL: build_c1_w6(drop_h22=False) != w3.tv_spread'
    print('F2 PASS: build_c1_w6(drop_h22=False) == w3.tv_spread '
          '(bit-identical).')

    dec0 = w3.decisions_for(dS0, bd0)
    _, _, m0a, m0b = w5.run_pnl(F0, dec0, c0, shift=0)
    assert (m0a['sharpe'], m0b['sharpe']) == (W5_PUB['aligned_x1'],
                                              W5_PUB['aligned_x2']), \
        'F3 FAIL: no-drop aligned %.2f/%.2f != published 1.21/0.99' \
        % (m0a['sharpe'], m0b['sharpe'])
    print('F3 PASS: no-drop no-delay C2 reproduces W4-A published x1 %.2f / '
          'x2 %.2f.' % (m0a['sharpe'], m0b['sharpe']))

    _, _, md1, md2 = w5.run_pnl(F0, dec0, c0, shift=1)
    assert (md1['sharpe'], md2['sharpe']) == (W5_PUB['delay1_x1'],
                                              W5_PUB['delay1_x2']), \
        'F4 FAIL: no-drop delay-1 %.2f/%.2f != published 1.19/0.98' \
        % (md1['sharpe'], md2['sharpe'])
    print('F4 PASS: no-drop delay-1 reproduces W5 published x1 %.2f / x2 %.2f.'
          % (md1['sharpe'], md2['sharpe']))

    mis0 = {}
    for sh, key in ((+1, 'mis_p1'), (-1, 'mis_m1')):
        _, _, Fm, cm, dSm, bdm = build_c2_w6(sh, drop_h22=False)
        decm = w3.decisions_for(dSm, bdm)
        _, _, mm1, _ = w5.run_pnl(Fm, decm, cm, shift=0)
        mis0[sh] = mm1['sharpe']
        assert mm1['sharpe'] == W5_PUB[key], \
            'F5 FAIL: no-drop no-delay CL%+dh %.2f != published %.2f' \
            % (sh, mm1['sharpe'], W5_PUB[key])
    print('F5 PASS: no-drop no-delay misalignment reproduces W5 published '
          '+1h %.2f / -1h %.2f.' % (mis0[+1], mis0[-1]))

    # ---------------- wave-6 constructions (hour-22 dropped at ingestion) ---
    print('\n================ WAVE-6 CONSTRUCTIONS (hour-22-UTC bars dropped '
          'at ingestion, joins recomputed) ================')
    LA, LB, FA, cA, dSA, bdA = build_c2_w6(0, drop_h22=True)
    yrsA = (FA.index[-1] - FA.index[0]).days / 365.25
    print('A  C2 aligned      : bars=%d (%d dropped vs no-drop join) %s -> %s '
          '(%.2fy, %.1f bars/day) censored=%d (%.2f%%)'
          % (len(FA), len(F0) - len(FA), FA.index[0].date(),
             FA.index[-1].date(), yrsA, bdA, int(cA.sum()), 100 * cA.mean()))
    for nm, L in (('BZ', LA), ('CL', LB)):
        st = float(((L['v'] == 0) | (L['h'] == L['l'])).mean() * 100)
        print('   %s joined-leg staleness proxy (v==0 or h==l) after drop: '
              '%.1f%%' % (nm, st))

    decA = w3.decisions_for(dSA, bdA)
    all_true = np.ones(len(FA), bool)
    dec_chk = w5.decisions_masked(dSA, bdA, all_true)
    for da, db in zip(decA, dec_chk):
        assert da['u'].equals(db['u']) and da['trades'] == db['trades']
    print('F6 PASS: w5.decide_masked(all-True) == w3.decide on the wave-6 A '
          'frame (u + trades bit-identical, all 8 subs).')

    misB = {}
    for sh in (+1, -1):
        _, _, Fm, cm, dSm, bdm = build_c2_w6(sh, drop_h22=True)
        h22 = int((Fm.index.hour == 22).sum())
        misB[sh] = dict(F=Fm, cens=cm, dec=w3.decisions_for(dSm, bdm),
                        bars=len(Fm))
        print('B  C2 CL%+dh       : bars=%d (%.1f bars/day) censored=%d | '
              'hour-22 stamps in join=%d (RB1)'
              % (sh, len(Fm), bdm, int(cm.sum()), h22))

    FC, cC, dSC, bdC = build_c1_w6(drop_h22=True)
    yrsC = (FC.index[-1] - FC.index[0]).days / 365.25
    print('C  C1 TV BRN1!-WTI : bars=%d (%d dropped vs no-drop join) %s -> %s '
          '(%.2fy, %.1f bars/day) censored=%d'
          % (len(FC), len(F1c) - len(FC), FC.index[0].date(),
             FC.index[-1].date(), yrsC, bdC, int(cC.sum())))
    decC = w3.decisions_for(dSC, bdC)

    # ---------------- THE EXPERIMENT: delayed execution as definition -------
    print('\n================ THE EXPERIMENT — ALL FILLS AT open[t+2] '
          '(entries, exits, stops, time-stops) ================')
    A = full_report('A  C2 aligned ', FA, decA, cA)
    Bp = full_report('B  C2 CL +1h  ', misB[+1]['F'], misB[+1]['dec'],
                     misB[+1]['cens'])
    Bm = full_report('B  C2 CL -1h  ', misB[-1]['F'], misB[-1]['dec'],
                     misB[-1]['cens'])
    C = full_report('C  C1 aligned ', FC, decC, cC)

    print('\nA extras:')
    print('  longest underwater: %.0f days' % w4.underwater_days(A['x1']))

    # ---------------- mechanism check ----------------
    print('\n================ MECHANISM CHECK — does the misalignment '
          'supercharge survive delay? ================')
    _, _, mA0, _ = w5.run_pnl(FA, decA, cA, shift=0)
    print('aligned A   no-delay x1 = %.2f | delay-1 x1 = %.2f  (hygiene '
          'applied; no-hygiene W5 pair was %.2f -> %.2f)'
          % (mA0['sharpe'], A['m1']['sharpe'], W5_PUB['aligned_x1'],
             W5_PUB['delay1_x1']))
    mech = {}
    for sh, R in ((+1, Bp), (-1, Bm)):
        _, _, mB0, _ = w5.run_pnl(misB[sh]['F'], misB[sh]['dec'],
                                  misB[sh]['cens'], shift=0)
        nd, dl = mB0['sharpe'], R['m1']['sharpe']
        exc_nd = nd - mA0['sharpe']
        exc_dl = dl - A['m1']['sharpe']
        surv_ratio = dl / nd if nd else np.nan
        surv_exc = exc_dl / exc_nd if exc_nd else np.nan
        mech[sh] = dict(nodelay=nd, delay=dl, exc_nd=exc_nd, exc_dl=exc_dl,
                        surv_ratio=surv_ratio, surv_exc=surv_exc)
        print('CL %+dh: no-delay x1 = %.2f (W5 no-hygiene %.2f) -> delay-1 '
              'x1 = %.2f | raw survival %.0f%% | supercharge EXCESS over '
              'aligned: %.2f -> %.2f (survival %.0f%%)'
              % (sh, nd, W5_PUB['mis_p1' if sh > 0 else 'mis_m1'], dl,
                 100 * surv_ratio, exc_nd, exc_dl, 100 * surv_exc))
    print('(Prediction registered in W5/README: the injected dCL[t] noise '
          'renews each bar; a fill delayed to open[t+2] should forfeit its '
          'reversion, so the supercharge should collapse under delay while a '
          'genuine multi-bar edge survives.)')

    # ---------------- A micro-neighborhood (reporting only) ----------------
    print('\n================ A PARAM MICRO-NEIGHBORHOOD (delayed engine, '
          'REPORTING only) ================')
    nbhd = {}
    for nm, kw in (('exit 0.50', dict(zx=0.50)), ('exit 1.00', dict(zx=1.00)),
                   ('detrend 21td', dict(detrend_d=21)),
                   ('detrend 35td', dict(detrend_d=35))):
        dv = w3.decisions_for(dSA, bdA, **kw)
        _, _, mv1, mv2 = w5.run_pnl(FA, dv, cA, shift=DELAY)
        nbhd[nm] = (mv1['sharpe'], mv2['sharpe'])
        print('  %-14s x1 = %5.2f   x2 = %5.2f' % (nm, mv1['sharpe'],
                                                   mv2['sharpe']))
    print('  %-14s x1 = %5.2f   x2 = %5.2f  (frozen)'
          % ('baseline', A['m1']['sharpe'], A['m2']['sharpe']))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.22, width_ratios=[1.5, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    for R, lbl, col, lw in (
            (A, 'A: C2 aligned (Sh %.2f)' % A['m1']['sharpe'], 'navy', 1.4),
            (Bp, 'B: C2 CL +1h misaligned (Sh %.2f)' % Bp['m1']['sharpe'],
             'firebrick', 1.0),
            (Bm, 'B: C2 CL -1h misaligned (Sh %.2f)' % Bm['m1']['sharpe'],
             'darkorange', 1.0),
            (C, 'C: C1 TV BRN1!-CFI_WTI (Sh %.2f)' % C['m1']['sharpe'],
             'gray', 1.0)):
        e = w3.trim(R['x1']).cumsum()
        ax.plot(e.index, e.values, lw=lw, color=col, label=lbl)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylabel('cum pnl (vu)')
    ax.set_title('W6 final engine — ALL fills at open[t+2], hour-22 bars '
                 'dropped at ingestion\nx1 equity: candidate vs misalignment '
                 'certification controls vs C1 consistency', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    names = ['A aligned', 'B +1h', 'B -1h', 'C (C1)']
    v1 = [A['m1']['sharpe'], Bp['m1']['sharpe'], Bm['m1']['sharpe'],
          C['m1']['sharpe']]
    v2 = [A['m2']['sharpe'], Bp['m2']['sharpe'], Bm['m2']['sharpe'],
          C['m2']['sharpe']]
    x = np.arange(len(names))
    ax.bar(x - 0.2, v1, 0.4, color='navy', label='x1')
    ax.bar(x + 0.2, v2, 0.4, color='firebrick', alpha=0.8, label='x2')
    for i in range(len(names)):
        ax.text(i - 0.2, v1[i] + 0.02, '%.2f' % v1[i], ha='center', fontsize=8)
        ax.text(i + 0.2, v2[i] + 0.02, '%.2f' % v2[i], ha='center', fontsize=8)
    ax.axhline(0.6, color='navy', ls='--', lw=0.8, label='A gate: x1 >= 0.6')
    ax.axhline(1.5 * A['m1']['sharpe'], color='black', ls=':', lw=0.8,
               label='B bound: < 1.5 x A (%.2f)' % (1.5 * A['m1']['sharpe']))
    ax.axhline(0.3, color='gray', ls='-.', lw=0.8, label='C bound: < 0.3')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('decision-rule Sharpes (delayed engine)\n[W5 no-delay '
                 'misalignment was 4.56 / 4.27]', fontsize=9)
    out = lab.ROOT / 'curves' / 'w6_final.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out)

    # ---------------- decision ----------------
    print('\n================ PRE-REGISTERED DECISION RULE (mechanical) '
          '================')
    aok = (A['m1']['sharpe'] >= 0.6 and A['m2']['sharpe'] >= 0.4
           and A['m1']['h1'] > 0 and A['m1']['h2'] > 0)
    worstB = max(Bp['m1']['sharpe'], Bm['m1']['sharpe'])
    bok = worstB < 1.5 * A['m1']['sharpe']
    cok = C['m1']['sharpe'] < 0.3
    crit = [
        ('1. A x1 >= 0.6 AND x2 >= 0.4 AND halves > 0', aok,
         'x1=%.2f x2=%.2f h1=%.2f h2=%.2f' % (A['m1']['sharpe'],
                                              A['m2']['sharpe'],
                                              A['m1']['h1'], A['m1']['h2'])),
        ('2. max(B+1h, B-1h) < 1.5 x A (x1)', bok,
         '+1h=%.2f -1h=%.2f bound=%.2f' % (Bp['m1']['sharpe'],
                                           Bm['m1']['sharpe'],
                                           1.5 * A['m1']['sharpe'])),
        ('3. C < 0.3 (x1)', cok, 'C=%.2f' % C['m1']['sharpe']),
    ]
    all_pass = True
    for name, ok, det in crit:
        all_pass &= bool(ok)
        print('  [%s] %s  (%s)' % ('PASS' if ok else 'FAIL', name, det))
    decision = 'PROMOTE-AS-BOUNDED-CANDIDATE' if all_pass else 'FINAL-KILL'
    print('\nDECISION (mechanical): %s' % decision)
    print('\nBINDING SCOPE STATEMENT (either verdict): the sample is 2.39y '
          'on a single construction (Yahoo BZ-CL). Any positive claim is '
          'bounded to "candidate requiring extended validation" — never '
          '"proven robust edge".')

    # ---------------- machine-readable ----------------
    print('\n===== SUMMARY (machine-readable) =====')
    for tag, R in (('A', A), ('BP1', Bp), ('BM1', Bm), ('C', C)):
        print('%s_X1=%.2f %s_X2=%.2f %s_H1=%.2f %s_H2=%.2f %s_EQR2=%.2f '
              '%s_GP=%.2f %s_MAXDD=%.1f %s_TRYR=%.1f %s_TICKS=%.1f/%.1f'
              % (tag, R['m1']['sharpe'], tag, R['m2']['sharpe'],
                 tag, R['m1']['h1'], tag, R['m1']['h2'],
                 tag, R['m1']['eqR2'], tag, R['m1']['gain_pain'],
                 tag, R['m1']['maxdd_vu'], tag, len(R['ep']) / R['yrs'],
                 tag, R['ep'].ticks.mean(), R['ep'].ticks.median()))
        print('%s_YEARLY=%s' % (tag, lab.yearly(w3.trim(R['x1'])).to_dict()))
    print('A_NODELAY_X1=%.2f A_UWDAYS=%.0f' % (mA0['sharpe'],
                                               w4.underwater_days(A['x1'])))
    print('MECH_P1=%s' % {k: round(v, 3) for k, v in mech[+1].items()})
    print('MECH_M1=%s' % {k: round(v, 3) for k, v in mech[-1].items()})
    print('NBHD=%s' % nbhd)
    print('GATES=%s' % {'g1_A': aok, 'g2_B': bok, 'g3_C': cok})
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
