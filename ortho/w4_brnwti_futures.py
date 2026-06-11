#!/usr/bin/env /usr/bin/python3
"""
W4-A — HONEST BRN-WTI SPREAD MR ON REAL FUTURES FEEDS ONLY (pre-registered, README
"Wave 4"). The only surviving lead after 14 falsifications: real ICE futures quotes
showed honest Sharpe ~0.93 over 3.2y (W3-B cross-feed) while the CFD feed shows
~0.37 over 12.3y. Settle whether a modest real-futures edge genuinely exists.

FROZEN RULE: exact Program-1 rule as implemented in w3_brnwti_honest.py (28td
detrend, z ensemble {9,12,18,24}td x {2.5,3.0}, exit 0.75, stops 4.0/10td,
5d-vol>6m-median gate, risk-frozen sizing, NEXT-OPEN fills, 1.5 ticks/leg/turn,
stress 3.0 ticks at x2). NO re-tuning. td->bars from each dataset's actual
bars/day. All engine machinery imported from w3_brnwti_honest (apparatus controls
already validated there: OU harvester 0.8+ / RW ~0 through the same fills/cost
accounting).

CONSTRUCTIONS (spreads built manually from legs; pre-built TV spread = corroboration
only):
  C1: TV 'ICEEUR_DLY_BRN1!, 60' minus 'CFI_WTI, 60' — exact W3-B tv_spread()
      (hour-floored, Sundays dropped, inner join, 8xMAD day-boundary roll censor).
      HONESTY FLAG: the WTI leg (CFI_WTI) is itself a CFD feed; only BRN1! is a
      real ICE futures quote.
  C2: Yahoo lab.yahoo('BZ') minus lab.yahoo('CL') — real futures legs, censor-aware
      (cens bars excluded from signal AND pnl via vec_pnl roll-censor path).
  C3 (corroboration only, NOT a judgment leg): pre-built TV spread
      cl_brn_spread_60.csv. Internship cohort manifest flags it CONTAMINATED
      (leg-stripped precompute, legs absent, H/L untrusted, mixed_tz, 175 large
      gaps). Pre-registration says it is CL-BRN -> flip sign; CONFIRMED
      empirically: lag-0 hourly diff corr vs C1 is -0.62 while all +/-1..4h lags
      are ~0, so orientation is opposite to BRN-WTI and alignment is correct.
  C4 (context only, cited not re-run): W3-B clean duka 12.3y honest = 0.37.

RESOLVED AMBIGUITIES (conservative, documented):
  R1. Yahoo legs are floored to the hour; partial trailing bars collapse into
      duplicates and are dropped (keep-first). Sundays dropped from ALL
      constructions (W3-B hygiene A1; tv_spread already does it; Yahoo Sunday
      bars are 1.4% of legs).
  R2. C2 censoring = union of leg cens flags on the joined index; dSsig zeroed
      and vec_pnl cens path used (signal AND pnl excluded, re-roll turn charged
      while held) — identical treatment to C1.
  R3. C3 gets the same 8xMAD day-boundary censor (rolling-504 MAD) on the single
      flipped series; its continuous legs roll inside the precompute.
  R4. Pooled book = equal-risk: each construction's trimmed x1 pnl divided by its
      own full-sample x1 std, summed on the union index (missing bars = 0, i.e.
      flat). x2 pool uses the SAME x1 normalizers so the cost drag stays visible.
  R5. 15-min corroboration uses the same construction as C1 with 15min flooring
      and the MAD window scaled x4 (2016 bars, min_periods 400) to keep the same
      ~21-day lookback. Corroboration only (~1y).
  R6. Per-trade edge in ticks = gross open->open move per position episode
      (s*(O[exit]-O[entry]) in $/bbl x 100), unweighted; round-trip cost under the
      frozen spec = 2 turns x $0.03 = 6 ticks ($0.06). Approximate for censored
      episodes (engine zeroes censored bars; the raw open walk does not).
  R7. C1-vs-C3 shared-window agreement = both x1 pnl series sliced to the common
      calendar window [max(starts), min(ends)]; report each Sharpe + daily-pnl
      correlation + feed diff-corr by year.

PRE-REGISTERED DECISION RULE (README Wave 4, applied mechanically):
  promote iff  x1 Sharpe >= 0.6 on BOTH C1 and C2
          AND  pooled halves both > 0
          AND  pooled x2-cost Sharpe >= 0.3
          AND  no contradiction: C3 must not contradict C1 over the shared window
               (operationalized: C3 honest x1 Sharpe over the shared window > 0
               if C1 passes), and the claim is limited to 'modest recent-regime
               real-futures edge' (12y CFD honest figure stays 0.37 — cited C4).
  kill otherwise.

Run: cd ortho && /usr/bin/python3 w4_brnwti_futures.py
Figure: ../curves/w4_brnwti_futures.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab
import w3_brnwti_honest as w3

C3_PATH = (Path.home() / 'Desktop' / 'internship-final-reports' / 'data' / 'raw'
           / 'cl_brn_spread_60.csv')
TICKS_RT = 2 * w3.COST_TURN / 0.01          # round-trip cost in $0.01 ticks = 6


# ------------------------------------------------------------------ builders

def _mad_censor_series(cvals, newday, win, minp):
    dc = np.r_[0.0, np.diff(cvals)]
    mad = pd.Series(dc).rolling(win, min_periods=minp).apply(
        lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
    return (np.abs(dc) > 8 * 1.4826 * np.where(mad > 0, mad, np.nan)) & newday


def build_c2():
    """Yahoo BZ - CL legs, hour-floored, Sundays dropped, censor union."""
    legs = []
    for sym in ('BZ', 'CL'):
        d = lab.yahoo(sym)
        d.index = d.index.floor('h')
        d = d[~d.index.duplicated()]
        d = d[d.index.dayofweek != 6]
        legs.append(d)
    A, B = w3.join(*legs)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    cens = (A['cens'].astype(bool) | B['cens'].astype(bool))
    dSsig = F['c'].diff().where(~cens, 0.0)
    return F, cens, dSsig, w3.bars_day(F.index)


def build_c3():
    """Pre-built TV CL-BRN spread, FLIPPED to BRN-WTI orientation (pre-registered;
    confirmed by lag-0 diff anti-correlation vs C1). o/c only; H/L untrusted."""
    d = pd.read_csv(C3_PATH)
    t = pd.to_datetime(d['time'], utc=True, format='ISO8601')
    d = d.set_index(t)[['open', 'close']]
    d.columns = ['o', 'c']
    d = d[~d.index.duplicated()].sort_index()
    d = d[d.index.dayofweek != 6]
    F = pd.DataFrame({'o': -d['o'], 'c': -d['c']}, index=d.index)      # flip sign
    day = F.index.tz_convert('America/New_York').date
    newday = np.r_[False, day[1:] != day[:-1]]
    cens = pd.Series(_mad_censor_series(F['c'].values, newday, 504, 100),
                     index=F.index)
    dSsig = F['c'].diff().where(~cens, 0.0)
    return F, cens, dSsig, w3.bars_day(F.index)


def build_15m():
    """15-min BRN1! minus CFI_WTI, same construction as C1, ~1y, corroboration."""
    out = []
    for f in ('ICEEUR_DLY_BRN1!, 15 (1).csv', 'CFI_WTI, 15.csv'):
        X = lab.tv(f)
        X.index = X.index.floor('15min')
        X = X[~X.index.duplicated()]
        X = X[X.index.dayofweek != 6]
        out.append(X)
    A, B = w3.join(*out)
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=A.index)
    day = F.index.tz_convert('America/New_York').date
    newday = np.r_[False, day[1:] != day[:-1]]
    cens = np.zeros(len(F), bool)
    for L in (A, B):
        cens |= _mad_censor_series(L['c'].values, newday, 2016, 400)
    cens = pd.Series(cens, index=F.index)
    dSsig = F['c'].diff().where(~cens, 0.0)
    return F, cens, dSsig, w3.bars_day(F.index)


# ------------------------------------------------------------------ batteries

def run_constr(F, cens, dSsig, bd):
    dec = w3.decisions_for(dSsig, bd)
    x1, ntr, subs = w3.ensemble_fixed(F, dec, w3.COST_TURN, 'open', 1.0, cens)
    x2, _, _ = w3.ensemble_fixed(F, dec, w3.COST_TURN, 'open', 2.0, cens)
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    return dict(F=F, cens=cens, dSsig=dSsig, bd=bd, dec=dec, x1=x1, x2=x2,
                ntr=ntr, subs=subs, yrs=yrs, m1=w3.met(x1), m2=w3.met(x2))


def trade_stats(F, dec, yrs):
    """episodes, gross open->open ticks per episode, longest flat stretch (days)."""
    O = F['o'].values
    gross = []
    inm = pd.Series(False, index=F.index)
    nepi = 0
    for d in dec:
        eps = w3.episodes(d['u'])
        nepi += len(eps)
        for (f, e, s, q) in eps:
            if e > f:
                gross.append(s * (O[e] - O[f]))
        inm |= (d['u'].shift(1).fillna(0.0) != 0)
    g = np.asarray(gross) * 100.0                                # ticks of $0.01
    t = F.index[inm.values]
    if len(t) > 1:
        flat = float(np.max((t[1:] - t[:-1]).total_seconds()) / 86400)
    else:
        flat = np.nan
    return dict(nepi=nepi, epi_yr=nepi / yrs, epi_yr_sub=nepi / yrs / len(dec),
                gross_ticks_mean=float(np.mean(g)) if len(g) else np.nan,
                gross_ticks_med=float(np.median(g)) if len(g) else np.nan,
                flat_days=flat, in_mkt_frac=float(inm.mean()))


def underwater_days(pnl):
    eq = w3.trim(pnl).cumsum()
    at_high = eq >= (eq.cummax() - 1e-12)
    t = eq.index[at_high.values]
    if len(t) < 2:
        return np.nan
    gaps = (t[1:] - t[:-1]).total_seconds() / 86400
    tail = (eq.index[-1] - t[-1]).total_seconds() / 86400
    return float(max(gaps.max(), tail))


def pool_pair(pa, pb, sa, sb):
    a, b = w3.trim(pa) / sa, w3.trim(pb) / sb
    idx = a.index.union(b.index)
    return (a.reindex(idx, fill_value=0.0) + b.reindex(idx, fill_value=0.0))


def fmt(m):
    return ('Sh=%5.2f H1=%5.2f H2=%5.2f eqR2=%4.2f G/P=%5.2f maxDD=%6.1f '
            'tot=%6.1f' % (m['sharpe'], m['h1'], m['h2'], m['eqR2'],
                           m['gain_pain'], m['maxdd_vu'], m['total_vu']))


# ================================================================== main

def main():
    print(__doc__)

    # ---------------- constructions ----------------
    print('================ CONSTRUCTIONS ================')
    F1, cens1, dS1, bd1 = w3.tv_spread()
    C1 = run_constr(F1, cens1, dS1, bd1)
    print('C1 TV BRN1!-CFI_WTI : bars=%d %s -> %s (%.2fy, %.1f bars/day) '
          'censored=%d (%.2f%%)' % (len(F1), F1.index[0].date(),
          F1.index[-1].date(), C1['yrs'], bd1, int(cens1.sum()),
          100 * cens1.mean()))
    print('  HONESTY FLAG: CFI_WTI leg is a CFD, not an exchange futures quote.')

    F2, cens2, dS2, bd2 = build_c2()
    C2 = run_constr(F2, cens2, dS2, bd2)
    print('C2 Yahoo BZ-CL      : bars=%d %s -> %s (%.2fy, %.1f bars/day) '
          'censored=%d (%.2f%%)' % (len(F2), F2.index[0].date(),
          F2.index[-1].date(), C2['yrs'], bd2, int(cens2.sum()),
          100 * cens2.mean()))

    F3, cens3, dS3, bd3 = build_c3()
    C3 = run_constr(F3, cens3, dS3, bd3)
    print('C3 TV pre-built (FLIPPED CL-BRN; corroboration ONLY, manifest: '
          'CONTAMINATED leg-stripped precompute): bars=%d %s -> %s (%.2fy, '
          '%.1f bars/day) censored=%d' % (len(F3), F3.index[0].date(),
          F3.index[-1].date(), C3['yrs'], bd3, int(cens3.sum())))
    print('C4 (context, cited not re-run): 12.3y duka CFD honest = 0.37 '
          '(x2 0.17), W3-B.')

    # ---------------- battery A: per-construction honest backtests ------------
    print('\n================ A. PER-CONSTRUCTION HONEST BACKTESTS '
          '(frozen rule, next-open, ticks) ================')
    for nm, C in (('C1', C1), ('C2', C2), ('C3*', C3)):
        print('%-3s x1: %s' % (nm, fmt(C['m1'])))
        print('    x2: %s' % fmt(C['m2']))
        print('    thirds=%s  entry-events=%d (%.0f/yr; %.0f/yr per sub)'
              % (w3.thirds(C['x1']), C['ntr'], C['ntr'] / C['yrs'],
                 C['ntr'] / C['yrs'] / 8))
        print('    yearly Sharpe: %s'
              % lab.yearly(w3.trim(C['x1'])).to_dict())
        print('    per-sub x1 Sharpes (zd x ze): %s'
              % [round(s, 2) for s in C['subs']])
    print('(* C3 is corroboration only — never a judgment leg.)')

    # ---------------- battery B: trade-level honesty stats --------------------
    print('\n================ B. TRADE-LEVEL HONESTY STATS ================')
    tstats = {}
    for nm, C in (('C1', C1), ('C2', C2)):
        ts = trade_stats(C['F'], C['dec'], C['yrs'])
        ts['uw_days'] = underwater_days(C['x1'])
        tstats[nm] = ts
        print('%-3s episodes=%d (%.0f/yr all subs, %.1f/yr/sub) | gross edge '
              'mean=%.1f ticks med=%.1f ticks vs ROUND-TRIP COST %.0f ticks | '
              'in-market %.0f%% | longest flat %.1f d | longest underwater %.0f d'
              % (nm, ts['nepi'], ts['epi_yr'], ts['epi_yr_sub'],
                 ts['gross_ticks_mean'], ts['gross_ticks_med'], TICKS_RT,
                 100 * ts['in_mkt_frac'], ts['flat_days'], ts['uw_days']))

    # censoring impact on C1 (BRN1! roll gaps)
    x1_nc, _, _ = w3.ensemble_fixed(F1, C1['dec'], w3.COST_TURN, 'open', 1.0, None)
    m_nc = w3.met(x1_nc)
    held_cens = int(sum(((d['u'].shift(1).fillna(0.0) != 0) & cens1).sum()
                        for d in C1['dec']))
    print('C1 roll-censor impact: censored bars=%d, censored-while-held '
          '(sub-bar count)=%d | Sharpe censored=%.2f vs UNcensored=%.2f '
          '(delta %.2f)' % (int(cens1.sum()), held_cens, C1['m1']['sharpe'],
                            m_nc['sharpe'], C1['m1']['sharpe'] - m_nc['sharpe']))

    # ---------------- battery C: pooled C1+C2 ----------------
    print('\n================ C. POOLED C1+C2 (equal-risk, union index) '
          '================')
    s1, s2 = w3.trim(C1['x1']).std(), w3.trim(C2['x1']).std()
    pool1 = pool_pair(C1['x1'], C2['x1'], s1, s2)
    pool2 = pool_pair(C1['x2'], C2['x2'], s1, s2)
    mp1, mp2 = w3.met(pool1), w3.met(pool2)
    print('pooled x1: %s' % fmt(mp1))
    print('pooled x2: %s' % fmt(mp2))
    print('pooled yearly Sharpe: %s' % lab.yearly(w3.trim(pool1)).to_dict())
    print('pooled longest underwater: %.0f d' % underwater_days(pool1))
    # overlap caveat: C1 and C2 trade the same economics on the shared 2y window
    ji = w3.trim(C1['x1']).resample('1D').sum().index.intersection(
        w3.trim(C2['x1']).resample('1D').sum().index)
    dcor = np.corrcoef(C1['x1'].resample('1D').sum().reindex(ji, fill_value=0),
                       C2['x1'].resample('1D').sum().reindex(ji, fill_value=0))[0, 1]
    print('C1-C2 daily pnl corr on overlap (same trade, different feeds — pooling '
          'is feed-averaging NOT diversification): %.2f' % dcor)

    # ---------------- battery D: param neighborhood on C1 ----------------
    print('\n================ D. PARAM NEIGHBORHOOD (C1, honest x1, reporting '
          'only) ================')
    variants = [('exit 0.50', dict(zx=0.50)), ('exit 1.00', dict(zx=1.00)),
                ('detrend 21td', dict(detrend_d=21)),
                ('detrend 35td', dict(detrend_d=35)),
                ('volgate OFF', dict(volgate=False))]
    var_out = {}
    for nm, kw in variants:
        dv = w3.decisions_for(dS1, bd1, **kw)
        ev, _, _ = w3.ensemble_fixed(F1, dv, w3.COST_TURN, 'open', 1.0, cens1)
        var_out[nm] = w3.met(ev)['sharpe']
        print('  %-14s ensemble Sharpe = %.2f' % (nm, var_out[nm]))
    print('  %-14s ensemble Sharpe = %.2f  (baseline)'
          % ('frozen', C1['m1']['sharpe']))

    # ---------------- battery E: event study + shuffle on C1 ----------------
    print('\n================ E. SIGNED ENTRY EVENT STUDY (C1, %d-draw placebo) '
          '================' % w3.N_PLACEBO)
    d18 = C1['dec'][w3.ZDS.index(18) * len(w3.ZES)]
    ev_rows = w3.event_study(F1, C1['dec'], d18)
    print('%-5s %6s %9s %7s %7s' % ('h', 'n_ev', 'mean_vu', 't', 'p_plc'))
    for h, nev, mm, tt, pp in ev_rows:
        print('%-5d %6d %9s %7s %7s' % (h, nev, mm, tt, pp))

    print('\n================ F. SHUFFLE TEST (C1, %d relocated episodes, random '
          'sides) ================' % w3.N_SHUFFLE)
    real_tot, tots, n_epi = w3.shuffle_test(F1, C1['dec'])
    p_shuf = float((tots >= real_tot).mean())
    print('episodes=%d  real total=%.1f | shuffle mean=%.1f sd=%.1f | '
          'P(shuffle>=real)=%.3f' % (n_epi, real_tot, tots.mean(), tots.std(),
                                     p_shuf))

    # ---------------- battery G: C1 vs C3 overlap ----------------
    print('\n================ G. OVERLAP C1 vs C3 (same period — do the feeds '
          'agree?) ================')
    s1c = (F1['c']).rename('c1')
    s3c = (F3['c']).rename('c3')
    ji = s1c.index.intersection(s3c.index)
    dd = pd.concat([s1c.loc[ji].diff(), s3c.loc[ji].diff()], axis=1).dropna()
    dcorr = float(np.corrcoef(dd['c1'], dd['c3'])[0, 1])
    print('shared spread bars=%d | hourly diff corr (C1 vs flipped C3) = %.3f'
          % (len(ji), dcorr))
    for yr in sorted(set(ji.year)):
        m = dd.index.year == yr
        if m.sum() > 200:
            print('  %d: diff corr %.3f (n=%d)' % (yr,
                  np.corrcoef(dd.loc[m, 'c1'], dd.loc[m, 'c3'])[0, 1],
                  int(m.sum())))
    lo = max(F1.index[0], F3.index[0])
    hi = min(F1.index[-1], F3.index[-1])
    m1_sh = lab.metrics(w3.trim(C1['x1']).loc[lo:hi])
    m3_sh = lab.metrics(w3.trim(C3['x1']).loc[lo:hi])
    print('shared window %s -> %s' % (lo.date(), hi.date()))
    print('  C1 x1 on shared window: %s' % fmt(m1_sh))
    print('  C3 x1 on shared window: %s' % fmt(m3_sh))
    days = pd.date_range(lo.normalize(), hi.normalize(), tz='UTC')
    p13 = np.corrcoef(
        C1['x1'].resample('1D').sum().reindex(days, fill_value=0),
        C3['x1'].resample('1D').sum().reindex(days, fill_value=0))[0, 1]
    print('  C1-C3 strategy daily pnl corr: %.2f' % p13)

    # ---------------- battery H: 15-min corroboration ----------------
    print('\n================ H. 15-MIN CORROBORATION (BRN1! - CFI_WTI, ~1y) '
          '================')
    try:
        Fm, censm, dSm, bdm = build_15m()
        Cm = run_constr(Fm, censm, dSm, bdm)
        print('bars=%d %s -> %s (%.2fy, %.0f bars/day) censored=%d'
              % (len(Fm), Fm.index[0].date(), Fm.index[-1].date(), Cm['yrs'],
                 bdm, int(censm.sum())))
        print('15m x1: %s' % fmt(Cm['m1']))
        print('15m x2: %s' % fmt(Cm['m2']))
    except FileNotFoundError as e:
        Cm = None
        print('15-min files missing (%s) — skipped, documented.' % e)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.4, 1.0], hspace=0.45,
                          wspace=0.25)
    panels = [('C1: TV BRN1!-CFI_WTI 60m (%.1fy)\n[WTI leg is a CFD]' % C1['yrs'],
               C1, fig.add_subplot(gs[0, 0])),
              ('C2: Yahoo BZ-CL legs (%.1fy)' % C2['yrs'], C2,
               fig.add_subplot(gs[0, 1])),
              ('C3: pre-built TV spread, flipped (corroboration ONLY)', C3,
               fig.add_subplot(gs[1, 0]))]
    for ttl, C, ax in panels:
        e1 = w3.trim(C['x1']).cumsum()
        e2 = w3.trim(C['x2']).cumsum()
        ax.plot(e1.index, e1.values, lw=1.0, color='navy',
                label='x1 Sh %.2f (H1 %.2f/H2 %.2f)' % (C['m1']['sharpe'],
                      C['m1']['h1'], C['m1']['h2']))
        ax.plot(e2.index, e2.values, lw=0.8, color='firebrick', alpha=0.7,
                label='x2 costs Sh %.2f' % C['m2']['sharpe'])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
    axm = fig.add_subplot(gs[1, 1])
    if Cm is not None:
        em = w3.trim(Cm['x1']).cumsum()
        axm.plot(em.index, em.values, lw=0.9, color='darkgreen',
                 label='15-min x1 Sh %.2f' % Cm['m1']['sharpe'])
        axm.legend(fontsize=7, loc='upper left')
    axm.axhline(0, color='gray', lw=0.5, ls=':')
    axm.set_title('15-min corroboration BRN1!-CFI_WTI (~1y)', fontsize=9)
    axm.grid(alpha=0.3)
    axp = fig.add_subplot(gs[2, :])
    ep = w3.trim(pool1).cumsum()
    axp.plot(ep.index, ep.values, lw=1.1, color='black',
             label='POOLED C1+C2 equal-risk x1: Sh %.2f H1 %.2f H2 %.2f | x2 Sh '
                   '%.2f' % (mp1['sharpe'], mp1['h1'], mp1['h2'], mp2['sharpe']))
    ddw = ep - ep.cummax()
    axp2 = axp.twinx()
    axp2.fill_between(ddw.index, ddw.values, 0, color='firebrick', alpha=0.35)
    axp2.set_ylabel('drawdown', fontsize=8)
    axp.legend(fontsize=8, loc='upper left')
    axp.grid(alpha=0.3)
    axp.set_title('pooled equity + drawdown', fontsize=9)
    fig.suptitle('W4-A honest BRN-WTI on real futures feeds — FROZEN Program-1 '
                 'rule, NEXT-OPEN fills, 1.5 ticks/leg/turn\n(12.3y CFD honest '
                 'context: 0.37)', fontsize=10)
    out = lab.ROOT / 'curves' / 'w4_brnwti_futures.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out)

    # ---------------- decision ----------------
    print('\n================ PRE-REGISTERED DECISION RULE (mechanical) '
          '================')
    c3_contra = (C1['m1']['sharpe'] >= 0.6) and (m3_sh['sharpe'] <= 0)
    crit = [
        ('C1 x1 Sharpe >= 0.6', C1['m1']['sharpe'] >= 0.6,
         'C1=%.2f' % C1['m1']['sharpe']),
        ('C2 x1 Sharpe >= 0.6', C2['m1']['sharpe'] >= 0.6,
         'C2=%.2f' % C2['m1']['sharpe']),
        ('pooled halves both > 0', mp1['h1'] > 0 and mp1['h2'] > 0,
         'h1=%.2f h2=%.2f' % (mp1['h1'], mp1['h2'])),
        ('pooled x2 Sharpe >= 0.3', mp2['sharpe'] >= 0.3,
         'x2=%.2f' % mp2['sharpe']),
        ('C3 does not contradict C1 on shared window', not c3_contra,
         'C3 shared=%.2f vs C1 shared=%.2f, diffcorr=%.2f'
         % (m3_sh['sharpe'], m1_sh['sharpe'], dcorr)),
    ]
    all_pass = True
    for name, ok, det in crit:
        all_pass &= bool(ok)
        print('  [%s] %s  (%s)' % ('PASS' if ok else 'FAIL', name, det))
    decision = 'PROMOTE' if all_pass else 'KILL'
    print('\nDECISION (mechanical): %s' % decision)
    if all_pass:
        print('claim limited to: MODEST RECENT-REGIME REAL-FUTURES EDGE '
              '(12y CFD honest figure remains 0.37 — no contradiction claimed).')

    # ---------------- machine-readable ----------------
    print('\n===== SUMMARY (machine-readable) =====')
    print('C1_X1=%.2f C1_H1=%.2f C1_H2=%.2f C1_X2=%.2f C1_EQR2=%.2f C1_GP=%.2f'
          % (C1['m1']['sharpe'], C1['m1']['h1'], C1['m1']['h2'],
             C1['m2']['sharpe'], C1['m1']['eqR2'], C1['m1']['gain_pain']))
    print('C2_X1=%.2f C2_H1=%.2f C2_H2=%.2f C2_X2=%.2f C2_EQR2=%.2f C2_GP=%.2f'
          % (C2['m1']['sharpe'], C2['m1']['h1'], C2['m1']['h2'],
             C2['m2']['sharpe'], C2['m1']['eqR2'], C2['m1']['gain_pain']))
    print('C3_X1=%.2f C3_X2=%.2f C3_SHARED=%.2f C1_SHARED=%.2f DIFFCORR=%.3f '
          'PNLCORR13=%.2f' % (C3['m1']['sharpe'], C3['m2']['sharpe'],
             m3_sh['sharpe'], m1_sh['sharpe'], dcorr, p13))
    print('POOL_X1=%.2f POOL_H1=%.2f POOL_H2=%.2f POOL_X2=%.2f'
          % (mp1['sharpe'], mp1['h1'], mp1['h2'], mp2['sharpe']))
    print('C1_NOCENS=%.2f CENS_BARS=%d HELD_CENS=%d'
          % (m_nc['sharpe'], int(cens1.sum()), held_cens))
    print('TRADESTATS=%s' % {k: {kk: (round(vv, 2) if isinstance(vv, float)
                                      else vv) for kk, vv in v.items()}
                             for k, v in tstats.items()})
    print('VARIANTS=%s' % var_out)
    print('EVENTSTUDY=%s' % ev_rows)
    print('P_SHUFFLE=%.3f' % p_shuf)
    if Cm is not None:
        print('M15_X1=%.2f M15_X2=%.2f' % (Cm['m1']['sharpe'],
                                           Cm['m2']['sharpe']))
    print('C12_POOLCORR=%.2f' % dcor)
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
