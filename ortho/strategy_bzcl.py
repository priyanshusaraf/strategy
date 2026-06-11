#!/usr/bin/env /usr/bin/python3
"""
strategy_bzcl.py — NYMEX BZ-CL HOURLY SPREAD MEAN REVERSION, DELAY-1 EXECUTION
==============================================================================
The PROMOTED BOUNDED CANDIDATE of the 2026 research program (README waves 1-6).
This is the production-readable, standalone implementation: every line of
strategy logic lives in this file. The research repo is used only for data
loading (lab.yahoo) and for the runtime fidelity assertion against the
adjudicated engine (ortho/w6_final.py, construction A).

STRATEGY
--------
Instrument: the Brent-WTI crude differential built MANUALLY from two
same-venue NYMEX front-month legs — BZ (Brent financial) minus CL (WTI) —
on hourly bars. Never a pre-built spread feed: cross-venue and pre-computed
spread constructions were proven async-artifact harvesters (W4-B/W5).

Frozen Program-1 rule (every constant frozen BEFORE the validation waves;
no re-tuning anywhere in waves 3-6):
  spread   dS = dBZ - dCL in $/bbl (fixed 1:-1 weights, leg $-changes;
           roll-gap bars censored: contribute 0 to signal and pnl)
  detrend  S = cumsum(dS - rollmean(dS, 28 td))          [causal, shift(1)]
  signal   z = (S - mean(S, n)) / std(S, n)              [causal, shift(1)]
  ensemble n in {9,12,18,24} td x entry {2.5,3.0} -> 8 subs, equal weight
  exit     |z| <= 0.75; stops |z| >= 4.0 and 10-td time stop
  vol gate enter only when 5d spread vol (ewm) > its rolling 6m median
  sizing   per-trade risk-frozen: units = sign / sigma_at_entry (ewm 63td
           std of dS), never resized intra-trade
  FILLS    DELAY-1 EXECUTION IS THE STRATEGY DEFINITION: every transition
           (entry, exit, z-stop, time-stop) decided at close[t] fills at
           open[t+2] — one full bar later than "next open". This is not a
           stress test; it is the certification that made the candidate
           admissible: under no-delay fills, deliberately misaligning the
           CL leg by +/-1h SUPERCHARGED the rule to Sharpe 4.48/4.28 (a
           z-MR rule harvests injected noise by construction); under
           delay-1 the same controls collapse to 0.11/0.01 while the
           aligned edge survives (w6_final.py mechanism check).
  costs    1.5 ticks/leg/turn ($0.03 per spread unit per turn; 6 ticks
           round-trip); the x2 stress line (3.0 ticks/leg) always reported
  hygiene  hour-22-UTC bars dropped at ingestion (Globex break, 67-94%
           stale); Sunday bars dropped; joins recomputed after the drop

WHY THE EDGE SHOULD EXIST
-------------------------
BZ and CL price the same barrel economics two basins apart. Residual hedging
and inventory flow between the two crude benchmarks (refinery slates,
freight/storage arbitrage, index roll flow) transiently pushes the
differential away from its short-run equilibrium, and it re-equilibrates
within days — the frozen pipeline's own synthetic-OU characterization shows
it only harvests half-lives <= 5 trading days, exactly this flow's
timescale. The edge is harvested ONLY at entries gated by a spread-vol
dislocation (5d vol above its 6m median): in calm tape the differential is
efficient to within transaction costs, and the gate keeps the book flat
there by construction.

BOUNDED-CLAIM CAVEATS (must travel with every artifact derived from this file)
------------------------------------------------------------------------------
* BOUNDED CANDIDATE, NOT A PROVEN EDGE. The sample is 2.39 years on a SINGLE
  construction (Yahoo BZ-CL legs). Pre-registered scope statement: any
  positive claim is bounded to "candidate requiring extended validation" —
  never "proven robust edge".
* Validated numbers (w6_final construction A): Sharpe 1.19 net at x1 costs /
  0.97 at x2, halves 1.10/1.28, eqR2 0.83, gain/pain 3.70, maxDD -26.8 vol
  units, 189 trades/yr across the 8 subs, yearly Sharpe 1.84 / 0.56 / 1.61.
* ADVERSE TEXTURE: the MEDIAN trade makes 7.0 gross ticks against a 6-tick
  round-trip cost — the median trade barely clears costs; the mean is 49.1
  ticks, i.e. the pnl is RIGHT-TAIL DEPENDENT. Expect long bleed punctuated
  by dislocation harvests: the longest underwater stretch is 182 days of a
  2.39y sample.
* REGIME RISK: the same rule family run honestly over 12 years of (CFD)
  history scores 0.37. 2024-26 is one regime with large energy dislocations;
  there is no evidence the edge survives a calm decade.
* RETRACTION HISTORY: this program previously retracted its own headline
  results (12y Sharpe 1.93; "10 smooth securities") after forensics found
  stale-placeholder-bar fabrication and same-close-fill alpha — see
  README.md, "RETRACTION of Program 1 headline results". This candidate is
  what survived the kill-biased rebuild; treat it with the same suspicion.

LIVE-TRADING NOTES
------------------
* FILL REALISM: the open[t+2] convention means a signal at close[t] gives a
  full hour to work the two-leg package and be filled by the following
  bar's open. Do NOT "improve" on this by assuming faster fills: the
  no-delay variant of this pipeline is exactly the engine that failed
  artifact certification.
* BZ LIQUIDITY: NYMEX BZ (Brent financial) is far thinner than CL,
  especially outside US/European hours. The W5 battery shows the edge does
  NOT live in thin hours (liquid-hours-only entries retain 86% of Sharpe;
  the thin tercile carries 10.4% of pnl), so restricting execution to
  liquid hours is safe. Size against BZ top-of-book depth, not CL's.
* CAPACITY: with a 7-tick median trade vs a 6-tick assumed cost there is
  essentially NO cost headroom at the median — execution quality is the
  whole game and capacity is small (a few contracts per 1/sigma unit before
  impact eats the median trade). The x2-cost line (0.97) is the realistic
  expectation for non-professional execution.
* UNITS: positions are in 1/sigma spread units (sigma = ewm 63td std of
  hourly dS, $/bbl); pnl is in those vol units (vu). 1 spread unit = long 1
  BZ + short 1 CL (1000 bbl each). Scale to contracts via your risk budget.

FIDELITY
--------
Running this file re-derives everything from raw data and ASSERTS:
  (a) bit-identical reproduction of w6_final construction A through the
      research machinery (frame, censor mask, all 8 decision series, x1/x2
      pnl series), and
  (b) the published headline numbers: x1 Sharpe == 1.19, x2 Sharpe == 0.97
      (2 decimals — the values promoted in README wave 6).

Run: cd ortho && /usr/bin/python3 strategy_bzcl.py
Figure: ../curves/strategy_bzcl.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab                      # data loading ONLY; all logic is in this file

# ============================================================== CONFIG (FROZEN)
LEG_LONG        = 'BZ'          # NYMEX Brent financial, front month (hourly)
LEG_SHORT       = 'CL'          # NYMEX WTI, front month (hourly)
DETREND_TD      = 28            # detrend window for spread $-changes (td)
Z_WINDOWS_TD    = (9, 12, 18, 24)       # z lookbacks (td)
Z_ENTRIES       = (2.5, 3.0)            # entry thresholds -> 8 ensemble subs
Z_EXIT          = 0.75          # take-profit: |z| <= 0.75
Z_STOP          = 4.0           # disaster stop: |z| >= 4.0
TIME_STOP_TD    = 10            # time stop (td)
VOLGATE_FAST_TD = 5             # gate: 5d ewm vol of dS ...
VOLGATE_SLOW_TD = 126           # ... must exceed its rolling 6m median
SIZING_VOL_TD   = 63            # 1/sigma sizing, ewm 63td std of dS
DELAY_BARS      = 1             # ALL fills at open[t+1+DELAY] = open[t+2]
TICK            = 0.01          # $/bbl
SLIP_TICKS_LEG  = 1.5           # ticks per leg per turn
COST_TURN       = SLIP_TICKS_LEG * 2 * TICK   # $0.03/spread-unit/turn
TICKS_RT        = 2 * COST_TURN / TICK        # 6-tick round trip
DROP_HOUR_UTC   = 22            # Globex-break bars (67-94% stale): dropped
PUBLISHED = dict(x1=1.19, x2=0.97, h1=1.10, h2=1.28, eqR2=0.83, gp=3.70,
                 maxdd=-26.8, trades_yr=189, ticks_med=7.0, ticks_mean=49.1,
                 uw_days=182)   # w6_final.py construction-A record


# ==================================================================== DATA
def load_leg(sym):
    """One Yahoo hourly front-month leg with the binding ingestion hygiene:
    hour-floored stamps, hour-22-UTC bars dropped, duplicates dropped,
    Sundays dropped. Roll-gap flags ('cens') come from lab.yahoo
    (day-boundary |dC| > 8 x 1.4826 x MAD), computed at ingestion before
    the drops."""
    d = lab.yahoo(sym)
    d.index = d.index.floor('h')
    d = d[d.index.hour != DROP_HOUR_UTC]
    d = d[~d.index.duplicated()]
    d = d[d.index.dayofweek != 6]
    return d


def build_spread():
    """Inner-join the legs; build the spread from leg prices with fixed 1:-1
    economic weights. A bar is censored when EITHER leg flags a roll gap:
    censored bars contribute 0 to the signal (dSsig) AND 0 to pnl, and a
    re-roll cost turn is charged if a position is held across them."""
    A, B = load_leg(LEG_LONG), load_leg(LEG_SHORT)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    F = pd.DataFrame({'o': A['o'] - B['o'], 'c': A['c'] - B['c']}, index=idx)
    cens = (A['cens'].astype(bool) | B['cens'].astype(bool))
    dSsig = F['c'].diff().where(~cens, 0.0)
    bd = len(idx) / max(1, len(np.unique(idx.date)))    # actual bars/day
    return F, cens, dSsig, bd


# ==================================================================== SIGNAL
def decide(dSsig, bd, z_days, z_entry):
    """One ensemble sub: the frozen decision state machine. Returns the
    DECIDED position series u[t] (sign x 1/sigma, frozen through the trade),
    indexed at the DECISION bar t. Fills are applied later by the execution
    engine — decisions and fills are deliberately separated.

    Trading-day windows convert to bars via the data's own bars/day, with
    the frozen floors (z >= 80 bars, detrend >= 150 bars) — verbatim the
    Program-1 rule as re-validated in w3_brnwti_honest.decide."""
    dS = dSsig
    z_n = max(80, int(z_days * bd))
    dt = max(150, int(DETREND_TD * bd))
    ts = int(TIME_STOP_TD * bd)
    # causal detrend of spread $-changes, then causal z of the level
    mu = dS.shift(1).rolling(dt, min_periods=dt // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    n = len(dS)
    # vol gate: enter only in a spread-vol dislocation regime
    sf = dS.ewm(span=int(VOLGATE_FAST_TD * bd), min_periods=50).std()
    ss = sf.rolling(int(VOLGATE_SLOW_TD * bd), min_periods=800).median()
    hot = (sf > ss).values
    # per-trade risk-frozen sizing
    sig = dS.ewm(span=int(SIZING_VOL_TD * bd), min_periods=200).std()
    ssafe = sig.replace(0, np.nan).ffill().bfill().values
    u = np.zeros(n)
    cur = 0; held = 0; q = 0.0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:                                    # exits first
            held += 1
            if (zz != zz or abs(zz) >= Z_STOP or held >= ts
                    or (cur > 0 and zz >= -Z_EXIT) or (cur < 0 and zz <= Z_EXIT)):
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:            # then entries
            if zz <= -z_entry:
                cur, held, q = 1, 0, 1.0 / ssafe[i]
            elif zz >= z_entry:
                cur, held, q = -1, 0, 1.0 / ssafe[i]
        if cur != 0:
            u[i] = cur * q
    return pd.Series(u, index=dS.index)


def positions(dSsig, bd):
    """The 8 frozen ensemble subs, equal weight."""
    return [decide(dSsig, bd, zd, ze) for zd in Z_WINDOWS_TD for ze in Z_ENTRIES]


# ================================================================ EXECUTION
def spread_pnl(F, u, cens, stress=1.0, delay=DELAY_BARS):
    """Delay-1 execution engine. The position decided at close[t] becomes
    effective at open[t+1+delay] (= open[t+2] for the frozen delay=1): the
    whole decided path is shifted, so EVERY transition — entries, exits,
    z-stops, time-stops — fills one bar later than next-open. Bar pnl is
    decomposed into intraday (held position x open->close) and overnight
    (previous position x close->open) so gaps accrue to the position that
    actually held them. Censored (roll) bars contribute 0 pnl and charge a
    re-roll turn while held. Costs: COST_TURN per unit turned, x stress."""
    ud = u.shift(delay).fillna(0.0) if delay else u
    C, O = F['c'], F['o']
    ua = ud.shift(1)                                    # fill at next open
    du = ua.diff().abs().fillna(0.0)
    day = (ua * (C - O)).where(~cens, 0.0)
    on = (ua.shift(1) * (O - C.shift(1))).where(~cens, 0.0)
    roll = (ua.abs() * cens).fillna(0.0)
    pnl = day.fillna(0.0) + on.fillna(0.0) - stress * COST_TURN * (du + roll)
    return pnl.fillna(0.0)


def ensemble_pnl(F, us, cens, stress=1.0):
    """Equal-weight mean of the 8 subs' delayed-execution pnl (vol units)."""
    parts = [spread_pnl(F, u, cens, stress) for u in us]
    return pd.concat(parts, axis=1).mean(axis=1)


def episodes(ud):
    """Contiguous same-sign position episodes of one sub, on the
    fill-aligned series (entry/exit at the bars whose OPENS the engine
    actually fills)."""
    ua = ud.shift(1).fillna(0.0).values
    n = len(ua)
    eps, i = [], 0
    while i < n:
        if ua[i] == 0:
            i += 1
            continue
        s = np.sign(ua[i]); q = abs(ua[i]); f = i
        j = i
        while j + 1 < n and ua[j + 1] != 0 and np.sign(ua[j + 1]) == s:
            j += 1
        eps.append((f, min(j + 1, n - 1), s, q))
        i = j + 1
    return eps


def episode_table(F, us, delay=DELAY_BARS):
    """Per-trade table across all 8 subs: gross open->open ticks and net pnl
    (2 cost turns per episode)."""
    O = F['o'].values
    rows = []
    for u in us:
        ud = u.shift(delay).fillna(0.0) if delay else u
        for (f, e, s, q) in episodes(ud):
            if e <= f:
                continue
            gross = s * (O[e] - O[f])
            rows.append(dict(f=f, e=e, s=s, q=q, ticks=gross / TICK,
                             pnl=q * gross - 2 * COST_TURN * q))
    return pd.DataFrame(rows)


# ================================================================== METRICS
def trim(pnl):
    nz = pnl[pnl != 0]
    return pnl.loc[nz.index[0]:] if len(nz) else pnl


def metrics(pnl):
    idx = pnl.index
    yrs = (idx[-1] - idx[0]).days / 365.25
    bpy = len(pnl) / max(yrs, 1e-9)
    def shp(x):
        s = x.std()
        return float(x.mean() / s * np.sqrt(bpy)) if s > 0 else 0.0
    eq = pnl.cumsum()
    n = len(eq)
    ddmin = float((eq - eq.cummax()).min())
    tot = float(eq.iloc[-1])
    x = np.arange(n, dtype=float)
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, eq.values, rcond=None)
    resid = eq.values - A @ coef
    r2 = float(1 - resid.var() / eq.values.var()) if eq.values.var() > 0 else 0.0
    return dict(sharpe=round(shp(pnl), 2),
                h1=round(shp(pnl.iloc[:n // 2]), 2),
                h2=round(shp(pnl.iloc[n // 2:]), 2),
                eqR2=round(r2, 2),
                gain_pain=round(tot / abs(ddmin), 2) if ddmin < 0 else float('inf'),
                maxdd_vu=round(ddmin, 1), total_vu=round(tot, 1))


def yearly_sharpe(pnl):
    yrs = (pnl.index[-1] - pnl.index[0]).days / 365.25
    bpy = len(pnl) / max(yrs, 1e-9)
    g = pnl.groupby(pnl.index.year)
    return g.apply(lambda x: round(float(x.mean() / x.std() * np.sqrt(bpy)), 2)
                   if x.std() > 0 else 0.0)


def underwater_days(pnl):
    eq = trim(pnl).cumsum()
    at_high = eq >= (eq.cummax() - 1e-12)
    t = eq.index[at_high.values]
    if len(t) < 2:
        return np.nan
    gaps = (t[1:] - t[:-1]).total_seconds() / 86400
    tail = (eq.index[-1] - t[-1]).total_seconds() / 86400
    return float(max(gaps.max(), tail))


# ================================================================= FIDELITY
def fidelity_check(F, cens, dSsig, bd, us, x1, x2, m1, m2):
    """Assert this standalone implementation == the adjudicated research
    engine (w6_final construction A), then assert the published numbers."""
    import w3_brnwti_honest as w3
    import w5_adjudication as w5
    import w6_final as w6

    _, _, Fw, cw, dSw, bdw = w6.build_c2_w6(0, drop_h22=True)
    assert Fw.equals(F) and cw.equals(cens) and dSw.equals(dSsig) and bdw == bd, \
        'FIDELITY FAIL: spread construction differs from w6.build_c2_w6'
    print('FIDELITY 1 PASS: spread frame, censor mask, dSsig and bars/day '
          'bit-identical to w6_final construction A.')

    decw = w3.decisions_for(dSw, bdw)
    for mine, ref in zip(us, decw):
        assert ref['u'].equals(mine), \
            'FIDELITY FAIL: decision series differ from frozen w3.decide'
    print('FIDELITY 2 PASS: all 8 ensemble position series bit-identical to '
          'the frozen research decisions (w3.decide).')

    w1, w2, mw1, mw2 = w5.run_pnl(Fw, decw, cw, shift=DELAY_BARS)
    assert w1.equals(x1) and w2.equals(x2), \
        'FIDELITY FAIL: delayed-execution pnl differs from w5.run_pnl(shift=1)'
    assert (mw1['sharpe'], mw2['sharpe']) == (m1['sharpe'], m2['sharpe'])
    print('FIDELITY 3 PASS: x1/x2 delayed-execution pnl series bit-identical '
          'to w5.run_pnl(shift=1) on the w6 frame.')

    assert m1['sharpe'] == PUBLISHED['x1'] and m2['sharpe'] == PUBLISHED['x2'], \
        ('FIDELITY FAIL: Sharpe %.2f/%.2f != published %.2f/%.2f'
         % (m1['sharpe'], m2['sharpe'], PUBLISHED['x1'], PUBLISHED['x2']))
    print('FIDELITY 4 PASS: reproduces the published construction-A numbers '
          'x1 Sharpe %.2f / x2 Sharpe %.2f (2dp).'
          % (m1['sharpe'], m2['sharpe']))


# =================================================================== FIGURE
RULE_LINE = ('frozen rule: detrend 28td | z ens {9,12,18,24}td x {2.5,3.0} | '
             'exit 0.75 | stops 4.0 / 10td | vol gate 5d>6m med | 1/sigma '
             'risk-frozen | ALL fills open[t+2] | 1.5 ticks/leg/turn')


def figure(x1, x2, m1, m2, uw, path):
    e1, e2 = trim(x1).cumsum(), trim(x2).cumsum()
    yr = yearly_sharpe(trim(x1))
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.0, 1.1], hspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.plot(e1.index, e1.values, lw=1.1, color='navy',
            label='net x1 costs: Sh %.2f (H1 %.2f / H2 %.2f), eqR2 %.2f, '
                  'G/P %.2f' % (m1['sharpe'], m1['h1'], m1['h2'], m1['eqR2'],
                                m1['gain_pain']))
    ax.plot(e2.index, e2.values, lw=0.9, color='firebrick', alpha=0.8,
            label='net x2 costs (3.0 ticks/leg): Sh %.2f' % m2['sharpe'])
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylabel('cum net pnl (vol units)')
    ax.set_title('net equity — median trade 7.0 ticks gross vs 6-tick cost '
                 '(right-tail dependent); %.0fd max underwater' % uw,
                 fontsize=9)

    ax = fig.add_subplot(gs[1])
    dd = e1 - e1.cummax()
    ax.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.6)
    ax.set_ylabel('drawdown (vu)')
    ax.grid(alpha=0.3)
    ax.set_title('drawdown (x1), maxDD %.1f vu' % m1['maxdd_vu'], fontsize=9)

    ax = fig.add_subplot(gs[2])
    cols = ['seagreen' if v > 0 else 'firebrick' for v in yr.values]
    ax.bar(yr.index.astype(str), yr.values, color=cols)
    for i, v in enumerate(yr.values):
        ax.text(i, v + (0.03 if v >= 0 else -0.10), '%.2f' % v,
                ha='center', fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('yearly Sharpe (x1)')
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('yearly Sharpe — one regime, 2.39y: candidate requiring '
                 'extended validation', fontsize=9)

    fig.suptitle('NYMEX BZ-CL hourly spread MR, delay-1 execution — '
                 'BOUNDED CANDIDATE 2.39y\n%s' % RULE_LINE, fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches='tight')
    print('figure saved -> %s' % path)


# ====================================================================== MAIN
def main():
    print(__doc__)
    print('================ CONSTRUCTION ================')
    F, cens, dSsig, bd = build_spread()
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    print('%s-%s joined bars=%d  %s -> %s  (%.2fy, %.1f bars/day)  '
          'roll-censored=%d (%.2f%%)'
          % (LEG_LONG, LEG_SHORT, len(F), F.index[0].date(),
             F.index[-1].date(), yrs, bd, int(cens.sum()), 100 * cens.mean()))
    print('cost: $%.3f per spread unit per turn (%.1f ticks/leg x 2 legs); '
          'round-trip %.0f ticks; stress x2 reported'
          % (COST_TURN, SLIP_TICKS_LEG, TICKS_RT))

    print('\n================ SIGNALS (8 frozen ensemble subs) ================')
    us = positions(dSsig, bd)
    for (zd, ze), u in zip([(zd, ze) for zd in Z_WINDOWS_TD
                            for ze in Z_ENTRIES], us):
        print('  sub zd=%2dtd ze=%.1f : in-market %4.1f%% of bars'
              % (zd, ze, 100 * (u != 0).mean()))

    print('\n================ BACKTEST (all fills at open[t+2]) ================')
    x1 = ensemble_pnl(F, us, cens, 1.0)
    x2 = ensemble_pnl(F, us, cens, 2.0)
    m1, m2 = metrics(trim(x1)), metrics(trim(x2))
    ep = episode_table(F, us)
    yr = yearly_sharpe(trim(x1))
    uw = underwater_days(x1)
    for tag, m in (('x1', m1), ('x2', m2)):
        print('%s: Sh=%5.2f H1=%5.2f H2=%5.2f eqR2=%4.2f G/P=%5.2f '
              'maxDD=%6.1f tot=%6.1f'
              % (tag, m['sharpe'], m['h1'], m['h2'], m['eqR2'],
                 m['gain_pain'], m['maxdd_vu'], m['total_vu']))
    print('trades: %d episodes (%.0f/yr all 8 subs; %.1f/yr/sub)'
          % (len(ep), len(ep) / yrs, len(ep) / yrs / len(us)))
    print('per-trade gross edge: mean=%.1f ticks  MEDIAN=%.1f ticks  vs '
          'round-trip cost %.0f ticks  <- median trade barely clears cost'
          % (ep.ticks.mean(), ep.ticks.median(), TICKS_RT))
    print('yearly Sharpe (x1): %s' % yr.to_dict())
    print('longest underwater: %.0f days' % uw)

    print('\n================ FIDELITY CHECK (vs ortho/w6_final.py '
          'construction A) ================')
    fidelity_check(F, cens, dSsig, bd, us, x1, x2, m1, m2)

    print('\n================ PUBLISHED-RECORD CROSS-CHECK ================')
    got = dict(x1=m1['sharpe'], x2=m2['sharpe'], h1=m1['h1'], h2=m1['h2'],
               eqR2=m1['eqR2'], gp=m1['gain_pain'], maxdd=m1['maxdd_vu'],
               trades_yr=round(len(ep) / yrs),
               ticks_med=round(float(ep.ticks.median()), 1),
               ticks_mean=round(float(ep.ticks.mean()), 1), uw_days=round(uw))
    for k in PUBLISHED:
        flag = 'OK' if abs(got[k] - PUBLISHED[k]) < 1e-9 else 'DIFF'
        print('  %-10s published=%8s  this run=%8s  [%s]'
              % (k, PUBLISHED[k], got[k], flag))

    figure(x1, x2, m1, m2, uw, lab.ROOT / 'curves' / 'strategy_bzcl.png')

    print('\nBINDING SCOPE STATEMENT: 2.39y, single construction (Yahoo '
          'BZ-CL). This is a candidate requiring extended validation, never '
          'a proven robust edge. Median trade ~= cost; right-tail dependent; '
          '182-day underwater stretch in-sample; the same rule family scores '
          '0.37 on 12y of (CFD) history.')


if __name__ == '__main__':
    main()
