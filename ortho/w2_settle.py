#!/usr/bin/env /usr/bin/python3
"""
W2-B — SETTLEMENT-WINDOW FADE (falsification gauntlet; wave-2).

Wave-1 H8 found an hour-specific (placebo p<=0.003) fade block at 17-19 UTC
(London close -> CME settlement), sub-cost at threshold 1.0 ATR. Pre-registered
question: does concentrating on BIGGER moves in that window clear costs?

FROZEN RULE (pre-registered, run exactly; grid is for ROBUSTNESS REPORTING):
  Event   : bar with UTC hour in {17,18,19} AND |C[t]-C[t-1]|/ATR100 >= thr.
  Side    : -sign(move)  (fade). Entry NEXT bar open.
  Horizons: h in {2, 4}.  thr grid {1.5, 2.0} — both reported.
  PRIMARY CELL (declared ex-ante): thr=1.5, h=4.
  De-overlap: adjacent-hour events can overlap at h in {2,4} -> events kept
  first-come with >= h bar gap (and lab.backtest is one-position anyway).

UNIVERSE: 9 duka commodities (CMD) + 34 fx (MAJ = 10 majors, CRX = 24
crosses+exotics); families reported separately (H8: FX majors strongest at
18-19; commodities NEGATIVE at NY hours — family divergence expected).

BATTERY (pre-registered):
  1. Hour-conditioned SIGNED placebo (as in h8): same event count drawn with
     replacement from the instrument's OWN >=thr move bars at ANY hour, same
     fade rule, 300 draws -> p tests hour-specificity beyond the generic
     fade-big-move effect.
  2. Pooled + per-instrument + family tables, both thr, both h
     (+ thr=1.0 context line for continuity with H8 — context only).
  3. Portfolio backtest at PRIMARY cell, x1 and stressed costs
     (x3 exotic FX, x2 everything else), figure -> ../curves/w2_settle.png.
  4. Halves / thirds / yearly.
  5. Cross-feed (UNTOUCHED by wave-1): OANDA_EURUSD/USDJPY, FOREXCOM_USDCHF,
     TVC_DXY 60-min (~3.4y) — same rule; 15-min files resampled to UTC-hour
     bars = exact 4-bars/hour equivalent (same event close, same next-open
     fill time), ~1y, INDICATIVE only.
  6. lab.shuffle_test (100 shuffles) per instrument at primary cell.
  7. Mon-Thu only line (Friday events excluded).

PRE-REGISTERED DECISION RULE (kill-biased): kill unless SOME pre-registered
cell has edge/cost >= 1 AND hour-conditioned placebo p <= 0.05 AND cross-feed
same-sign t >= 1.5; promote additionally requires portfolio Sharpe >= 0.5 x1
/ > 0 stressed with both halves positive.

Run: cd ortho && /usr/bin/python3 w2_settle.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab

HOURS      = (17, 18, 19)
THRS       = (1.5, 2.0)
HORIZONS   = (2, 4)
PRIMARY    = (1.5, 4)           # (thr, h) — declared ex-ante
N_PLACEBO  = 300
MIN_N      = 10
N_SHUFFLE  = 100
FIG        = '../curves/w2_settle.png'

FX_MAJ  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD',
           'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF']
EXOTICS = ['EURPLN', 'EURHUF', 'EURCZK', 'EURNOK', 'EURSEK', 'USDNOK',
           'USDSEK', 'AUDSGD']

SYMS = ([('CMD', s) for s in lab.DUKA_SYMS] +
        [('MAJ' if s in FX_MAJ else 'CRX', s) for s in lab.fx_syms()])
FAMS = ['CMD', 'MAJ', 'CRX']

TV60 = [('EURUSD', 'OANDA_EURUSD, 60 (1).csv', 1.5),
        ('USDJPY', 'OANDA_USDJPY, 60.csv', 1.5),
        ('USDCHF', 'FOREXCOM_USDCHF, 60.csv', 1.5),
        ('DXY',    'TVC_DXY, 60.csv', 2.0)]
TV15 = [('EURUSD', 'OANDA_EURUSD, 15 (2).csv', 1.5),
        ('USDJPY', 'OANDA_USDJPY, 15.csv', 1.5),
        ('USDCHF', 'FOREXCOM_USDCHF, 15.csv', 1.5),
        ('DXY',    'TVC_DXY, 15.csv', 2.0)]


def stress_mult(sym):
    """Cost stress multiplier: exotic FX may be UNDER-costed at 5bps -> x3."""
    return 3.0 if sym in EXOTICS else 2.0


def _deoverlap(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def inst_study(df, a, cb, thr, seed, hours=HOURS, weekdays=None):
    """Signed event study for one instrument at one thr, both horizons.
    Placebo (h8-style): same count drawn with replacement from the
    instrument's own >=thr big-move universe at ANY hour, same fade rule.
    Returns h -> dict(n, f, cv, pm, wd) or None if too few events."""
    c, o = df.c.values, df.o.values
    av = a.values
    mv = np.r_[np.nan, np.diff(c)] / av
    hr = df.index.hour.values
    wd_all = df.index.dayofweek.values
    N = len(df)
    ok = (np.isfinite(mv) & (np.abs(mv) >= thr) & (mv != 0) &
          np.isfinite(av) & (av > 0))
    rng = np.random.default_rng(seed)
    out = {}
    for h in HORIZONS:
        uni = np.flatnonzero(ok)
        uni = uni[uni + h < N - 1]
        f_all = -np.sign(mv[uni]) * (c[uni + h] - o[uni + 1]) / av[uni]
        fin = np.isfinite(f_all)
        uni, f_all = uni[fin], f_all[fin]
        sel = uni[np.isin(hr[uni], hours)]
        if weekdays is not None:
            sel = sel[np.isin(wd_all[sel], weekdays)]
        ev = _deoverlap(sel, h)
        if len(ev) < MIN_N or len(f_all) < MIN_N:
            out[h] = None
            continue
        f = -np.sign(mv[ev]) * (c[ev + h] - o[ev + 1]) / av[ev]
        cv = (cb / 1e4) * np.abs(o[ev + 1]) / av[ev]
        draws = rng.integers(0, len(f_all), size=(N_PLACEBO, len(ev)))
        pm = f_all[draws].mean(axis=1)
        out[h] = dict(n=len(ev), f=f, cv=cv, pm=pm, wd=wd_all[ev])
    return out


def pool(studies, h):
    """Pool one (thr,h) cell across instruments. Placebo p = count-weighted
    per-iteration pooled placebo mean vs |real pooled mean|."""
    fs, pms, ns, cvs = [], [], [], []
    for st in studies:
        d = None if st is None else st[h]
        if d is None:
            continue
        fs.append(d['f'])
        pms.append(d['pm'] * d['n'])
        ns.append(d['n'])
        cvs.append(d['cv'].sum())
    if not fs:
        return dict(n=0, mean=np.nan, t=np.nan, p=np.nan, cost_vu=np.nan,
                    e2c=np.nan, pos=0, tot=0)
    F = np.concatenate(fs)
    m = F.mean()
    t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
    ntot = sum(ns)
    PM = np.vstack(pms).sum(axis=0) / ntot
    p = float((np.abs(PM) >= abs(m)).mean())
    cost_vu = sum(cvs) / ntot
    return dict(n=len(F), mean=m, t=t, p=p, cost_vu=cost_vu,
                e2c=m / cost_vu if cost_vu > 0 else np.nan,
                pos=sum(1 for f in fs if f.mean() > 0), tot=len(fs))


def event_mask(df, a, thr, h, hours=HOURS, weekdays=None):
    """De-overlapped event mask + fade sides for lab.backtest."""
    c = df.c.values
    av = a.values
    mv = np.r_[np.nan, np.diff(c)] / av
    hr = df.index.hour.values
    wd = df.index.dayofweek.values
    raw = (np.isfinite(mv) & (np.abs(mv) >= thr) & (mv != 0) &
           np.isfinite(av) & (av > 0) & np.isin(hr, hours))
    if weekdays is not None:
        raw &= np.isin(wd, weekdays)
    pos = _deoverlap(np.flatnonzero(raw), h)
    mask = np.zeros(len(df), bool)
    mask[pos] = True
    sides = np.zeros(len(df))
    sides[mask] = -np.sign(mv[mask])
    return mask, sides


def portfolio(data, thr, h, stress=False, weekdays=None, exclude=()):
    parts, n_tr = [], 0
    for fam, sym, df, a in data:
        if sym in exclude:
            continue
        mask, sides = event_mask(df, a, thr, h, weekdays=weekdays)
        cb = lab.cost_bps(sym, stress=stress_mult(sym) if stress else 1.0)
        pnl, n = lab.backtest(df, mask, h=h, sides=sides, cost=cb, a=a)
        parts.append(pnl.rename(sym))
        n_tr += n
    P = pd.concat(parts, axis=1).sort_index()
    return P.fillna(0.0).sum(axis=1), n_tr, P


def thirds(port):
    n = len(port)
    return [lab.metrics(port.iloc[i * n // 3:(i + 1) * n // 3])['sharpe']
            for i in range(3)]


def fmt_cell(d):
    if d['n'] == 0:
        return '      -      -     -     -      -    -/-'
    return ('%6d %6.3f %5.2f %5.3f %6.2f %3d/%-3d' %
            (d['n'], d['mean'], d['t'], d['p'], d['e2c'], d['pos'], d['tot']))


def resample_hourly(d):
    """15-min -> UTC-hour bars: exact 4-bars/hour equivalent of the rule
    (hour close = :45-bar close; next hourly open = next 15-min open)."""
    r = d.resample('1h').agg({'o': 'first', 'h': 'max', 'l': 'min',
                              'c': 'last'})
    return r.dropna(subset=['o', 'c'])


def main():
    print(__doc__)

    # ------------------------------------------------ load 43 instruments
    print('loading %d instruments ...' % len(SYMS))
    data = []
    for fam, sym in SYMS:
        df = lab.duka(sym) if fam == 'CMD' else lab.fx(sym)
        data.append((fam, sym, df, lab.atr(df)))
    print('loaded.\n')

    # ------------------------------------------------ event studies, all cells
    # studies[thr] = list aligned with data; thr=1.0 is CONTEXT only.
    studies = {}
    for thr in THRS + (1.0,):
        studies[thr] = [inst_study(df, a, lab.cost_bps(sym),
                                   thr, seed=7000 + i)
                        for i, (fam, sym, df, a) in enumerate(data)]

    print('=========== MAIN GRID: fade |move|>=thr ATR at UTC hours 17-19 ===========')
    print('placebo p: same count from instrument\'s own >=thr move bars at ANY hour,')
    print('same fade rule, 300 draws -> tests HOUR conditioning. e2c at x1 costs.')
    print('%-4s %-2s %-4s| %6s %6s %5s %5s %6s %6s' %
          ('thr', 'h', 'fam', 'n', 'mean', 't', 'p', 'e2c', 'sgn+'))
    grid = {}
    for thr in THRS:
        for h in HORIZONS:
            for famkey in FAMS + ['ALL']:
                lst = [st for (fam, sym, df, a), st in zip(data, studies[thr])
                       if famkey == 'ALL' or fam == famkey]
                d = pool(lst, h)
                grid[(thr, h, famkey)] = d
                tag = ' <-- PRIMARY' if (thr, h) == PRIMARY and famkey == 'ALL' else ''
                print('%-4g %-2d %-4s| %s%s' % (thr, h, famkey, fmt_cell(d), tag))
            print()

    print('--- CONTEXT (continuity with H8, not a decision cell): thr=1.0 ---')
    for h in HORIZONS:
        for famkey in FAMS + ['ALL']:
            lst = [st for (fam, sym, df, a), st in zip(data, studies[1.0])
                   if famkey == 'ALL' or fam == famkey]
            print('%-4g %-2d %-4s| %s' % (1.0, h, famkey, fmt_cell(pool(lst, h))))
        print()

    # ------------------------------------------------ per-instrument table
    pt, ph = PRIMARY
    print('========= PER-INSTRUMENT (all 4 pre-registered cells; p at primary) =========')
    hdr = '%-8s %-4s' % ('sym', 'fam')
    for thr in THRS:
        for h in HORIZONS:
            hdr += ' | %5s %7s %6s' % ('n', 'mean', 't')
    hdr += ' | %5s %5s' % ('p*', 'e2c*')
    print(hdr + '    (*primary cell thr=%g h=%d)' % (pt, ph))
    for i, (fam, sym, df, a) in enumerate(data):
        row = '%-8s %-4s' % (sym, fam)
        for thr in THRS:
            for h in HORIZONS:
                d = studies[thr][i][h] if studies[thr][i] else None
                if d is None:
                    row += ' | %5s %7s %6s' % ('-', '-', '-')
                else:
                    f = d['f']
                    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
                    row += ' | %5d %7.4f %6.2f' % (d['n'], f.mean(), t)
        d = studies[pt][i][ph] if studies[pt][i] else None
        if d is None:
            row += ' | %5s %5s' % ('-', '-')
        else:
            p = float((np.abs(d['pm']) >= abs(d['f'].mean())).mean())
            e2c = d['f'].mean() / d['cv'].mean() if d['cv'].mean() > 0 else np.nan
            row += ' | %5.3f %5.2f' % (p, e2c)
        print(row)

    # ------------------------------------------------ Mon-Thu line (battery 7)
    print('\n========= MON-THU ONLY (Friday excluded), pooled, primary cell =========')
    mt_st = [inst_study(df, a, lab.cost_bps(sym), pt, seed=8000 + i,
                        weekdays=(0, 1, 2, 3))
             for i, (fam, sym, df, a) in enumerate(data)]
    for famkey in FAMS + ['ALL']:
        lst = [st for (fam, sym, df, a), st in zip(data, mt_st)
               if famkey == 'ALL' or fam == famkey]
        print('%-4s| %s' % (famkey, fmt_cell(pool(lst, ph))))

    # ------------------------------------------------ portfolio @ primary
    print('\n===== PORTFOLIO BACKTEST @ PRIMARY (thr=%g, h=%d, fade, per-sym costs) =====' %
          (pt, ph))
    port, n_tr, P = portfolio(data, pt, ph, stress=False)
    mt = lab.metrics(port)
    print('x1 costs, 43 instruments, trades=%d' % n_tr)
    print(mt)
    yr = lab.yearly(port)
    print('yearly sharpe:')
    print(yr.to_string())
    print('thirds sharpe: %s' % thirds(port))

    port2, n_tr2, _ = portfolio(data, pt, ph, stress=True)
    mt2 = lab.metrics(port2)
    print('\nstressed costs (x3 exotic FX, x2 all else), trades=%d' % n_tr2)
    print(mt2)

    port_noex, n_noex, _ = portfolio(data, pt, ph, exclude=EXOTICS)
    print('\ndrop exotics (35 instruments, x1 costs), trades=%d' % n_noex)
    print(lab.metrics(port_noex))

    port_mt, n_mt, _ = portfolio(data, pt, ph, weekdays=(0, 1, 2, 3))
    print('\nMon-Thu only portfolio (x1 costs), trades=%d' % n_mt)
    print(lab.metrics(port_mt))

    # family portfolios
    print('\nfamily portfolios @ primary, x1 costs:')
    for famkey in FAMS:
        sub = [d for d in data if d[0] == famkey]
        pf, nf, _ = portfolio(sub, pt, ph)
        print('  %-4s trades=%5d %s' % (famkey, nf, lab.metrics(pf)))

    # avg pairwise correlation of instrument pnl
    act = P.loc[:, P.std() > 0].fillna(0.0)
    C = act.corr().values
    n_inst = C.shape[0]
    avg_corr = (C.sum() - n_inst) / (n_inst * (n_inst - 1))
    print('\navg pairwise corr of %d instrument pnl series: %.4f' %
          (n_inst, avg_corr))

    # episodicity
    dly = port.resample('1D').sum()
    dly = dly[dly != 0]
    tot = port.sum()
    top5 = dly.nlargest(5).sum()
    mo = port.resample('ME').sum()
    mo = mo[mo != 0]
    print('episodicity: total=%.1f vu; top-5 days sum=%.1f (%.0f%% of total); '
          'months positive %d/%d (%.0f%%)' %
          (tot, top5, 100 * top5 / tot if tot != 0 else np.nan,
           (mo > 0).sum(), len(mo), 100 * (mo > 0).mean()))
    ya = port.groupby(port.index.year).sum()
    print('yearly total vu:')
    print(ya.round(1).to_string())

    # ------------------------------------------------ shuffle test (battery 6)
    print('\n========= SHUFFLE TEST (100 random-time random-sign shuffles) =========')
    sh = []
    for fam, sym, df, a in data:
        mask, sides = event_mask(df, a, pt, ph)
        if mask.sum() < MIN_N:
            continue
        rt, pfrac = lab.shuffle_test(df, mask, h=ph, sides=sides,
                                     cost=lab.cost_bps(sym),
                                     n_shuffle=N_SHUFFLE)
        sh.append((sym, fam, rt, pfrac))
    shp_le05 = sum(1 for _, _, _, p in sh if p <= 0.05)
    print('%-8s %-4s %9s %6s' % ('sym', 'fam', 'real_vu', 'p_shuf'))
    for sym, fam, rt, pfrac in sh:
        print('%-8s %-4s %9.1f %6.2f' % (sym, fam, rt, pfrac))
    print('instruments with shuffle p<=0.05: %d/%d (chance ~%.1f)' %
          (shp_le05, len(sh), 0.05 * len(sh)))
    print('median shuffle p: %.2f' % np.median([p for _, _, _, p in sh]))

    # ------------------------------------------------ cross-feed (battery 5)
    print('\n========= CROSS-FEED: TV 60-min (~3.4y), UNTOUCHED by wave-1 =========')
    xf60 = []
    for i, (sym, fname, cb) in enumerate(TV60):
        df = lab.tv(fname)
        a = lab.atr(df)
        st = inst_study(df, a, cb, pt, seed=9000 + i)
        xf60.append(st)
        for h in HORIZONS:
            d = st[h] if st else None
            if d is None:
                print('%-7s h=%d: too few events' % (sym, h))
                continue
            f = d['f']
            t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
            p = float((np.abs(d['pm']) >= abs(f.mean())).mean())
            e2c = f.mean() / d['cv'].mean() if d['cv'].mean() > 0 else np.nan
            print('%-7s h=%d: n=%4d mean=%7.4f t=%5.2f p_plc=%.3f e2c=%5.2f' %
                  (sym, h, d['n'], f.mean(), t, p, e2c))
        # thr=2.0 line for completeness
        st2 = inst_study(df, a, cb, 2.0, seed=9100 + i)
        for h in HORIZONS:
            d = st2[h] if st2 else None
            if d is None:
                print('%-7s h=%d thr=2.0: too few events' % (sym, h))
                continue
            f = d['f']
            t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
            print('%-7s h=%d thr=2.0: n=%4d mean=%7.4f t=%5.2f' %
                  (sym, h, d['n'], f.mean(), t))
    for h in HORIZONS:
        d = pool(xf60, h)
        tag = ' <-- decision line' if h == ph else ''
        print('POOLED TV60 thr=%g h=%d: n=%d mean=%.4f t=%.2f p_plc=%.3f '
              'e2c=%.2f sgn+ %d/%d%s' %
              (pt, h, d['n'], d['mean'], d['t'], d['p'], d['e2c'],
               d['pos'], d['tot'], tag))

    print('\n--- cross-feed 15-min resampled to UTC-hour bars (~1y, INDICATIVE) ---')
    xf15 = []
    for i, (sym, fname, cb) in enumerate(TV15):
        df = resample_hourly(lab.tv(fname))
        a = lab.atr(df)
        st = inst_study(df, a, cb, pt, seed=9500 + i)
        xf15.append(st)
        for h in HORIZONS:
            d = st[h] if st else None
            if d is None:
                print('%-7s h=%d: too few events' % (sym, h))
                continue
            f = d['f']
            t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
            print('%-7s h=%d: n=%4d mean=%7.4f t=%5.2f' %
                  (sym, h, d['n'], f.mean(), t))
    for h in HORIZONS:
        d = pool(xf15, h)
        print('POOLED TV15->hourly thr=%g h=%d: n=%d mean=%.4f t=%.2f sgn+ %d/%d' %
              (pt, h, d['n'], d['mean'], d['t'], d['pos'], d['tot']))

    # ------------------------------------------------ figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={'height_ratios': [3, 1, 1.4]})
    eq = port.cumsum()
    eq2 = port2.cumsum()
    axes[0].plot(eq.index, eq.values, lw=0.9, color='navy', label='x1 costs')
    axes[0].plot(eq2.index, eq2.values, lw=0.9, color='firebrick', alpha=0.8,
                 label='stressed (x3 exotics, x2 rest)')
    axes[0].set_title('W2-B settlement-window fade — hours 17-19 UTC, '
                      '|$\\Delta$C|/ATR$\\geq$%g, fade, h=%d, next-open fills, '
                      '43 instruments' % (pt, ph))
    axes[0].set_ylabel('cumulative PnL (vol units)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    dd = eq - eq.cummax()
    axes[1].fill_between(dd.index, dd.values, 0, color='maroon', alpha=0.6)
    axes[1].set_ylabel('drawdown (vu)')
    axes[1].grid(alpha=0.3)
    ya2 = port.groupby(port.index.year).sum()
    axes[2].bar([str(y) for y in ya2.index], ya2.values,
                color=['seagreen' if v > 0 else 'indianred' for v in ya2.values])
    axes[2].set_ylabel('yearly PnL (vu)')
    axes[2].set_xlabel('year')
    axes[2].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    print('\nfigure saved -> %s' % FIG)

    # ------------------------------------------------ mechanical decision rule
    print('\n================= PRE-REGISTERED DECISION RULE (mechanical) =================')
    xfp = {h: pool(xf60, h) for h in HORIZONS}
    any_pass = False
    for thr in THRS:
        for h in HORIZONS:
            for famkey in FAMS + ['ALL']:
                d = grid[(thr, h, famkey)]
                x = xfp[h]
                cond = (d['n'] > 0 and np.isfinite(d['e2c']) and d['e2c'] >= 1
                        and d['p'] <= 0.05
                        and x['n'] > 0 and np.isfinite(x['t'])
                        and np.sign(x['mean']) == np.sign(d['mean'])
                        and x['t'] >= 1.5)
                if cond:
                    any_pass = True
                    print('CELL PASSES kill-gate: thr=%g h=%d fam=%s '
                          '(e2c=%.2f p=%.3f xfeed t=%.2f)' %
                          (thr, h, famkey, d['e2c'], d['p'], x['t']))
    if not any_pass:
        print('NO pre-registered cell passes (need e2c>=1 AND placebo p<=0.05 AND')
        print('cross-feed same-sign t>=1.5)  ->  KILL')
    else:
        promo = (mt['sharpe'] >= 0.5 and mt2['sharpe'] > 0
                 and mt['h1'] > 0 and mt['h2'] > 0)
        print('kill-gate passed; promote-gate (Sharpe>=0.5 x1, >0 stressed, both '
              'halves>0): %s' % ('PASS -> PROMOTE' if promo else 'FAIL -> UNRESOLVED'))

    print('\nheadline @ primary: x1 %s | stressed sharpe %.2f | trades %d' %
          (mt, mt2['sharpe'], n_tr))


if __name__ == '__main__':
    main()
