#!/usr/bin/env /usr/bin/python3
"""
H8 — TIME-OF-DAY REVERSION CONDITIONING (hourly; 9 duka + 34 fx = 43 instruments).

Mechanism: liquidity and flow are periodic (session opens, WMR London fix,
CME settlements, option-expiry hedging). Mechanical/benchmark-tracking flows
push price away from fair value at specific clock times; once the flow window
closes the move partially reverts. If true, fading large 1-bar moves should
work better at those UTC hours than at random hours.

PRE-SPECIFIED DESIGN (everything below fixed before any results were seen):
  Event   : bar UTC hour == H (H = 0..23) AND |C[t]-C[t-1]| / ATR100 >= 1.0.
            (ATR from lab.atr, already causally shifted; move uses event-bar
            close -> computable at event-bar close; entry next bar open.)
  Side    : -sign(C[t]-C[t-1])  — fade the move (ex-ante from mechanism).
  Horizons: (1, 2, 4) bars.
  Families: CMD = 9 duka commodities; MAJ = 10 FX majors; CRX = 24 FX
            crosses+exotics; ALL = 43 pooled.
  Placebo : per instrument, same event count drawn (with replacement) from the
            instrument's OWN big-move universe across ALL hours, same fade
            side rule -> p tests the HOUR conditioning beyond any generic
            fade-after-big-move edge. 300 iterations, count-weighted pooling.
  De-overlap: same-hour events are >=~20 bars apart -> no overlap at h<=4.
  MULTIPLE TESTING discipline (fixed ex-ante): 24x3 cells per family; single
            hot cells are noise. Promising requires (a) a contiguous block of
            >=2 adjacent hours significant (|t|>=2) in the SAME direction
            across >=2 families, or (b) an economically-nameable hour
            (NY open 13-14, London fix 15-16 across DST, CME settle 18-19)
            with |t|>=3.5 within a family.
  NATURAL VARIANT (declared ex-ante, the ONLY cost-aware backtest): event
            hours {15,16} = London 4pm WMR fix hour in UTC across DST,
            h=2, fade, all 43 instruments, per-symbol costs (+x2 stress).
  Context : unconditional per-hour fade t (no size filter), same horizons.

Run: cd ortho && /usr/bin/python3 h8_hour_of_day.py
"""
import numpy as np
import pandas as pd
import lab

HORIZONS      = (1, 2, 4)
THR           = 1.0
NATURAL_HOURS = (15, 16)
NATURAL_H     = 2
N_PLACEBO     = 300
MIN_N         = 10

FX_MAJ = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD',
          'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF']

SYMS = ([('CMD', s) for s in lab.DUKA_SYMS] +
        [('MAJ' if s in FX_MAJ else 'CRX', s) for s in lab.fx_syms()])
FAMS = ['CMD', 'MAJ', 'CRX']


def _deoverlap(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def build_universe(df, a, cb, thr):
    """Per horizon: indices, UTC hour, signed fade fwd return (vu), cost (vu)
    of every bar with |1-bar move| >= thr ATR."""
    c, o = df.c.values, df.o.values
    av = a.values
    mv = np.r_[np.nan, np.diff(c)] / av
    hr = df.index.hour.values
    N = len(df)
    ok0 = np.isfinite(mv) & (np.abs(mv) >= thr) & (mv != 0) & \
          np.isfinite(av) & (av > 0)
    out = {}
    for h in HORIZONS:
        idx = np.flatnonzero(ok0)
        idx = idx[idx + h < N - 1]
        s = -np.sign(mv[idx])
        f = s * (c[idx + h] - o[idx + 1]) / av[idx]
        fin = np.isfinite(f)
        idx, f = idx[fin], f[fin]
        cv = (cb / 1e4) * np.abs(o[idx + 1]) / av[idx]
        out[h] = dict(idx=idx, hr=hr[idx], f=f, cv=cv)
    return out


def cell_stats(uni, rng):
    """For one instrument: per horizon, per hour -> (n, f_sel, cv_sel, pm).
    pm = N_PLACEBO placebo means: same count drawn from the instrument's full
    big-move universe (any hour), same fade rule."""
    res = {}
    for h in HORIZONS:
        u = uni[h]
        f_all = u['f']
        M = len(f_all)
        per_hour = {}
        for H in range(24):
            sel = u['hr'] == H
            n = int(sel.sum())
            if n < MIN_N or M < MIN_N:
                per_hour[H] = None
                continue
            draws = rng.integers(0, M, size=(N_PLACEBO, n))
            pm = f_all[draws].mean(axis=1)
            per_hour[H] = dict(n=n, f=f_all[sel], cv=u['cv'][sel], pm=pm)
        res[h] = per_hour
    return res


def pool_cells(cells_list, h, H):
    """Stack signed event returns across instruments for one (h, hour) cell;
    pooled placebo = per-iteration count-weighted mean."""
    fs, pms, ns, cvs = [], [], [], []
    for cells in cells_list:
        d = cells[h][H]
        if d is None:
            continue
        fs.append(d['f'])
        pms.append(d['pm'] * d['n'])
        ns.append(d['n'])
        cvs.append(d['cv'].sum())
    if not fs:
        return dict(n=0, mean=np.nan, t=np.nan, p=np.nan, cost_vu=np.nan,
                    pos=0, tot=0)
    F = np.concatenate(fs)
    m = F.mean()
    t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
    ntot = sum(ns)
    PM = np.vstack(pms).sum(axis=0) / ntot
    p = float((np.abs(PM) >= abs(m)).mean())
    return dict(n=len(F), mean=m, t=t, p=p, cost_vu=sum(cvs) / ntot,
                pos=sum(1 for f in fs if f.mean() > 0), tot=len(fs))


def natural_event(df, a, hours, h):
    """Mask+sides for the ex-ante natural variant; events de-overlapped at gap
    h (hours 15 & 16 same day can overlap at h=2)."""
    c = df.c.values
    av = a.values
    mv = np.r_[np.nan, np.diff(c)] / av
    hr = df.index.hour.values
    raw = np.isfinite(mv) & (np.abs(mv) >= THR) & (mv != 0) & \
          np.isfinite(av) & (av > 0) & np.isin(hr, hours)
    pos = _deoverlap(np.flatnonzero(raw), h)
    mask = np.zeros(len(df), bool)
    mask[pos] = True
    sides = np.zeros(len(df))
    sides[mask] = -np.sign(mv[mask])
    return mask, sides


def natural_stats(df, a, cb, hours, h, rng):
    """Signed study of the natural variant for one instrument + placebo from
    the instrument's full big-move universe."""
    c, o = df.c.values, df.o.values
    av = a.values
    mv = np.r_[np.nan, np.diff(c)] / av
    hr = df.index.hour.values
    N = len(df)
    ok0 = np.isfinite(mv) & (np.abs(mv) >= THR) & (mv != 0) & \
          np.isfinite(av) & (av > 0)
    uni = np.flatnonzero(ok0)
    uni = uni[uni + h < N - 1]
    f_all = -np.sign(mv[uni]) * (c[uni + h] - o[uni + 1]) / av[uni]
    fin = np.isfinite(f_all)
    uni, f_all = uni[fin], f_all[fin]
    pos = _deoverlap(uni[np.isin(hr[uni], hours)], h)
    if len(pos) < MIN_N or len(f_all) < MIN_N:
        return None
    f = -np.sign(mv[pos]) * (c[pos + h] - o[pos + 1]) / av[pos]
    cv = (cb / 1e4) * np.abs(o[pos + 1]) / av[pos]
    draws = rng.integers(0, len(f_all), size=(N_PLACEBO, len(pos)))
    pm = f_all[draws].mean(axis=1)
    return dict(n=len(f), f=f, cv=cv, pm=pm)


def main():
    print(__doc__)
    data = []
    print('loading %d instruments ...' % len(SYMS))
    for fam, sym in SYMS:
        df = lab.duka(sym) if fam == 'CMD' else lab.fx(sym)
        a = lab.atr(df)
        data.append((fam, sym, df, a))
    print('loaded.')

    # ------------- per-instrument cell stats (event: |move|>=1 ATR) -------------
    cells_by_fam = {f: [] for f in FAMS}
    cells_all = []
    for i, (fam, sym, df, a) in enumerate(data):
        uni = build_universe(df, a, lab.cost_bps(sym), THR)
        cells = cell_stats(uni, np.random.default_rng(1000 + i))
        cells_by_fam[fam].append(cells)
        cells_all.append(cells)

    # ------------- MAIN GRID: hour x horizon, per family + ALL -------------
    print('\n================ MAIN GRID: fade |move|>=1ATR at UTC hour H ================')
    print('placebo p: same count drawn from instrument\'s own big-move universe (any')
    print('hour), same fade rule -> tests the HOUR conditioning. * marks |t|>=2.')
    grid = {}   # (famkey, h, H) -> pooled dict
    for famkey, lst in list(cells_by_fam.items()) + [('ALL', cells_all)]:
        print('\n--- family %s (%d instruments) ---' % (famkey, len(lst)))
        hdr = 'hr | ' + ' | '.join('h=%d: %6s %6s %6s %5s' % (h, 'n', 'mean', 't', 'p')
                                   for h in HORIZONS)
        print(hdr)
        for H in range(24):
            parts = []
            for h in HORIZONS:
                d = pool_cells(lst, h, H)
                grid[(famkey, h, H)] = d
                if d['n'] == 0:
                    parts.append('h=%d: %6s %6s %6s %5s' % (h, '-', '-', '-', '-'))
                else:
                    star = '*' if np.isfinite(d['t']) and abs(d['t']) >= 2 else ' '
                    parts.append('h=%d: %6d %6.3f %5.2f%s %5.3f' %
                                 (h, d['n'], d['mean'], d['t'], star, d['p']))
            print('%2d | %s' % (H, ' | '.join(parts)))

    # ------------- significant-cell + contiguity summary -------------
    print('\n================ CELLS WITH |t|>=2 (out of 24x3 per family) ================')
    sig = {}
    for famkey in FAMS + ['ALL']:
        hits = []
        for h in HORIZONS:
            for H in range(24):
                d = grid[(famkey, h, H)]
                if d['n'] > 0 and np.isfinite(d['t']) and abs(d['t']) >= 2:
                    hits.append((H, h, d['t'], d['p']))
        sig[famkey] = hits
        print('%s: %d/72 cells |t|>=2 (chance ~3.6): %s' %
              (famkey, len(hits),
               ', '.join('H%d h%d t=%.2f p=%.3f' % x for x in sorted(hits))))
    print('\ncontiguity check (>=2 adjacent hours, same sign, |t|>=2, within a horizon):')
    for famkey in FAMS + ['ALL']:
        for h in HORIZONS:
            hrs = sorted([(H, grid[(famkey, h, H)]['t']) for H in range(24)
                          if grid[(famkey, h, H)]['n'] > 0 and
                          np.isfinite(grid[(famkey, h, H)]['t']) and
                          abs(grid[(famkey, h, H)]['t']) >= 2])
            for (H1, t1), (H2, t2) in zip(hrs, hrs[1:]):
                if H2 == H1 + 1 and np.sign(t1) == np.sign(t2):
                    print('  %s h=%d: hours %d-%d same-sign (t=%.2f, %.2f)' %
                          (famkey, h, H1, H2, t1, t2))

    # ------------- CONTEXT: unconditional per-hour fade t (no size filter) -------------
    print('\n========== CONTEXT: unconditional per-hour fade t (no size filter) ==========')
    uncond = {f: [] for f in FAMS}
    for fam, sym, df, a in data:
        u = build_universe(df, a, lab.cost_bps(sym), 0.0)
        uncond[fam].append(u)
    print('hr | ' + ' | '.join('%s: %6s %6s %6s' % (f, 't@1', 't@2', 't@4')
                               for f in FAMS))
    for H in range(24):
        parts = []
        for f in FAMS:
            ts = []
            for h in HORIZONS:
                fs = [u[h]['f'][u[h]['hr'] == H] for u in uncond[f]]
                fs = [x for x in fs if len(x) >= MIN_N]
                if not fs:
                    ts.append('     -')
                    continue
                F = np.concatenate(fs)
                t = F.mean() / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
                ts.append('%6.2f' % t)
            parts.append('%s: %s' % (f, ' '.join(ts)))
        print('%2d | %s' % (H, ' | '.join(parts)))
    del uncond

    # ------------- per-instrument table @ natural variant -------------
    print('\n========= PER-INSTRUMENT @ natural variant: hours {15,16}, h=%d =========' %
          NATURAL_H)
    print('%-8s %-4s %6s %9s %7s' % ('sym', 'fam', 'n_ev', 'mean_vu', 't'))
    nat = []
    pos_cnt, tot_cnt = 0, 0
    fam_pos = {}
    for i, (fam, sym, df, a) in enumerate(data):
        d = natural_stats(df, a, lab.cost_bps(sym), NATURAL_HOURS, NATURAL_H,
                          np.random.default_rng(5000 + i))
        nat.append(d)
        if d is None:
            print('%-8s %-4s %6s %9s %7s' % (sym, fam, 0, 'nan', 'nan'))
            continue
        m = d['f'].mean()
        t = m / (d['f'].std(ddof=1) / np.sqrt(len(d['f'])) + 1e-12)
        print('%-8s %-4s %6d %9.4f %7.2f' % (sym, fam, d['n'], m, t))
        tot_cnt += 1
        fam_pos.setdefault(fam, [0, 0])[1] += 1
        if m > 0:
            pos_cnt += 1
            fam_pos.setdefault(fam, [0, 0])[0] += 1
    print('\nsign consistency: %d/%d positive (%.0f%%)' %
          (pos_cnt, tot_cnt, 100.0 * pos_cnt / max(tot_cnt, 1)))
    for fam, (p_, t_) in fam_pos.items():
        print('  family %-4s: %d/%d positive' % (fam, p_, t_))
    print('NOTE: 34 FX pairs share legs (USD/EUR/JPY blocs) -> heavily cross-')
    print('      correlated; effective independent count is far below 43.')

    # pooled natural
    fs = [d['f'] for d in nat if d is not None]
    pms = [d['pm'] * d['n'] for d in nat if d is not None]
    ns = [d['n'] for d in nat if d is not None]
    cvs = [d['cv'].sum() for d in nat if d is not None]
    F = np.concatenate(fs)
    m = F.mean()
    t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
    PM = np.vstack(pms).sum(axis=0) / sum(ns)
    p = float((np.abs(PM) >= abs(m)).mean())
    cost_vu = sum(cvs) / sum(ns)
    e2c = m / cost_vu if cost_vu > 0 else np.nan
    print('\npooled @ natural (hours 15-16, h=%d): n=%d mean=%.4f t=%.2f '
          'p_placebo=%.3f' % (NATURAL_H, len(F), m, t, p))
    print('pooled round-trip cost = %.4f vu -> edge/cost = %.2f' % (cost_vu, e2c))

    # ------------- ONE cost-aware backtest: natural variant only -------------
    print('\n===== BACKTEST (natural variant ONLY: hours {15,16}, h=%d, fade, '
          'per-sym costs) =====' % NATURAL_H)
    parts, n_tr = [], 0
    for fam, sym, df, a in data:
        mask, sides = natural_event(df, a, NATURAL_HOURS, NATURAL_H)
        pnl, n = lab.backtest(df, mask, h=NATURAL_H, sides=sides,
                              cost=lab.cost_bps(sym), a=a)
        parts.append(pnl)
        n_tr += n
    port = pd.concat(parts, axis=1).fillna(0.0).sum(axis=1).sort_index()
    print('portfolio (sum of 43 single-unit instrument books), trades=%d' % n_tr)
    print(lab.metrics(port))
    print('yearly sharpe:')
    print(lab.yearly(port))

    parts2 = []
    for fam, sym, df, a in data:
        mask, sides = natural_event(df, a, NATURAL_HOURS, NATURAL_H)
        pnl, _ = lab.backtest(df, mask, h=NATURAL_H, sides=sides,
                              cost=lab.cost_bps(sym, stress=2.0), a=a)
        parts2.append(pnl)
    port2 = pd.concat(parts2, axis=1).fillna(0.0).sum(axis=1).sort_index()
    print('\nstress x2 costs:')
    print(lab.metrics(port2))

    print('\nbest ALL-pooled grid cell by |t| (completeness only; backtest above is')
    print('the ex-ante natural variant, NOT tuned):')
    best = max(((h, H) for h in HORIZONS for H in range(24)),
               key=lambda x: abs(grid[('ALL', x[0], x[1])]['t'])
               if grid[('ALL', x[0], x[1])]['n'] > 0 and
               np.isfinite(grid[('ALL', x[0], x[1])]['t']) else -1)
    b = grid[('ALL', best[0], best[1])]
    e2cb = b['mean'] / b['cost_vu'] if b['cost_vu'] and b['cost_vu'] > 0 else np.nan
    print('  H=%d h=%d: n=%d mean=%.4f t=%.2f p=%.3f e2c=%.2f sign+ %d/%d' %
          (best[1], best[0], b['n'], b['mean'], b['t'], b['p'], e2cb,
           b['pos'], b['tot']))

    # ------------- POST-HOC DIAGNOSTICS (labelled as such; NO second backtest) ----
    # Purpose: characterize the discovered 17-22 UTC hot block, NOT tune a strategy.
    # (a) e2c per family for H=17..22, h=2 (at NORMAL-hour flat-bps costs — these
    #     hours are the thinnest of the FX day, so real costs are higher);
    # (b) first-half vs second-half-of-sample stability of the 20-22 block;
    # (c) weekday split of the 20-22 block (Fri events enter at Sunday open ->
    #     weekend-gap confound; Sun events are thinnest bars of the week).
    print('\n================ POST-HOC DIAGNOSTICS (hot block 17-22 UTC) ================')
    print('(a) e2c at h=2, flat normal-hour costs (real costs at these hours are higher):')
    for famkey in FAMS + ['ALL']:
        row = []
        for H in range(17, 23):
            d = grid[(famkey, 2, H)]
            e = d['mean'] / d['cost_vu'] if d['n'] > 0 and d['cost_vu'] > 0 else np.nan
            row.append('H%d %.2f' % (H, e))
        print('  %-3s: %s' % (famkey, '  '.join(row)))

    def block_events(df, a, hours, h):
        c, o = df.c.values, df.o.values
        av = a.values
        mv = np.r_[np.nan, np.diff(c)] / av
        hr = df.index.hour.values
        N = len(df)
        raw = np.isfinite(mv) & (np.abs(mv) >= THR) & (mv != 0) & \
              np.isfinite(av) & (av > 0) & np.isin(hr, hours)
        pos = _deoverlap(np.flatnonzero(raw), h)
        pos = pos[pos + h < N - 1]
        f = -np.sign(mv[pos]) * (c[pos + h] - o[pos + 1]) / av[pos]
        fin = np.isfinite(f)
        pos, f = pos[fin], f[fin]
        return df.index[pos], f

    def _mt(F):
        if len(F) < MIN_N:
            return (len(F), np.nan, np.nan)
        return (len(F), F.mean(),
                F.mean() / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12))

    for hours, lbl in [((20, 21, 22), '20-22'), ((17, 18, 19), '17-19')]:
        h1f = {f: [] for f in FAMS}
        h2f = {f: [] for f in FAMS}
        dow = {f: {'fri': [], 'sun': [], 'monthu': []} for f in FAMS}
        for fam, sym, df, a in data:
            ts, f = block_events(df, a, hours, 2)
            if len(f) < MIN_N:
                continue
            mid = ts[len(ts) // 2]
            h1f[fam].append(f[ts < mid])
            h2f[fam].append(f[ts >= mid])
            wd = ts.dayofweek.values
            dow[fam]['fri'].append(f[wd == 4])
            dow[fam]['sun'].append(f[wd == 6])
            dow[fam]['monthu'].append(f[wd <= 3])
        print('\n(b) block %s h=2, first vs second half of sample:' % lbl)
        for famkey in FAMS:
            a1 = np.concatenate(h1f[famkey]) if h1f[famkey] else np.array([])
            a2 = np.concatenate(h2f[famkey]) if h2f[famkey] else np.array([])
            n1, m1, t1 = _mt(a1)
            n2, m2, t2 = _mt(a2)
            print('  %-3s: 1st n=%6d mean=%7.4f t=%6.2f | 2nd n=%6d mean=%7.4f t=%6.2f'
                  % (famkey, n1, m1, t1, n2, m2, t2))
        print('(c) block %s h=2, by weekday (Fri events fill at SUNDAY open):' % lbl)
        for famkey in FAMS:
            parts = []
            for k in ['fri', 'sun', 'monthu']:
                arr = ([x for x in dow[famkey][k]] or [np.array([])])
                A = np.concatenate(arr) if arr else np.array([])
                n, m, t = _mt(A)
                parts.append('%s n=%5d m=%7.4f t=%6.2f' % (k, n, m, t))
            print('  %-3s: %s' % (famkey, ' | '.join(parts)))


if __name__ == '__main__':
    main()
