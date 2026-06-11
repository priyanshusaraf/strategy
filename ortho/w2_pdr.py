#!/usr/bin/env /usr/bin/python3
"""
W2-A — POST-DISLOCATION RELAXATION (PDR). Wave-2 falsification gauntlet.

FROZEN RULE (pre-registered, distilled from wave-1 H3 extreme-vol fade /
H2 24-48h wave / H1 h48 cell; run EXACTLY, no tuning):
  RV24[t] = std of last 24 hourly close-changes (causal, through t).
  B[t]    = median of RV24 over the prior 240 bars, then shift(1).
  STATE   : RV24/B >= 3.0           (no crest condition — wave 1 falsified it).
  EVENT   : every state bar. side = -sign(C[t] - C[t-24]). Entry NEXT OPEN.
  EXIT    : fixed horizon, PRIMARY h=16 bars. One position per instrument
            (lab.backtest enforces).
NEIGHBORHOOD (robustness REPORTING only, never selection):
  k in {2.5, 3.0, 3.5} x h in {8, 16, 24, 48} (full cross incl. the registered
  {2.5,3.5}x{8,24,48} neighborhood and the primary cell).
UNIVERSE: 9 duka commodity CFDs (12.3y hourly) + 34 FX pairs (10.4y hourly).
CROSS-FEED OOS (untouched by wave 1):
  (a) TradingView 60-min exports (~4y, different venues/feeds). NOTE:
      'TVC_SILVER, 60.csv' from the registration is ABSENT on disk (only
      15-min/1D exist); excluded rather than substituting a different bar
      size -> 16 of 17 instruments.
  (b) all 25 lab.yahoo front-month legs (2y, real futures). CENSOR-AWARE:
      signal (RV24, 24h net move) built on roll-censored close-changes
      (cens-bar diff zeroed), pnl censored natively by lab.backtest, and
      event-study forward returns use the censored cumulative-change series.
BATTERY (all mandatory): signed event study with SIGNED placebo (random times
  + same -sign(24h move) side rule, 300 draws) pooled/per-instrument/per-family;
  primary-cell portfolio backtest (metrics, halves, thirds, yearly, figure);
  cost stress x1 / x2-all / x3-exotics / drop-exotics / commodities-only;
  episodicity; cross-feed validation; shuffle test (100); k x h grid.
PRE-REGISTERED DECISION RULE — promote iff ALL of:
  primary portfolio Sharpe >= 0.5 @ x1; both halves > 0; >= 60% of instruments
  positive total; cross-feed pooled t >= 2 same sign on BOTH feeds;
  x2-cost Sharpe > 0; drop-exotics portfolio > 0. Else kill.

Run: cd ortho && /usr/bin/python3 w2_pdr.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab

RV_N      = 24
BASE_N    = 240
K_PRI     = 3.0
H_PRI     = 16
KS        = (2.5, 3.0, 3.5)
HS        = (8, 16, 24, 48)
HORIZONS  = (2, 4, 8, 16, 24, 48)
N_PLACEBO = 300
N_SHUFFLE = 100
MIN_N     = 10

FX_MAJ = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD',
          'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF']
FX_EXO = ['EURNOK', 'EURSEK', 'USDNOK', 'USDSEK', 'EURPLN', 'EURHUF',
          'EURCZK', 'AUDSGD']
CORE_FAMS = ['CMD', 'MAJ', 'CRX', 'EXO']

# TradingView 60-min exports + sensible round-trip cost (bps), per registration.
TV_LIST = [
    ('ICEEUR_DLY_BRN1!, 60 (1).csv', 'TV_BRN',      4.0),
    ('CFI_WTI, 60.csv',              'TV_WTI',      4.0),
    ('COMEX_DL_GC2!, 60.csv',        'TV_GC',       2.0),
    ('COMEX_DL_SI2!, 60.csv',        'TV_SI',       6.0),
    ('COMEX_DL_HG1!, 60.csv',        'TV_HG',       8.0),
    ('NYMEX_DL_PA1!, 60.csv',        'TV_PA',      15.0),
    ('CBOT_DL_ZC1!, 60.csv',         'TV_ZC',       5.0),
    ('CAPITALCOM_COCOA, 60.csv',     'TV_COCOA',   12.0),
    ('FOREXCOM_COFFEE, 60.csv',      'TV_COFFEE',  10.0),
    ('FOREXCOM_COTTON, 60.csv',      'TV_COTTON',  10.0),
    ('OANDA_EURUSD, 60 (1).csv',     'TV_EURUSD',   1.5),
    ('OANDA_USDJPY, 60.csv',         'TV_USDJPY',   1.5),
    ('FOREXCOM_USDCHF, 60.csv',      'TV_USDCHF',   1.5),
    ('TVC_DXY, 60.csv',              'TV_DXY',      2.0),
    ('TVC_PLATINUM, 60.csv',         'TV_PLATINUM', 10.0),
    ('TVC_SILVER, 60.csv',           'TV_SILVER',   6.0),   # absent on disk; documented
    ('TVC_US10Y, 60.csv',            'TV_US10Y',    2.0),
]


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def build_signal(df):
    """ratio = RV24/B (causal), net = C[t]-C[t-24]. If 'cens' present (yahoo),
    both are built from roll-censored close-changes."""
    dc = df.c.diff()
    if 'cens' in df.columns:
        dc = dc.where(~df['cens'].astype(bool), 0.0)
    rv = dc.rolling(RV_N).std()
    base = rv.rolling(BASE_N, min_periods=BASE_N // 2).median().shift(1)
    ratio = (rv / base).values
    cc = dc.fillna(0.0).values.cumsum()
    net = np.full(len(df), np.nan)
    net[RV_N:] = cc[RV_N:] - cc[:-RV_N]
    return ratio, net


def mk_inst(name, fam, df, cb):
    a = lab.atr(df)
    ratio, net = build_signal(df)
    mask, sides = {}, {}
    for k in KS:
        m = np.isfinite(ratio) & (ratio >= k) & np.isfinite(net) & (net != 0)
        s = np.zeros(len(df))
        s[m] = -np.sign(net[m])
        mask[k], sides[k] = m, s
    return dict(name=name, fam=fam, df=df, a=a, cb=cb, mask=mask, sides=sides,
                net=net)


def signed_study(inst, horizons=HORIZONS, n_placebo=N_PLACEBO, seed=0):
    """Signed event study at the PRIMARY k. Forward returns signed by the fade
    rule; placebo = same count at random valid times with the SAME
    -sign(24h move) side rule (tests the dislocation conditioning, not the
    generic 24h fade). Censor-aware via the censored cumulative-change series."""
    df, net, cb = inst['df'], inst['net'], inst['cb']
    c, o = df.c.values, df.o.values
    av = inst['a'].values
    cens = (df['cens'].values.astype(bool) if 'cens' in df.columns
            else np.zeros(len(df), bool))
    dc = np.r_[0.0, np.diff(c)]
    dc[cens] = 0.0
    cc = dc.cumsum()

    def fwd(P, h):
        first = np.where(cens[P + 1], 0.0, c[P + 1] - o[P + 1])
        return (first + cc[P + h] - cc[P + 1]) / av[P]

    pos = np.flatnonzero(inst['mask'][K_PRI])
    N = len(df)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(av) & (av > 0) &
                           np.isfinite(net) & (net != 0))
    out = {}
    for h in horizons:
        P = _dedup(pos[(pos + h) < N - 1], h)
        P = P[np.isfinite(av[P]) & (av[P] > 0)]
        V = valid[(valid + h) < N - 1]
        if len(P) < MIN_N or len(V) < MIN_N:
            out[h] = dict(n=len(P), mean=np.nan, t=np.nan, f=np.array([]),
                          pm=np.full(n_placebo, np.nan), cost_vu=np.nan)
            continue
        s = -np.sign(net[P])
        f = s * fwd(P, h)
        ok = np.isfinite(f)
        f = f[ok]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        cost_vu = (cb / 1e4) * float(np.nanmean(np.abs(o[P + 1][ok]) / av[P][ok]))
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = V[rng.integers(0, len(V), size=len(P))]
            g = -np.sign(net[Q]) * fwd(Q, h)
            pm[i] = np.nanmean(g)
        out[h] = dict(n=len(f), mean=m, t=t, f=f, pm=pm, cost_vu=cost_vu)
    return out


def pool(results, h):
    """Stack signed event returns across instruments; pooled placebo =
    per-iteration count-weighted mean of per-instrument placebo means."""
    fs, pms, ns, costs = [], [], [], []
    for r in results:
        d = r[h]
        if d['n'] >= MIN_N and np.isfinite(d['mean']):
            fs.append(d['f'])
            pms.append(d['pm'] * d['n'])
            ns.append(d['n'])
            costs.append(d['cost_vu'] * d['n'])
    if not fs:
        return dict(n=0, mean=np.nan, t=np.nan, p=np.nan, cost_vu=np.nan,
                    pos=0, tot=0)
    F = np.concatenate(fs)
    m = F.mean()
    t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
    ntot = sum(ns)
    PM = np.vstack(pms).sum(axis=0) / ntot
    p = float((np.abs(PM) >= abs(m)).mean())
    return dict(n=len(F), mean=m, t=t, p=p, cost_vu=sum(costs) / ntot,
                pos=sum(1 for f in fs if f.mean() > 0), tot=len(fs))


def run_port(insts, k, h, stress=1.0, exo_stress=None, subset_fams=None,
             drop_fams=None, want_frame=False):
    parts, ntr, per = [], 0, {}
    for r in insts:
        if subset_fams is not None and r['fam'] not in subset_fams:
            continue
        if drop_fams is not None and r['fam'] in drop_fams:
            continue
        st = exo_stress if (exo_stress is not None and r['fam'] == 'EXO') else stress
        pnl, n = lab.backtest(r['df'], r['mask'][k], h=h, sides=r['sides'][k],
                              cost=r['cb'] * st, a=r['a'])
        parts.append(pnl.rename(r['name']))
        ntr += n
        per[r['name']] = (n, float(pnl.sum()))
    frame = pd.concat(parts, axis=1).fillna(0.0).sort_index()
    port = frame.sum(axis=1)
    return port, ntr, per, (frame if want_frame else None)


def thirds_sharpe(pnl):
    n = len(pnl)
    bpy = lab.bars_per_year(pnl.to_frame())
    out = []
    for i in range(3):
        seg = pnl.iloc[i * n // 3:(i + 1) * n // 3]
        out.append(round(float(seg.mean() / seg.std() * np.sqrt(bpy)), 2)
                   if seg.std() > 0 else 0.0)
    return out


def shuffle_port(insts, k, h, n_shuffle=N_SHUFFLE, seed=11):
    """Portfolio shuffle: per instrument, same NUMBER of event bars at random
    times with random +-1 sides; iteration i totals summed across instruments."""
    rng = np.random.default_rng(seed)
    tots = np.zeros(n_shuffle)
    for r in insts:
        N = len(r['df'])
        kk = int(r['mask'][k].sum())
        if kk == 0:
            continue
        for i in range(n_shuffle):
            m2 = np.zeros(N, bool)
            m2[rng.choice(N - 2, size=min(kk, N - 2), replace=False)] = True
            s2 = np.zeros(N)
            s2[m2] = rng.choice([-1, 1], size=int(m2.sum()))
            p2, _ = lab.backtest(r['df'], m2, h=h, sides=s2, cost=r['cb'],
                                 a=r['a'])
            tots[i] += p2.sum()
    return tots


def study_block(insts, label, seed0):
    """Run signed studies, print pooled grid (per family + ALL) and the
    per-instrument h=H_PRI table. Returns (studies, pooled@H_PRI for ALL)."""
    studies = []
    for i, r in enumerate(insts):
        studies.append(signed_study(r, seed=seed0 + i))
    fams = sorted(set(r['fam'] for r in insts))
    groups = [(f, [s for r, s in zip(insts, studies) if r['fam'] == f])
              for f in fams if len(fams) > 1]
    groups.append(('ALL', studies))
    print('\n---- %s: pooled signed event study (placebo = random times, same '
          'fade-24h side rule, %d draws) ----' % (label, N_PLACEBO))
    print('%-5s %-3s | %7s %9s %7s %6s %6s %6s' %
          ('fam', 'h', 'n_ev', 'mean_vu', 't', 'p_plc', 'e2c', 'sgn+'))
    pooled_all = {}
    for fam, lst in groups:
        for h in HORIZONS:
            d = pool(lst, h)
            if fam == 'ALL':
                pooled_all[h] = d
            e2c = d['mean'] / d['cost_vu'] if d['cost_vu'] and d['cost_vu'] > 0 \
                else np.nan
            print('%-5s %-3d | %7d %9.4f %7.2f %6.3f %6.2f %3d/%-3d' %
                  (fam, h, d['n'], d['mean'], d['t'], d['p'], e2c,
                   d['pos'], d['tot']))
        print()
    print('---- %s: per-instrument @ primary h=%d ----' % (label, H_PRI))
    print('%-12s %-4s %6s %9s %7s' % ('sym', 'fam', 'n_ev', 'mean_vu', 't'))
    pos_cnt = tot_cnt = 0
    for r, s in zip(insts, studies):
        d = s[H_PRI]
        if d['n'] < MIN_N or not np.isfinite(d['mean']):
            print('%-12s %-4s %6d %9s %7s' % (r['name'], r['fam'], d['n'],
                                              'nan', 'nan'))
            continue
        print('%-12s %-4s %6d %9.4f %7.2f' % (r['name'], r['fam'], d['n'],
                                              d['mean'], d['t']))
        tot_cnt += 1
        pos_cnt += d['mean'] > 0
    print('event-study sign consistency @h=%d: %d/%d positive (%.0f%%)' %
          (H_PRI, pos_cnt, tot_cnt, 100.0 * pos_cnt / max(tot_cnt, 1)))
    return studies, pooled_all


def main():
    print(__doc__)

    # ================= load core universe =================
    print('=== loading core universe (9 duka + 34 fx) ===')
    insts = []
    for s in lab.DUKA_SYMS:
        insts.append(mk_inst(s, 'CMD', lab.duka(s), lab.cost_bps(s)))
    for s in lab.fx_syms():
        fam = 'MAJ' if s in FX_MAJ else ('EXO' if s in FX_EXO else 'CRX')
        insts.append(mk_inst(s, fam, lab.fx(s), lab.cost_bps(s)))
    print('loaded %d core instruments.' % len(insts))

    # ================= 1. signed event study =================
    print('\n================ BATTERY 1: SIGNED EVENT STUDY (core 43) ================')
    _, pooled_core = study_block(insts, 'CORE 9duka+34fx', seed0=100)
    print('NOTE: 34 FX pairs share legs (USD/EUR/JPY blocs) -> heavily cross-'
          'correlated; effective independent count far below 43.')

    # ================= 2. primary-cell portfolio backtest =================
    print('\n================ BATTERY 2: PRIMARY BACKTEST k=%.1f h=%d (x1 costs) '
          '================' % (K_PRI, H_PRI))
    port1, ntr1, per1, frame1 = run_port(insts, K_PRI, H_PRI, want_frame=True)
    m1 = lab.metrics(port1)
    th = thirds_sharpe(port1)
    print('portfolio (sum of %d single-vol-unit books), trades=%d' %
          (len(insts), ntr1))
    print(m1)
    print('thirds sharpe: %s' % th)
    print('yearly sharpe:')
    yr_tab = lab.yearly(port1)
    print(yr_tab)

    cors = frame1.corr().values
    iu = np.triu_indices(cors.shape[0], 1)
    avg_corr = float(np.nanmean(cors[iu]))
    print('avg pairwise corr of instrument pnl: %.4f' % avg_corr)

    print('\nper-instrument backtest @ primary (x1):')
    print('%-10s %-4s %8s %10s' % ('sym', 'fam', 'trades', 'total_vu'))
    pos_bt = tot_bt = 0
    fam_bt = {f: [0, 0] for f in CORE_FAMS}
    for r in insts:
        n, tot = per1[r['name']]
        print('%-10s %-4s %8d %10.1f' % (r['name'], r['fam'], n, tot))
        if n >= 1:
            tot_bt += 1
            fam_bt[r['fam']][1] += 1
            if tot > 0:
                pos_bt += 1
                fam_bt[r['fam']][0] += 1
    pct_pos = 100.0 * pos_bt / max(tot_bt, 1)
    print('backtest sign consistency: %d/%d instruments positive total (%.0f%%)'
          % (pos_bt, tot_bt, pct_pos))
    for f in CORE_FAMS:
        print('  family %-4s: %d/%d positive' % (f, fam_bt[f][0], fam_bt[f][1]))

    # family sub-portfolios (decomposition)
    print('\nfamily sub-portfolio metrics @ primary (x1):')
    fam_m = {}
    for f in CORE_FAMS:
        pf, nf, _, _ = run_port(insts, K_PRI, H_PRI, subset_fams={f})
        fam_m[f] = lab.metrics(pf)
        print('  %-4s trades=%5d %s' % (f, nf, fam_m[f]))

    # ================= 3. cost stress =================
    print('\n================ BATTERY 3: COST STRESS ================')
    port2, ntr2, _, _ = run_port(insts, K_PRI, H_PRI, stress=2.0)
    m2 = lab.metrics(port2)
    print('x2 ALL           : %s' % m2)
    port_e3, _, _, _ = run_port(insts, K_PRI, H_PRI, stress=1.0, exo_stress=3.0)
    m_e3 = lab.metrics(port_e3)
    print('x3 EXOTICS only  : %s   (others x1)' % m_e3)
    port_e3b, _, _, _ = run_port(insts, K_PRI, H_PRI, stress=2.0, exo_stress=3.0)
    print('x2 all + x3 exot : %s' % lab.metrics(port_e3b))
    port_dx, ntr_dx, _, _ = run_port(insts, K_PRI, H_PRI, drop_fams={'EXO'})
    m_dx = lab.metrics(port_dx)
    print('DROP EXOTICS (x1): %s  trades=%d' % (m_dx, ntr_dx))
    port_cmd, ntr_cmd, _, _ = run_port(insts, K_PRI, H_PRI, subset_fams={'CMD'})
    m_cmd = lab.metrics(port_cmd)
    print('COMMODITIES ONLY : %s  trades=%d' % (m_cmd, ntr_cmd))

    # ================= 4. episodicity =================
    print('\n================ BATTERY 4: EPISODICITY ================')
    yrs_port = (port1.index[-1] - port1.index[0]).days / 365.25
    state_bars = sum(int(r['mask'][K_PRI].sum()) for r in insts)
    tot_bars = sum(len(r['df']) for r in insts)
    nz = port1.index[port1.values != 0]
    gaps = (nz[1:] - nz[:-1]).days if len(nz) > 1 else np.array([0])
    longest_flat = int(np.max(gaps))
    inpos_pct = 100.0 * float((port1.values != 0).mean())
    epy = ntr1 / yrs_port
    tpy = [per1[r['name']][0] /
           max((r['df'].index[-1] - r['df'].index[0]).days / 365.25, 1e-9)
           for r in insts]
    print('portfolio trades total=%d over %.1fy -> %.0f trades/yr (all books)' %
          (ntr1, yrs_port, epy))
    print('per-instrument trades/yr: min=%.1f median=%.1f max=%.1f' %
          (np.min(tpy), np.median(tpy), np.max(tpy)))
    print('%% bars in state (pooled %d instruments): %.2f%% (%d of %d bars)' %
          (len(insts), 100.0 * state_bars / tot_bars, state_bars, tot_bars))
    print('mean holding: fixed h=%d bars (~%dh trading time)' % (H_PRI, H_PRI))
    print('portfolio %% bars with open position (proxy: pnl!=0): %.1f%%' %
          inpos_pct)
    print('longest flat stretch (max gap between nonzero-pnl bars): %d days' %
          longest_flat)

    # ================= 5. cross-feed validation =================
    print('\n================ BATTERY 5: CROSS-FEED VALIDATION (de-facto OOS) '
          '================')
    print('--- feed (a): TradingView 60-min exports ---')
    tv_insts, missing = [], []
    for fname, name, cb in TV_LIST:
        try:
            df = lab.tv(fname)
        except FileNotFoundError:
            missing.append(fname)
            continue
        df = df[df.c.notna()].copy()
        tv_insts.append(mk_inst(name, 'TV', df, cb))
    for fn in missing:
        print("MISSING ON DISK (documented, excluded): '%s'" % fn)
    print('loaded %d TV instruments (registration listed 17).' % len(tv_insts))
    for r in tv_insts:
        print('  %-12s %6d bars  %s -> %s' %
              (r['name'], len(r['df']), r['df'].index[0].date(),
               r['df'].index[-1].date()))
    _, pooled_tv = study_block(tv_insts, 'TV 60-min', seed0=300)
    t_tv = pooled_tv[H_PRI]['t']
    port_tv, ntr_tv, per_tv, _ = run_port(tv_insts, K_PRI, H_PRI)
    m_tv = lab.metrics(port_tv)
    print('TV mini-portfolio @ primary (x1, mapped costs): trades=%d' % ntr_tv)
    print(m_tv)
    print('TV yearly sharpe:')
    print(lab.yearly(port_tv))

    print('\n--- feed (b): 25 yahoo futures legs (censor-aware) ---')
    yh_insts = [mk_inst('YH_' + s, 'YH', lab.yahoo(s), lab.cost_bps(s))
                for s in lab.yahoo_syms()]
    print('loaded %d yahoo legs.' % len(yh_insts))
    _, pooled_yh = study_block(yh_insts, 'YAHOO 2y futures', seed0=600)
    t_yh = pooled_yh[H_PRI]['t']
    port_yh, ntr_yh, per_yh, _ = run_port(yh_insts, K_PRI, H_PRI)
    m_yh = lab.metrics(port_yh)
    print('YH mini-portfolio @ primary (x1): trades=%d' % ntr_yh)
    print(m_yh)
    print('YH yearly sharpe:')
    print(lab.yearly(port_yh))
    print('\ncross-feed decision inputs: pooled t@h=%d  TV=%.2f  YH=%.2f' %
          (H_PRI, t_tv, t_yh))

    # ================= 6. shuffle test =================
    print('\n================ BATTERY 6: SHUFFLE TEST (%d shuffles, portfolio) '
          '================' % N_SHUFFLE)
    tots = shuffle_port(insts, K_PRI, H_PRI)
    real_tot = float(port1.sum())
    p_shuf = float((tots >= real_tot).mean())
    print('real portfolio total = %.1f vu | shuffle mean=%.1f sd=%.1f | '
          'P(shuffle >= real) = %.3f' %
          (real_tot, tots.mean(), tots.std(), p_shuf))

    # ================= 7. k x h neighborhood =================
    print('\n================ BATTERY 7: PARAM NEIGHBORHOOD (portfolio Sharpe '
          '@ x1) ================')
    print('registered neighborhood = {2.5,3.5}x{8,24,48}; full cross printed; '
          'primary cell marked *')
    grid_sh = {}
    print('%-6s | %s' % ('k', ' '.join('h=%-7d' % h for h in HS)))
    for k in KS:
        row = []
        for h in HS:
            pg, ng, _, _ = run_port(insts, k, h)
            sh = lab.metrics(pg)['sharpe']
            grid_sh[(k, h)] = (sh, ng)
            mark = '*' if (k == K_PRI and h == H_PRI) else ' '
            row.append('%5.2f%s ' % (sh, mark))
        print('%-6g | %s' % (k, ' '.join(row)))
    print('trade counts:')
    for k in KS:
        print('  k=%g: %s' % (k, '  '.join('h=%d:%d' % (h, grid_sh[(k, h)][1])
                                           for h in HS)))

    # ================= figure =================
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.2, 1.5, 2], hspace=0.5)
    eq = port1.cumsum()
    ax = fig.add_subplot(gs[0])
    ax.plot(eq.index, eq.values, lw=0.8, color='navy')
    ax.set_title('43-instrument portfolio equity @ x1 costs (Sharpe=%.2f, '
                 'trades=%d)' % (m1['sharpe'], ntr1))
    ax.set_ylabel('cum PnL (vol units)')
    ax.set_xlabel('date')
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[1])
    dd = eq - eq.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.6)
    ax2.set_ylabel('drawdown (vu)')
    ax2.set_xlabel('date')
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(gs[2])
    ytot = port1.groupby(port1.index.year).sum()
    ax3.bar(ytot.index.astype(str), ytot.values,
            color=['seagreen' if v > 0 else 'firebrick' for v in ytot.values])
    ax3.set_ylabel('yearly PnL (vu)')
    ax3.set_xlabel('year')
    ax3.grid(alpha=0.3, axis='y')
    ax4 = fig.add_subplot(gs[3])
    eqtv = port_tv.cumsum()
    eqyh = port_yh.cumsum()
    ax4.plot(eqtv.index, eqtv.values, lw=0.9, color='darkorange',
             label='TV 60-min feed (%d instr, t@16=%.2f)' % (len(tv_insts), t_tv))
    ax4.plot(eqyh.index, eqyh.values, lw=0.9, color='teal',
             label='Yahoo futures feed (25 legs, t@16=%.2f)' % t_yh)
    ax4.set_title('cross-feed OOS mini-portfolios (same frozen rule, x1 costs)')
    ax4.set_ylabel('cum PnL (vu)')
    ax4.set_xlabel('date')
    ax4.legend()
    ax4.grid(alpha=0.3)
    fig.suptitle('W2-A PDR — STATE: RV24/med240(RV24) >= 3.0; side = '
                 '-sign(C-C[-24]); entry next open; h=16; one pos/instrument',
                 fontsize=11)
    out_png = lab.ROOT / 'curves' / 'w2_pdr.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out_png)

    # ================= decision =================
    print('\n================ PRE-REGISTERED DECISION RULE ================')
    crit = [
        ('primary Sharpe >= 0.5 @ x1', m1['sharpe'] >= 0.5,
         'sharpe=%.2f' % m1['sharpe']),
        ('both halves > 0', m1['h1'] > 0 and m1['h2'] > 0,
         'h1=%.2f h2=%.2f' % (m1['h1'], m1['h2'])),
        ('>= 60%% instruments positive total', pct_pos >= 60.0,
         '%.0f%% (%d/%d)' % (pct_pos, pos_bt, tot_bt)),
        ('cross-feed pooled t >= 2, same sign, BOTH feeds',
         (t_tv >= 2.0) and (t_yh >= 2.0),
         'TV t=%.2f, YH t=%.2f' % (t_tv, t_yh)),
        ('x2-cost Sharpe > 0', m2['sharpe'] > 0, 'x2 sharpe=%.2f' % m2['sharpe']),
        ('drop-exotics portfolio > 0', m_dx['sharpe'] > 0,
         'dropEXO sharpe=%.2f total=%.1f' % (m_dx['sharpe'], m_dx['total_vu'])),
    ]
    all_pass = True
    for name, ok, det in crit:
        all_pass &= bool(ok)
        print('  [%s] %s  (%s)' % ('PASS' if ok else 'FAIL', name, det))
    decision = 'PROMOTE' if all_pass else 'KILL'
    print('\nDECISION (mechanical): %s' % decision)

    # ================= machine-readable summary =================
    print('\n===== SUMMARY (machine-readable) =====')
    print('SHARPE_X1=%.2f H1=%.2f H2=%.2f EQR2=%.2f GAINPAIN=%.2f NTRADES=%d '
          'SHARPE_X2=%.2f' % (m1['sharpe'], m1['h1'], m1['h2'], m1['eqR2'],
                              m1['gain_pain'], ntr1, m2['sharpe']))
    print('THIRDS=%s' % th)
    print('PCT_POS=%.0f AVG_CORR=%.4f' % (pct_pos, avg_corr))
    print('T_TV=%.2f T_YH=%.2f TV_SHARPE=%.2f YH_SHARPE=%.2f' %
          (t_tv, t_yh, m_tv['sharpe'], m_yh['sharpe']))
    print('P_SHUFFLE=%.3f REAL_TOT=%.1f' % (p_shuf, real_tot))
    print('DROPEXO_SHARPE=%.2f CMDONLY_SHARPE=%.2f EXO3_SHARPE=%.2f' %
          (m_dx['sharpe'], m_cmd['sharpe'], m_e3['sharpe']))
    print('GRID=%s' % {('%g' % k, h): grid_sh[(k, h)][0] for k in KS for h in HS})
    print('POOLED_CORE_T16=%.2f POOLED_CORE_P16=%.3f' %
          (pooled_core[H_PRI]['t'], pooled_core[H_PRI]['p']))
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
