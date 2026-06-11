#!/usr/bin/env /usr/bin/python3
"""
W3-A — VOLUME-CLIMAX / CAPITULATION REVERSAL (wave-3 orthogonal MR).

Mechanism: a price thrust executed on EXTREME participation (volume climax) marks
forced/panicked flow exhausting itself — the marginal seller/buyer is done, and
price relaxes. Volume distinguishes capitulation (revert) from quiet drift; wave-1
H2 showed price-only thrusts are ~5x sub-cost, so volume conditioning must
CONCENTRATE the edge to matter.

FROZEN RULE (pre-registered; run exactly, no tuning):
  vnorm[t] = v[t] / median(v over the prior 30 SAME-UTC-HOUR bars)   (causal;
             volume is strongly diurnal — never a flat rolling median).
  EVENT    : vnorm >= K  AND  |C[t]-C[t-1]| / ATR100 >= 1.5.
  SIDE     : -sign(C[t]-C[t-1]).  Entry NEXT bar open (lab.backtest).
  PRIMARY  : K = 3, h = 8.   GRID (robustness reporting, not selection):
             K in {3, 5} x h in {2, 4, 8, 16, 24}.
  VARIANT B (climax close, reported separately): additionally require close in
             the extreme 25% of the bar's own range in the move direction.

BINDING HYGIENE RULE (wave-2 forensics): for hourly FX, bars opening 20:00-22:59
UTC and ALL Sunday bars are excluded from event detection AND entry (Dukascopy
bid-candle rollover artifact). For duka commodities Sunday bars are excluded.
Yahoo legs: 'cens' roll-gap bars excluded from detection and entry; forward
returns / pnl roll-censored.

UNIVERSE : 9 duka commodity CFDs (12.3y hourly, REAL volume — primary)
           + 34 FX (10.4y hourly, tick volume — secondary).
CONFIRM  : 25 lab.yahoo front-month futures legs (2y, real volume), same rule,
           pooled stats.

BATTERY: signed placebo event study — placebo = random times among bars passing
  the |move|>=1.5 filter ALONE (volume-agnostic), SAME -sign(move) side rule,
  300 draws -> isolates the VOLUME conditioning specifically; per-instrument +
  family tables; portfolio backtest primary cell x1/x2 -> ../curves/
  w3_volclimax.png; halves/thirds/yearly; shuffle 100; episodicity; param grid.

PRE-REGISTERED DECISION RULE — promote iff ALL of:
  (1) volume-conditioned pooled t >= 3 vs the move-only placebo at the primary
      cell (t of real pooled mean minus placebo-mean expectation, pooled core);
  (2) pooled edge/cost >= 1 at the primary cell;
  (3) >= 60% of instruments same sign (per-instrument event mean > 0 @ primary);
  (4) Yahoo confirm feed same sign with pooled t >= 1.5 @ primary;
  (5) portfolio Sharpe >= 0.5 @ x1 costs AND > 0 @ x2 costs.
  kill otherwise (unresolved only for data failure).

Run: cd ortho && /usr/bin/python3 w3_volclimax.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab

MOVE_THR  = 1.5
VN_WIN    = 30
K_PRI     = 3.0
H_PRI     = 8
KS        = (3.0, 5.0)
HS        = (2, 4, 8, 16, 24)
HORIZONS  = (1, 2, 4, 8, 16, 24)
N_PLACEBO = 300
N_SHUFFLE = 100
MIN_N     = 10

FX_MAJ = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD',
          'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF']
FX_EXO = ['EURNOK', 'EURSEK', 'USDNOK', 'USDSEK', 'EURPLN', 'EURHUF',
          'EURCZK', 'AUDSGD']
CORE_FAMS = ['CMD', 'MAJ', 'CRX', 'EXO']


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def vnorm_series(df):
    """v[t] / median(v over prior 30 same-UTC-hour bars). Causal: shift(1)
    within the hour group BEFORE the rolling median."""
    med = df.v.groupby(df.index.hour).transform(
        lambda s: s.shift(1).rolling(VN_WIN, min_periods=VN_WIN).median())
    mv = med.values.astype(float)
    v = df.v.values.astype(float)
    vn = np.full(len(df), np.nan)
    ok = np.isfinite(mv) & (mv > 0) & np.isfinite(v)
    vn[ok] = v[ok] / mv[ok]
    return vn


def hygiene_ex(df, fam):
    """True on bars excluded from event detection AND entry."""
    wd = df.index.weekday.values            # Mon=0 .. Sun=6
    if fam in ('MAJ', 'CRX', 'EXO'):        # hourly FX: rollover artifact zone
        hr = df.index.hour.values
        ex = (wd == 6) | ((hr >= 20) & (hr <= 22))
    elif fam == 'CMD':                      # duka commodities
        ex = (wd == 6)
    else:                                   # yahoo futures
        ex = np.zeros(len(df), bool)
    if 'cens' in df.columns:
        ex = ex | df['cens'].values.astype(bool)
    return ex


def mk_inst(name, fam, df, cb):
    a = lab.atr(df)
    av = a.values
    h_, l, c = df.h.values, df.l.values, df.c.values
    mvu = np.full(len(df), np.nan)
    mvu[1:] = (c[1:] - c[:-1]) / av[1:]
    vn = vnorm_series(df)
    ex = hygiene_ex(df, fam)
    ex_entry = np.r_[ex[1:], True]          # entry bar t+1 must be clean too
    okbase = (np.isfinite(av) & (av > 0) & np.isfinite(mvu) & np.isfinite(vn)
              & ~ex & ~ex_entry)
    move_ok = okbase & (np.abs(mvu) >= MOVE_THR)   # placebo universe
    rng_ = h_ - l
    with np.errstate(invalid='ignore', divide='ignore'):
        clp = np.where(rng_ > 0, (c - l) / rng_, np.nan)
    climax = (((mvu > 0) & (clp >= 0.75)) | ((mvu < 0) & (clp <= 0.25)))
    masks = {}
    for K in KS:
        mA = move_ok & (vn >= K)
        masks[('A', K)] = mA
        masks[('B', K)] = mA & climax
    sides = np.zeros(len(df))
    sides[move_ok] = -np.sign(mvu[move_ok])
    return dict(name=name, fam=fam, df=df, a=a, cb=cb, masks=masks,
                sides=sides, move_ok=move_ok, mvu=mvu, vn=vn)


def signed_study(inst, variant, K, horizons=HORIZONS, n_placebo=N_PLACEBO,
                 seed=0):
    """Signed event study. Forward returns signed by -sign(move); placebo =
    same count at random times among bars passing the MOVE filter alone
    (volume-agnostic), SAME side rule -> isolates the volume conditioning.
    Censor-aware (yahoo): forward changes built on roll-censored diffs."""
    df, cb, mvu = inst['df'], inst['cb'], inst['mvu']
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

    pos = np.flatnonzero(inst['masks'][(variant, K)])
    V_all = np.flatnonzero(inst['move_ok'])
    N = len(df)
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        P = _dedup(pos[(pos + h) < N - 1], h)
        V = V_all[(V_all + h) < N - 1]
        if len(P) < MIN_N or len(V) < MIN_N:
            out[h] = dict(n=len(P), n_move=len(V), mean=np.nan, t=np.nan,
                          tvp=np.nan, f=np.array([]),
                          pm=np.full(n_placebo, np.nan), cost_vu=np.nan)
            continue
        s = -np.sign(mvu[P])
        f = s * fwd(P, h)
        ok = np.isfinite(f)
        f = f[ok]
        m = f.mean()
        se = f.std(ddof=1) / np.sqrt(len(f)) + 1e-12
        t = m / se
        cost_vu = (cb / 1e4) * float(np.nanmean(np.abs(o[P + 1][ok]) /
                                                av[P][ok]))
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = V[rng.integers(0, len(V), size=len(P))]
            g = -np.sign(mvu[Q]) * fwd(Q, h)
            pm[i] = np.nanmean(g)
        tvp = (m - np.nanmean(pm)) / se
        out[h] = dict(n=len(f), n_move=len(V), mean=m, t=t, tvp=tvp, f=f,
                      pm=pm, cost_vu=cost_vu)
    return out


def pool(results, h):
    """Stack signed event returns across instruments. Pooled placebo mean per
    iteration = count-weighted mean of per-instrument placebo means.
    tvp = (pooled real mean - pooled placebo expectation) / pooled SE."""
    fs, pms, ns, costs = [], [], [], []
    for r in results:
        d = r[h]
        if d['n'] >= MIN_N and np.isfinite(d['mean']):
            fs.append(d['f'])
            pms.append(d['pm'] * d['n'])
            ns.append(d['n'])
            costs.append(d['cost_vu'] * d['n'])
    if not fs:
        return dict(n=0, mean=np.nan, t=np.nan, tvp=np.nan, p2=np.nan,
                    pgt=np.nan, pm_mean=np.nan, cost_vu=np.nan, pos=0, tot=0)
    F = np.concatenate(fs)
    m = F.mean()
    se = F.std(ddof=1) / np.sqrt(len(F)) + 1e-12
    ntot = sum(ns)
    PM = np.vstack(pms).sum(axis=0) / ntot
    return dict(n=len(F), mean=m, t=m / se,
                tvp=(m - PM.mean()) / se,
                p2=float((np.abs(PM) >= abs(m)).mean()),
                pgt=float((PM >= m).mean()),
                pm_mean=float(PM.mean()),
                cost_vu=sum(costs) / ntot,
                pos=sum(1 for f in fs if f.mean() > 0), tot=len(fs))


def study_block(insts, studies, label):
    """Pooled tables per family + ALL, and per-instrument @ primary."""
    fams = sorted(set(r['fam'] for r in insts))
    groups = ([(f, [s for r, s in zip(insts, studies) if r['fam'] == f])
               for f in fams] if len(fams) > 1 else [])
    groups.append(('ALL', studies))
    print('\n---- %s: pooled signed study (placebo = move-only bars, same '
          'side rule, %d draws) ----' % (label, N_PLACEBO))
    print('%-5s %-3s | %7s %9s %7s %7s %9s %6s %6s %6s %6s' %
          ('fam', 'h', 'n_ev', 'mean_vu', 't', 't_vs_pl', 'plc_mean',
           'p2', 'p_gt', 'e2c', 'sgn+'))
    pooled_all = {}
    for fam, lst in groups:
        for h in HORIZONS:
            d = pool(lst, h)
            if fam == 'ALL':
                pooled_all[h] = d
            e2c = (d['mean'] / d['cost_vu']
                   if d['cost_vu'] and d['cost_vu'] > 0 else np.nan)
            print('%-5s %-3d | %7d %9.4f %7.2f %7.2f %9.4f %6.3f %6.3f '
                  '%6.2f %3d/%-3d' %
                  (fam, h, d['n'], d['mean'], d['t'], d['tvp'], d['pm_mean'],
                   d['p2'], d['pgt'], e2c, d['pos'], d['tot']))
        print()
    return pooled_all


def per_inst_table(insts, studies, label):
    print('---- %s: per-instrument @ primary K=%g h=%d ----' %
          (label, K_PRI, H_PRI))
    print('%-10s %-4s %8s %7s %9s %7s %7s' %
          ('sym', 'fam', 'n_move', 'n_ev', 'mean_vu', 't', 't_vs_pl'))
    pos_cnt = tot_cnt = 0
    for r, s in zip(insts, studies):
        d = s[H_PRI]
        if d['n'] < MIN_N or not np.isfinite(d['mean']):
            print('%-10s %-4s %8d %7d %9s %7s %7s' %
                  (r['name'], r['fam'], d['n_move'], d['n'], 'nan', 'nan',
                   'nan'))
            continue
        print('%-10s %-4s %8d %7d %9.4f %7.2f %7.2f' %
              (r['name'], r['fam'], d['n_move'], d['n'], d['mean'], d['t'],
               d['tvp']))
        tot_cnt += 1
        pos_cnt += d['mean'] > 0
    pct = 100.0 * pos_cnt / max(tot_cnt, 1)
    print('sign consistency @primary: %d/%d positive (%.0f%%)' %
          (pos_cnt, tot_cnt, pct))
    return pos_cnt, tot_cnt, pct


def run_port(insts, variant, K, h, stress=1.0, exo_stress=None,
             subset_fams=None, drop_fams=None, want_frame=False):
    parts, ntr, per = [], 0, {}
    for r in insts:
        if subset_fams is not None and r['fam'] not in subset_fams:
            continue
        if drop_fams is not None and r['fam'] in drop_fams:
            continue
        st = exo_stress if (exo_stress is not None and r['fam'] == 'EXO') \
            else stress
        pnl, n = lab.backtest(r['df'], r['masks'][(variant, K)], h=h,
                              sides=r['sides'], cost=r['cb'] * st, a=r['a'])
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


def shuffle_port(insts, variant, K, h, n_shuffle=N_SHUFFLE, seed=11):
    """Same NUMBER of events per instrument at random times with random +-1
    sides; iteration totals summed across instruments."""
    rng = np.random.default_rng(seed)
    tots = np.zeros(n_shuffle)
    for r in insts:
        N = len(r['df'])
        kk = int(r['masks'][(variant, K)].sum())
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


def main():
    print(__doc__)

    # ================= load core universe =================
    print('=== loading core universe (9 duka REAL volume + 34 fx tick volume) ===')
    insts = []
    for s in lab.DUKA_SYMS:
        insts.append(mk_inst(s, 'CMD', lab.duka(s), lab.cost_bps(s)))
    for s in lab.fx_syms():
        fam = 'MAJ' if s in FX_MAJ else ('EXO' if s in FX_EXO else 'CRX')
        insts.append(mk_inst(s, fam, lab.fx(s), lab.cost_bps(s)))
    print('loaded %d core instruments.' % len(insts))
    print('event counts (raw mask bars) and concentration:')
    print('%-10s %-4s %9s %9s %9s %9s %9s' %
          ('sym', 'fam', 'bars', 'move_ok', 'A_K3', 'A_K5', 'B_K3'))
    for r in insts:
        print('%-10s %-4s %9d %9d %9d %9d %9d' %
              (r['name'], r['fam'], len(r['df']), int(r['move_ok'].sum()),
               int(r['masks'][('A', 3.0)].sum()),
               int(r['masks'][('A', 5.0)].sum()),
               int(r['masks'][('B', 3.0)].sum())))

    # ================= 1. signed event studies =================
    print('\n================ BATTERY 1: SIGNED PLACEBO EVENT STUDY '
          '(core 43) ================')
    print('placebo isolates the VOLUME conditioning: random bars passing '
          '|move|>=%.1f ATR ALONE, same -sign(move) rule.' % MOVE_THR)
    studies = {}
    for variant in ('A', 'B'):
        for K in KS:
            studies[(variant, K)] = [
                signed_study(r, variant, K,
                             seed=1000 + i + int(K) * 37 +
                             (0 if variant == 'A' else 500))
                for i, r in enumerate(insts)]
    pooled = {}
    for K in KS:
        pooled[('A', K)] = study_block(insts, studies[('A', K)],
                                       'VARIANT A, K=%g' % K)
    pos_cnt, tot_cnt, pct_pos_study = per_inst_table(
        insts, studies[('A', K_PRI)], 'VARIANT A (primary K)')
    fam_pos = {f: [0, 0] for f in CORE_FAMS}
    for r, s in zip(insts, studies[('A', K_PRI)]):
        d = s[H_PRI]
        if d['n'] >= MIN_N and np.isfinite(d['mean']):
            fam_pos[r['fam']][1] += 1
            fam_pos[r['fam']][0] += d['mean'] > 0
    for f in CORE_FAMS:
        print('  family %-4s: %d/%d positive' % (f, fam_pos[f][0],
                                                 fam_pos[f][1]))
    print('NOTE: 34 FX pairs share legs (USD/EUR/JPY blocs) -> heavily '
          'cross-correlated; effective independent count far below 43.')
    print('\n-- VARIANT B (climax close) pooled tables (reported '
          'separately) --')
    for K in KS:
        pooled[('B', K)] = study_block(insts, studies[('B', K)],
                                       'VARIANT B, K=%g' % K)
    pe = pooled[('A', K_PRI)][H_PRI]
    e2c_pri = (pe['mean'] / pe['cost_vu']
               if pe['cost_vu'] and pe['cost_vu'] > 0 else np.nan)
    print('PRIMARY CELL (A, K=%g, h=%d): n=%d mean=%.4f t=%.2f '
          't_vs_placebo=%.2f placebo_mean=%.4f p2=%.3f p_gt=%.3f '
          'cost_vu=%.4f e/c=%.2f' %
          (K_PRI, H_PRI, pe['n'], pe['mean'], pe['t'], pe['tvp'],
           pe['pm_mean'], pe['p2'], pe['pgt'], pe['cost_vu'], e2c_pri))
    pe_cmd = pool([s for r, s in zip(insts, studies[('A', K_PRI)])
                   if r['fam'] == 'CMD'], H_PRI)
    print('DUKA-ONLY (real volume) @primary: n=%d mean=%.4f t=%.2f '
          't_vs_placebo=%.2f' %
          (pe_cmd['n'], pe_cmd['mean'], pe_cmd['t'], pe_cmd['tvp']))

    # ================= 2. primary backtest =================
    print('\n================ BATTERY 2: PRIMARY BACKTEST A K=%g h=%d '
          '(x1 costs) ================' % (K_PRI, H_PRI))
    port1, ntr1, per1, frame1 = run_port(insts, 'A', K_PRI, H_PRI,
                                         want_frame=True)
    m1 = lab.metrics(port1)
    th = thirds_sharpe(port1)
    print('portfolio (sum of %d single-vol-unit books), trades=%d' %
          (len(insts), ntr1))
    print(m1)
    print('thirds sharpe: %s' % th)
    print('yearly sharpe:')
    print(lab.yearly(port1))
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
    pct_pos_bt = 100.0 * pos_bt / max(tot_bt, 1)
    print('backtest sign consistency: %d/%d positive total (%.0f%%)' %
          (pos_bt, tot_bt, pct_pos_bt))
    for f in CORE_FAMS:
        print('  family %-4s: %d/%d positive' % (f, fam_bt[f][0],
                                                 fam_bt[f][1]))
    print('\nfamily sub-portfolio metrics @ primary (x1):')
    for f in CORE_FAMS:
        pf, nf, _, _ = run_port(insts, 'A', K_PRI, H_PRI, subset_fams={f})
        print('  %-4s trades=%5d %s' % (f, nf, lab.metrics(pf)))
    portB, ntrB, _, _ = run_port(insts, 'B', K_PRI, H_PRI)
    mB = lab.metrics(portB)
    print('\nVARIANT B portfolio @ primary (x1): trades=%d %s' % (ntrB, mB))

    # ================= 3. cost stress =================
    print('\n================ BATTERY 3: COST STRESS ================')
    port2, _, _, _ = run_port(insts, 'A', K_PRI, H_PRI, stress=2.0)
    m2 = lab.metrics(port2)
    print('x2 ALL           : %s' % m2)
    port_e3, _, _, _ = run_port(insts, 'A', K_PRI, H_PRI, stress=1.0,
                                exo_stress=3.0)
    print('x3 EXOTICS only  : %s   (others x1)' % lab.metrics(port_e3))
    port_dx, ntr_dx, _, _ = run_port(insts, 'A', K_PRI, H_PRI,
                                     drop_fams={'EXO'})
    m_dx = lab.metrics(port_dx)
    print('DROP EXOTICS (x1): %s  trades=%d' % (m_dx, ntr_dx))
    port_cmd, ntr_cmd, _, _ = run_port(insts, 'A', K_PRI, H_PRI,
                                       subset_fams={'CMD'})
    m_cmd = lab.metrics(port_cmd)
    print('COMMODITIES ONLY : %s  trades=%d' % (m_cmd, ntr_cmd))

    # ================= 4. episodicity =================
    print('\n================ BATTERY 4: EPISODICITY ================')
    yrs_port = (port1.index[-1] - port1.index[0]).days / 365.25
    ev_bars = sum(int(r['masks'][('A', K_PRI)].sum()) for r in insts)
    tot_bars = sum(len(r['df']) for r in insts)
    nz = port1.index[port1.values != 0]
    gaps = (nz[1:] - nz[:-1]).days if len(nz) > 1 else np.array([0])
    print('portfolio trades total=%d over %.1fy -> %.0f trades/yr '
          '(all books)' % (ntr1, yrs_port, ntr1 / yrs_port))
    tpy = [per1[r['name']][0] /
           max((r['df'].index[-1] - r['df'].index[0]).days / 365.25, 1e-9)
           for r in insts]
    print('per-instrument trades/yr: min=%.1f median=%.1f max=%.1f' %
          (np.min(tpy), np.median(tpy), np.max(tpy)))
    print('%% bars in event state (pooled): %.3f%% (%d of %d)' %
          (100.0 * ev_bars / tot_bars, ev_bars, tot_bars))
    print('portfolio %% bars with open position (pnl!=0 proxy): %.1f%%' %
          (100.0 * float((port1.values != 0).mean())))
    print('longest flat stretch: %d days' % int(np.max(gaps)))

    # ================= 5. confirm feed: yahoo futures =================
    print('\n================ BATTERY 5: CONFIRM FEED — 25 YAHOO FUTURES '
          'LEGS (2y, real volume, censor-aware) ================')
    yh_insts = [mk_inst('YH_' + s, 'YH', lab.yahoo(s), lab.cost_bps(s))
                for s in lab.yahoo_syms()]
    print('loaded %d yahoo legs.' % len(yh_insts))
    yh_studies = [signed_study(r, 'A', K_PRI, seed=9000 + i)
                  for i, r in enumerate(yh_insts)]
    pooled_yh = study_block(yh_insts, yh_studies, 'YAHOO A, K=%g' % K_PRI)
    per_inst_table(yh_insts, yh_studies, 'YAHOO A (primary K)')
    py = pooled_yh[H_PRI]
    port_yh, ntr_yh, _, _ = run_port(yh_insts, 'A', K_PRI, H_PRI)
    m_yh = lab.metrics(port_yh)
    print('YH mini-portfolio @ primary (x1): trades=%d %s' % (ntr_yh, m_yh))
    print('confirm-feed decision inputs: mean=%.4f t=%.2f (core mean=%.4f)' %
          (py['mean'], py['t'], pe['mean']))

    # ================= 6. shuffle test =================
    print('\n================ BATTERY 6: SHUFFLE TEST (%d shuffles, '
          'portfolio) ================' % N_SHUFFLE)
    tots = shuffle_port(insts, 'A', K_PRI, H_PRI)
    real_tot = float(port1.sum())
    p_shuf = float((tots >= real_tot).mean())
    print('real total=%.1f vu | shuffle mean=%.1f sd=%.1f | '
          'P(shuffle >= real)=%.3f' % (real_tot, tots.mean(), tots.std(),
                                       p_shuf))

    # ================= 7. param grid =================
    print('\n================ BATTERY 7: PARAM GRID (portfolio Sharpe @ x1) '
          '================')
    grid_sh = {}
    print('%-6s | %s' % ('K', ' '.join('h=%-7d' % h for h in HS)))
    for K in KS:
        row = []
        for h in HS:
            pg, ng, _, _ = run_port(insts, 'A', K, h)
            sh = lab.metrics(pg)['sharpe']
            grid_sh[(K, h)] = (sh, ng)
            mark = '*' if (K == K_PRI and h == H_PRI) else ' '
            row.append('%5.2f%s ' % (sh, mark))
        print('%-6g | %s' % (K, ' '.join(row)))
    print('trade counts:')
    for K in KS:
        print('  K=%g: %s' % (K, '  '.join('h=%d:%d' % (h, grid_sh[(K, h)][1])
                                           for h in HS)))

    # ================= figure =================
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.2, 1.5, 2], hspace=0.5)
    eq1, eq2 = port1.cumsum(), port2.cumsum()
    ax = fig.add_subplot(gs[0])
    ax.plot(eq1.index, eq1.values, lw=0.8, color='navy',
            label='x1 costs (Sharpe=%.2f)' % m1['sharpe'])
    ax.plot(eq2.index, eq2.values, lw=0.8, color='gray',
            label='x2 costs (Sharpe=%.2f)' % m2['sharpe'])
    ax.set_title('43-instrument portfolio equity, trades=%d' % ntr1)
    ax.set_ylabel('cum PnL (vol units)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[1])
    dd = eq1 - eq1.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.6)
    ax2.set_ylabel('drawdown (vu, x1)')
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(gs[2])
    ytot = port1.groupby(port1.index.year).sum()
    ax3.bar(ytot.index.astype(str), ytot.values,
            color=['seagreen' if v > 0 else 'firebrick' for v in ytot.values])
    ax3.set_ylabel('yearly PnL (vu, x1)')
    ax3.grid(alpha=0.3, axis='y')
    ax4 = fig.add_subplot(gs[3])
    eqy = port_yh.cumsum()
    eqb = portB.cumsum()
    ax4.plot(eqy.index, eqy.values, lw=0.9, color='teal',
             label='confirm feed: 25 yahoo legs (t@8=%.2f)' % py['t'])
    ax4.plot(eqb.index, eqb.values, lw=0.9, color='darkorange',
             label='variant B climax-close, core 43 (Sharpe=%.2f)'
                   % mB['sharpe'])
    ax4.set_title('confirm feed + variant B (same frozen rule, x1 costs)')
    ax4.set_ylabel('cum PnL (vu)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    fig.suptitle('W3-A VOLUME CLIMAX — vnorm = v/median(prior 30 same-UTC-hour'
                 ' v) >= 3 AND |dC|/ATR >= 1.5; side = -sign(move); entry next'
                 ' open; h=8; one pos/instrument', fontsize=10)
    out_png = lab.ROOT / 'curves' / 'w3_volclimax.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out_png)

    # ================= decision =================
    print('\n================ PRE-REGISTERED DECISION RULE ================')
    same_sign_yh = (np.isfinite(py['mean']) and np.isfinite(pe['mean'])
                    and np.sign(py['mean']) == np.sign(pe['mean']))
    crit = [
        ('pooled t >= 3 vs move-only placebo @ primary',
         np.isfinite(pe['tvp']) and pe['tvp'] >= 3.0,
         't_vs_placebo=%.2f (naive t=%.2f, placebo mean=%.4f)' %
         (pe['tvp'], pe['t'], pe['pm_mean'])),
        ('edge/cost >= 1 @ primary',
         np.isfinite(e2c_pri) and e2c_pri >= 1.0, 'e/c=%.2f' % e2c_pri),
        ('>= 60% instruments same sign @ primary',
         pct_pos_study >= 60.0,
         '%.0f%% (%d/%d)' % (pct_pos_study, pos_cnt, tot_cnt)),
        ('yahoo confirm same-sign t >= 1.5',
         same_sign_yh and np.isfinite(py['t']) and abs(py['t']) >= 1.5,
         'yh mean=%.4f t=%.2f, core mean=%.4f' %
         (py['mean'], py['t'], pe['mean'])),
        ('portfolio Sharpe >= 0.5 @ x1', m1['sharpe'] >= 0.5,
         'sharpe=%.2f' % m1['sharpe']),
        ('portfolio Sharpe > 0 @ x2', m2['sharpe'] > 0,
         'x2 sharpe=%.2f' % m2['sharpe']),
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
    print('POOLED_T=%.2f POOLED_TVP=%.2f POOLED_MEAN=%.4f PLACEBO_MEAN=%.4f '
          'P2=%.3f PGT=%.3f E2C=%.2f' %
          (pe['t'], pe['tvp'], pe['mean'], pe['pm_mean'], pe['p2'], pe['pgt'],
           e2c_pri))
    print('DUKA_T=%.2f DUKA_TVP=%.2f DUKA_MEAN=%.4f' %
          (pe_cmd['t'], pe_cmd['tvp'], pe_cmd['mean']))
    print('PCT_POS_STUDY=%.0f PCT_POS_BT=%.0f AVG_CORR=%.4f' %
          (pct_pos_study, pct_pos_bt, avg_corr))
    print('YH_T=%.2f YH_MEAN=%.4f YH_SHARPE=%.2f' %
          (py['t'], py['mean'], m_yh['sharpe']))
    print('P_SHUFFLE=%.3f REAL_TOT=%.1f' % (p_shuf, real_tot))
    print('DROPEXO_SHARPE=%.2f CMDONLY_SHARPE=%.2f VARB_SHARPE=%.2f '
          'VARB_TRADES=%d' % (m_dx['sharpe'], m_cmd['sharpe'], mB['sharpe'],
                              ntrB))
    print('GRID=%s' % {('%g' % K, h): grid_sh[(K, h)][0]
                       for K in KS for h in HS})
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
