#!/usr/bin/env /usr/bin/python3
"""
W3-C — PDR COMMODITIES-ONLY, REAL-FUTURES CONFIRMATION (wave 3).

Fresh pre-registration of the single defensible wave-2 residue. W2-A killed the
43-instrument PDR (CFD-feed-specific, Sharpe 0.41); its commodities-only
sub-book was the residue (Sharpe 0.51, both halves +). Question: does commodity
PDR exist on REAL FUTURES feeds?

FROZEN RULE (identical to W2-A, commodities only; run EXACTLY, no tuning):
  RV24[t] = std of last 24 hourly close-changes (causal, through t).
  B[t]    = median of RV24 over prior 240 bars, shift(1).
  STATE   : RV24/B >= 3.0.
  EVENT   : every state bar. side = -sign(C[t] - C[t-24]). Entry NEXT OPEN.
  EXIT    : fixed h=16 bars. One position per instrument. No FX.

HYGIENE (binding, wave-2 forensics): exclude Sunday bars from event detection
AND entry on ALL feeds. Yahoo legs censor-aware: 'cens' roll-gap bars excluded
from signals (censored close-changes), entries (event dropped if event bar or
entry bar is censored), AND pnl (lab.backtest zeroes cens-bar steps natively).

FEEDS:
  (a) CONTEXT (in-feed, was part of W2-A's evidence): 9 duka CFDs, 12.3y.
  (b) JUDGMENT (real futures venues, never part of W2-A's verdict):
      TV 60-min commodity exports (~4y, 11 instruments) AND all 25 lab.yahoo
      front-month legs (2y, censor-aware).

BATTERY: signed placebo event study per feed (300 draws, placebo = random valid
times + same -sign(24h move) side rule); per-instrument tables; mini-portfolio
backtests per feed x1/x2 -> ../curves/w3_pdr_cmd.png; halves/thirds; yearly;
param neighborhood k in {2.5,3.5} x h in {8,24} (robustness REPORTING only);
episodicity incl. n events per feed; shuffle 100 on the TV portfolio.

POWER DISCIPLINE: a judgment feed with < 30 de-overlapped events @ h=16 is
UNDERPOWERED -> feeds the unresolved branch, never a pass.

PRE-REGISTERED DECISION RULE (mechanical):
  PROMOTE iff pooled t@16 >= 2 SAME SIGN on BOTH judgment feeds (both powered)
          AND combined judgment-feed portfolio Sharpe >= 0.5 @x1 AND > 0 @x2.
  KILL    if either judgment feed wrong-signed with t <= -1
          OR combined judgment portfolio Sharpe < 0.25 @x1.
  UNRESOLVED otherwise (incl. underpowered) — state what data would settle it.

Run: cd ortho && /usr/bin/python3 w3_pdr_cmd.py
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
NEIGH     = [(2.5, 8), (2.5, 24), (3.5, 8), (3.5, 24)]   # registered corners
HORIZONS  = (2, 4, 8, 16, 24, 48)
N_PLACEBO = 300
N_SHUFFLE = 100
MIN_N     = 10
MIN_POWER = 30   # de-overlapped events @ h=16 per judgment feed

# TV 60-min commodity instruments (registered list; costs bps round-trip)
TV_LIST = [
    ('ICEEUR_DLY_BRN1!, 60 (1).csv', 'TV_BRN',       4.0),
    ('CFI_WTI, 60.csv',              'TV_WTI',       4.0),
    ('COMEX_DL_GC2!, 60.csv',        'TV_GC',        2.0),
    ('COMEX_DL_SI2!, 60.csv',        'TV_SI',        6.0),
    ('COMEX_DL_HG1!, 60.csv',        'TV_HG',        8.0),
    ('NYMEX_DL_PA1!, 60.csv',        'TV_PA',       15.0),
    ('CBOT_DL_ZC1!, 60.csv',         'TV_ZC',        5.0),
    ('CAPITALCOM_COCOA, 60.csv',     'TV_COCOA',    12.0),
    ('FOREXCOM_COFFEE, 60.csv',      'TV_COFFEE',   10.0),
    ('FOREXCOM_COTTON, 60.csv',      'TV_COTTON',   10.0),
    ('TVC_PLATINUM, 60.csv',         'TV_PLATINUM', 10.0),
]


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def mk_inst(name, feed, df, cb):
    """Build signal (cens-aware), apply hygiene exclusions (Sunday everywhere;
    cens on yahoo) to event detection AND entry (event dropped if event bar OR
    next/entry bar is excluded)."""
    N = len(df)
    a = lab.atr(df)
    cens = (df['cens'].astype(bool).values if 'cens' in df.columns
            else np.zeros(N, bool))
    dc = df.c.diff()
    dc = dc.where(~cens, 0.0)                      # signals on censored diffs
    rv = dc.rolling(RV_N).std()
    base = rv.rolling(BASE_N, min_periods=BASE_N // 2).median().shift(1)
    ratio = (rv / base).values
    cc = dc.fillna(0.0).values.cumsum()
    net = np.full(N, np.nan)
    net[RV_N:] = cc[RV_N:] - cc[:-RV_N]

    sun = np.asarray(df.index.dayofweek) == 6
    bad = sun | cens
    bad_evt = bad.copy()
    bad_evt[:-1] = bad_evt[:-1] | bad[1:]          # entry bar (t+1) excluded too

    masks, sides = {}, {}
    for k in KS:
        m = (np.isfinite(ratio) & (ratio >= k) & np.isfinite(net) & (net != 0)
             & ~bad_evt)
        s = np.zeros(N)
        s[m] = -np.sign(net[m])
        masks[k], sides[k] = m, s
    return dict(name=name, feed=feed, df=df, a=a, cb=cb, mask=masks,
                sides=sides, net=net, cens=cens, bad_evt=bad_evt,
                n_sun=int(sun.sum()), n_cens=int(cens.sum()))


def signed_study(inst, horizons=HORIZONS, n_placebo=N_PLACEBO, seed=0):
    """Signed event study at primary k. Forward returns signed by the fade rule;
    SIGNED placebo = same count at random valid (hygiene-clean) times with the
    SAME -sign(24h move) side rule -> tests the dislocation conditioning, not a
    generic 24h fade. Censor-aware via censored cumulative-change series."""
    df, net, cb = inst['df'], inst['net'], inst['cb']
    c, o = df.c.values, df.o.values
    av = inst['a'].values
    cens = inst['cens']
    dc = np.r_[0.0, np.diff(c)]
    dc[cens] = 0.0
    cc = dc.cumsum()

    def fwd(P, h):
        first = np.where(cens[P + 1], 0.0, c[P + 1] - o[P + 1])
        return (first + cc[P + h] - cc[P + 1]) / av[P]

    pos = np.flatnonzero(inst['mask'][K_PRI])
    N = len(df)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(av) & (av > 0) & np.isfinite(net)
                           & (net != 0) & ~inst['bad_evt'])
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


def study_block(insts, label, seed0):
    studies = [signed_study(r, seed=seed0 + i) for i, r in enumerate(insts)]
    print('\n---- %s: pooled signed event study (placebo = random clean times, '
          'same fade-24h side rule, %d draws) ----' % (label, N_PLACEBO))
    print('%-3s | %7s %9s %7s %6s %6s %7s' %
          ('h', 'n_ev', 'mean_vu', 't', 'p_plc', 'e2c', 'sgn+'))
    pooled = {}
    for h in HORIZONS:
        d = pool(studies, h)
        pooled[h] = d
        e2c = d['mean'] / d['cost_vu'] if d['cost_vu'] and d['cost_vu'] > 0 \
            else np.nan
        print('%-3d | %7d %9.4f %7.2f %6.3f %6.2f %4d/%-3d' %
              (h, d['n'], d['mean'], d['t'], d['p'], e2c, d['pos'], d['tot']))
    print('---- %s: per-instrument @ h=%d ----' % (label, H_PRI))
    print('%-13s %6s %9s %7s' % ('sym', 'n_ev', 'mean_vu', 't'))
    pos_cnt = tot_cnt = 0
    for r, s in zip(insts, studies):
        d = s[H_PRI]
        if d['n'] < MIN_N or not np.isfinite(d['mean']):
            print('%-13s %6d %9s %7s' % (r['name'], d['n'], 'nan', 'nan'))
            continue
        print('%-13s %6d %9.4f %7.2f' % (r['name'], d['n'], d['mean'], d['t']))
        tot_cnt += 1
        pos_cnt += d['mean'] > 0
    print('sign consistency @h=%d: %d/%d positive (%.0f%%)' %
          (H_PRI, pos_cnt, tot_cnt, 100.0 * pos_cnt / max(tot_cnt, 1)))
    return studies, pooled


def run_port(insts, k, h, stress=1.0):
    parts, ntr, per = [], 0, {}
    for r in insts:
        pnl, n = lab.backtest(r['df'], r['mask'][k], h=h, sides=r['sides'][k],
                              cost=r['cb'] * stress, a=r['a'])
        parts.append(pnl.rename(r['name']))
        ntr += n
        per[r['name']] = (n, float(pnl.sum()))
    frame = pd.concat(parts, axis=1).fillna(0.0).sort_index()
    return frame.sum(axis=1), ntr, per, frame


def combine(p1, p2):
    f = pd.concat([p1.rename('a'), p2.rename('b')], axis=1).fillna(0.0).sort_index()
    return f.sum(axis=1)


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
    """Per instrument: same NUMBER of event bars at random times with random
    +-1 sides; iteration totals summed across the feed."""
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


def episodicity(insts, port, ntr, per, label):
    yrs = (port.index[-1] - port.index[0]).days / 365.25
    state = sum(int(r['mask'][K_PRI].sum()) for r in insts)
    bars = sum(len(r['df']) for r in insts)
    nz = port.index[port.values != 0]
    gap = int(np.max((nz[1:] - nz[:-1]).days)) if len(nz) > 1 else 0
    tpy = [per[r['name']][0] /
           max((r['df'].index[-1] - r['df'].index[0]).days / 365.25, 1e-9)
           for r in insts]
    print('%s: state bars=%d (%.2f%% of %d) | trades=%d over %.1fy '
          '(%.0f/yr feed-wide; per-inst med %.1f/yr) | in-pos bars %.1f%% | '
          'longest flat %dd' %
          (label, state, 100.0 * state / bars, bars, ntr, yrs, ntr / yrs,
           float(np.median(tpy)), 100.0 * float((port.values != 0).mean()), gap))


def main():
    print(__doc__)

    # ================= load feeds =================
    print('=== loading feeds ===')
    duka_insts = [mk_inst(s, 'DUKA', lab.duka(s), lab.cost_bps(s))
                  for s in lab.DUKA_SYMS]
    tv_insts = []
    for fname, name, cb in TV_LIST:
        df = lab.tv(fname)
        df = df[df.c.notna()].copy()
        tv_insts.append(mk_inst(name, 'TV', df, cb))
    yh_insts = [mk_inst('YH_' + s, 'YH', lab.yahoo(s), lab.cost_bps(s))
                for s in lab.yahoo_syms()]
    for lst, lbl in [(duka_insts, 'DUKA context'), (tv_insts, 'TV judgment'),
                     (yh_insts, 'YH judgment')]:
        print('%s: %d instruments' % (lbl, len(lst)))
        for r in lst:
            print('  %-13s %7d bars  %s -> %s  (sun_excl=%d cens=%d)' %
                  (r['name'], len(r['df']), r['df'].index[0].date(),
                   r['df'].index[-1].date(), r['n_sun'], r['n_cens']))

    # ================= 1. signed event studies =================
    print('\n================ BATTERY 1: SIGNED PLACEBO EVENT STUDIES ============')
    _, pooled_duka = study_block(duka_insts, 'DUKA context (in-feed)', seed0=100)
    _, pooled_tv = study_block(tv_insts, 'TV 60-min judgment', seed0=300)
    _, pooled_yh = study_block(yh_insts, 'YAHOO futures judgment', seed0=600)
    t_duka = pooled_duka[H_PRI]['t']
    t_tv, n_tv_ev = pooled_tv[H_PRI]['t'], pooled_tv[H_PRI]['n']
    t_yh, n_yh_ev = pooled_yh[H_PRI]['t'], pooled_yh[H_PRI]['n']
    pow_tv = n_tv_ev >= MIN_POWER
    pow_yh = n_yh_ev >= MIN_POWER
    print('\ndecision inputs: pooled t@%d  DUKA(ctx)=%.2f  TV=%.2f (n=%d, '
          'powered=%s)  YH=%.2f (n=%d, powered=%s)' %
          (H_PRI, t_duka, t_tv, n_tv_ev, pow_tv, t_yh, n_yh_ev, pow_yh))

    # ================= 2. portfolios x1/x2 =================
    print('\n================ BATTERY 2: MINI-PORTFOLIOS @ k=%.1f h=%d =========='
          % (K_PRI, H_PRI))
    ports = {}
    for lbl, lst in [('DUKA', duka_insts), ('TV', tv_insts), ('YH', yh_insts)]:
        p1, n1, per1, _ = run_port(lst, K_PRI, H_PRI, stress=1.0)
        p2, _, _, _ = run_port(lst, K_PRI, H_PRI, stress=2.0)
        ports[lbl] = dict(x1=p1, x2=p2, ntr=n1, per=per1,
                          m1=lab.metrics(p1), m2=lab.metrics(p2))
        print('%-4s x1: trades=%4d %s' % (lbl, n1, ports[lbl]['m1']))
        print('%-4s x2:             %s' % (lbl, ports[lbl]['m2']))
        print('  per-instrument (x1): ' + '  '.join(
            '%s:%d/%.1f' % (k.replace('YH_', '').replace('TV_', ''), v[0], v[1])
            for k, v in per1.items()))
        pos = sum(1 for v in per1.values() if v[0] >= 1 and v[1] > 0)
        tot = sum(1 for v in per1.values() if v[0] >= 1)
        print('  sign consistency: %d/%d positive total' % (pos, tot))
        print('  yearly sharpe: %s' %
              lab.yearly(p1).to_dict())
        print('  thirds sharpe: %s' % thirds_sharpe(p1))

    comb1 = combine(ports['TV']['x1'], ports['YH']['x1'])
    comb2 = combine(ports['TV']['x2'], ports['YH']['x2'])
    m_c1, m_c2 = lab.metrics(comb1), lab.metrics(comb2)
    ntr_comb = ports['TV']['ntr'] + ports['YH']['ntr']
    print('\nCOMBINED JUDGMENT (TV+YH) x1: trades=%d %s' % (ntr_comb, m_c1))
    print('COMBINED JUDGMENT (TV+YH) x2:           %s' % m_c2)
    print('combined yearly sharpe: %s' % lab.yearly(comb1).to_dict())
    print('combined thirds sharpe: %s' % thirds_sharpe(comb1))

    # ================= 3. episodicity =================
    print('\n================ BATTERY 3: EPISODICITY ================')
    for lbl, lst in [('DUKA', duka_insts), ('TV', tv_insts), ('YH', yh_insts)]:
        episodicity(lst, ports[lbl]['x1'], ports[lbl]['ntr'], ports[lbl]['per'],
                    lbl)
    print('de-overlapped events @h=%d: TV=%d YH=%d (power threshold %d each)' %
          (H_PRI, n_tv_ev, n_yh_ev, MIN_POWER))

    # ================= 4. param neighborhood =================
    print('\n================ BATTERY 4: PARAM NEIGHBORHOOD (reporting only) ====')
    cells = [(K_PRI, H_PRI)] + NEIGH
    print('%-12s | %s' % ('cell', '  '.join('%-10s' % l for l in
                                            ['DUKA', 'TV', 'YH', 'COMB'])))
    grid = {}
    for k, h in cells:
        row = []
        sub = {}
        for lbl, lst in [('DUKA', duka_insts), ('TV', tv_insts),
                         ('YH', yh_insts)]:
            pg, ng, _, _ = run_port(lst, k, h)
            sub[lbl] = (lab.metrics(pg)['sharpe'], ng, pg)
            row.append('%5.2f/%-4d' % (sub[lbl][0], ng))
        pc = combine(sub['TV'][2], sub['YH'][2])
        shc = lab.metrics(pc)['sharpe']
        sub['COMB'] = (shc, sub['TV'][1] + sub['YH'][1], pc)
        row.append('%5.2f/%-4d' % (shc, sub['COMB'][1]))
        grid[(k, h)] = {l: (sub[l][0], sub[l][1]) for l in sub}
        mark = '*' if (k, h) == (K_PRI, H_PRI) else ' '
        print('k=%-3g h=%-3d%s | %s' % (k, h, mark, '  '.join(row)))
    print('(cells show Sharpe/trades @x1; * = primary, frozen)')

    # ================= 5. shuffle test (TV portfolio) =================
    print('\n================ BATTERY 5: SHUFFLE TEST (TV portfolio, %d) ========'
          % N_SHUFFLE)
    tots = shuffle_port(tv_insts, K_PRI, H_PRI)
    real_tv = float(ports['TV']['x1'].sum())
    p_shuf = float((tots >= real_tv).mean())
    print('real TV total=%.1f vu | shuffle mean=%.1f sd=%.1f | '
          'P(shuffle >= real)=%.3f' % (real_tv, tots.mean(), tots.std(), p_shuf))

    # ================= figure =================
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 2.2, 1.2, 1.4], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    for lbl, col in [('DUKA', 'gray'), ('TV', 'darkorange'), ('YH', 'teal')]:
        eq = ports[lbl]['x1'].cumsum()
        tag = 'context' if lbl == 'DUKA' else 'judgment'
        ax.plot(eq.index, eq.values, lw=0.9, color=col,
                label='%s %s (Sh=%.2f, n=%d)' % (lbl, tag,
                                                 ports[lbl]['m1']['sharpe'],
                                                 ports[lbl]['ntr']))
    ax.set_title('per-feed mini-portfolio equity @ x1 costs')
    ax.set_ylabel('cum PnL (vol units)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax1 = fig.add_subplot(gs[1])
    eqc = comb1.cumsum()
    eqc2 = comb2.cumsum()
    ax1.plot(eqc.index, eqc.values, lw=0.9, color='navy',
             label='combined judgment x1 (Sh=%.2f)' % m_c1['sharpe'])
    ax1.plot(eqc2.index, eqc2.values, lw=0.9, color='firebrick', ls='--',
             label='combined judgment x2 (Sh=%.2f)' % m_c2['sharpe'])
    ax1.set_title('COMBINED judgment portfolio (TV + YH)')
    ax1.set_ylabel('cum PnL (vu)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[2])
    dd = eqc - eqc.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.6)
    ax2.set_ylabel('combined DD (vu)')
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(gs[3])
    ytot = comb1.groupby(comb1.index.year).sum()
    ax3.bar(ytot.index.astype(str), ytot.values,
            color=['seagreen' if v > 0 else 'firebrick' for v in ytot.values])
    ax3.set_ylabel('combined yearly PnL (vu)')
    ax3.grid(alpha=0.3, axis='y')
    fig.suptitle('W3-C commodity PDR — RV24/med240 >= 3.0; side=-sign(C-C[-24]);'
                 ' entry next open; h=16; no Sunday events; cens-aware',
                 fontsize=11)
    out_png = lab.ROOT / 'curves' / 'w3_pdr_cmd.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print('\nfigure saved -> %s' % out_png)

    # ================= decision =================
    print('\n================ PRE-REGISTERED DECISION RULE ================')
    print('  TV powered (n_ev=%d >= %d): %s' % (n_tv_ev, MIN_POWER, pow_tv))
    print('  YH powered (n_ev=%d >= %d): %s' % (n_yh_ev, MIN_POWER, pow_yh))
    promote = (pow_tv and pow_yh and t_tv >= 2.0 and t_yh >= 2.0
               and m_c1['sharpe'] >= 0.5 and m_c2['sharpe'] > 0)
    kill = (t_tv <= -1.0) or (t_yh <= -1.0) or (m_c1['sharpe'] < 0.25)
    print('  promote leg: t_tv=%.2f>=2? %s | t_yh=%.2f>=2? %s | '
          'comb x1=%.2f>=0.5? %s | comb x2=%.2f>0? %s' %
          (t_tv, t_tv >= 2, t_yh, t_yh >= 2, m_c1['sharpe'],
           m_c1['sharpe'] >= 0.5, m_c2['sharpe'], m_c2['sharpe'] > 0))
    print('  kill leg: t_tv<=-1? %s | t_yh<=-1? %s | comb x1<0.25? %s' %
          (t_tv <= -1, t_yh <= -1, m_c1['sharpe'] < 0.25))
    decision = 'PROMOTE' if promote else ('KILL' if kill else 'UNRESOLVED')
    print('\nDECISION (mechanical): %s' % decision)

    # ================= machine-readable summary =================
    print('\n===== SUMMARY (machine-readable) =====')
    print('COMB_SHARPE_X1=%.2f H1=%.2f H2=%.2f EQR2=%.2f GAINPAIN=%.2f '
          'NTRADES=%d COMB_SHARPE_X2=%.2f' %
          (m_c1['sharpe'], m_c1['h1'], m_c1['h2'], m_c1['eqR2'],
           m_c1['gain_pain'], ntr_comb, m_c2['sharpe']))
    print('T_DUKA=%.2f T_TV=%.2f T_YH=%.2f N_TV_EV=%d N_YH_EV=%d' %
          (t_duka, t_tv, t_yh, n_tv_ev, n_yh_ev))
    print('TV_SHARPE_X1=%.2f TV_SHARPE_X2=%.2f YH_SHARPE_X1=%.2f '
          'YH_SHARPE_X2=%.2f DUKA_SHARPE_X1=%.2f DUKA_SHARPE_X2=%.2f' %
          (ports['TV']['m1']['sharpe'], ports['TV']['m2']['sharpe'],
           ports['YH']['m1']['sharpe'], ports['YH']['m2']['sharpe'],
           ports['DUKA']['m1']['sharpe'], ports['DUKA']['m2']['sharpe']))
    print('P_SHUFFLE_TV=%.3f REAL_TV_TOT=%.1f' % (p_shuf, real_tv))
    print('GRID=%s' % {('k%g_h%d' % c): grid[c]['COMB'][0] for c in grid})
    print('DECISION=%s' % decision)


if __name__ == '__main__':
    main()
