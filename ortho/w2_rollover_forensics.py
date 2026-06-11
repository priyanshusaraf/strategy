#!/usr/bin/env /usr/bin/python3
"""
W2-C — 20-22 UTC FX ANOMALY FORENSICS (artifact vs real).

Wave-1 H8 found a fade edge at 20-22 UTC in FX (|1-bar move| >= 1 ATR, fade,
t up to 31). But 21-22 UTC is the FX rollover / 5pm-ET window where spreads
widen 5-20x and Dukascopy candles are BID quotes. This script is FORENSIC:
it does not build a strategy, it determines whether the measured edge is a
quote/spread artifact or genuine multi-hour reversion.

FROZEN RULE (pre-registered, from wave-1 H8; characterization horizon h=2):
  event: UTC hour(t) in {20,21,22} AND |C[t]-C[t-1]| / ATR100 >= 1.0
  side : -sign(C[t]-C[t-1])  (fade);  entry next bar open; h in {1,2,4}.

MANDATORY TESTS
  1. Baseline replication on Dukascopy (6 representative pairs + pooled all-34,
     per-hour stats, signed placebo, time-clustered t).
  2. Delayed entry (open[t+2], open[t+3], same hold length) -> survival
     fraction; plus bar-by-bar decomposition of the h=4 fade return.
  3. Cross-feed: TV 60-min OANDA_EURUSD / OANDA_USDJPY / FOREXCOM_USDCHF /
     TVC_DXY vs the same pairs on Dukascopy in the same window.
  4. Fine structure: 15-min OANDA/FOREXCOM files — within-window timing.
  5. Bar integrity by hour: (H-L)/ATR, |O[t+1]-C[t]|/ATR, volume, bar counts.
  6. Break-even round-trip spread (bps) per family/hour vs realistic
     rollover-window spreads (majors ~2-10bps, crosses/exotics 10-50bps).
  7. Year-by-year stability of what survives 2-3.

PRE-REGISTERED DECISION RULE (mechanical):
  REAL requires ALL of:
    (a) 1-bar-delayed entry keeps >= 50% of baseline edge (pooled all-34, h=2);
    (b) same-sign t >= 2 on the OANDA/FOREXCOM feed (pooled 3 FX TV symbols, h=2);
    (c) implied break-even spread comfortably above realistic rollover spreads.
  Anything else = ARTIFACT (kill) or UNRESOLVED.

Run: cd ortho && /usr/bin/python3 w2_rollover_forensics.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lab

HOURS     = (20, 21, 22)
THR       = 1.0
HORIZONS  = (1, 2, 4)
H_CHAR    = 2          # characterization horizon (fixed ex-ante)
DELAYS    = (0, 1, 2)  # entry at open[t+1+d]
N_PLACEBO = 300
MIN_N     = 10

REP6   = ['EURUSD', 'USDJPY', 'EURSEK', 'EURNOK', 'GBPNZD', 'NZDCHF']
EXOT   = ['EURNOK', 'EURSEK', 'USDNOK', 'USDSEK', 'EURPLN', 'EURHUF',
          'EURCZK', 'AUDSGD']
MAJ    = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD',
          'NZDUSD', 'EURGBP', 'EURJPY', 'EURCHF']
TV_SYMS = [('OANDA_EURUSD',   'OANDA_EURUSD, 60 (1).csv', 1.5, 'EURUSD'),
           ('OANDA_USDJPY',   'OANDA_USDJPY, 60.csv',     1.5, 'USDJPY'),
           ('FOREXCOM_USDCHF','FOREXCOM_USDCHF, 60.csv',  1.5, 'USDCHF'),
           ('TVC_DXY',        'TVC_DXY, 60.csv',          2.0, None)]
TV15 = [('OANDA_EURUSD',   'OANDA_EURUSD, 15 (2).csv'),
        ('OANDA_USDJPY',   'OANDA_USDJPY, 15.csv'),
        ('FOREXCOM_USDCHF','FOREXCOM_USDCHF, 15.csv')]


def fam_of(sym):
    return 'MAJ' if sym in MAJ else ('EXO' if sym in EXOT else 'CRX')


def _deoverlap(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def prep(df):
    """Per-instrument arrays: close, open, atr, signed move (vu), hour."""
    a = lab.atr(df)
    av = a.values
    c, o = df.c.values, df.o.values
    mv = np.r_[np.nan, np.diff(c)] / av
    ok = np.isfinite(mv) & (mv != 0) & np.isfinite(av) & (av > 0)
    return dict(df=df, a=a, av=av, c=c, o=o, mv=mv, ok=ok,
                hr=df.index.hour.values, ts=df.index)


def block_pos(P, hours, thr, h, d_max=0):
    """Event positions for the hour-block rule, de-overlapped at gap h,
    with room for delay d_max and horizon h."""
    N = len(P['c'])
    raw = P['ok'] & (np.abs(P['mv']) >= thr) & np.isin(P['hr'], hours)
    pos = _deoverlap(np.flatnonzero(raw), h)
    return pos[pos + d_max + h < N - 1]


def fade_fwd(P, pos, h, d=0):
    """Signed fade fwd return (vu): s*(c[p+d+h]-o[p+d+1])/atr[p];
    also per-event break-even bps and entry timestamps."""
    s = -np.sign(P['mv'][pos])
    f = s * (P['c'][pos + d + h] - P['o'][pos + d + 1]) / P['av'][pos]
    fin = np.isfinite(f)
    pos, f, s = pos[fin], f[fin], s[fin]
    bps = f * P['av'][pos] / np.abs(P['o'][pos + d + 1]) * 1e4
    return pos, f, bps, s


def mt(F):
    F = np.asarray(F, float)
    F = F[np.isfinite(F)]
    if len(F) < MIN_N:
        return len(F), np.nan, np.nan
    m = F.mean()
    return len(F), m, m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)


def clustered_t(ts_list, f_list):
    """t over per-timestamp means (events across pairs at the same hour are
    one cluster -> kills the shared-leg/simultaneous-rollover inflation)."""
    s = pd.Series(np.concatenate(f_list),
                  index=pd.DatetimeIndex(np.concatenate(
                      [np.asarray(t) for t in ts_list])))
    g = s.groupby(s.index).mean()
    n, m, t = mt(g.values)
    return n, m, t


def signed_placebo(P, hours, thr, h, n_real, rng):
    """Placebo means: n_real events drawn from the instrument's OWN big-move
    fade universe across ALL hours (same side rule) -> tests the HOUR
    conditioning beyond generic fade-after-big-move. Returns (pm, n_real)."""
    N = len(P['c'])
    raw = P['ok'] & (np.abs(P['mv']) >= thr)
    uni = np.flatnonzero(raw)
    uni = uni[uni + h < N - 1]
    s = -np.sign(P['mv'][uni])
    f_all = s * (P['c'][uni + h] - P['o'][uni + 1]) / P['av'][uni]
    f_all = f_all[np.isfinite(f_all)]
    if len(f_all) < MIN_N or n_real < 1:
        return None
    draws = rng.integers(0, len(f_all), size=(N_PLACEBO, n_real))
    return f_all[draws].mean(axis=1)


def main():
    print(__doc__)
    syms = lab.fx_syms()
    print('loading %d duka FX pairs ...' % len(syms))
    D = {s: prep(lab.fx(s)) for s in syms}
    print('loaded. windows: %s -> %s' %
          (D['EURUSD']['ts'][0], D['EURUSD']['ts'][-1]))

    # =================================================================
    # TEST 1 — BASELINE REPLICATION (Dukascopy)
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 1 — BASELINE REPLICATION on Dukascopy: fade |move|>=1ATR at 20-22 UTC')
    print('=' * 78)
    print('\nper-hour, representative pairs (n / mean_vu / t):')
    hdr = '%-8s hr |' % 'sym' + ' |'.join(' h=%d %5s %7s %6s' % (h, 'n', 'mean', 't')
                                          for h in HORIZONS)
    print(hdr)
    for sym in REP6:
        P = D[sym]
        for H in HOURS:
            parts = []
            for h in HORIZONS:
                pos = block_pos(P, (H,), THR, h)
                _, f, _, _ = fade_fwd(P, pos, h)
                n, m, t = mt(f)
                parts.append(' h=%d %5d %7.3f %6.2f' % (h, n, m, t))
            print('%-8s %2d |%s' % (sym, H, ' |'.join(parts)))
        parts = []
        for h in HORIZONS:
            pos = block_pos(P, HOURS, THR, h)
            _, f, _, _ = fade_fwd(P, pos, h)
            n, m, t = mt(f)
            parts.append(' h=%d %5d %7.3f %6.2f' % (h, n, m, t))
        print('%-8s blk|%s' % (sym, ' |'.join(parts)))

    print('\npooled ALL-34, per hour and block (naive t / time-clustered t / placebo p):')
    rng = np.random.default_rng(7)
    base_pool = {}
    for grp_name, grp_hours in [('20', (20,)), ('21', (21,)), ('22', (22,)),
                                ('block 20-22', HOURS)]:
        for h in HORIZONS:
            fs, tss, pms, ns, bpss = [], [], [], [], []
            for i, sym in enumerate(syms):
                P = D[sym]
                pos = block_pos(P, grp_hours, THR, h)
                pos, f, bps, _ = fade_fwd(P, pos, h)
                if len(f) < MIN_N:
                    continue
                fs.append(f)
                bpss.append(bps)
                tss.append(P['ts'][pos + 1])
                pm = signed_placebo(P, grp_hours, THR, h, len(f),
                                    np.random.default_rng(100 + i))
                if pm is not None:
                    pms.append(pm * len(f))
                    ns.append(len(f))
            F = np.concatenate(fs)
            n, m, t = mt(F)
            nc, mc, tc = clustered_t(tss, fs)
            PM = np.vstack(pms).sum(axis=0) / sum(ns)
            p = float((np.abs(PM) >= abs(m)).mean())
            pospairs = sum(1 for f in fs if f.mean() > 0)
            base_pool[(grp_name, h)] = dict(n=n, m=m, t=t, tc=tc, p=p,
                                            bps=np.concatenate(bpss),
                                            pos=pospairs, tot=len(fs))
            print('  %-12s h=%d: n=%6d mean=%7.4f t=%6.2f | clusters=%5d '
                  'clust_t=%5.2f | placebo_p=%5.3f | pairs+ %d/%d' %
                  (grp_name, h, n, m, t, nc, tc, p, pospairs, len(fs)))

    print('\nNOTE: 34 pairs share legs and roll simultaneously -> naive pooled t is')
    print('inflated; the time-clustered t is the honest one.')

    # day-of-week mix of block events (context: Fri 21-22 fills at Sunday open)
    wd_cnt = np.zeros(7, int)
    for sym in syms:
        P = D[sym]
        pos = block_pos(P, HOURS, THR, H_CHAR)
        wd = P['ts'][pos].dayofweek
        for w in range(7):
            wd_cnt[w] += int((wd == w).sum())
    print('event day-of-week mix (Mon..Sun): %s' % wd_cnt.tolist())

    # =================================================================
    # TEST 2 — DELAYED ENTRY (Dukascopy)
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 2 — DELAYED ENTRY: enter open[t+1+d], same hold length h')
    print('=' * 78)
    surv = {}
    for h in HORIZONS:
        print('\nh=%d  (pooled all-34, identical event set across delays):' % h)
        res_d = {}
        for d in DELAYS:
            fs, tss = [], []
            fams = {'MAJ': [], 'CRX': [], 'EXO': []}
            for sym in syms:
                P = D[sym]
                pos = block_pos(P, HOURS, THR, h, d_max=max(DELAYS))
                posf, f, _, _ = fade_fwd(P, pos, h, d=d)
                if len(f) < MIN_N:
                    continue
                fs.append(f)
                tss.append(P['ts'][posf + 1 + d])
                fams[fam_of(sym)].append(f)
            F = np.concatenate(fs)
            n, m, t = mt(F)
            _, _, tc = clustered_t(tss, fs)
            res_d[d] = m
            fam_str = '  '.join('%s m=%7.4f t=%5.2f' % ((k,) + mt(np.concatenate(v))[1:])
                                for k, v in fams.items() if v)
            print('  d=%d (entry o[t+%d]): n=%6d mean=%7.4f t=%6.2f clust_t=%5.2f | %s'
                  % (d, d + 1, n, m, t, tc, fam_str))
        for d in (1, 2):
            sf = res_d[d] / res_d[0] if res_d[0] else np.nan
            surv[(h, d)] = sf
            print('  survival fraction d=%d: %.1f%%' % (d, 100 * sf))

    # bar-by-bar decomposition of the h=4 fade return (where does the PnL sit?)
    print('\nbar-by-bar decomposition (pooled all-34, block events, signed by fade side):')
    comps = {k: [] for k in ['gap c[t]->o[t+1]', 'bar o[t+1]->c[t+1]',
                             'bar c[t+1]->c[t+2]', 'bars c[t+2]->c[t+4]']}
    for sym in syms:
        P = D[sym]
        pos = block_pos(P, HOURS, THR, 4)
        if len(pos) < MIN_N:
            continue
        s = -np.sign(P['mv'][pos])
        av = P['av'][pos]
        c, o = P['c'], P['o']
        comps['gap c[t]->o[t+1]'].append(s * (o[pos + 1] - c[pos]) / av)
        comps['bar o[t+1]->c[t+1]'].append(s * (c[pos + 1] - o[pos + 1]) / av)
        comps['bar c[t+1]->c[t+2]'].append(s * (c[pos + 2] - c[pos + 1]) / av)
        comps['bars c[t+2]->c[t+4]'].append(s * (c[pos + 4] - c[pos + 2]) / av)
    for k, v in comps.items():
        n, m, t = mt(np.concatenate(v))
        print('  %-20s: n=%6d mean=%8.4f t=%6.2f' % (k, n, m, t))

    # long/short asymmetry: bid-quote artifact signature = profit concentrated
    # on the LONG side (buying after down moves measured on a depressed bid)
    print('\nfade-side asymmetry at h=%d (bid-candle artifact -> LONG side carries it):' % H_CHAR)
    for h in (1, H_CHAR):
        fl, fsh = [], []
        for sym in syms:
            P = D[sym]
            pos = block_pos(P, HOURS, THR, h)
            pos, f, _, s = fade_fwd(P, pos, h)
            fl.append(f[s > 0])
            fsh.append(f[s < 0])
        nl, ml, tl = mt(np.concatenate(fl))
        ns_, ms_, ts_ = mt(np.concatenate(fsh))
        print('  h=%d LONG  (fade down-move): n=%6d mean=%7.4f t=%6.2f' % (h, nl, ml, tl))
        print('  h=%d SHORT (fade up-move)  : n=%6d mean=%7.4f t=%6.2f' % (h, ns_, ms_, ts_))

    # =================================================================
    # TEST 3 — CROSS-FEED (TV 60-min, different liquidity pools)
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 3 — CROSS-FEED: TV 60-min (OANDA/FOREXCOM/TVC) vs Dukascopy, same window')
    print('=' * 78)
    TVD = {}
    for name, fname, cb, duk in TV_SYMS:
        TVD[name] = prep(lab.tv(fname))
    t0 = max(TVD[n]['ts'][0] for n in TVD)
    t1 = min(TVD[n]['ts'][-1] for n in TVD)
    print('TV window: %s -> %s' % (t0, t1))

    tv_pool = {h: ([], []) for h in HORIZONS}        # 3 FX TV symbols only
    tv_pool_d1 = {h: ([], []) for h in HORIZONS}
    print('\n%-16s %-10s |' % ('feed/sym', 'h') + '  d=0 (o[t+1])      d=1 (o[t+2])')
    for name, fname, cb, duk in TV_SYMS:
        P = TVD[name]
        for h in HORIZONS:
            pos = block_pos(P, HOURS, THR, h, d_max=1)
            p0, f0, _, _ = fade_fwd(P, pos, h, d=0)
            p1, f1, _, _ = fade_fwd(P, pos, h, d=1)
            n0, m0, tt0 = mt(f0)
            n1, m1, tt1 = mt(f1)
            print('%-16s h=%d | n=%4d m=%7.4f t=%5.2f | n=%4d m=%7.4f t=%5.2f' %
                  (name, h, n0, m0, tt0, n1, m1, tt1))
            if duk is not None and len(f0) >= 1:
                tv_pool[h][0].append(f0)
                tv_pool[h][1].append(P['ts'][p0 + 1])
                tv_pool_d1[h][0].append(f1)
                tv_pool_d1[h][1].append(P['ts'][p1 + 2])
        # same pair on duka, same window
        if duk is not None:
            dfw = D[duk]['df']
            Pw = prep(dfw[(dfw.index >= t0) & (dfw.index <= t1)])
            for h in HORIZONS:
                pos = block_pos(Pw, HOURS, THR, h, d_max=1)
                _, f0, _, _ = fade_fwd(Pw, pos, h, d=0)
                _, f1, _, _ = fade_fwd(Pw, pos, h, d=1)
                n0, m0, tt0 = mt(f0)
                n1, m1, tt1 = mt(f1)
                print('%-16s h=%d | n=%4d m=%7.4f t=%5.2f | n=%4d m=%7.4f t=%5.2f' %
                      ('  duka ' + duk, h, n0, m0, tt0, n1, m1, tt1))

    print('\npooled TV FX (OANDA_EURUSD + OANDA_USDJPY + FOREXCOM_USDCHF):')
    tv_t_char = {}
    for h in HORIZONS:
        F = np.concatenate(tv_pool[h][0])
        n, m, t = mt(F)
        _, _, tc = clustered_t(tv_pool[h][1], tv_pool[h][0])
        F1 = np.concatenate(tv_pool_d1[h][0])
        n1, m1, t1_ = mt(F1)
        tv_t_char[h] = (n, m, t, tc, m1, t1_)
        print('  h=%d: d=0 n=%5d mean=%7.4f t=%5.2f clust_t=%5.2f | d=1 mean=%7.4f t=%5.2f'
              % (h, n, m, t, tc, m1, t1_))
    # duka same-3-pairs, same window pooled (apples to apples)
    print('pooled duka same 3 pairs, TV window:')
    duka3_char = {}
    for h in HORIZONS:
        fs, tss = [], []
        for duk in ['EURUSD', 'USDJPY', 'USDCHF']:
            dfw = D[duk]['df']
            Pw = prep(dfw[(dfw.index >= t0) & (dfw.index <= t1)])
            pos = block_pos(Pw, HOURS, THR, h)
            posf, f, _, _ = fade_fwd(Pw, pos, h)
            fs.append(f)
            tss.append(Pw['ts'][posf + 1])
        F = np.concatenate(fs)
        n, m, t = mt(F)
        _, _, tc = clustered_t(tss, fs)
        duka3_char[h] = (n, m, t, tc)
        print('  h=%d: n=%5d mean=%7.4f t=%5.2f clust_t=%5.2f' % (h, n, m, t, tc))

    # =================================================================
    # TEST 4 — FINE STRUCTURE (15-min, ~10 months)
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 4 — FINE STRUCTURE: 15-min path after hourly 20-22 UTC events')
    print('=' * 78)
    steps = [1, 2, 3, 4, 6, 8, 12, 16]   # 15-min closes -> 15m..4h
    paths, gaps = [], []
    for name, fname in TV15:
        d15 = lab.tv(fname)
        hh = d15.resample('1h').agg({'o': 'first', 'h': 'max',
                                     'l': 'min', 'c': 'last'}).dropna()
        Ph = prep(hh)
        pos = block_pos(Ph, HOURS, THR, max(steps) // 4 + 1)
        idx15 = d15.index
        c15, o15 = d15.c.values, d15.o.values
        for p in pos:
            ev_close_time = Ph['ts'][p] + pd.Timedelta(hours=1)
            e = idx15.searchsorted(ev_close_time)
            if e + max(steps) >= len(d15):
                continue
            if (idx15[e] - ev_close_time) > pd.Timedelta(minutes=45):
                continue
            s = -np.sign(Ph['mv'][p])
            a = Ph['av'][p]
            entry = o15[e]
            gaps.append(s * (entry - Ph['c'][p]) / a)
            paths.append([s * (c15[e + k - 1] - entry) / a for k in steps])
    paths = np.asarray(paths)
    print('events: %d (3 symbols pooled, ~10 months of 15-min data)' % len(paths))
    if len(paths) >= MIN_N:
        n, mg, tg = mt(np.asarray(gaps))
        print('pre-entry gap eventclose->15m-open (uncapturable): mean=%.4f t=%.2f' % (mg, tg))
        print('cumulative fade pnl from 15-min entry (vu):')
        for j, k in enumerate(steps):
            n, m, t = mt(paths[:, j])
            print('  +%3d min: n=%4d mean=%7.4f t=%6.2f' % (k * 15, n, m, t))
    else:
        print('insufficient events for fine structure.')

    # =================================================================
    # TEST 5 — BAR INTEGRITY BY HOUR (Dukascopy, all 34 pooled)
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 5 — BAR INTEGRITY by UTC hour (duka, 34 pairs pooled)')
    print('=' * 78)
    rng_hl = {H: [] for H in range(24)}
    rng_gap = {H: [] for H in range(24)}
    rng_vol = {H: [] for H in range(24)}
    for sym in syms:
        P = D[sym]
        df = P['df']
        hl = (df.h.values - df.l.values) / P['av']
        gap = np.abs(np.r_[P['o'][1:], np.nan] - P['c']) / P['av']
        vmed = np.nanmedian(df.v.values) + 1e-12
        vrel = df.v.values / vmed
        for H in range(24):
            sel = (P['hr'] == H) & np.isfinite(hl)
            rng_hl[H].append(hl[sel])
            sel2 = (P['hr'] == H) & np.isfinite(gap)
            rng_gap[H].append(gap[sel2])
            rng_vol[H].append(vrel[(P['hr'] == H) & np.isfinite(vrel)])
    print('hr | med(H-L)/ATR p90 | med|O[t+1]-C|/ATR p90 | med rel.volume')
    integ = {}
    for H in range(24):
        a_ = np.concatenate(rng_hl[H])
        g_ = np.concatenate(rng_gap[H])
        v_ = np.concatenate(rng_vol[H])
        integ[H] = (np.median(a_), np.percentile(a_, 90),
                    np.median(g_), np.percentile(g_, 90), np.median(v_))
        flag = '  <-- rollover' if H in HOURS else ''
        print('%2d |   %5.2f  %5.2f   |    %5.3f   %5.3f     |   %5.2f%s' %
              ((H,) + integ[H] + (flag,)))

    # =================================================================
    # TEST 6 — BREAK-EVEN SPREAD
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 6 — BREAK-EVEN ROUND-TRIP SPREAD (bps) = per-event edge in bps')
    print('=' * 78)
    print('realistic 5pm-ET rollover-window spreads: majors ~2-10, crosses/exotics 10-50')
    be = {}
    for h in (1, H_CHAR):
        for d in (0, 1):
            fams = {'MAJ': [], 'CRX': [], 'EXO': []}
            for sym in syms:
                P = D[sym]
                pos = block_pos(P, HOURS, THR, h, d_max=1)
                _, _, bps, _ = fade_fwd(P, pos, h, d=d)
                fams[fam_of(sym)].append(bps)
            row = []
            for k in ['MAJ', 'CRX', 'EXO']:
                B = np.concatenate(fams[k])
                n, m, t = mt(B)
                be[(h, d, k)] = m
                row.append('%s %6.2f bps (t=%5.2f)' % (k, m, t))
            print('h=%d d=%d: %s' % (h, d, ' | '.join(row)))
    print('\nper-hour break-even (h=%d, d=0), majors only:' % H_CHAR)
    for H in HOURS:
        bs = []
        for sym in MAJ:
            P = D[sym]
            pos = block_pos(P, (H,), THR, H_CHAR)
            _, _, bps, _ = fade_fwd(P, pos, H_CHAR)
            bs.append(bps)
        B = np.concatenate(bs)
        n, m, t = mt(B)
        print('  hour %d: n=%5d break-even=%6.2f bps (t=%5.2f)' % (H, n, m, t))

    # =================================================================
    # TEST 7 — YEAR-BY-YEAR of what survives tests 2-3
    # =================================================================
    print('\n' + '=' * 78)
    print('TEST 7 — YEAR-BY-YEAR (h=%d): baseline d=0 and delayed d=1, duka all-34; TV pooled' % H_CHAR)
    print('=' * 78)
    rows = {}
    for d in (0, 1):
        fs, tss = [], []
        for sym in syms:
            P = D[sym]
            pos = block_pos(P, HOURS, THR, H_CHAR, d_max=1)
            posf, f, _, _ = fade_fwd(P, pos, H_CHAR, d=d)
            fs.append(f)
            tss.append(P['ts'][posf])
        s = pd.Series(np.concatenate(fs),
                      index=pd.DatetimeIndex(np.concatenate([np.asarray(t) for t in tss])))
        rows[('duka', d)] = s
    for d in (0, 1):
        key = tv_pool if d == 0 else tv_pool_d1
        s = pd.Series(np.concatenate(key[H_CHAR][0]),
                      index=pd.DatetimeIndex(np.concatenate(
                          [np.asarray(t) for t in key[H_CHAR][1]])))
        rows[('tv', d)] = s
    for (feed, d), s in rows.items():
        print('\n%s d=%d, per year (n / mean / t):' % (feed, d))
        for y, g in s.groupby(s.index.year):
            n, m, t = mt(g.values)
            print('  %d: n=%5d mean=%7.4f t=%6.2f' % (y, n, m, t))

    # =================================================================
    # CHARACTERIZATION BACKTEST (headline numbers only — NOT a strategy)
    # =================================================================
    print('\n' + '=' * 78)
    print('CHARACTERIZATION BACKTEST: frozen rule, h=%d, all-34 duka, per-sym costs' % H_CHAR)
    print('(this quantifies what the artifact LOOKS like if naively backtested)')
    print('=' * 78)
    parts, parts2x, parts_d1 = [], [], []
    n_tr = 0
    for sym in syms:
        P = D[sym]
        df = P['df']
        N = len(df)
        pos = block_pos(P, HOURS, THR, H_CHAR, d_max=1)
        mask = np.zeros(N, bool)
        mask[pos] = True
        sides = np.zeros(N)
        sides[pos] = -np.sign(P['mv'][pos])
        cb = lab.cost_bps(sym)
        stress = 3.0 if sym in EXOT else 2.0
        pnl, n = lab.backtest(df, mask, h=H_CHAR, sides=sides, cost=cb, a=P['a'])
        pnl2, _ = lab.backtest(df, mask, h=H_CHAR, sides=sides,
                               cost=lab.cost_bps(sym, stress), a=P['a'])
        # delayed entry d=1: shift events one bar
        mask1 = np.zeros(N, bool)
        mask1[pos + 1] = True
        sides1 = np.zeros(N)
        sides1[pos + 1] = sides[pos]
        pnl_d1, _ = lab.backtest(df, mask1, h=H_CHAR, sides=sides1, cost=cb, a=P['a'])
        parts.append(pnl.rename(sym))
        parts2x.append(pnl2)
        parts_d1.append(pnl_d1)
        n_tr += n
    M = pd.concat(parts, axis=1).fillna(0.0).sort_index()
    port = M.sum(axis=1)
    port2x = pd.concat(parts2x, axis=1).fillna(0.0).sum(axis=1).sort_index()
    port_d1 = pd.concat(parts_d1, axis=1).fillna(0.0).sum(axis=1).sort_index()
    met = lab.metrics(port)
    met2 = lab.metrics(port2x)
    metd = lab.metrics(port_d1)
    print('1x costs : %s  trades=%d' % (met, n_tr))
    print('stress (x2 normal / x3 exotic): %s' % met2)
    print('delayed-entry d=1, 1x costs   : %s' % metd)
    print('yearly sharpe (1x):')
    print(lab.yearly(port))
    # avg pairwise correlation on days with activity
    act = M.loc[(M != 0).any(axis=1)]
    C = act.corr().values
    apc = float(np.nanmean(C[np.triu_indices_from(C, 1)]))
    print('avg pairwise corr of instrument pnl (active bars): %.3f' % apc)

    # shuffle test, 6 representative pairs
    print('\nshuffle test (h=%d, 100 shuffles, same event count random times+signs):' % H_CHAR)
    for sym in REP6:
        P = D[sym]
        N = len(P['df'])
        pos = block_pos(P, HOURS, THR, H_CHAR)
        mask = np.zeros(N, bool)
        mask[pos] = True
        sides = np.zeros(N)
        sides[pos] = -np.sign(P['mv'][pos])
        rt, pf = lab.shuffle_test(P['df'], mask, h=H_CHAR, sides=sides,
                                  cost=lab.cost_bps(sym))
        print('  %-8s real_total=%8.1f vu  frac_shuffles>=real=%.2f' % (sym, rt, pf))

    # =================================================================
    # FIGURE
    # =================================================================
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('W2-C forensics: fade |move|>=1ATR at 20-22 UTC, FX (frozen rule, h=2)')
    a0 = ax[0, 0]
    a0.plot(port.index, port.cumsum().values, label='entry o[t+1] (1x costs)', lw=1)
    a0.plot(port_d1.index, port_d1.cumsum().values,
            label='entry o[t+2] (1-bar delay)', lw=1)
    a0.set_title('duka all-34 cumulative equity (vol units)')
    a0.set_xlabel('date'); a0.set_ylabel('vu'); a0.legend(); a0.grid(alpha=.3)
    a1 = ax[0, 1]
    if len(paths) >= MIN_N:
        mins = [k * 15 for k in steps]
        mp = paths.mean(axis=0)
        se = paths.std(axis=0, ddof=1) / np.sqrt(len(paths))
        a1.errorbar(mins, mp, yerr=2 * se, marker='o')
        a1.axhline(0, color='k', lw=.5)
    a1.set_title('15-min fade path after event (3 TV pairs, n=%d)' % len(paths))
    a1.set_xlabel('minutes after entry'); a1.set_ylabel('cum fade pnl (vu)')
    a1.grid(alpha=.3)
    a2 = ax[1, 0]
    hrs = list(range(24))
    a2.bar(hrs, [integ[H][0] for H in hrs], color=['tomato' if H in HOURS
                                                   else 'steelblue' for H in hrs])
    a2.set_title('median (H-L)/ATR by UTC hour (34 duka pairs)')
    a2.set_xlabel('UTC hour'); a2.set_ylabel('median (H-L)/ATR'); a2.grid(alpha=.3)
    a3 = ax[1, 1]
    labels, vals0, vals1 = [], [], []
    for h in HORIZONS:
        labels.append('TV h=%d' % h)
        vals0.append(tv_t_char[h][1])
        vals1.append(duka3_char[h][1])
    x = np.arange(len(labels))
    a3.bar(x - .18, vals1, .36, label='duka same 3 pairs, same window')
    a3.bar(x + .18, vals0, .36, label='TV OANDA/FOREXCOM')
    a3.axhline(0, color='k', lw=.5)
    a3.set_xticks(x); a3.set_xticklabels(labels)
    a3.set_title('cross-feed mean fade pnl (vu), 2023-2026')
    a3.set_ylabel('mean vu'); a3.legend(); a3.grid(alpha=.3)
    fig.tight_layout()
    out = lab.ROOT / 'curves' / 'w2_rollover.png'
    fig.savefig(out, dpi=110)
    print('\nfigure saved: %s' % out)

    # =================================================================
    # MECHANICAL DECISION
    # =================================================================
    print('\n' + '=' * 78)
    print('PRE-REGISTERED DECISION RULE (h=%d)' % H_CHAR)
    print('=' * 78)
    sf = surv[(H_CHAR, 1)]
    tvt = tv_t_char[H_CHAR][2]
    tvm = tv_t_char[H_CHAR][1]
    base_m = base_pool[('block 20-22', H_CHAR)]['m']
    cond_a = np.isfinite(sf) and sf >= 0.5
    cond_b = np.isfinite(tvt) and tvt >= 2.0 and np.sign(tvm) == np.sign(base_m)
    be_maj = be[(H_CHAR, 0, 'MAJ')]
    be_exo = be[(H_CHAR, 0, 'EXO')]
    cond_c = be_maj > 10 and be_exo > 50   # "comfortably above" upper realistic spread
    print('(a) delayed-entry survival >= 50%%: %.1f%% -> %s' %
          (100 * sf, 'PASS' if cond_a else 'FAIL'))
    print('(b) TV-feed same-sign t >= 2: t=%.2f mean=%.4f -> %s' %
          (tvt, tvm, 'PASS' if cond_b else 'FAIL'))
    print('(c) break-even spread comfortably above rollover spreads: '
          'MAJ %.1f bps (vs ~10), EXO %.1f bps (vs ~50) -> %s' %
          (be_maj, be_exo, 'PASS' if cond_c else 'FAIL'))
    verdict = 'REAL' if (cond_a and cond_b and cond_c) else 'ARTIFACT'
    print('\nVERDICT: %s' % verdict)


if __name__ == '__main__':
    main()
