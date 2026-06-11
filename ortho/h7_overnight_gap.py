#!/usr/bin/env /usr/bin/python3
"""
H7 — OVERNIGHT vs INTRADAY STRUCTURE + GAP FADE (daily, long history).

Mechanism: the settlement->open period has thin liquidity and inventory-driven
pricing; openings overshoot. Day-session two-way flow corrects the overshoot.

Universe:
  FUT: all 49 lab.daily_files() continuous futures, sliced >= 2000-01-01
       (opens synthetic O=H=L=C before ~2000). Back-adjusted: $-diffs valid,
       LEVELS distorted (BRN post-2000 is 32% negative) -> per-bar |px|/ATR cost
       is invalid; instead each future gets a CONSTANT cost_vu estimated from
       the LAST 250 bars (levels there = actual current contract).
       NOTE: 1!/2! pairs of the same root are near-duplicates -> effective
       independent count ~25; sign consistency also reported by unique root.
  EQ : 8 equities + 2 indices (DXY, US10Y) via lab.tv(). Levels true; verified
       splits are adjusted (NVDA >25% gaps are the real 2008-07-03 / 2023-05-25
       moves). DXY / US10Y are NON-TRADEABLE indices: stats only, excluded from
       the cost-aware backtest.

PRE-SPECIFIED DESIGN (fixed before results were seen):
  ATR    : lab.atr (n=100, shifted 1 bar) -> known at the open of day t.
  ON[t]  = (O[t]-C[t-1])/ATR[t]   overnight (gap) return, vol units
  ID[t]  = (C[t]-O[t])/ATR[t]     intraday (open->close) return, vol units
  Valid  : finite ON/ID, ATR>0, calendar gap (t-1 -> t) <= 14 days.
  (a) GAP FADE event: |ON[t]| >= g, g in {0.5, 1.0}. Side = -sign(ON[t]).
      Primary outcome (h=0): same-day r0 = side*ID[t]. Entry assumed AT the
      open print (flagged: optimistic for futures; ~implementable via MOO for
      equities). Events are 1-day, non-overlapping by construction.
      Follow-on: signed study entering next open o[t+1], h in {1,2,4,8} days,
      de-overlapped at h spacing.
      Placebo (300 iters): same count, random valid days, SAME side rule
      (-sign of that day's ON) -> tests the MAGNITUDE conditioning beyond any
      generic fade/drift.
  (b) corr(ON[t], ID[t]) per instrument + stacked pooled, full & post-2015.
  (c) corr(ID[t], ON[t+1]) same.
  NATURAL VARIANT (declared ex-ante): g=1.0 (clear overshoot), same-day
      open->close fade, per-instrument costs. ONE cost-aware backtest, FUT and
      EQ books separately, stress x2 reported. No iteration.
"""
import numpy as np
import pandas as pd
import lab

GS         = (0.5, 1.0)
NATURAL_G  = 1.0
FOLLOW_HS  = (1, 2, 4, 8)
N_PLACEBO  = 300
MAX_HOLE_D = 14
EQ_COST    = 5.0

EQ_FILES = ['BATS_MSFT, 1D.csv', 'BATS_NVDA, 1D.csv', 'BATS_PLTR, 1D.csv',
            'BSE_DLY_HDFCBANK, 1D.csv', 'BSE_DLY_ICICIBANK, 1D.csv',
            'BSE_DLY_TCS, 1D.csv', 'NSE_DLY_INFY, 1D.csv',
            'NSE_DLY_WIPRO, 1D.csv', 'TVC_DXY, 1D.csv', 'TVC_US10Y, 1D.csv']
NONTRADE = {'DXY', 'US10Y'}                      # indices: stats only

ROOT_ALIAS = {'BRN': 'BZ', 'WBS': 'CL', 'UHO': 'HO'}   # ULS -> default 10bps


def fut_root(fname):
    sym = fname.split(',')[0].split('_')[-1].rstrip('!')   # 'ZC1' -> root 'ZC'
    return sym[:-1] if sym[-1] in '12' else sym


def eq_name(fname):
    return fname.split(',')[0].split('_')[-1]


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def prep(df):
    """Returns dict of aligned numpy arrays + validity mask."""
    a = lab.atr(df)
    o, c, av = df.o.values, df.c.values, a.values
    on = np.full(len(df), np.nan)
    on[1:] = (o[1:] - c[:-1]) / av[1:]
    iday = (c - o) / av
    dt = df.index.values
    hole = np.full(len(df), True)
    hole[1:] = (dt[1:] - dt[:-1]) > np.timedelta64(MAX_HOLE_D, 'D')
    valid = np.isfinite(on) & np.isfinite(iday) & np.isfinite(av) & (av > 0) & ~hole
    return dict(o=o, c=c, a=av, on=on, iday=iday, valid=valid, idx=df.index)


def sameday_study(P, g, n_placebo=N_PLACEBO, seed=0):
    """Same-day fade: events |ON|>=g, outcome -sign(ON)*ID, entry at open print.
    Placebo: same count random valid days, same side rule."""
    ev = P['valid'] & (np.abs(P['on']) >= g)
    pos = np.flatnonzero(ev)
    f = -np.sign(P['on'][pos]) * P['iday'][pos]
    if len(f) < 10:
        return dict(n=len(f), mean=np.nan, t=np.nan, f=np.array([]),
                    pm=np.full(n_placebo, np.nan), pos=pos)
    m = f.mean()
    t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
    V = np.flatnonzero(P['valid'] & (P['on'] != 0))
    rng = np.random.default_rng(seed)
    pm = np.empty(n_placebo)
    for i in range(n_placebo):
        Q = V[rng.integers(0, len(V), size=len(pos))]
        pm[i] = np.mean(-np.sign(P['on'][Q]) * P['iday'][Q])
    return dict(n=len(f), mean=m, t=t, f=f, pm=pm, pos=pos)


def follow_study(P, g, horizons=FOLLOW_HS, n_placebo=N_PLACEBO, seed=0):
    """Signed forward study: entry o[t+1], exit c[t+h], side=-sign(ON[t])."""
    o, c, a, on = P['o'], P['c'], P['a'], P['on']
    N = len(o)
    pos0 = np.flatnonzero(P['valid'] & (np.abs(on) >= g))
    V = np.flatnonzero(P['valid'] & (on != 0))
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        Pp = _dedup(pos0[(pos0 + h) < N - 1], h)
        if len(Pp) < 10:
            out[h] = dict(n=len(Pp), mean=np.nan, t=np.nan, f=np.array([]),
                          pm=np.full(n_placebo, np.nan))
            continue
        s = -np.sign(on[Pp])
        f = s * (c[Pp + h] - o[Pp + 1]) / a[Pp]
        f = f[np.isfinite(f)]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        Vh = V[(V + h) < N - 1]
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = Vh[rng.integers(0, len(Vh), size=len(Pp))]
            g_ = -np.sign(on[Q]) * (c[Q + h] - o[Q + 1]) / a[Q]
            pm[i] = np.nanmean(g_)
        out[h] = dict(n=len(f), mean=m, t=t, f=f, pm=pm)
    return out


def pool(items):
    """items: list of dicts with f (signed returns) and pm (placebo means)."""
    fs = [d['f'] for d in items if d['n'] >= 10 and np.isfinite(d['mean'])]
    pms = [d['pm'] * d['n'] for d in items if d['n'] >= 10 and np.isfinite(d['mean'])]
    ns = [d['n'] for d in items if d['n'] >= 10 and np.isfinite(d['mean'])]
    if not fs:
        return dict(n=0, mean=np.nan, t=np.nan, p=np.nan, pos=0, tot=0)
    F = np.concatenate(fs)
    m = F.mean()
    t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
    PM = np.nansum(np.vstack(pms), axis=0) / sum(ns)
    return dict(n=len(F), mean=m, t=t, p=float((np.abs(PM) >= abs(m)).mean()),
                pos=sum(1 for f in fs if f.mean() > 0), tot=len(fs))


def corrstat(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum()
    if n < 30:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
    return r, t, int(n)


def main():
    # ---------------- load ----------------
    insts = []   # (group, name, df, P, cost_bps, cost_vu_const_or_None)
    for f in lab.daily_files():
        df = lab.tv_daily(f)
        df = df[df.index >= '2000-01-01']
        P = prep(df)
        root = fut_root(f)
        cb = lab.cost_bps(ROOT_ALIAS.get(root, root))
        tail = df.tail(250)
        at = lab.atr(df).reindex(tail.index)
        cvu = (cb / 1e4) * float(np.nanmedian(np.abs(tail.c.values) / at.values))
        name = f.split(',')[0].split('_')[-1]            # e.g. ZC1!
        insts.append(('FUT', name, df, P, cb, cvu))
    for f in EQ_FILES:
        df = lab.tv(f)
        P = prep(df)
        insts.append(('EQ', eq_name(f), df, P, EQ_COST, None))
    nf = sum(1 for g, *_ in insts if g == 'FUT')
    print('loaded %d futures (>=2000) + %d equity/index dailies' % (nf, len(insts) - nf))

    # ---------------- (a) gap fade: full pre-specified grid ----------------
    res_same = {g: [] for g in GS}            # per instrument, in insts order
    res_fol = {g: [] for g in GS}
    for grp, name, df, P, cb, cvu in insts:
        for g in GS:
            res_same[g].append(sameday_study(P, g))
            res_fol[g].append(follow_study(P, g))

    print('\n========== (a) GAP FADE — FULL GRID, pooled by group ==========')
    print('h=0 is SAME-DAY open->close (entry at open print); h>=1 enters next open.')
    print('placebo = same count random valid days, same -sign(ON) side rule.')
    hdr = '%-4s %-4s %-3s | %7s %9s %7s %6s %8s'
    print(hdr % ('grp', 'g', 'h', 'n_ev', 'mean_vu', 't', 'p_plc', 'sgn+'))
    grid = {}
    for grp in ('FUT', 'EQ'):
        sel = [i for i, it in enumerate(insts) if it[0] == grp]
        for g in GS:
            pe = pool([res_same[g][i] for i in sel])
            grid[(grp, g, 0)] = pe
            print('%-4s %-4g %-3d | %7d %9.4f %7.2f %6.3f %4d/%-3d' %
                  (grp, g, 0, pe['n'], pe['mean'], pe['t'], pe['p'], pe['pos'], pe['tot']))
            for h in FOLLOW_HS:
                pf = pool([res_fol[g][i][h] for i in sel])
                grid[(grp, g, h)] = pf
                print('%-4s %-4g %-3d | %7d %9.4f %7.2f %6.3f %4d/%-3d' %
                      (grp, g, h, pf['n'], pf['mean'], pf['t'], pf['p'], pf['pos'], pf['tot']))
            print()

    # ---------------- per-instrument table @ natural (g=1.0, same-day) ----------------
    g = NATURAL_G
    print('========== PER-INSTRUMENT @ natural g=%g, same-day O->C fade ==========' % g)
    print('%-10s %-4s %6s %9s %7s %9s' % ('sym', 'grp', 'n_ev', 'mean_vu', 't', 'cost_vu'))
    root_means = {}
    for (grp, name, df, P, cb, cvu), d in zip(insts, res_same[g]):
        cv = cvu if cvu is not None else \
            (cb / 1e4) * float(np.nanmedian(np.abs(P['o'][d['pos']]) / P['a'][d['pos']])) \
            if len(d['pos']) else np.nan
        if d['n'] < 10 or not np.isfinite(d['mean']):
            print('%-10s %-4s %6d %9s %7s %9s' % (name, grp, d['n'], 'nan', 'nan', '-'))
            continue
        print('%-10s %-4s %6d %9.4f %7.2f %9.4f' % (name, grp, d['n'], d['mean'], d['t'], cv))
        root = name.rstrip('!').rstrip('12') if grp == 'FUT' else name
        root_means.setdefault((grp, root), []).append(d['mean'])
    for grp in ('FUT', 'EQ'):
        roots = {r: np.mean(v) for (gg, r), v in root_means.items() if gg == grp}
        pos = sum(1 for v in roots.values() if v > 0)
        print('%s unique-root sign consistency: %d/%d positive' % (grp, pos, len(roots)))
    print('FLAG: 1!/2! pairs are near-duplicates; FX-style cross-correlation within')
    print('      sectors (grains bloc, energy bloc) further shrinks effective N.')
    print('FLAG: same-day leg assumes fill AT the open print (MOO-approximable for')
    print('      equities; optimistic for futures). Grain limit-lock days untradeable.')

    # ---------------- edge vs cost @ natural ----------------
    print('\n========== EDGE vs COST @ natural (g=%g, same-day) ==========' % g)
    for grp in ('FUT', 'EQ'):
        sel = [i for i, it in enumerate(insts) if it[0] == grp
               and (grp == 'FUT' or it[1] not in NONTRADE)]
        pe = pool([res_same[g][i] for i in sel])
        cvs, ns = [], []
        for i in sel:
            grp_, name, df, P, cb, cvu = insts[i]
            d = res_same[g][i]
            if d['n'] < 10:
                continue
            cv = cvu if cvu is not None else \
                (cb / 1e4) * float(np.nanmedian(np.abs(P['o'][d['pos']]) / P['a'][d['pos']]))
            cvs.append(cv * d['n'])
            ns.append(d['n'])
        cost = sum(cvs) / max(sum(ns), 1)
        e2c = pe['mean'] / cost if cost > 0 else np.nan
        print('%s (tradeable only): pooled mean=%.4f vu, mean cost=%.4f vu -> edge/cost=%.2f'
              % (grp, pe['mean'], cost, e2c))

    # ---------------- (b)/(c) overnight<->intraday transmission ----------------
    print('\n========== (b) corr(ON[t], ID[t])  and  (c) corr(ID[t], ON[t+1]) ==========')
    print('%-10s %-4s | %8s %7s %7s | %8s %7s | %8s %7s' %
          ('sym', 'grp', 'b_full', 't', 'n', 'b_2015+', 't', 'c_full', 't'))
    pooled = {('FUT', k): ([], []) for k in ('b', 'b15', 'c')}
    pooled.update({('EQ', k): ([], []) for k in ('b', 'b15', 'c')})
    for grp, name, df, P, cb, cvu in insts:
        on, iday, valid = P['on'], P['iday'], P['valid']
        on_next = np.full(len(on), np.nan)
        on_next[:-1] = np.where(valid[1:], on[1:], np.nan)
        x = np.where(valid, on, np.nan)
        y = np.where(valid, iday, np.nan)
        rb, tb, nb = corrstat(x, y)
        m15 = P['idx'] >= pd.Timestamp('2015-01-01', tz=P['idx'].tz)
        rb15, tb15, n15 = corrstat(x[m15], y[m15])
        rc, tc, nc = corrstat(y, on_next)
        print('%-10s %-4s | %8.3f %7.1f %7d | %8.3f %7.1f | %8.3f %7.1f' %
              (name, grp, rb, tb, nb, rb15, tb15, rc, tc))
        ok = np.isfinite(x) & np.isfinite(y)
        pooled[(grp, 'b')][0].append(x[ok]); pooled[(grp, 'b')][1].append(y[ok])
        ok = ok & m15
        pooled[(grp, 'b15')][0].append(x[ok]); pooled[(grp, 'b15')][1].append(y[ok])
        ok2 = np.isfinite(y) & np.isfinite(on_next)
        pooled[(grp, 'c')][0].append(y[ok2]); pooled[(grp, 'c')][1].append(on_next[ok2])
    print('--- pooled (stacked; observations within group are cross-correlated, t inflated) ---')
    for grp in ('FUT', 'EQ'):
        for k, lbl in (('b', 'ON->ID full'), ('b15', 'ON->ID 2015+'), ('c', 'ID->ONnext full')):
            X = np.concatenate(pooled[(grp, k)][0])
            Y = np.concatenate(pooled[(grp, k)][1])
            r, t, n = corrstat(X, Y)
            print('  %s %-15s r=%7.3f t=%7.1f n=%d' % (grp, lbl, r, t, n))

    # ---------------- ONE cost-aware backtest: natural variant only ----------------
    print('\n========== BACKTEST (natural ONLY: g=%g same-day O->C fade, real costs) ==========' % g)
    for grp in ('FUT', 'EQ'):
        for stress in (1.0, 2.0):
            parts, ntr = [], 0
            for grp_, name, df, P, cb, cvu in insts:
                if grp_ != grp or name in NONTRADE:
                    continue
                ev = P['valid'] & (np.abs(P['on']) >= g)
                pos = np.flatnonzero(ev)
                pnl = np.zeros(len(df))
                if cvu is not None:
                    cv = cvu * stress
                    pnl[pos] = -np.sign(P['on'][pos]) * P['iday'][pos] - cv
                else:
                    cvv = (cb * stress / 1e4) * np.abs(P['o'][pos]) / P['a'][pos]
                    pnl[pos] = -np.sign(P['on'][pos]) * P['iday'][pos] - cvv
                ntr += len(pos)
                parts.append(pd.Series(pnl, index=df.index))
            port = pd.concat(parts, axis=1).fillna(0.0).sum(axis=1).sort_index()
            mt = lab.metrics(port)
            tag = 'x%g costs' % stress
            print('%s book (%d trades, %s): %s' % (grp, ntr, tag, mt))
            if stress == 1.0:
                print('  yearly sharpe:')
                ys = lab.yearly(port)
                print('  ' + '  '.join('%d:%+.2f' % (yy, vv) for yy, vv in ys.items()))

    # ---------------- best grid cell, for completeness ----------------
    fin = {k: v for k, v in grid.items() if np.isfinite(v['t'])}
    best = max(fin, key=lambda k: abs(fin[k]['t']))
    bp = fin[best]
    print('\nbest grid cell by |t| (completeness; backtest above is the ex-ante natural')
    print('variant, NOT tuned): grp=%s g=%g h=%d: n=%d mean=%.4f t=%.2f p=%.3f' %
          (best[0], best[1], best[2], bp['n'], bp['mean'], bp['t'], bp['p']))


if __name__ == '__main__':
    main()
