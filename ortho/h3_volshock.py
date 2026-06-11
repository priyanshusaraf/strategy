#!/usr/bin/env /usr/bin/python3
"""
H3 — POST-VOLATILITY-SHOCK NORMALIZATION (hourly; 9 duka + 34 fx = 43 instruments).

Mechanism: vol shocks come with forced flows (margin calls, stop cascades, dealer
hedging) that push price beyond fair value; once the forcing stops (vol crests),
price relaxes back toward the pre-shock anchor.

PRE-SPECIFIED DESIGN (everything below fixed before any results were seen):
  State : RV24[t] = std of last 24 hourly close-changes (causal: uses close[t],
          entry is next bar open).
          B[t] = median RV24 over prior 240 bars (~10 days), shifted 1 bar.
  Shock : RV24/B > k, grid k in {2, 3}. FULL grid reported.
  Event : first bar where shock active AND RV24 declined 2 consecutive bars
          (crest passed). De-dup: >=24-bar gap between kept events (first event
          per episode).
  Side  : -sign(C[t] - C[t-24])  — fade the net shock-window move (ex-ante).
  Control: same shock state but RV24 RISING 2 consecutive bars (crest NOT
          passed), same side rule. Mechanism predicts control is worse.
  Horizons: (2, 4, 8, 16, 24, 48).
  Placebo: same event count at random valid times with the SAME side rule
          (-sign of trailing 24-bar move) -> tests the vol-crest CONDITIONING
          beyond any generic 24h-reversal/drift effect.
  NATURAL VARIANT (declared ex-ante): k=2, h=24 — inclusive shock threshold,
          one full day of relaxation matching the 24-bar shock window. The
          single cost-aware backtest runs ONLY on this variant.
"""
import numpy as np
import pandas as pd
import lab

HORIZONS   = (2, 4, 8, 16, 24, 48)
KS         = (2.0, 3.0)
NATURAL_K  = 2.0
NATURAL_H  = 24
N_PLACEBO  = 300
RV_N       = 24
BASE_N     = 240
DEDUP      = 24

SYMS = [('duka', s) for s in lab.DUKA_SYMS] + [('fx', s) for s in lab.fx_syms()]


def _dedup(pos, gap):
    out, last = [], -10**9
    for p in pos:
        if p - last >= gap:
            out.append(p)
            last = p
    return np.asarray(out, dtype=int)


def build_state(df):
    dc = df.c.diff()
    rv = dc.rolling(RV_N).std()
    base = rv.rolling(BASE_N, min_periods=BASE_N // 2).median().shift(1)
    ratio = (rv / base).values
    rv_v = rv.values
    down2 = np.zeros(len(df), bool)
    up2 = np.zeros(len(df), bool)
    down2[2:] = (rv_v[2:] < rv_v[1:-1]) & (rv_v[1:-1] < rv_v[:-2])
    up2[2:] = (rv_v[2:] > rv_v[1:-1]) & (rv_v[1:-1] > rv_v[:-2])
    net = (df.c - df.c.shift(RV_N)).values
    return ratio, down2, up2, net


def masks_for(df, ratio, trend2, net, k):
    raw = (ratio > k) & trend2 & np.isfinite(net) & (net != 0)
    pos = _dedup(np.flatnonzero(raw), DEDUP)
    mask = np.zeros(len(df), bool)
    mask[pos] = True
    sides = np.zeros(len(df))
    sides[mask] = -np.sign(net[mask])
    return mask, sides


def signed_study(df, mask, net, a, cb, horizons=HORIZONS, n_placebo=N_PLACEBO, seed=0):
    """Like lab.event_study but forward returns are SIGNED by the fade rule
    -sign(net 24-bar move). Placebo applies the SAME side rule at random valid
    times -> p tests the vol-crest conditioning, not the generic 24h fade.
    cb = round-trip cost bps for this instrument; cost_vu = (cb/1e4)*mean(|px|/ATR).
    Returns dict h -> dict(n, mean, t, f (signed returns), pm (placebo means),
    cost_vu)."""
    c, o = df.c.values, df.o.values
    av = a.values
    pos = np.flatnonzero(np.asarray(mask, bool))
    N = len(df)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(av) & (av > 0) & np.isfinite(net) & (net != 0))
    out = {}
    for h in horizons:
        P = _dedup(pos[(pos + h) < N - 1], h)
        P = P[np.isfinite(av[P]) & (av[P] > 0)]
        if len(P) < 10:
            out[h] = dict(n=len(P), mean=np.nan, t=np.nan,
                          f=np.array([]), pm=np.full(n_placebo, np.nan),
                          cost_vu=np.nan)
            continue
        s = -np.sign(net[P])
        f = s * (c[P + h] - o[P + 1]) / av[P]
        ok = np.isfinite(f)
        f = f[ok]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        cost_vu = (cb / 1e4) * float(np.nanmean(np.abs(o[P + 1][ok]) / av[P][ok]))
        V = valid[(valid + h) < N - 1]
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = V[rng.integers(0, len(V), size=len(P))]
            sq = -np.sign(net[Q])
            g = sq * (c[Q + h] - o[Q + 1]) / av[Q]
            pm[i] = np.nanmean(g)
        out[h] = dict(n=len(f), mean=m, t=t, f=f, pm=pm, cost_vu=cost_vu)
    return out


def pool(results, h):
    """Stack signed event returns across instruments; pooled placebo = per-
    iteration count-weighted mean of per-instrument placebo means."""
    fs, pms, ns, costs = [], [], [], []
    for r in results:
        d = r[h]
        if d['n'] >= 10 and np.isfinite(d['mean']):
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
    PM = np.nansum(np.vstack(pms), axis=0) / ntot
    p = float((np.abs(PM) >= abs(m)).mean())
    pos = sum(1 for f in fs if f.mean() > 0)
    return dict(n=len(F), mean=m, t=t, p=p, cost_vu=sum(costs) / ntot,
                pos=pos, tot=len(fs))


def main():
    data = {}
    print('loading %d instruments ...' % len(SYMS))
    for fam, sym in SYMS:
        df = lab.duka(sym) if fam == 'duka' else lab.fx(sym)
        data[sym] = (fam, df, lab.atr(df))

    # ---------------- event studies, full grid + control ----------------
    res = {k: {'event': [], 'control': []} for k in KS}
    ev_cache = {}   # (sym, k) -> (mask, sides) for natural backtest
    for fam, sym in SYMS:
        _, df, a = data[sym]
        ratio, down2, up2, net = build_state(df)
        cb = lab.cost_bps(sym)
        for k in KS:
            m_ev, s_ev = masks_for(df, ratio, down2, net, k)
            m_ct, _ = masks_for(df, ratio, up2, net, k)
            res[k]['event'].append(signed_study(df, m_ev, net, a, cb))
            res[k]['control'].append(signed_study(df, m_ct, net, a, cb))
            ev_cache[(sym, k)] = (m_ev, s_ev)
        print('  done %s' % sym)

    print('\n================ FULL PRE-SPECIFIED GRID (pooled, 43 instruments) ================')
    print('placebo = same count, random times, SAME fade-24h side rule (tests conditioning)')
    hdr = '%-4s %-3s | %7s %9s %6s %6s %6s %6s | %9s %6s   (control = RV still rising)'
    print(hdr % ('k', 'h', 'n_ev', 'mean_vu', 't', 'p_plc', 'e2c', 'sgn+', 'ctl_mean', 'ctl_t'))
    grid = {}
    for k in KS:
        for h in HORIZONS:
            pe = pool(res[k]['event'], h)
            pc = pool(res[k]['control'], h)
            grid[(k, h)] = (pe, pc)
            e2c = pe['mean'] / pe['cost_vu'] if pe['cost_vu'] and pe['cost_vu'] > 0 else np.nan
            print('%-4g %-3d | %7d %9.4f %6.2f %6.3f %6.2f %3d/%-3d | %9.4f %6.2f' %
                  (k, h, pe['n'], pe['mean'], pe['t'], pe['p'], e2c,
                   pe['pos'], pe['tot'], pc['mean'], pc['t']))
        print()

    # ---------------- per-instrument table at the natural variant ----------------
    k, h = NATURAL_K, NATURAL_H
    print('============ PER-INSTRUMENT @ natural variant k=%g h=%d ============' % (k, h))
    print('%-8s %-5s %6s %9s %7s' % ('sym', 'fam', 'n_ev', 'mean_vu', 't'))
    pos_cnt, tot_cnt = 0, 0
    fam_pos = {}
    for (fam, sym), r in zip(SYMS, res[k]['event']):
        d = r[h]
        if d['n'] < 10 or not np.isfinite(d['mean']):
            print('%-8s %-5s %6d %9s %7s' % (sym, fam, d['n'], 'nan', 'nan'))
            continue
        print('%-8s %-5s %6d %9.4f %7.2f' % (sym, fam, d['n'], d['mean'], d['t']))
        tot_cnt += 1
        if d['mean'] > 0:
            pos_cnt += 1
            fam_pos.setdefault(fam, [0, 0])[0] += 1
        fam_pos.setdefault(fam, [0, 0])[1] += 1
    pe, pc = grid[(k, h)]
    print('\nsign consistency: %d/%d instruments positive (%.0f%%)' %
          (pos_cnt, tot_cnt, 100.0 * pos_cnt / max(tot_cnt, 1)))
    for fam, (p_, t_) in fam_pos.items():
        print('  family %-5s: %d/%d positive' % (fam, p_, t_))
    print('NOTE: 34 FX pairs share legs (USD/EUR/JPY blocs) -> heavily cross-correlated;')
    print('      effective independent count is much lower than 43.')
    print('\npooled @ natural: n=%d mean=%.4f t=%.2f p_placebo=%.3f' %
          (pe['n'], pe['mean'], pe['t'], pe['p']))
    print('control @ natural: n=%d mean=%.4f t=%.2f p_placebo=%.3f' %
          (pc['n'], pc['mean'], pc['t'], pc['p']))

    # ---------------- edge vs cost at natural horizon ----------------
    e2c = pe['mean'] / pe['cost_vu'] if pe['cost_vu'] and pe['cost_vu'] > 0 else np.nan
    print('\npooled round-trip cost at natural horizon = %.4f vu -> edge/cost = %.2f' %
          (pe['cost_vu'], e2c))

    # ---------------- ONE cost-aware backtest: natural variant only ----------------
    print('\n============ BACKTEST (natural variant ONLY: k=%g h=%d, fade sides, per-sym costs) ============' % (k, h))
    parts, n_tr = [], 0
    for fam, sym in SYMS:
        _, df, a = data[sym]
        m_ev, s_ev = ev_cache[(sym, k)]
        pnl, n = lab.backtest(df, m_ev, h=h, sides=s_ev, cost=lab.cost_bps(sym), a=a)
        parts.append(pnl)
        n_tr += n
    port = pd.concat(parts, axis=1).fillna(0.0).sum(axis=1).sort_index()
    mt = lab.metrics(port)
    print('portfolio (sum of 43 single-unit instrument books), trades=%d' % n_tr)
    print(mt)
    print('yearly sharpe:')
    print(lab.yearly(port))

    parts2 = []
    for fam, sym in SYMS:
        _, df, a = data[sym]
        m_ev, s_ev = ev_cache[(sym, k)]
        pnl, _ = lab.backtest(df, m_ev, h=h, sides=s_ev,
                              cost=lab.cost_bps(sym, stress=2.0), a=a)
        parts2.append(pnl)
    port2 = pd.concat(parts2, axis=1).fillna(0.0).sum(axis=1).sort_index()
    mt2 = lab.metrics(port2)
    print('\nstress x2 costs:')
    print(mt2)

    print('\nbest grid cell by |t| (reported for completeness; backtest above is the')
    print('ex-ante natural variant, NOT tuned):')
    best = max(grid, key=lambda kh: abs(grid[kh][0]['t'])
               if np.isfinite(grid[kh][0]['t']) else -1)
    bp = grid[best][0]
    print('  k=%g h=%d: n=%d mean=%.4f t=%.2f p=%.3f' %
          (best[0], best[1], bp['n'], bp['mean'], bp['t'], bp['p']))


if __name__ == '__main__':
    main()
