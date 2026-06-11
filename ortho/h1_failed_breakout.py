"""
H1 — FAILED BREAKOUT / TRAPPED PARTICIPANTS (hourly).

Mechanism: price takes out a salient prior extreme intrabar (stop-run / breakout
entries) then closes back inside -> breakout traders trapped; their unwind plus
absent follow-through flow pushes price back. FADE: short failed upside breakout,
long failed downside breakout. Direction fixed ex-ante by mechanism.

Event (causal, computable at close of bar t):
  L_up = rolling max of HIGH over prior N bars (shift 1, excludes bar t)
  L_dn = rolling min of LOW  over prior N bars (shift 1)
  up-event  : H[t] > L_up and C[t] < L_up   -> side -1
  dn-event  : L[t] < L_dn and C[t] > L_dn   -> side +1
  ambiguous (both) -> dropped.
  fresh flag: level not touched in prior 24 bars
    up: rollmax(H,24).shift(1) < L_up ; dn: rollmin(L,24).shift(1) > L_dn

PRE-SPECIFIED grid (full grid reported, no tuning):
  N in {120, 480}; fresh in {False, True}; horizons (1,2,4,8,16,24,48).
NATURAL variant declared ex-ante: N=480 (~20d Donchian level, the salient
breakout level), fresh=False (base definition), h=8 (~1 trading day) -> the one
cost-aware backtest.

Universe: all 9 duka commodity CFDs + all 34 FX pairs (43 instruments).
Run: cd ortho && /usr/bin/python3 h1_failed_breakout.py
"""
import numpy as np
import pandas as pd
import lab

pd.set_option('display.width', 200)

HORIZONS = (1, 2, 4, 8, 16, 24, 48)
GRID_N = (120, 480)
FRESH = (False, True)
NAT_N, NAT_FRESH, NAT_H = 480, False, 8
N_PLACEBO = 300

FAMILY = {}
for s in ['WTI', 'BRENT', 'GAS', 'DIESEL']:
    FAMILY[s] = 'energy'
for s in ['COPPER', 'XAU', 'XAG']:
    FAMILY[s] = 'metal'
for s in ['COCOA', 'SUGAR']:
    FAMILY[s] = 'soft'


def fx_family(sym):
    if sym in lab._FX_MAJOR:
        return 'fx_major'
    if sym in lab._FX_CROSS:
        return 'fx_cross'
    return 'fx_exotic'


def build_sides(df, N, fresh):
    """Per-bar sides array: -1 fade failed upside break, +1 fade failed downside."""
    l_up = df.h.rolling(N).max().shift(1)
    l_dn = df.l.rolling(N).min().shift(1)
    up = (df.h > l_up) & (df.c < l_up)
    dn = (df.l < l_dn) & (df.c > l_dn)
    if fresh:
        up &= df.h.rolling(24).max().shift(1) < l_up
        dn &= df.l.rolling(24).min().shift(1) > l_dn
    amb = up & dn
    sides = np.zeros(len(df))
    sides[(up & ~amb).values] = -1.0
    sides[(dn & ~amb).values] = +1.0
    return sides


def signed_event_study(df, sides, a, cost_bp, seed=0):
    """Signed analogue of lab.event_study (sides applied; placebo = same count,
    random valid times, random +/-1 signs -> null kills drift, tests conditioning
    AND direction). Returns dict h -> dict(n, mean, t, sum, sumsq, placebo_means,
    cost_vu_mean[h==NAT_H only is used])."""
    c, o = df.c.values, df.o.values
    av = a.values
    pos = np.flatnonzero(sides != 0)
    Nn = len(df)
    rng = np.random.default_rng(seed)
    fwd_ok = np.isfinite(av) & (av > 0)
    out = {}
    for h in HORIZONS:
        P = lab._deoverlap(pos[(pos + h) < Nn - 1], h)
        P = P[fwd_ok[P]]
        valid = np.flatnonzero(fwd_ok & (np.arange(Nn) + h < Nn - 1))
        if len(P) < 10 or len(valid) < 100:
            out[h] = None
            continue
        f = sides[P] * (c[P + h] - o[P + 1]) / av[P]
        good = np.isfinite(f)
        f = f[good]
        Pg = P[good]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        # placebo: random times, random signs
        fv = (c[valid + h] - o[valid + 1]) / av[valid]
        fin = np.isfinite(fv)
        valid, fv = valid[fin], fv[fin]
        pm = np.empty(N_PLACEBO)
        k = len(f)
        for i in range(N_PLACEBO):
            idx = rng.integers(0, len(fv), size=k)
            sg = rng.choice([-1.0, 1.0], size=k)
            pm[i] = (sg * fv[idx]).mean()
        cost_vu = (cost_bp / 1e4) * np.abs(o[Pg + 1]) / av[Pg]
        out[h] = dict(n=len(f), mean=m, t=t, s=f.sum(), ss=(f ** 2).sum(),
                      pm=pm, cost_vu=cost_vu.mean())
    return out


def main():
    syms = [(s, 'duka') for s in lab.DUKA_SYMS] + [(s, 'fx') for s in lab.fx_syms()]
    print(f'Universe: {len(syms)} instruments ({len(lab.DUKA_SYMS)} duka + {len(lab.fx_syms())} fx)')
    data = {}
    for sym, src in syms:
        df = lab.duka(sym) if src == 'duka' else lab.fx(sym)
        data[sym] = (df, lab.atr(df))

    # ---------------- full grid, pooled across instruments ----------------
    res = {}   # (N, fresh) -> sym -> h -> stats
    for N in GRID_N:
        for fr in FRESH:
            cell = {}
            for i, (sym, src) in enumerate(syms):
                df, a = data[sym]
                sides = build_sides(df, N, fr)
                cell[sym] = signed_event_study(df, sides, a, lab.cost_bps(sym),
                                               seed=1000 * N + 100 * fr + i)
            res[(N, fr)] = cell

    print('\n================ FULL PRE-SPECIFIED GRID — POOLED (stacked events) ================')
    print('(mean_vu signed in trade direction; placebo = same-count random-time random-sign events)')
    grid_rows = []
    for (N, fr), cell in res.items():
        for h in HORIZONS:
            tot_n = tot_s = tot_ss = 0.0
            pms, cvs = [], []
            npos = ntot = 0
            for sym, st in cell.items():
                if st is None or st.get(h) is None:
                    continue
                d = st[h]
                tot_n += d['n']; tot_s += d['s']; tot_ss += d['ss']
                pms.append((d['n'], d['pm'])); cvs.append((d['n'], d['cost_vu']))
                ntot += 1; npos += d['mean'] > 0
            if tot_n < 50:
                continue
            pmean = tot_s / tot_n
            pvar = (tot_ss - tot_n * pmean ** 2) / (tot_n - 1)
            pt = pmean / np.sqrt(pvar / tot_n)
            pool_pm = np.zeros(N_PLACEBO)
            wsum = 0.0
            for n_i, pm_i in pms:
                pool_pm += n_i * pm_i; wsum += n_i
            pool_pm /= wsum
            p_plac = float((np.abs(pool_pm) >= abs(pmean)).mean())
            cost = sum(n_i * c_i for n_i, c_i in cvs) / wsum
            grid_rows.append(dict(N=N, fresh=fr, h=h, n=int(tot_n),
                                  mean_vu=round(pmean, 4), t=round(pt, 2),
                                  p_placebo=round(p_plac, 3),
                                  edge_cost=round(pmean / cost, 2),
                                  pos_syms=f'{npos}/{ntot}'))
    gdf = pd.DataFrame(grid_rows)
    print(gdf.to_string(index=False))

    # ---------------- per-instrument table, natural cell/horizon ----------------
    print(f'\n================ PER-INSTRUMENT — natural cell N={NAT_N} fresh={NAT_FRESH} h={NAT_H} ================')
    cell = res[(NAT_N, NAT_FRESH)]
    rows = []
    for sym, src in syms:
        st = cell[sym]
        d = None if st is None else st.get(NAT_H)
        fam = FAMILY.get(sym, fx_family(sym))
        if d is None:
            rows.append(dict(sym=sym, family=fam, n=0, mean_vu=np.nan, t=np.nan,
                             edge_cost=np.nan))
            continue
        rows.append(dict(sym=sym, family=fam, n=d['n'], mean_vu=round(d['mean'], 4),
                         t=round(d['t'], 2),
                         edge_cost=round(d['mean'] / d['cost_vu'], 2)))
    pdf = pd.DataFrame(rows)
    print(pdf.to_string(index=False))
    ok = pdf.dropna(subset=['mean_vu'])
    npos = int((ok.mean_vu > 0).sum())
    print(f'\nSign consistency at h={NAT_H}: {npos}/{len(ok)} instruments positive '
          f'({100 * npos / len(ok):.0f}%)')
    fam_tab = ok.groupby('family').apply(
        lambda g: pd.Series(dict(n_syms=len(g), n_events=int(g.n.sum()),
                                 ev_w_mean=round(float((g.mean_vu * g.n).sum() / g.n.sum()), 4),
                                 pos=f'{int((g.mean_vu > 0).sum())}/{len(g)}')),
        include_groups=False)
    print('\nBy family (NOTE: same-family instruments are correlated — FX pairs share '
          'legs, energy co-moves):')
    print(fam_tab.to_string())

    # ---------------- ONE cost-aware backtest: natural variant only ----------------
    print(f'\n================ BACKTEST — natural variant only: N={NAT_N} fresh={NAT_FRESH} '
          f'h={NAT_H}, per-sym lab.cost_bps ================')
    pnls, tn = [], 0
    per_sym = []
    for sym, src in syms:
        df, a = data[sym]
        sides = build_sides(df, NAT_N, NAT_FRESH)
        mask = sides != 0
        pnl, n = lab.backtest(df, mask, h=NAT_H, sides=sides,
                              cost=lab.cost_bps(sym), a=a)
        pnls.append(pnl.rename(sym)); tn += n
        per_sym.append((sym, n, round(float(pnl.sum()), 1)))
    port = pd.concat(pnls, axis=1).fillna(0.0).sum(axis=1)
    print(f'total trades: {tn}')
    print('portfolio metrics (sum of per-instrument bar pnl, vol units):')
    print(lab.metrics(port))
    print('\nyearly sharpe of portfolio pnl:')
    print(lab.yearly(port).to_string())
    print('\nper-sym (sym, n_trades, total_vu):')
    print(per_sym)


if __name__ == '__main__':
    main()
