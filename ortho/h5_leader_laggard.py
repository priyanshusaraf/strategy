"""
H5 — LEADER-LAGGARD INFORMATION DIFFUSION (hourly). NOT spread trading.

Mechanism: price discovery concentrates in the liquid leader; the laggard's makers
update with delay. After a big leader move the laggard hasn't matched, the laggard
drifts toward the leader's implied move (catch-up). Alternative: leader overshoots
and reverts. Both directions tested; direction of each trade follows ex-ante from
the stated mechanism (laggard side = sign of leader move; leader fade = -sign).

Event (causal, computable at close of event bar t, on the INNER-JOINED hourly index):
  zL = (cL[t] - cL[t-W]) / ATR_L[t]   (lab.atr already shifted -> causal)
  zB = (cB[t] - cB[t-W]) / ATR_B[t]
  event:  |zL| >= THR  and  |zB| <= 0.5*|zL|
  laggard trade side = sign(zL)  (catch-up);  leader check side = -sign(zL)

PRE-SPECIFIED GRID (full results reported, no tuning):
  THR in {1.5, 2.0, 2.5}  x  W in {2, 4}.  Headline/natural cell: THR=2.0, W=4.
  Natural horizon pre-specified: h=8 bars (information diffusion over a session).
  Horizons screened: (1,2,4,8,16,24).
ONE backtest only: natural cell (2.0, 4), h=8, catch-up sides, per-pair lab cost.

Run: cd /Users/priyanshusaraf/Desktop/strategy-dev-mr/ortho && /usr/bin/python3 h5_leader_laggard.py
"""
import numpy as np
import pandas as pd
import lab

HORIZONS = (1, 2, 4, 8, 16, 24)
GRID_THR = (1.5, 2.0, 2.5)
GRID_W   = (2, 4)
NAT_THR, NAT_W, NAT_H = 2.0, 4, 8
N_PLACEBO = 300

PAIRS = [('WTI','BRENT'), ('BRENT','WTI'), ('BRENT','DIESEL'), ('WTI','GAS'),
         ('XAU','XAG'), ('XAU','COPPER'), ('XAG','COPPER'),
         ('EURUSD','EURSEK'), ('EURUSD','EURNOK'), ('EURUSD','EURPLN'),
         ('EURUSD','EURHUF'), ('AUDUSD','NZDUSD'), ('AUDUSD','AUDCAD'),
         ('GBPUSD','GBPNZD'), ('USDJPY','CHFJPY')]

FAMILY = {('WTI','BRENT'):'oil', ('BRENT','WTI'):'oil', ('BRENT','DIESEL'):'oil',
          ('WTI','GAS'):'oil', ('XAU','XAG'):'metal', ('XAU','COPPER'):'metal',
          ('XAG','COPPER'):'metal', ('EURUSD','EURSEK'):'eurx',
          ('EURUSD','EURNOK'):'eurx', ('EURUSD','EURPLN'):'eurx',
          ('EURUSD','EURHUF'):'eurx', ('AUDUSD','NZDUSD'):'anzac',
          ('AUDUSD','AUDCAD'):'anzac', ('GBPUSD','GBPNZD'):'gbp',
          ('USDJPY','CHFJPY'):'jpy'}

_cache = {}
def load(sym):
    if sym not in _cache:
        _cache[sym] = lab.duka(sym) if sym in lab.DUKA_SYMS else lab.fx(sym)
    return _cache[sym]

def align_pair(ls, bs):
    dL, dB = load(ls), load(bs)
    common = dL.index.intersection(dB.index)
    aL = lab.atr(dL).reindex(common)
    aB = lab.atr(dB).reindex(common)
    return dL.loc[common], dB.loc[common], aL, aB

def make_events(dL, dB, aL, aB, thr, W):
    zL = (dL.c - dL.c.shift(W)) / aL
    zB = (dB.c - dB.c.shift(W)) / aB
    ok = zL.notna() & zB.notna() & (aL > 0) & (aB > 0)
    mask = (ok & (zL.abs() >= thr) & (zB.abs() <= 0.5 * zL.abs())).values
    sides = np.zeros(len(dL))
    sides[mask] = np.sign(zL.values[mask])
    return mask, sides

def signed_es(df, mask, sides, a, cost_bps, horizons=HORIZONS,
              n_placebo=N_PLACEBO, seed=0):
    """Signed event study modeled on lab.event_study: fwd ret next-open->close[t+h]
    in ATR units, multiplied by trade side. Placebo = same-count random-time events
    with RANDOM signs (drift cancels under the null -> tests conditioning+direction).
    Returns dict h -> (n, mean, t, p_placebo, f_array, placebo_mean_array, cost_vu)."""
    c, o = df.c.values, df.o.values
    av = np.asarray(a, float)
    sd = np.asarray(sides, float)
    pos = np.flatnonzero(np.asarray(mask, bool))
    N = len(df)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(av) & (av > 0))
    out = {}
    for h in horizons:
        P = lab._deoverlap(pos[(pos + h) < N - 1], h)
        P = P[np.isfinite(av[P]) & (av[P] > 0)]
        if len(P) < 10:
            out[h] = (len(P), np.nan, np.nan, np.nan,
                      np.array([]), np.full(n_placebo, np.nan), np.nan)
            continue
        f = sd[P] * (c[P + h] - o[P + 1]) / av[P]
        f = f[np.isfinite(f)]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        cost_vu = float(np.nanmean((cost_bps / 1e4) * np.abs(o[P + 1]) / av[P]))
        V = valid[(valid + h) < N - 1]
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = rng.choice(V, size=len(P), replace=False)
            sg = rng.choice([-1.0, 1.0], size=len(P))
            pm[i] = np.nanmean(sg * (c[Q + h] - o[Q + 1]) / av[Q])
        p = float((np.abs(pm) >= abs(m)).mean())
        out[h] = (len(f), m, t, p, f, pm, cost_vu)
    return out

def pool(results):
    """results: list of per-pair dicts from signed_es. Stack events; pooled placebo
    mean per iteration = event-count-weighted mean of per-pair placebo means."""
    rows = []
    for h in HORIZONS:
        fs, ws, pms, costs = [], [], [], []
        for r in results:
            n, m, t, p, f, pm, cv = r[h]
            if n >= 10 and len(f):
                fs.append(f); ws.append(len(f)); pms.append(pm); costs.append(cv * len(f))
        if not fs:
            rows.append((h, 0, np.nan, np.nan, np.nan, np.nan)); continue
        F = np.concatenate(fs)
        m = F.mean()
        t = m / (F.std(ddof=1) / np.sqrt(len(F)) + 1e-12)
        W = np.array(ws, float)
        PM = np.vstack(pms)                      # pairs x n_placebo
        pooled_pm = (PM * W[:, None]).sum(0) / W.sum()
        p = float((np.abs(pooled_pm) >= abs(m)).mean())
        cv = sum(costs) / W.sum()
        rows.append((h, len(F), m, t, p, cv))
    return pd.DataFrame(rows, columns=['h','n','mean_vu','t','p_placebo','cost_vu']
                        ).set_index('h')

def main():
    print('Loading + aligning pairs...')
    aligned = {}
    for ls, bs in PAIRS:
        dL, dB, aL, aB = align_pair(ls, bs)
        aligned[(ls, bs)] = (dL, dB, aL, aB)
        print(f'  {ls}->{bs}: {len(dL)} common bars '
              f'({dL.index[0].date()} .. {dL.index[-1].date()})')

    # ---------------- full pre-specified grid: pooled catch-up (laggard) ----------
    print('\n================ GRID (pooled across 15 pairs, LAGGARD catch-up, '
          'sides=sign(zL)) ================')
    grid_store = {}
    for thr in GRID_THR:
        for W in GRID_W:
            res_lag, res_led = [], []
            for k, (ls, bs) in enumerate(PAIRS):
                dL, dB, aL, aB = aligned[(ls, bs)]
                mask, sides = make_events(dL, dB, aL, aB, thr, W)
                r = signed_es(dB, mask, sides, aB, lab.cost_bps(bs),
                              seed=1000 + k)
                res_lag.append(r)
                rl = signed_es(dL, mask, -sides, aL, lab.cost_bps(ls),
                               seed=5000 + k)
                res_led.append(rl)
            pl = pool(res_lag)
            pf = pool(res_led)
            grid_store[(thr, W)] = (res_lag, res_led, pl, pf)
            print(f'\n--- THR={thr} W={W} --- catch-up (laggard):')
            print(pl.round(4).to_string())
            print(f'--- THR={thr} W={W} --- overshoot check (leader, fade):')
            print(pf.round(4).to_string())

    # ---------------- natural cell per-pair table ---------------------------------
    thr, W = NAT_THR, NAT_W
    res_lag, res_led, pl, pf = grid_store[(thr, W)]
    print(f'\n================ NATURAL CELL THR={thr} W={W}, per-pair at h={NAT_H} '
          '(LAGGARD catch-up) ================')
    rows = []
    for k, (ls, bs) in enumerate(PAIRS):
        n, m, t, p, f, pm, cv = res_lag[k][NAT_H]
        rows.append((f'{ls}->{bs}', FAMILY[(ls, bs)], n,
                     round(m, 4) if np.isfinite(m) else np.nan,
                     round(t, 2) if np.isfinite(t) else np.nan,
                     p, round(cv, 3) if np.isfinite(cv) else np.nan))
    tbl = pd.DataFrame(rows, columns=['pair','family','n','mean_vu','t',
                                      'p_placebo','cost_vu'])
    print(tbl.to_string(index=False))
    val = tbl.dropna(subset=['mean_vu'])
    npos = int((val.mean_vu > 0).sum())
    print(f'\nSign consistency at h={NAT_H}: {npos}/{len(val)} pairs positive')
    for fam in sorted(set(FAMILY.values())):
        sub = val[val.family == fam]
        print(f'  family {fam}: {int((sub.mean_vu>0).sum())}/{len(sub)} positive, '
              f'mean of means {sub.mean_vu.mean():+.4f}')
    print('\nNatural-cell pooled (laggard catch-up):')
    print(pl.round(4).to_string())
    print('Natural-cell pooled (leader fade / overshoot):')
    print(pf.round(4).to_string())
    e2c = pl.loc[NAT_H, 'mean_vu'] / pl.loc[NAT_H, 'cost_vu']
    print(f'\nEdge/cost at natural h={NAT_H}: {e2c:.2f}')

    # ---------------- ONE backtest: natural variant -------------------------------
    print(f'\n================ BACKTEST (single natural variant): THR={thr} W={W} '
          f'h={NAT_H}, laggard catch-up, per-pair lab cost ================')
    combined = None
    for k, (ls, bs) in enumerate(PAIRS):
        dL, dB, aL, aB = aligned[(ls, bs)]
        mask, sides = make_events(dL, dB, aL, aB, thr, W)
        pnl, n = lab.backtest(dB, mask, h=NAT_H, sides=sides,
                              cost=lab.cost_bps(bs), a=aB)
        print(f'  {ls}->{bs}: trades={n}, total_vu={pnl.sum():+.1f}')
        combined = pnl if combined is None else combined.add(pnl, fill_value=0.0)
    combined = combined.sort_index()
    print('\nCombined (sum across pairs) metrics:')
    print(lab.metrics(combined))
    print('Yearly Sharpe:')
    print(lab.yearly(combined).to_string())

if __name__ == '__main__':
    main()
