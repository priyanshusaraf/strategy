#!/usr/bin/env /usr/bin/python3
"""
H2 — RANGE THRUST / LIQUIDITY-VACUUM EXHAUSTION (hourly; all 9 duka + all 34 fx = 43).

Mechanism: an abnormally large fast move with a climactic close happens because the
book was thin (liquidity vacuum), not because of proportional information; when
liquidity refills, price retraces part of the vacuum move.  Fade direction is fixed
ex-ante: against the thrust.

PRE-SPECIFIED grid (reported in full, no tuning):
  Event A (single-bar climax): TrueRange[t] > k*ATR[t] AND close in the extreme 20%
      of the bar's own range in the move direction.  sides = -sign(C[t]-O[t]).
      k in {2.5, 3.5}.
  Event B (multi-bar thrust): |C[t]-C[t-4]| >= m*ATR[t].  sides = -sign(move).
      m in {3, 4.5}.
  Horizons (1,2,4,8,16,24,48).

Causality: TR[t], close-location[t], C[t]-C[t-4] are all known at the close of the
event bar; lab.atr() is already shifted one bar.  Entries are next-bar-open (lab).

NATURAL VARIANT (declared before seeing results): Event A, k=2.5, h=8 (one session
for the book to refill) -> the single cost-aware lab.backtest, per-instrument costs.

Placebo: same-count random-time events with random +/-1 sides (vectorized, drawn
with replacement; pools across instruments).  Random signs kill drift, so this tests
the conditioning (timing + direction) exactly.

Run: cd /Users/priyanshusaraf/Desktop/strategy-dev-mr/ortho && /usr/bin/python3 h2_thrust_exhaustion.py
"""
import numpy as np
import pandas as pd
import lab

HORIZONS  = (1, 2, 4, 8, 16, 24, 48)
N_PLACEBO = 300
NATURAL_H = 8
NATURAL_VARIANT = 'A k=2.5'

pd.set_option('display.width', 200)

# ---------------------------------------------------------------- events

def events_A(df, a, k):
    """Single-bar climax: TR > k*ATR, close in extreme 20% of range in move direction."""
    pc = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()],
                   axis=1).max(axis=1).values
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    av = a.values
    rng_ = h - l
    with np.errstate(invalid='ignore', divide='ignore'):
        clp = np.where(rng_ > 0, (c - l) / rng_, np.nan)
    dirn = np.sign(c - o)
    climax = ((dirn > 0) & (clp >= 0.8)) | ((dirn < 0) & (clp <= 0.2))
    mask = np.isfinite(av) & (av > 0) & np.isfinite(tr) & (tr > k * av) & climax
    sides = np.zeros(len(df))
    sides[mask] = -dirn[mask]
    return mask, sides

def events_B(df, a, m):
    """Multi-bar thrust: |C[t]-C[t-4]| >= m*ATR[t]."""
    c = df.c.values
    mv = np.full(len(df), np.nan)
    mv[4:] = c[4:] - c[:-4]
    av = a.values
    mask = np.isfinite(av) & (av > 0) & np.isfinite(mv) & (np.abs(mv) >= m * av)
    sides = np.zeros(len(df))
    sides[mask] = -np.sign(mv[mask])
    return mask, sides

VARIANTS = [('A k=2.5', events_A, 2.5), ('A k=3.5', events_A, 3.5),
            ('B m=3.0', events_B, 3.0), ('B m=4.5', events_B, 4.5)]

# ---------------------------------------------------------------- machinery

def signed_event_returns(df, mask, sides, a, h):
    """De-overlapped signed forward returns (sides * (C[p+h]-O[p+1])/ATR[p])."""
    c, o = df.c.values, df.o.values
    av = a.values
    N = len(df)
    pos = np.flatnonzero(np.asarray(mask, bool))
    pos = pos[(pos + h) < N - 1]
    pos = lab._deoverlap(pos, h)
    pos = pos[np.isfinite(av[pos]) & (av[pos] > 0)]
    if len(pos) == 0:
        return np.array([]), pos
    f = (c[pos + h] - o[pos + 1]) / av[pos]
    s = sides[pos]
    ok = np.isfinite(f)
    return (s * f)[ok], pos[ok]

def placebo_sums(df, a, h, n_ev, rng):
    """N_PLACEBO sums of n_ev random-time random-sign signed forward returns."""
    c, o = df.c.values, df.o.values
    av = a.values
    N = len(df)
    V = np.flatnonzero(np.isfinite(av) & (av > 0))
    V = V[(V + h) < N - 1]
    if len(V) == 0 or n_ev == 0:
        return np.zeros(N_PLACEBO), np.zeros(N_PLACEBO)
    Q = rng.choice(V, size=(N_PLACEBO, n_ev), replace=True)
    sg = rng.choice([-1.0, 1.0], size=(N_PLACEBO, n_ev))
    g = sg * (c[Q + h] - o[Q + 1]) / av[Q]
    return np.nansum(g, axis=1), np.isfinite(g).sum(axis=1)

def cost_vu_at_events(df, pos, a, sym):
    """Round-trip cost in vol units at each event (charged on next-bar-open entry)."""
    o = df.o.values
    av = a.values
    return (lab.cost_bps(sym) / 1e4) * np.abs(o[pos + 1]) / av[pos]

# ---------------------------------------------------------------- load

print('loading 9 duka + 34 fx ...')
data = {}
for s in lab.DUKA_SYMS:
    data[s] = lab.duka(s)
fx_list = lab.fx_syms()
for s in fx_list:
    data[s] = lab.fx(s)
atrs = {s: lab.atr(d) for s, d in data.items()}
print(f'{len(data)} instruments loaded')

ev_cache = {}   # (variant, sym) -> (mask, sides)
for vname, fn, param in VARIANTS:
    for sym, df in data.items():
        ev_cache[(vname, sym)] = fn(df, atrs[sym], param)

# ---------------------------------------------------------------- full grid

rng = np.random.default_rng(0)
grid_rows = []
inst_tables = {}     # vname -> per-instrument DataFrame at NATURAL_H

for vname, fn, param in VARIANTS:
    inst_rows = []
    for h in HORIZONS:
        stack, cost_stack = [], []
        psum = np.zeros(N_PLACEBO)
        pn   = np.zeros(N_PLACEBO)
        n_pos_inst, n_inst_used = 0, 0
        for sym, df in data.items():
            mask, sides = ev_cache[(vname, sym)]
            r, pos = signed_event_returns(df, mask, sides, atrs[sym], h)
            if len(r):
                stack.append(r)
                cost_stack.append(cost_vu_at_events(df, pos, atrs[sym], sym))
                s_, n_ = placebo_sums(df, atrs[sym], h, len(r), rng)
                psum += s_; pn += n_
                n_inst_used += 1
                if r.mean() > 0:
                    n_pos_inst += 1
            if h == NATURAL_H:
                t_i = (r.mean() / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-12)
                       if len(r) >= 3 else np.nan)
                inst_rows.append((sym, len(r),
                                  round(float(r.mean()), 4) if len(r) else np.nan,
                                  round(float(t_i), 2) if np.isfinite(t_i) else np.nan))
        allr = np.concatenate(stack) if stack else np.array([])
        allc = np.concatenate(cost_stack) if cost_stack else np.array([1.0])
        if len(allr) < 10:
            grid_rows.append((vname, h, len(allr), np.nan, np.nan, np.nan,
                              np.nan, np.nan, f'0/{n_inst_used}'))
            continue
        m_ = allr.mean()
        t_ = m_ / (allr.std(ddof=1) / np.sqrt(len(allr)) + 1e-12)
        pm = psum / np.maximum(pn, 1)
        p_ = float((np.abs(pm) >= abs(m_)).mean())
        cvu = float(allc.mean())
        grid_rows.append((vname, h, len(allr), round(float(m_), 4), round(float(t_), 2),
                          round(p_, 3), round(cvu, 3), round(float(m_) / cvu, 2),
                          f'{n_pos_inst}/{n_inst_used}'))
    inst_tables[vname] = pd.DataFrame(
        inst_rows, columns=['sym', 'n', 'mean_vu', 't']).set_index('sym')

grid = pd.DataFrame(grid_rows, columns=['variant', 'h', 'n', 'mean_vu', 'pooled_t',
                                        'p_placebo', 'cost_vu', 'edge/cost', 'pos_inst'])
print('\n================ FULL PRE-SPECIFIED GRID (pooled stacked events, signed = fade dir) ================')
print(grid.to_string(index=False))

print(f'\n================ PER-INSTRUMENT at h={NATURAL_H} (signed mean_vu in fade direction) ================')
for vname, _, _ in VARIANTS:
    tb = inst_tables[vname]
    np_ = int((tb.mean_vu > 0).sum()); nt = int(tb.mean_vu.notna().sum())
    print(f'\n--- {vname}  (positive {np_}/{nt}) ---')
    print(tb.to_string())

print('\nFAMILY FLAG: WTI/BRENT/DIESEL/GAS are one energy family; XAU/XAG one metals '
      'family; 34 FX pairs share 10 currencies (EUR/USD blocs heavily overlapping) — '
      'pooled t overstates independent evidence.')

# ---------------------------------------------------------------- one natural backtest

print(f'\n================ COST-AWARE BACKTEST: natural variant {NATURAL_VARIANT}, h={NATURAL_H} ================')
pnls, total_trades = [], 0
per_bt = []
for sym, df in data.items():
    mask, sides = ev_cache[(NATURAL_VARIANT, sym)]
    pnl, n = lab.backtest(df, mask, h=NATURAL_H, sides=sides,
                          cost=lab.cost_bps(sym), a=atrs[sym])
    pnls.append(pnl); total_trades += n
    per_bt.append((sym, n, round(float(pnl.sum()), 1)))
port = pd.concat(pnls, axis=1).fillna(0.0).sum(axis=1)
mt = lab.metrics(port)
print(f'portfolio (43 instruments, per-instrument lab.cost_bps): trades={total_trades}')
print('metrics:', mt)
print('yearly Sharpe:', dict(lab.yearly(port)))
print('\nper-instrument backtest (sym, trades, total_vu):')
print(pd.DataFrame(per_bt, columns=['sym', 'trades', 'total_vu']).to_string(index=False))

# stress x2 costs on natural variant (no iteration, just honesty check)
pnls2 = []
for sym, df in data.items():
    mask, sides = ev_cache[(NATURAL_VARIANT, sym)]
    pnl, _ = lab.backtest(df, mask, h=NATURAL_H, sides=sides,
                          cost=lab.cost_bps(sym, stress=2.0), a=atrs[sym])
    pnls2.append(pnl)
port2 = pd.concat(pnls2, axis=1).fillna(0.0).sum(axis=1)
print('\nstress x2 costs metrics:', lab.metrics(port2))
