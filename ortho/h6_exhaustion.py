#!/usr/bin/env /usr/bin/python3
"""
H6 — RUN-LENGTH / TREND EXHAUSTION + NONLINEAR RESPONSE CURVE
(hourly; all 9 duka + all 34 fx = 43 instruments).

Mechanism: extended one-way runs exhaust the marginal buyer/seller and accumulate
profit-taking pressure; reversion should appear only in the extreme tail of the
move distribution (nonlinear), not as a linear AR(1).  Fade direction fixed
ex-ante: against the run / move.

PRE-SPECIFIED grid (reported in full, no tuning):
  Event A (run-length): n consecutive same-sign hourly close-to-close changes,
      n in {6, 9, 12}; a zero change breaks the run.  sides = -sign(run).
  Event B (vol-adjusted overextension): |C[t]-C[t-24]| >= m*ATR[t] AND the maximum
      close-path retracement against the move within the 24-bar window is < 30% of
      |move| (i.e. the move was one-way).  m in {3, 4, 5}.  sides = -sign(move).
  Horizons (4, 8, 16, 24, 48).

RESPONSE CURVE (key falsifier): non-overlapping 24-bar samples of
  move = (C[t]-C[t-24])/ATR[t], pooled across all 43 instruments, bucketed into
  pooled deciles; table of mean 24-bar forward vu (raw, and signed by sign(move) =
  continuation convention).  Reversion-in-tails predicts NEGATIVE continuation-signed
  means ONLY in deciles 1 and 10.

Causality: run length, C[t]-C[t-24], and intra-window retracement use closes up to
the event bar only; lab.atr() is already shifted one bar.  Entries next-bar-open.

NATURAL VARIANT (declared before seeing results): Event B, m=4, h=24 — fade a
one-way >=4-ATR day-length move over the following day; this matches the
response-curve window exactly.  -> the single cost-aware lab.backtest.

Placebo: same-count random-time events with random +/-1 sides, pooled across
instruments (random signs kill drift -> tests the conditioning: timing+direction).

Run: cd /Users/priyanshusaraf/Desktop/strategy-dev-mr/ortho && /usr/bin/python3 h6_exhaustion.py
"""
import numpy as np
import pandas as pd
import lab

HORIZONS  = (4, 8, 16, 24, 48)
N_PLACEBO = 300
NATURAL_H = 24
NATURAL_VARIANT = 'B m=4'
W = 24                      # overextension / response-curve window

pd.set_option('display.width', 220)

# ---------------------------------------------------------------- events

def run_length(c):
    """Per-bar length of the current same-sign close-to-close run (0 if flat/nan)."""
    s = pd.Series(np.sign(np.diff(c, prepend=np.nan)))
    grp = (s != s.shift()).cumsum()
    rl = (s.groupby(grp).cumcount() + 1).astype(float)
    rl[(s == 0) | s.isna()] = 0.0
    return rl.values, s.fillna(0.0).values

def events_A(df, a, n):
    """Run-length exhaustion: >= n consecutive same-sign hourly closes."""
    rl, sgn = run_length(df.c.values)
    av = a.values
    mask = np.isfinite(av) & (av > 0) & (rl >= n)
    sides = np.zeros(len(df))
    sides[mask] = -sgn[mask]
    return mask, sides

def events_B(df, a, m):
    """One-way overextension: |C[t]-C[t-24]| >= m*ATR[t], retracement < 30% of move."""
    c = df.c.values
    av = a.values
    N = len(df)
    mask = np.zeros(N, bool)
    sides = np.zeros(N)
    if N <= W + 2:
        return mask, sides
    mv = np.full(N, np.nan)
    mv[W:] = c[W:] - c[:-W]
    # close-path retracement within each 25-point window ending at t
    win = np.lib.stride_tricks.sliding_window_view(c, W + 1)        # rows end at t=W..N-1
    dd_dn = (np.maximum.accumulate(win, axis=1) - win).max(axis=1)  # pullback in up-move
    dd_up = (win - np.minimum.accumulate(win, axis=1)).max(axis=1)  # bounce in down-move
    retr = np.full(N, np.nan)
    retr[W:] = np.where(mv[W:] > 0, dd_dn, dd_up)
    ok = np.isfinite(av) & (av > 0) & np.isfinite(mv)
    mask = ok & (np.abs(mv) >= m * av) & (retr < 0.3 * np.abs(mv))
    sides[mask] = -np.sign(mv[mask])
    return mask, sides

VARIANTS = [('A n=6',  events_A, 6),  ('A n=9', events_A, 9), ('A n=12', events_A, 12),
            ('B m=3',  events_B, 3.0), ('B m=4', events_B, 4.0), ('B m=5', events_B, 5.0)]

# ---------------------------------------------------------------- machinery (h2-style)

def signed_event_returns(df, mask, sides, a, h):
    """De-overlapped signed forward returns: sides * (C[p+h]-O[p+1])/ATR[p]."""
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
    o = df.o.values
    av = a.values
    return (lab.cost_bps(sym) / 1e4) * np.abs(o[pos + 1]) / av[pos]

# ---------------------------------------------------------------- load

print('loading 9 duka + 34 fx ...')
data = {}
for s in lab.DUKA_SYMS:
    data[s] = lab.duka(s)
for s in lab.fx_syms():
    data[s] = lab.fx(s)
atrs = {s: lab.atr(d) for s, d in data.items()}
print(f'{len(data)} instruments loaded')

ev_cache = {}
for vname, fn, param in VARIANTS:
    for sym, df in data.items():
        ev_cache[(vname, sym)] = fn(df, atrs[sym], param)

# ---------------------------------------------------------------- response curve

print(f'\n================ RESPONSE CURVE: pooled deciles of {W}-bar move/ATR '
      f'(non-overlapping samples) ================')
mvs, fwds, fams = [], [], []
for sym, df in data.items():
    c, o = df.c.values, df.o.values
    av = atrs[sym].values
    N = len(df)
    idx = np.arange(W, N - W - 1, W)           # non-overlapping forward windows
    if len(idx) == 0:
        continue
    mv = (c[idx] - c[idx - W]) / av[idx]
    fwd = (c[idx + W] - o[idx + 1]) / av[idx]
    ok = np.isfinite(mv) & np.isfinite(fwd) & np.isfinite(av[idx]) & (av[idx] > 0)
    mvs.append(mv[ok]); fwds.append(fwd[ok])
    fams.append(np.full(ok.sum(), 'duka' if sym in lab.DUKA_SYMS else 'fx'))
MV = np.concatenate(mvs); FW = np.concatenate(fwds); FAM = np.concatenate(fams)
edges = np.quantile(MV, np.linspace(0, 1, 11))
dec = np.clip(np.searchsorted(edges, MV, side='right') - 1, 0, 9)

def _row(sel, label):
    f, m_ = FW[sel], MV[sel]
    cont = f * np.sign(m_)                     # continuation-signed forward
    tf = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
    tc = cont.mean() / (cont.std(ddof=1) / np.sqrt(len(cont)) + 1e-12)
    return (label, len(f), round(float(m_.min()), 2), round(float(m_.max()), 2),
            round(float(f.mean()), 4), round(float(tf), 2),
            round(float(cont.mean()), 4), round(float(tc), 2))

rc_rows = [_row(dec == d, f'D{d+1}') for d in range(10)]
rc_rows += [_row(MV <= np.quantile(MV, 0.02), 'bot2%'),
            _row(MV >= np.quantile(MV, 0.98), 'top2%')]
rc = pd.DataFrame(rc_rows, columns=['bucket', 'n', 'mv_lo', 'mv_hi',
                                    'fwd_mean_vu', 't_fwd', 'cont_mean_vu', 't_cont'])
print(rc.to_string(index=False))
print('reversion-in-tails prediction: cont_mean_vu < 0 ONLY in D1/D10 (and 2% tails).')

print('\n-- tails by family (cont_mean_vu, t) --')
for fam in ('duka', 'fx'):
    for lab_, sel in (('D1', (dec == 0) & (FAM == fam)), ('D10', (dec == 9) & (FAM == fam))):
        f = FW[sel] * np.sign(MV[sel])
        t_ = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        print(f'  {fam:4s} {lab_:3s}: n={len(f):5d}  cont={f.mean():+.4f}  t={t_:+.2f}')

# ---------------------------------------------------------------- full grid

rng = np.random.default_rng(0)
grid_rows = []
inst_tables = {}

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

print('\nFAMILY FLAG: WTI/BRENT/DIESEL/GAS one energy family; XAU/XAG one metals family; '
      '34 FX pairs share 10 currencies (EUR/USD blocs heavily overlapping) — pooled t '
      'overstates independent evidence.')

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

pnls2 = []
for sym, df in data.items():
    mask, sides = ev_cache[(NATURAL_VARIANT, sym)]
    pnl, _ = lab.backtest(df, mask, h=NATURAL_H, sides=sides,
                          cost=lab.cost_bps(sym, stress=2.0), a=atrs[sym])
    pnls2.append(pnl)
port2 = pd.concat(pnls2, axis=1).fillna(0.0).sum(axis=1)
print('\nstress x2 costs metrics:', lab.metrics(port2))
