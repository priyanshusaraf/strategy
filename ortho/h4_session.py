#!/usr/bin/env /usr/bin/python3
"""
H4 — ILLIQUID-SESSION MOVE REVERSAL (hourly, true-UTC index).

Mechanism: moves made in the illiquid session are inventory/liquidity moves (thin
books, no primary participants); when the liquid session arrives, real two-way flow
re-prices them back. Moves made in the LIQUID session are information and should
NOT revert -> control arm must show materially less reversion.

PRE-SPECIFIED DESIGN (fixed before any results; no tuning afterwards):
  Classes / windows (UTC, fixed):
    CMD (9 duka):  illiquid 22:00-07:00 (bars h in {22,23,0..6}, require >=6 bars),
                   liquid 13:00-20:00 (bars h in {13..19}); treatment event bar =
                   first liquid bar of day d; session horizon h_sess=6
                   (entry next open ~14:00 -> close ~20:00).
    FX (34):       illiquid Asia 00:00-06:00 (bars h in {0..5}, require >=4 bars),
                   liquid London/NY 07:00-17:00 (bars h in {7..16}); event bar =
                   first liquid bar of day d; session horizon h_sess=9
                   (entry next open ~08:00 -> close ~17:00).
  Treatment event: |illiquid-window net move (first open -> last close)| / ATR(event
    bar) >= thr, thr grid {1.0, 1.5, 2.0}. side = -sign(illiquid move). Causal: the
    window ends hours before the event bar; ATR is lab.atr (already shifted).
  Control event:  same thresholds on the LIQUID-session net move of day d; event bar
    = first bar of the NEXT illiquid window (entered at next session start);
    side = -sign(liquid move). Gap from window end to event bar must be <= 12h
    (drops weekends/holidays in both arms).
  Horizons reported (full grid): 2, 4, sess (6 CMD / 9 FX), 8, 12.
  NATURAL VARIANT (fixed ex ante): thr=1.5, h=h_sess -> only backtested variant,
    per-instrument costs lab.cost_bps(sym) (+2x stress line).
  Placebo: per instrument, same-count random-time valid events with RANDOM signs
    (signed fade test -> symmetric null, drift cancels); pooled p = fraction of 300
    placebo sets whose |pooled mean| >= |real pooled mean|.
  Correlation flag: same-day events across 34 FX pairs are correlated; pooled
    stacked t overstates independence -> also report date-clustered t (average
    signed return per UTC date across instruments, t over dates) at the natural
    variant, plus per-instrument sign consistency.
"""
import numpy as np
import pandas as pd
import lab

THRS = (1.0, 1.5, 2.0)
NAT_THR = 1.5
HLABELS = ('2', '4', 'sess', '8', '12')
N_PLACEBO = 300
SEED = 0

CLASSES = {
    'CMD': dict(syms=lab.DUKA_SYMS, loader=lab.duka,
                ill_hours=frozenset(list(range(22, 24)) + list(range(0, 7))),
                ill_shift=2,                       # +2h maps 22:00 of d-1 -> day d
                liq_hours=frozenset(range(13, 20)),
                h_sess=6, min_ill=6,
                max_gap=np.timedelta64(12, 'h')),
    'FX': dict(syms=lab.fx_syms(), loader=lab.fx,
               ill_hours=frozenset(range(0, 6)),
               ill_shift=0,
               liq_hours=frozenset(range(7, 17)),
               h_sess=9, min_ill=4,
               max_gap=np.timedelta64(12, 'h')),
}


def window_stats(df, sel, key):
    """Per-window: net move (first open -> last close), bar count, last bar time,
    position of first bar."""
    if sel.sum() == 0:
        return pd.DataFrame(columns=['move', 'n', 't_end', 'p_first'])
    pos = np.arange(len(df))
    d = pd.DataFrame({'o': df.o.values[sel], 'c': df.c.values[sel],
                      'p': pos[sel], 'ts': df.index[sel]}, index=key[sel])
    g = d.groupby(level=0)
    return pd.DataFrame({'move': g['c'].last() - g['o'].first(), 'n': g['c'].size(),
                         't_end': g['ts'].last(), 'p_first': g['p'].first()})


def build_events(df, cfg, av):
    """Returns (tr_pos, tr_mvu, co_pos, co_mvu): event-bar positions and the
    conditioning move in ATR units (ATR at event bar). All causal."""
    idx = df.index
    hrs = idx.hour
    ill = np.isin(hrs, list(cfg['ill_hours']))
    liq = np.isin(hrs, list(cfg['liq_hours']))
    wk_ill = (idx + pd.Timedelta(hours=cfg['ill_shift'])).normalize()
    day = idx.normalize()

    iw = window_stats(df, ill, wk_ill)
    lw = window_stats(df, liq, day)

    def make(win, min_n, ev_first, key_shift):
        """win: conditioning windows; ev_first: Series day->first bar position of the
        entry session; key_shift: days added to window key to find the event session."""
        if len(win) == 0:
            return np.array([], int), np.array([])
        ok = win[win['n'] >= min_n] if min_n else win
        if len(ok) == 0:
            return np.array([], int), np.array([])
        keys = ok.index + pd.Timedelta(days=key_shift)
        ep = ev_first.reindex(keys)
        good = ep.notna().values
        ep = ep.values[good].astype(int)
        mv = ok['move'].values[good]
        tend = ok['t_end'].values[good]
        gap_ok = (idx.values[ep] - tend) <= cfg['max_gap']
        atr_ok = np.isfinite(av[ep]) & (av[ep] > 0)
        keep = gap_ok & atr_ok & np.isfinite(mv)
        ep, mv = ep[keep], mv[keep]
        return ep, mv / av[ep]

    liq_first = lw['p_first'] if len(lw) else pd.Series(dtype=float)
    ill_first = iw['p_first'] if len(iw) else pd.Series(dtype=float)

    # treatment: illiquid window of day d -> first liquid bar of day d
    tr_pos, tr_mvu = make(iw, cfg['min_ill'], liq_first, 0)
    # control: liquid window of day d -> first illiquid bar of window d+1.
    # NOTE: no bar-count filter on the conditioning liquid window would be needed for
    # causality either way (window is in the past at event time); we apply a fixed
    # min of 5 bars to avoid holiday stub sessions. Pre-specified, not tuned.
    co_pos, co_mvu = make(lw, 5, ill_first, 1)
    return tr_pos, tr_mvu, co_pos, co_mvu


class Inst:
    def __init__(self, name, cls):
        cfg = CLASSES[cls]
        self.name, self.cls = name, cls
        self.df = cfg['loader'](name)
        self.N = len(self.df)
        self.c, self.o = self.df.c.values, self.df.o.values
        self.atr = lab.atr(self.df)
        self.av = self.atr.values
        self.h_sess = cfg['h_sess']
        self.cost = lab.cost_bps(name)
        (self.tr_pos, self.tr_mvu,
         self.co_pos, self.co_mvu) = build_events(self.df, cfg, self.av)


def deoverlap_signed(P, S, h):
    keep, last = [], -10**9
    for i, p in enumerate(P):
        if p - last >= h:
            keep.append(i); last = p
    keep = np.asarray(keep, int)
    return P[keep], S[keep]


def fwd_signed(inst, P, S, h):
    """Signed forward returns, lab.event_study conventions: next-open entry,
    close[t+h] exit, ATR(event) sizing, de-overlapped."""
    k = (P + h) < inst.N - 1
    P, S = P[k], S[k]
    if len(P) == 0:
        return np.array([]), np.array([], int)
    P, S = deoverlap_signed(P, S, h)
    f = S * (inst.c[P + h] - inst.o[P + 1]) / inst.av[P]
    m = np.isfinite(f)
    return f[m], P[m]


def main():
    rng = np.random.default_rng(SEED)
    insts = []
    for cls in ('CMD', 'FX'):
        for s in CLASSES[cls]['syms']:
            insts.append(Inst(s, cls))
    print(f'instruments: {len(insts)}  '
          f'(CMD {sum(i.cls=="CMD" for i in insts)}, FX {sum(i.cls=="FX" for i in insts)})')
    for i in insts[:3] + insts[9:11]:
        print(f'  {i.name:8s} {i.cls} bars={i.N} {i.df.index[0].date()}..{i.df.index[-1].date()} '
              f'tr_windows={len(i.tr_pos)} co_windows={len(i.co_pos)}')

    # ---------------- grid: pooled stacked stats + placebo, per scope -------------
    scopes = ('CMD', 'FX', 'ALL')
    res = {}            # (arm, thr, hlabel, scope) -> (n, mean, t, p)
    nat_rows = []       # per-instrument @ natural variant (treatment)
    nat_dates_f = []    # (dates, f) for date-clustered t @ natural (treatment, ALL)
    nat_cost_vu = []    # per-event round-trip cost in vu @ natural (treatment)

    for hlabel in HLABELS:
        # accumulators
        fs = {(arm, thr, sc): [] for arm in 'TC' for thr in THRS for sc in scopes}
        ps = {(arm, thr, sc): np.zeros(N_PLACEBO) for arm in 'TC' for thr in THRS
              for sc in scopes}
        pn = {(arm, thr, sc): 0 for arm in 'TC' for thr in THRS for sc in scopes}
        for inst in insts:
            h = inst.h_sess if hlabel == 'sess' else int(hlabel)
            V = np.flatnonzero(np.isfinite(inst.av) & (inst.av > 0))
            V = V[(V + h) < inst.N - 1]
            fwd_all = (inst.c[V + h] - inst.o[V + 1]) / inst.av[V]
            fwd_all = fwd_all[np.isfinite(fwd_all)]
            for arm in 'TC':
                ep, mv = ((inst.tr_pos, inst.tr_mvu) if arm == 'T'
                          else (inst.co_pos, inst.co_mvu))
                for thr in THRS:
                    sel = np.abs(mv) >= thr
                    P, S = ep[sel], -np.sign(mv[sel])
                    f, Pk = fwd_signed(inst, P, S, h)
                    n = len(f)
                    for sc in (inst.cls, 'ALL'):
                        fs[(arm, thr, sc)].append(f)
                    if n >= 5 and len(fwd_all) > n:
                        draws = rng.choice(fwd_all, size=(N_PLACEBO, n))
                        signs = rng.integers(0, 2, size=(N_PLACEBO, n)) * 2 - 1
                        sums = (draws * signs).sum(axis=1)
                        for sc in (inst.cls, 'ALL'):
                            ps[(arm, thr, sc)] += sums
                            pn[(arm, thr, sc)] += n
                    if (arm == 'T' and thr == NAT_THR and hlabel == 'sess'):
                        m = f.mean() if n else np.nan
                        t = (m / (f.std(ddof=1) / np.sqrt(n) + 1e-12)
                             if n >= 5 else np.nan)
                        nat_rows.append((inst.name, inst.cls, n,
                                         round(m, 4) if n else np.nan,
                                         round(t, 2) if n >= 5 else np.nan))
                        if n:
                            nat_dates_f.append(pd.Series(
                                f, index=inst.df.index[Pk].normalize()))
                            cv = (inst.cost / 1e4) * np.abs(inst.c[Pk]) / inst.av[Pk]
                            nat_cost_vu.append(cv)
        for (arm, thr, sc), lst in fs.items():
            rkey = (arm, thr, hlabel, sc)
            f = np.concatenate(lst) if lst else np.array([])
            n = len(f)
            if n < 10:
                res[rkey] = (n, np.nan, np.nan, np.nan); continue
            m = f.mean()
            t = m / (f.std(ddof=1) / np.sqrt(n) + 1e-12)
            if pn[(arm, thr, sc)] > 0:
                pm = ps[(arm, thr, sc)] / pn[(arm, thr, sc)]
                p = float((np.abs(pm) >= abs(m)).mean())
            else:
                p = np.nan
            res[rkey] = (n, round(m, 4), round(t, 2), round(p, 3))

    # ---------------- print full grid ---------------------------------------------
    for arm, label in (('T', 'TREATMENT (fade ILLIQUID-session move at liquid open)'),
                       ('C', 'CONTROL (fade LIQUID-session move at next session start)')):
        print(f'\n================ {label} ================')
        hdr = (f'{"thr":>4} {"h":>5} | ' +
               ' | '.join(f'{sc:>26}' for sc in scopes))
        print(hdr)
        print(f'{"":>4} {"":>5} | ' +
              ' | '.join(f'{"n":>7} {"mean":>7} {"t":>5} {"p":>4}' for _ in scopes))
        for thr in THRS:
            for hlabel in HLABELS:
                cells = []
                for sc in scopes:
                    n, m, t, p = res[(arm, thr, hlabel, sc)]
                    cells.append(f'{n:>7} {m if m==m else float("nan"):>7.4f} '
                                 f'{t if t==t else float("nan"):>5.2f} '
                                 f'{p if p==p else float("nan"):>4.2f}')
                print(f'{thr:>4} {hlabel:>5} | ' + ' | '.join(cells))

    # ---------------- per-instrument table @ natural variant ----------------------
    print(f'\n===== per-instrument @ NATURAL variant: thr={NAT_THR}, h=h_sess '
          f'(CMD 6 / FX 9), treatment =====')
    print(f'{"sym":8s} {"cls":3s} {"n":>5} {"mean_vu":>8} {"t":>6}')
    npos = ntot = 0
    for name, cls, n, m, t in nat_rows:
        print(f'{name:8s} {cls:3s} {n:>5} {m:>8} {t:>6}')
        if n >= 10:
            ntot += 1
            if m == m and m > 0:
                npos += 1
    print(f'sign consistency (mean_vu>0, n>=10): {npos}/{ntot}')

    # date-clustered pooled t (handles cross-instrument same-day correlation)
    allf = pd.concat(nat_dates_f)
    byday = allf.groupby(level=0).mean()
    tclu = byday.mean() / (byday.std(ddof=1) / np.sqrt(len(byday)) + 1e-12)
    print(f'date-clustered pooled (natural): n_dates={len(byday)} '
          f'mean={byday.mean():.4f} t={tclu:.2f}')

    # costs
    cv = np.concatenate(nat_cost_vu)
    n_, m_, t_, p_ = res[('T', NAT_THR, 'sess', 'ALL')]
    print(f'mean round-trip cost @ natural events: {cv.mean():.4f} vu '
          f'(CMD/FX pooled); pooled edge {m_} vu -> edge/cost = {m_ / cv.mean():.2f}')
    for sc in ('CMD', 'FX'):
        cvs = np.concatenate([c for c, r in zip(nat_cost_vu, nat_rows)
                              if r[1] == sc and len(c)]) \
            if any(r[1] == sc for r in nat_rows) else np.array([np.nan])
        nn, mm, tt, pp = res[('T', NAT_THR, 'sess', sc)]
        print(f'  {sc}: edge {mm} vu, cost {cvs.mean():.4f} vu, '
              f'edge/cost {mm / cvs.mean():.2f}')

    # ---------------- single natural-variant backtest -----------------------------
    print(f'\n===== backtest: NATURAL variant only (thr={NAT_THR}, h=h_sess, '
          f'per-sym costs) =====')
    pnls, tot_tr = [], 0
    for inst in insts:
        sides = np.zeros(inst.N)
        sel = np.abs(inst.tr_mvu) >= NAT_THR
        sides[inst.tr_pos[sel]] = -np.sign(inst.tr_mvu[sel])
        mask = sides != 0
        if mask.sum() == 0:
            continue
        pnl, ntr = lab.backtest(inst.df, mask, h=inst.h_sess, sides=sides,
                                cost=inst.cost)
        pnls.append(pnl.rename(inst.name)); tot_tr += ntr
    port = pd.concat(pnls, axis=1).fillna(0).sum(axis=1)
    print(f'trades={tot_tr}  metrics={lab.metrics(port)}')
    print('yearly sharpe:'); print(lab.yearly(port))
    # 2x cost stress (same variant)
    pnls2 = []
    for inst in insts:
        sides = np.zeros(inst.N)
        sel = np.abs(inst.tr_mvu) >= NAT_THR
        sides[inst.tr_pos[sel]] = -np.sign(inst.tr_mvu[sel])
        mask = sides != 0
        if mask.sum() == 0:
            continue
        pnl, _ = lab.backtest(inst.df, mask, h=inst.h_sess, sides=sides,
                              cost=inst.cost * 2)
        pnls2.append(pnl.rename(inst.name))
    port2 = pd.concat(pnls2, axis=1).fillna(0).sum(axis=1)
    print(f'2x cost stress: {lab.metrics(port2)}')


if __name__ == '__main__':
    main()
