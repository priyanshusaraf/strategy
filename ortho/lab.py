"""
ortho/lab.py — shared infrastructure for the ORTHOGONAL mean-reversion program.

Scope: event/state-conditional reversion (exhaustion, failed breakouts, vol shocks,
session effects, leader-laggard, run-length). Explicitly NOT z-score/band/rolling-mean
deviation strategies (forbidden by mandate; the prior program already mined that vein).

Honesty rules (inherited from the validated prior apparatus):
  * fills at NEXT BAR OPEN, never same-bar close
  * costs charged per round trip, in bps of price, conservative; stress x2
  * all PnL in VOL UNITS (move / ATR at entry) -> scale-free, works on back-adjusted
    series with negative levels
  * Yahoo front-month legs: roll gaps censored (day-boundary move > 8x MAD)
  * Dukascopy epochs are +19800s (IST) behind true UTC -> corrected here (verified
    against CME Fri 22:00 UTC close and FX weekend-gap fingerprints)

Interpreter: /usr/bin/python3 (has pandas/numpy/matplotlib).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
DUKA  = ROOT / 'data_duka'
FXD   = ROOT / 'data_fx'
YH    = ROOT / 'data_hourly'
TVDIR = Path.home() / 'Downloads' / 'mean-reversion-data'
DAILY = Path.home() / 'Downloads' / 'commodities-data' / 'daily'

DUKA_TS_OFFSET = 19800   # fetcher wrote month-starts in IST; +5h30m restores true UTC
FX_TS_OFFSET   = 19800

DUKA_SYMS = ['WTI', 'BRENT', 'GAS', 'DIESEL', 'COPPER', 'XAU', 'XAG', 'COCOA', 'SUGAR']

def fx_syms():
    return sorted(p.stem for p in FXD.glob('*.csv'))

def yahoo_syms():
    return sorted(p.stem for p in YH.glob('*.csv'))

# conservative ROUND-TRIP cost, bps of price
_FX_MAJOR = dict.fromkeys(['EURUSD','GBPUSD','USDJPY','USDCHF','USDCAD','AUDUSD',
                           'NZDUSD','EURGBP','EURJPY','EURCHF'], 1.5)
_FX_CROSS = dict.fromkeys(['AUDCAD','AUDCHF','AUDNZD','CADCHF','CADJPY','CHFJPY',
                           'EURAUD','EURCAD','EURNZD','GBPAUD','GBPCAD','GBPCHF',
                           'GBPJPY','GBPNZD','NZDCAD','NZDCHF'], 2.5)
_FX_EXOT  = dict.fromkeys(['EURNOK','EURSEK','USDNOK','USDSEK','EURPLN','EURHUF',
                           'EURCZK','AUDSGD'], 5.0)
_CMD = {'WTI':4, 'BRENT':4, 'GAS':15, 'DIESEL':10, 'COPPER':8, 'XAU':2, 'XAG':8,
        'COCOA':12, 'SUGAR':12}
_YH  = {'CL':3,'BZ':3,'RB':6,'HO':6,'NG':8,'GC':2,'SI':6,'HG':6,'PL':10,'PA':15,
        'ZC':5,'ZS':5,'ZW':6,'ZM':6,'ZL':6,'KE':8,'ZO':15,'CC':10,'KC':10,'SB':8,
        'CT':10,'OJ':20,'LE':8,'GF':12,'HE':10}
COST_BPS = {**_FX_MAJOR, **_FX_CROSS, **_FX_EXOT, **_CMD, **_YH}

def cost_bps(sym, stress=1.0):
    return COST_BPS.get(sym, 10.0) * stress

# ---------------------------------------------------------------- loaders

def _epoch_df(path, ts_offset=0):
    d = pd.read_csv(path)
    t = pd.to_datetime(d['ts'] + ts_offset, unit='s', utc=True)
    d = d.set_index(t)[['open', 'high', 'low', 'close', 'volume']]
    d.columns = list('ohlcv')
    return d[~d.index.duplicated()].sort_index()

def duka(sym):
    """12.3y hourly commodity CFD, true-UTC index, placeholder bars dropped."""
    d = _epoch_df(DUKA / f'{sym}.csv', DUKA_TS_OFFSET)
    return d[~((d.h == d.l) & (d.v == 0))].copy()

def fx(sym):
    """10.4y hourly FX, true-UTC index."""
    d = _epoch_df(FXD / f'{sym}.csv', FX_TS_OFFSET)
    return d[~((d.h == d.l) & (d.v == 0))].copy()

def yahoo(sym):
    """2y hourly front-month future. Adds 'cens' col: True on roll-gap bars."""
    d = _epoch_df(YH / f'{sym}.csv', 0)
    d = d[d.c.notna()].copy()
    dc = d.c.diff()
    day = d.index.normalize()
    day_change = np.r_[True, day[1:] != day[:-1]]
    mad = (dc - dc.median()).abs().median() + 1e-12
    d['cens'] = day_change & (dc.abs() > 8 * 1.4826 * mad)
    return d

def tv(fname):
    """TradingView export (60/15-min or 1D) from ~/Downloads/mean-reversion-data."""
    d = pd.read_csv(TVDIR / fname)
    t = pd.to_datetime(d['time'], utc=True)
    cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in d.columns]
    d = d.set_index(t)[cols]
    d.columns = list('ohlcv'[:len(cols)])
    return d[~d.index.duplicated()].sort_index()

def tv_daily(fname):
    """Long daily continuous contract (back-adjusted: LEVELS distorted, diffs valid;
    opens synthetic O=H=L=C before ~2000)."""
    d = pd.read_csv(DAILY / fname)
    t = pd.to_datetime(d['time'])
    d = d.set_index(t)[['open', 'high', 'low', 'close']]
    d.columns = list('ohlc')
    return d[~d.index.duplicated()].sort_index()

def daily_files():
    return sorted(p.name for p in DAILY.glob('*.csv'))

# ---------------------------------------------------------------- vol units

def atr(df, n=100):
    """Causal ATR (shifted 1 bar): rolling mean true range in price units."""
    pc = df.c.shift(1)
    tr = pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n // 2).mean().shift(1)

def bars_per_year(df):
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    return len(df) / max(yrs, 1e-9)

# ---------------------------------------------------------------- event study

def _deoverlap(pos, h):
    out, last = [], -10**9
    for p in pos:
        if p - last >= h:
            out.append(p); last = p
    return np.asarray(out, dtype=int)

def event_study(df, mask, horizons=(1, 2, 4, 8, 16, 24, 48), n_placebo=300, seed=0,
                a=None):
    """Forward returns (next-bar-open -> close[t+h], in ATR units) conditional on mask.

    Returns DataFrame indexed by horizon: n, mean_vu, t, p_placebo.
    p_placebo: two-sided — fraction of random-event placebo sets whose |mean| >= |real|.
    Null includes any unconditional drift, so this tests the CONDITIONING.
    """
    c, o = df.c.values, df.o.values
    a = (atr(df) if a is None else a).values
    mask = np.asarray(mask, bool)
    pos = np.flatnonzero(mask)
    N = len(df)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(a) & (a > 0))
    rows = []
    for h in horizons:
        P = _deoverlap(pos[(pos + h) < N - 1], h)
        P = P[np.isfinite(a[P]) & (a[P] > 0)]
        if len(P) < 10:
            rows.append((h, len(P), np.nan, np.nan, np.nan)); continue
        f = (c[P + h] - o[P + 1]) / a[P]
        f = f[np.isfinite(f)]
        m = f.mean()
        t = m / (f.std(ddof=1) / np.sqrt(len(f)) + 1e-12)
        V = valid[(valid + h) < N - 1]
        pm = np.empty(n_placebo)
        for i in range(n_placebo):
            Q = rng.choice(V, size=len(P), replace=False)
            g = (c[Q + h] - o[Q + 1]) / a[Q]
            pm[i] = np.nanmean(g)
        rows.append((h, len(f), round(m, 4), round(t, 2),
                     round(float((np.abs(pm) >= abs(m)).mean()), 3)))
    return pd.DataFrame(rows, columns=['h', 'n', 'mean_vu', 't', 'p_placebo']).set_index('h')

# ---------------------------------------------------------------- backtest

def backtest(df, mask, h=8, side=1, sides=None, cost=3.0, stop_vu=None, a=None):
    """Honest event backtest. Entry next bar OPEN after event bar; hold h bars
    (exit at close); ONE position at a time; size 1/ATR(entry) (PnL in vol units).
    sides: optional array aligned to df (+1/-1/0 at event bars) overriding `side`.
    cost: round-trip bps of price, charged at entry.
    Returns (bar-level pnl Series in vol units, n_trades).
    """
    o, c = df.o.values, df.c.values
    a = (atr(df) if a is None else a).values
    cens = df['cens'].values if 'cens' in df.columns else None
    mask = np.asarray(mask, bool)
    sd = None if sides is None else np.asarray(sides, float)
    pos = np.flatnonzero(mask)
    N = len(df)
    pnl = np.zeros(N)
    until = -1; n = 0
    for p in pos:
        if p <= until or p + 2 >= N or not np.isfinite(a[p]) or a[p] <= 0:
            continue
        s = sd[p] if sd is not None else side
        if not np.isfinite(s) or s == 0:
            continue
        end = min(p + h, N - 1)
        entry = o[p + 1]
        prev = entry
        cum = 0.0
        for j in range(p + 1, end + 1):
            step = 0.0 if (cens is not None and cens[j]) else s * (c[j] - prev) / a[p]
            pnl[j] += step; cum += step; prev = c[j]
            if stop_vu is not None and cum <= -stop_vu:
                end = j; break
        pnl[p + 1] -= (cost / 1e4) * abs(entry) / a[p]
        until = end; n += 1
    return pd.Series(pnl, index=df.index), n

def metrics(pnl):
    idx = pnl.index
    yrs = (idx[-1] - idx[0]).days / 365.25
    bpy = len(pnl) / max(yrs, 1e-9)
    sd = pnl.std()
    def shp(x):
        s = x.std()
        return float(x.mean() / s * np.sqrt(bpy)) if s > 0 else 0.0
    eq = pnl.cumsum()
    n = len(eq)
    ddmin = float((eq - eq.cummax()).min())
    tot = float(eq.iloc[-1])
    x = np.arange(n, dtype=float)
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, eq.values, rcond=None)
    resid = eq.values - A @ coef
    r2 = float(1 - resid.var() / eq.values.var()) if eq.values.var() > 0 else 0.0
    return dict(sharpe=round(shp(pnl), 2),
                h1=round(shp(pnl.iloc[:n // 2]), 2),
                h2=round(shp(pnl.iloc[n // 2:]), 2),
                eqR2=round(r2, 2),
                gain_pain=round(tot / abs(ddmin), 2) if ddmin < 0 else float('inf'),
                maxdd_vu=round(ddmin, 1), total_vu=round(tot, 1))

def yearly(pnl):
    g = pnl.groupby(pnl.index.year)
    bpy = bars_per_year(pnl.to_frame())
    return g.apply(lambda x: round(float(x.mean() / x.std() * np.sqrt(bpy)), 2)
                   if x.std() > 0 else 0.0)

def shuffle_test(df, mask, h, side=1, sides=None, cost=3.0, n_shuffle=100, seed=1):
    """Randomization sanity check: same NUMBER of events at random times.
    Returns (real_total_vu, fraction of shuffles with total >= real)."""
    real, _ = backtest(df, mask, h=h, side=side, sides=sides, cost=cost)
    rt = real.sum()
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, bool)
    k = int(mask.sum()); N = len(df)
    tots = []
    for _ in range(n_shuffle):
        m2 = np.zeros(N, bool)
        m2[rng.choice(N - 2, size=min(k, N - 2), replace=False)] = True
        s2 = None
        if sides is not None:
            s2 = np.zeros(N)
            s2[m2] = rng.choice([-1, 1], size=m2.sum())
        p2, _ = backtest(df, m2, h=h, side=side, sides=s2, cost=cost)
        tots.append(p2.sum())
    return float(rt), float((np.asarray(tots) >= rt).mean())
