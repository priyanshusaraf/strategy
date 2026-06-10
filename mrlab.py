#!/usr/bin/env python3
"""
mrlab — mean-reversion research lab for economically-linked commodity spreads.

DATA REALITY (drives the whole design):
  The continuous contracts in ~/Downloads/commodities-data/daily are BACK-ADJUSTED.
  Back-adjustment preserves daily $ CHANGES (true PnL of a rolled position) but makes
  LEVELS meaningless far from the present (ZM/ULS/CC/BRN go negative for years).
  Therefore:
    - a spread is defined by its daily $ change:  dS_t = sum_i w_i * dP_(i,t)
    - the signal runs on the synthetic level S = cumsum(dS) using ROLLING stats only
      (z-scores are location-invariant, so the arbitrary level offset cancels)
    - sizing is RISK-based (units scaled to hit a target daily $ vol), never notional-based
    - costs are risk-proportional: trading |du| units costs lam * sigma_t * |du| dollars
      (lam=0.05 ~ paying 5% of one daily stdev per unit turned — roughly 1 tick on liquid mkts;
       stress at 0.10)

Causality: every stat uses data through t-1 to set the position that earns dS_t.
"""
import glob, math, os
import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/Downloads/commodities-data/daily")
DATA2 = os.path.expanduser(
    "~/Desktop/internship-final-reports/data/raw/more-mean-reversion-data")
_cache = {}

def leg_df(sym):
    """Daily open+close of one continuous contract, e.g. 'RB1'. Cached."""
    if sym in _cache:
        return _cache[sym]
    hits = glob.glob(f"{DATA}/*_{sym}!, 1D.csv") or glob.glob(f"{DATA2}/*_{sym}!, 1D.csv")
    if not hits:
        raise FileNotFoundError(sym)
    df = pd.read_csv(hits[0])
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = df["time"]
    # epoch-seconds (more-mean-reversion-data export) vs ISO date string (main daily set)
    if pd.api.types.is_numeric_dtype(tcol) or tcol.astype(str).str.fullmatch(r"\d{9,11}").all():
        df["date"] = pd.to_datetime(pd.to_numeric(tcol, errors="coerce"),
                                    unit="s", errors="coerce").dt.normalize()  # floor to calendar date
    else:
        df["date"] = pd.to_datetime(tcol.astype(str).str[:10], errors="coerce")
    out = df.dropna(subset=["date", "close", "open"]).set_index("date")[["open", "close"]].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    _cache[sym] = out
    return out

def leg(sym):
    return leg_df(sym)["close"]

def combo(legs, weights):
    """Spread open/close levels (back-adjusted offsets cancel in any rolling stat).
    Returns DataFrame with columns o, c."""
    O = pd.concat([leg_df(l)["open"] for l in legs], axis=1, keys=range(len(legs))).dropna()
    C = pd.concat([leg_df(l)["close"] for l in legs], axis=1, keys=range(len(legs))).dropna()
    idx = O.index.intersection(C.index)
    So = sum(w * O.loc[idx, i] for i, w in enumerate(weights))
    Sc = sum(w * C.loc[idx, i] for i, w in enumerate(weights))
    return pd.DataFrame({"o": So, "c": Sc})

def logratio(a, b):
    """log(A/B) open/close levels — only valid where both legs positive."""
    A, B = leg_df(a), leg_df(b)
    df = A.join(B, lsuffix="_a", rsuffix="_b").dropna()
    df = df[(df > 0).all(axis=1)]
    return pd.DataFrame({"o": np.log(df.open_a) - np.log(df.open_b),
                         "c": np.log(df.close_a) - np.log(df.close_b)})

# ---------------------------------------------------------------------------
# UNIVERSE — economically linked spreads only. Fixed weights (no estimated betas).
# 42 gal/bbl for cracks; gasoil ~7.45 bbl/tonne.
# ---------------------------------------------------------------------------
UNIVERSE = {
    # energy: refinery economics & location/quality arb (canonical NYMEX cracks: CL/HO legs
    # from DATA2 -> same venue, synchronized settles -> same-close execution is legitimate)
    "RB_crack":   lambda: combo(["RB1", "CL1"], [42, -1]),
    "HO_crack":   lambda: combo(["HO1", "CL1"], [42, -1]),
    "GO_crack":   lambda: combo(["ULS1", "BRN1"], [1 / 7.45, -1]),   # ASYNC settles - open exec only
    "crack_321":  lambda: combo(["RB1", "HO1", "CL1"], [28, 14, -1]),
    "BRN_WTI":    lambda: combo(["BRN1", "WBS1"], [1, -1]),
    "RB_HO":      lambda: combo(["RB1", "HO1"], [42, -42]),
    "CL_cal":     lambda: combo(["CL1", "CL2"], [1, -1]),
    "HO_cal":     lambda: combo(["HO1", "HO2"], [42, -42]),
    "KC_cal":     lambda: combo(["KC1", "KC2"], [1, -1]),
    "OJ_cal":     lambda: combo(["OJ1", "OJ2"], [1, -1]),
    "FCPO_cal":   lambda: combo(["FCPO1", "FCPO2"], [1, -1]),
    # energy: storage / calendar
    "NG_cal":     lambda: combo(["NG1", "NG2"], [1, -1]),
    "RB_cal":     lambda: combo(["RB1", "RB2"], [42, -42]),
    "BRN_cal":    lambda: combo(["BRN1", "BRN2"], [1, -1]),
    "WBS_cal":    lambda: combo(["WBS1", "WBS2"], [1, -1]),
    "ULS_cal":    lambda: combo(["ULS1", "ULS2"], [1 / 7.45, -1 / 7.45]),
    # grains & oilseeds: processing margins and substitution
    "crush":      lambda: combo(["ZM1", "ZL1", "ZS1"], [0.022, 0.11, -0.01]),
    "ZW_KE":      lambda: combo(["ZW1", "KE1"], [1, -1]),       # SRW vs HRW wheat
    "ZC_ZW":      lambda: combo(["ZC1", "ZW1"], [1, -1]),       # feed substitution
    "ZO_ZC":      lambda: combo(["ZO1", "ZC1"], [1, -1]),       # feed substitution
    "ZS_cal":     lambda: combo(["ZS1", "ZS2"], [1, -1]),
    "ZC_cal":     lambda: combo(["ZC1", "ZC2"], [1, -1]),
    "ZW_cal":     lambda: combo(["ZW1", "ZW2"], [1, -1]),
    # livestock: feeding economics
    "LE_GF":      lambda: combo(["LE1", "GF1"], [1, -1]),       # live vs feeder cattle
    "HE_LE":      lambda: combo(["HE1", "LE1"], [1, -1]),       # hogs vs cattle
    "LE_cal":     lambda: combo(["LE1", "LE2"], [1, -1]),
    "HE_cal":     lambda: combo(["HE1", "HE2"], [1, -1]),
    # more calendars (the carry-floor trade applies to every storable)
    "ZL_cal":     lambda: combo(["ZL1", "ZL2"], [1, -1]),
    "ZM_cal":     lambda: combo(["ZM1", "ZM2"], [1, -1]),
    "KE_cal":     lambda: combo(["KE1", "KE2"], [1, -1]),
    "ZO_cal":     lambda: combo(["ZO1", "ZO2"], [1, -1]),
    "GF_cal":     lambda: combo(["GF1", "GF2"], [1, -1]),
    "CT_cal":     lambda: combo(["CT1", "CT2"], [1, -1]),
    "CC_cal":     lambda: combo(["CC1", "CC2"], [1, -1]),
    "SB_cal":     lambda: combo(["SB1", "SB2"], [1, -1]),
    "UHO_cal":    lambda: combo(["UHO1", "UHO2"], [42, -42]),
    "HG_cal":     lambda: combo(["HG1", "HG2"], [1, -1]),
    "PL_cal":     lambda: combo(["PL1", "PL2"], [1, -1]),
    "PA_cal":     lambda: combo(["PA1", "PA2"], [1, -1]),
    # metals: relative value (log-ratio — legs never negative)
    "GC_SI":      lambda: logratio("GC1", "SI1"),
    "SI_HG":      lambda: logratio("SI1", "HG1"),
    "GC_PL":      lambda: logratio("GC1", "PL1"),
    "PL_PA":      lambda: logratio("PL1", "PA1"),
    "GC_cal":     lambda: logratio("GC1", "GC2"),
    "SI_cal":     lambda: logratio("SI1", "SI2"),
}

# ---------------------------------------------------------------------------
# signal components (all causal: stats through t-1 decide position earning dS_t)
# ---------------------------------------------------------------------------
def zscore(S, n):
    m = S.shift(1).rolling(n).mean()
    sd = S.shift(1).rolling(n).std()
    return (S - m) / sd

def deseasonalize(S, years=5, min_years=3):
    """Causal day-of-year seasonal removal on the detrended level.
    detr = S - 252d rolling mean; seasonal(t) = mean of detr at ~same doy in past `years`;
    returns S - seasonal (falls back to S where history is short)."""
    detr = S - S.shift(1).rolling(252, min_periods=100).mean()
    vals = detr.values
    idx = S.index
    rows = []
    for k in range(1, years + 1):
        tgt = idx - pd.DateOffset(years=k)
        pos = np.clip(idx.searchsorted(tgt), 0, len(idx) - 1)
        good = np.abs((idx[pos] - tgt).days) <= 7
        rows.append(np.where(good, vals[pos], np.nan))
    mat = np.vstack(rows)
    cnt = (~np.isnan(mat)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        seasonal = np.where(cnt >= min_years, np.nanmean(mat, axis=0), 0.0)
    seasonal = np.nan_to_num(seasonal)
    return S - pd.Series(seasonal, index=idx)

def ou_halflife(S, win):
    """Rolling OU half-life of S (window through t-1): dS = a + b*S_lag, HL = -ln2/b."""
    x = S.shift(1)            # everything through t-1
    d = x.diff()
    lag = x.shift(1)
    mlag = lag.rolling(win).mean()
    md = d.rolling(win).mean()
    cov = (lag * d).rolling(win).mean() - mlag * md
    var = lag.rolling(win).var()
    b = cov / var
    hl = -math.log(2) / b
    hl[b >= 0] = np.nan
    return hl

# ---------------------------------------------------------------------------
# strategies: map synthetic level S -> target weight in [-1, 1]
# ---------------------------------------------------------------------------
def strat_binary(S, p):
    """Baseline: enter |z|>=ze, exit |z|<=zx / stop / time / gate-break."""
    z = zscore(S, p["z_n"]).values
    hl = ou_halflife(S, p["hl_win"]).values if p.get("use_gate", True) else None
    n = len(S)
    w = np.zeros(n)
    cur, held = 0, 0
    for i in range(max(p["z_n"], p.get("hl_win", 0)) + 2, n):
        gate = True
        tstop = p.get("t_stop", 60)
        if hl is not None:
            gate = (hl[i] == hl[i]) and p["hl_lo"] <= hl[i] <= p["hl_hi"]
            if hl[i] == hl[i]:
                tstop = max(5, int(p["t_mult"] * hl[i]))
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= p["z_stop"] or held >= tstop or not gate
                    or (cur > 0 and zz >= -p["z_exit"]) or (cur < 0 and zz <= p["z_exit"])):
                cur, held = 0, 0
        if cur == 0 and gate and zz == zz:
            if zz <= -p["z_entry"]:
                cur, held = 1, 0
            elif zz >= p["z_entry"]:
                cur, held = -1, 0
        w[i] = cur
    return pd.Series(w, index=S.index)

def strat_linear(S, p):
    """Continuous: w = -clip(z/zcap, -1, 1), deadband |z|<db -> 0, optional OU gate."""
    z = zscore(S, p["z_n"])
    w = (-(z / p["z_cap"])).clip(-1, 1)
    w[z.abs() < p.get("deadband", 0.0)] = 0.0
    if p.get("use_gate", False):
        hl = ou_halflife(S, p["hl_win"])
        ok = (hl >= p["hl_lo"]) & (hl <= p["hl_hi"])
        w[~ok] = 0.0
    return w.fillna(0.0)

def strat_linear_ens(S, p):
    """Ensemble of z lookbacks averaged — robustness over any single window."""
    ws = []
    for n in p["z_ns"]:
        z = zscore(S, n)
        wi = (-(z / p["z_cap"])).clip(-1, 1)
        wi[z.abs() < p.get("deadband", 0.0)] = 0.0
        ws.append(wi)
    w = pd.concat(ws, axis=1).mean(axis=1)
    if p.get("use_gate", False):
        hl = ou_halflife(S, p["hl_win"])
        ok = (hl >= p["hl_lo"]) & (hl <= p["hl_hi"])
        w[~ok] = 0.0
    return w.fillna(0.0)

STRATS = {"binary": strat_binary, "linear": strat_linear, "linear_ens": strat_linear_ens}

# ---------------------------------------------------------------------------
# execution & metrics
# ---------------------------------------------------------------------------
# spreads whose legs settle in the same venue/window: exchange-listed spread instruments
# exist and fill at the settle difference, so same-close execution is legitimate.
SYNC = {
    "RB_crack", "HO_crack", "crack_321", "RB_HO", "CL_cal", "HO_cal", "RB_cal",
    "NG_cal", "BRN_WTI", "BRN_cal", "ULS_cal", "WBS_cal", "UHO_cal",
    "crush", "ZW_KE", "ZC_ZW", "ZO_ZC", "ZS_cal", "ZC_cal", "ZW_cal", "ZL_cal",
    "ZM_cal", "KE_cal", "ZO_cal",
    "LE_GF", "HE_LE", "LE_cal", "HE_cal", "GF_cal",
    "GC_SI", "GC_cal", "SI_cal", "HG_cal", "PL_PA", "PL_cal", "PA_cal",
    "CC_cal", "CT_cal", "SB_cal", "KC_cal", "OJ_cal",
}

def backtest(F, strat, p, lam=0.05, seasonal=False, band=0.0, vol_span=63,
             ann_vol_target=0.10, start=None, detrend=0, exec_mode="open"):
    """NEXT-OPEN execution: signal at close t -> trade at open t+1 -> hold to open t+2.
    Artifact-free (bar opens are synchronous across legs even where settles are not)
    while losing only overnight signal decay.
    F: DataFrame with columns o, c (spread open/close levels, offset-arbitrary).
    detrend>0: signal level = cumsum of drift-adjusted close changes (kills roll-carry
    drift that a lagging z-score otherwise fights on back-adjusted calendars).
    Returns daily $ PnL series (target daily vol = ann_vol_target/sqrt(252) on capital 1)."""
    F = F.dropna()
    if start:
        F = F[F.index >= start]
    dS = F["c"].diff()
    if detrend > 0:
        mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
        S = (dS - mu).cumsum()
    else:
        S = F["c"]
    if seasonal:
        S = deseasonalize(S)
    w = STRATS[strat](S, p).clip(-1, 1)
    sig = dS.ewm(span=vol_span, min_periods=20).std()             # through close t
    tgt = ann_vol_target / math.sqrt(252)
    units = (w * tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if p.get("rebal_dow") is not None:   # weekly inertia: only update target on this weekday
        keep = units.index.dayofweek == p["rebal_dow"]
        units = units.where(keep).ffill().fillna(0.0)
    if band > 0:   # trade band: hold current units until target drifts by > band * full-size
        u = units.values.copy()
        full = (tgt / sig).replace([np.inf, -np.inf], np.nan).fillna(0).values
        held = 0.0
        for i in range(len(u)):
            if full[i] > 0 and abs(u[i] - held) > band * full[i]:
                held = u[i]
            u[i] = held
        units = pd.Series(u, index=units.index)
    if exec_mode == "close":
        # settlement-order fill at the same close the signal uses — ONLY valid for SYNC
        # spreads (exchange-listed spread instruments price at the settle difference)
        du = units.diff().abs().fillna(0.0)
        pnl = units.shift(1) * dS - lam * sig.shift(1) * du
    else:
        ua = units.shift(1)                       # position held during day t (set at open t)
        du = ua.diff().abs().fillna(0.0)          # traded at open t
        pnl = (ua * (F["c"] - F["o"])              # day session
               + ua.shift(1) * (F["o"] - F["c"].shift(1))   # overnight with previous position
               - lam * sig.shift(1) * du)          # cost at the open
    return pnl.fillna(0.0)

def meta_filter(r, lookback=756, min_sharpe=0.0):
    """Causal spread-selection overlay: scale today's return by whether the strategy's own
    trailing `lookback`-day Sharpe (through t-1) clears min_sharpe. Kills persistent bleeders
    without lookahead."""
    mu = r.shift(1).rolling(lookback, min_periods=lookback // 2).mean()
    sd = r.shift(1).rolling(lookback, min_periods=lookback // 2).std()
    trailing = mu / sd * math.sqrt(252)
    on = (trailing > min_sharpe).astype(float)
    return r * on, on

def metrics(r, is_frac=0.6):
    """Full / IS / OOS metrics. r = daily return series (capital=1)."""
    nz = r[r != 0]
    if len(nz) < 100:
        return None
    r = r.loc[nz.index[0]:]
    eq = (1 + r).cumprod()
    def block(rr):
        if len(rr) < 50 or rr.std() == 0:
            return dict(sharpe=0.0, cagr=0.0, maxdd=0.0, r2=0.0)
        e = (1 + rr).cumprod()
        sh = rr.mean() / rr.std() * math.sqrt(252)
        cagr = e.iloc[-1] ** (252 / len(rr)) - 1 if e.iloc[-1] > 0 else -1.0
        dd = (e / e.cummax() - 1).min()
        le = np.log(e.clip(lower=1e-9).values)
        x = np.arange(len(le))
        b1, b0 = np.polyfit(x, le, 1)
        ssr = ((le - (b1 * x + b0)) ** 2).sum()
        sst = max(((le - le.mean()) ** 2).sum(), 1e-12)
        return dict(sharpe=sh, cagr=cagr, maxdd=dd, r2=1 - ssr / sst)
    cut = int(len(r) * is_frac)
    out = dict(full=block(r), IS=block(r.iloc[:cut]), OOS=block(r.iloc[cut:]),
               eq=eq, n=len(r), trades=int((r != 0).astype(int).diff().abs().sum() // 2))
    return out
