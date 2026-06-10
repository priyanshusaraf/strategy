#!/usr/bin/env python3
"""HOURLY spread MR lab — the construction validated on 3.3y of CL-BRN hourly data.

Strategy (FROZEN from that validation, applied unchanged to every spread):
  detrend window 504 bars, z_n=336 bars, entry |z|>=2.5, exit |z|<=0.75,
  disaster 4.0, time stop 10 trading days of bars, next-bar-OPEN execution.

Data: Yahoo hourly front-month legs (730d). Rolls are NOT back-adjusted, so:
  ROLL CENSORING: any leg diff at a day boundary with |diff| > 8x rolling MAD is
  treated as a roll: that bar contributes 0 to the spread (signal AND PnL) and one
  extra cost turn is charged if a position is held (you re-roll the spread).

Costs: per unit turn = SLIP x sum(|w_i| * tick_i) dollars, SLIP=1.5 (entry+exit each).
"""
import math, os, sys
import numpy as np, pandas as pd

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_hourly")
SLIP = 1.5

TICK = dict(CL=0.01, BZ=0.01, RB=0.0001, HO=0.0001, NG=0.001,
            GC=0.10, SI=0.005, HG=0.0005, PL=0.10, PA=0.50,
            ZC=0.25, ZS=0.25, ZW=0.25, KE=0.25, ZL=0.01, ZM=0.10, ZO=0.25,
            LE=0.025, HE=0.025, GF=0.025,
            KC=0.05, CC=1.00, CT=0.01, SB=0.01, OJ=0.05, FCPO=1.0)

_cache = {}
def leg(sym):
    if sym in _cache:
        return _cache[sym]
    df = pd.read_csv(f"{DIR}/{sym}.csv")
    df["t"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("t").sort_index()[["open", "close"]]
    df = df[~df.index.duplicated(keep="last")]
    _cache[sym] = df
    return df

# legs -> weights; spreads in native quote-unit dollars
SPREADS = {
    "RB_crack":  (["RB", "CL"], [42, -1]),
    "HO_crack":  (["HO", "CL"], [42, -1]),
    "crack_321": (["RB", "HO", "CL"], [28, 14, -1]),
    "BZ_CL":     (["BZ", "CL"], [1, -1]),
    "RB_HO":     (["RB", "HO"], [42, -42]),
    "crush":     (["ZM", "ZL", "ZS"], [0.022, 0.11, -0.01]),
    "ZW_KE":     (["ZW", "KE"], [1, -1]),
    "ZC_ZW":     (["ZC", "ZW"], [1, -1]),
    "ZO_ZC":     (["ZO", "ZC"], [1, -1]),
    "LE_GF":     (["LE", "GF"], [1, -1]),
    "HE_LE":     (["HE", "LE"], [1, -1]),
    "GC_SI":     (["GC", "SI"], [1, None]),     # None -> vol-matched on first 60 sessions
    "GC_PL":     (["GC", "PL"], [1, None]),
    "SI_HG":     (["SI", "HG"], [1, None]),
    "PL_PA":     (["PL", "PA"], [1, None]),
    "KC_CC":     (["KC", "CC"], [1, None]),
    # additional real economic links buildable from the same legs
    "BOHO":      (["ZL", "HO"], [0.0766, -1]),  # soyoil $/gal vs heating oil $/gal (biodiesel arb)
    "RB_BZ":     (["RB", "BZ"], [42, -1]),      # transatlantic gasoline arb (USGC vs Brent)
    "HO_BZ":     (["HO", "BZ"], [42, -1]),      # transatlantic distillate arb
    "crack321b": (["RB", "HO", "BZ"], [28, 14, -1]),
    "NG_HO":     (["NG", "HO"], [None, None]),  # fuel-switching (vol-matched)
    "NG_CL":     (["NG", "CL"], [None, None]),  # BTU arb / fuel competition (vol-matched)
    "ZS_ZC":     (["ZS", "ZC"], [1, None]),     # acreage substitution
    "ZM_ZL":     (["ZM", "ZL"], [1, None]),     # oilshare within crush
    "ZC_SB":     (["ZC", "SB"], [1, None]),     # ethanol feedstock arb
    # ---- expanded economic relative-value set (for book diversification) ----
    # metals: every pairwise relative value within precious+base (vol-matched)
    "GC_HG":     (["GC", "HG"], [1, None]),     # gold-copper (risk vs growth metal)
    "SI_PL":     (["SI", "PL"], [1, None]),
    "HG_PL":     (["HG", "PL"], [1, None]),     # copper-platinum (industrial)
    "GC_PA":     (["GC", "PA"], [1, None]),
    "SI_PA":     (["SI", "PA"], [1, None]),
    # grains: substitution & relative value across the complex
    "ZC_KE":     (["ZC", "KE"], [1, None]),     # corn vs HRW wheat (feed)
    "ZS_ZW":     (["ZS", "ZW"], [1, None]),
    "ZW_ZL":     (["ZW", "ZL"], [1, None]),
    # softs: relative value across the soft complex
    "KC_SB":     (["KC", "SB"], [1, None]),
    "CC_SB":     (["CC", "SB"], [1, None]),
    "CT_SB":     (["CT", "SB"], [1, None]),
    "KC_CT":     (["KC", "CT"], [1, None]),
    # energy: product & quality relative value
    "RB_CL":     (["RB", "CL"], [None, None]),  # gasoline vs crude (vol-matched, not 42x)
    "HO_CL":     (["HO", "CL"], [None, None]),
    "NG_BZ":     (["NG", "BZ"], [None, None]),
}

# horizons in TRADING DAYS (validated on 3.3y CL-BRN hourly: 336 bars / ~18.6 bars-day);
# converted per spread to bars so short-session markets keep the same calendar horizon.
# volgate validated independently on CL-BRN (worst-half Sharpe sum +0.6 -> +2.5):
# enter only when trailing 5d spread vol > rolling 6m median (dislocation regime).
P = dict(detrend_days=28, z_days=18, ze=2.5, zx=0.75, zstop=4.0, tstop_days=10, volgate=True)

def build(name):
    legs, w = SPREADS[name]
    try:
        dfs = [leg(s) for s in legs]
    except FileNotFoundError:
        return None
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    if len(idx) < 2000:
        return None
    O = np.column_stack([d.loc[idx, "open"].values for d in dfs])
    C = np.column_stack([d.loc[idx, "close"].values for d in dfs])
    w = list(w)
    if w[1] is None:                        # vol-match second leg on first ~60 sessions
        n0 = min(1400, len(idx) // 4)
        v0 = np.nanstd(np.diff(C[:n0, 0]))
        v1 = np.nanstd(np.diff(C[:n0, 1]))
        w[1] = -v0 / v1
    if w[0] is None:                        # both None: vol-match leg0 too (unit gross)
        n0 = min(1400, len(idx) // 4)
        w[0] = 1.0 / np.nanstd(np.diff(C[:n0, 0]))
        w[1] = -1.0 / np.nanstd(np.diff(C[:n0, 1]))
    W = np.array(w, float)
    So, Sc = O @ W, C @ W
    # per-leg roll censoring at day boundaries
    day = idx.tz_convert("America/New_York").date
    newday = np.r_[False, day[1:] != day[:-1]]
    censor = np.zeros(len(idx), bool)
    for j in range(C.shape[1]):
        d = np.r_[0.0, np.diff(C[:, j])]
        mad = pd.Series(d).rolling(504, min_periods=100).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True).values
        big = np.abs(d) > 8 * 1.4826 * np.where(mad > 0, mad, np.nan)
        censor |= (big & newday)
    cost_unit = SLIP * sum(abs(ww) * TICK[s] for ww, s in zip(W, legs))
    return pd.DataFrame({"o": So, "c": Sc, "censor": censor}, index=idx), cost_unit

def run(name, p=P, slip_mult=1.0, half=None):
    out = build(name)
    if out is None:
        return None
    F, cost = out
    cost *= slip_mult
    dS = F["c"].diff().where(~F["censor"], 0.0)
    bars_day = len(F) / max(1, len(np.unique(F.index.date)))
    tstop = int(p["tstop_days"] * bars_day)
    z_n = max(60, int(p["z_days"] * bars_day))
    detrend = max(100, int(p["detrend_days"] * bars_day))
    mu = dS.shift(1).rolling(detrend, min_periods=detrend // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    n = len(F)
    if p.get("volgate", False):
        sig_f = dS.ewm(span=int(5 * bars_day), min_periods=50).std()
        sig_s = sig_f.rolling(int(126 * bars_day), min_periods=500).median()
        hot = (sig_f > sig_s).values
    else:
        hot = np.ones(n, bool)
    w = np.zeros(n); cur = 0; held = 0
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= p["zstop"] or held >= tstop
                    or (cur > 0 and zz >= -p["zx"]) or (cur < 0 and zz <= p["zx"])):
                cur, held = 0, 0
        if cur == 0 and zz == zz and hot[i]:
            if zz <= -p["ze"]: cur, held = 1, 0
            elif zz >= p["ze"]: cur, held = -1, 0
        w[i] = cur
    # per-trade risk normalization: units = sign / sigma_at_entry, FROZEN through the
    # trade (no intra-trade resizing). Equalizes risk per trade across vol regimes —
    # standard practice; without it high-vol episodes dominate the curve and the DD.
    sig = dS.ewm(span=int(bars_day * 63), min_periods=100).std()
    sig_safe = sig.replace(0, np.nan).ffill().bfill().values
    u = np.zeros(n)
    for i in range(1, n):
        if w[i] == 0:
            u[i] = 0.0
        elif w[i - 1] == 0 or np.sign(w[i]) != np.sign(w[i - 1]):
            u[i] = w[i] / sig_safe[i]            # fresh entry sized at current vol
        else:
            u[i] = u[i - 1]                       # hold size through the trade
    u = pd.Series(u, index=F.index)
    ua = u.shift(1)
    du = ua.diff().abs().fillna(0.0)
    roll_cost = (ua.abs() * F["censor"]).fillna(0.0)      # re-roll held position
    day_pnl = (ua * (F["c"] - F["o"])).where(~F["censor"], 0.0)
    on_pnl = (ua.shift(1) * (F["o"] - F["c"].shift(1))).where(~F["censor"], 0.0)
    pnl = day_pnl + on_pnl - cost * (du + roll_cost)
    pnl = pnl.fillna(0.0)
    # pnl is now ~unit daily-vol per active bar; scale to 10%/yr target on capital 1
    bpy = len(F) / 730 * 365
    ret = pnl * (0.10 / math.sqrt(bpy))
    return dict(ret=ret, trades=int(du.gt(0).sum() // 2), censored=int(F["censor"].sum()),
                cost=cost, bars=n, w=pd.Series(w, index=F.index))

def stats(ret, bpy):
    ann = math.sqrt(bpy)
    def block(rr):
        if len(rr) < 200 or rr.std() == 0:
            return (0.0, 0.0, 0.0)
        eq = rr.cumsum()
        dd = (eq - eq.cummax()).min()
        x = np.arange(len(eq))
        b1, b0 = np.polyfit(x, eq.values, 1)
        ssr = ((eq.values - (b1 * x + b0)) ** 2).sum()
        sst = max(((eq.values - eq.values.mean()) ** 2).sum(), 1e-12)
        return (rr.mean() / rr.std() * ann, dd, 1 - ssr / sst)
    h = len(ret) // 2
    return block(ret), block(ret.iloc[:h]), block(ret.iloc[h:])

if __name__ == "__main__":
    slip_mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(f"HOURLY OU-z MR (frozen: z336/2.5/0.75, tstop10d, next-open, slip x{slip_mult})")
    print(f"{'spread':<11}{'bars':>7}{'trades':>7}{'rolls':>6}{'fullSh':>8}{'ISsh':>7}{'OOSsh':>7}"
          f"{'maxDD':>8}{'eqR2':>6}")
    book = {}
    for nm in SPREADS:
        res = run(nm, slip_mult=slip_mult)
        if res is None:
            print(f"{nm:<11} insufficient data")
            continue
        bpy = res["bars"] / 730 * 365
        (fs, fdd, fr2), (is_, idd, _), (os_, odd, _) = stats(res["ret"], bpy)
        book[nm] = res["ret"]
        print(f"{nm:<11}{res['bars']:>7}{res['trades']:>7}{res['censored']:>6}"
              f"{fs:>8.2f}{is_:>7.2f}{os_:>7.2f}{fdd*100:>7.1f}%{fr2:>6.2f}")
    # book: align on union hourly index, sum (each already vol-targeted)
    B = pd.DataFrame(book)
    act = B.notna().sum(axis=1)
    port = B.fillna(0).sum(axis=1) / act.clip(lower=1).pow(0.5)   # ~risk-parity-ish
    port = port[act >= 4]
    bpy = len(port) / 730 * 365
    (fs, fdd, fr2), (is_, _, _), (os_, _, _) = stats(port, bpy)
    print(f"{'BOOK':<11}{len(port):>7}{'':>7}{'':>6}{fs:>8.2f}{is_:>7.2f}{os_:>7.2f}"
          f"{fdd*100:>7.1f}%{fr2:>6.2f}")
