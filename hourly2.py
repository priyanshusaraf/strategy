#!/usr/bin/env python3
"""Construction selection on the INDEPENDENT 3.3y CL-BRN hourly set (validation data),
before re-touching the 20-spread panel:
  sizing:   constant-unit vs per-trade risk-normalized
  vol gate: none vs enter only when trailing 5d spread vol > rolling 6m median
  z window: 9/12/18/24 trading days (ze=2.5, zx=0.75)
Decision rule: pick the construction with the best WORST-half Sharpe summed across
z windows (robustness, not peak)."""
import math
import numpy as np, pandas as pd

F = pd.read_csv("/Users/priyanshusaraf/Desktop/internship-final-reports/data/raw/cl_brn_spread_60.csv")
F["time"] = pd.to_datetime(F["time"], utc=True)
F = F.set_index("time").sort_index()[["open", "close"]].dropna()
dS = F["close"].diff()
BARS_DAY = len(F) / max(1, len(np.unique(F.index.date)))
BPY = len(F) / ((F.index.max() - F.index.min()).days / 365.25)
COST = 0.03   # $: ~1.5 ticks per leg

def run(zd, sizing="pertrade", volgate=False, ze=2.5, zx=0.75):
    z_n = int(zd * BARS_DAY)
    dt = int(28 * BARS_DAY)
    mu = dS.shift(1).rolling(dt, min_periods=dt // 2).mean().fillna(0.0)
    S = (dS - mu).cumsum()
    m = S.shift(1).rolling(z_n).mean()
    sd = S.shift(1).rolling(z_n).std()
    z = ((S - m) / sd).values
    sig_f = dS.ewm(span=int(5 * BARS_DAY), min_periods=50).std()        # fast (5d)
    sig_s = sig_f.rolling(int(126 * BARS_DAY), min_periods=500).median() # slow (6m)
    hot = (sig_f > sig_s).values
    sig = dS.ewm(span=int(63 * BARS_DAY), min_periods=100).std()
    sig_safe = sig.replace(0, np.nan).ffill().bfill().values
    n = len(F); w = np.zeros(n); cur = 0; held = 0
    tstop = int(10 * BARS_DAY)
    for i in range(z_n + 2, n):
        zz = z[i]
        if cur != 0:
            held += 1
            if (zz != zz or abs(zz) >= 4.0 or held >= tstop
                    or (cur > 0 and zz >= -zx) or (cur < 0 and zz <= zx)):
                cur, held = 0, 0
        if cur == 0 and zz == zz and (not volgate or hot[i]):
            if zz <= -ze: cur, held = 1, 0
            elif zz >= ze: cur, held = -1, 0
        w[i] = cur
    u = np.zeros(n)
    for i in range(1, n):
        if w[i] == 0:
            u[i] = 0.0
        elif w[i - 1] == 0:
            u[i] = w[i] / (sig_safe[i] if sizing == "pertrade" else 1.0)
        else:
            u[i] = u[i - 1]
    u = pd.Series(u, index=F.index)
    ua = u.shift(1); du = ua.diff().abs().fillna(0.0)
    pnl = (ua * (F["close"] - F["open"]) + ua.shift(1) * (F["open"] - F["close"].shift(1))
           - COST * du * (1.0 if sizing != "pertrade" else 1.0)).fillna(0.0)
    ann = math.sqrt(BPY)
    h = len(pnl) // 2
    s_full = pnl.mean() / pnl.std() * ann if pnl.std() > 0 else 0
    s_a = pnl.iloc[:h].mean() / pnl.iloc[:h].std() * ann if pnl.iloc[:h].std() > 0 else 0
    s_b = pnl.iloc[h:].mean() / pnl.iloc[h:].std() * ann if pnl.iloc[h:].std() > 0 else 0
    return s_full, s_a, s_b

print(f"{'sizing':<10}{'gate':<6}{'zd':>4}{'full':>7}{'H1':>7}{'H2':>7}")
for sizing in ["constant", "pertrade"]:
    for volgate in [False, True]:
        worst_sum = 0
        for zd in [9, 12, 18, 24]:
            f, a, b = run(zd, sizing, volgate)
            worst_sum += min(a, b)
            print(f"{sizing:<10}{str(volgate):<6}{zd:>4}{f:>7.2f}{a:>7.2f}{b:>7.2f}")
        print(f"{sizing:<10}{str(volgate):<6} sum of worst-half Sharpes: {worst_sum:+.2f}\n")
