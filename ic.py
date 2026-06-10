#!/usr/bin/env python3
"""Information-content map: does z_t predict forward spread change dS over the next k days,
with execution delay 1 (so forward window starts at t+2)?  IC = corr(-z_t, fwd dS) and t-stat.
Also conditional IC for |z|>1.5 (tail reversion) and per-decade stability for the winners."""
import math
import numpy as np, pandas as pd
import mrlab

START = "1990-01-01"
DELAY = 1   # signal close t -> execute close t+1 -> earn from t+2

def fwd_ret(dS, k):
    """f_t = sum of dS over t+1+DELAY .. t+k+DELAY (what a delay-executed position earns over k days)."""
    return dS.rolling(k).sum().shift(-(k + DELAY))

rows = []
for nm, fn in mrlab.UNIVERSE.items():
    dS = fn().dropna()
    dS = dS[dS.index >= START]
    if len(dS) < 2000:
        continue
    S = dS.cumsum()
    best = None
    for z_n in [10, 20, 40, 60, 120, 250]:
        z = mrlab.zscore(S, z_n)
        sig = dS.ewm(span=63, min_periods=20).std().shift(1)
        for k in [5, 10, 21, 42]:
            f = fwd_ret(dS, k) / (sig * math.sqrt(k))
            df = pd.concat([z, f], axis=1, keys=["z", "f"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(df) < 500:
                continue
            df["f"] = df.f.clip(-10, 10)
            ic = np.corrcoef(-df.z, df.f)[0, 1]
            t = ic * math.sqrt(len(df) / max(k, 1))      # crude overlap-corrected t
            tail = df[np.abs(df.z) > 1.5]
            ict = np.corrcoef(-tail.z, tail.f)[0, 1] if len(tail) > 200 else np.nan
            if best is None or t > best[4]:
                best = (nm, z_n, k, ic, t, ict, len(df))
    rows.append(best)

rows.sort(key=lambda r: -r[4])
print(f"delay={DELAY}.  IC = corr(-z, vol-normalized fwd dS).  t = overlap-adjusted t-stat.")
print(f"{'spread':<12}{'z_n':>5}{'k':>4}{'IC':>8}{'t':>7}{'tailIC':>8}{'n':>7}")
for nm, z_n, k, ic, t, ict, n in rows:
    flag = " ***" if t > 3 else ("  *" if t > 2 else "")
    print(f"{nm:<12}{z_n:>5}{k:>4}{ic:>8.3f}{t:>7.2f}{(ict if ict==ict else 0):>8.3f}{n:>7}{flag}")
