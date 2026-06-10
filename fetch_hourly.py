#!/usr/bin/env python3
"""Fetch 730d of hourly bars for futures legs from Yahoo chart API -> data_hourly/*.csv"""
import json, os, time, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_hourly")
os.makedirs(OUT, exist_ok=True)
TICKERS = ["CL=F", "BZ=F", "RB=F", "HO=F", "NG=F",
           "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
           "ZC=F", "ZS=F", "ZW=F", "KE=F", "ZL=F", "ZM=F", "ZO=F",
           "LE=F", "HE=F", "GF=F",
           "KC=F", "CC=F", "CT=F", "SB=F", "OJ=F"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

for tk in TICKERS:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{tk}"
           f"?interval=60m&range=730d&includePrePost=false")
    try:
        req = urllib.request.Request(url, headers=UA)
        data = json.load(urllib.request.urlopen(req, timeout=30))
        res = data["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        rows = []
        for i, t in enumerate(ts):
            o, c = q["open"][i], q["close"][i]
            h, l, v = q["high"][i], q["low"][i], q["volume"][i]
            if o is None or c is None:
                continue
            rows.append(f"{t},{o},{h},{l},{c},{v if v is not None else 0}")
        path = os.path.join(OUT, tk.replace("=F", "") + ".csv")
        with open(path, "w") as f:
            f.write("ts,open,high,low,close,volume\n" + "\n".join(rows) + "\n")
        print(f"{tk:<6} {len(rows):6d} bars  -> {path}")
    except Exception as e:
        print(f"{tk:<6} FAILED: {e}")
    time.sleep(1.0)
