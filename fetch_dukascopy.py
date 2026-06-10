#!/usr/bin/env python3
"""Download multi-year HOURLY candles from Dukascopy -> data_duka/<INST>.csv
Monthly files: /datafeed/<INST>/<YYYY>/<MM-1>/BID_candles_hour_1.bi5  (LZMA).
Record = >iiiiif : (sec offset in month, open, close, low, high, volume); prices x1000.
"""
import lzma, struct, os, time, urllib.request, datetime, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_duka")
os.makedirs(OUT, exist_ok=True)
INSTR = {  # friendly -> Dukascopy id
    "WTI": "LIGHTCMDUSD", "BRENT": "BRENTCMDUSD", "GAS": "GASCMDUSD", "DIESEL": "DIESELCMDUSD",
    "XAU": "XAUUSD", "XAG": "XAGUSD", "COPPER": "COPPERCMDUSD",
    "COCOA": "COCOACMDUSD", "SUGAR": "SUGARCMDUSD",
}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Referer": "https://www.dukascopy.com/"}
START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2014
END = datetime.date(2026, 6, 1)

def fetch_month(inst_id, y, m):
    url = f"https://datafeed.dukascopy.com/datafeed/{inst_id}/{y}/{m-1:02d}/BID_candles_hour_1.bi5"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=30).read()
            if not raw:
                return []
            data = lzma.decompress(raw)
            rows = []
            base = datetime.datetime(y, m, 1)
            for i in range(0, len(data) - 23, 24):
                off, o, c, lo, hi, vol = struct.unpack(">iiiiif", data[i:i+24])
                ts = base + datetime.timedelta(seconds=off)
                rows.append((int(ts.timestamp()), o/1000, hi/1000, lo/1000, c/1000, vol))
            return rows
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None  # failed

for friendly, inst_id in INSTR.items():
    path = os.path.join(OUT, f"{friendly}.csv")
    allrows = []; fails = 0
    y, m = START_YEAR, 1
    while datetime.date(y, m, 1) <= END:
        r = fetch_month(inst_id, y, m)
        if r is None:
            fails += 1
        elif r:
            allrows.extend(r)
        m += 1
        if m > 12: m = 1; y += 1
    allrows.sort()
    seen = set(); dedup = []
    for row in allrows:
        if row[0] not in seen:
            seen.add(row[0]); dedup.append(row)
    with open(path, "w") as f:
        f.write("ts,open,high,low,close,volume\n")
        for ts, o, h, l, c, v in dedup:
            f.write(f"{ts},{o},{h},{l},{c},{v}\n")
    span = ""
    if dedup:
        span = f"{datetime.date.fromtimestamp(dedup[0][0])} -> {datetime.date.fromtimestamp(dedup[-1][0])}"
    print(f"{friendly:<7} {len(dedup):6d} bars  fails={fails}  {span}", flush=True)
