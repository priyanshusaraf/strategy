#!/usr/bin/env python3
"""Download 12y of hourly FX (Dukascopy) -> data_fx/<PAIR>.csv. JPY pairs scale 1e3, else 1e5."""
import lzma, struct, os, time, urllib.request, datetime, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fx")
os.makedirs(OUT, exist_ok=True)
PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY","USDNOK","USDSEK",
         "EURGBP","EURCHF","EURJPY","EURNOK","EURSEK","AUDNZD","AUDCAD","GBPJPY","EURAUD",
         "CADJPY","CHFJPY","GBPCHF","NZDCAD"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Referer": "https://www.dukascopy.com/"}
START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2014
END = datetime.date(2026, 6, 1)

def fetch_month(pair, y, m, scale):
    url = f"https://datafeed.dukascopy.com/datafeed/{pair}/{y}/{m-1:02d}/BID_candles_hour_1.bi5"
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            if not raw: return []
            data = lzma.decompress(raw); rows = []; base = datetime.datetime(y, m, 1)
            for i in range(0, len(data)-23, 24):
                off, o, c, lo, hi, vol = struct.unpack(">iiiiif", data[i:i+24])
                ts = base + datetime.timedelta(seconds=off)
                rows.append((int(ts.timestamp()), o/scale, hi/scale, lo/scale, c/scale, vol))
            return rows
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return None

for pair in PAIRS:
    scale = 1000.0 if pair.endswith("JPY") else 100000.0
    allrows = []; fails = 0; y, m = START_YEAR, 1
    while datetime.date(y, m, 1) <= END:
        r = fetch_month(pair, y, m, scale)
        if r is None: fails += 1
        elif r: allrows.extend(r)
        m += 1
        if m > 12: m = 1; y += 1
    allrows.sort(); seen = set(); dd = []
    for row in allrows:
        if row[0] not in seen: seen.add(row[0]); dd.append(row)
    with open(os.path.join(OUT, f"{pair}.csv"), "w") as f:
        f.write("ts,open,high,low,close,volume\n")
        for ts,o,h,l,c,v in dd: f.write(f"{ts},{o},{h},{l},{c},{v}\n")
    span = f"{datetime.date.fromtimestamp(dd[0][0])} -> {datetime.date.fromtimestamp(dd[-1][0])}" if dd else ""
    print(f"{pair:<8} {len(dd):6d} bars fails={fails} {span}", flush=True)
