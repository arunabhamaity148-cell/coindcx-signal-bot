# fetch_5coins_30d.py
import os, requests, pandas as pd, time

os.makedirs("data", exist_ok=True)
coins = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT".split(",")

for c in coins:
    url = f"https://api.binance.com/api/v3/klines?symbol={c}&interval=15m&limit=2000"
    data = requests.get(url, timeout=20).json()
    if not data:
        continue
    df = pd.DataFrame(
        data,
        columns="ot o h l c v ct qv trades tb tq i".split()
    )[["ot", "o", "h", "l", "c"]]
    df["ot"] = pd.to_datetime(df["ot"], unit="ms")
    df.to_csv(f"data/{c.lower()}_15m_30d.csv", index=False)
    print("✅", c)

print("🎉 5 coins 30d CSV ready!")
