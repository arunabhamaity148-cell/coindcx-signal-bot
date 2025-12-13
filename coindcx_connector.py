import requests
import pandas as pd
from datetime import datetime

class CoinDCXConnector:
    def __init__(self):
        self.base = "https://public.coindcx.com"
        self.symbols = [
            "B-BTC_USDT","B-ETH_USDT","B-SOL_USDT","B-BNB_USDT","B-XRP_USDT",
            "B-DOGE_USDT","B-ADA_USDT","B-AVAX_USDT","B-MATIC_USDT","B-DOT_USDT",
            "B-LINK_USDT","B-UNI_USDT","B-ATOM_USDT","B-LTC_USDT","B-TRX_USDT"
        ]

    def get_ohlcv(self, symbol, interval="15m", limit=200):
        url = f"{self.base}/market_data/candles"
        r = requests.get(url, params={
            "pair": symbol,
            "interval": interval,
            "limit": limit
        }).json()
        if not r:
            return None
        df = pd.DataFrame(r, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df

    def get_orderbook(self, symbol):
        r = requests.get(
            f"{self.base}/market_data/orderbook",
            params={"pair": symbol}
        ).json()
        if not r:
            return None
        bids = [[float(x["price"]), float(x["quantity"])] for x in r["bids"]]
        asks = [[float(x["price"]), float(x["quantity"])] for x in r["asks"]]
        return {
            "bids": bids,
            "asks": asks,
            "imbalance": sum(b[1] for b in bids) / max(1, sum(a[1] for a in asks)),
            "spread": asks[0][0] - bids[0][0]
        }