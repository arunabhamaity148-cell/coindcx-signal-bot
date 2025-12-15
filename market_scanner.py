"""
📊 Market Scanner - CoinDCX (FINAL STABLE)
Logic 1–10: Market Health + Pair Filtering
DEBUG ENABLED
"""

import requests
import pandas as pd
from datetime import datetime
from config import *

class MarketScanner:
    def __init__(self):
        self.base_url = "https://public.coindcx.com"
        self.market_health_score = 0
        self.btc_volatility = None
        self.market_regime = "UNKNOWN"

    # -------------------------------------------------
    # DATA FETCH
    # -------------------------------------------------
    def get_market_data(self, symbol, interval="5m", limit=200):
        try:
            url = f"{self.base_url}/market_data/candles"
            params = {
                "pair": f"B-{symbol}_USDT",
                "interval": interval,
                "limit": limit
            }
            r = requests.get(url, params=params, timeout=10)

            if r.status_code != 200:
                if DEBUG_MODE:
                    print(f"❌ {symbol} candles API {r.status_code}")
                return None

            data = r.json()
            if not data or len(data) < 20:
                if DEBUG_MODE:
                    print(f"❌ {symbol} empty/low candle data")
                return None

            df = pd.DataFrame(
                data,
                columns=["time","open","high","low","close","volume"]
            )

            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df.dropna(inplace=True)

            if len(df) < 20:
                if DEBUG_MODE:
                    print(f"❌ {symbol} insufficient candles after clean")
                return None

            return df

        except Exception as e:
            print(f"❌ Exception fetching {symbol}: {e}")
            return None

    # -------------------------------------------------
    # MARKET HEALTH
    # -------------------------------------------------
    def check_btc_calm(self):
        df = self.get_market_data("BTC", "1h", 24)

        if df is None:
            print("⚠️ BTC candle missing → market health REDUCED")
            self.btc_volatility = None
            return False

        returns = df["close"].pct_change().abs()
        self.btc_volatility = returns.mean() * 100

        calm = self.btc_volatility < BTC_VOLATILITY_THRESHOLD
        print(f"🔹 BTC Volatility: {self.btc_volatility:.2f}% {'✅' if calm else '⚠️'}")
        return calm

    def detect_market_regime(self):
        df = self.get_market_data("BTC", "15m", 100)
        if df is None:
            self.market_regime = "UNKNOWN"
            return self.market_regime

        returns = df["close"].pct_change().dropna()
        if returns.std() > 0.01:
            self.market_regime = "VOLATILE"
        elif abs(returns.mean()) > 0.0015:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"

        print(f"🔹 Market Regime: {self.market_regime}")
        return self.market_regime

    def check_fear_greed(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=5)
            v = int(r.json()["data"][0]["value"])
            ok = FEAR_GREED_EXTREME[0] < v < FEAR_GREED_EXTREME[1]
            print(f"🔹 Fear & Greed: {v} {'✅' if ok else '⚠️'}")
            return ok
        except:
            print("⚠️ Fear & Greed API fail (ignored)")
            return True

    def calculate_market_health(self):
        score = 0

        if self.check_btc_calm():
            score += 2

        regime = self.detect_market_regime()
        if regime in ["TRENDING","RANGING"]:
            score += 2

        if self.check_fear_greed():
            score += 2

        now = datetime.now().time()
        if not any(s <= now <= e for s,e in AVOID_NEWS_HOURS):
            score += 2

        if self.btc_volatility and self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2

        self.market_health_score = score
        print(f"\n🏥 MARKET HEALTH SCORE: {score}/10\n")
        return score

    # -------------------------------------------------
    # PAIR SCAN
    # -------------------------------------------------
    def scan_all_pairs(self):
        tradeable = []

        print(f"🔍 Scanning {TEST_PAIRS_LIMIT} pairs...")

        for symbol in WATCHLIST[:TEST_PAIRS_LIMIT]:
            df = self.get_market_data(symbol, "5m", 120)

            if df is None:
                if DEBUG_MODE:
                    print(f"⛔ {symbol} blocked: no candle data")
                continue

            df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

            if pd.isna(df["vol_ma"].iloc[-1]):
                if DEBUG_MODE:
                    print(f"⛔ {symbol} blocked: volume MA NaN")
                continue

            vol_ratio = df["volume"].iloc[-1] / df["vol_ma"].iloc[-1]

            if vol_ratio >= 1.0:
                tradeable.append({
                    "symbol": symbol,
                    "price": df["close"].iloc[-1],
                    "volume_ratio": round(vol_ratio,2),
                    "data": df
                })
                print(f"✅ {symbol} SELECTED | Vol x{vol_ratio:.2f}")
            else:
                if DEBUG_MODE:
                    print(f"⛔ {symbol} blocked: low volume ({vol_ratio:.2f}x)")

        print(f"\n✅ Found {len(tradeable)} tradeable pairs\n")
        return tradeable