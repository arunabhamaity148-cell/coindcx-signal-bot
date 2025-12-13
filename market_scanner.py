"""
📊 Market Scanner - Health Checks + Pair Selection
Implements Market Health + Volume Logic
FINAL FIXED VERSION
"""

import requests
import pandas as pd
from datetime import datetime
from config import (
    WATCHLIST,
    BTC_VOLATILITY_THRESHOLD,
    FEAR_GREED_EXTREME,
    MIN_VOLUME_24H,
    MAX_SPREAD_PERCENT,
    VOLUME_MA_PERIOD,
    AVOID_NEWS_HOURS,
)

class MarketScanner:
    def __init__(self):
        self.base_url = "https://public.coindcx.com"
        self.btc_volatility = 0.0
        self.market_regime = "RANGING"

    # =========================
    # DATA FETCH
    # =========================
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
                return None

            data = r.json()
            if not data or len(data) < 20:
                return None

            df = pd.DataFrame(
                data,
                columns=["time", "open", "high", "low", "close", "volume"]
            )
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df.dropna(inplace=True)
            if len(df) < 20:
                return None

            return df

        except Exception:
            return None

    # =========================
    # MARKET HEALTH
    # =========================
    def check_btc_calm(self):
        df = self.get_market_data("BTC", "1h", 24)
        if df is None:
            self.btc_volatility = 1.0
            return True

        returns = df["close"].pct_change().abs()
        self.btc_volatility = returns.mean() * 100
        return self.btc_volatility < BTC_VOLATILITY_THRESHOLD

    def detect_market_regime(self):
        df = self.get_market_data("BTC", "15m", 100)
        if df is None:
            self.market_regime = "RANGING"
            return self.market_regime

        returns = df["close"].pct_change().dropna()
        if returns.std() > 0.01:
            self.market_regime = "VOLATILE"
        elif abs(returns.mean()) > 0.0015:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"

        return self.market_regime

    def check_fear_greed(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=5)
            v = int(r.json()["data"][0]["value"])
            return FEAR_GREED_EXTREME[0] < v < FEAR_GREED_EXTREME[1]
        except Exception:
            return True

    def calculate_market_health(self):
        score = 0

        if self.check_btc_calm():
            score += 2

        regime = self.detect_market_regime()
        if regime in ["TRENDING", "RANGING"]:
            score += 2

        if self.check_fear_greed():
            score += 2

        now = datetime.now().time()
        avoid = any(start <= now <= end for start, end in AVOID_NEWS_HOURS)
        if not avoid:
            score += 2

        if self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2

        print(f"🏥 MARKET HEALTH SCORE: {score}/10")
        return score

    # =========================
    # PAIR SCAN
    # =========================
    def scan_all_pairs(self):
        tradeable = []

        print(f"🔍 Scanning {len(WATCHLIST)} pairs...")

        for symbol in WATCHLIST:
            df = self.get_market_data(symbol, "5m", 120)
            if df is None:
                continue

            df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
            if df["vol_ma"].isna().all():
                continue

            current_vol = df["volume"].iloc[-1]
            avg_vol = df["vol_ma"].iloc[-1]
            if avg_vol <= 0:
                continue

            volume_ratio = current_vol / avg_vol

            if volume_ratio >= 1.2:
                tradeable.append({
                    "symbol": symbol,
                    "price": df["close"].iloc[-1],
                    "volume_ratio": round(volume_ratio, 2),
                    "data": df   # 🔥 MOST IMPORTANT FIX
                })
                print(f"  ✅ {symbol} SELECTED | Vol x{volume_ratio:.2f}")

        print(f"✅ Found {len(tradeable)} tradeable pairs")
        return tradeable