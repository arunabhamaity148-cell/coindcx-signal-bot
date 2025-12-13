"""
📊 Market Scanner - FINAL FIXED VERSION
- Railway safe
- WATCHLIST auto-fallback
- Soft volume filter (signals আসবে)
- No missing config crash
"""

import requests
import pandas as pd
from datetime import datetime, time
import traceback

# =========================
# SAFE CONFIG IMPORT
# =========================
try:
    from config import (
        WATCHLIST,
        BTC_VOLATILITY_THRESHOLD,
        FEAR_GREED_EXTREME,
        MIN_VOLUME_24H,
        MAX_SPREAD_PERCENT,
        VOLUME_MA_PERIOD,
        AVOID_NEWS_HOURS
    )
except Exception:
    # 🔥 Railway safe defaults
    WATCHLIST = []
    BTC_VOLATILITY_THRESHOLD = 2.5
    FEAR_GREED_EXTREME = (10, 90)
    MIN_VOLUME_24H = 1_000_000
    MAX_SPREAD_PERCENT = 0.15
    VOLUME_MA_PERIOD = 20
    AVOID_NEWS_HOURS = [
        (time(17, 30), time(18, 30)),
        (time(13, 0), time(13, 30)),
    ]

# =========================
# FORCE WATCHLIST (IMPORTANT)
# =========================
if not WATCHLIST:
    WATCHLIST = [
        "BTC", "ETH", "BNB", "SOL", "XRP",
        "ADA", "AVAX", "DOGE", "MATIC", "DOT",
        "LINK", "LTC", "UNI", "ATOM", "FIL",
        "NEAR", "APT", "ARB", "OP", "INJ"
    ]


class MarketScanner:
    def __init__(self):
        self.base_url = "https://public.coindcx.com"
        self.market_health_score = 0
        self.btc_volatility = 0.0
        self.market_regime = "UNKNOWN"

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
                print(f"⚠️ API Error {symbol}: {r.status_code}")
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
            return df if len(df) >= 20 else None

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
        print(f"🔹 BTC Volatility: {self.btc_volatility:.2f}%")
        return self.btc_volatility < BTC_VOLATILITY_THRESHOLD

    def detect_market_regime(self):
        df = self.get_market_data("BTC", "15m", 100)
        if df is None:
            self.market_regime = "RANGING"
            return self.market_regime

        returns = df["close"].pct_change().dropna()
        vol = returns.rolling(20).std()

        if vol.iloc[-1] > vol.mean() * 1.5:
            self.market_regime = "VOLATILE"
        elif abs(returns.rolling(20).mean().iloc[-1]) > 0.002:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"

        print(f"🔹 Market Regime: {self.market_regime}")
        return self.market_regime

    def check_fear_greed(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=5)
            fgi = int(r.json()["data"][0]["value"])
            print(f"🔹 Fear & Greed: {fgi}")
            return FEAR_GREED_EXTREME[0] < fgi < FEAR_GREED_EXTREME[1]
        except Exception:
            return True

    def calculate_market_health(self):
        score = 0

        if self.check_btc_calm():
            score += 2

        if self.detect_market_regime() in ["TRENDING", "RANGING"]:
            score += 2

        if self.check_fear_greed():
            score += 2

        now = datetime.now().time()
        avoid = any(start <= now <= end for start, end in AVOID_NEWS_HOURS)
        if not avoid:
            score += 2

        if self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2

        self.market_health_score = score
        print(f"🏥 MARKET HEALTH SCORE: {score}/10")
        return score

    # =========================
    # SCAN PAIRS
    # =========================
    def scan_all_pairs(self):
        tradeable = []

        print(f"🔍 Scanning {len(WATCHLIST)} pairs...")

        # 🔥 Dynamic looseness
        volume_threshold = 0.9 if self.market_health_score >= 8 else 1.1

        for symbol in WATCHLIST:
            try:
                df = self.get_market_data(symbol, "5m", 50)
                if df is None:
                    continue

                df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
                if df["vol_ma"].isna().all():
                    continue

                ratio = df["volume"].iloc[-1] / df["vol_ma"].iloc[-1]
                print(f"  🔎 {symbol} | Volume ratio: {ratio:.2f}")

                if ratio >= volume_threshold:
                    tradeable.append({
                        "symbol": symbol,
                        "price": df["close"].iloc[-1],
                        "volume_ratio": ratio
                    })
                    print(f"  ✅ {symbol} SELECTED")

            except Exception:
                print(f"⚠️ Scan error {symbol}")
                traceback.print_exc()

        print(f"\n✅ Found {len(tradeable)} tradeable pairs")
        return tradeable