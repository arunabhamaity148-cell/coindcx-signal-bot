"""
📊 Market Scanner - Health Checks + 50 Pair Monitoring
Implements Logic 1–10: Market Health Filters
Railway-safe | CoinDCX compatible | Defensive config handling
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict

# ===============================
# SAFE CONFIG IMPORT (NO CRASH)
# ===============================
try:
    from config import (
        WATCHLIST,
        BTC_VOLATILITY_THRESHOLD,
        FEAR_GREED_EXTREME,
        MIN_VOLUME_24H,
        MAX_SPREAD_PERCENT,
        VOLUME_MA_PERIOD,
        AVOID_NEWS_HOURS,
    )
except ImportError:
    # ---- Fallback defaults (Railway crash protection) ----
    WATCHLIST = []
    BTC_VOLATILITY_THRESHOLD = 2.5
    FEAR_GREED_EXTREME = [10, 90]
    MIN_VOLUME_24H = 1_000_000
    MAX_SPREAD_PERCENT = 0.15
    VOLUME_MA_PERIOD = 20
    AVOID_NEWS_HOURS = []

# ===============================
# MARKET SCANNER
# ===============================
class MarketScanner:
    def __init__(self):
        self.base_url = "https://public.coindcx.com"
        self.market_health_score = 0
        self.btc_volatility = 0.0
        self.market_regime = "UNKNOWN"

    # --------------------------------------------------
    # DATA FETCH
    # --------------------------------------------------
    def get_market_data(self, symbol: str, interval: str = "5m", limit: int = 200):
        """Fetch OHLCV data from CoinDCX"""
        try:
            endpoint = f"{self.base_url}/market_data/candles"
            params = {
                "pair": f"B-{symbol}_USDT",
                "interval": interval,
                "limit": limit,
            }

            r = requests.get(endpoint, params=params, timeout=10)
            if r.status_code != 200:
                print(f"⚠️ API Error for {symbol}: {r.status_code}")
                return None

            data = r.json()
            if not isinstance(data, list) or len(data) < 20:
                print(f"⚠️ No/insufficient data for {symbol}")
                return None

            df = pd.DataFrame(
                data, columns=["time", "open", "high", "low", "close", "volume"]
            )

            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df.dropna(inplace=True)
            if len(df) < 20:
                return None

            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None

    # --------------------------------------------------
    # LOGIC 1: BTC CALM CHECK
    # --------------------------------------------------
    def check_btc_calm(self) -> bool:
        btc = self.get_market_data("BTC", "1h", 24)
        if btc is None:
            print("⚠️ BTC data unavailable, assuming calm")
            self.btc_volatility = 1.0
            return True

        returns = btc["close"].pct_change().abs()
        self.btc_volatility = returns.mean() * 100

        is_calm = self.btc_volatility < BTC_VOLATILITY_THRESHOLD
        print(f"🔹 BTC Volatility: {self.btc_volatility:.2f}% {'✅' if is_calm else '⚠️'}")
        return is_calm

    # --------------------------------------------------
    # LOGIC 2: MARKET REGIME
    # --------------------------------------------------
    def detect_market_regime(self) -> str:
        btc = self.get_market_data("BTC", "15m", 120)
        if btc is None:
            self.market_regime = "RANGING"
            return self.market_regime

        returns = btc["close"].pct_change().dropna()
        if len(returns) < 20:
            self.market_regime = "RANGING"
            return self.market_regime

        vol_now = returns.rolling(20).std().iloc[-1]
        vol_avg = returns.rolling(20).std().mean()
        trend = returns.rolling(20).mean().iloc[-1]

        if vol_now > vol_avg * 1.5:
            self.market_regime = "VOLATILE"
        elif abs(trend) > 0.002:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"

        print(f"🔹 Market Regime: {self.market_regime}")
        return self.market_regime

    # --------------------------------------------------
    # LOGIC 4: FEAR & GREED
    # --------------------------------------------------
    def check_fear_greed(self) -> bool:
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=5)
            val = int(r.json()["data"][0]["value"])
            ok = FEAR_GREED_EXTREME[0] < val < FEAR_GREED_EXTREME[1]
            print(f"🔹 Fear & Greed: {val} {'✅' if ok else '⚠️'}")
            return ok
        except:
            return True

    # --------------------------------------------------
    # LOGIC 6: NEWS TIME FILTER
    # --------------------------------------------------
    def is_news_time(self) -> bool:
        if not AVOID_NEWS_HOURS:
            return False

        now = datetime.now().time()
        return any(start <= now <= end for start, end in AVOID_NEWS_HOURS)

    # --------------------------------------------------
    # LOGIC 7: SPREAD CHECK
    # --------------------------------------------------
    def check_spread(self, symbol: str) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/market_data/orderbook",
                params={"pair": f"B-{symbol}_USDT"},
                timeout=5,
            )
            data = r.json()
            if not data.get("bids") or not data.get("asks"):
                return False

            bid = float(data["bids"][0]["price"])
            ask = float(data["asks"][0]["price"])
            spread_pct = ((ask - bid) / bid) * 100
            return spread_pct < MAX_SPREAD_PERCENT
        except:
            return False

    # --------------------------------------------------
    # LOGIC 8: LIQUIDITY CHECK
    # --------------------------------------------------
    def check_liquidity(self, symbol: str) -> bool:
        try:
            r = requests.get(f"{self.base_url}/market_data/ticker", timeout=5)
            for t in r.json():
                if t.get("market") == f"B-{symbol}_USDT":
                    return float(t.get("volume", 0)) > MIN_VOLUME_24H
            return False
        except:
            return False

    # --------------------------------------------------
    # MARKET HEALTH SCORE (1–10)
    # --------------------------------------------------
    def calculate_market_health(self) -> int:
        score = 0

        if self.check_btc_calm():
            score += 2

        regime = self.detect_market_regime()
        if regime in ["TRENDING", "RANGING"]:
            score += 2

        if self.check_fear_greed():
            score += 2

        if not self.is_news_time():
            score += 2

        if self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2

        self.market_health_score = score
        print(f"\n🏥 MARKET HEALTH SCORE: {score}/10 {'✅' if score >= 6 else '⚠️'}\n")
        return score

    # --------------------------------------------------
    # PAIR SCAN (VOLUME SPIKE)
    # --------------------------------------------------
    def scan_all_pairs(self) -> List[Dict]:
        print(f"🔍 Scanning {len(WATCHLIST)} pairs...")
        tradeable = []

        for symbol in WATCHLIST:
            df = self.get_market_data(symbol, "5m", 60)
            if df is None:
                continue

            df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
            if df["vol_ma"].isna().iloc[-1]:
                continue

            ratio = df["volume"].iloc[-1] / df["vol_ma"].iloc[-1]
            if ratio >= 1.2:
                tradeable.append(
                    {
                        "symbol": symbol,
                        "price": df["close"].iloc[-1],
                        "volume_ratio": round(ratio, 2),
                    }
                )
                print(f"  ✅ {symbol} | Vol x{ratio:.2f}")

        print(f"\n✅ Found {len(tradeable)} tradeable pairs\n")
        return tradeable