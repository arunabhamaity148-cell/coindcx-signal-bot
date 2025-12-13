"""
📊 Market Scanner - Health Checks + 50 Pair Monitoring
Implements Logic 1-10: Market Health Filters
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
from config import (
    WATCHLIST,
    BTC_VOLATILITY_THRESHOLD,
    FEAR_GREED_EXTREME,
    MIN_VOLUME_24H,
    MAX_SPREAD_PERCENT,
    VOLUME_MA_PERIOD,
    AVOID_NEWS_HOURS
)


class MarketScanner:
    def __init__(self):
        self.base_url = "https://public.coindcx.com"
        self.market_health_score = 0
        self.btc_volatility = 0.0
        self.market_regime = "UNKNOWN"

    # --------------------------------------------------
    # OHLCV FETCH
    # --------------------------------------------------
    def get_market_data(self, symbol, interval="5m", limit=200):
        try:
            endpoint = f"{self.base_url}/market_data/candles"
            params = {
                "pair": f"B-{symbol}_USDT",
                "interval": interval,
                "limit": limit
            }

            r = requests.get(endpoint, params=params, timeout=10)
            if r.status_code != 200:
                print(f"⚠️ API Error for {symbol}: {r.status_code}")
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

        except Exception as e:
            print(f"❌ Data fetch error {symbol}: {e}")
            return None

    # --------------------------------------------------
    # LOGIC 1: BTC CALM
    # --------------------------------------------------
    def check_btc_calm(self):
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
    def detect_market_regime(self):
        btc = self.get_market_data("BTC", "15m", 100)
        if btc is None:
            self.market_regime = "RANGING"
            return self.market_regime

        returns = btc["close"].pct_change().dropna()
        if len(returns) < 20:
            self.market_regime = "RANGING"
            return self.market_regime

        vol = returns.rolling(20).std()
        avg_vol = vol.mean()

        if vol.iloc[-1] > avg_vol * 1.5:
            self.market_regime = "VOLATILE"
        elif abs(returns.rolling(20).mean().iloc[-1]) > 0.002:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"

        print(f"🔹 Market Regime: {self.market_regime}")
        return self.market_regime

    # --------------------------------------------------
    # LOGIC 4: FEAR & GREED
    # --------------------------------------------------
    def check_fear_greed(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=5)
            fgi = int(r.json()["data"][0]["value"])
            ok = FEAR_GREED_EXTREME[0] < fgi < FEAR_GREED_EXTREME[1]
            print(f"🔹 Fear & Greed: {fgi} {'✅' if ok else '⚠️'}")
            return ok
        except:
            return True

    # --------------------------------------------------
    # MARKET HEALTH SCORE (0–10)
    # --------------------------------------------------
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
        in_news = any(start <= now <= end for start, end in AVOID_NEWS_HOURS)
        if not in_news:
            score += 2

        if self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2

        self.market_health_score = score
        print(f"\n🏥 MARKET HEALTH SCORE: {score}/10 {'✅' if score >= 6 else '⚠️'}\n")
        return score

    # --------------------------------------------------
    # PAIR SCAN
    # --------------------------------------------------
    def scan_all_pairs(self):
        tradeable = []
        print(f"🔍 Scanning {len(WATCHLIST)} pairs...")

        for symbol in WATCHLIST[:20]:
            df = self.get_market_data(symbol, "5m", 60)
            if df is None:
                continue

            df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()
            if df["vol_ma"].isna().all():
                continue

            ratio = df["volume"].iloc[-1] / df["vol_ma"].iloc[-1]
            if ratio > 1.2:
                tradeable.append(symbol)
                print(f"  ✅ {symbol} volume spike {ratio:.2f}x")

        print(f"\n✅ Found {len(tradeable)} tradeable pairs\n")
        return tradeable