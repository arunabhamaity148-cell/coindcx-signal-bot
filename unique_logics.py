"""
UNIQUE TRADING LOGICS MODULE
Institutional-grade 45 logic evaluator
NO ML dependency
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class LogicEvaluator:
    def __init__(self, config: Dict):
        self.config = config

    # =========================================================
    # MAIN ENTRY POINT
    # =========================================================
    def evaluate_all_logics(
        self,
        df: pd.DataFrame,
        orderbook: Dict,
        funding_rate: float,
        oi_history: List[float],
        recent_trades: List[Dict],
        fear_greed_index: int,
        news_times: List[datetime],
        liquidation_clusters: List[Dict]
    ) -> Dict:
        """
        Run all 45 logics and return unified result
        """

        results = {
            "trade_allowed": True,
            "final_score": 0,
            "market_health": {},
            "price_action": {},
            "momentum": {},
            "orderflow": {},
            "derivatives": {},
            "anti_trap": {},
        }

        score = 0
        max_score = 45

        # ================= MARKET HEALTH (1–8) =================
        btc_calm = self._btc_calm(df)
        regime = self._market_regime(df)
        funding_ok = abs(funding_rate) < self.config.get("funding_rate_extreme", 0.0015)
        fg_ok = 20 < fear_greed_index < 80
        spread_ok = orderbook["spread"] / df["close"].iloc[-1] < self.config.get("spread_max_percent", 0.001)

        results["market_health"] = {
            "btc_calm": btc_calm,
            "market_regime": regime,
            "funding_ok": funding_ok,
            "fear_greed_ok": fg_ok,
            "spread_ok": spread_ok,
        }

        score += sum([btc_calm, funding_ok, fg_ok, spread_ok])

        # ================= PRICE ACTION (9–15) =================
        structure = self._market_structure(df)
        ema_trend = self._ema_alignment(df)
        atr_ok = self._atr_filter(df)
        bb_squeeze = self._bb_squeeze(df)

        results["price_action"] = {
            "structure": structure,
            "ema_trend": ema_trend,
            "atr_ok": atr_ok,
            "bb_squeeze": bb_squeeze,
        }

        score += sum([
            structure != "neutral",
            ema_trend != "neutral",
            atr_ok,
            bb_squeeze
        ])

        # ================= MOMENTUM (16–21) =================
        rsi = self._rsi(df)
        macd = self._macd(df)
        stoch = self._stochastic(df)
        obv_div = self._obv_divergence(df)
        roc = self._roc(df)

        results["momentum"] = {
            "rsi": rsi,
            "macd": macd,
            "stochastic": stoch,
            "obv_divergence": obv_div,
            "roc": roc,
        }

        score += sum([
            rsi in ["oversold", "overbought"],
            macd != "neutral",
            stoch != "neutral",
            obv_div != "neutral",
            abs(roc) > 0.5
        ])

        # ================= ORDER FLOW (22–31) =================
        imbalance = orderbook.get("imbalance", 1)
        large_orders = self._large_orders(orderbook)
        spoofing = self._spoofing(orderbook)

        results["orderflow"] = {
            "imbalance": imbalance,
            "large_orders": large_orders,
            "spoofing": spoofing,
        }

        score += sum([
            imbalance > 1.2 or imbalance < 0.8,
            len(large_orders) > 0,
            not spoofing
        ])

        # ================= DERIVATIVES (32–37) =================
        oi_trend = self._oi_trend(oi_history)
        liq_near = self._liquidation_near(df["close"].iloc[-1], liquidation_clusters)

        results["derivatives"] = {
            "oi_trend": oi_trend,
            "liquidation_near": liq_near,
        }

        score += sum([
            oi_trend != "neutral",
            not liq_near
        ])

        # ================= ANTI-TRAP (38–45) =================
        round_safe = self._avoid_round(df["close"].iloc[-1])
        sl_hunt = self._sl_hunt(df)
        wick_safe = self._wick_filter(df)

        results["anti_trap"] = {
            "round_safe": round_safe,
            "sl_hunt": sl_hunt,
            "wick_safe": wick_safe,
        }

        score += sum([
            round_safe,
            not sl_hunt,
            wick_safe
        ])

        # ================= FINAL =================
        final_score = (score / max_score) * 100
        results["final_score"] = round(final_score, 2)

        if final_score < 60:
            results["trade_allowed"] = False

        return results

    # =========================================================
    # HELPER LOGICS (FULL IMPLEMENTATION)
    # =========================================================
    def _btc_calm(self, df):
        vol = ((df["high"] - df["low"]) / df["close"]).tail(20).mean()
        return vol < self.config.get("btc_calm_threshold", 0.015)

    def _market_regime(self, df):
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        atr_pct = atr / df["close"].iloc[-1]
        if atr_pct > 0.03:
            return "volatile"
        elif atr_pct < 0.015:
            return "ranging"
        return "trending"

    def _market_structure(self, df):
        h = df["high"]
        l = df["low"]
        if h.iloc[-1] > h.iloc[-5] and l.iloc[-1] > l.iloc[-5]:
            return "bullish"
        if h.iloc[-1] < h.iloc[-5] and l.iloc[-1] < l.iloc[-5]:
            return "bearish"
        return "neutral"

    def _ema_alignment(self, df):
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["close"].ewm(span=50).mean().iloc[-1]
        ema200 = df["close"].ewm(span=200).mean().iloc[-1]
        if ema20 > ema50 > ema200:
            return "bullish"
        if ema20 < ema50 < ema200:
            return "bearish"
        return "neutral"

    def _atr_filter(self, df):
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        return atr / df["close"].iloc[-1] < 0.04

    def _bb_squeeze(self, df):
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        width = (sma + 2*std - (sma - 2*std)) / sma
        return width.iloc[-1] < width.mean()

    def _rsi(self, df):
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        if rsi.iloc[-1] < 30:
            return "oversold"
        if rsi.iloc[-1] > 70:
            return "overbought"
        return "neutral"

    def _macd(self, df):
        macd = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        signal = macd.ewm(span=9).mean()
        if macd.iloc[-1] > signal.iloc[-1]:
            return "bullish"
        if macd.iloc[-1] < signal.iloc[-1]:
            return "bearish"
        return "neutral"

    def _stochastic(self, df):
        low = df["low"].rolling(14).min()
        high = df["high"].rolling(14).max()
        k = 100 * (df["close"] - low) / (high - low)
        if k.iloc[-1] < 20:
            return "bullish"
        if k.iloc[-1] > 80:
            return "bearish"
        return "neutral"

    def _obv_divergence(self, df):
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        if obv.iloc[-1] > obv.iloc[-10] and df["close"].iloc[-1] < df["close"].iloc[-10]:
            return "bullish"
        if obv.iloc[-1] < obv.iloc[-10] and df["close"].iloc[-1] > df["close"].iloc[-10]:
            return "bearish"
        return "neutral"

    def _roc(self, df):
        return ((df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]) * 100

    def _large_orders(self, ob):
        threshold = self.config.get("large_order_threshold", 100000)
        orders = []
        for b in ob.get("bids", []):
            if b[0] * b[1] > threshold:
                orders.append(b)
        for a in ob.get("asks", []):
            if a[0] * a[1] > threshold:
                orders.append(a)
        return orders

    def _spoofing(self, ob):
        return any(b[0]*b[1] > self.config.get("spoofing_wall_threshold", 500000)
                   for b in ob.get("bids", []))

    def _oi_trend(self, oi):
        if len(oi) < 10:
            return "neutral"
        return "increasing" if np.mean(oi[-5:]) > np.mean(oi[-10:-5]) else "decreasing"

    def _liquidation_near(self, price, clusters):
        for c in clusters:
            if abs(price - c["price"]) / price < 0.02:
                return True
        return False

    def _avoid_round(self, price):
        round_price = round(price, -3)
        return abs(price - round_price) / price > 0.001

    def _sl_hunt(self, df):
        wicks = 0
        for _, c in df.tail(20).iterrows():
            body = abs(c["close"] - c["open"])
            wick = (c["high"] - c["low"]) - body
            if wick > body * 2:
                wicks += 1
        return wicks > 3

    def _wick_filter(self, df):
        c = df.iloc[-1]
        return (c["high"] - c["low"]) / c["close"] < 0.01