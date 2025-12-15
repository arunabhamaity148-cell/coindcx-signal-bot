"""
🧠 SIMPLE + SMART SIGNAL ENGINE
Manual Trading Friendly
Goal: 15–25 clean signals/day
"""

from datetime import datetime
import pandas as pd
from config import *

class SignalEngine:
    def __init__(self):
        self.last_signal_time = {}

    # =====================
    # INDICATORS
    # =====================
    def ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # =====================
    # CORE SIGNAL LOGIC
    # =====================
    def generate_signal(self, pair_data, mode="quick"):
        symbol = pair_data["symbol"]
        df = pair_data["data"].copy()

        if len(df) < 50:
            print(f"❌ {symbol} blocked: low candles")
            return None

        close = df["close"]
        volume = df["volume"]

        # EMA Trend
        ema20 = self.ema(close, 20)
        ema50 = self.ema(close, 50)

        trend = None
        if ema20.iloc[-1] > ema50.iloc[-1]:
            trend = "BUY"
        elif ema20.iloc[-1] < ema50.iloc[-1]:
            trend = "SELL"
        else:
            print(f"❌ {symbol} blocked: EMA flat")
            return None

        # RSI Filter (relaxed but smart)
        rsi_val = self.rsi(close).iloc[-1]
        if trend == "BUY" and not (40 <= rsi_val <= 65):
            print(f"❌ {symbol} BUY blocked: RSI {rsi_val:.1f}")
            return None
        if trend == "SELL" and not (35 <= rsi_val <= 60):
            print(f"❌ {symbol} SELL blocked: RSI {rsi_val:.1f}")
            return None

        # Volume confirmation
        vol_ma = volume.rolling(VOLUME_MA_PERIOD).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma if vol_ma else 0

        if vol_ratio < 1.2:
            print(f"❌ {symbol} blocked: Volume weak ({vol_ratio:.2f}x)")
            return None

        # Smart candle confirmation
        last = df.iloc[-1]
        if trend == "BUY" and last["close"] <= last["open"]:
            print(f"❌ {symbol} BUY blocked: red candle")
            return None
        if trend == "SELL" and last["close"] >= last["open"]:
            print(f"❌ {symbol} SELL blocked: green candle")
            return None

        # Soft cooldown (only same coin, same mode)
        key = f"{symbol}_{mode}"
        if key in self.last_signal_time:
            mins = (datetime.now() - self.last_signal_time[key]).seconds / 60
            if mins < 5:
                print(f"⏸ {symbol} cooldown {mins:.1f} min")
                return None

        # SCORE (transparent & simple)
        score = 0
        score += 2  # EMA trend
        score += 2  # RSI zone
        score += 2  # Volume
        score += 1  # Candle direction

        if score < 5:
            print(f"❌ {symbol} blocked: low score {score}")
            return None

        # Save cooldown
        self.last_signal_time[key] = datetime.now()

        print(f"✅ SIGNAL {symbol} {trend} | RSI {rsi_val:.1f} | Vol {vol_ratio:.2f}x")

        return {
            "symbol": symbol,
            "direction": "LONG" if trend == "BUY" else "SHORT",
            "entry_price": close.iloc[-1],
            "mode": mode,
            "score": score,
            "timestamp": datetime.now(),
            "indicators": {
                "ema20": ema20.iloc[-1],
                "ema50": ema50.iloc[-1],
                "rsi": rsi_val,
                "volume_ratio": round(vol_ratio, 2)
            }
        }