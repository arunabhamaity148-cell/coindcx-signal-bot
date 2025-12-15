"""
🧠 Signal Engine - 45 Logic Implementation (FINAL)
FULL DEBUG VERSION – shows WHY signals are blocked
CoinDCX + Railway READY
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import *

class SignalEngine:
    def __init__(self):
        self.last_signal_time = {}
        self.consecutive_losses = 0

    # =====================
    # INDICATORS
    # =====================
    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data):
        ema_fast = self.calculate_ema(data, MACD_FAST)
        ema_slow = self.calculate_ema(data, MACD_SLOW)
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, MACD_SIGNAL)
        hist = macd - signal
        return macd, signal, hist

    # =====================
    # LOG HELPERS
    # =====================
    def log_block(self, symbol, reasons, long_score, short_score):
        print(f"\n🚫 {symbol} BLOCKED")
        for r in reasons:
            print(f" ├─ {r}")
        print(f" └─ Final Score → LONG={long_score} SHORT={short_score} (need ≥6)")

    # =====================
    # LOGICS
    # =====================
    def check_ema_trend(self, df):
        if len(df) < EMA_SLOW:
            return "NEUTRAL"
        ema20 = self.calculate_ema(df['close'], EMA_FAST)
        ema50 = self.calculate_ema(df['close'], EMA_MID)
        ema200 = self.calculate_ema(df['close'], EMA_SLOW)
        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            return "BULL"
        if ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            return "BEAR"
        return "NEUTRAL"

    def check_market_structure(self, df):
        if df['high'].iloc[-1] > df['high'].iloc[-2]:
            return "BULLISH_STRUCTURE"
        if df['low'].iloc[-1] < df['low'].iloc[-2]:
            return "BEARISH_STRUCTURE"
        return "NEUTRAL_STRUCTURE"

    def check_rsi(self, df):
        rsi = self.calculate_rsi(df['close'], RSI_PERIOD)
        df['rsi'] = rsi
        signals = []
        if rsi.iloc[-2] < RSI_OVERSOLD and rsi.iloc[-1] > RSI_OVERSOLD:
            signals.append("RSI_OVERSOLD_REVERSAL")
        if rsi.iloc[-2] > RSI_OVERBOUGHT and rsi.iloc[-1] < RSI_OVERBOUGHT:
            signals.append("RSI_OVERBOUGHT_REJECTION")
        return signals

    def check_macd(self, df):
        macd, signal, hist = self.calculate_macd(df['close'])
        signals = []
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append("MACD_BULLISH_CROSS")
        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append("MACD_BEARISH_CROSS")
        return signals

    def check_volume(self, df):
        ma = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
        if df['volume'].iloc[-1] > ma.iloc[-1] * 1.5:
            return "VOLUME_SPIKE"
        return "VOLUME_NORMAL"

    def check_vwap(self, df):
        tp = (df['high'] + df['low'] + df['close']) / 3
        vwap = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        if df['close'].iloc[-1] > vwap.iloc[-1]:
            return "VWAP_ABOVE"
        return "VWAP_BELOW"

    def manipulation_candle(self, df):
        c = df.iloc[-1]
        body = abs(c['close'] - c['open'])
        total = c['high'] - c['low']
        if total == 0:
            return True
        return (body / total) < 0.3

    # =====================
    # FINAL SIGNAL
    # =====================
    def generate_signal(self, pair_data, mode="mid"):
        symbol = pair_data['symbol']
        df = pair_data['data'].copy()
        reasons = []

        if len(df) < 50:
            reasons.append("Not enough candles (<50)")
            self.log_block(symbol, reasons, 0, 0)
            return None

        if self.manipulation_candle(df):
            reasons.append("Manipulation candle")
            self.log_block(symbol, reasons, 0, 0)
            return None

        if symbol in self.last_signal_time:
            mins = (datetime.now() - self.last_signal_time[symbol]).seconds / 60
            if mins < SIGNAL_MODES[mode]['hold_time']:
                reasons.append(f"Cooldown active ({mins:.1f}m)")
                self.log_block(symbol, reasons, 0, 0)
                return None

        ema = self.check_ema_trend(df)
        structure = self.check_market_structure(df)
        rsi = self.check_rsi(df)
        macd = self.check_macd(df)
        volume = self.check_volume(df)
        vwap = self.check_vwap(df)

        long_score = 0
        short_score = 0

        if ema == "BULL": long_score += 2
        else: reasons.append(f"EMA: {ema}")

        if structure == "BULLISH_STRUCTURE": long_score += 2
        else: reasons.append(f"Structure: {structure}")

        if "RSI_OVERSOLD_REVERSAL" in rsi: long_score += 2
        else: reasons.append("RSI no reversal")

        if "MACD_BULLISH_CROSS" in macd: long_score += 2
        else: reasons.append("MACD no cross")

        if volume == "VOLUME_SPIKE": long_score += 2
        else: reasons.append("Volume normal")

        if vwap == "VWAP_ABOVE": long_score += 1
        else: reasons.append("Below VWAP")

        if long_score >= 6:
            print(f"\n✅ {symbol} LONG SIGNAL | Score={long_score}")
            self.last_signal_time[symbol] = datetime.now()
            return {
                "symbol": symbol,
                "direction": "LONG",
                "entry_price": df['close'].iloc[-1],
                "mode": mode,
                "score": long_score,
                "timestamp": datetime.now(),
                "indicators": {
                    "ema_trend": ema,
                    "rsi": df['rsi'].iloc[-1]
                }
            }

        self.log_block(symbol, reasons, long_score, short_score)
        return None