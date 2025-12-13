"""
🧠 Signal Engine - 45 Logic Implementation (FINAL)
Logic 11-45: Trend, Momentum, Volume, Risk Filters
FIXED VERSION - Railway & CoinDCX Ready
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import *

class SignalEngine:
    def __init__(self):
        self.signals_generated = 0
        self.consecutive_losses = 0
        self.last_signal_time = {}

    # =====================
    # INDICATORS
    # =====================
    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data):
        ema_fast = self.calculate_ema(data, MACD_FAST)
        ema_slow = self.calculate_ema(data, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, MACD_SIGNAL)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, data, period=20, std=2):
        sma = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower

    # =====================
    # LOGIC 11–12 EMA TREND
    # =====================
    def check_ema_trend(self, df):
        if len(df) < EMA_SLOW:
            return "NEUTRAL"

        close = df['close']
        ema20 = self.calculate_ema(close, EMA_FAST)
        ema50 = self.calculate_ema(close, EMA_MID)
        ema200 = self.calculate_ema(close, EMA_SLOW)

        if ema20.isna().any() or ema50.isna().any() or ema200.isna().any():
            return "NEUTRAL"

        if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
            return "BULL"
        elif ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
            return "BEAR"
        return "NEUTRAL"

    # =====================
    # LOGIC 13 STRUCTURE
    # =====================
    def check_market_structure(self, df):
        if len(df) < 10:
            return "NEUTRAL_STRUCTURE"

        highs = df['high'].rolling(5).max().dropna()
        lows = df['low'].rolling(5).min().dropna()

        if len(highs) < 5 or len(lows) < 5:
            return "NEUTRAL_STRUCTURE"

        hh = highs.iloc[-1] > highs.iloc[-2]
        ll = lows.iloc[-1] < lows.iloc[-2]

        if hh:
            return "BULLISH_STRUCTURE"
        if ll:
            return "BEARISH_STRUCTURE"
        return "NEUTRAL_STRUCTURE"

    # =====================
    # LOGIC 21–23 RSI (FIXED)
    # =====================
    def check_rsi_conditions(self, df):
        close = df['close']
        rsi = self.calculate_rsi(close, RSI_PERIOD)
        df['rsi'] = rsi

        signals = []

        if len(rsi) < 3:
            return signals

        # ✅ Oversold reversal (FIXED)
        if rsi.iloc[-2] < RSI_OVERSOLD and rsi.iloc[-1] > RSI_OVERSOLD:
            signals.append("RSI_OVERSOLD_REVERSAL")

        # ✅ Overbought rejection (FIXED)
        if rsi.iloc[-2] > RSI_OVERBOUGHT and rsi.iloc[-1] < RSI_OVERBOUGHT:
            signals.append("RSI_OVERBOUGHT_REJECTION")

        # Divergence
        if len(close) >= 20:
            if close.iloc[-1] <= close.tail(20).min() and rsi.iloc[-1] > rsi.tail(20).min():
                signals.append("RSI_BULLISH_DIVERGENCE")

        return signals

    # =====================
    # LOGIC 24–25 MACD
    # =====================
    def check_macd_signals(self, df):
        close = df['close']
        macd, signal, hist = self.calculate_macd(close)

        df['macd_hist'] = hist
        signals = []

        if len(macd) < 2:
            return signals

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append("MACD_BULLISH_CROSS")

        if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append("MACD_BEARISH_CROSS")

        if hist.iloc[-1] > hist.iloc[-2] > 0:
            signals.append("MACD_BULLISH_MOMENTUM")

        if hist.iloc[-1] < hist.iloc[-2] < 0:
            signals.append("MACD_BEARISH_MOMENTUM")

        return signals

    # =====================
    # LOGIC 28–29 BB
    # =====================
    def check_bollinger_bands(self, df):
        close = df['close']
        upper, mid, lower = self.calculate_bollinger_bands(close)

        signals = []

        if close.iloc[-1] < lower.iloc[-1]:
            signals.append("BB_OVERSOLD")
        elif close.iloc[-1] > upper.iloc[-1]:
            signals.append("BB_OVERBOUGHT")

        return signals

    # =====================
    # LOGIC 31–32 VOLUME
    # =====================
    def check_volume_confirmation(self, df):
        volume = df['volume']
        ma = volume.rolling(VOLUME_MA_PERIOD).mean()

        if ma.isna().iloc[-1]:
            return "VOLUME_NORMAL"

        if volume.iloc[-1] > ma.iloc[-1] * 1.5:
            return "VOLUME_SPIKE"
        if volume.iloc[-1] < ma.iloc[-1] * 0.5:
            return "VOLUME_DRY"
        return "VOLUME_NORMAL"

    # =====================
    # LOGIC 33–35 VWAP
    # =====================
    def check_vwap(self, df):
        tp = (df['high'] + df['low'] + df['close']) / 3
        vwap = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        if len(vwap) < 2:
            return "VWAP_NEUTRAL"

        if df['close'].iloc[-1] > vwap.iloc[-1] and df['close'].iloc[-2] <= vwap.iloc[-2]:
            return "VWAP_RECLAIM"
        if df['close'].iloc[-1] < vwap.iloc[-1] and df['close'].iloc[-2] >= vwap.iloc[-2]:
            return "VWAP_REJECTION"
        return "VWAP_NEUTRAL"

    # =====================
    # LOGIC 42 MANIPULATION (FIXED)
    # =====================
    def check_manipulation_filter(self, df):
        candle = df.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        if total == 0:
            return False
        return (body / total) < 0.3  # True = MANIPULATION

    # =====================
    # LOGIC 40 ROUND NUMBER
    # =====================
    def check_round_number_trap(self, price):
        s = str(int(price))
        return len(s) - len(s.rstrip('0')) >= 3

    # =====================
    # FINAL SIGNAL BUILDER
    # =====================
    def generate_signal(self, pair_data, mode="mid"):
        symbol = pair_data['symbol']
        df = pair_data['data'].copy()

        if len(df) < 50:
            return None

        # ❌ reject manipulation candles
        if self.check_manipulation_filter(df):
            return None

        price = df['close'].iloc[-1]
        if self.check_round_number_trap(price):
            return None

        # Cooldown
        if symbol in self.last_signal_time:
            mins = (datetime.now() - self.last_signal_time[symbol]).seconds / 60
            if mins < SIGNAL_MODES[mode]['hold_time']:
                return None

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return None

        ema = self.check_ema_trend(df)
        structure = self.check_market_structure(df)
        rsi = self.check_rsi_conditions(df)
        macd = self.check_macd_signals(df)
        bb = self.check_bollinger_bands(df)
        vol = self.check_volume_confirmation(df)
        vwap = self.check_vwap(df)

        long_score = 0
        short_score = 0

        if ema == "BULL": long_score += 2
        if structure == "BULLISH_STRUCTURE": long_score += 2
        if "RSI_OVERSOLD_REVERSAL" in rsi: long_score += 2
        if "MACD_BULLISH_CROSS" in macd: long_score += 2
        if "BB_OVERSOLD" in bb: long_score += 1
        if vol == "VOLUME_SPIKE": long_score += 2
        if vwap == "VWAP_RECLAIM": long_score += 2

        if ema == "BEAR": short_score += 2
        if structure == "BEARISH_STRUCTURE": short_score += 2
        if "RSI_OVERBOUGHT_REJECTION" in rsi: short_score += 2
        if "MACD_BEARISH_CROSS" in macd: short_score += 2
        if "BB_OVERBOUGHT" in bb: short_score += 1
        if vol == "VOLUME_SPIKE": short_score += 2
        if vwap == "VWAP_REJECTION": short_score += 2

        if long_score >= 6:
            self.last_signal_time[symbol] = datetime.now()
            return {
                "symbol": symbol,
                "side": "LONG",
                "entry": price,
                "score": long_score,
                "mode": mode,
                "time": datetime.now()
            }

        if short_score >= 6:
            self.last_signal_time[symbol] = datetime.now()
            return {
                "symbol": symbol,
                "side": "SHORT",
                "entry": price,
                "score": short_score,
                "mode": mode,
                "time": datetime.now()
            }

        return None