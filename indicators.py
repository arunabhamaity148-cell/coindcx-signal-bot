import numpy as np
import pandas as pd

class TechnicalIndicators:

    @staticmethod
    def calculate_ema(data, period):
        if len(data) < period:
            return None
        return pd.Series(data).ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return None, None, None

        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    @staticmethod
    def calculate_atr(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return None

        tr_list = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)

        atr = np.mean(tr_list[-period:])
        return atr

    @staticmethod
    def calculate_adx(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return None

        plus_dm = []
        minus_dm = []

        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)

            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)

        atr = TechnicalIndicators.calculate_atr(highs, lows, closes, period)
        if not atr or atr == 0:
            return None

        plus_di = (np.mean(plus_dm[-period:]) / atr) * 100
        minus_di = (np.mean(minus_dm[-period:]) / atr) * 100

        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0

        return dx
    
    @staticmethod
    def calculate_volatility(prices, atr):
        if not prices or not atr:
            return None
        avg_price = np.mean(prices[-20:])
        if avg_price == 0:
            return None
        return (atr / avg_price) * 100