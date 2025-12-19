import pandas as pd
import numpy as np

class Indicators:
    """Technical Indicators for Trading Analysis"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def macd(data: pd.Series, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = Indicators.ema(data, fast)
        ema_slow = Indicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Average Directional Index - Trend Strength"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Average True Range - Volatility"""
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        sma = Indicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        
        return k, d
    
    @staticmethod
    def volume_surge(volume: pd.Series, period: int = 20) -> pd.Series:
        """Detect volume surge (above average)"""
        avg_volume = volume.rolling(window=period).mean()
        surge_ratio = volume / avg_volume
        return surge_ratio
    
    @staticmethod
    def trend_regime(close: pd.Series, ema_fast: int = 50, ema_slow: int = 200) -> str:
        """Determine market regime: TRENDING or RANGING"""
        ema_f = Indicators.ema(close, ema_fast)
        ema_s = Indicators.ema(close, ema_slow)
        
        separation = abs(ema_f.iloc[-1] - ema_s.iloc[-1]) / ema_s.iloc[-1]
        
        if separation > 0.02:  # 2% separation = trending
            return "TRENDING"
        else:
            return "RANGING"
    
    @staticmethod
    def mtf_trend(close_5m: pd.Series, close_15m: pd.Series, close_1h: pd.Series) -> str:
        """Multi-Timeframe Trend Analysis"""
        
        # Check each timeframe
        trend_5m = "UP" if close_5m.iloc[-1] > close_5m.iloc[-10] else "DOWN"
        trend_15m = "UP" if close_15m.iloc[-1] > close_15m.iloc[-10] else "DOWN"
        trend_1h = "UP" if close_1h.iloc[-1] > close_1h.iloc[-10] else "DOWN"
        
        # All aligned = strong trend
        if trend_5m == trend_15m == trend_1h == "UP":
            return "STRONG_UP"
        elif trend_5m == trend_15m == trend_1h == "DOWN":
            return "STRONG_DOWN"
        elif trend_15m == trend_1h:
            return f"MODERATE_{trend_1h}"
        else:
            return "MIXED"