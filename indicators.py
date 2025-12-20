import pandas as pd
import numpy as np

class Indicators:
    """Technical Indicators for Trading Analysis"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        return data.rolling(window=period).mean()
    
    @staticmethod
    def macd(data: pd.Series, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        if len(data) < slow:
            empty = pd.Series([np.nan] * len(data), index=data.index)
            return empty, empty, empty
            
        ema_fast = Indicators.ema(data, fast)
        ema_slow = Indicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index - FIXED VERSION
        Handles edge cases and prevents NaN propagation
        """
        if len(data) < period + 1:
            return pd.Series([np.nan] * len(data), index=data.index)
        
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate average gain and loss using rolling mean
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # ✅ CRITICAL FIX: Handle division by zero
        # Replace zero losses with small number to avoid inf/nan
        avg_loss = avg_loss.replace(0, 1e-10)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # ✅ Clean up any remaining NaN or inf values
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        # ✅ Fill initial NaN values with 50 (neutral RSI)
        # This is better than leaving NaN for the first 'period' values
        # rsi = rsi.fillna(50)
        
        return rsi
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """
        Average Directional Index - FIXED VERSION
        Trend Strength indicator
        """
        if len(high) < period + 1:
            empty = pd.Series([np.nan] * len(high), index=high.index)
            return empty, empty, empty
        
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Set negative values to zero
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ✅ Smooth TR and DM using rolling mean
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        # ✅ Handle division by zero
        atr_safe = atr.replace(0, 1e-10)
        
        plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm_smooth / atr_safe)
        minus_di = 100 * (minus_dm_smooth / atr_safe)
        
        # Calculate DX
        di_sum = plus_di + minus_di
        di_sum_safe = di_sum.replace(0, 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / di_sum_safe
        
        # Calculate ADX (smoothed DX)
        adx = dx.rolling(window=period, min_periods=period).mean()
        
        # ✅ Clean up inf/nan
        adx = adx.replace([np.inf, -np.inf], np.nan)
        plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
        minus_di = minus_di.replace([np.inf, -np.inf], np.nan)
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """
        Average True Range - FIXED VERSION
        Volatility indicator
        """
        if len(high) < period + 1:
            return pd.Series([np.nan] * len(high), index=high.index)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # True Range is the maximum
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        # ✅ Clean up
        atr = atr.replace([np.inf, -np.inf], np.nan)
        
        return atr
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        if len(data) < period:
            empty = pd.Series([np.nan] * len(data), index=data.index)
            return empty, empty, empty
            
        sma = Indicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Stochastic Oscillator"""
        if len(high) < period:
            empty = pd.Series([np.nan] * len(high), index=high.index)
            return empty, empty
            
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        # ✅ Handle division by zero
        range_hl = highest_high - lowest_low
        range_hl_safe = range_hl.replace(0, 1e-10)
        
        k = 100 * ((close - lowest_low) / range_hl_safe)
        d = k.rolling(window=3).mean()
        
        # ✅ Clean up
        k = k.replace([np.inf, -np.inf], np.nan)
        d = d.replace([np.inf, -np.inf], np.nan)
        
        return k, d
    
    @staticmethod
    def volume_surge(volume: pd.Series, period: int = 20) -> pd.Series:
        """Detect volume surge (above average)"""
        if len(volume) < period:
            return pd.Series([1.0] * len(volume), index=volume.index)
            
        avg_volume = volume.rolling(window=period).mean()
        
        # ✅ Handle division by zero
        avg_volume_safe = avg_volume.replace(0, 1e-10)
        
        surge_ratio = volume / avg_volume_safe
        
        # ✅ Clean up
        surge_ratio = surge_ratio.replace([np.inf, -np.inf], 1.0)
        surge_ratio = surge_ratio.fillna(1.0)
        
        return surge_ratio
    
    @staticmethod
    def trend_regime(close: pd.Series, ema_fast: int = 50, ema_slow: int = 200) -> str:
        """Determine market regime: TRENDING or RANGING"""
        if len(close) < ema_slow:
            return "UNKNOWN"
            
        ema_f = Indicators.ema(close, ema_fast)
        ema_s = Indicators.ema(close, ema_slow)
        
        # Get last valid values
        ema_f_val = ema_f.dropna().iloc[-1] if len(ema_f.dropna()) > 0 else 0
        ema_s_val = ema_s.dropna().iloc[-1] if len(ema_s.dropna()) > 0 else 0
        
        if ema_s_val == 0:
            return "UNKNOWN"
        
        separation = abs(ema_f_val - ema_s_val) / ema_s_val
        
        if separation > 0.02:  # 2% separation = trending
            return "TRENDING"
        else:
            return "RANGING"
    
    @staticmethod
    def mtf_trend(close_5m: pd.Series, close_15m: pd.Series, close_1h: pd.Series) -> str:
        """Multi-Timeframe Trend Analysis"""
        
        if len(close_5m) < 10 or len(close_15m) < 10 or len(close_1h) < 10:
            return "UNKNOWN"
        
        # Check each timeframe
        try:
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
        except:
            return "UNKNOWN"