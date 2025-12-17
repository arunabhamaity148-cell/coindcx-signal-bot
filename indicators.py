"""
Technical indicators for trading analysis
"""
import pandas as pd
import numpy as np

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain = pd.Series(gain).rolling(window=period).mean()
    loss = pd.Series(loss).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD Indicator"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    high = df["high"]
    low = df["low"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr_atr = atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx_val = dx.rolling(window=period).mean()
    
    return adx_val

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def stochastic(df: pd.DataFrame, period: int = 14) -> tuple:
    """Stochastic Oscillator"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(window=3).mean()
    
    return k, d

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 10) -> str:
    """Detect bullish/bearish divergence"""
    if len(price) < lookback + 2:
        return None
    
    price_trend = price.iloc[-1] - price.iloc[-lookback]
    ind_trend = indicator.iloc[-1] - indicator.iloc[-lookback]
    
    if price_trend < -0.01 and ind_trend > 0.01:
        return "bullish_divergence"
    
    if price_trend > 0.01 and ind_trend < -0.01:
        return "bearish_divergence"
    
    return None