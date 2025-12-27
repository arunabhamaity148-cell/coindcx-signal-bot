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
        """Relative Strength Index"""
        if len(data) < period + 1:
            return pd.Series([np.nan] * len(data), index=data.index)

        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        avg_loss = avg_loss.replace(0, 1e-10)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.replace([np.inf, -np.inf], np.nan)

        return rsi

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Average Directional Index"""
        if len(high) < period + 1:
            empty = pd.Series([np.nan] * len(high), index=high.index)
            return empty, empty, empty

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()

        atr_safe = atr.replace(0, 1e-10)

        plus_dm_smooth = plus_dm.rolling(window=period, min_periods=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period, min_periods=period).mean()

        plus_di = 100 * (plus_dm_smooth / atr_safe)
        minus_di = 100 * (minus_dm_smooth / atr_safe)

        di_sum = plus_di + minus_di
        di_sum_safe = di_sum.replace(0, 1e-10)

        dx = 100 * abs(plus_di - minus_di) / di_sum_safe

        adx = dx.rolling(window=period, min_periods=period).mean()

        adx = adx.replace([np.inf, -np.inf], np.nan)
        plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
        minus_di = minus_di.replace([np.inf, -np.inf], np.nan)

        return adx, plus_di, minus_di

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
        """Average True Range"""
        if len(high) < period + 1:
            return pd.Series([np.nan] * len(high), index=high.index)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=period).mean()

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

        range_hl = highest_high - lowest_low
        range_hl_safe = range_hl.replace(0, 1e-10)

        k = 100 * ((close - lowest_low) / range_hl_safe)
        d = k.rolling(window=3).mean()

        k = k.replace([np.inf, -np.inf], np.nan)
        d = d.replace([np.inf, -np.inf], np.nan)

        return k, d

    @staticmethod
    def volume_surge(volume: pd.Series, period: int = 20) -> pd.Series:
        """Detect volume surge"""
        if len(volume) < period:
            return pd.Series([1.0] * len(volume), index=volume.index)

        avg_volume = volume.rolling(window=period).mean()

        avg_volume_safe = avg_volume.replace(0, 1e-10)

        surge_ratio = volume / avg_volume_safe

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

        ema_f_val = ema_f.dropna().iloc[-1] if len(ema_f.dropna()) > 0 else 0
        ema_s_val = ema_s.dropna().iloc[-1] if len(ema_s.dropna()) > 0 else 0

        if ema_s_val == 0:
            return "UNKNOWN"

        separation = abs(ema_f_val - ema_s_val) / ema_s_val

        if separation > 0.02:
            return "TRENDING"
        else:
            return "RANGING"

    @staticmethod
    def mtf_trend(close_5m: pd.Series, close_15m: pd.Series, close_1h: pd.Series) -> str:
        """Multi-Timeframe Trend Analysis"""

        if len(close_5m) < 10 or len(close_15m) < 10 or len(close_1h) < 10:
            return "UNKNOWN"

        try:
            trend_5m = "UP" if close_5m.iloc[-1] > close_5m.iloc[-10] else "DOWN"
            trend_15m = "UP" if close_15m.iloc[-1] > close_15m.iloc[-10] else "DOWN"
            trend_1h = "UP" if close_1h.iloc[-1] > close_1h.iloc[-10] else "DOWN"

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

    @staticmethod
    def check_htf_alignment(direction: str, mode: str, close_1h: pd.Series, close_4h: pd.Series = None) -> tuple[bool, str]:
        """Mode-specific HTF alignment check"""
        try:
            if len(close_1h) < 10:
                return False, "Insufficient 1H data"
            
            trend_1h = "UP" if close_1h.iloc[-1] > close_1h.iloc[-10] else "DOWN"
            
            if mode == 'QUICK':
                if direction == "LONG" and trend_1h == "DOWN":
                    return False, f"HTF_AGAINST: 1H {trend_1h} vs LONG"
                if direction == "SHORT" and trend_1h == "UP":
                    return False, f"HTF_AGAINST: 1H {trend_1h} vs SHORT"
                return True, "HTF_OK: 1H not against"
            
            elif mode == 'MID':
                if direction == "LONG" and trend_1h != "UP":
                    return False, f"HTF_AGAINST: 1H {trend_1h} vs LONG"
                if direction == "SHORT" and trend_1h != "DOWN":
                    return False, f"HTF_AGAINST: 1H {trend_1h} vs SHORT"
                return True, "HTF_OK: 1H aligned"
            
            elif mode == 'TREND':
                if close_4h is None or len(close_4h) < 10:
                    return False, "Insufficient 4H data"
                
                trend_4h = "UP" if close_4h.iloc[-1] > close_4h.iloc[-10] else "DOWN"
                
                if direction == "LONG":
                    if trend_1h != "UP" or trend_4h != "UP":
                        return False, f"HTF_AGAINST: 1H={trend_1h}, 4H={trend_4h} vs LONG"
                else:
                    if trend_1h != "DOWN" or trend_4h != "DOWN":
                        return False, f"HTF_AGAINST: 1H={trend_1h}, 4H={trend_4h} vs SHORT"
                
                return True, "HTF_OK: 1H+4H aligned"
            
            return True, "HTF_OK: Mode unknown"
            
        except Exception as e:
            return False, f"HTF_ERROR: {str(e)[:50]}"

    @staticmethod
    def find_order_blocks(candles_4h: pd.DataFrame, direction: str) -> list:
        """Find institutional order blocks from 4H impulse moves"""
        try:
            if candles_4h is None or len(candles_4h) < 20:
                return []
            
            order_blocks = []
            
            for i in range(10, len(candles_4h)):
                current_close = float(candles_4h.iloc[i]['close'])
                prev_close = float(candles_4h.iloc[i-3]['close'])
                
                move_pct = abs(current_close - prev_close) / prev_close * 100
                
                if move_pct > 5:
                    ob_candle = candles_4h.iloc[i-1]
                    ob_price = float(ob_candle['close'])
                    
                    is_bullish_impulse = current_close > prev_close
                    
                    if direction == "LONG" and is_bullish_impulse:
                        order_blocks.append({
                            'price': ob_price,
                            'type': 'BULLISH_OB',
                            'strength': move_pct
                        })
                    elif direction == "SHORT" and not is_bullish_impulse:
                        order_blocks.append({
                            'price': ob_price,
                            'type': 'BEARISH_OB',
                            'strength': move_pct
                        })
            
            return order_blocks[-5:] if order_blocks else []
            
        except Exception as e:
            return []

    @staticmethod
    def is_near_order_block(current_price: float, order_blocks: list, threshold: float = 1.5) -> tuple[bool, dict]:
        """Check if current price is near an order block"""
        try:
            if not order_blocks:
                return False, {}
            
            for ob in order_blocks:
                distance_pct = abs(current_price - ob['price']) / current_price * 100
                
                if distance_pct < threshold:
                    return True, {
                        'distance': round(distance_pct, 2),
                        'ob_price': ob['price'],
                        'type': ob['type'],
                        'strength': round(ob['strength'], 1)
                    }
            
            return False, {}
            
        except Exception as e:
            return False, {}

    @staticmethod
    def detect_liquidity_sweep(candles: pd.DataFrame, direction: str) -> tuple[bool, dict]:
        """Detect if price recently swept liquidity (stop hunt)"""
        try:
            if candles is None or len(candles) < 30:
                return False, {}
            
            recent_20 = candles.iloc[-30:-10]
            last_10 = candles.iloc[-10:]
            
            if direction == "LONG":
                recent_low = float(recent_20['low'].min())
                
                for idx, candle in last_10.iterrows():
                    candle_low = float(candle['low'])
                    candle_close = float(candle['close'])
                    
                    if candle_low < recent_low and candle_close > recent_low:
                        wick_size = candle_close - candle_low
                        body_size = abs(candle_close - float(candle['open']))
                        
                        if wick_size > body_size * 1.5:
                            return True, {
                                'type': 'BULLISH_SWEEP',
                                'swept_level': recent_low,
                                'rejection_strength': round(wick_size / body_size, 2)
                            }
            
            else:
                recent_high = float(recent_20['high'].max())
                
                for idx, candle in last_10.iterrows():
                    candle_high = float(candle['high'])
                    candle_close = float(candle['close'])
                    
                    if candle_high > recent_high and candle_close < recent_high:
                        wick_size = candle_high - candle_close
                        body_size = abs(candle_close - float(candle['open']))
                        
                        if wick_size > body_size * 1.5:
                            return True, {
                                'type': 'BEARISH_SWEEP',
                                'swept_level': recent_high,
                                'rejection_strength': round(wick_size / body_size, 2)
                            }
            
            return False, {}
            
        except Exception as e:
            return False, {}

    @staticmethod
    def detect_fvg(candles: pd.DataFrame, direction: str) -> tuple[bool, dict]:
        """
        Detect Fair Value Gap (3-candle imbalance)
        Returns: (has_fvg, fvg_info)
        """
        try:
            if candles is None or len(candles) < 10:
                return False, {}
            
            last_10 = candles.tail(10)
            
            for i in range(len(last_10) - 3, max(len(last_10) - 7, 0), -1):
                c1 = last_10.iloc[i]
                c2 = last_10.iloc[i+1]
                c3 = last_10.iloc[i+2]
                
                c1_high = float(c1['high'])
                c1_low = float(c1['low'])
                c3_high = float(c3['high'])
                c3_low = float(c3['low'])
                
                if direction == "LONG":
                    if c1_high < c3_low:
                        gap_size = c3_low - c1_high
                        gap_pct = (gap_size / c1_high) * 100
                        
                        return True, {
                            'type': 'BULLISH_FVG',
                            'gap_start': c1_high,
                            'gap_end': c3_low,
                            'gap_size_pct': round(gap_pct, 2)
                        }
                
                else:
                    if c1_low > c3_high:
                        gap_size = c1_low - c3_high
                        gap_pct = (gap_size / c1_low) * 100
                        
                        return True, {
                            'type': 'BEARISH_FVG',
                            'gap_start': c3_high,
                            'gap_end': c1_low,
                            'gap_size_pct': round(gap_pct, 2)
                        }
            
            return False, {}
            
        except Exception as e:
            return False, {}

    @staticmethod
    def is_price_in_fvg(current_price: float, fvg_info: dict) -> bool:
        """Check if current price is filling FVG"""
        try:
            if not fvg_info:
                return False
            
            gap_start = fvg_info.get('gap_start', 0)
            gap_end = fvg_info.get('gap_end', 0)
            
            if gap_start < gap_end:
                return gap_start <= current_price <= gap_end
            else:
                return gap_end <= current_price <= gap_start
                
        except:
            return False

    @staticmethod
    def get_daily_key_levels(candles_daily: pd.DataFrame) -> dict:
        """
        Get key levels from daily timeframe
        Returns: Previous Day High/Low, Weekly High/Low
        """
        try:
            if candles_daily is None or len(candles_daily) < 10:
                return {}
            
            prev_day = candles_daily.iloc[-2]
            pdh = float(prev_day['high'])
            pdl = float(prev_day['low'])
            
            last_7_days = candles_daily.tail(7)
            weekly_high = float(last_7_days['high'].max())
            weekly_low = float(last_7_days['low'].min())
            
            return {
                'pdh': pdh,
                'pdl': pdl,
                'weekly_high': weekly_high,
                'weekly_low': weekly_low
            }
            
        except Exception as e:
            return {}

    @staticmethod
    def check_key_level_proximity(current_price: float, key_levels: dict, direction: str, threshold: float = 2.0) -> tuple[bool, str]:
        """
        Check if entry is near key support/resistance
        Returns: (is_near_key_level, level_info)
        """
        try:
            if not key_levels:
                return False, ""
            
            pdh = key_levels.get('pdh', 0)
            pdl = key_levels.get('pdl', 0)
            weekly_high = key_levels.get('weekly_high', 0)
            weekly_low = key_levels.get('weekly_low', 0)
            
            if direction == "LONG":
                pdl_distance = abs(current_price - pdl) / current_price * 100
                weekly_low_distance = abs(current_price - weekly_low) / current_price * 100
                
                if pdl_distance < threshold:
                    return True, f"Near PDL (₹{pdl:,.2f}) - {pdl_distance:.1f}% away"
                elif weekly_low_distance < threshold:
                    return True, f"Near Weekly Low (₹{weekly_low:,.2f}) - {weekly_low_distance:.1f}% away"
            
            else:
                pdh_distance = abs(current_price - pdh) / current_price * 100
                weekly_high_distance = abs(current_price - weekly_high) / current_price * 100
                
                if pdh_distance < threshold:
                    return True, f"Near PDH (₹{pdh:,.2f}) - {pdh_distance:.1f}% away"
                elif weekly_high_distance < threshold:
                    return True, f"Near Weekly High (₹{weekly_high:,.2f}) - {weekly_high_distance:.1f}% away"
            
            return False, ""
            
        except Exception as e:
            return False, ""

    @staticmethod
    def detect_market_regime(candles_daily: pd.DataFrame) -> str:
        """
        Detect current market regime: VOLATILE, RANGING, or NORMAL
        """
        try:
            if candles_daily is None or len(candles_daily) < 30:
                return "UNKNOWN"
            
            high = candles_daily['high']
            low = candles_daily['low']
            close = candles_daily['close']
            
            atr_series = Indicators.atr(high, low, close, 20)
            
            if atr_series.empty:
                return "UNKNOWN"
            
            current_atr = float(atr_series.iloc[-1])
            avg_atr = float(atr_series.tail(20).mean())
            
            if avg_atr == 0:
                return "UNKNOWN"
            
            atr_ratio = current_atr / avg_atr
            
            if atr_ratio > 1.5:
                return "VOLATILE"
            elif atr_ratio < 0.7:
                return "RANGING"
            else:
                return "NORMAL"
                
        except Exception as e:
            return "UNKNOWN"