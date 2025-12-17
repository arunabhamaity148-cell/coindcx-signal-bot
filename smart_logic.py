import numpy as np

class SmartMoneyLogic:
    
    @staticmethod
    def detect_market_regime(prices, atr, adx):
        """Detect market regime: Trending, Ranging, or Volatile"""
        if adx is None or atr is None:
            return 'unknown'
        
        price_range = max(prices[-20:]) - min(prices[-20:])
        avg_price = np.mean(prices[-20:])
        
        volatility_ratio = (atr / avg_price) * 100
        
        if adx > 25 and volatility_ratio < 3:
            return 'trending'
        elif adx < 20 and volatility_ratio < 2:
            return 'ranging'
        elif volatility_ratio > 3:
            return 'volatile'
        else:
            return 'mixed'
    
    @staticmethod
    def detect_fair_value_gap(candles, min_gap_size=0.002):
        """Detect Fair Value Gaps (FVG)"""
        if len(candles) < 3:
            return None
        
        first = candles[-3]
        second = candles[-2]
        third = candles[-1]
        
        # Bullish FVG: gap between first high and third low
        bullish_gap = third['low'] - first['high']
        if bullish_gap > 0 and bullish_gap / first['close'] > min_gap_size:
            return {'type': 'bullish', 'size': bullish_gap, 'level': (first['high'] + third['low']) / 2}
        
        # Bearish FVG: gap between first low and third high
        bearish_gap = first['low'] - third['high']
        if bearish_gap > 0 and bearish_gap / first['close'] > min_gap_size:
            return {'type': 'bearish', 'size': bearish_gap, 'level': (first['low'] + third['high']) / 2}
        
        return None
    
    @staticmethod
    def detect_liquidity_grab(candles, lookback=20):
        """Detect liquidity grab patterns"""
        if len(candles) < lookback:
            return None
        
        recent_candles = candles[-lookback:]
        highs = [c['high'] for c in recent_candles]
        lows = [c['low'] for c in recent_candles]
        
        swing_high = max(highs[:-1])
        swing_low = min(lows[:-1])
        
        current = candles[-1]
        
        # Bullish liquidity grab: swept below swing low then reversed up
        if current['low'] < swing_low and current['close'] > current['open']:
            return 'bullish_sweep'
        
        # Bearish liquidity grab: swept above swing high then reversed down
        if current['high'] > swing_high and current['close'] < current['open']:
            return 'bearish_sweep'
        
        return None
    
    @staticmethod
    def calculate_order_flow_imbalance(candles):
        """Simulate order flow imbalance"""
        if len(candles) < 5:
            return 0
        
        buying_pressure = 0
        selling_pressure = 0
        
        for candle in candles[-5:]:
            body = candle['close'] - candle['open']
            volume_estimate = (candle['high'] - candle['low']) * 1000
            
            if body > 0:
                buying_pressure += abs(body) * volume_estimate
            else:
                selling_pressure += abs(body) * volume_estimate
        
        total_pressure = buying_pressure + selling_pressure
        if total_pressure == 0:
            return 0
        
        imbalance = (buying_pressure - selling_pressure) / total_pressure
        return imbalance
    
    @staticmethod
    def calculate_volume_profile_poc(candles, lookback=50):
        """Calculate Point of Control from volume profile"""
        if len(candles) < lookback:
            return None
        
        recent_candles = candles[-lookback:]
        
        # Create price bins
        all_prices = []
        for c in recent_candles:
            all_prices.extend([c['high'], c['low'], c['close']])
        
        min_price = min(all_prices)
        max_price = max(all_prices)
        
        if max_price == min_price:
            return None
        
        bins = 20
        bin_size = (max_price - min_price) / bins
        
        volume_at_price = [0] * bins
        
        for candle in recent_candles:
            avg_price = (candle['high'] + candle['low'] + candle['close']) / 3
            bin_index = int((avg_price - min_price) / bin_size)
            if 0 <= bin_index < bins:
                volume_estimate = candle['high'] - candle['low']
                volume_at_price[bin_index] += volume_estimate
        
        # Find POC (highest volume bin)
        poc_bin = volume_at_price.index(max(volume_at_price))
        poc_price = min_price + (poc_bin * bin_size) + (bin_size / 2)
        
        return poc_price
    
    @staticmethod
    def calculate_fibonacci_levels(candles, lookback=50):
        """Calculate Fibonacci retracement levels"""
        if len(candles) < lookback:
            return None
        
        recent_candles = candles[-lookback:]
        highs = [c['high'] for c in recent_candles]
        lows = [c['low'] for c in recent_candles]
        
        swing_high = max(highs)
        swing_low = min(lows)
        
        diff = swing_high - swing_low
        
        levels = {
            '0.236': swing_high - (diff * 0.236),
            '0.382': swing_high - (diff * 0.382),
            '0.500': swing_high - (diff * 0.500),
            '0.618': swing_high - (diff * 0.618),
            '0.786': swing_high - (diff * 0.786)
        }
        
        return levels
