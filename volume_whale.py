import numpy as np

class VolumeWhaleDetector:
    """
    Advanced volume and whale activity detection
    """
    
    @staticmethod
    def detect_volume_surge(candles, lookback=20):
        """
        Detect institutional volume surge
        Returns: (is_surge: bool, surge_ratio: float, direction: str)
        """
        if len(candles) < lookback + 1:
            return False, 0, 'none'
        
        recent = candles[-lookback:]
        current = candles[-1]
        
        # Calculate average volume
        avg_volume = np.mean([c['volume'] for c in recent[:-1]])
        current_volume = current['volume']
        
        if avg_volume == 0:
            return False, 0, 'none'
        
        surge_ratio = current_volume / avg_volume
        
        # 3x volume = institutional interest
        if surge_ratio >= 3.0:
            # Determine direction
            body = current['close'] - current['open']
            direction = 'bullish' if body > 0 else 'bearish'
            return True, surge_ratio, direction
        
        return False, surge_ratio, 'none'
    
    @staticmethod
    def detect_whale_candle(candle):
        """
        Detect single large move (whale activity)
        Returns: (is_whale: bool, move_pct: float, direction: str)
        """
        if candle['open'] == 0:
            return False, 0, 'none'
        
        body = candle['close'] - candle['open']
        body_pct = abs(body) / candle['open'] * 100
        
        # 2%+ single candle move = whale
        if body_pct >= 2.0:
            direction = 'bullish' if body > 0 else 'bearish'
            return True, body_pct, direction
        
        return False, body_pct, 'none'
    
    @staticmethod
    def calculate_order_flow_strength(candles, period=5):
        """
        Advanced order flow calculation
        Returns: float between -1 (strong sell) to +1 (strong buy)
        """
        if len(candles) < period:
            return 0
        
        recent = candles[-period:]
        
        buy_pressure = 0
        sell_pressure = 0
        
        for candle in recent:
            body = candle['close'] - candle['open']
            wick_high = candle['high'] - max(candle['open'], candle['close'])
            wick_low = min(candle['open'], candle['close']) - candle['low']
            
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                continue
            
            # Weighted by candle range and body
            weight = abs(body) / total_range if total_range > 0 else 0
            
            if body > 0:
                # Bullish candle
                buy_pressure += (abs(body) + wick_low * 0.5) * weight
            else:
                # Bearish candle
                sell_pressure += (abs(body) + wick_high * 0.5) * weight
        
        total = buy_pressure + sell_pressure
        if total == 0:
            return 0
        
        return (buy_pressure - sell_pressure) / total
    
    @staticmethod
    def detect_liquidity_sweep_advanced(candles, lookback=30):
        """
        Advanced liquidity sweep detection
        Returns: (is_sweep: bool, sweep_type: str, strength: int)
        """
        if len(candles) < lookback:
            return False, 'none', 0
        
        recent = candles[-lookback:]
        current = candles[-1]
        
        # Find recent highs/lows (excluding current)
        highs = [c['high'] for c in recent[:-1]]
        lows = [c['low'] for c in recent[:-1]]
        
        swing_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        swing_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
        
        # Check for bullish sweep (wick below low, close above)
        if current['low'] < swing_low:
            if current['close'] > current['open']:
                # Strong reversal
                wick = min(current['open'], current['close']) - current['low']
                body = current['close'] - current['open']
                
                if wick > body * 0.5:
                    strength = 3 if wick > body else 2
                    return True, 'bullish_sweep', strength
        
        # Check for bearish sweep (wick above high, close below)
        if current['high'] > swing_high:
            if current['close'] < current['open']:
                # Strong reversal
                wick = current['high'] - max(current['open'], current['close'])
                body = current['open'] - current['close']
                
                if wick > body * 0.5:
                    strength = 3 if wick > body else 2
                    return True, 'bearish_sweep', strength
        
        return False, 'none', 0