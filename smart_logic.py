import numpy as np

class SmartMoneyLogic:
    
    @staticmethod
    def detect_market_regime(prices, atr, adx):
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
    def detect_liquidity_grab(candles, lookback=20):
        if len(candles) < lookback:
            return None
        
        recent_candles = candles[-lookback:]
        highs = [c['high'] for c in recent_candles]
        lows = [c['low'] for c in recent_candles]
        
        swing_high = max(highs[:-1])
        swing_low = min(lows[:-1])
        
        current = candles[-1]
        
        if current['low'] < swing_low and current['close'] > current['open']:
            return 'bullish_sweep'
        
        if current['high'] > swing_high and current['close'] < current['open']:
            return 'bearish_sweep'
        
        return None
    
    @staticmethod
    def calculate_order_flow(candles):
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
        
        total = buying_pressure + selling_pressure
        if total == 0:
            return 0
        
        return (buying_pressure - selling_pressure) / total