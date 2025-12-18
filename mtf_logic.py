from indicators import TechnicalIndicators

class MTFLogic:
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def get_trend_direction(self, candles, fast=9, slow=21):
        """Returns 'bullish', 'bearish', or 'neutral'"""
        if not candles or len(candles) < slow:
            return 'neutral'
        
        closes = [c['close'] for c in candles]
        ema_fast = self.indicators.calculate_ema(closes, fast)
        ema_slow = self.indicators.calculate_ema(closes, slow)
        
        if not ema_fast or not ema_slow:
            return 'neutral'
        
        if ema_fast > ema_slow * 1.002:
            return 'bullish'
        elif ema_fast < ema_slow * 0.998:
            return 'bearish'
        else:
            return 'neutral'
    
    def check_mtf_alignment(self, trend_15m, bias_1h, signal_direction):
        """
        Check if 5m signal aligns with higher timeframes
        Returns: (aligned: bool, reason: str)
        """
        if signal_direction == 'LONG':
            # LONG allowed if 15m bullish and 1h bullish/neutral
            if trend_15m == 'bullish' and bias_1h in ['bullish', 'neutral']:
                return True, 'MTF aligned'
            else:
                return False, f'MTF mismatch: 15m={trend_15m}, 1h={bias_1h}'
        
        elif signal_direction == 'SHORT':
            # SHORT allowed if 15m bearish and 1h bearish/neutral
            if trend_15m == 'bearish' and bias_1h in ['bearish', 'neutral']:
                return True, 'MTF aligned'
            else:
                return False, f'MTF mismatch: 15m={trend_15m}, 1h={bias_1h}'
        
        return False, 'Unknown direction'
    
    def get_mtf_score(self, trend_15m, bias_1h, signal_direction):
        """Returns MTF alignment score (0-10)"""
        aligned, _ = self.check_mtf_alignment(trend_15m, bias_1h, signal_direction)
        
        if not aligned:
            return 0
        
        # Full alignment = 10 points
        if signal_direction == 'LONG':
            if trend_15m == 'bullish' and bias_1h == 'bullish':
                return 10
            elif trend_15m == 'bullish' and bias_1h == 'neutral':
                return 8
        
        elif signal_direction == 'SHORT':
            if trend_15m == 'bearish' and bias_1h == 'bearish':
                return 10
            elif trend_15m == 'bearish' and bias_1h == 'neutral':
                return 8
        
        return 5