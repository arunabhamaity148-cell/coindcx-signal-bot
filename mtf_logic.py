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
        
        # RELAXED threshold - 0.5% difference
        if ema_fast > ema_slow * 1.005:
            return 'bullish'
        elif ema_fast < ema_slow * 0.995:
            return 'bearish'
        else:
            return 'neutral'
    
    def check_mtf_alignment(self, trend_15m, bias_1h, signal_direction, strict_mode=False):
        """
        BALANCED MTF: Allow neutral states
        """
        if not strict_mode:
            # RELAXED: Both neutral = OK, opposing = block
            if signal_direction == 'LONG':
                # Block only if 15m clearly bearish
                if trend_15m == 'bearish':
                    return False, '15m bearish blocks LONG'
                # Neutral is fine
                return True, 'MTF OK'
            
            elif signal_direction == 'SHORT':
                # Block only if 15m clearly bullish
                if trend_15m == 'bullish':
                    return False, '15m bullish blocks SHORT'
                # Neutral is fine
                return True, 'MTF OK'
        
        else:
            # STRICT: Require alignment
            if signal_direction == 'LONG':
                if trend_15m == 'bullish' and bias_1h in ['bullish', 'neutral']:
                    return True, 'MTF aligned'
                return False, f'MTF mismatch: 15m={trend_15m}, 1h={bias_1h}'
            
            elif signal_direction == 'SHORT':
                if trend_15m == 'bearish' and bias_1h in ['bearish', 'neutral']:
                    return True, 'MTF aligned'
                return False, f'MTF mismatch: 15m={trend_15m}, 1h={bias_1h}'
        
        return False, 'Unknown direction'
    
    def get_mtf_score(self, trend_15m, bias_1h, signal_direction):
        """
        Returns MTF alignment score (0-10)
        More generous scoring for INR futures
        """
        if signal_direction == 'LONG':
            if trend_15m == 'bullish' and bias_1h == 'bullish':
                return 10
            elif trend_15m == 'bullish' and bias_1h == 'neutral':
                return 8
            elif trend_15m == 'neutral' and bias_1h == 'bullish':
                return 7
            elif trend_15m == 'neutral' and bias_1h == 'neutral':
                return 5
            elif trend_15m == 'bearish':
                return 0
            else:
                return 3
        
        elif signal_direction == 'SHORT':
            if trend_15m == 'bearish' and bias_1h == 'bearish':
                return 10
            elif trend_15m == 'bearish' and bias_1h == 'neutral':
                return 8
            elif trend_15m == 'neutral' and bias_1h == 'bearish':
                return 7
            elif trend_15m == 'neutral' and bias_1h == 'neutral':
                return 5
            elif trend_15m == 'bullish':
                return 0
            else:
                return 3
        
        return 0