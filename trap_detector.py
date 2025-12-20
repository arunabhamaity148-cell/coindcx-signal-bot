import pandas as pd
import numpy as np
from datetime import datetime, time
from config import config

class TrapDetector:
    """Detects 10 types of market traps to avoid false signals"""
    
    @staticmethod
    def check_ny_pullback_trap(candles: pd.DataFrame) -> bool:
        """
        Trap #1: New York pullback trap
        False breakout during NY session open
        """
        now = datetime.now()
        ny_open_hours = [21, 22, 23]  # 9:30 PM IST onwards (NY open)
        
        if now.hour in ny_open_hours:
            # Check for sudden reversal
            last_3_changes = candles['close'].diff().iloc[-3:]
            if len(set(np.sign(last_3_changes))) >= 2:
                return True
        return False
    
    @staticmethod
    def check_liquidity_grab(candles: pd.DataFrame) -> bool:
        """
        Trap #2: Liquidity grab filter
        
        NOW USES REAL BINANCE DATA - Proper detection!
        """
        if len(candles) < 2:
            return False
        
        last_candle = candles.iloc[-1]
        
        body = abs(last_candle['close'] - last_candle['open'])
        
        if body < 0.0001:
            body = 0.0001
        
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # With REAL Binance data: STRICT threshold
        # Wick > 3x body = liquidity grab
        if upper_wick > body * 3.0:
            return True
        if lower_wick > body * 3.0:
            return True
        
        return False
    
    @staticmethod
    def check_wick_manipulation(candles: pd.DataFrame) -> bool:
        """
        Trap #3: Wick manipulation filter
        
        NOW USES REAL BINANCE DATA - Enable full protection!
        Real historical candles = Proper wick detection
        """
        if len(candles) < 3:
            return False
        
        last_3 = candles.iloc[-3:]
        wick_traps = 0
        
        for _, candle in last_3.iterrows():
            body = abs(candle['close'] - candle['open'])
            
            if body < 0.0001:
                body = 0.0001
                
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            # With REAL data, use STRICT threshold
            # Wick > 2.5x body = manipulation
            if upper_wick > body * 2.5 or lower_wick > body * 2.5:
                wick_traps += 1
        
        # Need 2+ wicks for confirmation
        return wick_traps >= 2
    
    @staticmethod
    def check_news_spike(candles: pd.DataFrame) -> bool:
        """
        Trap #4: News spike avoid
        Sudden price spike (>3%) = news event
        """
        returns = candles['close'].pct_change()
        last_return = abs(returns.iloc[-1])
        
        return last_return > config.NEWS_SPIKE_THRESHOLD
    
    @staticmethod
    def check_market_maker_trap(candles: pd.DataFrame) -> bool:
        """
        Trap #5: Market maker trap
        False breakout with immediate reversal
        """
        if len(candles) < 5:
            return False
        
        # Check if last candle breaks high/low then closes back
        last = candles.iloc[-1]
        prev_high = candles['high'].iloc[-5:-1].max()
        prev_low = candles['low'].iloc[-5:-1].min()
        
        # Breakout up but close below previous high
        if last['high'] > prev_high and last['close'] < prev_high:
            return True
        
        # Breakout down but close above previous low
        if last['low'] < prev_low and last['close'] > prev_low:
            return True
        
        return False
    
    @staticmethod
    def check_orderblock_rejection(candles: pd.DataFrame) -> bool:
        """
        Trap #6: Order block rejection
        Strong rejection from supply/demand zone
        """
        if len(candles) < 10:
            return False
        
        # Find recent high/low zones
        recent_high = candles['high'].iloc[-10:].max()
        recent_low = candles['low'].iloc[-10:].min()
        
        last_close = candles['close'].iloc[-1]
        
        # Rejection = price touches zone but closes far away
        zone_distance_pct = 0.005  # 0.5%
        rejection_distance_pct = 0.01  # 1%
        
        # Near high zone?
        if abs(last_close - recent_high) / recent_high < zone_distance_pct:
            # But closed significantly lower
            if (recent_high - last_close) / recent_high > rejection_distance_pct:
                return True
        
        # Near low zone?
        if abs(last_close - recent_low) / recent_low < zone_distance_pct:
            # But closed significantly higher
            if (last_close - recent_low) / recent_low > rejection_distance_pct:
                return True
        
        return False
    
    @staticmethod
    def check_entry_timing_lag(timestamp: datetime) -> bool:
        """
        Trap #7: Entry timing lag protection
        Avoid stale signals (>5 minutes old)
        """
        now = datetime.now()
        age_seconds = (now - timestamp).total_seconds()
        
        return age_seconds > 300  # 5 minutes
    
    @staticmethod
    def check_spread_expansion(bid: float, ask: float) -> bool:
        """
        Trap #8: Spread expansion filter
        Wide spread = low liquidity trap
        """
        if bid <= 0:
            return True
        
        spread_pct = ((ask - bid) / bid) * 100
        return spread_pct > config.MAX_SPREAD_PERCENT
    
    @staticmethod
    def check_three_candle_reversal(candles: pd.DataFrame) -> bool:
        """
        Trap #9: 3-candle reversal trap
        Alternating red/green candles = indecision
        """
        if len(candles) < 3:
            return False
        
        last_3_changes = candles['close'].diff().iloc[-3:]
        signs = np.sign(last_3_changes)
        
        # All different signs = reversal pattern
        unique_signs = len(set(signs))
        return unique_signs >= 2
    
    @staticmethod
    def check_indicator_overfit(rsi: float, adx: float, macd_hist: float) -> bool:
        """
        Trap #10: Indicator over-optimization trap
        
        NOW USES REAL DATA - Proper detection!
        """
        
        # Check if indicators are unrealistically perfect
        perfect_count = 0
        
        # RSI at exact key levels
        if abs(rsi - 50) < 0.5 or abs(rsi - 30) < 0.5 or abs(rsi - 70) < 0.5:
            perfect_count += 1
        
        # ADX at exact thresholds
        if abs(adx - 25) < 0.5 or abs(adx - 50) < 0.5:
            perfect_count += 1
        
        # MACD too close to zero
        if abs(macd_hist) < 0.005:
            perfect_count += 1
        
        # If 3 indicators are "perfect" = suspicious
        # (Very rare in real data)
        return perfect_count >= 3
    
    @staticmethod
    def check_all_traps(candles: pd.DataFrame, bid: float = 0, ask: float = 0, 
                        rsi: float = 50, adx: float = 25, macd_hist: float = 0) -> dict:
        """
        Run all trap checks and return results
        Returns: {trap_name: is_trapped, ...}
        """
        traps = {
            'ny_pullback': TrapDetector.check_ny_pullback_trap(candles),
            'liquidity_grab': TrapDetector.check_liquidity_grab(candles),
            'wick_manipulation': TrapDetector.check_wick_manipulation(candles),
            'news_spike': TrapDetector.check_news_spike(candles),
            'market_maker': TrapDetector.check_market_maker_trap(candles),
            'orderblock_rejection': TrapDetector.check_orderblock_rejection(candles),
            'entry_timing_lag': TrapDetector.check_entry_timing_lag(datetime.now()),
            'spread_expansion': TrapDetector.check_spread_expansion(bid, ask) if bid and ask else False,
            'three_candle_reversal': TrapDetector.check_three_candle_reversal(candles),
            'indicator_overfit': TrapDetector.check_indicator_overfit(rsi, adx, macd_hist)
        }
        
        return traps
    
    @staticmethod
    def is_trapped(traps: dict) -> bool:
        """Check if any trap is triggered"""
        return any(traps.values())
    
    @staticmethod
    def get_trap_reasons(traps: dict) -> list:
        """Get list of triggered trap names"""
        return [trap_name for trap_name, is_trapped in traps.items() if is_trapped]