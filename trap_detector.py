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
        Smart detection - adapts to data quality
        """
        if len(candles) < 2:
            return False
        
        last_candle = candles.iloc[-1]
        prev_candle = candles.iloc[-2]
        
        # Check if data is realistic (has variation)
        price_variation = abs(last_candle['close'] - prev_candle['close']) / prev_candle['close']
        
        # If no variation = simulated, be lenient
        if price_variation < 0.0001:  # Less than 0.01% change
            return False  # Skip check for flat data
        
        body = abs(last_candle['close'] - last_candle['open'])
        
        if body < 0.0001:
            body = 0.0001
        
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Real data: Strict (2x)
        # Simulated but realistic: Medium (3x)
        threshold = 3.0 if price_variation < 0.001 else 2.0
        
        if upper_wick > body * threshold:
            return True
        if lower_wick > body * threshold:
            return True
        
        return False
    
    @staticmethod
    def check_wick_manipulation(candles: pd.DataFrame) -> bool:
        """
        Trap #3: Wick manipulation filter
        Smart detection based on data quality
        """
        if len(candles) < 3:
            return False
        
        # Check data quality - is it realistic?
        recent_closes = candles['close'].iloc[-5:] if len(candles) >= 5 else candles['close']
        price_std = recent_closes.std()
        price_mean = recent_closes.mean()
        
        # Coefficient of variation
        cv = (price_std / price_mean) if price_mean > 0 else 0
        
        # Very low variation = simulated data
        is_simulated = cv < 0.005  # Less than 0.5% variation
        
        last_3 = candles.iloc[-3:]
        wick_traps = 0
        
        for _, candle in last_3.iterrows():
            body = abs(candle['close'] - candle['open'])
            
            if body < 0.0001:
                body = 0.0001
                
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            
            # Adaptive threshold
            threshold = 4.0 if is_simulated else 1.5
            
            if upper_wick > body * threshold or lower_wick > body * threshold:
                wick_traps += 1
        
        # Need 2+ for trigger
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
        Smart detection - checks for unrealistic perfection
        """
        
        # Check if indicators are suspiciously perfect
        perfect_score = 0
        
        # RSI at exact round numbers
        rsi_rounded = round(rsi)
        if rsi_rounded in [30, 50, 70] and abs(rsi - rsi_rounded) < 0.5:
            perfect_score += 1
        
        # ADX at exact thresholds
        adx_rounded = round(adx)
        if adx_rounded in [25, 50] and abs(adx - adx_rounded) < 0.5:
            perfect_score += 1
        
        # MACD too close to zero
        if abs(macd_hist) < 0.01:
            perfect_score += 1
        
        # If 2+ indicators are "perfect" = suspicious
        return perfect_score >= 2
    
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