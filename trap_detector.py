import pandas as pd
import numpy as np
from datetime import datetime
from config import config

class TrapDetector:
    """
    ADVANCED TRAP DETECTOR (10 TRAPS)
    - All previous bugs fixed
    - Three-candle reversal optimized
    - Entry timing lag uses real timestamps
    - Spread trap improved
    - Liquidity/wick trap lenient but accurate
    """

    @staticmethod
    def check_ny_pullback_trap(candles: pd.DataFrame) -> bool:
        """Trap #1: New York pullback trap"""
        now = datetime.now()
        ny_open_hours = [21, 22, 23]  # 9:30 PM IST → NY open

        if now.hour not in ny_open_hours:
            return False

        last_3 = candles["close"].diff().iloc[-3:]
        signs = np.sign(last_3)

        # zig-zag only → +, -, + or -, +, -
        zigzag = list(signs) in ([1, -1, 1], [-1, 1, -1])
        return zigzag

    @staticmethod
    def check_liquidity_grab(candles: pd.DataFrame) -> bool:
        """Trap #2: Liquidity grab (extreme wick only)"""
        if len(candles) < 2:
            return False

        c = candles.iloc[-1]
        body = abs(c.close - c.open) or 0.0001
        upper = c.high - max(c.open, c.close)
        lower = min(c.open, c.close) - c.low

        return (upper > body * 5) or (lower > body * 5)

    @staticmethod
    def check_wick_manipulation(candles: pd.DataFrame) -> bool:
        """Trap #3: Extreme wick manipulation"""
        if len(candles) < 3:
            return False

        extreme = 0
        for _, c in candles.iloc[-3:].iterrows():
            body = abs(c.close - c.open) or 0.0001
            upper = c.high - max(c.open, c.close)
            lower = min(c.open, c.close) - c.low
            if upper > body * 4 or lower > body * 4:
                extreme += 1

        return extreme == 3  # all 3 must be extreme

    @staticmethod
    def check_news_spike(candles: pd.DataFrame) -> bool:
        """Trap #4: News spike (>3% 1-candle move)"""
        returns = candles["close"].pct_change()
        return abs(returns.iloc[-1]) > config.NEWS_SPIKE_THRESHOLD

    @staticmethod
    def check_market_maker_trap(candles: pd.DataFrame) -> bool:
        """Trap #5: Market maker false breakout"""
        if len(candles) < 5:
            return False

        last = candles.iloc[-1]
        prev_high = candles["high"].iloc[-5:-1].max()
        prev_low = candles["low"].iloc[-5:-1].min()

        if last.high > prev_high and last.close < prev_high:
            return True
        if last.low < prev_low and last.close > prev_low:
            return True

        return False

    @staticmethod
    def check_orderblock_rejection(candles: pd.DataFrame) -> bool:
        """Trap #6: Order block rejection"""
        if len(candles) < 10:
            return False

        recent_high = candles["high"].iloc[-10:].max()
        recent_low = candles["low"].iloc[-10:].min()
        close = candles["close"].iloc[-1]

        zone_dist = 0.005
        reject_dist = 0.01

        # High zone rejection
        if abs(close - recent_high) / recent_high < zone_dist:
            if (recent_high - close) / recent_high > reject_dist:
                return True

        # Low zone rejection
        if abs(close - recent_low) / recent_low < zone_dist:
            if (close - recent_low) / recent_low > reject_dist:
                return True

        return False

    @staticmethod
    def check_entry_timing_lag(timestamp: datetime) -> bool:
        """Trap #7: Stale signal (>5 minutes old)"""
        age = (datetime.now() - timestamp).total_seconds()
        return age > 300  # more than 5 min old = trap

    @staticmethod
    def check_spread_expansion(bid: float, ask: float) -> bool:
        """Trap #8: Spread trap (low liquidity)"""
        if bid <= 0 or ask <= 0:
            return True

        spread_pct = ((ask - bid) / bid) * 100
        return spread_pct > config.MAX_SPREAD_PERCENT

    @staticmethod
    def check_three_candle_reversal(candles: pd.DataFrame) -> bool:
        """
        Trap #9: 3-candle reversal (fixed)
        - Must be alternating AND low volatility
        This prevents false triggering every time.
        """
        if len(candles) < 4:
            return False

        last_3 = candles["close"].diff().iloc[-3:]
        signs = list(np.sign(last_3))

        zigzag = signs in ([1, -1, 1], [-1, 1, -1])
        low_volatility = abs(last_3).mean() / candles["close"].iloc[-1] < 0.005  # <0.5%

        return zigzag and low_volatility

    @staticmethod
    def check_indicator_overfit(rsi: float, adx: float, macd_hist: float) -> bool:
        """Trap #10: Over-optimized indicators"""
        perfect = 0

        if abs(rsi - 50) < 0.5 or abs(rsi - 30) < 0.5 or abs(rsi - 70) < 0.5:
            perfect += 1
        if abs(adx - 25) < 0.5 or abs(adx - 50) < 0.5:
            perfect += 1
        if abs(macd_hist) < 0.005:
            perfect += 1

        return perfect >= 3

    @staticmethod
    def check_all_traps(candles: pd.DataFrame, bid: float = 0, ask: float = 0,
                        rsi: float = 50, adx: float = 25, macd_hist: float = 0) -> dict:
        """Runs all traps with FIXED logic"""

        try:
            last_timestamp = candles.index[-1]
            if not isinstance(last_timestamp, datetime):
                last_timestamp = datetime.now()
        except:
            last_timestamp = datetime.now()

        return {
            'ny_pullback': TrapDetector.check_ny_pullback_trap(candles),
            'liquidity_grab': TrapDetector.check_liquidity_grab(candles),
            'wick_manipulation': TrapDetector.check_wick_manipulation(candles),
            'news_spike': TrapDetector.check_news_spike(candles),
            'market_maker': TrapDetector.check_market_maker_trap(candles),
            'orderblock_rejection': TrapDetector.check_orderblock_rejection(candles),
            'entry_timing_lag': TrapDetector.check_entry_timing_lag(last_timestamp),
            'spread_expansion': TrapDetector.check_spread_expansion(bid, ask),
            'three_candle_reversal': TrapDetector.check_three_candle_reversal(candles),
            'indicator_overfit': TrapDetector.check_indicator_overfit(rsi, adx, macd_hist)
        }

    @staticmethod
    def get_trap_reasons(traps: dict):
        return [k for k, v in traps.items() if v]