"""
UNIQUE & PROPRIETARY Trading Logic
Nobody else has this combination
Smart, Different, Effective
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import logging

from helpers import TechnicalIndicators

logger = logging.getLogger(__name__)

class UniqueSignalGenerator:
    """
    PROPRIETARY ALGORITHM
    Combines unique indicators nobody else uses together
    """
    
    def __init__(self):
        self.signal_memory = {}  # Track recent signals per coin
        self.market_pulse = {}   # Track market momentum shifts
    
    # ========== UNIQUE LOGIC 1: MOMENTUM WAVE DETECTION ==========
    def detect_momentum_wave(self, df: pd.DataFrame) -> Tuple[str, int]:
        """
        PROPRIETARY: Wave-based momentum detection
        Uses price velocity + acceleration together
        """
        
        # Calculate price velocity (rate of change)
        velocity = df['close'].pct_change(5)  # 5-candle velocity
        
        # Calculate acceleration (change in velocity)
        acceleration = velocity.diff(3)  # 3-candle acceleration
        
        current_vel = velocity.iloc[-1]
        current_acc = acceleration.iloc[-1]
        
        # Wave states
        if current_vel > 0.01 and current_acc > 0:
            return "ACCELERATING_UP", 15  # Strongest signal
        elif current_vel > 0.005 and current_acc > 0:
            return "BUILDING_UP", 10
        elif current_vel < -0.01 and current_acc < 0:
            return "ACCELERATING_DOWN", 15
        elif current_vel < -0.005 and current_acc < 0:
            return "BUILDING_DOWN", 10
        elif abs(current_vel) < 0.002:
            return "DORMANT", 0  # Skip dormant coins
        else:
            return "TRANSITIONING", 3
    
    # ========== UNIQUE LOGIC 2: SMART MONEY TRACKER ==========
    def track_smart_money(self, df: pd.DataFrame, orderbook: dict) -> Tuple[str, int]:
        """
        PROPRIETARY: Detect institutional/smart money activity
        Combines volume profile + orderbook walls + price action
        """
        
        score = 0
        signals = []
        
        # 1. Volume Profile Analysis
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = df['volume'].iloc[-50:].mean()
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
        
        # 2. Large Candle Detection (Smart money entry)
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        candle_range = last_candle['high'] - last_candle['low']
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        # Smart money signature: Large volume + Strong body + Low wicks
        if vol_surge > 2.0 and body_ratio > 0.7:
            if last_candle['close'] > last_candle['open']:
                signals.append("SMART_BUY")
                score += 12
            else:
                signals.append("SMART_SELL")
                score += 12
        
        # 3. Orderbook Wall Detection (Support/Resistance from big players)
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            bid_sizes = [b[1] for b in orderbook['bids'][:10]]
            ask_sizes = [a[1] for a in orderbook['asks'][:10]]
            
            avg_bid = np.mean(bid_sizes)
            avg_ask = np.mean(ask_sizes)
            max_bid = max(bid_sizes)
            max_ask = max(ask_sizes)
            
            # Large wall = Smart money positioning
            if max_bid > avg_bid * 4:
                signals.append("BID_WALL")
                score += 5
            if max_ask > avg_ask * 4:
                signals.append("ASK_WALL")
                score += 5
        
        direction = "BULLISH" if "SMART_BUY" in signals or "BID_WALL" in signals else \
                   "BEARISH" if "SMART_SELL" in signals or "ASK_WALL" in signals else \
                   "NEUTRAL"
        
        return direction, min(score, 15)
    
    # ========== UNIQUE LOGIC 3: CHAOS THEORY OSCILLATOR ==========
    def chaos_oscillator(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        PROPRIETARY: Chaos-based market state detection
        Measures market disorder vs order
        """
        
        # Calculate fractal dimension (market complexity)
        highs = df['high'].iloc[-20:].values
        lows = df['low'].iloc[-20:].values
        
        # Range of highs and lows
        price_range = max(highs) - min(lows)
        
        # Path length (total price movement)
        path_length = sum(abs(df['close'].iloc[-20:].diff().dropna()))
        
        # Chaos ratio: Higher = more chaotic (volatile)
        chaos_ratio = path_length / price_range if price_range > 0 else 0
        
        # Interpret chaos
        if chaos_ratio > 15:
            return chaos_ratio, 0  # Too chaotic, avoid
        elif 10 < chaos_ratio <= 15:
            return chaos_ratio, 5  # High volatility, risky
        elif 6 < chaos_ratio <= 10:
            return chaos_ratio, 12  # Optimal: Moving with structure
        elif 3 < chaos_ratio <= 6:
            return chaos_ratio, 8  # Good movement
        else:
            return chaos_ratio, 0  # Too flat, no opportunity
    
    # ========== UNIQUE LOGIC 4: LIQUIDITY VACUUM ZONES ==========
    def find_liquidity_vacuum(self, df: pd.DataFrame) -> Tuple[bool, List[float], int]:
        """
        PROPRIETARY: Find zones with no resistance/support
        Price moves fast through these zones
        """
        
        # Volume profile: Find low-volume price zones
        price_levels = np.linspace(df['low'].min(), df['high'].max(), 20)
        volume_at_levels = []
        
        for level in price_levels:
            # Find candles that touched this level
            touched = df[(df['low'] <= level) & (df['high'] >= level)]
            vol_sum = touched['volume'].sum() if len(touched) > 0 else 0
            volume_at_levels.append(vol_sum)
        
        # Find vacuum zones (very low volume areas)
        avg_volume = np.mean(volume_at_levels)
        vacuum_zones = []
        
        for i, vol in enumerate(volume_at_levels):
            if vol < avg_volume * 0.3:  # Less than 30% of average
                vacuum_zones.append(price_levels[i])
        
        current_price = df['close'].iloc[-1]
        
        # Check if price is near a vacuum zone
        near_vacuum = False
        score = 0
        
        for zone in vacuum_zones:
            if abs(current_price - zone) / current_price < 0.01:  # Within 1%
                near_vacuum = True
                score = 10  # High score: Price will move fast through vacuum
                break
        
        return near_vacuum, vacuum_zones, score
    
    # ========== UNIQUE LOGIC 5: MARKET SYNCHRONY INDEX ==========
    def calculate_market_sync(self, df: pd.DataFrame) -> Tuple[str, int]:
        """
        PROPRIETARY: Measure how in-sync price, volume, and volatility are
        When all align = high probability move
        """
        
        # Normalize indicators to 0-1 scale
        price_change = df['close'].pct_change(10).iloc[-1]
        volume_change = (df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean()) - 1
        
        atr = TechnicalIndicators.atr(df, 14).iloc[-1]
        atr_avg = TechnicalIndicators.atr(df, 14).iloc[-50:].mean()
        volatility_change = (atr / atr_avg) - 1 if atr_avg > 0 else 0
        
        # Check synchrony
        bullish_sync = price_change > 0 and volume_change > 0 and volatility_change > 0
        bearish_sync = price_change < 0 and volume_change > 0 and volatility_change > 0
        
        # Calculate sync strength
        sync_strength = abs(price_change) + abs(volume_change) + abs(volatility_change)
        
        if bullish_sync and sync_strength > 0.5:
            return "BULLISH_SYNC", 12
        elif bearish_sync and sync_strength > 0.5:
            return "BEARISH_SYNC", 12
        elif bullish_sync:
            return "WEAK_BULLISH_SYNC", 6
        elif bearish_sync:
            return "WEAK_BEARISH_SYNC", 6
        else:
            return "DESYNC", 0
    
    # ========== UNIQUE LOGIC 6: REVERSAL EXHAUSTION DETECTOR ==========
    def detect_exhaustion(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        """
        PROPRIETARY: Detect when trends are exhausted
        Uses diminishing momentum + volume divergence
        """
        
        # Price making new highs/lows but...
        recent_high = df['high'].iloc[-5:].max()
        recent_low = df['low'].iloc[-5:].min()
        prev_high = df['high'].iloc[-20:-5].max()
        prev_low = df['low'].iloc[-20:-5].min()
        
        # Volume comparison
        recent_vol = df['volume'].iloc[-5:].mean()
        prev_vol = df['volume'].iloc[-20:-5].mean()
        
        # RSI for overbought/oversold
        rsi = TechnicalIndicators.rsi(df, 14).iloc[-1]
        
        # Bullish Exhaustion: New high + Lower volume + Overbought
        if recent_high > prev_high and recent_vol < prev_vol and rsi > 65:
            return True, "BULLISH_EXHAUSTION", 14
        
        # Bearish Exhaustion: New low + Lower volume + Oversold
        elif recent_low < prev_low and recent_vol < prev_vol and rsi < 35:
            return True, "BEARISH_EXHAUSTION", 14
        
        return False, "NO_EXHAUSTION", 0
    
    # ========== UNIQUE LOGIC 7: TIME-WEIGHTED MOMENTUM ==========
    def time_weighted_momentum(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        PROPRIETARY: Recent momentum matters more than old
        Exponential weighting on momentum
        """
        
        # Calculate momentum for last 10 candles
        momentum = df['close'].pct_change().iloc[-10:].values
        
        # Apply exponential weights (recent = more weight)
        weights = np.exp(np.linspace(0, 1, 10))
        weights = weights / weights.sum()  # Normalize
        
        weighted_momentum = np.sum(momentum * weights)
        
        # Score based on weighted momentum
        if weighted_momentum > 0.02:
            return weighted_momentum, 12
        elif weighted_momentum > 0.01:
            return weighted_momentum, 8
        elif weighted_momentum < -0.02:
            return weighted_momentum, 12
        elif weighted_momentum < -0.01:
            return weighted_momentum, 8
        else:
            return weighted_momentum, 2
    
    # ========== UNIQUE LOGIC 8: ADAPTIVE VOLATILITY BANDS ==========
    def adaptive_bands(self, df: pd.DataFrame) -> Tuple[str, int]:
        """
        PROPRIETARY: Bands that adapt to market volatility
        Not static like Bollinger Bands
        """
        
        # Calculate adaptive period based on recent volatility
        recent_volatility = df['close'].pct_change().iloc[-20:].std()
        
        if recent_volatility > 0.03:
            period = 10  # Fast bands in volatile market
        elif recent_volatility > 0.015:
            period = 20
        else:
            period = 30  # Slow bands in calm market
        
        # Calculate adaptive bands
        sma = df['close'].rolling(period).mean().iloc[-1]
        std = df['close'].rolling(period).std().iloc[-1]
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        current_price = df['close'].iloc[-1]
        
        # Band position
        if current_price < lower:
            return "BELOW_LOWER_BAND", 10  # Oversold
        elif current_price > upper:
            return "ABOVE_UPPER_BAND", 10  # Overbought
        elif current_price > sma:
            return "ABOVE_MIDDLE", 5
        elif current_price < sma:
            return "BELOW_MIDDLE", 5
        else:
            return "AT_MIDDLE", 3
    
    # ========== UNIQUE LOGIC 9: STEALTH ACCUMULATION DETECTOR ==========
    def detect_stealth_accumulation(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """
        PROPRIETARY: Detect quiet accumulation before breakout
        Price flat + Volume increasing = Smart money accumulating
        """
        
        # Price movement (should be small)
        price_std = df['close'].iloc[-20:].std()
        price_mean = df['close'].iloc[-20:].mean()
        price_cv = price_std / price_mean if price_mean > 0 else 0  # Coefficient of variation
        
        # Volume trend (should be increasing)
        vol_early = df['volume'].iloc[-20:-10].mean()
        vol_recent = df['volume'].iloc[-10:].mean()
        vol_increase = (vol_recent / vol_early) - 1 if vol_early > 0 else 0
        
        # Stealth accumulation: Low price variance + High volume increase
        if price_cv < 0.02 and vol_increase > 0.3:
            return True, 15  # Strong signal
        elif price_cv < 0.03 and vol_increase > 0.2:
            return True, 10
        
        return False, 0
    
    # ========== UNIQUE LOGIC 10: SENTIMENT SHIFT DETECTOR ==========
    def detect_sentiment_shift(self, df: pd.DataFrame) -> Tuple[str, int]:
        """
        PROPRIETARY: Detect rapid sentiment changes
        Sudden change in candle patterns
        """
        
        # Analyze last 5 candles
        bullish_candles = 0
        bearish_candles = 0
        
        for i in range(-5, 0):
            candle = df.iloc[i]
            if candle['close'] > candle['open']:
                bullish_candles += 1
            else:
                bearish_candles += 1
        
        # Previous 5 candles
        prev_bullish = 0
        prev_bearish = 0
        
        for i in range(-10, -5):
            candle = df.iloc[i]
            if candle['close'] > candle['open']:
                prev_bullish += 1
            else:
                prev_bearish += 1
        
        # Detect shift
        if prev_bearish >= 4 and bullish_candles >= 4:
            return "BEARISH_TO_BULLISH_SHIFT", 13
        elif prev_bullish >= 4 and bearish_candles >= 4:
            return "BULLISH_TO_BEARISH_SHIFT", 13
        elif bullish_candles >= 4:
            return "STRONG_BULLISH", 8
        elif bearish_candles >= 4:
            return "STRONG_BEARISH", 8
        
        return "NO_CLEAR_SHIFT", 2
    
    # ========== MAIN SIGNAL GENERATION ==========
    async def generate_signal(
        self,
        market: str,
        candles: pd.DataFrame,
        orderbook: dict,
        timeframe: str,
        current_price_inr: float
    ) -> Dict:
        """
        PROPRIETARY SIGNAL GENERATION
        Combines all unique logics
        """
        
        if len(candles) < 100:
            return None
        
        total_score = 0
        details = {}
        
        # === UNIQUE LOGIC 1: Momentum Wave ===
        wave_state, wave_score = self.detect_momentum_wave(candles)
        total_score += wave_score
        details['momentum_wave'] = wave_state
        
        # === UNIQUE LOGIC 2: Smart Money ===
        smart_money, sm_score = self.track_smart_money(candles, orderbook)
        total_score += sm_score
        details['smart_money'] = smart_money
        
        # === UNIQUE LOGIC 3: Chaos Oscillator ===
        chaos_value, chaos_score = self.chaos_oscillator(candles)
        total_score += chaos_score
        details['chaos_ratio'] = round(chaos_value, 2)
        
        # === UNIQUE LOGIC 4: Liquidity Vacuum ===
        near_vacuum, vacuum_zones, vacuum_score = self.find_liquidity_vacuum(candles)
        total_score += vacuum_score
        details['near_vacuum'] = near_vacuum
        
        # === UNIQUE LOGIC 5: Market Sync ===
        sync_state, sync_score = self.calculate_market_sync(candles)
        total_score += sync_score
        details['market_sync'] = sync_state
        
        # === UNIQUE LOGIC 6: Exhaustion ===
        exhausted, exhaust_type, exhaust_score = self.detect_exhaustion(candles)
        total_score += exhaust_score
        details['exhaustion'] = exhaust_type
        
        # === UNIQUE LOGIC 7: Time-Weighted Momentum ===
        tw_momentum, tw_score = self.time_weighted_momentum(candles)
        total_score += tw_score
        details['tw_momentum'] = round(tw_momentum, 4)
        
        # === UNIQUE LOGIC 8: Adaptive Bands ===
        band_position, band_score = self.adaptive_bands(candles)
        total_score += band_score
        details['band_position'] = band_position
        
        # === UNIQUE LOGIC 9: Stealth Accumulation ===
        accumulating, accum_score = self.detect_stealth_accumulation(candles)
        total_score += accum_score
        details['accumulation'] = accumulating
        
        # === UNIQUE LOGIC 10: Sentiment Shift ===
        sentiment, sentiment_score = self.detect_sentiment_shift(candles)
        total_score += sentiment_score
        details['sentiment'] = sentiment
        
        # === CALCULATE FINAL SCORE ===
        max_possible = 130
        final_score = int((total_score / max_possible) * 100)
        
        # Minimum threshold
        if final_score < 45:
            return None
        
        # === DETERMINE DIRECTION ===
        bullish_signals = 0
        bearish_signals = 0
        
        # Wave
        if "UP" in wave_state:
            bullish_signals += 3
        elif "DOWN" in wave_state:
            bearish_signals += 3
        
        # Smart Money
        if smart_money == "BULLISH":
            bullish_signals += 3
        elif smart_money == "BEARISH":
            bearish_signals += 3
        
        # Sync
        if "BULLISH" in sync_state:
            bullish_signals += 2
        elif "BEARISH" in sync_state:
            bearish_signals += 2
        
        # Exhaustion (reverse signal)
        if "BULLISH_EXHAUSTION" in exhaust_type:
            bearish_signals += 2  # Reversal coming
        elif "BEARISH_EXHAUSTION" in exhaust_type:
            bullish_signals += 2
        
        # Time-weighted momentum
        if tw_momentum > 0:
            bullish_signals += 2
        else:
            bearish_signals += 2
        
        # Band position
        if "BELOW" in band_position:
            bullish_signals += 1
        elif "ABOVE" in band_position:
            bearish_signals += 1
        
        # Accumulation
        if accumulating:
            bullish_signals += 2
        
        # Sentiment
        if "BULLISH" in sentiment:
            bullish_signals += 1
        elif "BEARISH" in sentiment:
            bearish_signals += 1
        
        if bullish_signals <= bearish_signals:
            if bearish_signals - bullish_signals < 2:
                return None  # Not clear enough
            side = "SELL"
        else:
            if bullish_signals - bearish_signals < 2:
                return None
            side = "BUY"
        
        # === CALCULATE LEVELS ===
        atr = TechnicalIndicators.atr(candles, 14).iloc[-1]
        
        if side == "BUY":
            entry = current_price_inr
            sl = entry - (atr * 1.8)
            tp = entry + (atr * 3.2)
        else:
            entry = current_price_inr
            sl = entry + (atr * 1.8)
            tp = entry - (atr * 3.2)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        # === CONFIDENCE ===
        if final_score >= 70:
            confidence = "HIGH"
        elif final_score >= 55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # === MODE ===
        mode = {"5m": "QUICK", "15m": "MID", "1h": "TREND"}.get(timeframe, "MID")
        
        return {
            'market': market,
            'timeframe': timeframe,
            'side': side,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'rr_ratio': round(rr_ratio, 1),
            'logic_score': final_score,
            'confidence': confidence,
            'mode': mode,
            'details': details
        }