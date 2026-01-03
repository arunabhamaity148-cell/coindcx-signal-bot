"""
entry_filters.py - Entry Timing Filters & Multi-Confirmation System
"""

from typing import Dict, Tuple, List
import pandas as pd
from indicators import Indicators
from coindcx_api import CoinDCXAPI


class EntryFilters:
    """
    Advanced entry filters to improve win rate
    """
    
    @staticmethod
    def check_pullback_entry(candles: pd.DataFrame, trend: str, 
                            mode: str) -> Tuple[bool, str]:
        """
        Check if price has pulled back to optimal entry zone
        Prevents late/chasing entries
        """
        
        close = candles['close']
        
        # Calculate EMAs
        ema_fast = Indicators.ema(close, 9)
        ema_slow = Indicators.ema(close, 21)
        
        if len(ema_fast) < 5 or len(ema_slow) < 5:
            return True, "Insufficient data"  # Allow by default
        
        current_price = float(close.iloc[-1])
        current_ema_fast = float(ema_fast.iloc[-1])
        current_ema_slow = float(ema_slow.iloc[-1])
        
        if trend == "LONG":
            # Confirm uptrend first
            if current_ema_fast <= current_ema_slow:
                return False, "Not in uptrend"
            
            # Distance from EMAs
            distance_from_fast = (current_price - current_ema_fast) / current_price * 100
            distance_from_slow = (current_price - current_ema_slow) / current_price * 100
            
            # Mode-specific entry zones
            if mode == 'TREND':
                # Should be near EMA21
                if 0 < distance_from_slow < 1.0:
                    return True, "Pullback to EMA21 (TREND)"
            
            elif mode == 'MID':
                # Should be near EMA9
                if 0 < distance_from_fast < 0.5:
                    return True, "Pullback to EMA9 (MID)"
            
            elif mode == 'QUICK':
                # Can be slightly above EMA9
                if -0.2 < distance_from_fast < 0.3:
                    return True, "Near EMA9 (QUICK)"
            
            # Price too far = late entry
            if distance_from_fast > 1.5:
                return False, f"Too far above EMA ({distance_from_fast:.2f}%)"
            
            # Price too far below = weak trend
            if distance_from_fast < -1.0:
                return False, "Price below EMA - trend weak"
        
        else:  # SHORT
            # Confirm downtrend
            if current_ema_fast >= current_ema_slow:
                return False, "Not in downtrend"
            
            distance_from_fast = (current_ema_fast - current_price) / current_price * 100
            distance_from_slow = (current_ema_slow - current_price) / current_price * 100
            
            if mode == 'TREND':
                if 0 < distance_from_slow < 1.0:
                    return True, "Pullback to EMA21 (TREND)"
            
            elif mode == 'MID':
                if 0 < distance_from_fast < 0.5:
                    return True, "Pullback to EMA9 (MID)"
            
            elif mode == 'QUICK':
                if -0.2 < distance_from_fast < 0.3:
                    return True, "Near EMA9 (QUICK)"
            
            if distance_from_fast > 1.5:
                return False, f"Too far below EMA ({distance_from_fast:.2f}%)"
            
            if distance_from_fast < -1.0:
                return False, "Price above EMA - trend weak"
        
        return True, "Entry timing acceptable"
    
    @staticmethod
    def multi_confirmation_check(pair: str, candles: pd.DataFrame,
                                 trend: str, mode: str) -> Tuple[bool, int, List[str]]:
        """
        7-point confirmation system
        Minimum required: TREND=5, MID=4, QUICK=3
        """
        
        confirmations = []
        score = 0
        
        close = candles['close']
        high = candles['high']
        low = candles['low']
        volume = candles['volume']
        
        # 1. EMA Alignment
        ema_9 = Indicators.ema(close, 9)
        ema_21 = Indicators.ema(close, 21)
        ema_50 = Indicators.ema(close, 50)
        
        if len(ema_9) >= 3 and len(ema_21) >= 3 and len(ema_50) >= 3:
            e9 = float(ema_9.iloc[-1])
            e21 = float(ema_21.iloc[-1])
            e50 = float(ema_50.iloc[-1])
            
            if trend == "LONG":
                if e9 > e21 > e50:
                    confirmations.append("✓ EMA Stack Bullish")
                    score += 1
            else:
                if e9 < e21 < e50:
                    confirmations.append("✓ EMA Stack Bearish")
                    score += 1
        
        # 2. MACD Histogram Growing
        macd_line, signal_line, histogram = Indicators.macd(close)
        
        if len(histogram) >= 3:
            h1 = float(histogram.iloc[-1])
            h2 = float(histogram.iloc[-2])
            h3 = float(histogram.iloc[-3])
            
            if trend == "LONG":
                if h1 > h2 > h3 and h1 > 0:
                    confirmations.append("✓ MACD Momentum Growing")
                    score += 1
            else:
                if h1 < h2 < h3 and h1 < 0:
                    confirmations.append("✓ MACD Momentum Growing")
                    score += 1
        
        # 3. Volume Confirmation
        if len(volume) >= 3:
            v1 = float(volume.iloc[-1])
            v2 = float(volume.iloc[-2])
            avg_vol = float(volume.tail(10).mean())
            
            if v1 > v2 and v1 > avg_vol * 1.1:
                confirmations.append("✓ Volume Increasing")
                score += 1
        
        # 4. RSI Healthy
        rsi = Indicators.rsi(close)
        
        if len(rsi) >= 1:
            current_rsi = float(rsi.iloc[-1])
            
            if 35 <= current_rsi <= 65:
                confirmations.append("✓ RSI Healthy")
                score += 1
        
        # 5. Price Action Pattern
        if len(candles) >= 3:
            if trend == "LONG":
                l1 = float(low.iloc[-1])
                l2 = float(low.iloc[-2])
                l3 = float(low.iloc[-3])
                
                if l1 > l2 > l3:
                    confirmations.append("✓ Higher Lows Pattern")
                    score += 1
            else:
                h1 = float(high.iloc[-1])
                h2 = float(high.iloc[-2])
                h3 = float(high.iloc[-3])
                
                if h1 < h2 < h3:
                    confirmations.append("✓ Lower Highs Pattern")
                    score += 1
        
        # 6. ADX Strong
        adx, plus_di, minus_di = Indicators.adx(high, low, close)
        
        if len(adx) >= 1:
            current_adx = float(adx.iloc[-1])
            
            MIN_ADX = {'TREND': 28, 'MID': 24, 'QUICK': 22}.get(mode, 24)
            
            if current_adx >= MIN_ADX:
                confirmations.append(f"✓ ADX Strong ({current_adx:.1f})")
                score += 1
        
        # 7. Higher Timeframe Aligned
        try:
            candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)
            
            if not candles_1h.empty:
                close_1h = candles_1h['close']
                ema_20_1h = Indicators.ema(close_1h, 20)
                
                if len(ema_20_1h) >= 1:
                    current_price = float(close.iloc[-1])
                    ema_1h = float(ema_20_1h.iloc[-1])
                    
                    if trend == "LONG":
                        if current_price > ema_1h:
                            confirmations.append("✓ 1H Aligned")
                            score += 1
                    else:
                        if current_price < ema_1h:
                            confirmations.append("✓ 1H Aligned")
                            score += 1
        except:
            pass
        
        # Minimum confirmations by mode
        MIN_CONFIRMATIONS = {
            'TREND': 5,
            'MID': 4,
            'QUICK': 3,
            'SCALP': 3
        }.get(mode, 4)
        
        is_confirmed = score >= MIN_CONFIRMATIONS
        
        return is_confirmed, score, confirmations
    
    @staticmethod
    def check_market_context(pair: str) -> Tuple[bool, str, Dict]:
        """
        Check overall market conditions
        """
        
        context = {
            'btc_trend': 'UNKNOWN',
            'overall_volume': 'NORMAL',
            'altcoin_favorable': False
        }
        
        try:
            # BTC Trend Check
            btc_candles = CoinDCXAPI.get_candles('BTCUSDT', '1h', 100)
            
            if not btc_candles.empty:
                btc_close = btc_candles['close']
                btc_ema_20 = Indicators.ema(btc_close, 20)
                btc_ema_50 = Indicators.ema(btc_close, 50)
                
                current_btc = float(btc_close.iloc[-1])
                ema20 = float(btc_ema_20.iloc[-1])
                ema50 = float(btc_ema_50.iloc[-1])
                
                # Classify BTC trend
                if current_btc > ema20 > ema50:
                    context['btc_trend'] = 'STRONG_UP'
                    btc_multiplier = 1.0
                elif current_btc > ema20 or current_btc > ema50:
                    context['btc_trend'] = 'MODERATE_UP'
                    btc_multiplier = 0.9
                elif current_btc < ema20 < ema50:
                    context['btc_trend'] = 'STRONG_DOWN'
                    btc_multiplier = 0.7
                else:
                    context['btc_trend'] = 'CHOPPY'
                    btc_multiplier = 0.6
            else:
                btc_multiplier = 1.0
            
            # Volume Context
            candles = CoinDCXAPI.get_candles(pair, '1h', 50)
            
            if not candles.empty:
                current_volume = float(candles['volume'].iloc[-1])
                avg_volume = float(candles['volume'].tail(20).mean())
                
                volume_ratio = current_volume / avg_volume
                
                if volume_ratio > 1.5:
                    context['overall_volume'] = 'HIGH'
                    volume_multiplier = 1.1
                elif volume_ratio > 1.0:
                    context['overall_volume'] = 'NORMAL'
                    volume_multiplier = 1.0
                elif volume_ratio > 0.7:
                    context['overall_volume'] = 'LOW'
                    volume_multiplier = 0.8
                else:
                    context['overall_volume'] = 'VERY_LOW'
                    volume_multiplier = 0.5
            else:
                volume_multiplier = 1.0
            
            # Context Score
            context_score = btc_multiplier * volume_multiplier
            
            if context_score >= 0.85:
                return True, f"Good context (score: {context_score:.2f})", context
            elif context_score >= 0.65:
                return True, f"Moderate context (score: {context_score:.2f})", context
            else:
                return False, f"Poor context (score: {context_score:.2f})", context
        
        except Exception as e:
            print(f"⚠️ Context check error: {e}")
            return True, "Context check skipped", context