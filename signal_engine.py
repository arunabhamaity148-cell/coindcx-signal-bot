"""
🧠 Signal Engine - 45 Logic Implementation
Logic 11-45: Trend, Momentum, Volume, Risk Filters
WITH DETAILED DEBUG LOGGING
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import *

class SignalEngine:
    def __init__(self):
        self.signals_generated = 0
        self.consecutive_losses = 0
        self.last_signal_time = {}
        
    def calculate_ema(self, data, period):
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, MACD_FAST)
        ema_slow = self.calculate_ema(data, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, MACD_SIGNAL)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, period=20, std=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    def check_ema_trend(self, df):
        """Logic 11-12: EMA Trend Alignment"""
        if len(df) < EMA_SLOW:
            return "NEUTRAL"
        
        close = df['close']
        ema20 = self.calculate_ema(close, EMA_FAST)
        ema50 = self.calculate_ema(close, EMA_MID)
        ema200 = self.calculate_ema(close, EMA_SLOW)
        
        # Drop NaN values
        ema20 = ema20.dropna()
        ema50 = ema50.dropna()
        ema200 = ema200.dropna()
        
        if len(ema20) == 0 or len(ema50) == 0 or len(ema200) == 0:
            return "NEUTRAL"
        
        df['ema20'] = ema20
        df['ema50'] = ema50
        df['ema200'] = ema200
        
        # Bull trend: 20 > 50 > 200
        bull_trend = (ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1])
        
        # Bear trend: 20 < 50 < 200
        bear_trend = (ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1])
        
        if bull_trend:
            return "BULL"
        elif bear_trend:
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def check_market_structure(self, df):
        """Logic 13: Higher High / Lower Low Detection"""
        if len(df) < 10:
            return "NEUTRAL_STRUCTURE"
        
        highs = df['high'].rolling(5).max().dropna()
        lows = df['low'].rolling(5).min().dropna()
        
        if len(highs) < 5 or len(lows) < 5:
            return "NEUTRAL_STRUCTURE"
        
        # Check last 3 swings
        recent_highs = highs.tail(10).values
        recent_lows = lows.tail(10).values
        
        # Higher Highs (HH)
        hh = len([i for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1]]) >= 2
        
        # Lower Lows (LL)
        ll = len([i for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1]]) >= 2
        
        if hh:
            return "BULLISH_STRUCTURE"
        elif ll:
            return "BEARISH_STRUCTURE"
        return "NEUTRAL_STRUCTURE"
    
    def check_rsi_conditions(self, df):
        """Logic 21-23: RSI Oversold/Overbought/Divergence"""
        close = df['close']
        rsi = self.calculate_rsi(close, RSI_PERIOD)
        df['rsi'] = rsi
        
        current_rsi = rsi.iloc[-1]
        
        signals = []
        
        # Oversold reversal (Logic 21)
        if current_rsi < RSI_OVERSOLD and rsi.iloc[-2] >= RSI_OVERSOLD:
            signals.append("RSI_OVERSOLD_REVERSAL")
        
        # Overbought rejection (Logic 22)
        if current_rsi > RSI_OVERBOUGHT and rsi.iloc[-2] <= RSI_OVERBOUGHT:
            signals.append("RSI_OVERBOUGHT_REJECTION")
        
        # Bullish divergence (Logic 23)
        if len(close) > 20:
            price_low = close.tail(20).min()
            rsi_low = rsi.tail(20).min()
            if close.iloc[-1] <= price_low and rsi.iloc[-1] > rsi_low:
                signals.append("RSI_BULLISH_DIVERGENCE")
        
        return signals
    
    def check_macd_signals(self, df):
        """Logic 24-25: MACD Cross & Histogram"""
        close = df['close']
        macd_line, signal_line, histogram = self.calculate_macd(close)
        
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        
        signals = []
        
        # Bullish cross (Logic 24)
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signals.append("MACD_BULLISH_CROSS")
        
        # Bearish cross
        if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            signals.append("MACD_BEARISH_CROSS")
        
        # Histogram momentum (Logic 25)
        if histogram.iloc[-1] > histogram.iloc[-2] > 0:
            signals.append("MACD_BULLISH_MOMENTUM")
        elif histogram.iloc[-1] < histogram.iloc[-2] < 0:
            signals.append("MACD_BEARISH_MOMENTUM")
        
        return signals
    
    def check_bollinger_bands(self, df):
        """Logic 28-29: BB Squeeze & Mean Reversion"""
        close = df['close']
        upper, middle, lower = self.calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        signals = []
        
        # Squeeze expansion (Logic 28)
        bb_width = (upper - lower) / middle
        if bb_width.iloc[-1] > bb_width.rolling(20).mean().iloc[-1]:
            signals.append("BB_EXPANSION")
        
        # Mean reversion (Logic 29)
        if close.iloc[-1] < lower.iloc[-1]:
            signals.append("BB_OVERSOLD")
        elif close.iloc[-1] > upper.iloc[-1]:
            signals.append("BB_OVERBOUGHT")
        
        return signals
    
    def check_volume_confirmation(self, df):
        """Logic 31-32: Volume Spike & Dry-up"""
        volume = df['volume']
        volume_ma = volume.rolling(VOLUME_MA_PERIOD).mean()
        
        df['volume_ma'] = volume_ma
        
        current_vol = volume.iloc[-1]
        avg_vol = volume_ma.iloc[-1]
        
        if current_vol > avg_vol * 1.2:  # Lowered from 1.5
            return "VOLUME_SPIKE"
        elif current_vol < avg_vol * 0.5:
            return "VOLUME_DRY"
        
        return "VOLUME_NORMAL"
    
    def check_vwap(self, df):
        """Logic 33-35: VWAP Logic"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        df['vwap'] = vwap
        
        current_price = df['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        if current_price > current_vwap and df['close'].iloc[-2] <= vwap.iloc[-2]:
            return "VWAP_RECLAIM"
        elif current_price < current_vwap and df['close'].iloc[-2] >= vwap.iloc[-2]:
            return "VWAP_REJECTION"
        
        return "VWAP_NEUTRAL"
    
    def check_manipulation_filter(self, df):
        """Logic 42: Long Wick Filter"""
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        
        if total_range == 0:
            return False
        
        wick_ratio = body / total_range
        
        # If body > 30% of candle, it's valid (loosened from 70%)
        return wick_ratio > 0.2
    
    def check_round_number_trap(self, price):
        """Logic 40: Round Number Trap Avoidance"""
        # Avoid prices like 50000, 100000, etc.
        str_price = str(int(price))
        trailing_zeros = len(str_price) - len(str_price.rstrip('0'))
        
        return trailing_zeros >= 4  # 4+ trailing zeros = trap zone
    
    def generate_signal(self, pair_data, mode="mid"):
        """Main signal generation combining all 45 logics"""
        symbol = pair_data['symbol']
        df = pair_data['data'].copy()
        
        print(f"\n🔎 Analyzing {symbol}...")
        
        if len(df) < 50:
            print(f"  ❌ BLOCKED: Insufficient data ({len(df)} candles)")
            return None
        
        try:
            # Apply all filters
            ema_trend = self.check_ema_trend(df)
            print(f"  📊 EMA Trend: {ema_trend}")
            
            structure = self.check_market_structure(df)
            print(f"  📈 Structure: {structure}")
            
            rsi_signals = self.check_rsi_conditions(df)
            print(f"  📉 RSI Signals: {rsi_signals if rsi_signals else 'None'}")
            
            macd_signals = self.check_macd_signals(df)
            print(f"  🔄 MACD Signals: {macd_signals if macd_signals else 'None'}")
            
            bb_signals = self.check_bollinger_bands(df)
            print(f"  📊 BB Signals: {bb_signals if bb_signals else 'None'}")
            
            volume_status = self.check_volume_confirmation(df)
            print(f"  📦 Volume: {volume_status}")
            
            vwap_status = self.check_vwap(df)
            print(f"  💹 VWAP: {vwap_status}")
            
            # Anti-trap filters
            if not self.check_manipulation_filter(df):
                print(f"  ❌ BLOCKED: Manipulation candle detected")
                return None
            
            current_price = df['close'].iloc[-1]
            if self.check_round_number_trap(current_price):
                print(f"  ❌ BLOCKED: Round number trap at ₹{current_price:.0f}")
                return None
            
            # Logic 45: One-side lock
            if symbol in self.last_signal_time:
                time_diff = (datetime.now() - self.last_signal_time[symbol]).seconds / 60
                if time_diff < SIGNAL_MODES[mode]['hold_time']:
                    print(f"  ❌ BLOCKED: Cooldown active ({time_diff:.0f}/{SIGNAL_MODES[mode]['hold_time']} min)")
                    return None
            
            # Logic 43: Consecutive loss cooldown
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                print(f"  ❌ BLOCKED: Max consecutive losses ({self.consecutive_losses})")
                return None
            
            # LONG SIGNAL LOGIC
            long_score = 0
            if ema_trend == "BULL": long_score += 2
            if structure == "BULLISH_STRUCTURE": long_score += 2
            if "RSI_OVERSOLD_REVERSAL" in rsi_signals: long_score += 2
            if "MACD_BULLISH_CROSS" in macd_signals: long_score += 2
            if "BB_OVERSOLD" in bb_signals: long_score += 1
            if volume_status == "VOLUME_SPIKE": long_score += 2
            if vwap_status == "VWAP_RECLAIM": long_score += 2
            
            # SHORT SIGNAL LOGIC
            short_score = 0
            if ema_trend == "BEAR": short_score += 2
            if structure == "BEARISH_STRUCTURE": short_score += 2
            if "RSI_OVERBOUGHT_REJECTION" in rsi_signals: short_score += 2
            if "MACD_BEARISH_CROSS" in macd_signals: short_score += 2
            if "BB_OVERBOUGHT" in bb_signals: short_score += 1
            if volume_status == "VOLUME_SPIKE": short_score += 2
            if vwap_status == "VWAP_REJECTION": short_score += 2
            
            print(f"  🎯 LONG Score: {long_score}/15")
            print(f"  🎯 SHORT Score: {short_score}/15")
            
            # Get RSI value safely
            rsi_value = df.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in df.columns else 50
            macd_hist_value = df.get('macd_hist', pd.Series([0])).iloc[-1] if 'macd_hist' in df.columns else 0
            
            # Use MIN_SIGNAL_SCORE from config
            if long_score >= MIN_SIGNAL_SCORE:
                print(f"  ✅ LONG SIGNAL PASSED! Score: {long_score}/15")
                self.last_signal_time[symbol] = datetime.now()
                return {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'score': long_score,
                    'mode': mode,
                    'timestamp': datetime.now(),
                    'indicators': {
                        'ema_trend': ema_trend,
                        'rsi': rsi_value,
                        'macd_hist': macd_hist_value,
                        'volume_ratio': volume_status
                    }
                }
            
            elif short_score >= MIN_SIGNAL_SCORE:
                print(f"  ✅ SHORT SIGNAL PASSED! Score: {short_score}/15")
                self.last_signal_time[symbol] = datetime.now()
                return {
                    'symbol': symbol,
                    'direction': 'SHORT',
                    'entry_price': current_price,
                    'score': short_score,
                    'mode': mode,
                    'timestamp': datetime.now(),
                    'indicators': {
                        'ema_trend': ema_trend,
                        'rsi': rsi_value,
                        'macd_hist': macd_hist_value,
                        'volume_ratio': volume_status
                    }
                }
            else:
                print(f"  ❌ BLOCKED: Score too low (LONG: {long_score}, SHORT: {short_score}, Need: {MIN_SIGNAL_SCORE})")
            
        except Exception as e:
            print(f"  ⚠️ Signal generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return None