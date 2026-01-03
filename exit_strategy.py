"""
exit_strategy.py - Dynamic Exit Management System
Handles: Partial booking, Time-based, Momentum, RSI divergence, Volume exits
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pandas as pd
from indicators import Indicators
from coindcx_api import CoinDCXAPI


class ExitStrategy:
    """
    Manages all exit conditions for active positions
    """
    
    def __init__(self):
        self.exit_history = []
    
    def check_all_exits(self, pair: str, signal: Dict, 
                       current_candles: pd.DataFrame) -> Tuple[bool, str, int]:
        """
        Check all exit conditions and return decision
        
        Returns:
            (should_exit, reason, booking_percent)
        """
        
        current_price = float(current_candles['close'].iloc[-1])
        entry = signal['entry']
        direction = signal['direction']
        mode = signal['mode']
        
        # Calculate current profit
        if direction == "LONG":
            profit_pct = ((current_price - entry) / entry) * 100
        else:
            profit_pct = ((entry - current_price) / entry) * 100
        
        # Priority 1: Quick Profit Booking
        should_exit, reason, pct = self._check_quick_profit(profit_pct, mode)
        if should_exit:
            return True, reason, pct
        
        # Priority 2: RSI Divergence
        should_exit, reason = self._check_rsi_divergence(
            current_candles, direction, profit_pct
        )
        if should_exit:
            return True, reason, 100
        
        # Priority 3: Time-Based Exit
        should_exit, reason = self._check_time_based_exit(
            signal, profit_pct, mode
        )
        if should_exit:
            return True, reason, 100
        
        # Priority 4: Momentum Exhaustion
        should_exit, reason = self._check_momentum_exhaustion(
            current_candles, direction, profit_pct
        )
        if should_exit:
            return True, reason, 80
        
        # Priority 5: Volume Dry-Up
        should_exit, reason = self._check_volume_dryup(
            current_candles, profit_pct
        )
        if should_exit:
            return True, reason, 70
        
        # Priority 6: Support/Resistance Hit
        should_exit, reason = self._check_key_levels(
            pair, current_price, direction, profit_pct
        )
        if should_exit:
            return True, reason, 100
        
        # No exit condition met
        return False, "Hold", 0
    
    def _check_quick_profit(self, profit_pct: float, mode: str) -> Tuple[bool, str, int]:
        """
        Quick profit targets - take money off the table early
        """
        
        QUICK_TARGETS = {
            'TREND': {
                'tp1': (0.8, 50),   # 0.8% profit = book 50%
                'tp2': (1.5, 30),   # 1.5% profit = book 30% more
                'tp3': (2.5, 20)    # 2.5% profit = book rest
            },
            'MID': {
                'tp1': (0.5, 50),
                'tp2': (1.0, 30),
                'tp3': (1.8, 20)
            },
            'QUICK': {
                'tp1': (0.3, 70),   # Quick mode = aggressive booking
                'tp2': (0.6, 20),
                'tp3': (1.0, 10)
            },
            'SCALP': {
                'tp1': (0.2, 80),
                'tp2': (0.4, 15),
                'tp3': (0.7, 5)
            }
        }
        
        targets = QUICK_TARGETS.get(mode, QUICK_TARGETS['MID'])
        
        # Check each target
        for tp_name, (target_pct, book_pct) in targets.items():
            if profit_pct >= target_pct:
                return True, f"Quick profit {tp_name}: {profit_pct:.2f}%", book_pct
        
        return False, "", 0
    
    def _check_rsi_divergence(self, candles: pd.DataFrame, 
                             direction: str, profit_pct: float) -> Tuple[bool, str]:
        """
        Detect RSI divergence - early warning of reversal
        """
        
        if len(candles) < 5:
            return False, ""
        
        close = candles['close']
        rsi = Indicators.rsi(close)
        
        if len(rsi) < 5:
            return False, ""
        
        # Get recent values
        current_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        
        if direction == "LONG":
            # Bearish divergence: price higher, RSI lower
            price_higher = current_price > prev_price
            rsi_lower = current_rsi < prev_rsi
            
            if price_higher and rsi_lower and current_rsi > 65:
                if profit_pct > 0.2:  # At least some profit
                    return True, f"Bearish divergence at {profit_pct:.2f}%"
        
        else:  # SHORT
            # Bullish divergence: price lower, RSI higher
            price_lower = current_price < prev_price
            rsi_higher = current_rsi > prev_rsi
            
            if price_lower and rsi_higher and current_rsi < 35:
                if profit_pct > 0.2:
                    return True, f"Bullish divergence at {profit_pct:.2f}%"
        
        return False, ""
    
    def _check_time_based_exit(self, signal: Dict, 
                               profit_pct: float, mode: str) -> Tuple[bool, str]:
        """
        Time-based exit with profit requirement
        """
        
        TIME_LIMITS = {
            'TREND': 16,    # hours
            'MID': 8,
            'QUICK': 4,
            'SCALP': 2
        }
        
        max_hours = TIME_LIMITS.get(mode, 8)
        
        entry_time = datetime.fromisoformat(signal['timestamp'])
        elapsed_hours = (datetime.now() - entry_time).seconds / 3600
        
        if elapsed_hours >= max_hours:
            if profit_pct > 0.1:  # Any profit
                return True, f"Time limit {elapsed_hours:.1f}h with {profit_pct:.2f}%"
            elif profit_pct > -0.3:  # Small loss acceptable
                return True, f"Time limit reached, minimize loss"
        
        return False, ""
    
    def _check_momentum_exhaustion(self, candles: pd.DataFrame,
                                  direction: str, profit_pct: float) -> Tuple[bool, str]:
        """
        Detect momentum exhaustion via MACD histogram
        """
        
        if len(candles) < 3:
            return False, ""
        
        close = candles['close']
        macd_line, signal_line, histogram = Indicators.macd(close)
        
        if len(histogram) < 3:
            return False, ""
        
        # Get last 3 histogram values
        h1 = float(histogram.iloc[-1])
        h2 = float(histogram.iloc[-2])
        h3 = float(histogram.iloc[-3])
        
        if direction == "LONG":
            # Histogram declining for 2 bars
            momentum_fading = h1 < h2 < h3
            
            if momentum_fading and profit_pct > 0.3:
                return True, f"Momentum fading at {profit_pct:.2f}%"
        
        else:  # SHORT
            momentum_fading = h1 > h2 > h3
            
            if momentum_fading and profit_pct > 0.3:
                return True, f"Momentum fading at {profit_pct:.2f}%"
        
        return False, ""
    
    def _check_volume_dryup(self, candles: pd.DataFrame, 
                           profit_pct: float) -> Tuple[bool, str]:
        """
        Exit when volume dries up with profit
        """
        
        volume = candles['volume']
        
        if len(volume) < 10:
            return False, ""
        
        current_vol = float(volume.iloc[-1])
        avg_vol = float(volume.tail(10).mean())
        
        # Volume dropped to 50% of average
        if current_vol < avg_vol * 0.5 and profit_pct > 0.4:
            return True, f"Volume dried at {profit_pct:.2f}%"
        
        return False, ""
    
    def _check_key_levels(self, pair: str, current_price: float,
                         direction: str, profit_pct: float) -> Tuple[bool, str]:
        """
        Exit near daily support/resistance levels
        """
        
        try:
            candles_daily = CoinDCXAPI.get_candles(pair, '1d', 30)
            
            if candles_daily.empty:
                return False, ""
            
            key_levels = Indicators.get_daily_key_levels(candles_daily)
            
            # Check distance to key levels
            for level_type, level_price in key_levels.items():
                distance_pct = abs(current_price - level_price) / current_price * 100
                
                # Within 0.3% of key level
                if distance_pct < 0.3:
                    if direction == "LONG" and level_type == "resistance":
                        if profit_pct > 0.3:
                            return True, f"Resistance at {profit_pct:.2f}%"
                    
                    elif direction == "SHORT" and level_type == "support":
                        if profit_pct > 0.3:
                            return True, f"Support at {profit_pct:.2f}%"
            
            return False, ""
        
        except Exception as e:
            print(f"Key level check error: {e}")
            return False, ""
    
    def log_exit(self, pair: str, signal: Dict, exit_price: float, 
                 exit_reason: str, booking_pct: int):
        """
        Log exit for analytics
        """
        
        exit_record = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'mode': signal['mode'],
            'direction': signal['direction'],
            'entry': signal['entry'],
            'exit': exit_price,
            'exit_reason': exit_reason,
            'booking_percent': booking_pct
        }
        
        self.exit_history.append(exit_record)
        
        return exit_record