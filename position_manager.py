"""
position_manager.py - Position Tracking & Trailing Stop Management
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
from indicators import Indicators


class TrailingStopManager:
    """
    Manages trailing stops for active positions
    """
    
    def __init__(self):
        self.trailing_stops = {}  # {pair: trail_config}
    
    def initialize_trail(self, pair: str, signal: Dict):
        """
        Initialize trailing stop for a new position
        """
        
        self.trailing_stops[pair] = {
            'entry': signal['entry'],
            'initial_sl': signal['sl'],
            'current_sl': signal['sl'],
            'direction': signal['direction'],
            'mode': signal['mode'],
            'highest_price': signal['entry'] if signal['direction'] == 'LONG' else signal['entry'],
            'lowest_price': signal['entry'] if signal['direction'] == 'SHORT' else signal['entry'],
            'trail_activated': False,
            'lock_profit_at': None,
            'last_update': datetime.now()
        }
        
        print(f"🎯 Trailing stop initialized for {pair}")
    
    def update_trailing_stop(self, pair: str, current_price: float,
                            current_candles: pd.DataFrame) -> Tuple[Optional[float], str]:
        """
        Update trailing stop based on current price
        
        Returns:
            (new_sl, update_message)
        """
        
        if pair not in self.trailing_stops:
            return None, "No active trail"
        
        trail = self.trailing_stops[pair]
        entry = trail['entry']
        direction = trail['direction']
        mode = trail['mode']
        current_sl = trail['current_sl']
        
        # Calculate current profit
        if direction == "LONG":
            profit_pct = ((current_price - entry) / entry) * 100
            
            # Update highest price
            if current_price > trail['highest_price']:
                trail['highest_price'] = current_price
        else:
            profit_pct = ((entry - current_price) / entry) * 100
            
            # Update lowest price
            if current_price < trail['lowest_price']:
                trail['lowest_price'] = current_price
        
        # STAGE 1: Move to Breakeven (0.4% profit)
        if profit_pct >= 0.4 and not trail['trail_activated']:
            new_sl = entry
            trail['trail_activated'] = True
            trail['current_sl'] = new_sl
            trail['last_update'] = datetime.now()
            
            return new_sl, f"🔒 Breakeven at {profit_pct:.2f}%"
        
        # STAGE 2: Lock 50% Profit (0.8% profit)
        if profit_pct >= 0.8 and trail['lock_profit_at'] is None:
            if direction == "LONG":
                profit_to_lock = (current_price - entry) * 0.5
                new_sl = entry + profit_to_lock
            else:
                profit_to_lock = (entry - current_price) * 0.5
                new_sl = entry - profit_to_lock
            
            trail['lock_profit_at'] = 0.5
            trail['current_sl'] = new_sl
            trail['last_update'] = datetime.now()
            
            return new_sl, f"🔒 50% Profit Locked at {profit_pct:.2f}%"
        
        # STAGE 3: Dynamic Trailing (1.2%+ profit)
        if profit_pct >= 1.2:
            atr = Indicators.atr(
                current_candles['high'],
                current_candles['low'],
                current_candles['close']
            )
            
            if len(atr) >= 1:
                current_atr = float(atr.iloc[-1])
                
                # Trail distance by mode
                TRAIL_MULTIPLIER = {
                    'TREND': 2.0,
                    'MID': 1.5,
                    'QUICK': 1.2,
                    'SCALP': 1.0
                }.get(mode, 1.5)
                
                trail_distance = current_atr * TRAIL_MULTIPLIER
                
                if direction == "LONG":
                    potential_sl = trail['highest_price'] - trail_distance
                    
                    # Only move SL up
                    if potential_sl > current_sl:
                        trail['current_sl'] = potential_sl
                        trail['last_update'] = datetime.now()
                        return potential_sl, f"📈 Trailing UP at {profit_pct:.2f}%"
                
                else:  # SHORT
                    potential_sl = trail['lowest_price'] + trail_distance
                    
                    # Only move SL down
                    if potential_sl < current_sl:
                        trail['current_sl'] = potential_sl
                        trail['last_update'] = datetime.now()
                        return potential_sl, f"📉 Trailing DOWN at {profit_pct:.2f}%"
        
        # STAGE 4: Aggressive Lock (2%+ profit)
        if profit_pct >= 2.0:
            if direction == "LONG":
                profit_to_lock = (current_price - entry) * 0.7
                new_sl = entry + profit_to_lock
            else:
                profit_to_lock = (entry - current_price) * 0.7
                new_sl = entry - profit_to_lock
            
            # Only update if better
            if direction == "LONG":
                if new_sl > current_sl:
                    trail['current_sl'] = new_sl
                    trail['lock_profit_at'] = 0.7
                    trail['last_update'] = datetime.now()
                    return new_sl, f"🔒 70% Profit Locked at {profit_pct:.2f}%"
            else:
                if new_sl < current_sl:
                    trail['current_sl'] = new_sl
                    trail['lock_profit_at'] = 0.7
                    trail['last_update'] = datetime.now()
                    return new_sl, f"🔒 70% Profit Locked at {profit_pct:.2f}%"
        
        return current_sl, "Holding current SL"
    
    def check_stop_hit(self, pair: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if trailing stop has been hit
        """
        
        if pair not in self.trailing_stops:
            return False, "No active trail"
        
        trail = self.trailing_stops[pair]
        current_sl = trail['current_sl']
        direction = trail['direction']
        
        if direction == "LONG":
            if current_price <= current_sl:
                del self.trailing_stops[pair]
                return True, f"Trailing SL Hit at ₹{current_price:.6f}"
        else:
            if current_price >= current_sl:
                del self.trailing_stops[pair]
                return True, f"Trailing SL Hit at ₹{current_price:.6f}"
        
        return False, "SL not hit"
    
    def remove_trail(self, pair: str):
        """
        Remove trailing stop (position closed)
        """
        
        if pair in self.trailing_stops:
            del self.trailing_stops[pair]
            print(f"✅ Trailing stop removed for {pair}")
    
    def get_current_sl(self, pair: str) -> Optional[float]:
        """
        Get current stop loss for a pair
        """
        
        if pair in self.trailing_stops:
            return self.trailing_stops[pair]['current_sl']
        
        return None


class PositionManager:
    """
    Manages all active positions and their lifecycle
    """
    
    def __init__(self):
        self.active_positions = {}
        self.trail_manager = TrailingStopManager()
        self.closed_positions = []
    
    def add_position(self, pair: str, signal: Dict):
        """
        Add new position to tracking
        """
        
        self.active_positions[pair] = {
            'signal': signal,
            'entry_time': datetime.now(),
            'entry_price': signal['entry'],
            'current_sl': signal['sl'],
            'status': 'ACTIVE',
            'partial_exits': []
        }
        
        # Initialize trailing stop
        self.trail_manager.initialize_trail(pair, signal)
        
        print(f"✅ Position added: {pair} {signal['direction']}")
    
    def update_position(self, pair: str, current_price: float,
                       current_candles: pd.DataFrame) -> Dict:
        """
        Update position with current market data
        
        Returns status and any actions needed
        """
        
        if pair not in self.active_positions:
            return {'action': 'NONE', 'reason': 'No position'}
        
        position = self.active_positions[pair]
        signal = position['signal']
        
        # Update trailing stop
        new_sl, update_msg = self.trail_manager.update_trailing_stop(
            pair, current_price, current_candles
        )
        
        if new_sl != position['current_sl']:
            position['current_sl'] = new_sl
            print(f"🔄 {pair}: {update_msg}")
            
            return {
                'action': 'UPDATE_SL',
                'new_sl': new_sl,
                'message': update_msg
            }
        
        # Check if stop hit
        hit, hit_msg = self.trail_manager.check_stop_hit(pair, current_price)
        
        if hit:
            return {
                'action': 'CLOSE',
                'reason': hit_msg,
                'price': current_price
            }
        
        return {'action': 'HOLD', 'reason': 'Monitoring'}
    
    def close_position(self, pair: str, exit_price: float, 
                      exit_reason: str, booking_pct: int = 100):
        """
        Close position (full or partial)
        """
        
        if pair not in self.active_positions:
            return
        
        position = self.active_positions[pair]
        signal = position['signal']
        
        # Calculate profit
        entry = signal['entry']
        direction = signal['direction']
        
        if direction == "LONG":
            profit_pct = ((exit_price - entry) / entry) * 100
        else:
            profit_pct = ((entry - exit_price) / entry) * 100
        
        # Record exit
        exit_record = {
            'pair': pair,
            'mode': signal['mode'],
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'profit_pct': profit_pct,
            'exit_reason': exit_reason,
            'booking_pct': booking_pct,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(),
            'hold_hours': (datetime.now() - position['entry_time']).seconds / 3600
        }
        
        position['partial_exits'].append(exit_record)
        
        # Full exit
        if booking_pct >= 100:
            self.closed_positions.append(exit_record)
            del self.active_positions[pair]
            self.trail_manager.remove_trail(pair)
            
            print(f"✅ Position closed: {pair} | {profit_pct:.2f}% | {exit_reason}")
        else:
            print(f"💰 Partial exit: {pair} | {booking_pct}% @ {profit_pct:.2f}%")
        
        return exit_record
    
    def get_active_positions(self) -> Dict:
        """
        Get all active positions
        """
        return self.active_positions
    
    def get_position_count(self) -> int:
        """
        Get count of active positions
        """
        return len(self.active_positions)
    
    def has_position(self, pair: str) -> bool:
        """
        Check if position exists for pair
        """
        return pair in self.active_positions