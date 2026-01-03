"""
risk_manager.py - Position Sizing & Daily Risk Limits
"""

from datetime import datetime, date
from typing import Dict, Optional, Tuple
import pandas as pd
from config import config


class RiskManager:
    """
    Manages position sizing and enforces daily risk limits
    """
    
    def __init__(self, base_capital: float = 10000):
        self.base_capital = base_capital
        self.daily_pnl = 0.0
        self.daily_loss_limit = base_capital * (config.RISK_LIMITS['daily_loss_percent'] / 100)
        self.positions_today = 0
        self.last_reset_date = date.today()
        self.trade_history_today = []
        self.kill_switch_active = False
    
    def reset_daily_counters(self):
        """
        Reset daily tracking at midnight
        """
        
        today = date.today()
        
        if today != self.last_reset_date:
            print(f"\n🔄 Daily reset: {today}")
            print(f"   Yesterday P&L: ₹{self.daily_pnl:,.2f}")
            print(f"   Positions taken: {self.positions_today}")
            
            self.daily_pnl = 0.0
            self.positions_today = 0
            self.trade_history_today = []
            self.kill_switch_active = False
            self.last_reset_date = today
    
    def check_daily_limits(self) -> Tuple[bool, str]:
        """
        Check if daily limits allow new positions
        """
        
        self.reset_daily_counters()
        
        # Kill switch check
        if self.kill_switch_active:
            return False, "🛑 KILL SWITCH ACTIVE - Daily loss limit reached"
        
        # Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            self.kill_switch_active = True
            loss_pct = (abs(self.daily_pnl) / self.base_capital) * 100
            return False, f"🛑 Daily loss limit hit: {loss_pct:.1f}%"
        
        # Max positions per day
        max_positions = config.RISK_LIMITS['max_positions']
        if self.positions_today >= max_positions:
            return False, f"Daily position limit reached: {max_positions}"
        
        return True, "Limits OK"
    
    def calculate_position_size(self, signal: Dict, 
                               market_context: Dict) -> Dict:
        """
        Dynamic position sizing based on signal quality
        """
        
        score = signal['score']
        mode = signal['mode']
        
        # Base risk per trade
        risk_per_trade_pct = config.RISK_LIMITS['capital_per_trade']
        base_risk_amount = self.base_capital * (risk_per_trade_pct / 100)
        
        # Score-based multiplier
        if score >= 85:
            score_multiplier = 1.5      # High confidence
            risk_level = "HIGH"
        elif score >= 75:
            score_multiplier = 1.2
            risk_level = "MEDIUM"
        elif score >= 65:
            score_multiplier = 1.0
            risk_level = "NORMAL"
        else:
            score_multiplier = 0.7
            risk_level = "LOW"
        
        # Market context multiplier
        btc_trend = market_context.get('btc_trend', 'UNKNOWN')
        
        if btc_trend in ['STRONG_UP', 'STRONG_DOWN']:
            context_multiplier = 1.1
        elif btc_trend in ['MODERATE_UP', 'MODERATE_DOWN']:
            context_multiplier = 1.0
        else:
            context_multiplier = 0.7
        
        # Mode-based multiplier
        mode_multiplier = {
            'TREND': 1.3,
            'MID': 1.1,
            'QUICK': 0.9,
            'SCALP': 0.7
        }.get(mode, 1.0)
        
        # Recent performance multiplier
        recent_win_rate = self._calculate_recent_win_rate(mode)
        
        if recent_win_rate >= 0.70:
            streak_multiplier = 1.2
        elif recent_win_rate >= 0.55:
            streak_multiplier = 1.0
        else:
            streak_multiplier = 0.8
        
        # Adjusted risk
        adjusted_risk = (base_risk_amount *
                        score_multiplier *
                        context_multiplier *
                        mode_multiplier *
                        streak_multiplier)
        
        # Calculate position size
        entry = signal['entry']
        sl = signal['sl']
        sl_distance_pct = abs(entry - sl) / entry
        
        # Position size = Risk / SL Distance
        position_size_usdt = adjusted_risk / sl_distance_pct
        
        # Apply leverage
        leverage = signal['leverage']
        actual_margin = position_size_usdt / leverage
        
        # Safety cap - never use more than 20% of capital
        MAX_POSITION_PCT = 0.20
        max_position = self.base_capital * MAX_POSITION_PCT
        
        if actual_margin > max_position:
            actual_margin = max_position
            position_size_usdt = actual_margin * leverage
        
        return {
            'position_size_usdt': round(position_size_usdt, 2),
            'margin_required': round(actual_margin, 2),
            'risk_amount': round(adjusted_risk, 2),
            'risk_level': risk_level,
            'score_multiplier': score_multiplier,
            'context_multiplier': context_multiplier,
            'mode_multiplier': mode_multiplier,
            'streak_multiplier': streak_multiplier,
            'effective_leverage': leverage,
            'position_pct_of_capital': round(actual_margin / self.base_capital * 100, 2)
        }
    
    def record_trade_result(self, pair: str, mode: str, 
                           profit_pct: float, profit_amount: float):
        """
        Record trade result for tracking
        """
        
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'mode': mode,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'win': profit_pct > 0
        }
        
        self.trade_history_today.append(trade_record)
        self.daily_pnl += profit_amount
        
        # Check if kill switch should activate
        if self.daily_pnl <= -self.daily_loss_limit:
            self.kill_switch_active = True
            print(f"\n🛑 KILL SWITCH ACTIVATED")
            print(f"   Daily Loss: ₹{abs(self.daily_pnl):,.2f}")
            print(f"   Limit: ₹{self.daily_loss_limit:,.2f}")
            print(f"   Trading stopped for today")
    
    def record_position_opened(self):
        """
        Increment position counter
        """
        self.positions_today += 1
    
    def _calculate_recent_win_rate(self, mode: str, lookback: int = 10) -> float:
        """
        Calculate recent win rate for a mode
        """
        
        try:
            # Read from performance log
            df = pd.read_csv('trade_results.csv')
            
            # Filter by mode
            mode_trades = df[df['mode'] == mode].tail(lookback)
            
            if len(mode_trades) == 0:
                return 0.55  # Default assumption
            
            wins = len(mode_trades[mode_trades['win'] == True])
            return wins / len(mode_trades)
        
        except:
            return 0.55  # Default
    
    def get_daily_stats(self) -> Dict:
        """
        Get current day statistics
        """
        
        self.reset_daily_counters()
        
        wins = len([t for t in self.trade_history_today if t['win']])
        losses = len([t for t in self.trade_history_today if not t['win']])
        
        win_rate = (wins / len(self.trade_history_today) * 100) if self.trade_history_today else 0
        
        remaining_loss_buffer = self.daily_loss_limit + self.daily_pnl
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.base_capital) * 100,
            'positions_today': self.positions_today,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'kill_switch_active': self.kill_switch_active,
            'remaining_loss_buffer': remaining_loss_buffer,
            'loss_limit': self.daily_loss_limit
        }
    
    def check_correlation(self, pair: str, active_positions: Dict) -> Tuple[bool, str]:
        """
        Check if new position is correlated with existing ones
        Prevents over-exposure to same coin
        """
        
        coin = pair.replace('USDT', '')
        
        # Check if same coin in different mode
        for active_pair in active_positions:
            active_coin = active_pair.replace('USDT', '')
            
            if coin == active_coin:
                return False, f"Already have position in {active_coin}"
        
        # Check highly correlated pairs (simplified)
        CORRELATION_GROUPS = [
            ['BTC', 'ETH'],  # Major crypto correlation
            ['SOL', 'AVAX', 'ATOM'],  # L1 platforms
            ['MATIC', 'ARB', 'OP'],  # L2 solutions
        ]
        
        for group in CORRELATION_GROUPS:
            if coin in group:
                for active_pair in active_positions:
                    active_coin = active_pair.replace('USDT', '')
                    if active_coin in group and active_coin != coin:
                        return False, f"Correlated with {active_coin}"
        
        return True, "No correlation issues"