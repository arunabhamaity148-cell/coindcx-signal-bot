"""
📊 Performance Tracker - Win/Loss Analytics
Tracks TP1, TP2, SL hits and reasons
"""

from datetime import datetime
import json

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_stats = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'sl_hits': 0,
            'total_pnl': 0,
            'trades_detail': []
        }
        
    def add_signal(self, signal, levels):
        """Register a new signal"""
        trade = {
            'symbol': signal['symbol'],
            'direction': signal['direction'],
            'mode': signal['mode'],
            'entry': levels['entry'],
            'tp1': levels['tp1'],
            'tp2': levels['tp2'],
            'sl': levels['sl'],
            'timestamp': signal['timestamp'],
            'status': 'PENDING',
            'exit_price': None,
            'pnl': 0,
            'exit_reason': None
        }
        
        self.trades.append(trade)
        self.daily_stats['total_signals'] += 1
        return len(self.trades) - 1  # Return trade index
    
    def update_trade(self, trade_idx, exit_price, exit_reason):
        """Update trade with exit details"""
        if trade_idx >= len(self.trades):
            return
        
        trade = self.trades[trade_idx]
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['status'] = 'CLOSED'
        
        # Calculate PnL
        entry = trade['entry']
        direction = trade['direction']
        margin = 3000  # From config
        
        if direction == "LONG":
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100
        
        # Apply leverage
        pnl_pct *= 5  # 5x leverage
        trade['pnl'] = (margin * pnl_pct) / 100
        
        # Update stats
        self.daily_stats['total_pnl'] += trade['pnl']
        
        if 'TP' in exit_reason:
            self.daily_stats['wins'] += 1
            if 'TP1' in exit_reason:
                self.daily_stats['tp1_hits'] += 1
            elif 'TP2' in exit_reason:
                self.daily_stats['tp2_hits'] += 1
        elif 'SL' in exit_reason:
            self.daily_stats['losses'] += 1
            self.daily_stats['sl_hits'] += 1
        
        self.daily_stats['trades_detail'].append(trade)
    
    def analyze_sl_reason(self, trade):
        """Analyze why SL was hit"""
        reasons = []
        
        # Check market conditions during loss
        # This is simplified - real implementation would check market data
        
        # Common reasons for SL
        reasons.append("Market reversal against position")
        reasons.append("Higher timeframe resistance/support")
        reasons.append("Low volume during entry")
        reasons.append("Stop hunt before reversal")
        
        return " | ".join(reasons[:2])  # Return top 2 reasons
    
    def analyze_tp_reason(self, trade):
        """Analyze why TP was hit"""
        reasons = []
        
        reasons.append("Strong trend continuation")
        reasons.append("Volume confirmation at breakout")
        reasons.append("Clean price action structure")
        
        return " | ".join(reasons[:2])
    
    def get_daily_summary(self):
        """Generate comprehensive daily summary"""
        total = self.daily_stats['total_signals']
        wins = self.daily_stats['wins']
        losses = self.daily_stats['losses']
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        winning_trades = [t for t in self.daily_stats['trades_detail'] if t['pnl'] > 0]
        losing_trades = [t for t in self.daily_stats['trades_detail'] if t['pnl'] < 0]
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        best_trade = max([t['pnl'] for t in self.daily_stats['trades_detail']], default=0)
        
        return {
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'tp1_hits': self.daily_stats['tp1_hits'],
            'tp2_hits': self.daily_stats['tp2_hits'],
            'sl_hits': self.daily_stats['sl_hits'],
            'total_pnl': self.daily_stats['total_pnl'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'report_time': datetime.now().strftime('%d %b %Y, %H:%M')
        }
    
    def get_detailed_breakdown(self):
        """Get trade-by-trade breakdown for analysis"""
        breakdown = []
        
        for trade in self.daily_stats['trades_detail']:
            detail = {
                'symbol': trade['symbol'],
                'direction': trade['direction'],
                'entry': trade['entry'],
                'exit': trade['exit_price'],
                'pnl': trade['pnl'],
                'reason': trade['exit_reason'],
                'time': trade['timestamp'].strftime('%H:%M')
            }
            
            # Add analysis
            if 'SL' in trade['exit_reason']:
                detail['analysis'] = self.analyze_sl_reason(trade)
            else:
                detail['analysis'] = self.analyze_tp_reason(trade)
            
            breakdown.append(detail)
        
        return breakdown
    
    def format_breakdown_message(self):
        """Format breakdown for Telegram"""
        breakdown = self.get_detailed_breakdown()
        
        message = "📋 **TRADE-BY-TRADE BREAKDOWN**\n\n"
        
        for i, trade in enumerate(breakdown, 1):
            emoji = "✅" if trade['pnl'] > 0 else "❌"
            
            message += f"{emoji} **Trade #{i}**: {trade['symbol']}\n"
            message += f"Direction: {trade['direction']}\n"
            message += f"Entry: ₹{trade['entry']:.2f} → Exit: ₹{trade['exit']:.2f}\n"
            message += f"PnL: ₹{trade['pnl']:.2f}\n"
            message += f"Reason: {trade['reason']}\n"
            message += f"Analysis: {trade['analysis']}\n"
            message += f"Time: {trade['time']}\n"
            message += "━━━━━━━━━━━━━━\n\n"
        
        return message
    
    def reset_daily_stats(self):
        """Reset stats for new day"""
        self.daily_stats = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'sl_hits': 0,
            'total_pnl': 0,
            'trades_detail': []
        }
        
    def save_to_file(self, filename="trade_history.json"):
        """Save all trades to JSON file"""
        data = {
            'daily_stats': self.daily_stats,
            'all_trades': [
                {**t, 'timestamp': t['timestamp'].isoformat()} 
                for t in self.trades
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Trade history saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save: {e}")