"""
performance_tracker.py - Performance Analytics & Trade Logging
"""

import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class PerformanceTracker:
    """
    Track and analyze trading performance
    """
    
    def __init__(self):
        self.trades_log = []
        self.daily_stats = {}
        self.trade_results_file = 'trade_results.csv'
    
    def log_trade_result(self, signal: Dict, exit_price: float,
                        exit_reason: str, exit_time: datetime,
                        booking_pct: int = 100) -> Dict:
        """
        Log completed trade result
        """
        
        entry = signal['entry']
        direction = signal['direction']
        
        # Calculate profit
        if direction == "LONG":
            profit_pct = ((exit_price - entry) / entry) * 100
        else:
            profit_pct = ((entry - exit_price) / entry) * 100
        
        # Calculate R multiple
        sl_distance = abs(entry - signal['sl']) / entry * 100
        r_multiple = profit_pct / sl_distance if sl_distance > 0 else 0
        
        # Hold time
        entry_time = datetime.fromisoformat(signal['timestamp'])
        hold_hours = (exit_time - entry_time).seconds / 3600
        
        trade_record = {
            'pair': signal['pair'],
            'mode': signal['mode'],
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'signal_score': signal['score'],
            'profit_pct': round(profit_pct, 2),
            'r_multiple': round(r_multiple, 2),
            'hold_hours': round(hold_hours, 2),
            'exit_reason': exit_reason,
            'booking_pct': booking_pct,
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'win': profit_pct > 0,
            'rsi': signal['rsi'],
            'adx': signal['adx'],
            'volume_surge': signal['volume_surge'],
            'mtf_trend': signal['mtf_trend']
        }
        
        self.trades_log.append(trade_record)
        
        # Save to CSV
        self._save_to_csv(trade_record)
        
        print(f"\n📊 Trade Logged:")
        print(f"   {signal['pair']} {direction}")
        print(f"   Profit: {profit_pct:.2f}% ({r_multiple:.2f}R)")
        print(f"   Hold: {hold_hours:.1f}h")
        print(f"   Reason: {exit_reason}")
        
        return trade_record
    
    def _save_to_csv(self, record: Dict):
        """
        Save trade result to CSV
        """
        
        try:
            # Check if file exists
            file_exists = False
            try:
                with open(self.trade_results_file, 'r') as f:
                    file_exists = True
            except FileNotFoundError:
                pass
            
            # Write to CSV
            with open(self.trade_results_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(record)
        
        except Exception as e:
            print(f"⚠️ CSV logging error: {e}")
    
    def get_performance_stats(self, mode: Optional[str] = None,
                             days: int = 30) -> Optional[Dict]:
        """
        Calculate performance statistics
        """
        
        try:
            df = pd.read_csv(self.trade_results_file)
            
            # Filter by timeframe
            cutoff_date = datetime.now() - timedelta(days=days)
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df = df[df['exit_time'] >= cutoff_date]
            
            # Filter by mode if specified
            if mode:
                df = df[df['mode'] == mode]
            
            if len(df) == 0:
                return None
            
            # Calculate stats
            total_trades = len(df)
            wins = len(df[df['win'] == True])
            losses = len(df[df['win'] == False])
            
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            
            avg_win = df[df['win'] == True]['profit_pct'].mean() if wins > 0 else 0
            avg_loss = df[df['win'] == False]['profit_pct'].mean() if losses > 0 else 0
            
            avg_r_multiple = df['r_multiple'].mean()
            
            # Profit factor
            total_wins_pct = df[df['win'] == True]['profit_pct'].sum()
            total_loss_pct = abs(df[df['win'] == False]['profit_pct'].sum())
            
            profit_factor = total_wins_pct / total_loss_pct if total_loss_pct > 0 else 0
            
            # Average hold time
            avg_hold = df['hold_hours'].mean()
            
            # Best and worst
            best_trade = df.loc[df['profit_pct'].idxmax()] if len(df) > 0 else None
            worst_trade = df.loc[df['profit_pct'].idxmin()] if len(df) > 0 else None
            
            stats = {
                'period_days': days,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_r_multiple': round(avg_r_multiple, 2),
                'profit_factor': round(profit_factor, 2),
                'avg_hold_hours': round(avg_hold, 2),
                'best_trade': {
                    'pair': best_trade['pair'],
                    'profit': round(best_trade['profit_pct'], 2),
                    'date': best_trade['exit_time']
                } if best_trade is not None else None,
                'worst_trade': {
                    'pair': worst_trade['pair'],
                    'profit': round(worst_trade['profit_pct'], 2),
                    'date': worst_trade['exit_time']
                } if worst_trade is not None else None
            }
            
            return stats
        
        except FileNotFoundError:
            print("⚠️ No trade results file found")
            return None
        except Exception as e:
            print(f"⚠️ Stats calculation error: {e}")
            return None
    
    def analyze_by_mode(self) -> Dict:
        """
        Compare performance across different modes
        """
        
        try:
            df = pd.read_csv(self.trade_results_file)
            
            if len(df) == 0:
                return {}
            
            mode_stats = {}
            
            for mode in df['mode'].unique():
                mode_df = df[df['mode'] == mode]
                
                total = len(mode_df)
                wins = len(mode_df[mode_df['win'] == True])
                
                mode_stats[mode] = {
                    'total_trades': total,
                    'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                    'avg_profit': round(mode_df['profit_pct'].mean(), 2),
                    'avg_hold_hours': round(mode_df['hold_hours'].mean(), 1)
                }
            
            return mode_stats
        
        except Exception as e:
            print(f"⚠️ Mode analysis error: {e}")
            return {}
    
    def analyze_by_pair(self, top_n: int = 10) -> List[Dict]:
        """
        Find best performing pairs
        """
        
        try:
            df = pd.read_csv(self.trade_results_file)
            
            if len(df) == 0:
                return []
            
            pair_stats = df.groupby('pair').agg({
                'profit_pct': ['count', 'mean', 'sum'],
                'win': lambda x: (x.sum() / len(x) * 100)
            }).round(2)
            
            pair_stats.columns = ['trades', 'avg_profit', 'total_profit', 'win_rate']
            pair_stats = pair_stats.sort_values('total_profit', ascending=False)
            
            return pair_stats.head(top_n).to_dict('index')
        
        except Exception as e:
            print(f"⚠️ Pair analysis error: {e}")
            return []
    
    def generate_daily_report(self) -> str:
        """
        Generate formatted daily report
        """
        
        stats = self.get_performance_stats(days=1)
        
        if not stats:
            return "📊 No trades today"
        
        report = f"""
📊 DAILY PERFORMANCE REPORT
{'='*40}

Trades: {stats['total_trades']}
Wins: {stats['wins']} | Losses: {stats['losses']}
Win Rate: {stats['win_rate']}%

Average Win: {stats['avg_win']}%
Average Loss: {stats['avg_loss']}%
Profit Factor: {stats['profit_factor']}

Average R: {stats['avg_r_multiple']}R
Average Hold: {stats['avg_hold_hours']:.1f}h

Best Trade: {stats['best_trade']['pair']} ({stats['best_trade']['profit']}%)
Worst Trade: {stats['worst_trade']['pair']} ({stats['worst_trade']['profit']}%)
"""
        
        return report.strip()
    
    def generate_weekly_report(self) -> str:
        """
        Generate formatted weekly report
        """
        
        stats = self.get_performance_stats(days=7)
        mode_stats = self.analyze_by_mode()
        
        if not stats:
            return "📊 No trades this week"
        
        report = f"""
📊 WEEKLY PERFORMANCE REPORT
{'='*40}

OVERALL:
Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']}%
Profit Factor: {stats['profit_factor']}
Average R: {stats['avg_r_multiple']}R

BY MODE:
"""
        
        for mode, mode_stat in mode_stats.items():
            report += f"""
{mode}:
  Trades: {mode_stat['total_trades']}
  Win Rate: {mode_stat['win_rate']}%
  Avg Profit: {mode_stat['avg_profit']}%
  Avg Hold: {mode_stat['avg_hold_hours']}h
"""
        
        return report.strip()
    
    def suggest_optimizations(self) -> List[str]:
        """
        Analyze data and suggest improvements
        """
        
        suggestions = []
        
        try:
            df = pd.read_csv(self.trade_results_file)
            
            if len(df) < 20:
                return ["Need more trade data (minimum 20 trades)"]
            
            # Analyze by mode
            mode_stats = df.groupby('mode').agg({
                'win': lambda x: (x.sum() / len(x) * 100)
            }).round(1)
            
            for mode in mode_stats.index:
                win_rate = mode_stats.loc[mode, 'win']
                
                if win_rate < 45:
                    suggestions.append(
                        f"⚠️ {mode} mode underperforming (WR: {win_rate}%) - Consider disabling"
                    )
                elif win_rate > 70:
                    suggestions.append(
                        f"✅ {mode} mode excellent (WR: {win_rate}%) - Increase allocation"
                    )
            
            # Analyze exit timing
            avg_hold = df['hold_hours'].mean()
            avg_profit = df['profit_pct'].mean()
            
            if avg_hold > 10 and avg_profit < 1.0:
                suggestions.append(
                    f"⚠️ Long holds with low profits - Consider tighter exits"
                )
            
            # Analyze score correlation
            high_score = df[df['signal_score'] >= 80]
            low_score = df[df['signal_score'] < 70]
            
            if len(high_score) > 5 and len(low_score) > 5:
                high_wr = (high_score['win'].sum() / len(high_score)) * 100
                low_wr = (low_score['win'].sum() / len(low_score)) * 100
                
                if high_wr > low_wr + 15:
                    suggestions.append(
                        f"✅ High scores perform better ({high_wr:.1f}% vs {low_wr:.1f}%) - Raise minimum score"
                    )
            
            return suggestions
        
        except Exception as e:
            print(f"⚠️ Optimization analysis error: {e}")
            return ["Analysis unavailable"]