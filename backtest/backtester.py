 """
Complete Backtesting System
Tests trading strategy on historical data
Calculates comprehensive performance metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, initial_capital=10000, fee_rate=0.0005, slippage=0.0005):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            fee_rate: Trading fee (0.05%)
            slippage: Slippage percentage (0.05%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.open_position = None
        
        # Performance metrics
        self.metrics = {}
    
    def load_data(self, filepath):
        """Load historical data from CSV"""
        logger.info(f"📂 Loading data from: {filepath}")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        logger.info(f"✅ Loaded {len(df):,} candles")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals (simplified for backtest)
        In real implementation, this would use ML models + 45 logics
        """
        logger.info("🧠 Generating signals...")
        
        # Example: Simple strategy for testing
        # RSI-based signals
        df['signal'] = 1  # HOLD by default
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df.loc[df['rsi'] < 30, 'signal'] = 0  # BUY
        df.loc[df['rsi'] > 70, 'signal'] = 2  # SELL
        
        # Add confidence (random for demo, would be ML prediction)
        df['confidence'] = 0.75
        df.loc[df['signal'] != 1, 'confidence'] = np.random.uniform(0.70, 0.85, 
                                                                     size=(df['signal'] != 1).sum())
        
        logger.info(f"✅ Signals generated")
        logger.info(f"   BUY signals: {(df['signal'] == 0).sum()}")
        logger.info(f"   SELL signals: {(df['signal'] == 2).sum()}")
        logger.info(f"   HOLD signals: {(df['signal'] == 1).sum()}")
        
        return df
    
    def calculate_position_size(self, confidence, volatility=0.02):
        """Calculate position size based on risk"""
        base_size = self.capital * 0.15  # 15% per trade
        
        # Adjust for confidence
        if confidence > 0.80:
            base_size *= 1.2
        elif confidence < 0.65:
            base_size *= 0.8
        
        return min(base_size, self.capital * 0.50)
    
    def calculate_leverage(self, volatility):
        """Calculate leverage based on volatility"""
        if volatility > 0.03:
            return 3
        elif volatility > 0.02:
            return 5
        else:
            return 7
    
    def open_position(self, row, side, size, leverage):
        """Open a position"""
        entry_price = row['close']
        
        # Apply slippage
        if side == 'LONG':
            entry_price *= (1 + self.slippage)
        else:
            entry_price *= (1 - self.slippage)
        
        # Calculate stop loss and take profit
        if side == 'LONG':
            stop_loss = entry_price * 0.98  # 2% SL
            take_profit = entry_price * 1.03  # 3% TP
        else:
            stop_loss = entry_price * 1.02
            take_profit = entry_price * 0.97
        
        # Calculate fees
        position_value = size * leverage
        fee = position_value * self.fee_rate
        
        self.open_position = {
            'entry_time': row.name,
            'entry_price': entry_price,
            'side': side,
            'size': size,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'fee_open': fee,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        # Deduct fee from capital
        self.capital -= fee
        
        logger.debug(f"📈 Opened {side} position @ {entry_price:.2f}")
    
    def close_position(self, row, reason='manual'):
        """Close the open position"""
        if not self.open_position:
            return
        
        exit_price = row['close']
        
        # Apply slippage
        if self.open_position['side'] == 'LONG':
            exit_price *= (1 - self.slippage)
        else:
            exit_price *= (1 + self.slippage)
        
        # Calculate P&L
        entry = self.open_position['entry_price']
        size = self.open_position['size']
        leverage = self.open_position['leverage']
        
        if self.open_position['side'] == 'LONG':
            pnl_percent = (exit_price - entry) / entry
        else:
            pnl_percent = (entry - exit_price) / entry
        
        pnl = pnl_percent * size * leverage
        
        # Exit fee
        position_value = size * leverage
        fee_close = position_value * self.fee_rate
        pnl -= fee_close
        
        # Update capital
        self.capital += pnl
        
        # Record trade
        trade = {
            **self.open_position,
            'exit_time': row.name,
            'exit_price': exit_price,
            'fee_close': fee_close,
            'pnl': pnl,
            'pnl_percent': pnl / size * 100,
            'return': pnl / size,
            'duration_hours': (row.name - self.open_position['entry_time']).total_seconds() / 3600,
            'reason': reason
        }
        
        self.trades.append(trade)
        self.open_position = None
        
        logger.debug(f"📉 Closed position @ {exit_price:.2f} | P&L: ₹{pnl:.2f} | Reason: {reason}")
    
    def update_position(self, row):
        """Update open position (check TP/SL)"""
        if not self.open_position:
            return
        
        current_price = row['close']
        high = row['high']
        low = row['low']
        
        # Update highest/lowest
        self.open_position['highest_price'] = max(self.open_position['highest_price'], high)
        self.open_position['lowest_price'] = min(self.open_position['lowest_price'], low)
        
        # Check stop loss
        if self.open_position['side'] == 'LONG':
            if low <= self.open_position['stop_loss']:
                self.close_position(row, 'stop_loss')
                return
            # Check take profit
            if high >= self.open_position['take_profit']:
                self.close_position(row, 'take_profit')
                return
        else:  # SHORT
            if high >= self.open_position['stop_loss']:
                self.close_position(row, 'stop_loss')
                return
            if low <= self.open_position['take_profit']:
                self.close_position(row, 'take_profit')
                return
        
        # Check time limit (4 hours)
        time_open = (row.name - self.open_position['entry_time']).total_seconds() / 3600
        if time_open >= 4:
            self.close_position(row, 'time_limit')
            return
    
    def run_backtest(self, df):
        """Run backtest on historical data"""
        logger.info("="*80)
        logger.info("🚀 STARTING BACKTEST")
        logger.info("="*80)
        logger.info(f"Initial capital: ₹{self.initial_capital:,.2f}")
        logger.info(f"Fee rate: {self.fee_rate:.2%}")
        logger.info(f"Slippage: {self.slippage:.2%}")
        logger.info("="*80)
        
        # Generate signals
        df = self.generate_signals(df)
        
        # Main backtest loop
        for idx, row in df.iterrows():
            # Update existing position
            if self.open_position:
                self.update_position(row)
            
            # Check for new signals (only if no open position)
            if not self.open_position:
                signal = row['signal']
                confidence = row['confidence']
                
                if signal in [0, 2] and confidence >= 0.70:
                    # Calculate position parameters
                    volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                    size = self.calculate_position_size(confidence, volatility)
                    leverage = self.calculate_leverage(volatility)
                    
                    # Open position
                    side = 'LONG' if signal == 0 else 'SHORT'
                    self.open_position(row, side, size, leverage)
            
            # Record equity
            current_equity = self.capital
            if self.open_position:
                # Add unrealized P&L
                entry = self.open_position['entry_price']
                size = self.open_position['size']
                leverage = self.open_position['leverage']
                
                if self.open_position['side'] == 'LONG':
                    unrealized = (row['close'] - entry) / entry * size * leverage
                else:
                    unrealized = (entry - row['close']) / entry * size * leverage
                
                current_equity += unrealized
            
            self.equity_curve.append({
                'timestamp': idx,
                'equity': current_equity
            })
        
        # Close any remaining position
        if self.open_position:
            self.close_position(df.iloc[-1], 'backtest_end')
        
        logger.info("✅ Backtest completed")
        logger.info(f"Total trades: {len(self.trades)}")
        logger.info(f"Final capital: ₹{self.capital:,.2f}")
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        logger.info("\n📊 CALCULATING PERFORMANCE METRICS")
        logger.info("="*80)
        
        if not self.trades:
            logger.warning("No trades executed!")
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        wins = len(winning_trades)
        losses = len(losing_trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if wins > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if losses > 0 else 0
        largest_win = winning_trades['pnl'].max() if wins > 0 else 0
        largest_loss = losing_trades['pnl'].min() if losses > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if wins > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if losses > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Return metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (simplified)
        returns = trades_df['return'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Recovery factor
        recovery_factor = total_pnl / abs(max_drawdown * self.initial_capital) if max_drawdown != 0 else 0
        
        # Trade duration
        avg_duration = trades_df['duration_hours'].mean()
        
        # Exit reasons
        exit_reasons = trades_df['reason'].value_counts()
        
        self.metrics = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'expectancy': expectancy,
            'recovery_factor': recovery_factor,
            'avg_duration_hours': avg_duration,
            'exit_reasons': exit_reasons.to_dict(),
            'final_capital': self.capital,
            'initial_capital': self.initial_capital
        }
        
        return self.metrics
    
    def print_results(self):
        """Print backtest results"""
        if not self.metrics:
            self.calculate_metrics()
        
        m = self.metrics
        
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        print(f"\n💰 CAPITAL")
        print(f"   Initial: ₹{m['initial_capital']:,.2f}")
        print(f"   Final:   ₹{m['final_capital']:,.2f}")
        print(f"   P&L:     ₹{m['total_pnl']:,.2f}")
        print(f"   Return:  {m['total_return']:.2%}")
        
        print(f"\n📈 TRADE STATISTICS")
        print(f"   Total trades:    {m['total_trades']}")
        print(f"   Winning trades:  {m['wins']} ({m['wins']/m['total_trades']:.1%})")
        print(f"   Losing trades:   {m['losses']} ({m['losses']/m['total_trades']:.1%})")
        print(f"   Win rate:        {m['win_rate']:.2%}")
        
        print(f"\n💵 P&L BREAKDOWN")
        print(f"   Average win:     ₹{m['avg_win']:,.2f}")
        print(f"   Average loss:    ₹{m['avg_loss']:,.2f}")
        print(f"   Largest win:     ₹{m['largest_win']:,.2f}")
        print(f"   Largest loss:    ₹{m['largest_loss']:,.2f}")
        print(f"   Profit factor:   {m['profit_factor']:.2f}")
        print(f"   Expectancy:      ₹{m['expectancy']:,.2f}")
        
        print(f"\n📉 RISK METRICS")
        print(f"   Max drawdown:    {m['max_drawdown']:.2%}")
        print(f"   Sharpe ratio:    {m['sharpe_ratio']:.2f}")
        print(f"   Recovery factor: {m['recovery_factor']:.2f}")
        
        print(f"\n⏱️ TRADE DURATION")
        print(f"   Average:         {m['avg_duration_hours']:.1f} hours")
        
        print(f"\n📋 EXIT REASONS")
        for reason, count in m['exit_reasons'].items():
            print(f"   {reason:15} {count:3} ({count/m['total_trades']:.1%})")
        
        print("\n" + "="*80)
        
        # Performance grade
        grade = self._calculate_grade()
        print(f"\n🏆 PERFORMANCE GRADE: {grade}")
        print("="*80)
    
    def _calculate_grade(self):
        """Calculate performance grade"""
        score = 0
        
        # Win rate (max 25 points)
        if self.metrics['win_rate'] >= 0.70:
            score += 25
        elif self.metrics['win_rate'] >= 0.65:
            score += 20
        elif self.metrics['win_rate'] >= 0.60:
            score += 15
        else:
            score += 10
        
        # Profit factor (max 25 points)
        if self.metrics['profit_factor'] >= 2.0:
            score += 25
        elif self.metrics['profit_factor'] >= 1.5:
            score += 20
        elif self.metrics['profit_factor'] >= 1.2:
            score += 15
        else:
            score += 10
        
        # Return (max 25 points)
        if self.metrics['total_return'] >= 2.0:
            score += 25
        elif self.metrics['total_return'] >= 1.0:
            score += 20
        elif self.metrics['total_return'] >= 0.5:
            score += 15
        else:
            score += 10
        
        # Max drawdown (max 25 points)
        if self.metrics['max_drawdown'] >= -0.10:
            score += 25
        elif self.metrics['max_drawdown'] >= -0.15:
            score += 20
        elif self.metrics['max_drawdown'] >= -0.20:
            score += 15
        else:
            score += 10
        
        # Grade
        if score >= 90:
            return "A+ (EXCELLENT)"
        elif score >= 80:
            return "A (VERY GOOD)"
        elif score >= 70:
            return "B (GOOD)"
        elif score >= 60:
            return "C (ACCEPTABLE)"
        else:
            return "D (NEEDS IMPROVEMENT)"
    
    def save_results(self, output_dir='backtest_results'):
        """Save backtest results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(f'{output_dir}/metrics.json', 'w') as f:
            # Convert exit_reasons to serializable format
            metrics_copy = self.metrics.copy()
            json.dump(metrics_copy, f, indent=4, default=str)
        
        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(f'{output_dir}/trades.csv', index=False)
        
        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(f'{output_dir}/equity_curve.csv', index=False)
        
        logger.info(f"✅ Results saved to: {output_dir}/")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Backtest Trading Strategy')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data CSV')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--fee', type=float, default=0.0005,
                        help='Trading fee rate (default: 0.0005)')
    parser.add_argument('--save', action='store_true',
                        help='Save results to files')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=args.capital,
        fee_rate=args.fee
    )
    
    # Load data
    df = backtester.load_data(args.data)
    
    # Run backtest
    backtester.run_backtest(df)
    
    # Calculate and print results
    backtester.calculate_metrics()
    backtester.print_results()
    
    # Save results
    if args.save:
        backtester.save_results()


if __name__ == "__main__":
    main()


# ==================== USAGE EXAMPLES ====================
"""
# Basic backtest
python backtest/backtester.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv

# Custom capital
python backtest/backtester.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --capital 50000

# Save results
python backtest/backtester.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --save
"""