"""
Chart Visualization System
Creates beautiful charts for analysis:
- Price charts with indicators
- Equity curve
- Trade distribution
- Performance heatmaps
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")


class ChartPlotter:
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.output_dir = 'charts'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_price_chart(self, df, trades=None, indicators=True, save=True):
        """
        Plot price chart with indicators and trades
        
        Args:
            df: DataFrame with OHLCV data
            trades: List of trade dictionaries
            indicators: Whether to plot indicators
            save: Save to file
        """
        logger.info("📊 Creating price chart...")
        
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        
        # Plot candlesticks (simplified - using line chart)
        ax1.plot(df.index, df['close'], label='Price', color='cyan', linewidth=1)
        
        # Plot indicators
        if indicators and 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], label='SMA 20', color='orange', linewidth=1, alpha=0.7)
            ax1.plot(df.index, df['sma_50'], label='SMA 50', color='yellow', linewidth=1, alpha=0.7)
        
        # Plot trades
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Buy signals
            buys = trades_df[trades_df['side'] == 'LONG']
            if not buys.empty:
                ax1.scatter(buys['entry_time'], buys['entry_price'], 
                           color='lime', marker='^', s=100, label='Buy', zorder=5)
                ax1.scatter(buys['exit_time'], buys['exit_price'], 
                           color='red', marker='v', s=100, label='Sell', zorder=5)
            
            # Short signals
            shorts = trades_df[trades_df['side'] == 'SHORT']
            if not shorts.empty:
                ax1.scatter(shorts['entry_time'], shorts['entry_price'], 
                           color='red', marker='v', s=100, label='Short', zorder=5)
                ax1.scatter(shorts['exit_time'], shorts['exit_price'], 
                           color='lime', marker='^', s=100, label='Cover', zorder=5)
        
        ax1.set_title('Price Chart with Trades', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in df.columns:
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.plot(df.index, df['rsi'], color='purple', linewidth=1)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax3.fill_between(df.index, 30, 70, alpha=0.1)
            ax3.set_ylabel('RSI', fontsize=12)
            ax3.set_ylim([0, 100])
            ax3.grid(True, alpha=0.3)
        
        # MACD
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        # Simplified MACD (would calculate if not present)
        ax4.set_ylabel('MACD', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/price_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Chart saved: {filename}")
        
        plt.show()
        plt.close()
    
    def plot_equity_curve(self, equity_data, trades=None, save=True):
        """
        Plot equity curve over time
        
        Args:
            equity_data: List of dicts with timestamp and equity
            trades: List of trades for annotations
            save: Save to file
        """
        logger.info("📈 Creating equity curve...")
        
        equity_df = pd.DataFrame(equity_data)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])
        
        # Equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], 
                color='cyan', linewidth=2, label='Equity')
        ax1.fill_between(equity_df['timestamp'], equity_df['equity'], 
                         alpha=0.3, color='cyan')
        
        # Add initial capital line
        initial = equity_df['equity'].iloc[0]
        ax1.axhline(y=initial, color='yellow', linestyle='--', 
                   label=f'Initial Capital (₹{initial:,.0f})', alpha=0.7)
        
        # Mark trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Winning trades
            wins = trades_df[trades_df['pnl'] > 0]
            ax1.scatter(wins['exit_time'], [equity_df[equity_df['timestamp'] == t]['equity'].values[0] 
                       if len(equity_df[equity_df['timestamp'] == t]) > 0 else 0 
                       for t in wins['exit_time']], 
                       color='lime', marker='o', s=50, alpha=0.6, label='Wins')
            
            # Losing trades
            losses = trades_df[trades_df['pnl'] < 0]
            ax1.scatter(losses['exit_time'], [equity_df[equity_df['timestamp'] == t]['equity'].values[0] 
                       if len(equity_df[equity_df['timestamp'] == t]) > 0 else 0 
                       for t in losses['exit_time']], 
                       color='red', marker='o', s=50, alpha=0.6, label='Losses')
        
        ax1.set_title('Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Equity (₹)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                         color='red', alpha=0.5)
        ax2.plot(equity_df['timestamp'], equity_df['drawdown'], 
                color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/equity_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Chart saved: {filename}")
        
        plt.show()
        plt.close()
    
    def plot_trade_distribution(self, trades, save=True):
        """
        Plot trade distribution and analysis
        
        Args:
            trades: List of trade dictionaries
            save: Save to file
        """
        logger.info("📊 Creating trade distribution charts...")
        
        trades_df = pd.DataFrame(trades)
        
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. P&L Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(trades_df['pnl'], bins=30, color='cyan', edgecolor='white', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('P&L (₹)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss Pie Chart
        ax2 = fig.add_subplot(gs[0, 1])
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] < 0])
        colors = ['lime', 'red']
        ax2.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')
        
        # 3. Trade Duration
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(trades_df['duration_hours'], bins=20, color='purple', 
                edgecolor='white', alpha=0.7)
        ax3.set_title('Trade Duration', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. P&L by Side
        ax4 = fig.add_subplot(gs[1, 0])
        side_pnl = trades_df.groupby('side')['pnl'].sum()
        ax4.bar(side_pnl.index, side_pnl.values, color=['lime', 'red'], alpha=0.7)
        ax4.set_title('P&L by Side', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Total P&L (₹)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Exit Reasons
        ax5 = fig.add_subplot(gs[1, 1])
        exit_counts = trades_df['reason'].value_counts()
        ax5.bar(range(len(exit_counts)), exit_counts.values, 
               tick_label=exit_counts.index, color='orange', alpha=0.7)
        ax5.set_title('Exit Reasons', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Count')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        # 6. Return % Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(trades_df['pnl_percent'], bins=30, color='yellow', 
                edgecolor='white', alpha=0.7)
        ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax6.set_title('Return % Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Return %')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Trade Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/trade_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Chart saved: {filename}")
        
        plt.show()
        plt.close()
    
    def plot_performance_heatmap(self, trades, save=True):
        """
        Plot performance heatmap by hour and day
        
        Args:
            trades: List of trade dictionaries
            save: Save to file
        """
        logger.info("🔥 Creating performance heatmap...")
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        trades_df['day'] = trades_df['entry_time'].dt.day_name()
        
        # Create pivot table
        pivot = trades_df.pivot_table(values='pnl', index='day', 
                                       columns='hour', aggfunc='mean', fill_value=0)
        
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in days_order if d in pivot.index])
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                   center=0, linewidths=1, cbar_kws={'label': 'Avg P&L (₹)'},
                   ax=ax)
        
        ax.set_title('Performance Heatmap (Avg P&L by Day & Hour)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/performance_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Chart saved: {filename}")
        
        plt.show()
        plt.close()
    
    def create_dashboard(self, data_file, trades_file, equity_file, save=True):
        """
        Create comprehensive dashboard with all charts
        
        Args:
            data_file: Path to price data CSV
            trades_file: Path to trades CSV
            equity_file: Path to equity curve CSV
            save: Save charts
        """
        logger.info("🎨 Creating comprehensive dashboard...")
        
        # Load data
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        trades = pd.read_csv(trades_file).to_dict('records')
        equity = pd.read_csv(equity_file).to_dict('records')
        
        # Create all charts
        self.plot_price_chart(df, trades, save=save)
        self.plot_equity_curve(equity, trades, save=save)
        self.plot_trade_distribution(trades, save=save)
        self.plot_performance_heatmap(trades, save=save)
        
        logger.info("✅ Dashboard creation complete!")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Create Trading Charts')
    parser.add_argument('--type', type=str, required=True,
                        choices=['price', 'equity', 'distribution', 'heatmap', 'dashboard'],
                        help='Type of chart to create')
    parser.add_argument('--data', type=str, help='Path to price data CSV')
    parser.add_argument('--trades', type=str, help='Path to trades CSV')
    parser.add_argument('--equity', type=str, help='Path to equity curve CSV')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save charts')
    
    args = parser.parse_args()
    
    plotter = ChartPlotter()
    save = not args.no_save
    
    if args.type == 'dashboard':
        if not all([args.data, args.trades, args.equity]):
            logger.error("❌ Dashboard requires --data, --trades, and --equity files")
            return
        plotter.create_dashboard(args.data, args.trades, args.equity, save=save)
    
    elif args.type == 'price':
        if not args.data:
            logger.error("❌ Price chart requires --data file")
            return
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        trades = pd.read_csv(args.trades).to_dict('records') if args.trades else None
        plotter.plot_price_chart(df, trades, save=save)
    
    elif args.type == 'equity':
        if not args.equity:
            logger.error("❌ Equity curve requires --equity file")
            return
        equity = pd.read_csv(args.equity).to_dict('records')
        trades = pd.read_csv(args.trades).to_dict('records') if args.trades else None
        plotter.plot_equity_curve(equity, trades, save=save)
    
    elif args.type == 'distribution':
        if not args.trades:
            logger.error("❌ Distribution chart requires --trades file")
            return
        trades = pd.read_csv(args.trades).to_dict('records')
        plotter.plot_trade_distribution(trades, save=save)
    
    elif args.type == 'heatmap':
        if not args.trades:
            logger.error("❌ Heatmap requires --trades file")
            return
        trades = pd.read_csv(args.trades).to_dict('records')
        plotter.plot_performance_heatmap(trades, save=save)


if __name__ == "__main__":
    main()


# ==================== USAGE EXAMPLES ====================
"""
# Create price chart
python visualization/chart_plotter.py --type price --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --trades backtest_results/trades.csv

# Create equity curve
python visualization/chart_plotter.py --type equity --equity backtest_results/equity_curve.csv --trades backtest_results/trades.csv

# Create trade distribution
python visualization/chart_plotter.py --type distribution --trades backtest_results/trades.csv

# Create performance heatmap
python visualization/chart_plotter.py --type heatmap --trades backtest_results/trades.csv

# Create full dashboard
python visualization/chart_plotter.py --type dashboard --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --trades backtest_results/trades.csv --equity backtest_results/equity_curve.csv
""" 