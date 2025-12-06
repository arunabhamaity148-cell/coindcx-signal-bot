# ================================================================
# backtesting.py - Complete Backtesting Framework
# ================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import logging

log = logging.getLogger("backtesting")


class Trade:
    """Represents a single backtest trade"""
    
    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        signal_data: dict
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size
        self.signal_data = signal_data
        
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.duration_hours = 0
        self.exit_reason = ""
        
        # Risk parameters from signal
        self.stop_loss = signal_data.get("stop_loss")
        self.take_profit = signal_data.get("take_profit")
        self.max_duration_hours = signal_data.get("max_duration", 24)
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate P&L
        if self.side == "long":
            self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100
        
        self.pnl = (self.pnl_pct / 100) * (self.size * self.entry_price)
        self.duration_hours = (exit_time - self.entry_time).total_seconds() / 3600
    
    def check_exit(self, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if trade should be exited"""
        
        # Stop loss
        if self.stop_loss:
            if (self.side == "long" and current_price <= self.stop_loss) or \
               (self.side == "short" and current_price >= self.stop_loss):
                return "STOP_LOSS"
        
        # Take profit
        if self.take_profit:
            if (self.side == "long" and current_price >= self.take_profit) or \
               (self.side == "short" and current_price <= self.take_profit):
                return "TAKE_PROFIT"
        
        # Max duration
        hours_open = (current_time - self.entry_time).total_seconds() / 3600
        if hours_open >= self.max_duration_hours:
            return "MAX_DURATION"
        
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "duration_hours": self.duration_hours,
            "exit_reason": self.exit_reason,
            "size": self.size,
            "confidence": self.signal_data.get("confidence", 0)
        }


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 2.0,
        max_open_trades: int = 3,
        commission: float = 0.1  # 0.1% per trade
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_open_trades = max_open_trades
        self.commission = commission
        
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve = []
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        risk_percent: float
    ) -> float:
        """
        Calculate position size based on risk management
        
        Formula: Position Size = (Account * Risk%) / (Entry - Stop Loss)
        """
        
        if not stop_loss or stop_loss <= 0:
            # Default to 1% of capital if no stop loss
            return (account_balance * 0.01) / entry_price
        
        risk_amount = account_balance * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size_usd = risk_amount / (price_risk / entry_price)
        position_size_coins = position_size_usd / entry_price
        
        # Max 10% of capital per trade
        max_size = (account_balance * 0.10) / entry_price
        
        return min(position_size_coins, max_size)
    
    async def backtest(
        self,
        signals: List[dict],
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        Run backtest on historical signals
        
        Args:
            signals: List of signal dicts with timestamp, symbol, side, etc.
            price_data: Dict of {symbol: DataFrame with OHLC data}
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results dict
        """
        
        log.info(f"Starting backtest: {start_date} to {end_date}")
        log.info(f"Signals: {len(signals)}, Initial capital: ${self.initial_capital}")
        
        # Filter signals by date
        valid_signals = [
            s for s in signals
            if start_date <= s["timestamp"] <= end_date
        ]
        
        log.info(f"Valid signals in date range: {len(valid_signals)}")
        
        # Sort signals by timestamp
        valid_signals.sort(key=lambda x: x["timestamp"])
        
        # Process each signal
        for signal in valid_signals:
            await self._process_signal(signal, price_data)
        
        # Close any remaining open trades at end date
        for trade in self.open_trades[:]:
            symbol = trade.symbol
            if symbol in price_data:
                df = price_data[symbol]
                end_price = df[df.index <= end_date]["close"].iloc[-1]
                self._close_trade(trade, end_price, end_date, "BACKTEST_END")
        
        # Calculate statistics
        results = self._calculate_statistics()
        
        log.info(f"Backtest complete: {len(self.closed_trades)} trades")
        log.info(f"Final capital: ${self.current_capital:.2f}")
        log.info(f"Total return: {results['total_return']:.2f}%")
        
        return results
    
    async def _process_signal(self, signal: dict, price_data: Dict[str, pd.DataFrame]):
        """Process a single signal"""
        
        symbol = signal["symbol"]
        timestamp = signal["timestamp"]
        
        # Check if we can take more trades
        if len(self.open_trades) >= self.max_open_trades:
            return
        
        # Get price data
        if symbol not in price_data:
            return
        
        df = price_data[symbol]
        price_at_signal = df[df.index >= timestamp]["open"].iloc[0]
        
        # Calculate position size
        stop_loss = signal.get("stop_loss", price_at_signal * 0.98)
        position_size = self.calculate_position_size(
            price_at_signal,
            stop_loss,
            self.current_capital,
            self.risk_per_trade
        )
        
        if position_size <= 0:
            return
        
        # Calculate commission
        trade_value = position_size * price_at_signal
        commission_cost = trade_value * (self.commission / 100)
        
        # Check if we have enough capital
        if trade_value + commission_cost > self.current_capital:
            return
        
        # Open trade
        trade = Trade(
            symbol=symbol,
            side=signal["side"],
            entry_price=price_at_signal,
            entry_time=timestamp,
            size=position_size,
            signal_data=signal
        )
        
        self.open_trades.append(trade)
        self.current_capital -= commission_cost
        
        # Update equity curve
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": self.current_capital,
            "open_trades": len(self.open_trades)
        })
        
        # Check for exits in future candles
        future_data = df[df.index > timestamp]
        
        for idx, row in future_data.iterrows():
            exit_reason = trade.check_exit(row["close"], idx)
            
            if exit_reason:
                self._close_trade(trade, row["close"], idx, exit_reason)
                break
    
    def _close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """Close a trade"""
        
        trade.close(exit_price, exit_time, reason)
        
        # Calculate commission
        trade_value = trade.size * exit_price
        commission_cost = trade_value * (self.commission / 100)
        
        # Update capital
        self.current_capital += trade.pnl - commission_cost
        
        # Move to closed trades
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append({
            "timestamp": exit_time,
            "equity": self.current_capital,
            "open_trades": len(self.open_trades)
        })
    
    def _calculate_statistics(self) -> dict:
        """Calculate backtest statistics"""
        
        if not self.closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_return": 0,
                "sharpe_ratio": 0
            }
        
        trades_df = pd.DataFrame([t.to_dict() for t in self.closed_trades])
        
        # Basic stats
        total_trades = len(self.closed_trades)
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100
        
        # P&L stats
        total_pnl = trades_df["pnl"].sum()
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = trades_df["pnl_pct"].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series([e["equity"] for e in self.equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df["pnl"].idxmax()] if len(trades_df) > 0 else None
        worst_trade = trades_df.loc[trades_df["pnl"].idxmin()] if len(trades_df) > 0 else None
        
        # Average duration
        avg_duration = trades_df["duration_hours"].mean()
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return": round(total_return, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "best_trade_pnl": round(best_trade["pnl"], 2) if best_trade is not None else 0,
            "worst_trade_pnl": round(worst_trade["pnl"], 2) if worst_trade is not None else 0,
            "avg_duration_hours": round(avg_duration, 2),
            "initial_capital": self.initial_capital,
            "final_capital": round(self.current_capital, 2),
            "equity_curve": self.equity_curve,
            "trades": [t.to_dict() for t in self.closed_trades]
        }


class RiskManager:
    """Advanced risk management"""
    
    @staticmethod
    def calculate_position_size(
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Kelly Criterion-inspired position sizing
        
        Args:
            account_balance: Total capital
            risk_percent: Risk per trade (1-5%)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in base currency
        """
        
        if not stop_loss_price or stop_loss_price <= 0:
            # Conservative fallback
            return (account_balance * 0.01) / entry_price
        
        # Risk amount in dollars
        risk_amount = account_balance * (risk_percent / 100)
        
        # Distance to stop loss
        stop_distance = abs(entry_price - stop_loss_price)
        stop_distance_pct = stop_distance / entry_price
        
        if stop_distance_pct == 0:
            return 0
        
        # Position size
        position_size_usd = risk_amount / stop_distance_pct
        position_size = position_size_usd / entry_price
        
        # Max 20% of capital per trade
        max_position_usd = account_balance * 0.20
        max_position = max_position_usd / entry_price
        
        return min(position_size, max_position)
    
    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        
        Formula: f* = (bp - q) / b
        where:
            f* = optimal bet fraction
            b = odds (avg_win / avg_loss)
            p = probability of winning
            q = probability of losing (1-p)
        """
        
        if avg_loss == 0:
            return 0
        
        p = win_rate / 100
        q = 1 - p
        b = avg_win / avg_loss
        
        kelly = (b * p - q) / b
        
        # Use half-kelly for safety
        return max(0, min(kelly / 2, 0.25))  # Cap at 25%
    
    @staticmethod
    def check_max_drawdown(
        current_capital: float,
        peak_capital: float,
        max_dd_percent: float = 20.0
    ) -> dict:
        """
        Check if max drawdown exceeded
        
        Returns:
            {
                "current_dd": float,
                "exceeded": bool,
                "should_stop": bool
            }
        """
        
        if peak_capital == 0:
            return {"current_dd": 0, "exceeded": False, "should_stop": False}
        
        current_dd = ((peak_capital - current_capital) / peak_capital) * 100
        exceeded = current_dd > max_dd_percent
        
        return {
            "current_dd": round(current_dd, 2),
            "exceeded": exceeded,
            "should_stop": exceeded,
            "capital_to_recover": peak_capital - current_capital if exceeded else 0
        }


# Convenience function
async def quick_backtest(
    signals: List[dict],
    price_data: Dict[str, pd.DataFrame],
    initial_capital: float = 10000
) -> dict:
    """
    Quick backtest with default parameters
    
    Usage:
        results = await quick_backtest(signals, price_data)
        print(f"Win rate: {results['win_rate']}%")
    """
    
    engine = BacktestEngine(initial_capital=initial_capital)
    
    # Determine date range from signals
    start_date = min(s["timestamp"] for s in signals)
    end_date = max(s["timestamp"] for s in signals)
    
    results = await engine.backtest(signals, price_data, start_date, end_date)
    
    return results