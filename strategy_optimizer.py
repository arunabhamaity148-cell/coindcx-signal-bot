# ================================================================
# strategy_optimizer.py - Strategy Parameter Optimization
# ================================================================

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import asyncio
import logging
from itertools import product
from backtesting import BacktestEngine

log = logging.getLogger("optimizer")


class StrategyOptimizer:
    """Optimize strategy parameters using grid search and walk-forward analysis"""
    
    def __init__(self):
        self.best_params = {}
        self.optimization_results = []
    
    async def grid_search(
        self,
        signals: List[dict],
        price_data: Dict[str, pd.DataFrame],
        param_grid: dict,
        metric: str = "sharpe_ratio",
        initial_capital: float = 10000
    ) -> dict:
        """
        Grid search optimization
        
        Args:
            signals: Historical signals
            price_data: Price data for backtesting
            param_grid: Dict of parameters to test
                Example: {
                    "min_confidence": [60, 65, 70, 75],
                    "risk_per_trade": [1.0, 1.5, 2.0],
                    "max_open_trades": [2, 3, 4]
                }
            metric: Optimization metric (sharpe_ratio, profit_factor, win_rate)
            initial_capital: Starting capital
            
        Returns:
            Best parameters and results
        """
        
        log.info("Starting grid search optimization...")
        log.info(f"Parameter grid: {param_grid}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        log.info(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            log.info(f"Testing {i+1}/{len(combinations)}: {params}")
            
            # Filter signals based on min_confidence if present
            filtered_signals = signals
            if "min_confidence" in params:
                min_conf = params["min_confidence"]
                filtered_signals = [
                    s for s in signals 
                    if s.get("confidence", 0) >= min_conf
                ]
            
            # Run backtest with these parameters
            engine = BacktestEngine(
                initial_capital=initial_capital,
                risk_per_trade=params.get("risk_per_trade", 2.0),
                max_open_trades=params.get("max_open_trades", 3)
            )
            
            start_date = min(s["timestamp"] for s in filtered_signals)
            end_date = max(s["timestamp"] for s in filtered_signals)
            
            backtest_results = await engine.backtest(
                filtered_signals,
                price_data,
                start_date,
                end_date
            )
            
            # Store results
            result = {
                "params": params,
                "total_trades": backtest_results["total_trades"],
                "win_rate": backtest_results["win_rate"],
                "total_return": backtest_results["total_return"],
                "profit_factor": backtest_results["profit_factor"],
                "sharpe_ratio": backtest_results["sharpe_ratio"],
                "max_drawdown": backtest_results["max_drawdown"],
                "final_capital": backtest_results["final_capital"]
            }
            
            results.append(result)
        
        # Sort by optimization metric
        results.sort(key=lambda x: x[metric], reverse=True)
        
        best = results[0]
        self.best_params = best["params"]
        self.optimization_results = results
        
        log.info(f"Optimization complete! Best {metric}: {best[metric]:.2f}")
        log.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_metric": best[metric],
            "all_results": results
        }
    
    async def walk_forward_analysis(
        self,
        signals: List[dict],
        price_data: Dict[str, pd.DataFrame],
        param_grid: dict,
        train_periods: int = 3,
        test_periods: int = 1,
        metric: str = "sharpe_ratio"
    ) -> dict:
        """
        Walk-forward optimization
        
        Divides data into training and testing periods,
        optimizes on training data, validates on test data.
        
        Args:
            signals: Historical signals
            price_data: Price data
            param_grid: Parameters to optimize
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            metric: Optimization metric
            
        Returns:
            Walk-forward results
        """
        
        log.info("Starting walk-forward analysis...")
        
        # Sort signals by date
        sorted_signals = sorted(signals, key=lambda x: x["timestamp"])
        
        # Calculate period size
        total_periods = train_periods + test_periods
        period_size = len(sorted_signals) // total_periods
        
        log.info(f"Total signals: {len(sorted_signals)}")
        log.info(f"Period size: {period_size} signals")
        
        walk_forward_results = []
        
        # Rolling window
        for i in range(0, len(sorted_signals) - period_size * total_periods, period_size):
            # Training data
            train_start = i
            train_end = i + (period_size * train_periods)
            train_signals = sorted_signals[train_start:train_end]
            
            # Test data
            test_start = train_end
            test_end = test_start + (period_size * test_periods)
            test_signals = sorted_signals[test_start:test_end]
            
            if len(test_signals) < period_size // 2:
                break
            
            log.info(f"\nWalk-forward window {len(walk_forward_results) + 1}")
            log.info(f"Training: {len(train_signals)} signals")
            log.info(f"Testing: {len(test_signals)} signals")
            
            # Optimize on training data
            train_opt = await self.grid_search(
                train_signals,
                price_data,
                param_grid,
                metric=metric
            )
            
            # Test on out-of-sample data
            best_params = train_opt["best_params"]
            
            # Filter test signals
            filtered_test = test_signals
            if "min_confidence" in best_params:
                min_conf = best_params["min_confidence"]
                filtered_test = [
                    s for s in test_signals
                    if s.get("confidence", 0) >= min_conf
                ]
            
            # Run test backtest
            test_engine = BacktestEngine(
                risk_per_trade=best_params.get("risk_per_trade", 2.0),
                max_open_trades=best_params.get("max_open_trades", 3)
            )
            
            if filtered_test:
                test_start_date = min(s["timestamp"] for s in filtered_test)
                test_end_date = max(s["timestamp"] for s in filtered_test)
                
                test_results = await test_engine.backtest(
                    filtered_test,
                    price_data,
                    test_start_date,
                    test_end_date
                )
                
                walk_forward_results.append({
                    "train_best_params": best_params,
                    "train_metric": train_opt["best_metric"],
                    "test_return": test_results["total_return"],
                    "test_win_rate": test_results["win_rate"],
                    "test_sharpe": test_results["sharpe_ratio"],
                    "test_trades": test_results["total_trades"]
                })
        
        # Aggregate results
        avg_test_return = np.mean([r["test_return"] for r in walk_forward_results])
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in walk_forward_results])
        avg_test_win_rate = np.mean([r["test_win_rate"] for r in walk_forward_results])
        
        log.info("\nWalk-forward analysis complete!")
        log.info(f"Average out-of-sample return: {avg_test_return:.2f}%")
        log.info(f"Average out-of-sample Sharpe: {avg_test_sharpe:.2f}")
        log.info(f"Average out-of-sample win rate: {avg_test_win_rate:.2f}%")
        
        return {
            "walk_forward_results": walk_forward_results,
            "avg_test_return": avg_test_return,
            "avg_test_sharpe": avg_test_sharpe,
            "avg_test_win_rate": avg_test_win_rate,
            "stability_score": np.std([r["test_return"] for r in walk_forward_results])
        }
    
    async def sensitivity_analysis(
        self,
        signals: List[dict],
        price_data: Dict[str, pd.DataFrame],
        base_params: dict,
        vary_param: str,
        param_range: List
    ) -> dict:
        """
        Analyze sensitivity to a single parameter
        
        Args:
            signals: Historical signals
            price_data: Price data
            base_params: Base parameter set
            vary_param: Parameter to vary
            param_range: Range of values to test
            
        Returns:
            Sensitivity analysis results
        """
        
        log.info(f"Sensitivity analysis for {vary_param}")
        log.info(f"Range: {param_range}")
        
        results = []
        
        for value in param_range:
            params = base_params.copy()
            params[vary_param] = value
            
            # Filter signals if needed
            filtered_signals = signals
            if vary_param == "min_confidence":
                filtered_signals = [
                    s for s in signals
                    if s.get("confidence", 0) >= value
                ]
            
            # Run backtest
            engine = BacktestEngine(
                risk_per_trade=params.get("risk_per_trade", 2.0),
                max_open_trades=params.get("max_open_trades", 3)
            )
            
            if filtered_signals:
                start_date = min(s["timestamp"] for s in filtered_signals)
                end_date = max(s["timestamp"] for s in filtered_signals)
                
                backtest_results = await engine.backtest(
                    filtered_signals,
                    price_data,
                    start_date,
                    end_date
                )
                
                results.append({
                    "param_value": value,
                    "total_return": backtest_results["total_return"],
                    "sharpe_ratio": backtest_results["sharpe_ratio"],
                    "win_rate": backtest_results["win_rate"],
                    "total_trades": backtest_results["total_trades"]
                })
        
        return {
            "parameter": vary_param,
            "results": results,
            "optimal_value": max(results, key=lambda x: x["sharpe_ratio"])["param_value"]
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy robustness"""
    
    @staticmethod
    async def simulate(
        trades: List[dict],
        num_simulations: int = 1000,
        initial_capital: float = 10000
    ) -> dict:
        """
        Run Monte Carlo simulation on trade results
        
        Randomly resamples trade sequence to estimate distribution
        of possible outcomes
        
        Args:
            trades: Historical trade results
            num_simulations: Number of simulations
            initial_capital: Starting capital
            
        Returns:
            Simulation statistics
        """
        
        log.info(f"Running {num_simulations} Monte Carlo simulations...")
        
        final_capitals = []
        max_drawdowns = []
        
        for i in range(num_simulations):
            # Randomly resample trades with replacement
            simulated_trades = np.random.choice(trades, size=len(trades), replace=True)
            
            # Calculate equity curve
            capital = initial_capital
            equity_curve = [capital]
            peak = capital
            max_dd = 0
            
            for trade in simulated_trades:
                pnl = trade["pnl"]
                capital += pnl
                equity_curve.append(capital)
                
                # Track drawdown
                if capital > peak:
                    peak = capital
                else:
                    dd = ((peak - capital) / peak) * 100
                    max_dd = max(max_dd, dd)
            
            final_capitals.append(capital)
            max_drawdowns.append(max_dd)
        
        # Calculate statistics
        final_capitals = np.array(final_capitals)
        returns = ((final_capitals - initial_capital) / initial_capital) * 100
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)  # Value at Risk (95%)
        cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
        
        # Probability of profit
        prob_profit = (final_capitals > initial_capital).sum() / num_simulations * 100
        
        log.info(f"Simulation complete!")
        log.info(f"Expected return: {returns.mean():.2f}%")
        log.info(f"Probability of profit: {prob_profit:.1f}%")
        
        return {
            "expected_return": returns.mean(),
            "std_return": returns.std(),
            "median_return": np.median(returns),
            "best_case": returns.max(),
            "worst_case": returns.min(),
            "prob_profit": prob_profit,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "avg_max_drawdown": np.mean(max_drawdowns),
            "worst_drawdown": max(max_drawdowns)
        }


# Convenience functions
async def optimize_confidence_threshold(
    signals: List[dict],
    price_data: Dict[str, pd.DataFrame]
) -> dict:
    """
    Quick optimization of confidence threshold only
    
    Usage:
        result = await optimize_confidence_threshold(signals, price_data)
        print(f"Best confidence: {result['best_params']['min_confidence']}")
    """
    
    optimizer = StrategyOptimizer()
    
    param_grid = {
        "min_confidence": [50, 55, 60, 65, 70, 75, 80]
    }
    
    return await optimizer.grid_search(
        signals,
        price_data,
        param_grid,
        metric="sharpe_ratio"
    )