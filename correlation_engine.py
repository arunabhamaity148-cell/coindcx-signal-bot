# ================================================================
# correlation_engine.py - Market Correlation Analysis
# ================================================================

import numpy as np
import pandas as pd
import logging
from typing import Optional

log = logging.getLogger("correlation")


def calc_return_series(prices: np.ndarray) -> np.ndarray:
    """Calculate percentage returns from price series"""
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1]


def correlation_with_btc(sym_prices: list, btc_prices: list, window: int = 50) -> float:
    """
    Calculate rolling correlation with BTC
    
    Args:
        sym_prices: Symbol price array
        btc_prices: BTC price array
        window: Lookback window
        
    Returns:
        Correlation coefficient (-1 to 1)
    """
    
    if len(sym_prices) < 20 or len(btc_prices) < 20:
        return 0.0
    
    try:
        sym_array = np.array(sym_prices, dtype=float)
        btc_array = np.array(btc_prices, dtype=float)
        
        # Calculate returns
        r1 = calc_return_series(sym_array)
        r2 = calc_return_series(btc_array)
        
        # Use last 'window' periods
        n = min(window, len(r1), len(r2))
        r1 = r1[-n:]
        r2 = r2[-n:]
        
        # Calculate correlation
        if len(r1) < 2 or len(r2) < 2:
            return 0.0
        
        corr_matrix = np.corrcoef(r1, r2)
        corr = corr_matrix[0, 1]
        
        # Handle NaN
        if np.isnan(corr):
            return 0.0
        
        return float(corr)
    
    except Exception as e:
        log.error(f"Correlation calculation error: {e}")
        return 0.0


class CorrelationAnalyzer:
    """Advanced correlation analysis"""
    
    @staticmethod
    def analyze_market_correlation(
        symbol_data: dict,
        btc_data: dict,
        window: int = 50
    ) -> dict:
        """
        Comprehensive correlation analysis
        
        Args:
            symbol_data: {"prices": [...], "volumes": [...]}
            btc_data: {"prices": [...], "volumes": [...]}
            window: Analysis window
            
        Returns:
            {
                "price_corr": float,
                "volume_corr": float,
                "strength": str,
                "divergence": bool
            }
        """
        
        sym_prices = symbol_data.get("prices", [])
        btc_prices = btc_data.get("prices", [])
        
        # Price correlation
        price_corr = correlation_with_btc(sym_prices, btc_prices, window)
        
        # Volume correlation (if available)
        volume_corr = 0.0
        if "volumes" in symbol_data and "volumes" in btc_data:
            sym_vols = symbol_data["volumes"]
            btc_vols = btc_data["volumes"]
            if len(sym_vols) >= 20 and len(btc_vols) >= 20:
                volume_corr = correlation_with_btc(sym_vols, btc_vols, window)
        
        # Correlation strength classification
        abs_corr = abs(price_corr)
        if abs_corr >= 0.7:
            strength = "STRONG"
        elif abs_corr >= 0.4:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Detect divergence (price up, corr down or vice versa)
        divergence = False
        if len(sym_prices) >= 10 and len(btc_prices) >= 10:
            sym_change = (sym_prices[-1] - sym_prices[-10]) / sym_prices[-10]
            btc_change = (btc_prices[-1] - btc_prices[-10]) / btc_prices[-10]
            
            # Divergence: opposite directions with strong correlation
            if abs_corr > 0.6:
                if (sym_change > 0 and btc_change < 0) or (sym_change < 0 and btc_change > 0):
                    divergence = True
        
        return {
            "price_corr": round(price_corr, 3),
            "volume_corr": round(volume_corr, 3),
            "strength": strength,
            "divergence": divergence,
            "independent": abs_corr < 0.3  # Moving independently from BTC
        }
    
    @staticmethod
    def get_correlation_signal(corr_data: dict, signal_side: str) -> dict:
        """
        Evaluate if correlation supports the signal
        
        Args:
            corr_data: Correlation analysis result
            signal_side: "long" or "short"
            
        Returns:
            {
                "supported": bool,
                "reason": str,
                "confidence_adj": float  # Adjustment to confidence (-10 to +10)
            }
        """
        
        price_corr = corr_data["price_corr"]
        strength = corr_data["strength"]
        divergence = corr_data["divergence"]
        independent = corr_data["independent"]
        
        confidence_adj = 0.0
        supported = True
        reason = ""
        
        # Strong positive correlation (follows BTC)
        if price_corr > 0.7:
            reason = "Strong BTC correlation"
            confidence_adj = 5.0  # Boost confidence
        
        # Strong negative correlation (inverse BTC)
        elif price_corr < -0.7:
            reason = "Strong inverse BTC correlation"
            confidence_adj = 3.0
        
        # Independent movement (good for signals)
        elif independent:
            reason = "Independent from BTC"
            confidence_adj = 8.0  # Best case - own movement
        
        # Divergence detected
        elif divergence:
            reason = "Divergence from BTC detected"
            confidence_adj = -5.0  # Risky
            supported = False
        
        # Weak correlation
        else:
            reason = f"{strength} correlation with BTC"
            confidence_adj = 2.0
        
        return {
            "supported": supported,
            "reason": reason,
            "confidence_adj": confidence_adj
        }


# Convenience function
async def get_btc_correlation(
    symbol_prices: list,
    btc_prices: list
) -> dict:
    """
    Quick correlation check
    
    Usage:
        corr = await get_btc_correlation(eth_prices, btc_prices)
        print(f"Correlation: {corr['price_corr']}")
    """
    
    analyzer = CorrelationAnalyzer()
    
    symbol_data = {"prices": symbol_prices}
    btc_data = {"prices": btc_prices}
    
    return analyzer.analyze_market_correlation(symbol_data, btc_data)