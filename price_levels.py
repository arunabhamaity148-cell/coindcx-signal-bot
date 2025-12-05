# ================================================================
# price_levels.py — Support/Resistance & Key Levels
# ================================================================

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


class PriceLevels:
    """Calculate support/resistance and key price levels"""
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, order: int = 5) -> dict:
        """
        Find support and resistance levels from OHLCV data
        
        Args:
            df: OHLCV DataFrame
            order: Number of candles to look for local extrema
            
        Returns:
            {
                "support": [level1, level2, ...],
                "resistance": [level1, level2, ...],
                "nearest_support": float,
                "nearest_resistance": float
            }
        """
        
        if len(df) < order * 2:
            return None
        
        # Find local minima (support)
        local_min_idx = argrelextrema(
            df['low'].values, 
            np.less_equal, 
            order=order
        )[0]
        
        # Find local maxima (resistance)
        local_max_idx = argrelextrema(
            df['high'].values, 
            np.greater_equal, 
            order=order
        )[0]
        
        support_levels = df['low'].iloc[local_min_idx].tolist()
        resistance_levels = df['high'].iloc[local_max_idx].tolist()
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Find nearest levels
        support_below = [s for s in support_levels if s < current_price]
        resistance_above = [r for r in resistance_levels if r > current_price]
        
        nearest_support = max(support_below) if support_below else None
        nearest_resistance = min(resistance_above) if resistance_above else None
        
        return {
            "support": sorted(support_levels, reverse=True)[:3],
            "resistance": sorted(resistance_levels)[:3],
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "current_price": current_price
        }
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> dict:
        """
        Calculate classic pivot points
        
        Returns:
            {
                "pivot": float,
                "r1": float, "r2": float, "r3": float,
                "s1": float, "s2": float, "s3": float
            }
        """
        
        if len(df) < 1:
            return None
        
        # Use last complete candle
        high = float(df['high'].iloc[-2])
        low = float(df['low'].iloc[-2])
        close = float(df['close'].iloc[-2])
        
        # Classic pivot calculation
        pivot = (high + low + close) / 3
        
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": round(pivot, 6),
            "r1": round(r1, 6),
            "r2": round(r2, 6),
            "r3": round(r3, 6),
            "s1": round(s1, 6),
            "s2": round(s2, 6),
            "s3": round(s3, 6)
        }
    
    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
        """
        Calculate Fibonacci retracement levels
        
        Returns:
            {
                "0%": high,
                "23.6%": level,
                "38.2%": level,
                "50%": level,
                "61.8%": level,
                "100%": low
            }
        """
        
        if len(df) < lookback:
            return None
        
        recent = df.tail(lookback)
        high = float(recent['high'].max())
        low = float(recent['low'].min())
        diff = high - low
        
        return {
            "0%": round(high, 6),
            "23.6%": round(high - (diff * 0.236), 6),
            "38.2%": round(high - (diff * 0.382), 6),
            "50%": round(high - (diff * 0.5), 6),
            "61.8%": round(high - (diff * 0.618), 6),
            "78.6%": round(high - (diff * 0.786), 6),
            "100%": round(low, 6)
        }
    
    @staticmethod
    def get_price_distance_percent(current: float, target: float) -> float:
        """Calculate percentage distance between two prices"""
        return ((target - current) / current) * 100


class KeyLevelAnalyzer:
    """Analyze price in relation to key levels"""
    
    @staticmethod
    def analyze_position(current_price: float, levels: dict, pivots: dict) -> dict:
        """
        Analyze current price position relative to key levels
        
        Returns:
            {
                "near_support": bool,
                "near_resistance": bool,
                "between_levels": str,
                "pivot_zone": str,
                "risk_reward_ratio": float
            }
        """
        
        analysis = {}
        
        # Check proximity to support/resistance
        if levels and levels.get("nearest_support"):
            support_dist = abs((current_price - levels["nearest_support"]) / current_price * 100)
            analysis["near_support"] = support_dist < 0.5
            analysis["support_distance_pct"] = round(support_dist, 2)
        
        if levels and levels.get("nearest_resistance"):
            resistance_dist = abs((levels["nearest_resistance"] - current_price) / current_price * 100)
            analysis["near_resistance"] = resistance_dist < 0.5
            analysis["resistance_distance_pct"] = round(resistance_dist, 2)
        
        # Pivot zone analysis
        if pivots:
            pivot = pivots["pivot"]
            if current_price > pivots["r1"]:
                analysis["pivot_zone"] = "ABOVE_R1"
            elif current_price > pivot:
                analysis["pivot_zone"] = "ABOVE_PIVOT"
            elif current_price > pivots["s1"]:
                analysis["pivot_zone"] = "BELOW_PIVOT"
            else:
                analysis["pivot_zone"] = "BELOW_S1"
        
        # Risk/Reward calculation
        if levels and levels.get("nearest_support") and levels.get("nearest_resistance"):
            support = levels["nearest_support"]
            resistance = levels["nearest_resistance"]
            
            potential_gain = resistance - current_price
            potential_loss = current_price - support
            
            if potential_loss > 0:
                rr_ratio = potential_gain / potential_loss
                analysis["risk_reward_ratio"] = round(rr_ratio, 2)
            else:
                analysis["risk_reward_ratio"] = None
        
        return analysis


# Usage example
async def get_complete_levels(symbol: str, df: pd.DataFrame) -> dict:
    """Get all price levels and analysis for a symbol"""
    
    price_calc = PriceLevels()
    analyzer = KeyLevelAnalyzer()
    
    # Calculate all levels
    sr_levels = price_calc.find_support_resistance(df)
    pivots = price_calc.calculate_pivot_points(df)
    fibs = price_calc.calculate_fibonacci_levels(df)
    
    current_price = float(df['close'].iloc[-1])
    
    # Analyze position
    position_analysis = analyzer.analyze_position(current_price, sr_levels, pivots)
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "support_resistance": sr_levels,
        "pivots": pivots,
        "fibonacci": fibs,
        "analysis": position_analysis
    }