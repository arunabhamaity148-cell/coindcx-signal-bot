# ================================================================
# volume_analysis.py — Volume Profile & Institutional Detection
# ================================================================

import pandas as pd
import numpy as np
from collections import defaultdict


class VolumeProfile:
    """Analyze volume distribution at price levels"""
    
    @staticmethod
    def calculate_vp(df: pd.DataFrame, bins: int = 20) -> dict:
        """
        Calculate Volume Profile (Volume at Price)
        
        Returns:
            {
                "poc": float,  # Point of Control (highest volume)
                "vah": float,  # Value Area High
                "val": float,  # Value Area Low
                "profile": [(price, volume), ...]
            }
        """
        
        if len(df) < 10:
            return None
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        bin_size = price_range / bins
        
        # Aggregate volume at each price level
        volume_at_price = defaultdict(float)
        
        for idx, row in df.iterrows():
            # Distribute volume across the candle's range
            candle_range = row['high'] - row['low']
            if candle_range == 0:
                bin_idx = int((row['close'] - price_min) / bin_size)
                price_level = price_min + (bin_idx * bin_size)
                volume_at_price[price_level] += row['vol']
            else:
                # Distribute proportionally
                num_bins_in_candle = max(1, int(candle_range / bin_size))
                vol_per_bin = row['vol'] / num_bins_in_candle
                
                for i in range(num_bins_in_candle):
                    price_level = row['low'] + (i * bin_size)
                    volume_at_price[price_level] += vol_per_bin
        
        # Sort by volume
        sorted_profile = sorted(
            volume_at_price.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Point of Control (POC) - price with highest volume
        poc = sorted_profile[0][0] if sorted_profile else price_min
        
        # Value Area (70% of volume)
        total_volume = sum(v for _, v in sorted_profile)
        target_volume = total_volume * 0.70
        
        cumulative_vol = 0
        value_area_prices = []
        
        for price, vol in sorted_profile:
            cumulative_vol += vol
            value_area_prices.append(price)
            if cumulative_vol >= target_volume:
                break
        
        vah = max(value_area_prices) if value_area_prices else price_max
        val = min(value_area_prices) if value_area_prices else price_min
        
        return {
            "poc": round(poc, 6),
            "vah": round(vah, 6),
            "val": round(val, 6),
            "profile": sorted_profile[:10]  # Top 10 levels
        }
    
    @staticmethod
    def is_price_in_value_area(current_price: float, vp_data: dict) -> bool:
        """Check if current price is within value area"""
        if not vp_data:
            return False
        return vp_data["val"] <= current_price <= vp_data["vah"]


class SmartMoneyDetector:
    """Detect institutional/smart money activity"""
    
    @staticmethod
    def detect_large_orders(trades: list, percentile: float = 95) -> dict:
        """
        Detect unusually large orders (smart money)
        
        Args:
            trades: List of trade dicts with 'p', 'q', 'm'
            percentile: Size threshold percentile
            
        Returns:
            {
                "large_buys": int,
                "large_sells": int,
                "buy_volume": float,
                "sell_volume": float,
                "smart_money_direction": "BUY|SELL|NEUTRAL"
            }
        """
        
        if not trades or len(trades) < 20:
            return None
        
        df = pd.DataFrame(trades)
        df['q'] = df['q'].astype(float)
        
        # Calculate large order threshold
        threshold = df['q'].quantile(percentile / 100)
        
        # Filter large orders
        large_orders = df[df['q'] >= threshold].copy()
        
        if len(large_orders) == 0:
            return {
                "large_buys": 0,
                "large_sells": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "smart_money_direction": "NEUTRAL"
            }
        
        # Separate buys and sells (m=True means sell, m=False means buy)
        buys = large_orders[large_orders['m'] == False]
        sells = large_orders[large_orders['m'] == True]
        
        buy_count = len(buys)
        sell_count = len(sells)
        buy_vol = buys['q'].sum() if buy_count > 0 else 0
        sell_vol = sells['q'].sum() if sell_count > 0 else 0
        
        # Determine direction
        if buy_vol > sell_vol * 1.5:
            direction = "BUY"
        elif sell_vol > buy_vol * 1.5:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return {
            "large_buys": int(buy_count),
            "large_sells": int(sell_count),
            "buy_volume": float(buy_vol),
            "sell_volume": float(sell_vol),
            "smart_money_direction": direction,
            "threshold": float(threshold)
        }
    
    @staticmethod
    def detect_absorption(trades: list, window: int = 50) -> dict:
        """
        Detect price absorption (large volume, small price change)
        Indicates accumulation/distribution
        """
        
        if not trades or len(trades) < window:
            return None
        
        df = pd.DataFrame(trades[-window:])
        df['p'] = df['p'].astype(float)
        df['q'] = df['q'].astype(float)
        
        # Price change
        price_change_pct = abs((df['p'].iloc[-1] - df['p'].iloc[0]) / df['p'].iloc[0] * 100)
        
        # Volume
        total_volume = df['q'].sum()
        avg_volume = df['q'].mean()
        
        # Absorption detected if high volume but low price movement
        is_absorbing = price_change_pct < 0.1 and total_volume > avg_volume * window * 0.8
        
        # Direction of absorption
        df['side'] = np.where(df['m'], -1, 1)
        net_delta = (df['q'] * df['side']).sum()
        
        if is_absorbing:
            absorption_type = "ACCUMULATION" if net_delta > 0 else "DISTRIBUTION"
        else:
            absorption_type = None
        
        return {
            "is_absorbing": is_absorbing,
            "absorption_type": absorption_type,
            "price_change_pct": round(price_change_pct, 3),
            "volume_ratio": round(total_volume / (avg_volume * window), 2)
        }


class VolumeAnalyzer:
    """Comprehensive volume analysis"""
    
    @staticmethod
    def analyze_volume_trend(df: pd.DataFrame, period: int = 20) -> dict:
        """Analyze volume trend and strength"""
        
        if len(df) < period:
            return None
        
        recent = df.tail(period)
        
        # Volume moving average
        vol_ma = recent['vol'].mean()
        current_vol = recent['vol'].iloc[-1]
        
        # Volume trend
        vol_increasing = recent['vol'].iloc[-5:].mean() > recent['vol'].iloc[-10:-5].mean()
        
        # Volume spike detection
        vol_spike = current_vol > vol_ma * 1.5
        
        # Price-volume correlation
        price_changes = recent['close'].pct_change()
        vol_changes = recent['vol'].pct_change()
        correlation = price_changes.corr(vol_changes)
        
        return {
            "current_volume": float(current_vol),
            "avg_volume": float(vol_ma),
            "volume_ratio": round(current_vol / vol_ma, 2),
            "volume_increasing": vol_increasing,
            "volume_spike": vol_spike,
            "price_volume_correlation": round(correlation, 2)
        }


# Usage example
async def get_volume_insights(df: pd.DataFrame, trades: list) -> dict:
    """Get complete volume analysis"""
    
    vp = VolumeProfile()
    smd = SmartMoneyDetector()
    va = VolumeAnalyzer()
    
    # Calculate all metrics
    volume_profile = vp.calculate_vp(df)
    smart_money = smd.detect_large_orders(trades)
    absorption = smd.detect_absorption(trades)
    volume_trend = va.analyze_volume_trend(df)
    
    current_price = float(df['close'].iloc[-1])
    in_value_area = vp.is_price_in_value_area(current_price, volume_profile)
    
    return {
        "volume_profile": volume_profile,
        "smart_money": smart_money,
        "absorption": absorption,
        "volume_trend": volume_trend,
        "in_value_area": in_value_area
    }