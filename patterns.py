"""
Candlestick pattern detection
"""
import pandas as pd

def detect_patterns(df: pd.DataFrame) -> dict:
    """Detect multiple candlestick patterns"""
    if len(df) < 3:
        return {"pattern": None, "strength": 0, "all_patterns": []}
    
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]
    
    patterns = []
    strength = 0
    
    # Body and shadow calculations for last candle
    body = abs(c3["close"] - c3["open"])
    total_range = c3["high"] - c3["low"]
    lower_shadow = min(c3["open"], c3["close"]) - c3["low"]
    upper_shadow = c3["high"] - max(c3["open"], c3["close"])
    
    # === BULLISH PATTERNS ===
    
    # Bullish Engulfing
    if (c2["close"] < c2["open"] and c3["close"] > c3["open"] and
        c3["close"] > c2["open"] and c3["open"] < c2["close"]):
        patterns.append("bullish_engulfing")
        strength += 35
    
    # Morning Star
    if (c1["close"] < c1["open"] and
        abs(c2["close"] - c2["open"]) < (c1["high"] - c1["low"]) * 0.3 and
        c3["close"] > c3["open"] and 
        c3["close"] > (c1["open"] + c1["close"]) / 2):
        patterns.append("morning_star")
        strength += 40
    
    # Hammer
    if (total_range > 0 and body > 0 and
        lower_shadow > body * 2 and 
        upper_shadow < body * 0.5 and 
        c3["close"] > c3["open"]):
        patterns.append("hammer")
        strength += 30
    
    # Three White Soldiers
    if (c1["close"] > c1["open"] and 
        c2["close"] > c2["open"] and 
        c3["close"] > c3["open"] and
        c2["close"] > c1["close"] and 
        c3["close"] > c2["close"]):
        patterns.append("three_white_soldiers")
        strength += 45
    
    # === BEARISH PATTERNS ===
    
    # Bearish Engulfing
    if (c2["close"] > c2["open"] and c3["close"] < c3["open"] and
        c3["close"] < c2["open"] and c3["open"] > c2["close"]):
        patterns.append("bearish_engulfing")
        strength += 35
    
    # Evening Star
    if (c1["close"] > c1["open"] and
        abs(c2["close"] - c2["open"]) < (c1["high"] - c1["low"]) * 0.3 and
        c3["close"] < c3["open"] and 
        c3["close"] < (c1["open"] + c1["close"]) / 2):
        patterns.append("evening_star")
        strength += 40
    
    # Shooting Star
    if (total_range > 0 and body > 0 and
        upper_shadow > body * 2 and 
        lower_shadow < body * 0.5 and 
        c3["close"] < c3["open"]):
        patterns.append("shooting_star")
        strength += 30
    
    # Three Black Crows
    if (c1["close"] < c1["open"] and 
        c2["close"] < c2["open"] and 
        c3["close"] < c3["open"] and
        c2["close"] < c1["close"] and 
        c3["close"] < c2["close"]):
        patterns.append("three_black_crows")
        strength += 45
    
    # Doji (indecision)
    if body < (total_range * 0.1) and total_range > 0:
        patterns.append("doji")
        strength += 5
    
    return {
        "pattern": patterns[0] if patterns else None,
        "all_patterns": patterns,
        "strength": min(strength, 100)
    }

def is_bullish_pattern(pattern: str) -> bool:
    """Check if pattern is bullish"""
    bullish = ["bullish_engulfing", "morning_star", "hammer", 
               "three_white_soldiers", "piercing_line"]
    return pattern in bullish

def is_bearish_pattern(pattern: str) -> bool:
    """Check if pattern is bearish"""
    bearish = ["bearish_engulfing", "evening_star", "shooting_star",
               "three_black_crows", "dark_cloud_cover"]
    return pattern in bearish