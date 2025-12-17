"""
Main signal generation logic
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from indicators import *
from patterns import detect_patterns, is_bullish_pattern, is_bearish_pattern
from config import *

# Tracking
last_signal_time = {}
daily_signal_count = {}

def can_send_signal(market: str) -> bool:
    """Check cooldown and daily limits"""
    now = datetime.now()
    
    # Cooldown check
    if market in last_signal_time:
        elapsed = (now - last_signal_time[market]).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES:
            return False
    
    # Daily limit
    today = now.date()
    if today not in daily_signal_count:
        daily_signal_count.clear()
        daily_signal_count[today] = 0
    
    if daily_signal_count[today] >= MAX_SIGNALS_PER_DAY:
        return False
    
    return True

def mark_signal_sent(market: str):
    """Mark signal as sent"""
    now = datetime.now()
    last_signal_time[market] = now
    
    today = now.date()
    if today not in daily_signal_count:
        daily_signal_count[today] = 0
    daily_signal_count[today] += 1

def calculate_score(conditions: dict) -> int:
    """Advanced scoring system"""
    score = 0
    
    if conditions.get("htf_trend"):
        score += 28
    
    if conditions.get("rsi_ok") and conditions.get("macd_ok") and conditions.get("ema_ok"):
        score += 24
    
    if conditions.get("volume_spike"):
        score += 16
    
    if conditions.get("adx_strong"):
        score += 14
    
    score += min(conditions.get("pattern_strength", 0) // 10, 12)
    
    if conditions.get("divergence"):
        score += 6
    
    return min(score, 100)

def analyze_market(df: pd.DataFrame, market: str) -> dict:
    """Main technical analysis"""
    if df is None or len(df) < 60:
        return None
    
    # Calculate indicators
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df["atr"] = atr(df, ATR_PERIOD)
    df["adx"] = adx(df, ADX_PERIOD)
    df["volume_ma"] = sma(df["volume"], 20)
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Pattern detection
    pattern_info = detect_patterns(df) if ENABLE_PATTERN_DETECTION else {"pattern": None, "strength": 0}
    
    # Divergence
    divergence = detect_divergence(df["close"].tail(20), df["rsi"].tail(20), 10)
    
    # BUY conditions
    buy_cond = {
        "htf_trend": current["ema_slow"] > current["ema_trend"],
        "rsi_ok": (prev["rsi"] < 35 and current["rsi"] >= 35) or (30 < current["rsi"] < 55),
        "macd_ok": current["macd"] > current["macd_signal"] and current["macd_hist"] > 0,
        "ema_ok": current["ema_fast"] > current["ema_slow"] and current["close"] > current["ema_fast"],
        "volume_spike": current["volume"] > current["volume_ma"] * 1.4 if ENABLE_VOLUME_FILTER else True,
        "adx_strong": current["adx"] > 24,
        "pattern_strength": pattern_info["strength"] if is_bullish_pattern(pattern_info["pattern"]) else 0,
        "divergence": divergence == "bullish_divergence"
    }
    
    # SELL conditions
    sell_cond = {
        "htf_trend": current["ema_slow"] < current["ema_trend"],
        "rsi_ok": (prev["rsi"] > 65 and current["rsi"] <= 65) or (45 < current["rsi"] < 70),
        "macd_ok": current["macd"] < current["macd_signal"] and current["macd_hist"] < 0,
        "ema_ok": current["ema_fast"] < current["ema_slow"] and current["close"] < current["ema_fast"],
        "volume_spike": current["volume"] > current["volume_ma"] * 1.4 if ENABLE_VOLUME_FILTER else True,
        "adx_strong": current["adx"] > 24,
        "pattern_strength": pattern_info["strength"] if is_bearish_pattern(pattern_info["pattern"]) else 0,
        "divergence": divergence == "bearish_divergence"
    }
    
    buy_score = calculate_score(buy_cond)
    sell_score = calculate_score(sell_cond)
    
    if buy_score < MIN_SIGNAL_SCORE and sell_score < MIN_SIGNAL_SCORE:
        return None
    
    direction = "LONG" if buy_score >= sell_score else "SHORT"
    score = buy_score if direction == "LONG" else sell_score
    
    # SL/TP calculation
    entry = float(current["close"])
    atr_val = float(current["atr"]) if not np.isnan(current["atr"]) else entry * 0.012
    
    if direction == "LONG":
        sl = entry - (ATR_MULTIPLIER_SL * atr_val)
        tp1 = entry + (ATR_MULTIPLIER_TP1 * atr_val)
        tp2 = entry + (ATR_MULTIPLIER_TP2 * atr_val)
    else:
        sl = entry + (ATR_MULTIPLIER_SL * atr_val)
        tp1 = entry - (ATR_MULTIPLIER_TP1 * atr_val)
        tp2 = entry - (ATR_MULTIPLIER_TP2 * atr_val)
    
    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    rr = reward / risk if risk > 0 else 0
    
    # Risk/Reward filter
    if rr < RISK_REWARD_MIN:
        return None
    
    return {
        "market": market,
        "direction": direction,
        "score": score,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "rsi": float(current["rsi"]),
        "macd_hist": float(current["macd_hist"]),
        "adx": float(current["adx"]),
        "ema_fast": float(current["ema_fast"]),
        "ema_slow": float(current["ema_slow"]),
        "volume_spike": buy_cond["volume_spike"] if direction == "LONG" else sell_cond["volume_spike"],
        "pattern": pattern_info["pattern"],
        "divergence": divergence
    }