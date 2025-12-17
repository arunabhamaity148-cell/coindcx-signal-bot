"""
Advanced Smart Trading Logic
"""
import pandas as pd
import numpy as np
import requests
from indicators import *
from config import CANDLE_INTERVAL

def detect_market_regime(df: pd.DataFrame) -> str:
    """Detect market regime: TRENDING, RANGING, VOLATILE"""
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = bollinger_bands(df['close'], 20, 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['adx_calc'] = adx(df, 14)
    
    current_bb = df['bb_width'].iloc[-1]
    avg_bb = df['bb_width'].rolling(50).mean().iloc[-1]
    current_adx = df['adx_calc'].iloc[-1]
    
    if current_bb > avg_bb * 1.5 and current_adx > 30:
        return "VOLATILE_TRENDING"
    elif current_adx > 25:
        return "TRENDING"
    elif current_adx < 20 and current_bb < avg_bb * 0.8:
        return "RANGING"
    
    return "NEUTRAL"

def detect_order_flow(df: pd.DataFrame) -> dict:
    """Simulate order flow imbalance"""
    df['buy_pressure'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['sell_pressure'] = df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
    df['delta'] = df['buy_pressure'] - df['sell_pressure']
    df['cum_delta'] = df['delta'].rolling(20).sum()
    
    current = df['cum_delta'].iloc[-1]
    prev = df['cum_delta'].iloc[-2]
    slope = current - prev
    
    if current > 0 and slope > 0:
        strength = min(abs(slope) / df['volume'].mean() * 100, 100)
        return {"type": "BUY_IMBALANCE", "strength": strength}
    elif current < 0 and slope < 0:
        strength = min(abs(slope) / df['volume'].mean() * 100, 100)
        return {"type": "SELL_IMBALANCE", "strength": strength}
    
    return {"type": "NEUTRAL", "strength": 0}

def detect_liquidity_grab(df: pd.DataFrame) -> dict:
    """Detect liquidity grab patterns"""
    if len(df) < 30:
        return None
    
    swing_high = df['high'].rolling(10).max().shift(1).iloc[-1]
    swing_low = df['low'].rolling(10).min().shift(1).iloc[-1]
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Bullish grab
    if (current['low'] < swing_low and 
        current['close'] > swing_low and
        current['close'] > prev['close']):
        return {"type": "BULLISH_GRAB", "level": swing_low, "strength": 70}
    
    # Bearish grab
    if (current['high'] > swing_high and
        current['close'] < swing_high and
        current['close'] < prev['close']):
        return {"type": "BEARISH_GRAB", "level": swing_high, "strength": 70}
    
    return None

def multi_timeframe_momentum(market: str) -> dict:
    """Check momentum across multiple timeframes"""
    timeframes = ['5m', '15m', '1h']
    scores = {}
    
    for tf in timeframes:
        try:
            url = "https://public.coindcx.com/market_data/candles"
            params = {"pair": market, "interval": tf, "limit": 50}
            r = requests.get(url, params=params, timeout=8)
            data = r.json()
            
            if not data:
                continue
            
            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            df['ema20'] = ema(df['close'], 20)
            
            momentum = df['ema20'].iloc[-1] - df['ema20'].iloc[-5]
            scores[tf] = 1 if momentum > 0 else -1
        except:
            scores[tf] = 0
    
    total = sum(scores.values())
    
    if total == 3:
        return {"alignment": "STRONG_BULLISH", "score": 90}
    elif total == -3:
        return {"alignment": "STRONG_BEARISH", "score": 90}
    elif abs(total) == 2:
        return {"alignment": "MODERATE", "score": 60}
    
    return {"alignment": "MIXED", "score": 30}

def volume_profile_poc(df: pd.DataFrame) -> dict:
    """Volume Profile Point of Control"""
    if len(df) < 50:
        return None
    
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.linspace(price_min, price_max, 20)
    
    vol_at_price = {}
    for i in range(len(bins) - 1):
        mask = (df['close'] >= bins[i]) & (df['close'] < bins[i+1])
        vol_at_price[bins[i]] = df.loc[mask, 'volume'].sum()
    
    if not vol_at_price:
        return None
    
    poc = max(vol_at_price, key=vol_at_price.get)
    current = df['close'].iloc[-1]
    dist_pct = abs(current - poc) / poc * 100
    
    if current < poc and dist_pct > 1.5:
        return {"poc": poc, "signal": "BELOW_POC", "strength": min(dist_pct * 10, 100)}
    elif current > poc and dist_pct > 1.5:
        return {"poc": poc, "signal": "ABOVE_POC", "strength": min(dist_pct * 10, 100)}
    
    return {"poc": poc, "signal": "AT_POC", "strength": 40}

def sentiment_score(df: pd.DataFrame) -> dict:
    """Price momentum + volume sentiment"""
    df['vol_ma'] = sma(df['volume'], 20)
    df['price_mom'] = df['close'].pct_change(5) * 100
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    current = df.iloc[-1]
    mom = current['price_mom']
    vol_r = current['vol_ratio']
    
    if mom > 2 and vol_r > 1.5:
        return {"sentiment": "STRONG_BULLISH", "score": 85}
    elif mom < -2 and vol_r > 1.5:
        return {"sentiment": "STRONG_BEARISH", "score": 85}
    elif mom > 0.5 and vol_r > 1.2:
        return {"sentiment": "MODERATE_BULLISH", "score": 60}
    elif mom < -0.5 and vol_r > 1.2:
        return {"sentiment": "MODERATE_BEARISH", "score": 60}
    
    return {"sentiment": "NEUTRAL", "score": 30}

def fibonacci_levels(df: pd.DataFrame) -> dict:
    """Fibonacci retracement confluence"""
    if len(df) < 50:
        return None
    
    recent = df.tail(30)
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    
    fib_levels = {
        0.236: high - (diff * 0.236),
        0.382: high - (diff * 0.382),
        0.5: high - (diff * 0.5),
        0.618: high - (diff * 0.618),
        0.786: high - (diff * 0.786)
    }
    
    current = df['close'].iloc[-1]
    
    for level, price in fib_levels.items():
        dist = abs(current - price) / price * 100
        if dist < 0.5:
            return {
                "fib_level": level,
                "fib_price": price,
                "signal": "NEAR_FIB",
                "strength": 80 if level == 0.618 else 60
            }
    
    return None

def detect_fair_value_gap(df: pd.DataFrame) -> dict:
    """Smart Money Fair Value Gap"""
    if len(df) < 5:
        return None
    
    c1 = df.iloc[-3]
    c3 = df.iloc[-1]
    
    # Bullish FVG
    if c1['high'] < c3['low']:
        gap = c3['low'] - c1['high']
        return {"type": "BULLISH_FVG", "gap": gap, "strength": min(gap / c3['close'] * 5000, 100)}
    
    # Bearish FVG
    if c1['low'] > c3['high']:
        gap = c1['low'] - c3['high']
        return {"type": "BEARISH_FVG", "gap": gap, "strength": min(gap / c3['close'] * 5000, 100)}
    
    return None

def get_smart_signals(df: pd.DataFrame, market: str) -> dict:
    """Master combiner of all smart logic"""
    regime = detect_market_regime(df)
    order_flow = detect_order_flow(df)
    liquidity = detect_liquidity_grab(df)
    mtf = multi_timeframe_momentum(market)
    vol_prof = volume_profile_poc(df)
    sentiment = sentiment_score(df)
    fib = fibonacci_levels(df)
    fvg = detect_fair_value_gap(df)
    
    smart_score = 0
    signals = []
    
    # Regime
    if regime in ["TRENDING", "VOLATILE_TRENDING"]:
        smart_score += 12
        signals.append(f"📊 {regime}")
    
    # Order flow
    if order_flow['type'] == "BUY_IMBALANCE":
        smart_score += order_flow['strength'] * 0.18
        signals.append(f"💹 Buy Pressure: {order_flow['strength']:.0f}%")
    elif order_flow['type'] == "SELL_IMBALANCE":
        smart_score -= order_flow['strength'] * 0.18
        signals.append(f"📉 Sell Pressure: {order_flow['strength']:.0f}%")
    
    # Liquidity grab
    if liquidity:
        multiplier = 0.15 if liquidity['type'] == "BULLISH_GRAB" else -0.15
        smart_score += liquidity['strength'] * multiplier
        signals.append(f"🎯 {liquidity['type']}")
    
    # MTF
    if mtf['alignment'] == "STRONG_BULLISH":
        smart_score += 18
        signals.append("⏰ MTF: Strong Bull")
    elif mtf['alignment'] == "STRONG_BEARISH":
        smart_score -= 18
        signals.append("⏰ MTF: Strong Bear")
    
    # Volume Profile
    if vol_prof and vol_prof['signal'] == "BELOW_POC":
        smart_score += 12
        signals.append(f"📍 Below POC ({vol_prof['poc']:.2f})")
    elif vol_prof and vol_prof['signal'] == "ABOVE_POC":
        smart_score -= 12
        signals.append(f"📍 Above POC ({vol_prof['poc']:.2f})")
    
    # Sentiment
    if "BULLISH" in sentiment['sentiment']:
        smart_score += sentiment['score'] * 0.12
        signals.append(f"💚 {sentiment['sentiment']}")
    elif "BEARISH" in sentiment['sentiment']:
        smart_score -= sentiment['score'] * 0.12
        signals.append(f"❤️ {sentiment['sentiment']}")
    
    # Fibonacci
    if fib:
        smart_score += fib['strength'] * 0.08
        signals.append(f"🔢 Fib {fib['fib_level']}")
    
    # FVG
    if fvg:
        multiplier = 0.12 if fvg['type'] == "BULLISH_FVG" else -0.12
        smart_score += fvg['strength'] * multiplier
        signals.append(f"🧠 {fvg['type']}")
    
    final_score = min(abs(smart_score), 100)
    direction = "LONG" if smart_score > 0 else "SHORT"
    confidence = "HIGH" if final_score > 70 else "MODERATE" if final_score > 50 else "LOW"
    
    return {
        "smart_score": final_score,
        "direction": direction,
        "signals": signals,
        "regime": regime,
        "confidence": confidence
    }