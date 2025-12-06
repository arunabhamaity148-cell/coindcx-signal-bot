# ================================================================
# scorer.py — FINAL (With Correlation & All Features)
# ================================================================

import numpy as np
import logging

from helpers import (
    build_ohlcv_from_trades,
    calc_rsi,
    calc_atr,
    calc_vwap_from_trades,
    orderflow_metrics,
    btc_calm_check,
)
from price_levels import get_complete_levels
from volume_analysis import get_volume_insights
from signal_confidence import SignalConfidence, enhance_signal
from correlation_engine import CorrelationAnalyzer

log = logging.getLogger("scorer")

# Strategy thresholds
STRAT_MIN = {
    "QUICK": 6.8,
    "MID": 7.4,
    "TREND": 8.0
}


# --------------------------------------------------------------
# Compute MTF EMA structure (1m/5m/15m/60m)
# --------------------------------------------------------------
def compute_mtf(df1, df5, df15, df60):
    """Multi-timeframe trend analysis"""
    def ema_state(df):
        if df is None or len(df) < 60:
            return None
        close = df["close"]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        return 1 if ema20 > ema50 else -1

    states = [ema_state(df1), ema_state(df5), ema_state(df15), ema_state(df60)]
    bull = sum(1 for s in states if s == 1)
    bear = sum(1 for s in states if s == -1)
    return {"bull": bull, "bear": bear}


# --------------------------------------------------------------
# 47 LOGIC EVALUATOR
# --------------------------------------------------------------
def evaluate_logic(params):
    """Evaluate all 47 logic points"""
    L = {}

    rsi = params["rsi"]
    last = params["last"]
    vwap = params["vwap"]
    imb = params["imb"]
    depth = params["depth"]
    spread = params["spread"]
    atr = params["atr"]
    mom1 = params["mom1"]
    mom5 = params["mom5"]
    mtf = params["mtf"]

    # ========== Trend & MTF Logic ==========
    L["mtf_bull"] = 1 if mtf["bull"] >= 3 else 0
    L["mtf_bear"] = 1 if mtf["bear"] >= 3 else 0
    L["atr_stable"] = 1 if atr < (last * 0.002) else 0
    L["low_atr"] = 1 if atr < last * 0.0018 else 0

    # ========== Momentum Logic ==========
    L["mom1"] = 1 if abs(mom1) > 0.0008 else 0
    L["mom5"] = 1 if abs(mom5) > 0.0015 else 0
    L["imbalance"] = 1 if abs(imb) > 0.05 else 0
    L["spread_ok"] = 1 if spread < 0.32 else 0
    L["depth_ok"] = 1 if depth > 30000 else 0

    # ========== RSI Logic ==========
    L["rsi_mid"] = 1 if 45 <= rsi <= 55 else 0
    L["rsi_flip"] = 1 if (rsi > 50 and mom1 > 0) else 0
    L["rsi_div"] = 1 if (mom1 > 0 and rsi < 50) or (mom1 < 0 and rsi > 50) else 0

    # ========== VWAP Logic ==========
    L["vwap_close"] = 1 if abs(last - vwap) / vwap < 0.002 else 0
    L["vwap_small_dev"] = 1 if abs(last - vwap) / vwap < 0.0025 else 0
    L["vwap_momentum"] = 1 if (last > vwap and mom1 > 0) else 0

    # ========== Orderflow Logic ==========
    L["imb_buy"] = 1 if imb > 0.05 else 0
    L["imb_sell"] = 1 if imb < -0.05 else 0

    return L


# --------------------------------------------------------------
# Hybrid score aggregator
# --------------------------------------------------------------
def aggregate_score(logic_dict):
    """Calculate total score from logic"""
    score = 0
    for k, v in logic_dict.items():
        score += v
    return float(score)


# --------------------------------------------------------------
# Extract price series from OHLCV
# --------------------------------------------------------------
def extract_prices(df):
    """Extract close prices as list"""
    if df is None or len(df) == 0:
        return []
    return df["close"].tolist()


# --------------------------------------------------------------
# Compute final signal for a symbol
# --------------------------------------------------------------
async def compute_signal(sym: str, strat: str, trade_buffer: dict, ob_cache: dict):
    """
    Main signal computation with all features
    
    Args:
        sym: Symbol name
        strat: Strategy type
        trade_buffer: Trade data buffer
        ob_cache: Orderbook cache
        
    Returns:
        Enhanced signal dict or None
    """
    
    # -------------------------
    # BTC Calm mandatory
    # -------------------------
    if sym != "BTCUSDT":
        ok = await btc_calm_check(buffer_dict=trade_buffer)
        if not ok:
            return None

    # -------------------------
    # Build MTF OHLC
    # -------------------------
    o1 = await build_ohlcv_from_trades(sym, "1min", 200, buffer_dict=trade_buffer)
    if o1 is None or len(o1) < 60:
        return None

    o5 = await build_ohlcv_from_trades(sym, "5min", 200, buffer_dict=trade_buffer)
    o15 = await build_ohlcv_from_trades(sym, "15min", 200, buffer_dict=trade_buffer)
    o60 = await build_ohlcv_from_trades(sym, "60min", 200, buffer_dict=trade_buffer)

    mtf = compute_mtf(o1, o5, o15, o60)

    # -------------------------
    # Indicators
    # -------------------------
    last = float(o1["close"].iloc[-1])
    mom1 = (last - float(o1["close"].iloc[-2])) / last
    mom5 = 0
    if o5 is not None and len(o5) > 2:
        mom5 = (last - float(o5["close"].iloc[-2])) / last

    rsi = float(calc_rsi(o1["close"]).iloc[-1])
    atr = float(calc_atr(o1))

    vwap = await calc_vwap_from_trades(sym, buffer_dict=trade_buffer)
    if vwap is None:
        return None

    flow = await orderflow_metrics(sym, buffer_dict=trade_buffer, ob_cache=ob_cache)
    if not flow:
        return None

    # -------------------------
    # BTC Correlation
    # -------------------------
    correlation = {}
    if sym != "BTCUSDT":
        try:
            sym_prices = extract_prices(o1)
            btc_o1 = await build_ohlcv_from_trades("BTCUSDT", "1min", 200, buffer_dict=trade_buffer)
            
            if btc_o1 is not None and len(btc_o1) >= 60:
                btc_prices = extract_prices(btc_o1)
                
                analyzer = CorrelationAnalyzer()
                correlation = analyzer.analyze_market_correlation(
                    {"prices": sym_prices},
                    {"prices": btc_prices},
                    window=50
                )
        except Exception as e:
            log.error(f"Correlation calc error for {sym}: {e}")
            correlation = {}

    # -------------------------
    # Build params dict
    # -------------------------
    params = {
        "rsi": rsi,
        "atr": atr,
        "vwap": vwap,
        "imb": flow["imbalance"],
        "depth": flow["depth_usd"],
        "spread": flow["spread_pct"],
        "last": last,
        "mom1": mom1,
        "mom5": mom5,
        "mtf": mtf,
    }

    # -------------------------
    # 47 Logic Evaluation
    # -------------------------
    L = evaluate_logic(params)
    score = aggregate_score(L)

    if score < STRAT_MIN[strat]:
        return None

    side = "long" if params["imb"] > 0 else "short"

    # -------------------------
    # Build base signal
    # -------------------------
    signal = {
        "symbol": sym,
        "side": side,
        "score": score,
        "last": last,
        "strategy": strat,
        "logic": L,
        "passed": [k for k, v in L.items() if v == 1]
    }

    # -------------------------
    # Add price levels
    # -------------------------
    try:
        levels = await get_complete_levels(sym, o1)
        signal["levels"] = levels
    except Exception as e:
        log.error(f"Price levels error for {sym}: {e}")
        signal["levels"] = None

    # -------------------------
    # Add volume analysis
    # -------------------------
    try:
        trades = list(trade_buffer.get(sym, []))
        volume = await get_volume_insights(o1, trades)
        signal["volume"] = volume
    except Exception as e:
        log.error(f"Volume analysis error for {sym}: {e}")
        signal["volume"] = None

    # -------------------------
    # Add correlation
    # -------------------------
    signal["correlation"] = correlation

    # -------------------------
    # Enhance with confidence
    # -------------------------
    try:
        signal = enhance_signal(signal, params)
        
        # Adjust confidence based on correlation
        if correlation and "confidence_adj" in correlation:
            adj = correlation.get("confidence_adj", 0)
            signal["confidence"] = max(0, min(100, signal["confidence"] + adj))
    
    except Exception as e:
        log.error(f"Signal enhancement error for {sym}: {e}")

    return signal