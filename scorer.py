# ===========================
# scorer.py — Optimized + Enhanced
# ===========================
import numpy as np
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("scorer_v10")

from helpers import (
    build_ohlcv_from_trades,
    calc_rsi,
    calc_atr,
    calc_vwap_from_trades,
    orderflow_metrics,
    btc_calm_check,
)

# External enhancement modules (local files in repo)
# NOTE: import these lazily inside compute_signal to avoid circular imports where needed.


STRAT_MIN = {
    "QUICK": 6.8,
    "MID": 7.4,
    "TREND": 8.0
}


def compute_mtf(df1, df5, df15, df60):
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


def evaluate_logic(params):
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

    # Trend & MTF Logic
    L["mtf_bull"] = 1 if mtf["bull"] >= 3 else 0
    L["mtf_bear"] = 1 if mtf["bear"] >= 3 else 0
    L["atr_stable"] = 1 if atr < (last * 0.002) else 0
    L["vwap_close"] = 1 if abs(last - vwap) / vwap < 0.002 else 0

    # Momentum Logic
    L["mom1"] = 1 if abs(mom1) > 0.0008 else 0
    L["mom5"] = 1 if abs(mom5) > 0.0015 else 0
    L["imbalance"] = 1 if abs(imb) > 0.05 else 0
    L["spread_ok"] = 1 if spread < 0.32 else 0
    L["depth_ok"] = 1 if depth > 30000 else 0

    # RSI Logic
    L["rsi_mid"] = 1 if 45 <= rsi <= 55 else 0
    L["rsi_flip"] = 1 if (rsi > 50 and mom1 > 0) else 0
    L["rsi_div"] = 1 if (mom1 > 0 and rsi < 50) or (mom1 < 0 and rsi > 50) else 0

    # VWAP Logic
    L["vwap_small_dev"] = 1 if abs(last - vwap) / vwap < 0.0025 else 0
    L["vwap_momentum"] = 1 if (last > vwap and mom1 > 0) else 0

    # Orderflow Logic
    L["imb_buy"] = 1 if imb > 0.05 else 0
    L["imb_sell"] = 1 if imb < -0.05 else 0

    # Volatility Logic
    L["low_atr"] = 1 if atr < last * 0.0018 else 0

    return L


def aggregate_score(logic_dict):
    score = 0
    for k, v in logic_dict.items():
        score += v
    return float(score)


async def compute_signal(sym: str, strat: str):
    """
    Compute signal using in-memory buffers and enhanced modules.
    This function should be called from main where TRADE_BUFFER and OB_CACHE exist.
    """
    # Local import to avoid circular imports
    from main import TRADE_BUFFER, OB_CACHE  # noqa: F401
    try:
        from signal_confidence import SignalConfidence, enhance_signal
        from price_levels import get_complete_levels
        from volume_analysis import get_volume_insights
        from telegram_formatter import TelegramFormatter  # if needed
        from chart_image import generate_chart_image
        from correlation_engine import correlation_with_btc
    except Exception as e:
        log.debug(f"Optional enhancement modules missing: {e}")
        SignalConfidence = None
        enhance_signal = None
        get_complete_levels = None
        get_volume_insights = None
        generate_chart_image = None
        correlation_with_btc = None

    # BTC calm check
    if sym != "BTCUSDT":
        ok = await btc_calm_check(buffer_dict=TRADE_BUFFER)
        if not ok:
            return None

    # Build MTF OHLC
    o1 = await build_ohlcv_from_trades(sym, "1min", 200, buffer_dict=TRADE_BUFFER)
    if o1 is None or len(o1) < 60:
        return None

    o5 = await build_ohlcv_from_trades(sym, "5min", 200, buffer_dict=TRADE_BUFFER)
    o15 = await build_ohlcv_from_trades(sym, "15min", 200, buffer_dict=TRADE_BUFFER)
    o60 = await build_ohlcv_from_trades(sym, "60min", 200, buffer_dict=TRADE_BUFFER)

    mtf = compute_mtf(o1, o5, o15, o60)

    # Indicators
    last = float(o1["close"].iloc[-1])
    mom1 = (last - float(o1["close"].iloc[-2])) / last
    mom5 = 0.0
    if o5 is not None and len(o5) > 2:
        mom5 = (last - float(o5["close"].iloc[-2])) / last

    rsi = float(calc_rsi(o1["close"]).iloc[-1])
    atr = float(calc_atr(o1))

    vwap = await calc_vwap_from_trades(sym, buffer_dict=TRADE_BUFFER)
    if vwap is None:
        return None

    flow = await orderflow_metrics(sym, buffer_dict=TRADE_BUFFER, ob_cache=OB_CACHE)
    if not flow:
        return None

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

    # Correlation with BTC (optional)
    try:
        if correlation_with_btc and "BTCUSDT" in TRADE_BUFFER:
            btc = TRADE_BUFFER["BTCUSDT"]
            btc_closes = [float(t["p"]) for t in list(btc)[-200:]] if btc else []
            sym_closes = list(map(float, o1["close"].values))
            if btc_closes and sym_closes:
                corr_btc = correlation_with_btc(sym_closes, btc_closes)
                params["corr_btc"] = corr_btc
    except Exception:
        params["corr_btc"] = 0.0

    # 47 Logic Evaluation
    L = evaluate_logic(params)
    score = aggregate_score(L)

    if score < STRAT_MIN.get(strat, 0):
        return None

    side = "long" if params["imb"] > 0 else "short"

    # Base signal
    signal = {
        "symbol": sym,
        "side": side,
        "score": score,
        "last": last,
        "strategy": strat,
        "logic": L,
        "passed": [k for k, v in L.items() if v == 1],
        "timestamp": datetime.utcnow().isoformat()
    }

    # Attach advanced modules if available
    # 1) Price Levels
    try:
        if get_complete_levels:
            levels = await get_complete_levels(sym, o1)
            signal["levels"] = levels
    except Exception as e:
        log.debug(f"levels error: {e}")

    # 2) Volume Insights
    try:
        if get_volume_insights:
            trades = list(TRADE_BUFFER.get(sym, []))
            volume = await get_volume_insights(o1, trades)
            signal["volume"] = volume
    except Exception as e:
        log.debug(f"volume error: {e}")

    # 3) Chart image (TradingView or generate)
    try:
        if generate_chart_image:
            chart_url = await generate_chart_image(sym, "1")
            if chart_url:
                signal["chart"] = chart_url
    except Exception as e:
        log.debug(f"chart image error: {e}")

    # 4) Correlation flags to logic
    try:
        corr_val = params.get("corr_btc", 0.0)
        L["corr_positive"] = 1 if corr_val > 0.4 else 0
        L["corr_negative"] = 1 if corr_val < -0.4 else 0
        signal["logic"] = L
    except Exception:
        pass

    # 5) Enhance with confidence (optional)
    try:
        if SignalConfidence and enhance_signal:
            # enhance_signal expects signal dict and params; adapt if signature differs
            enhanced = enhance_signal(signal, params)
            # ensure we merge fields
            if isinstance(enhanced, dict):
                signal.update(enhanced)
    except Exception as e:
        log.debug(f"confidence error: {e}")

    return signal