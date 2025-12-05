# ============================================================
# scorer.py — ADVANCED SCORING ENGINE (Quick / Mid / Trend)
# ============================================================

import asyncio
import logging
from typing import Dict, Any, Optional

from helpers import (
    build_ohlcv_from_trades,
    calc_rsi,
    calc_atr,
    calc_vwap_from_trades,
    orderflow_metrics,
    btc_calm_check,
    score_from_components,
)

log = logging.getLogger("scorer")


# STRATEGY CONFIG
STRATEGY = {
    "QUICK": {
        "min_score": 6.2,
        "tp1_mult": 1.2,
        "tp2_mult": 1.8,
        "sl_mult": 0.8,
    },
    "MID": {
        "min_score": 6.6,
        "tp1_mult": 1.5,
        "tp2_mult": 2.3,
        "sl_mult": 1.0,
    },
    "TREND": {
        "min_score": 7.4,
        "tp1_mult": 2.0,
        "tp2_mult": 3.2,
        "sl_mult": 1.2,
    }
}


# -----------------------------
# CORE SCORE BUILDER
# -----------------------------
async def compute_score(sym: str, strategy: str) -> Optional[Dict[str, Any]]:
    """
    Build complete scoring result for a symbol.
    Returns None if insufficient data OR filters fail.
    Output:
    {
        'side': 'long/short',
        'score': float,
        'entry': price,
        'done': 'RSI|VWAP|FLOW|...',
        ...
    }
    """

    # 1️⃣ BTC calm check (mandatory)
    calm = await btc_calm_check()
    if not calm:
        log.debug(f"{sym} blocked → BTC not calm")
        return None

    # 2️⃣ Build OHLC
    ohlc = await build_ohlcv_from_trades(sym, tf="1min", limit=120)
    if ohlc is None or len(ohlc) < 30:
        log.debug(f"{sym} no OHLC data")
        return None

    close = ohlc["close"]
    last = float(close.iloc[-1])

    # 3️⃣ Indicators
    # RSI
    rsi = float(calc_rsi(close).iloc[-1] or 50.0)

    # ATR
    atr = float(calc_atr(ohlc, period=14))

    # VWAP
    vwap = await calc_vwap_from_trades(sym)
    if vwap is None:
        return None

    # Orderflow
    flow = await orderflow_metrics(sym)
    if not flow:
        return None

    imb = flow["imbalance"]
    spread = flow["spread_pct"] if flow["spread_pct"] else 0.2
    depth = flow["depth_usd"]

    # 4️⃣ Filters
    passed = []

    # RSI filter
    if abs(rsi - 50) < 25:
        passed.append("RSI")

    # VWAP filter
    if abs(last - vwap) / vwap < 0.0025:
        passed.append("VWAP")

    # Imbalance filter
    if abs(imb) > 0.05:
        passed.append("Flow")

    # Depth filter
    if depth > 30000:
        passed.append("Depth")

    # Spread filter
    if spread < 0.3:
        passed.append("Spread")

    # No filters → reject
    if len(passed) == 0:
        return None

    # 5️⃣ Side detection
    side = "long" if imb > 0 else "short"

    # 6️⃣ Composite Score
    score = score_from_components(rsi, imb, spread, atr)

    # 7️⃣ Strategy threshold
    req = STRATEGY[strategy]["min_score"]
    if score < req:
        return None

    # 8️⃣ Return final dict
    return {
        "symbol": sym,
        "side": side,
        "score": float(score),
        "entry": last,
        "rsi": rsi,
        "vwap": vwap,
        "imb": imb,
        "spread": spread,
        "depth": depth,
        "done": " | ".join(passed),
        "strategy_cfg": STRATEGY[strategy],
    }