# helpers.py
# Async helpers: OHLC builder, RSI, ATR, VWAP, Orderflow, BTC calm, TP/SL, Iceberg, Redis utils
# Requires: aiohttp, redis, pandas, numpy
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
from redis.asyncio import Redis

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis: Optional[Redis] = None

# Default config (tune via env)
DEFAULTS = {
    "RISK_PERC": float(os.getenv("RISK_PERC", "0.7")),    # percent of equity risk
    "EQUITY_USD": float(os.getenv("EQUITY_USD", "30000")), 
    "MIN_LEV": int(os.getenv("MIN_LEV", "15")),
    "MAX_LEV": int(os.getenv("MAX_LEV", "30")),
    "COOLDOWN_MIN": int(os.getenv("COOLDOWN_MIN", "30")),
    "BTC_CALM_PCT": float(os.getenv("BTC_CALM_PCT", "0.25")),  # percent momentum threshold
    "PRICE_TOL_PCT": float(os.getenv("PRICE_TOL_PCT", "0.35")), # for consensus
}


async def redis_client() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            await _redis.ping()
            log.info("✓ Redis connected")
        except Exception as e:
            log.error("Redis connect failed: %s", e)
            raise
    return _redis


# ---------------------------
# Raw data readers (from binance_ws keys)
# ---------------------------
async def get_trades(sym: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Read trades list stored at tr:{sym} (most recent first).
    Each trade is json: {"p": price, "q": qty, "m": is_sell, "t": ts}
    Returns list ordered oldest -> newest (for pandas convenience).
    """
    r = await redis_client()
    raw = await r.lrange(f"tr:{sym}", 0, limit - 1)
    if not raw:
        return []
    # incoming list is newest-first; reverse to oldest-first
    rows = [json.loads(x) for x in raw[::-1]]
    return rows


async def get_orderbook_snapshot(sym: str) -> Optional[Dict[str, Any]]:
    """
    Read orderbook top stored at ob:{sym} produced by binance_ws.
    Returns dict with bid, ask, t
    """
    r = await redis_client()
    val = await r.get(f"ob:{sym}")
    if not val:
        return None
    try:
        return json.loads(val)
    except:
        return None


async def get_ticker(sym: str) -> Optional[Dict[str, Any]]:
    """
    Get ticker saved at tk:{sym} (close for BTC or others)
    """
    r = await redis_client()
    val = await r.get(f"tk:{sym}")
    if not val:
        return None
    try:
        return json.loads(val)
    except:
        return None


# ---------------------------
# OHLC builder from trades
# ---------------------------
async def build_ohlcv_from_trades(sym: str, tf: str = "1min", limit: int = 200) -> Optional[pd.DataFrame]:
    """
    Build OHLCV using trades list stored in Redis.
    tf: pandas-compatible frequency e.g. "1min", "5min"
    Returns DataFrame with columns open, high, low, close, volume indexed by timestamp (UTC)
    """
    trades = await get_trades(sym, limit=2000)
    if not trades:
        return None
    # Create DataFrame
    df = pd.DataFrame(trades)
    if "p" not in df.columns or "q" not in df.columns or "t" not in df.columns:
        return None
    # convert
    df["p"] = df["p"].astype(float)
    df["q"] = df["q"].astype(float)
    # timestamps likely in ms
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").sort_index()
    ohlc = df["p"].resample(tf).ohlc().dropna()
    vol = df["q"].resample(tf).sum().rename("volume")
    out = ohlc.join(vol).dropna()
    if out.empty:
        return None
    out = out.tail(limit)
    # rename columns to standard
    out.columns = ["open", "high", "low", "close", "volume"]
    return out


# ---------------------------
# Indicators
# ---------------------------
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(ohlc: pd.DataFrame, period: int = 14) -> float:
    """
    Requires DataFrame with high, low, close columns
    returns last ATR value (float)
    """
    if ohlc is None or len(ohlc) < period + 2:
        return 0.001
    h = ohlc["high"]
    l = ohlc["low"]
    c = ohlc["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.001


async def calc_vwap_from_trades(sym: str) -> Optional[float]:
    """
    VWAP using available trades snapshot (recent). If insufficient, returns None
    """
    trades = await get_trades(sym, limit=500)
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["p"] = df["p"].astype(float)
    df["q"] = df["q"].astype(float)
    pv = (df["p"] * df["q"]).sum()
    vol = df["q"].sum()
    if vol == 0:
        return None
    return float(pv / vol)


async def orderflow_metrics(sym: str, depth_levels: int = 10) -> Optional[Dict[str, Any]]:
    """
    Compute imbalance, recent delta, depth estimation from trades and orderbook.
    Returns: {'imbalance':..., 'delta':..., 'recent_delta':..., 'depth_usd':...}
    """
    # from trades
    trades = await get_trades(sym, limit=500)
    if not trades:
        return None
    vols = [float(x["q"]) for x in trades]
    avg, std = (np.mean(vols), np.std(vols)) if vols else (0, 0)
    delta = sum(float(x["q"]) if not x.get("m", False) else -float(x["q"]) for x in trades[-200:])  # net buy-vol recent
    recent = sum(float(x["q"]) if not x.get("m", False) else -float(x["q"]) for x in trades[-40:])
    # depth estimate from orderbook snapshot (best 10)
    ob = await get_orderbook_snapshot(sym)
    depth_usd = 0.0
    imbalance = 0.0
    spread = None
    if ob:
        bid = ob.get("bid")
        ask = ob.get("ask")
        try:
            bid_price = float(bid)
            ask_price = float(ask)
            spread = (ask_price - bid_price) / (bid_price + 1e-12) * 100
        except:
            spread = None
        # We don't have levels list in this minimal ob; depth estimation not precise
        # but we can estimate by sampling recent trades value as proxy
        depth_usd = float(sum(float(x["p"]) * float(x["q"]) for x in trades[-40:]))  # approx
        # use trade-side imbalance as proxy
        buy_vol = sum(float(x["q"]) for x in trades[-200:] if not x.get("m", False))
        sell_vol = sum(float(x["q"]) for x in trades[-200:] if x.get("m", False))
        if buy_vol + sell_vol > 0:
            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)
    return {"delta": float(delta), "recent_delta": float(recent), "imbalance": float(imbalance), "spread_pct": spread, "depth_usd": float(depth_usd)}


# ---------------------------
# BTC calm check
# ---------------------------
async def btc_calm_check(threshold_pct: Optional[float] = None) -> bool:
    """
    Return True if BTC is calm (within threshold momentum on 1m)
    threshold_pct in percent (e.g., 0.25)
    """
    if threshold_pct is None:
        threshold_pct = DEFAULTS["BTC_CALM_PCT"]
    ohlc = await build_ohlcv_from_trades("BTCUSDT", tf="1min", limit=6)
    if ohlc is None or len(ohlc) < 3:
        # Not enough data -> be conservative and return False (not calm)
        log.debug("btc_calm_check: insufficient ohlc")
        return False
    # compute percent momentum over last 1 bar
    last = float(ohlc["close"].iloc[-1])
    prev = float(ohlc["close"].iloc[-2])
    pct = abs((last - prev) / (prev + 1e-12)) * 100
    log.debug("BTC momentum pct: %.4f", pct)
    return pct <= threshold_pct


# ---------------------------
# TP / SL & leverage & iceberg
# ---------------------------
def suggest_leverage(liq_dist_pct: float) -> int:
    """
    Suggest leverage by approximate liquidation distance (in percent).
    Liq_dist_pct = absolute percent distance between SL and theoretical liquidation price.
    """
    if liq_dist_pct >= 10.0:
        return DEFAULTS["MAX_LEV"]
    if liq_dist_pct >= 7.0:
        return max(DEFAULTS["MIN_LEV"], min(DEFAULTS["MAX_LEV"], 25))
    if liq_dist_pct >= 5.0:
        return max(DEFAULTS["MIN_LEV"], 20)
    if liq_dist_pct >= 3.5:
        return 18
    return DEFAULTS["MIN_LEV"]


async def calc_tp_sl(entry: float, side: str, strategy_cfg: Dict[str, float]) -> Tuple[float, float, float, int, float]:
    """
    Calculate tp1, tp2, sl, recommended leverage, and liq_dist_pct.
    strategy_cfg expects keys: tp1_mult, tp2_mult, sl_mult
    Returns: tp1, tp2, sl, lev, liq_dist_pct
    """
    # use ATR as volatility unit
    # For simplicity, user will pass a symbol-specific atr externally if needed. We'll fetch ATR for estimation when possible.
    # Here we assume entry is not None and strategy_cfg contains mult params.
    # default lev
    lev = DEFAULTS["MIN_LEV"]
    # approximate sl distance using a percent of entry via atr-like fallback
    # we'll use a small fallback base distance if ATR unavailable
    # Try to compute ATR from the entry symbol context is caller's responsibility; here we assume caller passed strategy_cfg with sl_mult representing multiplier of typical tick
    base_pct = strategy_cfg.get("sl_base_pct", 0.9)  # percent of price as base SL (fallback)
    sl_dist = (base_pct / 100.0) * entry * strategy_cfg.get("sl_mult", 1.0)
    if side.lower() == "long":
        sl = entry - sl_dist
        tp1 = entry + sl_dist * strategy_cfg.get("tp1_mult", 1.2)
        tp2 = entry + sl_dist * strategy_cfg.get("tp2_mult", 1.8)
    else:
        sl = entry + sl_dist
        tp1 = entry - sl_dist * strategy_cfg.get("tp1_mult", 1.2)
        tp2 = entry - sl_dist * strategy_cfg.get("tp2_mult", 1.8)

    # compute naive liquidation price estimation and liq distance percent
    # approximate liq as entry * (1 - 1/lev) for long; we'll choose a conservative lev first
    lev = DEFAULTS["MIN_LEV"]
    liq = entry * (1 - 1.0 / lev) if side.lower() == "long" else entry * (1 + 1.0 / lev)
    liq_dist_pct = abs((sl - liq) / (liq + 1e-12)) * 100
    lev = suggest_leverage(liq_dist_pct, )  # type: ignore
    # recompute liq_dist with suggested lev
    liq = entry * (1 - 1.0 / lev) if side.lower() == "long" else entry * (1 + 1.0 / lev)
    liq_dist_pct = abs((sl - liq) / (liq + 1e-12)) * 100

    return tp1, tp2, sl, lev, liq_dist_pct


def iceberg_size(equity_usd: float, entry: float, sl: float, lev: int) -> Dict[str, Any]:
    """
    Compute iceberg orders: total qty, per-slice qty, number of slices (4).
    qty based on risk percent DEFAULTS["RISK_PERC"].
    """
    risk = equity_usd * (DEFAULTS["RISK_PERC"] / 100.0)
    dist = abs(entry - sl)
    if dist <= 0:
        return {"total": 0.0, "each": 0.0, "orders": 0}
    qty_base = (risk / dist) / lev
    iq = qty_base / 4.0
    return {"total": float(qty_base), "each": float(iq), "orders": 4}


# ---------------------------
# Utility: Scoring helpers & safe checks used by scanner
# ---------------------------
def score_from_components(rsi: float, imbalance: float, spread_pct: float, atr: float) -> float:
    """
    Combine standardized components into a score.
    This is a simple, tunable aggregator — scorer.py will apply strategy thresholds.
    """
    score = 0.0
    # RSI proximity to 50 (neutral) is positive (higher = better)
    score += (50.0 - abs(rsi - 50.0)) / 50.0 * 3.0
    # imbalance: buy pressure positive -> add
    score += imbalance * 5.0
    # spread penalty
    score += max(0.0, (1.0 - (spread_pct / 0.5))) * 2.0
    # volatility penalty (smaller atr -> more stable)
    score += max(0.0, 1.0 - (atr * 1000.0)) * 1.0
    return float(score)


async def cdx_safe_checks_placeholder(*args, **kwargs) -> bool:
    """
    Placeholder for CoinDCX safety checks if you later add hybrid.
    Keep signature compatible: can be awaited in scanner.
    """
    return True


# ---------------------------
# Cleanup / close
# ---------------------------
async def close_redis():
    global _redis
    try:
        if _redis:
            await _redis.close()
            _redis = None
            log.info("Redis closed")
    except Exception as e:
        log.debug("close_redis error %s", e)


# ---------------------------
# Quick test CLI (optional)
# ---------------------------
if __name__ == "__main__":
    # quick smoke test if you run helpers directly (requires redis running + binance_ws feeding)
    async def _test():
        ok = await redis_client()
        print("redis ok")
        ohlc = await build_ohlcv_from_trades("BTCUSDT", tf="1min", limit=40)
        if ohlc is not None:
            print("OHLC rows:", len(ohlc))
            print(ohlc.tail(3))
        else:
            print("No OHLC")
        calm = await btc_calm_check()
        print("BTC calm:", calm)
        of = await orderflow_metrics("BTCUSDT")
        print("orderflow:", of)

    asyncio.run(_test())