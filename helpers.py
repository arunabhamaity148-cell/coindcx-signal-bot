# helpers.py — Optimized (Redis off by default, buffer-first)
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

log = logging.getLogger("helpers")

# ------------------------------
# Redis (optional)
# ------------------------------
REDIS_URL = os.getenv("REDIS_URL", "")  # empty => no redis
redis = None
USE_REDIS = bool(REDIS_URL)

if USE_REDIS:
    try:
        from redis.asyncio import Redis
        redis = Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5, socket_keepalive=True, health_check_interval=30)
        log.info("helpers: Redis enabled")
    except Exception as e:
        log.warning(f"helpers: Redis import/connect failed, proceeding without Redis: {e}")
        redis = None
        USE_REDIS = False
else:
    log.info("helpers: Redis disabled (REDIS_URL not set)")

# ------------------------------
# PAIRS (example list; keep in sync with main.PAIRS)
# ------------------------------
PAIRS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","TRXUSDT",
    "MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","LTCUSDT","ETCUSDT","FILUSDT",
    "APTUSDT","ARBUSDT","NEARUSDT","OPUSDT","SUIUSDT","INJUSDT","RUNEUSDT","AXSUSDT",
]

# ------------------------------
# Fetch trades (buffer-first, redis-fallback)
# ------------------------------
async def fetch_trades(sym: str, buffer_dict: dict = None, max_items: int = 500) -> list:
    """
    Return list of trades for sym.
    - Prefer buffer_dict (in-memory) if provided
    - Fallback to Redis if configured
    Each trade expected: {"p": price, "q": qty, "m": bool, "t": timestamp_ms}
    """
    try:
        # 1) buffer-first
        if buffer_dict and sym in buffer_dict:
            trades = list(buffer_dict[sym])
            if trades:
                # return up to max_items most recent
                return trades[-max_items:]
        # 2) redis fallback (if enabled)
        if USE_REDIS and redis:
            try:
                raw = await redis.lrange(f"tr:{sym}", 0, max_items-1)
                trades = []
                for item in raw:
                    try:
                        t = json.loads(item)
                        if all(k in t for k in ("p","q","m","t")):
                            trades.append(t)
                    except:
                        continue
                # redis lrange returns newest-first if stored that way; keep as-is
                return trades[-max_items:]
            except Exception as e:
                log.debug(f"helpers.fetch_trades: redis read failed for {sym}: {e}")
                return []
        return []
    except Exception as e:
        log.error(f"helpers.fetch_trades error for {sym}: {e}")
        return []

# ------------------------------
# Build OHLCV from trades
# ------------------------------
async def build_ohlcv_from_trades(sym: str, interval: str, limit: int = 200, buffer_dict: dict = None):
    """
    Build OHLCV pandas DataFrame from available trades.
    interval: '1min','5min','15min','60min'
    """
    try:
        trades = await fetch_trades(sym, buffer_dict, max_items=limit*8)
        if not trades or len(trades) < 10:
            return None

        df = pd.DataFrame(trades)
        # ensure columns
        if "t" not in df.columns or "p" not in df.columns or "q" not in df.columns:
            return None

        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)
        df = df.set_index("t").sort_index()

        rule = {"1min": "1T", "5min": "5T", "15min": "15T", "60min": "60T"}.get(interval, "1T")
        ohlc = df["p"].resample(rule).ohlc()
        vol = df["q"].resample(rule).sum()
        final = ohlc.join(vol.rename("vol")).dropna()
        if final.empty:
            return None
        return final.tail(limit)
    except Exception as e:
        log.error(f"build_ohlcv_from_trades error for {sym}/{interval}: {e}")
        return None

# ------------------------------
# Indicators: RSI, ATR, VWAP
# ------------------------------
def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        log.error(f"calc_rsi error: {e}")
        return None

def calc_atr(df: pd.DataFrame, period: int = 14):
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        previous = close.shift(1)
        tr = np.maximum(high - low, np.maximum(abs(high - previous), abs(low - previous)))
        return tr.rolling(period).mean().iloc[-1]
    except Exception as e:
        log.error(f"calc_atr error: {e}")
        return float(0)

async def calc_vwap_from_trades(sym: str, buffer_dict: dict = None):
    try:
        trades = await fetch_trades(sym, buffer_dict)
        if not trades:
            return None
        df = pd.DataFrame(trades)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)
        total_pv = (df["p"] * df["q"]).sum()
        total_vol = df["q"].sum() + 1e-9
        return float(total_pv / total_vol)
    except Exception as e:
        log.error(f"calc_vwap_from_trades error for {sym}: {e}")
        return None

# ------------------------------
# Orderflow metrics (buffer-first, ob_cache fallback)
# ------------------------------
async def orderflow_metrics(sym: str, buffer_dict: dict = None, ob_cache: dict = None):
    """
    Return:
      - delta: total signed qty
      - recent_delta: last N trades delta
      - imbalance: recent_delta / abs(total_delta)
      - spread_pct, depth_usd
    """
    try:
        trades = await fetch_trades(sym, buffer_dict, max_items=500)
        if not trades:
            return None
        df = pd.DataFrame(trades)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)
        if "m" not in df.columns:
            df["m"] = False
        df["side"] = np.where(df["m"], -1, 1)
        df["delta"] = df["q"] * df["side"]
        total_delta = float(df["delta"].sum())
        recent_delta = float(df["delta"].tail(40).sum())

        # try ob_cache first
        if ob_cache and sym in ob_cache:
            ob = ob_cache[sym]
            try:
                bid = float(ob.get("bid", 0.0))
                ask = float(ob.get("ask", bid))
                spread_pct = (ask - bid) / max(bid, 1e-9) * 100
                depth_usd = (bid + ask) * 50
            except:
                spread_pct = 0.15
                depth_usd = 20000
        else:
            # redis fallback for orderbook if present
            if USE_REDIS and redis:
                try:
                    ob_raw = await redis.get(f"ob:{sym}")
                    if ob_raw:
                        ob = json.loads(ob_raw)
                        bid = float(ob.get("bid", 0.0))
                        ask = float(ob.get("ask", bid))
                        spread_pct = (ask - bid) / max(bid, 1e-9) * 100
                        depth_usd = (bid + ask) * 50
                    else:
                        spread_pct = 0.15
                        depth_usd = 20000
                except Exception:
                    spread_pct = 0.15
                    depth_usd = 20000
            else:
                spread_pct = 0.15
                depth_usd = 20000

        imbalance = float(recent_delta / (abs(total_delta) + 1e-9)) if (abs(total_delta) > 0 or recent_delta != 0) else 0.0

        return {
            "delta": total_delta,
            "recent_delta": recent_delta,
            "imbalance": imbalance,
            "spread_pct": spread_pct,
            "depth_usd": depth_usd
        }
    except Exception as e:
        log.error(f"orderflow_metrics error for {sym}: {e}")
        return None

# ------------------------------
# BTC calm check (buffer-first)
# ------------------------------
async def btc_calm_check(buffer_dict: dict = None, threshold: float = 0.35):
    """
    Return True if BTC volatility (recent trades) < threshold (percent)
    """
    try:
        trades = await fetch_trades("BTCUSDT", buffer_dict, max_items=500)
        if not trades or len(trades) < 10:
            return True
        df = pd.DataFrame(trades)
        prices = df["p"].astype(float)
        vol = (prices.max() - prices.min()) / (prices.mean() + 1e-9) * 100
        return vol < threshold
    except Exception as e:
        log.debug(f"btc_calm_check error: {e}")
        return True

# ------------------------------
# Last price (ticker cache first)
# ------------------------------
async def get_last_price(sym: str, ticker_cache: dict = None):
    try:
        if ticker_cache and sym in ticker_cache:
            return float(ticker_cache[sym].get("last", 0.0))
        if USE_REDIS and redis:
            try:
                tk = await redis.get(f"tk:{sym}")
                if tk:
                    data = json.loads(tk)
                    return float(data.get("last", 0.0))
            except:
                return None
        return None
    except:
        return None