# ============================================================
# helpers.py — Binance Data + Redis Engine (Error Safe)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from redis.asyncio import Redis
from datetime import datetime
import logging

log = logging.getLogger("helpers")

# ------------------------------
# Redis Connection
# ------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis = Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
    health_check_interval=30
)

# ------------------------------
# PAIRS
# ------------------------------
PAIRS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","TRXUSDT",
    "MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","LTCUSDT","ETCUSDT","FILUSDT",
    "APTUSDT","ARBUSDT","NEARUSDT","OPUSDT","SUIUSDT","INJUSDT","RUNEUSDT","AXSUSDT",
    "GRTUSDT","FLUXUSDT","1INCHUSDT","IMXUSDT","STXUSDT","FETUSDT","MASKUSDT","TWTUSDT",
    "CHRUSDT","BLURUSDT","DYDXUSDT","RNDRUSDT","SANDUSDT","MANAUSDT","KAVAUSDT","ZECUSDT",
    "CRVUSDT","YFIUSDT","COMPUSDT","UNIUSDT","ALGOUSDT","HBARUSDT","VETUSDT","FTMUSDT",
    "WAVESUSDT"
]


async def fetch_trades(sym: str) -> list:
    try:
        data = await redis.lrange(f"tr:{sym}", 0, 500)
        if not data:
            return []
        
        trades = []
        for item in data:
            try:
                trade = json.loads(item)
                # Validate structure
                if all(k in trade for k in ["p", "q", "m", "t"]):
                    trades.append(trade)
            except:
                continue
        
        return trades
    except Exception as e:
        log.error(f"Fetch trades error for {sym}: {e}")
        return []


async def build_ohlcv_from_trades(sym: str, interval: str, limit: int = 200):
    try:
        trades = await fetch_trades(sym)
        if len(trades) < 20:
            return None

        df = pd.DataFrame(trades)
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)
        df = df.set_index("t")

        rule = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "60min": "60T"
        }.get(interval, "1T")

        ohlc = df["p"].resample(rule).ohlc()
        vol = df["q"].resample(rule).sum()

        final = ohlc.join(vol.rename("vol")).dropna()
        return final.tail(limit)
    except Exception as e:
        log.error(f"OHLCV build error for {sym}/{interval}: {e}")
        return None


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    previous = close.shift(1)

    tr = np.maximum(high - low,
                    np.maximum(abs(high - previous), abs(low - previous)))

    return tr.rolling(period).mean().iloc[-1]


async def calc_vwap_from_trades(sym: str):
    try:
        trades = await fetch_trades(sym)
        if not trades:
            return None

        df = pd.DataFrame(trades)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)

        total_pv = (df["p"] * df["q"]).sum()
        total_vol = df["q"].sum() + 1e-9

        return total_pv / total_vol
    except Exception as e:
        log.error(f"VWAP calc error for {sym}: {e}")
        return None


async def orderflow_metrics(sym: str):
    try:
        trades = await fetch_trades(sym)
        if not trades:
            return None

        df = pd.DataFrame(trades)
        df["p"] = df["p"].astype(float)
        df["q"] = df["q"].astype(float)
        df["side"] = np.where(df["m"], -1, 1)

        df["delta"] = df["q"] * df["side"]
        total_delta = df["delta"].sum()
        recent_delta = df["delta"].tail(40).sum()

        try:
            ob_raw = await redis.get(f"ob:{sym}")
            if ob_raw:
                ob = json.loads(ob_raw)
                bid = float(ob["bid"])
                ask = float(ob["ask"])
                spread_pct = (ask - bid) / bid * 100
                depth_usd = (bid + ask) * 50
            else:
                spread_pct = 0.15
                depth_usd = 20000
        except:
            spread_pct = 0.15
            depth_usd = 20000

        return {
            "delta": float(total_delta),
            "recent_delta": float(recent_delta),
            "imbalance": float(recent_delta / (abs(total_delta) + 1e-9)),
            "spread_pct": spread_pct,
            "depth_usd": depth_usd
        }
    except Exception as e:
        log.error(f"Orderflow error for {sym}: {e}")
        return None


async def btc_calm_check(threshold=0.35):
    try:
        trades = await fetch_trades("BTCUSDT")
        if len(trades) < 20:
            return True

        df = pd.DataFrame(trades)
        prices = df["p"].astype(float)

        vol = (prices.max() - prices.min()) / prices.mean() * 100

        return vol < threshold
    except:
        return True


async def get_last_price(sym: str):
    try:
        tk = await redis.get(f"tk:{sym}")
        if not tk:
            return None
        data = json.loads(tk)
        return float(data["last"])
    except:
        return None