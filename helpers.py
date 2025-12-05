# ============================================================
# helpers.py — Binance Data + Redis Engine (FINAL v1.0)
# ============================================================

import json
import numpy as np
import pandas as pd
from redis.asyncio import Redis
from datetime import datetime

# ------------------------------
# Redis Connection
# ------------------------------
redis = Redis.from_url(
    "redis://default:redispw@localhost:6379",
    decode_responses=True
)

# ------------------------------
# 50 COIN LIST (GLOBAL)
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

# ------------------------------
# Fetch trades from Redis
# ------------------------------
async def fetch_trades(sym: str) -> list:
    data = await redis.lrange(f"tr:{sym}", 0, 500)
    if not data:
        return []
    return [json.loads(x) for x in data]


# -----------------------------------------------------------
# Build OHLCV from trades (1m/5m/15m/1h)
# -----------------------------------------------------------
async def build_ohlcv_from_trades(sym: str, interval: str, limit: int = 200):
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


# -----------------------------------------------------------
# RSI Calculation
# -----------------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# -----------------------------------------------------------
# ATR calculation
# -----------------------------------------------------------
def calc_atr(df: pd.DataFrame, period: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    previous = close.shift(1)

    tr = np.maximum(high - low,
                    np.maximum(abs(high - previous), abs(low - previous)))

    return tr.rolling(period).mean().iloc[-1]


# -----------------------------------------------------------
# VWAP from trades
# -----------------------------------------------------------
async def calc_vwap_from_trades(sym: str):
    trades = await fetch_trades(sym)
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df["p"] = df["p"].astype(float)
    df["q"] = df["q"].astype(float)
    
    total_pv = (df["p"] * df["q"]).sum()
    total_vol = df["q"].sum() + 1e-9

    return total_pv / total_vol


# -----------------------------------------------------------
# ORDERFLOW METRICS
# -----------------------------------------------------------
async def orderflow_metrics(sym: str):
    trades = await fetch_trades(sym)
    if not trades:
        return None

    df = pd.DataFrame(trades)
    df["p"] = df["p"].astype(float)
    df["q"] = df["q"].astype(float)
    df["side"] = np.where(df["m"], -1, 1)

    # delta & imbalance
    df["delta"] = df["q"] * df["side"]
    total_delta = df["delta"].sum()
    recent_delta = df["delta"].tail(40).sum()

    # depth (from orderbook)
    ob_raw = await redis.get(f"ob:{sym}")
    if ob_raw:
        ob = json.loads(ob_raw)
        bid = float(ob["bid"])
        ask = float(ob["ask"])
        spread_pct = (ask - bid) / bid * 100
        depth_usd = (bid + ask) * 50  # approximate
    else:
        spread_pct = 0.15
        depth_usd = 20000

    return {
        "delta": float(total_delta),
        "recent_delta": float(recent_delta),
        "imbalance": float(recent_delta / (abs(total_delta) + 1e-9)),
        "spread_pct": spread_pct,
        "depth_usd": depth_usd
    }


# -----------------------------------------------------------
# BTC CALM CHECK
# -----------------------------------------------------------
async def btc_calm_check(threshold=0.35):
    trades = await fetch_trades("BTCUSDT")
    if len(trades) < 20:
        return True

    df = pd.DataFrame(trades)
    prices = df["p"].astype(float)

    vol = (prices.max() - prices.min()) / prices.mean() * 100

    return vol < threshold


# -----------------------------------------------------------
# TICKER FETCH (LTP)
# -----------------------------------------------------------
async def get_last_price(sym: str):
    tk = await redis.get(f"tk:{sym}")
    if not tk:
        return None
    data = json.loads(tk)
    return float(data["last"])