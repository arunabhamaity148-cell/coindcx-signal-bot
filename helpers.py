# helpers.py
import os
import json
import asyncio
import logging
from datetime import datetime
import aiohttp
import pandas as pd
import numpy as np
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

BASE = "https://public.coindcx.com/market_data"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,MATICUSDT,XRPUSDT").split(",")
TICKER_INTERVAL = float(os.getenv("TICKER_INTERVAL", 2.0))   # seconds
ORDERBOOK_INTERVAL = float(os.getenv("ORDERBOOK_INTERVAL", 6.0))  # seconds
CANDLES_INTERVAL = float(os.getenv("CANDLES_INTERVAL", 60.0))  # seconds
REDIS_PRICE_LIST_MAX = int(os.getenv("REDIS_PRICE_LIST_MAX", 2000))

# single session + redis client
_http_session: aiohttp.ClientSession | None = None
_redis: Redis | None = None

async def http_session():
    global _http_session
    if _http_session is None:
        _http_session = aiohttp.ClientSession()
    return _http_session

async def redis():
    global _redis
    if _redis is None:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
        await _redis.ping()
        log.info("✓ Redis connected")
    return _redis

# -------------------------
# Discovery helpers
# -------------------------
async def get_symbols():
    s = await http_session()
    url = f"{BASE}/symbols"
    try:
        async with s.get(url, timeout=8) as r:
            data = await r.json()
            return data
    except Exception as e:
        log.debug("symbols fetch failed: %s", e)
        return []

async def normalize_pair(query):
    """
    Try to find the exact instrument/pair string from symbols endpoint.
    Returns the exact pair string to use in other API calls, else returns query.
    """
    syms = await get_symbols()
    q = query.upper()
    for item in syms:
        text = json.dumps(item).upper()
        if q in text:
            # try common keys
            for key in ("pair", "symbol", "instrument", "id", "name"):
                if isinstance(item, dict) and key in item:
                    val = item[key]
                    if isinstance(val, str) and q in val.upper():
                        return val
            # fallback: return stringified item if contains q
            return item
    return query

# -------------------------
# CoinDCX public endpoints
# -------------------------
async def fetch_json(url):
    s = await http_session()
    try:
        async with s.get(url, timeout=8) as r:
            return await r.json()
    except Exception as e:
        log.debug("fetch_json failed %s -> %s", url, e)
        return None

async def get_ticker(pair):
    url = f"{BASE}/current_market_price/{pair}"
    data = await fetch_json(url)
    if not data:
        return None, None
    price = float(data.get("price", 0) or 0)
    ts = int(data.get("timestamp") or int(datetime.utcnow().timestamp() * 1000))
    return price, ts

async def get_orderbook(pair, depth=50):
    url = f"{BASE}/orderbook?pair={pair}&depth={depth}"
    data = await fetch_json(url)
    if not data:
        return None, None
    bids = [[float(x[0]), float(x[1])] for x in data.get("bids", [])]
    asks = [[float(x[0]), float(x[1])] for x in data.get("asks", [])]
    return bids, asks

async def get_candles(pair, interval="1m", limit=100):
    url = f"{BASE}/candles/?pair={pair}&interval={interval}&limit={limit}"
    data = await fetch_json(url)
    if not data:
        return None
    # Expecting list of [timestamp, open, high, low, close, volume] or similar
    return data

# -------------------------
# Redis store helpers (prices -> build candles locally)
# -------------------------
async def push_price(pair, price, ts):
    r = await redis()
    await r.lpush(f"px:{pair}", json.dumps({"p": price, "t": ts}))
    await r.ltrim(f"px:{pair}", 0, REDIS_PRICE_LIST_MAX)

async def get_price_history(pair, limit=2000):
    r = await redis()
    raw = await r.lrange(f"px:{pair}", 0, limit)
    if not raw:
        return None
    rows = [json.loads(x) for x in raw]
    df = pd.DataFrame(rows)
    if "t" not in df.columns or "p" not in df.columns:
        return None
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").sort_index()
    return df

async def build_ohlcv(pair, tf="1m", bars=120):
    df = await get_price_history(pair)
    if df is None or df.empty:
        return None
    ohlc = df["p"].resample(tf).ohlc().dropna()
    if ohlc is None or len(ohlc) == 0:
        return None
    return ohlc.tail(bars).rename(columns={"open":"open","high":"high","low":"low","close":"close"})

# -------------------------
# Indicators & Orderbook metrics
# -------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ob_metrics(bids, asks):
    if not bids or not asks:
        return None
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = (best_ask - best_bid) / (best_bid + 1e-12) * 100
    bid_vol = sum(b[1] for b in bids[:10])
    ask_vol = sum(a[1] for a in asks[:10])
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    depth_usd = sum(b[0]*b[1] for b in bids[:10]) + sum(a[0]*a[1] for a in asks[:10])
    return {"best_bid": best_bid, "best_ask": best_ask, "spread": spread, "imbalance": imbalance, "depth_usd": depth_usd}

async def calc_atr(pair):
    ohlc = await build_ohlcv(pair, "1m", 100)
    if ohlc is None or len(ohlc) < 20:
        return 0.001
    h = ohlc["high"]; l = ohlc["low"]; c = ohlc["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.001

# -------------------------
# Score (simple & robust)
# -------------------------
async def get_score(pair):
    # Normalize pair if needed (once)
    # pair = await normalize_pair(pair)  # optional if discovery required

    price, ts = await get_ticker(pair)
    if price is None:
        return None

    bids, asks = await get_orderbook(pair, depth=50)
    if bids is None or asks is None:
        return None

    # store instant price for candle reconstruction
    await push_price(pair, price, ts)

    ohlc = await build_ohlcv(pair, "1m", 40)
    if ohlc is None or len(ohlc) < 20:
        return None

    rsi = calc_rsi(ohlc["close"]).iloc[-1]
    atr = await calc_atr(pair)
    ob = ob_metrics(bids, asks)
    if ob is None:
        return None

    if ob["spread"] > 1.2:      # filter wide spread
        return None
    if ob["depth_usd"] < 20000:  # low depth filter (tune)
        return None

    score = 0.0
    # RSI proximity to 50 (higher is neutral bias)
    score += (50 - abs(rsi - 50)) / 50 * 3
    # orderbook imbalance weight
    score += ob["imbalance"] * 5
    # spread penalty (smaller spread -> better)
    score += max(0, (1 - ob["spread"]/0.5)) * 2
    # ATR / volatility factor (smaller atr => smoother score)
    score += max(0, 1 - (atr * 1000)) * 1

    return {
        "pair": pair,
        "price": float(price),
        "rsi": float(rsi),
        "atr": float(atr),
        "imbalance": float(ob["imbalance"]),
        "spread": float(ob["spread"]),
        "depth_usd": float(ob["depth_usd"]),
        "score": float(score)
    }

# -------------------------
# Pollers (run as background tasks)
# -------------------------
async def poll_ticker(pair):
    """
    Poll current ticker frequently and store in Redis px:{pair}
    """
    while True:
        try:
            price, ts = await get_ticker(pair)
            if price is not None:
                await push_price(pair, price, ts)
            await asyncio.sleep(TICKER_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.debug("poll_ticker error %s: %s", pair, e)
            await asyncio.sleep(2)

async def poll_orderbook(pair):
    """
    Poll orderbook periodically and store snapshot in Redis hash d:{pair}
    """
    r = await redis()
    while True:
        try:
            bids, asks = await get_orderbook(pair, depth=50)
            if bids is not None and asks is not None:
                await r.hset(f"d:{pair}", mapping={
                    "bids": json.dumps(bids[:50]),
                    "asks": json.dumps(asks[:50]),
                    "ts": int(datetime.utcnow().timestamp()*1000)
                })
            await asyncio.sleep(ORDERBOOK_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.debug("poll_orderbook error %s: %s", pair, e)
            await asyncio.sleep(3)

# -------------------------
# Cleanup
# -------------------------
async def close():
    global _http_session, _redis
    try:
        if _http_session:
            await _http_session.close()
            _http_session = None
    except: pass
    try:
        if _redis:
            await _redis.close()
            _redis = None
    except: pass

log.info("✓ helpers (CoinDCX minimal) loaded")