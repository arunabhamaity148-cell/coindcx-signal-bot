# helpers.py (CoinDCX minimal) — with symbol discovery & normalization
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
# initial PAIRS (UI-style) — these will be normalized on startup if possible
PAIRS = os.getenv("PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT,MATICUSDT,XRPUSDT").split(",")
TICKER_INTERVAL = float(os.getenv("TICKER_INTERVAL", 2.0))   # seconds
ORDERBOOK_INTERVAL = float(os.getenv("ORDERBOOK_INTERVAL", 6.0))  # seconds
CANDLES_INTERVAL = float(os.getenv("CANDLES_INTERVAL", 60.0))  # seconds
REDIS_PRICE_LIST_MAX = int(os.getenv("REDIS_PRICE_LIST_MAX", 2000))

# single session + redis client
_http_session: aiohttp.ClientSession | None = None
_redis: Redis | None = None

# symbol discovery cache
_SYMBOLS_CACHE = None
_NORMALIZED_MAP = {}   # e.g. {"BTCUSDT": "B-BTC_USDT"}

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
async def fetch_json(url):
    s = await http_session()
    try:
        async with s.get(url, timeout=12) as r:
            return await r.json()
    except Exception as e:
        log.debug("fetch_json failed %s -> %s", url, e)
        return None

async def get_symbols():
    global _SYMBOLS_CACHE
    if _SYMBOLS_CACHE is not None:
        return _SYMBOLS_CACHE
    url = f"{BASE}/symbols"
    data = await fetch_json(url)
    if not data:
        _SYMBOLS_CACHE = []
    else:
        _SYMBOLS_CACHE = data
    log.info("✓ symbols loaded: %d", len(_SYMBOLS_CACHE) if _SYMBOLS_CACHE else 0)
    return _SYMBOLS_CACHE

def _item_to_text(item):
    try:
        return json.dumps(item).upper()
    except:
        return str(item).upper()

async def find_matching_symbols(query):
    """
    Return list of symbol items from /symbols that likely match 'query' (case-insensitive).
    Query can be 'BTCUSDT' or 'BTC/USDT' etc.
    """
    syms = await get_symbols()
    q = query.replace("/", "").replace("-", "").replace("_", "").upper()
    matches = []
    for item in syms:
        text = _item_to_text(item)
        if q in text:
            matches.append(item)
    return matches

async def normalize_pair_once(ui_pair):
    """
    Try to map a UI-style pair like 'BTCUSDT' -> API instrument like 'B-BTC_USDT' or 'BTC_USDT_PERP', etc.
    Returns normalized string (if found) else returns original ui_pair.
    Caches results in _NORMALIZED_MAP.
    """
    if ui_pair in _NORMALIZED_MAP:
        return _NORMALIZED_MAP[ui_pair]

    syms = await get_symbols()
    q = ui_pair.replace("/", "").replace("-", "").replace("_", "").upper()
    # first pass: try common keys and exact-like matches
    for item in syms:
        if isinstance(item, dict):
            for k in ("pair", "symbol", "instrument", "id", "name"):
                if k in item and isinstance(item[k], str):
                    val = item[k].upper()
                    if q == val.replace("-", "").replace("_", "").replace("/", ""):
                        _NORMALIZED_MAP[ui_pair] = item[k]
                        return item[k]
            # fallback: check if UI q appears anywhere inside the JSON text
            txt = json.dumps(item).upper()
            if q in txt:
                # try to prefer 'pair' or 'symbol' key if available
                for prefer in ("pair", "symbol", "instrument", "id", "name"):
                    if prefer in item and isinstance(item[prefer], str):
                        _NORMALIZED_MAP[ui_pair] = item[prefer]
                        return item[prefer]
                # else return stringified item (not ideal)
                _NORMALIZED_MAP[ui_pair] = txt
                return txt
        else:
            # item may already be a string
            try:
                if isinstance(item, str) and q in item.upper():
                    _NORMALIZED_MAP[ui_pair] = item
                    return item
            except:
                pass

    # if nothing matched, just keep ui_pair as-is
    _NORMALIZED_MAP[ui_pair] = ui_pair
    return ui_pair

async def normalize_all_pairs():
    """
    Normalize global PAIRS list and return normalized list.
    """
    normalized = []
    for p in PAIRS:
        npair = await normalize_pair_once(p)
        normalized.append(npair)
        if npair != p:
            log.info("Normalized %s -> %s", p, npair)
    return normalized

# -------------------------
# CoinDCX public endpoints (wrapper)
# -------------------------
async def get_ticker(pair):
    url = f"{BASE}/current_market_price/{pair}"
    data = await fetch_json(url)
    if not data:
        return None, None
    # Some endpoints return nested dict, try to handle both
    price = None
    ts = None
    if isinstance(data, dict):
        price = float(data.get("price") or data.get("last") or 0)
        ts = int(data.get("timestamp") or int(datetime.utcnow().timestamp() * 1000))
    else:
        try:
            price = float(data[0].get("price") or 0)
            ts = int(data[0].get("timestamp") or int(datetime.utcnow().timestamp() * 1000))
        except:
            price, ts = None, None
    return price, ts

async def get_orderbook(pair, depth=50):
    url = f"{BASE}/orderbook?pair={pair}&depth={depth}"
    data = await fetch_json(url)
    if not data:
        return None, None
    bids = [[float(x[0]), float(x[1])] for x in data.get("bids", [])] if isinstance(data, dict) else []
    asks = [[float(x[0]), float(x[1])] for x in data.get("asks", [])] if isinstance(data, dict) else []
    return bids, asks

async def get_candles(pair, interval="1m", limit=100):
    url = f"{BASE}/candles/?pair={pair}&interval={interval}&limit={limit}"
    data = await fetch_json(url)
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
    # Normalize if we have mapping
    normalized_pair = await normalize_pair_once(pair)

    price, ts = await get_ticker(normalized_pair)
    if price is None:
        log.debug("get_score: no ticker for %s (normalized %s)", pair, normalized_pair)
        return None

    bids, asks = await get_orderbook(normalized_pair, depth=50)
    if bids is None or asks is None or len(bids) == 0 or len(asks) == 0:
        log.debug("get_score: no orderbook for %s (normalized %s)", pair, normalized_pair)
        return None

    # store instant price for candle reconstruction (use normalized key so px:<normalized> used)
    await push_price(normalized_pair, price, ts)

    ohlc = await build_ohlcv(normalized_pair, "1m", 40)
    if ohlc is None or len(ohlc) < 20:
        log.debug("get_score: insufficient ohlc for %s (normalized %s) len=%s", pair, normalized_pair, None if ohlc is None else len(ohlc))
        return None

    rsi = calc_rsi(ohlc["close"]).iloc[-1]
    atr = await calc_atr(normalized_pair)
    ob = ob_metrics(bids, asks)
    if ob is None:
        return None

    # filters (tuneable)
    if ob["spread"] > 1.2:
        return None
    if ob["depth_usd"] < 20000:
        return None

    score = 0.0
    score += (50 - abs(rsi - 50)) / 50 * 3
    score += ob["imbalance"] * 5
    score += max(0, (1 - ob["spread"]/0.5)) * 2
    score += max(0, 1 - (atr * 1000)) * 1

    return {
        "pair": normalized_pair,
        "orig_pair": pair,
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
    Poll current ticker frequently and store in Redis px:{normalized_pair}
    """
    # normalize once for stable key usage
    normalized = await normalize_pair_once(pair)
    while True:
        try:
            price, ts = await get_ticker(normalized)
            if price is not None:
                await push_price(normalized, price, ts)
            await asyncio.sleep(TICKER_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.debug("poll_ticker error %s (%s): %s", pair, normalized, e)
            await asyncio.sleep(2)

async def poll_orderbook(pair):
    """
    Poll orderbook periodically and store snapshot in Redis hash d:{normalized_pair}
    """
    normalized = await normalize_pair_once(pair)
    r = await redis()
    while True:
        try:
            bids, asks = await get_orderbook(normalized, depth=50)
            if bids is not None and asks is not None and len(bids) and len(asks):
                await r.hset(f"d:{normalized}", mapping={
                    "bids": json.dumps(bids[:50]),
                    "asks": json.dumps(asks[:50]),
                    "ts": int(datetime.utcnow().timestamp()*1000)
                })
            await asyncio.sleep(ORDERBOOK_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.debug("poll_orderbook error %s (%s): %s", pair, normalized, e)
            await asyncio.sleep(3)

# -------------------------
# Utility: print candidate matches for a UI pair
# -------------------------
async def pair_candidates(ui_pair):
    matches = await find_matching_symbols(ui_pair)
    return matches

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

log.info("✓ helpers (CoinDCX minimal with discovery) loaded")