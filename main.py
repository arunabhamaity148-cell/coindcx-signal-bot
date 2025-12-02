# main.py — Level-2 PRO Scanner (REST klines + orderbook depth + caching)
import os
import asyncio
import aiohttp
import time
from dotenv import load_dotenv

load_dotenv()

from helpers import (
    process_data_with_ai,
    btc_calm_check,
    format_signal,
    ema  # helper ema function from helpers.py
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))

# watchlist — keep as you had (edit as needed)
WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","BNBUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","TRXUSDT",
    "DOTUSDT","LTCUSDT","LINKUSDT","MATICUSDT","OPUSDT","ARBUSDT","FILUSDT","AAVEUSDT","SANDUSDT",
    "ATOMUSDT","NEARUSDT","INJUSDT","FXSUSDT","DYDXUSDT","EGLDUSDT","APTUSDT","RNDRUSDT","TIAUSDT",
    "SEIUSDT","BONKUSBT","FTMUSDT","RUNEUSDT","PYTHUSDT","WLDUSDT","SKLUSUSDT","BLURUSDT","MINAUSDT",
    "JTOUSDT","MEWUSDT","1000PEPEUSDT"
]

# concurrency limiter to avoid bursting API
SEM = asyncio.Semaphore(6)

# simple in-memory cache to reduce REST calls
CACHE = {
    # key -> (timestamp, data)
    # e.g. ("klines", "SOLUSDT", "1m") : (ts, [...])
}

BINANCE_REST = "https://api.binance.com"

# ---------------------------
# helper: cache get/set
# ---------------------------
def cache_get(key):
    entry = CACHE.get(key)
    if not entry:
        return None
    ts, data, ttl = entry
    if time.time() - ts > ttl:
        return None
    return data

def cache_set(key, data, ttl=5):
    CACHE[key] = (time.time(), data, ttl)

# ---------------------------
# Telegram send
# ---------------------------
async def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❗ Telegram not configured in .env")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=5)
    except Exception as e:
        print("Telegram send error:", e)

# ---------------------------
# Basic Binance REST wrappers (with caching)
# ---------------------------
async def fetch_klines(session, symbol, interval="1m", limit=100, ttl=5):
    key = ("klines", symbol, interval)
    cached = cache_get(key)
    if cached:
        return cached
    url = f"{BINANCE_REST}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url, timeout=8) as r:
            if r.status != 200:
                return None
            data = await r.json()
            # cache a bit longer for higher TF
            cache_set(key, data, ttl=ttl)
            return data
    except Exception:
        return None

async def fetch_orderbook(session, symbol, limit=50, ttl=2):
    key = ("depth", symbol, limit)
    cached = cache_get(key)
    if cached:
        return cached
    url = f"{BINANCE_REST}/api/v3/depth?symbol={symbol}&limit={limit}"
    try:
        async with session.get(url, timeout=5) as r:
            if r.status != 200:
                return None
            data = await r.json()
            cache_set(key, data, ttl=ttl)
            return data
    except Exception:
        return None

async def fetch_24hr(session, symbol):
    key = ("24hr", symbol)
    cached = cache_get(key)
    if cached:
        return cached
    url = f"{BINANCE_REST}/api/v3/ticker/24hr?symbol={symbol}"
    try:
        async with session.get(url, timeout=5) as r:
            if r.status != 200:
                return None
            d = await r.json()
            cache_set(key, d, ttl=10)
            return d
    except Exception:
        return None

# ---------------------------
# Utilities to compute fields expected by helpers.py
# ---------------------------
def closes_from_klines(klines):
    # kline: [ openTime, open, high, low, close, volume, ... ]
    try:
        return [float(k[4]) for k in klines]
    except Exception:
        return []

def sum_vol_from_klines(klines):
    try:
        return sum(float(k[5]) for k in klines)
    except:
        return 0.0

# naive orderbook analysis to set flags used by helpers
def analyze_orderbook(depth):
    # depth contains 'bids' and 'asks' arrays of [price, qty]
    out = {
        "best_bid": None,
        "best_ask": None,
        "spread": None,
        "top_bid_size": 0.0,
        "top_ask_size": 0.0,
        "wall_shift": False,
        "liq_wall": False,
        "liq_bend": False
    }
    try:
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        if bids:
            best_bid_p = float(bids[0][0]); best_bid_q = float(bids[0][1])
            out["best_bid"] = best_bid_p; out["top_bid_size"] = best_bid_q
        if asks:
            best_ask_p = float(asks[0][0]); best_ask_q = float(asks[0][1])
            out["best_ask"] = best_ask_p; out["top_ask_size"] = best_ask_q
        if out["best_bid"] and out["best_ask"]:
            out["spread"] = (out["best_ask"] - out["best_bid"]) / ((out["best_ask"]+out["best_bid"])/2)
        # simple heuristics: a big wall if top qty >> median of top5
        try:
            bid_sizes = [float(x[1]) for x in bids[:8]] if bids else []
            ask_sizes = [float(x[1]) for x in asks[:8]] if asks else []
            if bid_sizes:
                median_bid = sorted(bid_sizes)[len(bid_sizes)//2]
                if out["top_bid_size"] > max(3*median_bid, 1000):
                    out["liq_wall"] = True
            if ask_sizes:
                median_ask = sorted(ask_sizes)[len(ask_sizes)//2]
                if out["top_ask_size"] > max(3*median_ask, 1000):
                    out["liq_wall"] = True
            # wall_shift heuristic: compare top sizes relative (bid vs ask)
            if out["top_bid_size"] / (out["top_ask_size"] + 1e-9) > 4:
                out["wall_shift"] = True
            if out["top_ask_size"] / (out["top_bid_size"] + 1e-9) > 4:
                out["wall_shift"] = True
            # liq_bend: if near spread some bigger orders inside top10
            if bids and asks:
                # check deeper levels for asymmetric liquidity
                deeper_bid = sum(bid_sizes[1:5]) if len(bid_sizes) > 1 else 0
                deeper_ask = sum(ask_sizes[1:5]) if len(ask_sizes) > 1 else 0
                if deeper_bid > deeper_ask * 2 or deeper_ask > deeper_bid * 2:
                    out["liq_bend"] = True
        except Exception:
            pass
    except Exception:
        pass
    return out

# ---------------------------
# Compose live_data for helpers
# ---------------------------
async def build_live_data(session, symbol):
    """
    Fetch klines and orderbook, compute lightweight indicators and return dict
    that matches fields expected by helpers.py logic functions.
    """
    # fetch parallel
    async with SEM:
        tasks = [
            fetch_klines(session, symbol, "1m", limit=60, ttl=3),
            fetch_klines(session, symbol, "5m", limit=60, ttl=6),
            fetch_klines(session, symbol, "15m", limit=60, ttl=12),
            fetch_klines(session, symbol, "1h", limit=100, ttl=30),
            fetch_klines(session, symbol, "4h", limit=100, ttl=120),
            fetch_orderbook(session, symbol, limit=50, ttl=2),
            fetch_24hr(session, symbol)
        ]
        res = await asyncio.gather(*tasks, return_exceptions=True)

    k1, k5, k15, k1h, k4h, depth, t24 = res

    # if price fetch failed, bail out
    if not k1 or not k5 or not t24:
        return None

    # compute closes and basic EMAs using helpers.ema
    closes_1m = closes_from_klines(k1)
    closes_5m = closes_from_klines(k5)
    closes_15m = closes_from_klines(k15) if k15 else closes_5m
    closes_1h = closes_from_klines(k1h) if k1h else closes_15m
    closes_4h = closes_from_klines(k4h) if k4h else closes_1h

    price = closes_1m[-1] if closes_1m else None
    if price is None:
        return None

    # EMAs — choose reasonable lengths (these are heuristic)
    try:
        ema_15m = ema(closes_15m[-50:] if len(closes_15m) >= 50 else closes_15m, 21) or price
        ema_1h = ema(closes_1h[-50:] if len(closes_1h) >= 50 else closes_1h, 21) or price
        ema_4h = ema(closes_4h[-50:] if len(closes_4h) >= 50 else closes_4h, 21) or price
        ema_8h = ema(closes_4h[-100:] if len(closes_4h) >= 100 else closes_4h, 42) or ema_4h
    except Exception:
        ema_15m = ema_1h = ema_4h = ema_8h = price

    # volumes
    vol_1m = sum_vol_from_klines(k1[-5:]) if k1 else 0.0
    vol_5m = sum_vol_from_klines(k5[-5:]) if k5 else 0.0

    # orderbook analysis
    ob = analyze_orderbook(depth or {})
    spread = ob.get("spread", 0.0) or 0.0

    # 24hr change
    pct24 = None
    try:
        pct24 = abs(float(t24.get("priceChangePercent", 0.0)))
    except:
        pct24 = 0.0

    live = {
        "price": price,
        "ema_15m": ema_15m,
        "ema_1h": ema_1h,
        "ema_4h": ema_4h,
        "ema_8h": ema_8h,
        "trend": "bull_break" if price > ema_1h else "reject" if price < ema_1h else "neutral",
        "fvg": False,  # advanced: can detect FVG from higher TFs later
        "trend_strength": abs((price - ema_1h) / (ema_1h + 1e-9)),
        "exhaustion": False,
        "micro_pb": False,
        "wick_ratio": 1.0,
        "liquidation_sweep": False,
        "vol_1m": vol_1m,
        "vol_5m": vol_5m,
        "delta_1m": False,
        "delta_htf": False,
        "iceberg_1m": False,
        "iceberg_v2": False,
        "wall_shift": ob.get("wall_shift", False),
        "liq_wall": ob.get("liq_wall", False),
        "liq_bend": ob.get("liq_bend", False),
        "adr_ok": True,
        "atr_expanding": False,
        "phase_shift": False,
        "compression": False,
        "speed_imbalance": False,
        "taker_pressure": False,
        "vol_imprint": vol_1m > (vol_5m * 1.5 if vol_5m > 0 else 0),
        "cluster_tiny": False,
        "absorption": False,
        "weakness": False,
        "spread_snap_05": spread > 0.0005,
        "spread_snap_025": spread > 0.00025,
        "spread": spread,
        "be_lock": False,
        "liq_dist": 0.5,
        "kill_5m": False,
        "kill_htf": False,
        "kill_fast": False,
        "kill_primary": False,
        "news_risk": False,
        "recheck_ok": True,
        "btc_calm": True,  # btc_calm filled by main loop using btc_calm_check
        "btc_trending_fast": False,
        "funding_oi_combo": False,
        "funding_extreme": False,
        "funding_delta": False,
        "arb_opportunity": False,
        "oi_spike": False,
        "oi_sustained": False,
        "beta_div": False,
        "gamma_flip": False,
        "heat_sweep": False,
        "slippage": False,
        "orderblock": False
    }

    return live

# ---------------------------
# Symbol task
# ---------------------------
async def handle_symbol(session, symbol, btc_ok):
    try:
        live = await build_live_data(session, symbol)
        if not live:
            # could not build live data
            # print for debug
            # print(f"no live data for {symbol}")
            return None
        live["btc_calm"] = btc_ok
        # send to decision pipeline
        try:
            decision = await process_data_with_ai(symbol, "quick", live)
        except Exception as e:
            print(f"process_data error for {symbol}:", e)
            decision = None

        if decision:
            msg = format_signal(decision)
            await send_telegram(msg)
            print(f"📤 SIGNAL SENT → {decision['symbol']} | {decision['mode']} | score={decision.get('score')}")
        return decision
    except Exception as e:
        print(f"error handling {symbol}:", e)
        return None

# ---------------------------
# Main scanner loop
# ---------------------------
async def scanner():
    print("🚀 Level-2 PRO Scanner Started...")
    async with aiohttp.ClientSession() as session:
        while True:
            btc_ok = await btc_calm_check(session)
            if not btc_ok:
                print("⚠️ BTC Volatile — AI layer evaluating risk (scanning continues)")

            # create tasks per symbol with concurrency
            tasks = []
            for symbol in WATCHLIST:
                tasks.append(handle_symbol(session, symbol, btc_ok))

            # run with concurrency limit via gather in chunks
            results = []
            # chunked gather to limit total concurrency
            chunk_size = 8
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i:i+chunk_size]
                res = await asyncio.gather(*chunk, return_exceptions=True)
                results.extend(res)

            await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(scanner())
    except KeyboardInterrupt:
        print("Stopped.")