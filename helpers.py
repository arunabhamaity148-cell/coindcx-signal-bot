# helpers.py — REAL BINANCE SIGNAL ENGINE (logging & explain + safe parsers)
import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json
from typing import Optional

# -------------------------
# ENV / CONFIG
# -------------------------
BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SECRET = os.getenv("BINANCE_SECRET_KEY", "").encode() if os.getenv("BINANCE_SECRET_KEY") else b""
FUTURES = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

# AI disabled by default
AI_VERIFY_ENABLED = False
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Core thresholds (tweakable via env)
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "1800"))   # 30 min default
INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "8"))

MODE_THRESH = {
    "quick": int(os.getenv("THRESH_QUICK", "65")),
    "mid": int(os.getenv("THRESH_MID", "72")),
    "trend": int(os.getenv("THRESH_TREND", "78")),
}

TP_SL = {
    "quick": (float(os.getenv("TP_QUICK", "1.2")), float(os.getenv("SL_QUICK", "0.7"))),
    "mid": (float(os.getenv("TP_MID", "1.8")), float(os.getenv("SL_MID", "1.0"))),
    "trend": (float(os.getenv("TP_TREND", "2.5")), float(os.getenv("SL_TREND", "1.2"))),
}

LEVERAGE = int(os.getenv("LEVERAGE", "50"))

# Market safety filters
BTC_STABILITY_THRESHOLD = float(os.getenv("BTC_STABILITY_THRESHOLD", "1.2"))  # percent 24h change
MIN_24H_VOL = float(os.getenv("MIN_24H_VOL", "20000"))  # minimum quote volume over 24h (in USDT)
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.06"))  # 0.06% max spread
MAX_LIQ_DIST = float(os.getenv("MAX_LIQ_DIST", "1.0"))  # percent mark->liq threshold for safety

# Logging verbosity
VERBOSE_SCORE_MIN = int(os.getenv("VERBOSE_SCORE_MIN", "45"))

# -------------------------
# STATE + CACHE
# -------------------------
_last_signal = {}
_cache = {}
_cache_time = {}

def cache_get(key, ttl=5):
    v = _cache.get(key)
    if not v:
        return None
    t = _cache_time.get(key, 0)
    if time.time() - t > ttl:
        _cache.pop(key, None)
        _cache_time.pop(key, None)
        return None
    return v

def cache_set(key, val):
    _cache[key] = val
    _cache_time[key] = time.time()

# -------------------------
# COOLDOWN HELPERS
# -------------------------
def cooldown_ok(symbol: str) -> bool:
    return (time.time() - _last_signal.get(symbol, 0)) >= COOLDOWN

def update_cd(symbol: str):
    _last_signal[symbol] = time.time()

# -------------------------
# SIGN / API HELPERS
# -------------------------
def sign(query: str) -> str:
    if not BIN_SECRET:
        return ""
    return hmac.new(BIN_SECRET, query.encode(), hashlib.sha256).hexdigest()

async def public(session: aiohttp.ClientSession, path: str, params: str = ""):
    url = f"{FUTURES}{path}"
    if params:
        url = f"{url}?{params}"
    try:
        async with session.get(url, timeout=8) as r:
            return await r.json()
    except Exception:
        return None

async def private(session: aiohttp.ClientSession, path: str, params: str = ""):
    ts = int(time.time()*1000)
    q = f"{params}&timestamp={ts}" if params else f"timestamp={ts}"
    sig = sign(q)
    url = f"{FUTURES}{path}?{q}&signature={sig}"
    headers = {"X-MBX-APIKEY": BIN_KEY} if BIN_KEY else {}
    try:
        async with session.get(url, headers=headers, timeout=8) as r:
            return await r.json()
    except Exception:
        return None

# -------------------------
# BINANCE DATA CALLS
# -------------------------
async def depth(session: aiohttp.ClientSession, symbol: str, limit: int = 50):
    return await public(session, "/fapi/v1/depth", f"symbol={symbol}&limit={limit}")

async def mark_price(session: aiohttp.ClientSession, symbol: str):
    return await public(session, "/fapi/v1/premiumIndex", f"symbol={symbol}")

async def funding_rate(session: aiohttp.ClientSession, symbol: str):
    res = await public(session, "/fapi/v1/fundingRate", f"symbol={symbol}&limit=1")
    try:
        if res and isinstance(res, list) and len(res) > 0:
            return float(res[0].get("fundingRate", 0.0))
    except Exception:
        pass
    return 0.0

async def oi_history(session: aiohttp.ClientSession, symbol: str, limit: int = 5):
    return await public(session, "/fapi/v1/openInterestHist", f"symbol={symbol}&period=5m&limit={limit}")

async def trades(session: aiohttp.ClientSession, symbol: str, limit: int = 50):
    return await public(session, "/fapi/v1/aggTrades", f"symbol={symbol}&limit={limit}")

async def ticker_24h(session: aiohttp.ClientSession, symbol: str):
    return await public(session, "/fapi/v1/ticker/24hr", f"symbol={symbol}")

async def kline(session: aiohttp.ClientSession, symbol: str, interval: str = "1m", limit: int = 60):
    return await public(session, "/fapi/v1/klines", f"symbol={symbol}&interval={interval}&limit={limit}")

# -------------------------
# SAFE TRADE PARSER
# -------------------------
async def safe_last_trade(session: aiohttp.ClientSession, symbol: str):
    try:
        ts = await trades(session, symbol, limit=50)
        if not ts or not isinstance(ts, list) or len(ts) == 0:
            return {"price": None, "qty": None}
        last = ts[-1]
        p = None
        q = None
        if isinstance(last, dict):
            p = last.get("p") or last.get("price")
            q = last.get("q") or last.get("quantity")
        else:
            try:
                p = last[4]
                q = last[5]
            except Exception:
                p = None
                q = None
        try:
            price = float(p) if p is not None else None
        except Exception:
            price = None
        try:
            qty = float(q) if q is not None else None
        except Exception:
            qty = None
        if price is None or price == 0:
            return {"price": None, "qty": None}
        return {"price": price, "qty": qty}
    except Exception:
        return {"price": None, "qty": None}

# -------------------------
# EMA (robust)
# -------------------------
def ema(values, length=None):
    if values is None:
        return 0.0
    vals = list(values)
    if not vals:
        return 0.0
    if length is None:
        length = len(vals)
    try:
        length = int(length)
        if length <= 0:
            length = len(vals)
    except Exception:
        length = len(vals)
    k = 2.0 / (length + 1.0)
    e = float(vals[0])
    for v in vals[1:]:
        e = e * (1 - k) + float(v) * k
    return float(e)

# -------------------------
# HTF FETCHER (15m/1h/4h)
# -------------------------
async def get_htf(session: aiohttp.ClientSession, symbol: str):
    cache_key = f"htf:{symbol}"
    cached = cache_get(cache_key, ttl=6)
    if cached:
        return cached

    k15_task = kline(session, symbol, "15m", 60)
    k1h_task = kline(session, symbol, "1h", 60)
    k4h_task = kline(session, symbol, "4h", 60)

    k15, k1h, k4h = await asyncio.gather(k15_task, k1h_task, k4h_task)

    if not k15 or not k1h or not k4h:
        return None

    try:
        c15 = [float(x[4]) for x in k15]
        c1h = [float(x[4]) for x in k1h]
        c4h = [float(x[4]) for x in k4h]
    except Exception:
        return None

    ema_15 = ema(c15[-15:], 15)
    ema_50 = ema(c15[-50:], 50) if len(c15) >= 50 else ema_15
    ema_200 = ema(c15[-200:], 200) if len(c15) >= 200 else ema_50

    ema_1h_50 = ema(c1h[-50:], 50) if len(c1h) >= 50 else c1h[-1]
    ema_4h_50 = ema(c4h[-50:], 50) if len(c4h) >= 50 else c4h[-1]

    out = {
        "close_15": c15[-1],
        "prev_15": c15[-2] if len(c15) >= 2 else c15[-1],
        "ema_15": ema_15,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "ema_1h": ema_1h_50,
        "ema_4h": ema_4h_50,
        "trend_15": "up" if c15[-1] > ema_50 else "down",
        "trend_1h": "up" if c1h[-1] > ema_1h_50 else "down",
        "trend_4h": "up" if c4h[-1] > ema_4h_50 else "down",
    }

    cache_set(cache_key, out)
    return out

# -------------------------
# BTC calm check
# -------------------------
async def btc_calm_check(session: aiohttp.ClientSession):
    try:
        r = await public(session, "/fapi/v1/ticker/24hr", "symbol=BTCUSDT")
        if not r or "priceChangePercent" not in r:
            return True
        change = abs(float(r.get("priceChangePercent", 0.0)))
        return change < BTC_STABILITY_THRESHOLD
    except Exception:
        return True

# -------------------------
# Orderbook filters & imbalance
# -------------------------
def compute_spread_pct(best_bid: float, best_ask: float) -> float:
    if best_ask <= 0:
        return 999.0
    return abs(best_ask - best_bid) / ((best_ask + best_bid) / 2.0) * 100.0

def depth_imbalance(depth_json) -> float:
    try:
        bids = depth_json.get("bids", [])[:10]
        asks = depth_json.get("asks", [])[:10]
        bid_vol = sum([float(b[1]) * float(b[0]) for b in bids])
        ask_vol = sum([float(a[1]) * float(a[0]) for a in asks])
        if ask_vol <= 0:
            return 999.0
        return bid_vol / ask_vol
    except Exception:
        return 1.0

async def check_liquidity_and_spread(session: aiohttp.ClientSession, symbol: str):
    d = await depth(session, symbol, limit=20)
    t = await ticker_24h(session, symbol)
    if not d or not t:
        return False, "no_data"
    try:
        best_bid = float(d["bids"][0][0])
        best_ask = float(d["asks"][0][0])
    except Exception:
        return False, "no_depth"
    spread_pct = compute_spread_pct(best_bid, best_ask)
    vol_quote = 0.0
    try:
        vol_quote = float(t.get("quoteVolume", 0)) if t.get("quoteVolume") else float(t.get("volume", 0)) * ((best_bid+best_ask)/2)
    except Exception:
        vol_quote = 0.0
    if spread_pct > MAX_SPREAD_PCT:
        return False, "wide_spread"
    if vol_quote < MIN_24H_VOL:
        return False, "low_volume"
    imb = depth_imbalance(d)
    return True, {"spread_pct": round(spread_pct,4), "imbalance": round(imb,3), "vol_quote": vol_quote}

# -------------------------
# DIRECTION ENGINE
# -------------------------
def get_direction(htf: dict):
    try:
        ema_bull = htf["ema_15"] > htf["ema_50"] > htf["ema_200"]
        ema_bear = htf["ema_15"] < htf["ema_50"] < htf["ema_200"]
        bull = htf["trend_15"] == "up" and htf["trend_1h"] == "up"
        bear = htf["trend_15"] == "down" and htf["trend_1h"] == "down"
        if ema_bull and bull:
            return "BUY"
        if ema_bear and bear:
            return "SELL"
    except Exception:
        return None
    return None

# -------------------------
# SCORE ENGINE (orderflow focused)
# -------------------------
def score_engine(htf: dict, order_meta: dict, mode: str):
    s = 0
    if mode == "quick":
        s += 15 if htf.get("trend_15") == "up" else 0
        s += 10 if htf.get("ema_15",0) > htf.get("ema_50",0) else 0
        s += 12 if order_meta.get("imbalance",1) > 1.2 else 0
        s += 10 if order_meta.get("vol_quote",0) > MIN_24H_VOL*1.5 else 0
        s += 8 if abs(order_meta.get("funding",0)) < 0.02 else 0
    elif mode == "mid":
        s += 20 if htf.get("trend_1h") == "up" else 0
        s += 12 if htf.get("ema_50",0) > htf.get("ema_200",0) else 0
        s += 12 if order_meta.get("imbalance",1) > 1.3 else 0
        s += 10 if order_meta.get("oi_sustained", False) else 0
        s += 8 if abs(order_meta.get("funding",0)) < 0.03 else 0
    else:  # trend
        s += 25 if htf.get("trend_4h") == "up" else 0
        s += 15 if htf.get("trend_1h") == "up" else 0
        s += 15 if htf.get("ema_50",0) > htf.get("ema_200",0) else 0
        s += 10 if order_meta.get("oi_sustained", False) else 0
        s += 10 if order_meta.get("imbalance",1) > 1.4 else 0

    if abs(order_meta.get("funding",0)) > 0.05:
        s -= 15
    if order_meta.get("liq_risk", False):
        s -= 25

    s = max(0, min(100, int(s)))
    return s

# -------------------------
# Logging explain helper
# -------------------------
def explain_data_for_log(htf: dict, order_meta: dict):
    return {
        "ema_15": round(htf.get("ema_15", 0), 6) if htf.get("ema_15") is not None else None,
        "ema_50": round(htf.get("ema_50", 0), 6) if htf.get("ema_50") is not None else None,
        "trend_1h": htf.get("trend_1h"),
        "imbalance": order_meta.get("imbalance"),
        "spread_pct": order_meta.get("spread_pct"),
        "vol_quote": int(order_meta.get("vol_quote")) if order_meta.get("vol_quote") else 0,
        "funding": round(order_meta.get("funding", 0), 6),
        "liq_risk": order_meta.get("liq_risk", False),
    }

# -------------------------
# final_process with verbose logs
# -------------------------
async def final_process(session: aiohttp.ClientSession, symbol: str, mode: str):
    # 1. BTC calm
    btc_ok = await btc_calm_check(session)
    if not btc_ok:
        print(f"[SKIP] {symbol} → BTC volatile (BTC calm check failed)")
        return None

    # 2. HTF
    htf = await get_htf(session, symbol)
    if not htf:
        print(f"[SKIP] {symbol} → Binance returned no OHLCV or HTF failed")
        return None

    # 3. liquidity & spread
    liq_ok, liq_meta = await check_liquidity_and_spread(session, symbol)
    if not liq_ok:
        reason = liq_meta if isinstance(liq_meta, str) else "liquidity/spread check failed"
        print(f"[SKIP] {symbol} → {reason}")
        return None

    # 4. funding + OI + liq dist
    funding = await funding_rate(session, symbol)
    oi = await oi_history(session, symbol, limit=3)
    oi_sustained = False
    try:
        if oi and isinstance(oi, list) and len(oi) >= 2:
            vals = []
            for x in oi:
                # openInterest or sumOpenInterest
                v = x.get("sumOpenInterest") or x.get("openInterest") or x.get("value")
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)
            if len(vals) >= 2:
                oi_sustained = vals[-1] > vals[0]
    except Exception:
        oi_sustained = False

    # 5. liquidation distance approx
    d = await depth(session, symbol, limit=20)
    mark = await mark_price(session, symbol)
    liq_risk = False
    try:
        if d and mark:
            mp = None
            if isinstance(mark, list) and len(mark) > 0:
                mp = float(mark[0].get("markPrice", mark[0].get("lastPrice", 0)))
            elif isinstance(mark, dict):
                mp = float(mark.get("markPrice", 0))
            if mp and d.get("bids") and d.get("asks"):
                best_bid = float(d["bids"][0][0])
                best_ask = float(d["asks"][0][0])
                dist_pct = min(abs(mp - best_bid), abs(best_ask - mp)) / mp * 100.0
                if dist_pct < MAX_LIQ_DIST:
                    liq_risk = True
    except Exception:
        liq_risk = False

    order_meta = {
        "spread_pct": liq_meta.get("spread_pct") if isinstance(liq_meta, dict) else None,
        "imbalance": liq_meta.get("imbalance") if isinstance(liq_meta, dict) else 1.0,
        "vol_quote": liq_meta.get("vol_quote") if isinstance(liq_meta, dict) else 0,
        "funding": funding,
        "oi_sustained": oi_sustained,
        "liq_risk": liq_risk
    }

    # 6. direction
    direction = get_direction(htf)
    if not direction:
        print(f"[SKIP] {symbol} → No direction (data incomplete)")
        return None

    # 7. score
    score = score_engine(htf, order_meta, mode)

    # Verbose score print if above threshold for logging convenience
    if score >= VERBOSE_SCORE_MIN:
        print(f"[SCORE] {symbol} → {score} | RAW: {explain_data_for_log(htf, order_meta)}")

    # 8. threshold check
    mode_thresh = MODE_THRESH.get(mode, 100)
    if score < mode_thresh:
        print(f"[SKIP] {symbol} ({mode}) → score {score} < threshold {mode_thresh}")
        return None

    # 9. cooldown
    if not cooldown_ok(symbol):
        print(f"[SKIP] {symbol} ({mode}) → cooldown active")
        return None

    # 10. compute tp/sl (direction-aware)
    price = float(htf["close_15"])
    tp_pct, sl_pct = TP_SL.get(mode, (1.8, 1.0))
    if direction == "BUY":
        tp = price * (1 + tp_pct/100.0)
        sl = price * (1 - sl_pct/100.0)
    else:  # SELL
        tp = price * (1 - tp_pct/100.0)
        sl = price * (1 + sl_pct/100.0)

    signal = {
        "symbol": symbol,
        "direction": direction,
        "mode": mode,
        "price": round(price, 6),
        "tp": round(tp, 6),
        "sl": round(sl, 6),
        "score": int(score),
        "meta": order_meta
    }

    last_trade = await safe_last_trade(session, symbol)
    if last_trade.get("price") is not None:
        signal["last_trade_price"] = last_trade["price"]
        signal["last_trade_qty"] = last_trade["qty"]

    # update cooldown and return
    update_cd(symbol)
    print(f"[SEND] {symbol} ({mode}) → direction={direction} score={score} tp={signal['tp']} sl={signal['sl']}")
    return signal

# -------------------------
# TELEGRAM FORMAT
# -------------------------
def format_signal(sig: dict) -> str:
    meta = sig.get("meta", {})
    imb = meta.get("imbalance", "N/A")
    sp = meta.get("spread_pct", "N/A")
    vol = meta.get("vol_quote", "N/A")
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Entry:</b> <code>{sig['price']}</code>
<b>Score:</b> <b>{sig['score']}</b> / 100

🎯 <b>TP:</b> <code>{sig['tp']}</code>
🛑 <b>SL:</b> <code>{sig['sl']}</code>

📊 Imbalance: {imb} | Spread%: {sp}
📉 24h Vol(quote): {int(vol) if isinstance(vol,(int,float)) else vol}
⚡ Leverage: {LEVERAGE}x
⏱ Cooldown: {int(COOLDOWN/60)}m

#HybridAI #ArunSystem
"""

# End of helpers.py