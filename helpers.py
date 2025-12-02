# helpers.py — REAL BINANCE + AI DECISION ENGINE (FIXED ema signature + safe parsers)
import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json

# -------------------------
# ENV / CONFIG
# -------------------------
BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SECRET = os.getenv("BINANCE_SECRET_KEY", "").encode() if os.getenv("BINANCE_SECRET_KEY") else b""
FUTURES = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "1800"))   # default 30 minutes
MODE_THRESH = {
    "quick": int(os.getenv("THRESH_QUICK", "55")),
    "mid": int(os.getenv("THRESH_MID", "62")),
    "trend": int(os.getenv("THRESH_TREND", "70")),
}
TP_SL = {
    "quick": (float(os.getenv("TP_QUICK", "1.2")), float(os.getenv("SL_QUICK", "0.7"))),
    "mid": (float(os.getenv("TP_MID", "1.8")), float(os.getenv("SL_MID", "1.0"))),
    "trend": (float(os.getenv("TP_TREND", "2.5")), float(os.getenv("SL_TREND", "1.2"))),
}
LEVERAGE = int(os.getenv("LEVERAGE", "50"))

# -------------------------
# GLOBAL STATE
# -------------------------
_last_signal = {}
_cache = {}
_cache_time = {}

# -------------------------
# CACHE HELPERS
# -------------------------
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
def cooldown_ok(symbol):
    return (time.time() - _last_signal.get(symbol, 0)) >= COOLDOWN

def update_cd(symbol):
    _last_signal[symbol] = time.time()

# -------------------------
# SIGNING / API HELPERS
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
# BINANCE DATA FUNCTIONS
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

async def trades(session: aiohttp.ClientSession, symbol: str, limit: int = 50):
    return await public(session, "/fapi/v1/aggTrades", f"symbol={symbol}&limit={limit}")

async def ticker_24h(session: aiohttp.ClientSession, symbol: str):
    return await public(session, "/fapi/v1/ticker/24hr", f"symbol={symbol}")

async def kline(session: aiohttp.ClientSession, symbol: str, interval: str = "1m", limit: int = 60):
    return await public(session, "/fapi/v1/klines", f"symbol={symbol}&interval={interval}&limit={limit}")

# -------------------------
# SAFE TRADE PARSER (PATCHED)
# -------------------------
async def safe_last_trade(session: aiohttp.ClientSession, symbol: str):
    """
    Return dict: {"price": float|null, "qty": float|null}
    Handles empty responses gracefully (PEPE/SHIB issue).
    """
    try:
        ts = await trades(session, symbol, limit=50)
        if not ts or not isinstance(ts, list) or len(ts) == 0:
            return {"price": None, "qty": None}

        last = ts[-1]
        p = None
        q = None

        if isinstance(last, dict):
            p = last.get("p") or last.get("price") or last.get("P")
            q = last.get("q") or last.get("quantity") or last.get("Q")
        else:
            # fallback if list-like
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
# EMA (fixed signature: length optional)
# -------------------------
def ema(values, length=None):
    """
    Robust EMA:
    - values: iterable of numbers (will convert to list)
    - length: optional int. if None -> use len(values)
    Returns float (0.0 on empty)
    """
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

    # if not enough values, use what we have
    if len(vals) < 1:
        return float(vals[-1]) if vals else 0.0
    # initialize with first value
    k = 2.0 / (length + 1.0)
    e = float(vals[0])
    for v in vals[1:]:
        e = e * (1 - k) + float(v) * k
    return float(e)

# -------------------------
# HTF FETCHER (15m / 1h / 4h)
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
# SCORE ENGINE V3
# -------------------------
def score_v3(htf: dict, mode: str) -> int:
    s = 0
    try:
        prev = htf.get("prev_15", htf.get("close_15"))
        momentum = abs(htf["close_15"] - prev) / max(prev, 1)
        if mode == "quick":
            if momentum > 0.004: s += 20
            if htf.get("trend_15") == "up": s += 10
            if htf.get("ema_15", 0) > htf.get("ema_50", 0): s += 10
        elif mode == "mid":
            if htf.get("trend_1h") == "up": s += 20
            if htf.get("ema_50", 0) > htf.get("ema_200", 0): s += 15
            if momentum > 0.002: s += 10
        elif mode == "trend":
            if htf.get("trend_4h") == "up": s += 25
            if htf.get("trend_1h") == "up": s += 20
            if htf.get("ema_50", 0) > htf.get("ema_200", 0): s += 15
    except Exception:
        pass
    return min(100, int(s))

# -------------------------
# AI VERIFY (fail-open)
# -------------------------
async def ai_verify(signal: dict):
    """
    Send short verification prompt to OpenAI.
    Fail-open: on any error, return {"approve": True, "tp": None, "sl": None}
    """
    if not OPENAI_KEY:
        return {"approve": True, "tp": None, "sl": None, "note": "no_openai_key_fail_open"}

    prompt = (
        "You are a conservative crypto signal verifier. "
        "Input: JSON with keys: symbol, direction, mode, price, tp, sl, score. "
        "Return ONLY a single JSON object with keys: approve (bool), tp (number|null), sl (number|null), note (string).\n\n"
        f"Input: {json.dumps(signal)}\n"
        "Example output: {\"approve\": true, \"tp\": null, \"sl\": null, \"note\": \"ok\"}"
    )

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise and strict risk manager."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 120
    }

    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=6) as resp:
                if resp.status != 200:
                    return {"approve": True, "tp": None, "sl": None, "note": f"openai_status_{resp.status}"}
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                try:
                    parsed = json.loads(content)
                    return {
                        "approve": bool(parsed.get("approve", True)),
                        "tp": float(parsed["tp"]) if parsed.get("tp") not in (None, "", False) else None,
                        "sl": float(parsed["sl"]) if parsed.get("sl") not in (None, "", False) else None,
                        "note": str(parsed.get("note", ""))[:200]
                    }
                except Exception:
                    return {"approve": True, "tp": None, "sl": None, "note": "openai_parse_fail"}
    except Exception as e:
        return {"approve": True, "tp": None, "sl": None, "note": f"openai_error_{str(e)[:100]}"}

# -------------------------
# FINAL PROCESS (main decision)
# -------------------------
async def final_process(session: aiohttp.ClientSession, symbol: str, mode: str):
    """
    Build HTF, get direction, score, compute TP/SL, verify with AI (fail-open).
    Returns signal dict or None.
    """
    htf = await get_htf(session, symbol)
    if not htf:
        return None

    direction = get_direction(htf)
    if not direction:
        return None

    score = score_v3(htf, mode)
    if score < MODE_THRESH.get(mode, 60):
        return None

    price = float(htf["close_15"])
    tp_pct, sl_pct = TP_SL.get(mode, (1.8, 1.0))
    tp = price * (1 + tp_pct/100.0)
    sl = price * (1 - sl_pct/100.0)

    signal = {
        "symbol":symbol,
        "direction":direction,
        "mode":mode,
        "price":round(price, 6),
        "tp":round(tp, 6),
        "sl":round(sl, 6),
        "score":int(score),
    }

    last_trade = await safe_last_trade(session, symbol)
    if last_trade.get("price") is not None:
        signal["last_trade_price"] = last_trade["price"]
        signal["last_trade_qty"] = last_trade["qty"]

    ai = await ai_verify(signal)
    if not ai.get("approve", True):
        return None

    if ai.get("tp") is not None:
        try:
            signal["tp"] = float(ai.get("tp"))
        except Exception:
            pass
    if ai.get("sl") is not None:
        try:
            signal["sl"] = float(ai.get("sl"))
        except Exception:
            pass

    return signal

# -------------------------
# TELEGRAM FORMAT
# -------------------------
def format_signal(sig: dict) -> str:
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Entry:</b> <code>{sig['price']}</code>
<b>Score:</b> <b>{sig['score']}</b> / 100

🎯 <b>TP:</b> <code>{sig['tp']}</code>
🛑 <b>SL:</b> <code>{sig['sl']}</code>

📉 LiQ Dist: {sig.get('liq_dist', 'N/A')}
⚡ Leverage: {LEVERAGE}x
⏱ Cooldown: {int(COOLDOWN/60)}m

#HybridAI #ArunSystem
"""

# End of helpers.py