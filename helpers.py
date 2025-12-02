# helpers.py — FULL AI DECISION LAYER + Direction + Score + HTF + TP/SL + Format
import aiohttp
import asyncio
import os
import time
import json
import math
from collections import defaultdict

# ==========================================================
# CONFIG
# ==========================================================
COOLDOWN = 1800  # 30 minutes
MODE_THRESH = {"quick": 55, "mid": 62, "trend": 70}
TP_SL = {"quick": (1.2, 0.7), "mid": (1.8, 1.0), "trend": (2.5, 1.2)}
LEVERAGE = 50

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
AUTO_APPROVE = True   # FULL AUTO (NO MANUAL APPROVAL)

# ==========================================================
# CACHE SYSTEM (HTF optimization)
# ==========================================================
_cache = {}
_cache_time = {}

def cache_get(key, ttl=5):
    v = _cache.get(key)
    t = _cache_time.get(key, 0)
    if not v:
        return None
    if time.time() - t > ttl:
        return None
    return v

def cache_set(key, val):
    _cache[key] = val
    _cache_time[key] = time.time()

# ==========================================================
# UTILS
# ==========================================================
def ema(values, length):
    if len(values) < 1:
        return values[-1] if values else 0
    k = 2 / (length + 1)
    e = values[0]
    for v in values[1:]:
        e = e * (1 - k) + v * k
    return e

_last_signal = {}

def cooldown_ok(symbol):
    t = _last_signal.get(symbol, 0)
    return (time.time() - t) >= COOLDOWN

def update_cd(symbol):
    _last_signal[symbol] = time.time()

# ==========================================================
# HTF FETCH MODULE
# ==========================================================
BIN = "https://api.binance.com"

async def fetch_klines(session, symbol, interval, limit):
    key = f"{symbol}:{interval}:{limit}"
    c = cache_get(key, ttl=10)
    if c: return c
    url = f"{BIN}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url, timeout=8) as r:
            data = await r.json()
            cache_set(key, data)
            return data
    except:
        return None

async def get_htf(session, symbol):
    k15, k1h, k4h = await asyncio.gather(
        fetch_klines(session, symbol, "15m", 60),
        fetch_klines(session, symbol, "1h", 60),
        fetch_klines(session, symbol, "4h", 60)
    )

    if not k15 or not k1h or not k4h:
        return None

    c15 = [float(x[4]) for x in k15]
    c1h = [float(x[4]) for x in k1h]
    c4h = [float(x[4]) for x in k4h]

    ema15 = ema(c15[-15:], 15)
    ema50 = ema(c15[-50:], 50) if len(c15) >= 50 else ema15
    ema200 = ema(c15[-200:], 200) if len(c15) >= 200 else ema50

    ema1h_50 = ema(c1h[-50:], 50) if len(c1h) >= 50 else c1h[-1]
    ema4h_50 = ema(c4h[-50:], 50) if len(c4h) >= 50 else c4h[-1]

    return {
        "close_15": c15[-1],
        "prev_15": c15[-2],
        "ema_15": ema15,
        "ema_50": ema50,
        "ema_200": ema200,
        "ema_1h": ema1h_50,
        "ema_4h": ema4h_50,
        "trend_15": "up" if c15[-1] > ema50 else "down",
        "trend_1h": "up" if c1h[-1] > ema1h_50 else "down",
        "trend_4h": "up" if c4h[-1] > ema4h_50 else "down",
        "closes": c15
    }

# ==========================================================
# DIRECTION ENGINE v2
# ==========================================================
def get_direction(htf):
    ema_bull = htf["ema_15"] > htf["ema_50"] > htf["ema_200"]
    ema_bear = htf["ema_15"] < htf["ema_50"] < htf["ema_200"]

    bull = htf["trend_15"] == "up" and htf["trend_1h"] == "up"
    bear = htf["trend_15"] == "down" and htf["trend_1h"] == "down"

    if ema_bull and bull:
        return "BUY"
    if ema_bear and bear:
        return "SELL"
    return None

# ==========================================================
# SCORE ENGINE V3
# ==========================================================
def score_v3(htf, mode):
    s = 0
    momentum = abs(htf["close_15"] - htf["prev_15"]) / max(htf["prev_15"], 1)

    if mode == "quick":
        if momentum > 0.004: s += 15
        if htf["trend_15"] == "up": s += 10
        if htf["ema_15"] > htf["ema_50"]: s += 10

    if mode == "mid":
        if htf["trend_1h"] == "up": s += 20
        if htf["ema_50"] > htf["ema_200"]: s += 15
        if momentum > 0.002: s += 10

    if mode == "trend":
        if htf["trend_4h"] == "up": s += 25
        if htf["trend_1h"] == "up": s += 20
        if htf["ema_50"] > htf["ema_200"]: s += 15

    return min(100, s)

# ==========================================================
# AI VERIFIER (AUTO DECISION)
# ==========================================================
async def ai_verify(sig):
    if not OPENAI_KEY:
        return {"approve": True, "tp": None, "sl": None, "note": "no_key_fail_open"}

    prompt = f"""
You are a crypto risk manager. Verify this signal:

{json.dumps(sig)}

Return only JSON:
{{"approve": true/false, "tp": null or number, "sl": null or number, "note": "short text"}}
"""

    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are strict but concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 100
    }

    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=6) as r:
                data = await r.json()
                c = data["choices"][0]["message"]["content"]
                j = json.loads(c)
                return j
    except:
        return {"approve": True, "tp": None, "sl": None, "note": "fail_open"}

# ==========================================================
# FINAL PROCESS
# ==========================================================
async def final_process(session, symbol, mode):
    htf = await get_htf(session, symbol)
    if not htf:
        return None

    direction = get_direction(htf)
    if not direction:
        return None

    score = score_v3(htf, mode)
    if score < MODE_THRESH[mode]:
        return None

    price = htf["close_15"]
    tp_pct, sl_pct = TP_SL[mode]
    tp = price * (1 + tp_pct / 100)
    sl = price * (1 - sl_pct / 100)

    signal = {
        "symbol": symbol,
        "direction": direction,
        "mode": mode,
        "price": round(price, 6),
        "tp": round(tp, 6),
        "sl": round(sl, 6),
        "score": score,
        "liq_dist": 0.6
    }

    # AI DECISION
    ai = await ai_verify(signal)

    if not ai.get("approve", True):
        return None

    if ai.get("tp") is not None:
        signal["tp"] = ai["tp"]
    if ai.get("sl") is not None:
        signal["sl"] = ai["sl"]

    return signal

# ==========================================================
# TELEGRAM FORMAT
# ==========================================================
def format_signal(sig):
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Entry:</b> <code>{sig['price']}</code>
<b>Score:</b> {sig['score']}

🎯 <b>TP:</b> <code>{sig['tp']}</code>
🛑 <b>SL:</b> <code>{sig['sl']}</code>

⚡ Leverage: {LEVERAGE}x
📉 LiQ Dist: {sig['liq_dist']}%
⏱ Cooldown: 30m

#ArunSystem #HybridAI
"""