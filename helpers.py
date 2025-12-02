# helpers.py — REAL BINANCE + AI DECISION ENGINE (FINAL)
import aiohttp
import asyncio
import os
import time
import hmac, hashlib, json

# ------------------------------------------------------
# ENV CONFIG
# ------------------------------------------------------
BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SECRET = os.getenv("BINANCE_SECRET_KEY").encode()
FUTURES = "https://fapi.binance.com"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

COOLDOWN = 1800
MODE_THRESH = {"quick": 55, "mid": 62, "trend": 70}
TP_SL = {"quick": (1.2,0.7), "mid": (1.8,1.0), "trend": (2.5,1.2)}
LEVERAGE = 50

# ------------------------------------------------------
# COOLING
# ------------------------------------------------------
_last_signal = {}

def cooldown_ok(symbol):
    return (time.time() - _last_signal.get(symbol,0)) >= COOLDOWN

def update_cd(symbol):
    _last_signal[symbol] = time.time()

# ------------------------------------------------------
# SIGNING HELPERS
# ------------------------------------------------------
def sign(q):
    return hmac.new(BIN_SECRET, q.encode(), hashlib.sha256).hexdigest()

async def public(session, path, params=""):
    url = f"{FUTURES}{path}?{params}"
    async with session.get(url) as r:
        return await r.json()

async def private(session, path, params=""):
    ts = int(time.time()*1000)
    q = f"{params}&timestamp={ts}"
    s = sign(q)
    url = f"{FUTURES}{path}?{q}&signature={s}"
    async with session.get(url, headers={"X-MBX-APIKEY": BIN_KEY}) as r:
        return await r.json()

# ------------------------------------------------------
# REAL BINANCE FUTURES DATA
# ------------------------------------------------------
async def depth(session, symbol):
    return await public(session, "/fapi/v1/depth", f"symbol={symbol}&limit=50")

async def mark_price(session, symbol):
    return await public(session, "/fapi/v1/premiumIndex", f"symbol={symbol}")

async def funding_rate(session, symbol):
    d = await public(session, "/fapi/v1/fundingRate", f"symbol={symbol}&limit=1")
    return float(d[0]["fundingRate"]) if d else 0

async def trades(session, symbol):
    return await public(session, "/fapi/v1/aggTrades", f"symbol={symbol}&limit=50")

async def ticker_24h(session, symbol):
    return await public(session, "/fapi/v1/ticker/24hr", f"symbol={symbol}")

async def kline(session, symbol, interval="1m", limit=60):
    return await public(session, "/fapi/v1/klines", f"symbol={symbol}&interval={interval}&limit={limit}")

# ------------------------------------------------------
# EMA
# ------------------------------------------------------
def ema(values, length):
    if not values:
        return 0
    k = 2/(length+1)
    e = values[0]
    for v in values[1:]:
        e = e*(1-k)+v*k
    return e

# ------------------------------------------------------
# HTF FETCHER
# ------------------------------------------------------
async def get_htf(session, symbol):
    k15, k1h, k4h = await asyncio.gather(
        kline(session, symbol, "15m", 60),
        kline(session, symbol, "1h", 60),
        kline(session, symbol, "4h", 60)
    )

    if not k15 or not k1h or not k4h:
        return None

    c15 = [float(x[4]) for x in k15]
    c1h = [float(x[4]) for x in k1h]
    c4h = [float(x[4]) for x in k4h]

    return {
        "close_15": c15[-1],
        "prev_15": c15[-2],

        "ema_15": ema(c15[-15:], 15),
        "ema_50": ema(c15[-50:], 50),
        "ema_200": ema(c15[-200:], 200) if len(c15)>=200 else ema(c15[-50:],50),

        "ema_1h": ema(c1h[-50:], 50),
        "ema_4h": ema(c4h[-50:], 50),

        "trend_15": "up" if c15[-1] > ema(c15[-50:],50) else "down",
        "trend_1h": "up" if c1h[-1] > ema(c1h[-50:],50) else "down",
        "trend_4h": "up" if c4h[-1] > ema(c4h[-50:],50) else "down",
    }

# ------------------------------------------------------
# DIRECTION ENGINE
# ------------------------------------------------------
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

# ------------------------------------------------------
# SCORE ENGINE V3 (HYBRID)
# ------------------------------------------------------
def score_v3(htf, mode):
    s = 0
    momentum = abs(htf["close_15"] - htf["prev_15"]) / max(htf["prev_15"],1)

    if mode == "quick":
        if momentum > 0.004: s += 20
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

# ------------------------------------------------------
# AI VERIFIER
# ------------------------------------------------------
async def ai_verify(signal):
    if not OPENAI_KEY:
        return {"approve": True, "tp": None, "sl": None}

    prompt = f"""
Verify this crypto signal strictly:

{json.dumps(signal)}

Return only:
{ '{"approve": true/false, "tp": null or number, "sl": null or number, "note": "short"}' }
"""
    body = {
        "model": OPENAI_MODEL,
        "messages":[
            {"role":"system","content":"Strict crypto risk manager."},
            {"role":"user","content":prompt}
        ],
        "temperature":0,
        "max_tokens":100
    }

    headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"}

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=6) as r:
                data = await r.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
    except:
        return {"approve": True, "tp": None, "sl": None}

# ------------------------------------------------------
# FINAL PROCESSOR
# ------------------------------------------------------
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
    tp = price * (1 + tp_pct/100)
    sl = price * (1 - sl_pct/100)

    signal = {
        "symbol":symbol,
        "direction":direction,
        "mode":mode,
        "price":round(price,6),
        "tp":round(tp,6),
        "sl":round(sl,6),
        "score":score
    }

    ai = await ai_verify(signal)
    if not ai.get("approve",True):
        return None

    if ai.get("tp") is not None: signal["tp"] = ai["tp"]
    if ai.get("sl") is not None: signal["sl"] = ai["sl"]

    return signal

# ------------------------------------------------------
# TELEGRAM FORMAT
# ------------------------------------------------------
def format_signal(sig):
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Entry:</b> <code>{sig['price']}</code>
<b>Score:</b> {sig['score']} / 100

🎯 <b>TP:</b> <code>{sig['tp']}</code>
🛑 <b>SL:</b> <code>{sig['sl']}</code>

⚡ Leverage: {LEVERAGE}x
⏱ Cooldown: 30m
#HybridAI #ArunSystem
"""