# helpers.py — REAL BINANCE ENGINE + DEBUG_LOG toggle
import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json

# ============================
# CONFIG & ENV
# ============================

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() == "true"

BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SECRET = os.getenv("BINANCE_SECRET_KEY", "").encode() if os.getenv("BINANCE_SECRET_KEY") else b""
FUTURES = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

AI_VERIFY_ENABLED = False   # AI off

COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "1800"))
INTERVAL = int(os.getenv("SCAN_INTERVAL_SECONDS", "8"))

MODE_THRESH = {
    "quick": int(os.getenv("THRESH_QUICK", "65")),
    "mid": int(os.getenv("THRESH_MID", "72")),
    "trend": int(os.getenv("THRESH_TREND", "78")),
}

TP_SL = {
    "quick": (1.2, 0.7),
    "mid": (1.8, 1.0),
    "trend": (2.5, 1.2),
}

LEVERAGE = int(os.getenv("LEVERAGE", "50"))

BTC_STABILITY_THRESHOLD = float(os.getenv("BTC_STABILITY_THRESHOLD", "1.2"))
MIN_24H_VOL = float(os.getenv("MIN_24H_VOL", "20000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.06"))
MAX_LIQ_DIST = float(os.getenv("MAX_LIQ_DIST", "1.0"))

VERBOSE_SCORE_MIN = int(os.getenv("VERBOSE_SCORE_MIN", "45"))

# ============================
# CACHE & COOLDOWN
# ============================

_last_signal = {}
_cache = {}
_cache_time = {}

def cache_get(key, ttl=5):
    v = _cache.get(key)
    if not v:
        return None
    if time.time() - _cache_time.get(key, 0) > ttl:
        _cache.pop(key, None)
        _cache_time.pop(key, None)
        return None
    return v

def cache_set(key, val):
    _cache[key] = val
    _cache_time[key] = time.time()

def cooldown_ok(symbol: str) -> bool:
    return (time.time() - _last_signal.get(symbol, 0)) >= COOLDOWN

def update_cd(symbol: str):
    _last_signal[symbol] = time.time()


# ============================
# API HELPERS
# ============================

def sign(query: str) -> str:
    if not BIN_SECRET:
        return ""
    return hmac.new(BIN_SECRET, query.encode(), hashlib.sha256).hexdigest()

async def public(session, path, params=""):
    url = f"{FUTURES}{path}"
    if params:
        url += f"?{params}"
    try:
        async with session.get(url, timeout=8) as r:
            return await r.json()
    except:
        return None

async def private(session, path, params=""):
    ts = int(time.time()*1000)
    q = f"{params}&timestamp={ts}" if params else f"timestamp={ts}"
    sig = sign(q)
    url = f"{FUTURES}{path}?{q}&signature={sig}"
    headers = {"X-MBX-APIKEY": BIN_KEY} if BIN_KEY else {}
    try:
        async with session.get(url, headers=headers, timeout=8) as r:
            return await r.json()
    except:
        return None


# ============================
# DATA CALLS
# ============================

async def depth(session, symbol, limit=50):
    return await public(session, "/fapi/v1/depth", f"symbol={symbol}&limit={limit}")

async def mark_price(session, symbol):
    return await public(session, "/fapi/v1/premiumIndex", f"symbol={symbol}")

async def funding_rate(session, symbol):
    r = await public(session, "/fapi/v1/fundingRate", f"symbol={symbol}&limit=1")
    try:
        if r and isinstance(r, list):
            return float(r[0].get("fundingRate", 0.0))
    except:
        return 0.0
    return 0.0

async def oi_history(session, symbol, limit=5):
    return await public(session, "/fapi/v1/openInterestHist", f"symbol={symbol}&period=5m&limit={limit}")

async def trades(session, symbol, limit=50):
    return await public(session, "/fapi/v1/aggTrades", f"symbol={symbol}&limit={limit}")

async def ticker_24h(session, symbol):
    return await public(session, "/fapi/v1/ticker/24hr", f"symbol={symbol}")

async def kline(session, symbol, interval="1m", limit=60):
    return await public(session, "/fapi/v1/klines", f"symbol={symbol}&interval={interval}&limit={limit}")


# ============================
# SAFE LAST TRADE
# ============================

async def safe_last_trade(session, symbol):
    try:
        ts = await trades(session, symbol, 50)
        if not ts or not isinstance(ts, list):
            return {"price": None, "qty": None}
        last = ts[-1]
        p = last.get("p") if isinstance(last, dict) else last[4]
        q = last.get("q") if isinstance(last, dict) else last[5]
        return {"price": float(p), "qty": float(q)}
    except:
        return {"price": None, "qty": None}


# ============================
# EMA
# ============================

def ema(values, length=None):
    if not values:
        return 0.0
    vals = list(values)
    if length is None:
        length = len(vals)
    try:
        length = int(length)
    except:
        length = len(vals)
    k = 2/(length+1)
    e = float(vals[0])
    for v in vals[1:]:
        e = e*(1-k) + float(v)*k
    return e


# ============================
# HTF Fetch
# ============================

async def get_htf(session, symbol):
    cache_key = f"htf:{symbol}"
    c = cache_get(cache_key, ttl=6)
    if c:
        return c

    k15, k1h, k4h = await asyncio.gather(
        kline(session, symbol, "15m", 60),
        kline(session, symbol, "1h", 60),
        kline(session, symbol, "4h", 60)
    )

    if not k15 or not k1h or not k4h:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → HTF fetch failed")
        return None

    try:
        c15 = [float(x[4]) for x in k15]
        c1h = [float(x[4]) for x in k1h]
        c4h = [float(x[4]) for x in k4h]
    except:
        return None

    ema15 = ema(c15[-15:], 15)
    ema50 = ema(c15[-50:], 50) if len(c15)>=50 else ema15
    ema200 = ema(c15[-200:], 200) if len(c15)>=200 else ema50

    ema1h = ema(c1h[-50:], 50) if len(c1h)>=50 else c1h[-1]
    ema4h = ema(c4h[-50:], 50) if len(c4h)>=50 else c4h[-1]

    out = {
        "close_15": c15[-1],
        "ema_15": ema15,
        "ema_50": ema50,
        "ema_200": ema200,
        "ema_1h": ema1h,
        "ema_4h": ema4h,
        "trend_15": "up" if c15[-1] > ema50 else "down",
        "trend_1h": "up" if c1h[-1] > ema1h else "down",
        "trend_4h": "up" if c4h[-1] > ema4h else "down",
    }

    cache_set(cache_key, out)
    return out


# ============================
# BTC calm check
# ============================

async def btc_calm_check(session):
    try:
        r = await ticker_24h(session, "BTCUSDT")
        if not r:
            return True
        change = abs(float(r.get("priceChangePercent", 0)))
        return change < BTC_STABILITY_THRESHOLD
    except:
        return True


# ============================
# Liquidity, Spread, Imbalance
# ============================

def compute_spread_pct(bid, ask):
    if ask <= 0: return 999
    return abs(ask-bid)/((ask+bid)/2)*100

def depth_imbalance(d):
    try:
        bids = d.get("bids", [])[:10]
        asks = d.get("asks", [])[:10]
        bidv = sum(float(b[0])*float(b[1]) for b in bids)
        askv = sum(float(a[0])*float(a[1]) for a in asks)
        if askv==0: return 999
        return bidv/askv
    except:
        return 1

async def check_liq(session, symbol):
    d = await depth(session, symbol, 20)
    t = await ticker_24h(session, symbol)
    if not d or not t:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → no depth/ticker")
        return False, "no_data"
    try:
        bid = float(d["bids"][0][0])
        ask = float(d["asks"][0][0])
    except:
        return False, "bad_depth"

    spread = compute_spread_pct(bid, ask)
    if spread > MAX_SPREAD_PCT:
        return False, "wide_spread"

    vol = float(t.get("quoteVolume", 0))
    if vol < MIN_24H_VOL:
        return False, "low_volume"

    imb = depth_imbalance(d)
    return True, {"spread_pct": spread, "imbalance": imb, "vol_quote": vol}


# ============================
# Direction
# ============================

def get_direction(htf):
    try:
        bull = htf["ema_15"] > htf["ema_50"] > htf["ema_200"] and htf["trend_1h"]=="up"
        bear = htf["ema_15"] < htf["ema_50"] < htf["ema_200"] and htf["trend_1h"]=="down"
        if bull: return "BUY"
        if bear: return "SELL"
    except:
        return None
    return None


# ============================
# Score Engine
# ============================

def score_engine(htf, meta, mode):
    s = 0
    if mode=="quick":
        s += 15 if htf["trend_15"]=="up" else 0
        s += 10 if htf["ema_15"]>htf["ema_50"] else 0
        s += 12 if meta["imbalance"]>1.2 else 0
        s += 10 if meta["vol_quote"]>MIN_24H_VOL*1.5 else 0
        s += 8 if abs(meta["funding"])<0.02 else 0
    elif mode=="mid":
        s += 20 if htf["trend_1h"]=="up" else 0
        s += 12 if htf["ema_50"]>htf["ema_200"] else 0
        s += 12 if meta["imbalance"]>1.3 else 0
        s += 10 if meta["oi_sustained"] else 0
        s += 8 if abs(meta["funding"])<0.03 else 0
    else:
        s += 25 if htf["trend_4h"]=="up" else 0
        s += 15 if htf["trend_1h"]=="up" else 0
        s += 15 if htf["ema_50"]>htf["ema_200"] else 0
        s += 10 if meta["oi_sustained"] else 0
        s += 10 if meta["imbalance"]>1.4 else 0

    if abs(meta["funding"])>0.05: s-=15
    if meta["liq_risk"]: s-=25

    return max(0, min(100, int(s)))


# ============================
# Final Processing
# ============================

async def final_process(session, symbol, mode):
    if DEBUG_LOG:
        print(f"[CHECK] {symbol} ({mode}) scanning...")

    # BTC calm
    if not await btc_calm_check(session):
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → BTC volatile")
        return None

    # HTF
    htf = await get_htf(session, symbol)
    if not htf:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → HTF fail")
        return None

    # liquidity
    ok, meta = await check_liq(session, symbol)
    if not ok:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → {meta}")
        return None

    # funding
    funding = await funding_rate(session, symbol)

    # OI
    oi = await oi_history(session, symbol, 3)
    oi_sustained = False
    try:
        if oi and len(oi)>=2:
            a=float(oi[0].get("sumOpenInterest",0))
            b=float(oi[-1].get("sumOpenInterest",0))
            oi_sustained = b>a
    except:
        oi_sustained=False

    # liq distance
    d = await depth(session, symbol, 20)
    mark = await mark_price(session, symbol)
    liq_risk=False
    try:
        mp = float(mark[0].get("markPrice",0)) if isinstance(mark,list) else float(mark.get("markPrice",0))
        bid=float(d["bids"][0][0]); ask=float(d["asks"][0][0])
        dist=min(abs(mp-bid),abs(ask-mp))/mp*100
        if dist<MAX_LIQ_DIST: liq_risk=True
    except:
        liq_risk=False

    meta = {
        "spread_pct": meta.get("spread_pct"),
        "imbalance": meta.get("imbalance"),
        "vol_quote": meta.get("vol_quote"),
        "funding": funding,
        "oi_sustained": oi_sustained,
        "liq_risk": liq_risk
    }

    # direction
    direction = get_direction(htf)
    if not direction:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → No direction")
        return None

    # score
    score = score_engine(htf, meta, mode)
    if score>=VERBOSE_SCORE_MIN and DEBUG_LOG:
        print(f"[SCORE] {symbol} → {score} | RAW={meta}")

    # threshold
    if score < MODE_THRESH[mode]:
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} ({mode}) → score {score} < {MODE_THRESH[mode]}")
        return None

    # cooldown
    if not cooldown_ok(symbol):
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} ({mode}) → cooldown active")
        return None

    # final price
    price = float(htf["close_15"])
    tp_pct, sl_pct = TP_SL[mode]

    if direction=="BUY":
        tp = price*(1+tp_pct/100)
        sl = price*(1-sl_pct/100)
    else:
        tp = price*(1-tp_pct/100)
        sl = price*(1+sl_pct/100)

    sig = {
        "symbol": symbol,
        "direction": direction,
        "mode": mode,
        "price": round(price,6),
        "tp": round(tp,6),
        "sl": round(sl,6),
        "score": score,
        "meta": meta
    }

    update_cd(symbol)

    if DEBUG_LOG:
        print(f"[SEND] {symbol} ({mode}) → DIR={direction}, SCORE={score}, TP={round(tp,6)}, SL={round(sl,6)}")

    return sig


# ============================
# Telegram Formatting
# ============================

def format_signal(sig):
    m=sig["meta"]
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Entry:</b> <code>{sig['price']}</code>
<b>Score:</b> {sig['score']}

🎯 TP: <code>{sig['tp']}</code>
🛑 SL: <code>{sig['sl']}</code>

📊 Imbalance: {m['imbalance']}  
📉 Spread%: {m['spread_pct']}
💰 24h Vol: {int(m['vol_quote'])}
⚡ Leverage: {LEVERAGE}x

#HybridAI #ArunSystem
"""