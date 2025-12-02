# === helpers.py PART 1 ===
import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json

DEBUG_LOG = os.getenv("DEBUG_LOG", "false").lower() == "true"

BIN_KEY = os.getenv("BINANCE_API_KEY")
BIN_SECRET = os.getenv("BINANCE_SECRET_KEY", "").encode() if os.getenv("BINANCE_SECRET_KEY") else b""
FUTURES = os.getenv("BINANCE_FUTURES_URL", "https://fapi.binance.com")

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

ENABLE_ICEBERG = os.getenv("ENABLE_ICEBERG", "true").lower() == "true"
ENABLE_VOLSWEEP = os.getenv("ENABLE_VOLSWEEP", "true").lower() == "true"
ENABLE_OI_SPIKE = os.getenv("ENABLE_OI_SPIKE", "true").lower() == "true"

ICEBERG_FACTOR = float(os.getenv("ICEBERG_FACTOR", "3.0"))
VOLSWEEP_MULT = float(os.getenv("VOLSWEEP_MULT", "3.0"))
OI_SPIKE_PCT = float(os.getenv("OI_SPIKE_PCT", "5.0"))

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

def cooldown_ok(symbol):
    return (time.time() - _last_signal.get(symbol, 0)) >= COOLDOWN

def update_cd(symbol):
    _last_signal[symbol] = time.time()

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

async def depth(session, symbol, limit=50):
    return await public(session, "/fapi/v1/depth", f"symbol={symbol}&limit={limit}")

async def ticker_24h(session, symbol):
    return await public(session, "/fapi/v1/ticker/24hr", f"symbol={symbol}")

async def kline(session, symbol, interval="1m", limit=60):
    return await public(session, "/fapi/v1/klines", f"symbol={symbol}&interval={interval}&limit={limit}")

async def trades(session, symbol, limit=500):
    return await public(session, "/fapi/v1/aggTrades", f"symbol={symbol}&limit={limit}")

async def funding_rate(session, symbol):
    r = await public(session, "/fapi/v1/fundingRate", f"symbol={symbol}&limit=1")
    try:
        return float(r[0].get("fundingRate", 0))
    except:
        return 0.0

async def oi_history(session, symbol, limit=5):
    return await public(session, "/fapi/futures/data/openInterestHist", f"symbol={symbol}&period=5m&limit={limit}")

def ema(values, length=None):
    if not values:
        return 0.0
    vals = list(values)
    if length is None:
        length = len(vals)
    k = 2/(length+1)
    e = float(vals[0])
    for v in vals[1:]:
        e = e*(1-k) + float(v)*k
    return e

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

    c15 = [float(x[4]) for x in k15]
    c1h = [float(x[4]) for x in k1h]
    c4h = [float(x[4]) for x in k4h]

    out = {
        "close_15": c15[-1],
        "ema_15": ema(c15[-15:], 15),
        "ema_50": ema(c15[-50:], 50) if len(c15)>=50 else ema(c15),
        "ema_200": ema(c15[-200:], 200) if len(c15)>=200 else ema(c15),
        "ema_1h": ema(c1h[-50:], 50),
        "ema_4h": ema(c4h[-50:], 50),
        "trend_15": "up" if c15[-1] > ema(c15[-50:], 50) else "down",
        "trend_1h": "up" if c1h[-1] > ema(c1h[-50:], 50) else "down",
        "trend_4h": "up" if c4h[-1] > ema(c4h[-50:], 50) else "down",
    }

    cache_set(cache_key, out)
    return out
# === helpers.py PART 2 ===

async def detect_iceberg(session, symbol, lookback_trades=200, large_trade_factor=ICEBERG_FACTOR):
    try:
        ts = await trades(session, symbol, limit=lookback_trades)
        if not ts:
            return False
        sizes = []
        prices = []
        for t in ts:
            try:
                p = float(t["p"]); q = float(t["q"])
            except:
                try:
                    p = float(t[4]); q = float(t[5])
                except:
                    continue
            prices.append(p); sizes.append(q)

        if len(sizes) < 20:
            return False

        sorted_sizes = sorted(sizes)
        med = sorted_sizes[len(sorted_sizes)//2]
        threshold = med * large_trade_factor

        large_idx = [i for i,s in enumerate(sizes) if s >= threshold]
        if len(large_idx) < 3:
            return False

        large_prices = [prices[i] for i in large_idx]
        avg_lp = sum(large_prices)/len(large_prices)
        if (max(large_prices)-min(large_prices))/avg_lp > 0.001:
            return False

        if DEBUG_LOG:
            print(f"[DETECT] {symbol} → ICEBERG TRUE")
        return True
    except:
        return False


async def detect_vol_sweep_1m(session, symbol, lookback=6, mult=VOLSWEEP_MULT):
    try:
        kl = await kline(session, symbol, "1m", lookback+1)
        if not kl:
            return False

        prev_vols = [float(c[5]) for c in kl[:-1]]
        last_vol = float(kl[-1][5])
        avg_prev = sum(prev_vols)/len(prev_vols)

        if last_vol >= avg_prev * mult:
            if DEBUG_LOG:
                print(f"[DETECT] {symbol} → VOL_SWEEP TRUE")
            return True
        return False
    except:
        return False


async def detect_oi_spike(session, symbol, pct=OI_SPIKE_PCT):
    try:
        oi = await oi_history(session, symbol, 3)
        if not oi or len(oi)<2:
            return False

        first = float(oi[0].get("sumOpenInterest", oi[0].get("openInterest", 0)))
        last  = float(oi[-1].get("sumOpenInterest", oi[-1].get("openInterest", 0)))

        if first > 0:
            change = (last-first)/first*100
            if change >= pct:
                if DEBUG_LOG:
                    print(f"[DETECT] {symbol} → OI_SPIKE TRUE ({change:.2f}%)")
                return True
        return False
    except:
        return False


def score_engine(htf, meta, mode):
    s = 0

    if mode=="quick":
        s += 15 if htf["trend_15"]=="up" else 0
        s += 10 if htf["ema_15"] > htf["ema_50"] else 0
        s += 12 if meta.get("imbalance",1) > 1.2 else 0
        s += 10 if meta.get("vol_quote",0) > MIN_24H_VOL*1.5 else 0
        s += 8  if abs(meta.get("funding",0)) < 0.02 else 0

    elif mode=="mid":
        s += 20 if htf["trend_1h"]=="up" else 0
        s += 12 if htf["ema_50"] > htf["ema_200"] else 0
        s += 12 if meta.get("imbalance",1) > 1.3 else 0
        s += 10 if meta.get("oi_sustained",False) else 0
        s += 8  if abs(meta.get("funding",0)) < 0.03 else 0

    else:
        s += 25 if htf["trend_4h"]=="up" else 0
        s += 15 if htf["trend_1h"]=="up" else 0
        s += 15 if htf["ema_50"] > htf["ema_200"] else 0
        s += 10 if meta.get("oi_sustained",False) else 0
        s += 10 if meta.get("imbalance",1)>1.4 else 0

    if abs(meta.get("funding",0)) > 0.05:
        s -= 15
    if meta.get("liq_risk", False):
        s -= 25

    # --- TOP3 bonuses ---
    if meta.get("iceberg"):
        s += 3 if mode=="quick" else 4 if mode=="mid" else 5

    if meta.get("vol_sweep_1m"):
        s += 4

    if meta.get("oi_spike"):
        s += 3

    return max(0, min(100, int(s)))
# === helpers.py PART 3 ===

def get_direction(htf):
    try:
        bull = htf["ema_15"] > htf["ema_50"] > htf["ema_200"] and htf["trend_1h"]=="up"
        bear = htf["ema_15"] < htf["ema_50"] < htf["ema_200"] and htf["trend_1h"]=="down"
        return "BUY" if bull else "SELL" if bear else None
    except:
        return None

async def btc_calm_check(session):
    try:
        r = await ticker_24h(session, "BTCUSDT")
        if not r:
            return True
        change = abs(float(r.get("priceChangePercent", 0)))
        return change < BTC_STABILITY_THRESHOLD
    except:
        return True


async def final_process(session, symbol, mode):

    if DEBUG_LOG:
        print(f"[CHECK] {symbol} ({mode}) scanning...")

    if not await btc_calm_check(session):
        if DEBUG_LOG:
            print(f"[SKIP] {symbol} → BTC volatile")
        return None

    htf = await get_htf(session, symbol)
    if not htf:
        return None

    ok, liq = await check_liq(session, symbol)
    if not ok:
        return None

    funding = await funding_rate(session, symbol)

    oi = await oi_history(session, symbol, 3)
    oi_sustained = False
    try:
        a = float(oi[0].get("sumOpenInterest", oi[0].get("openInterest",0)))
        b = float(oi[-1].get("sumOpenInterest", oi[-1].get("openInterest",0)))
        oi_sustained = b > a
    except:
        pass

    meta = {
        "spread_pct": liq.get("spread_pct",0),
        "imbalance": liq.get("imbalance",1),
        "vol_quote": liq.get("vol_quote",0),
        "funding": funding,
        "oi_sustained": oi_sustained,
        "liq_risk": False
    }

    # ---- Top3 detection ----
    meta["iceberg"] = await detect_iceberg(session, symbol) if ENABLE_ICEBERG else False
    meta["vol_sweep_1m"] = await detect_vol_sweep_1m(session, symbol) if ENABLE_VOLSWEEP else False
    meta["oi_spike"] = await detect_oi_spike(session, symbol) if ENABLE_OI_SPIKE else False

    direction = get_direction(htf)
    if not direction:
        return None

    score = score_engine(htf, meta, mode)
    if score < MODE_THRESH.get(mode,100):
        return None

    if not cooldown_ok(symbol):
        return None

    price = float(htf["close_15"])
    tp_pct, sl_pct = TP_SL.get(mode, (1.8, 1.0))

    if direction=="BUY":
        tp = price*(1+tp_pct/100)
        sl = price*(1-sl_pct/100)
    else:
        tp = price*(1-tp_pct/100)
        sl = price*(1+sl_pct/100)

    sig = {
        "symbol":symbol,
        "direction":direction,
        "mode":mode,
        "price":round(price,6),
        "tp":round(tp,6),
        "sl":round(sl,6),
        "score":score,
        "meta":meta
    }

    update_cd(symbol)

    if DEBUG_LOG:
        print(f"[SEND] {symbol} ({mode}) SCORE={score}")

    return sig


def format_signal(sig):
    m = sig["meta"]
    return f"""
🔥 <b>{sig['mode'].upper()} {sig['direction']} SIGNAL</b>

Pair: {sig['symbol']}
Entry: <code>{sig['price']}</code>
Score: {sig['score']}

🎯 TP → <code>{sig['tp']}</code>
🛑 SL → <code>{sig['sl']}</code>

📊 Imbalance: {m.get('imbalance')}
⚡ VolSweep1m: {m.get('vol_sweep_1m')}
🧊 Iceberg: {m.get('iceberg')}
📈 OI Spike: {m.get('oi_spike')}
💰 24h Vol: {m.get('vol_quote')}

#HybridAI #ArunSystem
"""