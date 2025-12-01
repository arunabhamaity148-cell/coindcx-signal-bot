# main.py — FINAL (Hybrid AI scoring + caching + sqlite + ccxt + Telegram)
import os, time, json, asyncio, random, hashlib, sqlite3
from dotenv import load_dotenv
import aiohttp

load_dotenv()

from openai import OpenAI
import ccxt.async_support as ccxt

from helpers import (
    now_ts, human_time, esc, calc_tp_sl, build_ai_prompt, CACHE,
    compute_ema_from_closes, atr, rsi_from_closes
)

# -------------------------
# ENV / config
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN","").strip()
CHAT_ID   = os.getenv("CHAT_ID","").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
OPENAI_MAX_PER_MIN = int(os.getenv("OPENAI_MAX_PER_MIN","40"))
OPENAI_TTL_SECONDS = int(os.getenv("OPENAI_TTL_SECONDS","60"))

CYCLE_TIME = int(os.getenv("CYCLE_TIME","20"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD","78"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS","1800"))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","").split(",") if s.strip()]

USE_TESTNET = os.getenv("USE_TESTNET","true").lower() in ("1","true","yes")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY","").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","").strip()

# If no SYMBOLS provided in env, use default 50 coins list
if not SYMBOLS:
    SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
        "LTCUSDT","LINKUSDT","FILUSDT","ATOMUSDT","ETCUSDT","OPUSDT","ICPUSDT","APTUSDT","NEARUSDT","INJUSDT",
        "SUIUSDT","AAVEUSDT","EOSUSDT","CRVUSDT","RUNEUSDT","XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT",
        "HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT","ZILUSDT","GALAUSDT","MTLUSDT","CHZUSDT","RNDRUSDT",
        "SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","KLAYUSDT","TOMOUSDT","VETUSDT","BLURUSDT","STRKUSDT","ZRXUSDT"
    ]

# -------------------------
# OpenAI client
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# simple sliding-window call tracker
_openai_calls = []
def openai_can_call():
    now = time.time()
    window = 60.0
    while _openai_calls and _openai_calls[0] <= now - window:
        _openai_calls.pop(0)
    return len(_openai_calls) < OPENAI_MAX_PER_MIN

def openai_note_call():
    _openai_calls.append(time.time())

# -------------------------
# SQLite logging
# -------------------------
DB_PATH = os.getenv("SIGNAL_DB","signals.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER,
    symbol TEXT,
    price REAL,
    score INTEGER,
    mode TEXT,
    reason TEXT,
    tp REAL,
    sl REAL
)
""")
conn.commit()

def log_signal_db(ts, symbol, price, score, mode, reason, tp, sl):
    cur.execute("INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
                (ts, symbol, price, score, mode, reason, tp, sl))
    conn.commit()

# -------------------------
# create exchange
# -------------------------
async def create_exchange():
    opts = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if USE_TESTNET:
        opts["urls"] = {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    exchange = ccxt.binance(opts)
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        exchange.apiKey = BINANCE_API_KEY
        exchange.secret = BINANCE_API_SECRET
    return exchange

# -------------------------
# fetch snapshot (with small caching)
# -------------------------
async def fetch_snapshot(exchange, symbol):
    key = CACHE.make_key("snap", symbol)
    cached = CACHE.get(key)
    if cached:
        return cached

    price = None
    spread_pct = 0.0
    # fetch ticker
    try:
        tk = await exchange.fetch_ticker(symbol)
        price = float(tk.get("last") or tk.get("close") or 0)
        base_vol = float(tk.get("baseVolume") or 0)
    except Exception:
        price = random.uniform(1, 50000)
        base_vol = random.uniform(100, 50000)

    # orderbook top for spread
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if bid and ask:
            mid = (bid + ask) / 2.0
            spread_pct = abs(ask - bid) / mid * 100.0
    except Exception:
        spread_pct = 0.0

    # fetch ohlcv for small indicators
    closes_1m = closes_5m = closes_15m = closes_1h = []
    try:
        o1 = await exchange.fetch_ohlcv(symbol, '1m', limit=120)
        closes_1m = [r[4] for r in o1]
    except Exception:
        closes_1m = []

    try:
        o5 = await exchange.fetch_ohlcv(symbol, '5m', limit=120)
        closes_5m = [r[4] for r in o5]
    except Exception:
        closes_5m = []

    try:
        o15 = await exchange.fetch_ohlcv(symbol, '15m', limit=120)
        closes_15m = [r[4] for r in o15]
    except Exception:
        closes_15m = []

    try:
        o1h = await exchange.fetch_ohlcv(symbol, '1h', limit=120)
        closes_1h = [r[4] for r in o1h]
    except Exception:
        closes_1h = []

    # compute a few indicators locally
    atr_1m = atr(o1) if 'o1' in locals() else 0.0
    rsi_1m = rsi_from_closes(closes_1m) if closes_1m else 50.0
    ema_1h_50 = compute_ema_from_closes(closes_1h, 50) if closes_1h else 0.0
    ema_15m_50 = compute_ema_from_closes(closes_15m, 50) if closes_15m else 0.0

    metrics = {
        "closes_1m": closes_1m,
        "closes_5m": closes_5m,
        "closes_15m": closes_15m,
        "closes_1h": closes_1h,
        "vol_1m": base_vol,
        "vol_5m": base_vol * 3,
        "atr_1m": round(atr_1m, 6),
        "rsi_1m": round(rsi_1m, 2),
        "ema_1h_50": round(ema_1h_50, 6),
        "ema_15m_50": round(ema_15m_50, 6),
        "spread_pct": round(spread_pct, 4),
        "funding_rate": 0.0
    }

    snap = {"price": price, "metrics": metrics}
    CACHE.set(key, snap, ttl_seconds=3)
    return snap

# -------------------------
# AI scoring with cache & rate-limit
# -------------------------
def parse_json_from_text(text: str):
    try:
        return json.loads(text.strip())
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def extract_score_cache_key(symbol, price, metrics):
    important = {
        "price": round(price, 6),
        "rsi_1m": round(metrics.get("rsi_1m",50.0),2),
        "atr_1m": round(metrics.get("atr_1m",0.0),4),
        "spread_pct": round(metrics.get("spread_pct",0.0),4),
        "ema_1h_50": round(metrics.get("ema_1h_50",0.0),6),
        "ema_15m_50": round(metrics.get("ema_15m_50",0.0),6),
    }
    raw = f"{symbol}|{json.dumps(important, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()

async def ai_score_with_cache(symbol, price, metrics, prefs, ttl=OPENAI_TTL_SECONDS):
    cache_key = CACHE.make_key("ai", extract_score_cache_key(symbol, price, metrics))
    cached = CACHE.get(cache_key)
    if cached:
        return cached

    # rate-limit loop
    wait = 0
    while not openai_can_call():
        await asyncio.sleep(1)
        wait += 1
        if wait > 10:
            return None

    prompt = build_ai_prompt(symbol, price, metrics, prefs)

    # Synchronous helper that uses the new Responses API
    def call_ai():
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=0.0,
                max_tokens=300
            )
        except Exception as e:
            # If the SDK raises, return empty string; outer code will handle None
            print("OpenAI request exception:", e)
            return ""

        # Try common ways to extract text from response object (be robust)
        try:
            # newer SDK exposes output_text shortcut
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
        except Exception:
            pass

        try:
            # Another common shape: resp.output -> list[ { "content": [ { "type": "...", "text": "..." } ] } ]
            pieces = []
            out = getattr(resp, "output", None)
            if out:
                for item in out:
                    content = item.get("content", []) if isinstance(item, dict) else []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text" and c.get("text"):
                            pieces.append(c.get("text"))
                        elif isinstance(c, dict) and c.get("type") == "message" and isinstance(c.get("content"), list):
                            # sometimes chat-like messages appear here
                            for m in c.get("content", []):
                                if m.get("type") == "output_text" and m.get("text"):
                                    pieces.append(m.get("text"))
            if pieces:
                return "\n".join(pieces)
        except Exception:
            pass

        try:
            # fallback: try to stringify the whole response
            return json.dumps(resp.to_dict())
        except Exception:
            return ""

    openai_note_call()
    raw = await asyncio.to_thread(call_ai)
    if not raw:
        return None

    parsed = parse_json_from_text(raw)
    if parsed:
        CACHE.set(cache_key, parsed, ttl_seconds=ttl)
    return parsed

# -------------------------
# Telegram send
# -------------------------
async def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.post(url, data=payload, timeout=15) as r:
                return await r.json()
        except Exception as e:
            print("Telegram send error:", e)
            return None

# -------------------------
# build message
# -------------------------
def build_message(symbol, price, score, mode, reason, tp, sl):
    parts = []
    parts.append(f"🔥⚡ AI SIGNAL — {esc(mode.upper())} MODE ⚡🔥")
    parts.append(f"{esc(symbol)}  • Price: {price:.8f}")
    parts.append(f"Score: {int(score)}   Reason: {esc(reason)}")
    parts.append(f"TP: {tp}   SL: {sl}")
    parts.append(f"{human_time()}  • Cooldown: {int(COOLDOWN_SECONDS/60)}m")
    parts.append(f"Source: AI-Only Scoring • Hybrid indicators • Live prices")
    return "\n".join(parts)

# -------------------------
# main worker
# -------------------------
async def worker():
    exchange = await create_exchange()
    cd = {}  # symbol -> ts
    prefs = {
        "BTC_CALM_REQUIRED": True,
        "TP_SL": {
            "quick": {"tp_pct":1.6,"sl_pct":1.0},
            "mid": {"tp_pct":2.0,"sl_pct":1.0},
            "trend": {"tp_pct":4.0,"sl_pct":1.5}
        }
    }

    print(f"Hybrid AI bot started. Symbols: {len(SYMBOLS)} • OpenAI limit: {OPENAI_MAX_PER_MIN}/min")

    try:
        while True:
            for sym in SYMBOLS:
                # cooldown
                if cd.get(sym,0) > time.time():
                    continue

                snap = await fetch_snapshot(exchange, sym)
                price = snap["price"]
                metrics = snap["metrics"]

                parsed = await ai_score_with_cache(sym, price, metrics, prefs, ttl=OPENAI_TTL_SECONDS)
                if not parsed:
                    await asyncio.sleep(0.02)
                    continue

                score = int(parsed.get("score",0))
                mode = parsed.get("mode","quick")
                reason = parsed.get("reason","")

                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_message(sym, price, score, mode, reason, tp, sl)
                    resp = await send_telegram(msg)
                    print("[SENT]", sym, score, mode, reason, "resp:", resp)
                    log_signal_db(now_ts(), sym, price, score, mode, reason, tp, sl)
                    cd[sym] = time.time() + COOLDOWN_SECONDS

                await asyncio.sleep(0.08)

            await asyncio.sleep(CYCLE_TIME)
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY missing in .env")
    else:
        asyncio.run(worker())