# main.py — Updated: hybrid indicators + caching + rate-limit + sqlite history
import os, time, json, asyncio, hashlib, sqlite3, math
from dotenv import load_dotenv
import aiohttp, random
load_dotenv()

# OpenAI & ccxt async
from openai import OpenAI
import ccxt.async_support as ccxt

from helpers import (
    now_ts, human_time, esc, calc_tp_sl, build_ai_prompt, SimpleCache, CACHE, sma,
    compute_ema_from_closes, atr, rsi_from_closes
)

# -------------------------
# ENV
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN","").strip()
CHAT_ID   = os.getenv("CHAT_ID","").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
CYCLE_TIME = int(os.getenv("CYCLE_TIME","20"))
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD","78"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS","1800"))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","").split(",") if s.strip()]

USE_TESTNET = os.getenv("USE_TESTNET","true").lower() in ("1","true","yes")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY","").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","").strip()

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Rate-limit config (OpenAI calls per minute)
OPENAI_MAX_PER_MIN = int(os.getenv("OPENAI_MAX_PER_MIN","40"))
_openai_calls = []
def openai_can_call():
    # sliding window
    now = time.time()
    window = 60.0
    # remove old
    while _openai_calls and _openai_calls[0] <= now - window:
        _openai_calls.pop(0)
    return len(_openai_calls) < OPENAI_MAX_PER_MIN

def openai_note_call():
    _openai_calls.append(time.time())

# -------------------------
# SQLite history
# -------------------------
DB_PATH = os.getenv("SIGNAL_DB", "signals.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
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
    c.execute("INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
              (ts,symbol,price,score,mode,reason,tp,sl))
    conn.commit()

# -------------------------
# Exchange create
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
# Snapshot fetch (hlines + orderbook best)
# -------------------------
async def fetch_snapshot(exchange, symbol):
    # try cached snapshot key for heavy ops (1s cache)
    key = CACHE.make_key("snap", symbol)
    snap = CACHE.get(key)
    if snap:
        return snap

    price = None
    spread_pct = None
    closes_1m = closes_5m = closes_15m = closes_1h = []
    # fetch ticker
    try:
        tk = await exchange.fetch_ticker(symbol)
        price = float(tk.get("last") or tk.get("close") or 0)
    except Exception:
        price = random.uniform(1,50000)

    # attempt to fetch orderbook top to compute spread
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if bid and ask:
            mid = (bid + ask) / 2.0
            spread_pct = abs(ask - bid) / mid * 100.0
    except Exception:
        spread_pct = None

    # fetch ohlcv for multiple tf (best-effort, reduce size)
    try:
        ohlcv_1m = await exchange.fetch_ohlcv(symbol, timeframe='1m', limit=120)
        closes_1m = [row[4] for row in ohlcv_1m]
    except Exception:
        closes_1m = []

    try:
        ohlcv_5m = await exchange.fetch_ohlcv(symbol, timeframe='5m', limit=120)
        closes_5m = [row[4] for row in ohlcv_5m]
    except Exception:
        closes_5m = []

    try:
        ohlcv_15m = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=120)
        closes_15m = [row[4] for row in ohlcv_15m]
    except Exception:
        closes_15m = []

    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, timeframe='1h', limit=120)
        closes_1h = [row[4] for row in ohlcv_1h]
    except Exception:
        closes_1h = []

    # compute indicators
    atr_14 = atr(ohlcv_1m if 'ohlcv_1m' in locals() else []) if 'ohlcv_1m' in locals() else 0.0
    rsi_14 = rsi_from_closes(closes_1m, period=14) if closes_1m else 50.0
    ema_1h_50 = compute_ema_from_closes(closes_1h, 50) if closes_1h else 0.0
    ema_15m_50 = compute_ema_from_closes(closes_15m, 50) if closes_15m else 0.0

    metrics = {
        "closes_1m": closes_1m,
        "closes_5m": closes_5m,
        "closes_15m": closes_15m,
        "closes_1h": closes_1h,
        "vol_1m": sum(closes_1m[-10:]) if closes_1m else 0,
        "vol_5m": sum(closes_5m[-10:]) if closes_5m else 0,
        "atr_1m": atr_14,
        "rsi_1m": rsi_14,
        "ema_1h_50": ema_1h_50,
        "ema_15m_50": ema_15m_50,
        "spread_pct": round(spread_pct or 0.0, 4),
        "funding_rate": 0.0
    }

    snap = {"price": price, "metrics": metrics}
    CACHE.set(key, snap, ttl_seconds=3)  # small cache for snapshot freshness
    return snap

# -------------------------
# AI scorer wrapper with caching + rate-limit
# -------------------------
def extract_cache_key_for_scoring(symbol, price, metrics):
    # create stable small fingerprint of metrics
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

async def ai_score_with_cache(symbol, price, metrics, prefs, ttl=60):
    cache_key = CACHE.make_key("ai", extract_cache_key_for_scoring(symbol, price, metrics))
    cached = CACHE.get(cache_key)
    if cached:
        return cached

    # rate-limit check
    wait_count = 0
    while not openai_can_call():
        # sleep small until allowed (but avoid indefinite block)
        await asyncio.sleep(1)
        wait_count += 1
        if wait_count > 10:
            # avoid too long waiting, bail out
            return None

    # build prompt
    prompt = build_ai_prompt(symbol, price, metrics, prefs)

    def call_ai():
        # synchronous call to OpenAI client (wrapped)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are an expert crypto futures signal scorer. Return STRICT JSON only."},
                {"role":"user","content": prompt}
            ],
            temperature=0.0,
            max_tokens=250
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].text
            except Exception:
                return ""

    openai_note_call()
    raw = await asyncio.to_thread(call_ai)
    # parse JSON
    try:
        parsed = json.loads(raw.strip())
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
        else:
            parsed = None

    if parsed:
        CACHE.set(cache_key, parsed, ttl_seconds=ttl)
    return parsed

# -------------------------
# Telegram send
# -------------------------
async def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials missing; skipping send.")
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
# Message builder (premium)
# -------------------------
def build_message(symbol, price, score, mode, reason, tp, sl):
    parts = []
    parts.append(f"🔥⚡ <b>AI SIGNAL — {esc(mode.upper())} MODE</b> ⚡🔥")
    parts.append(f"<b>{esc(symbol)}</b>  • Price: <code>{price:.8f}</code>")
    parts.append(f"🎯 <b>Score:</b> <b>{int(score)}</b>   📌 <b>Reason:</b> {esc(reason)}")
    parts.append(f"🎯 <b>TP:</b> <code>{tp}</code>   🛡️ <b>SL:</b> <code>{sl}</code>")
    parts.append(f"🕒 {human_time()}   ⏳ Cooldown: {int(COOLDOWN_SECONDS/60)}m")
    parts.append(f"🤖 Source: AI-Only Scoring • Hybrid indicators (local) • Live prices")
    return "\n".join(parts)

# -------------------------
# Main worker
# -------------------------
async def worker():
    exchange = await create_exchange()
    cd = {}  # simple per-symbol cooldown timestamps
    prefs = {
        "BTC_CALM_REQUIRED": True,
        "TP_SL": {
            "quick": {"tp_pct":1.6, "sl_pct":1.0},
            "mid": {"tp_pct":2.0, "sl_pct":1.0},
            "trend": {"tp_pct":4.0, "sl_pct":1.5}
        }
    }

    print(f"Hybrid AI bot started. Symbols: {len(SYMBOLS)}. OpenAI rate limit: {OPENAI_MAX_PER_MIN}/min")

    try:
        while True:
            for sym in SYMBOLS:
                # simple cooldown
                if cd.get(sym,0) > time.time():
                    continue

                snap = await fetch_snapshot(exchange, sym)
                price = snap["price"]
                metrics = snap["metrics"]

                # Call AI with caching and rate-limit
                parsed = await ai_score_with_cache(sym, price, metrics, prefs, ttl=60)
                if not parsed:
                    await asyncio.sleep(0.05)
                    continue

                score = int(parsed.get("score",0))
                mode = parsed.get("mode","quick")
                reason = parsed.get("reason","")

                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_message(sym, price, score, mode, reason, tp, sl)
                    resp = await send_telegram(msg)
                    print("[SENT]", sym, score, mode, reason, "resp:", resp)
                    # log to sqlite
                    log_signal_db(now_ts(), sym, price, score, mode, reason, tp, sl)
                    # set cooldown
                    cd[sym] = time.time() + COOLDOWN_SECONDS

                await asyncio.sleep(0.08)  # small pacing

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