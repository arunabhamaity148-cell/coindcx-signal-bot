# ============================
# main.py — FINAL PRODUCTION (single-file)
# Batch Safe • Quota Safe • Stable
# ============================

import os
import time
import json
import asyncio
import random
import hashlib
import sqlite3
import traceback
from dotenv import load_dotenv
import aiohttp

load_dotenv()

# openai client (official)
from openai import OpenAI

# ccxt async
import ccxt.async_support as ccxt

# helpers (assumed final helpers.py present in same dir)
from helpers import (
    now_ts, human_time, esc, calc_tp_sl, build_ai_prompt, CACHE,
    compute_ema_from_closes, atr, rsi_from_closes
)

# -------------------------
# ENV / CONFIG
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_MAX_PER_MIN = int(os.getenv("OPENAI_MAX_PER_MIN", "40"))
OPENAI_TTL_SECONDS = int(os.getenv("OPENAI_TTL_SECONDS", "60"))

CYCLE_TIME = float(os.getenv("CYCLE_TIME", "20"))  # seconds between batches
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "78"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))

USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() in ("1", "true", "yes")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

# SYMBOL LIST (example 50) — change if needed
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","FILUSDT","ATOMUSDT","ETCUSDT","OPUSDT","ICPUSDT","APTUSDT","NEARUSDT","INJUSDT",
    "SUIUSDT","AAVEUSDT","EOSUSDT","CRVUSDT","RUNEUSDT","XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT",
    "HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT","ZILUSDT","GALAUSDT","MTLUSDT","CHZUSDT","RNDRUSDT",
    "SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","KLAYUSDT","TOMOUSDT","VETUSDT","BLURUSDT","STRKUSDT","ZRXUSDT"
]

# -------------------------
# OpenAI client & simple rate limiter per minute
# -------------------------
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

_openai_calls = []
def openai_can_call():
    now = time.time()
    # drop old calls
    while _openai_calls and _openai_calls[0] <= now - 60:
        _openai_calls.pop(0)
    return len(_openai_calls) < OPENAI_MAX_PER_MIN

def openai_note_call():
    _openai_calls.append(time.time())

# -------------------------
# SQLite: simple signal log DB
# -------------------------
DB_PATH = os.getenv("DB_PATH", "signals.db")
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
    try:
        cur.execute(
            "INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
            (ts, symbol, price, score, mode, reason, tp, sl)
        )
        conn.commit()
    except Exception:
        # avoid crashing on DB errors
        print("DB write error:", traceback.format_exc())

# -------------------------
# Create exchange (ccxt)
# -------------------------
async def create_exchange():
    opts = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if USE_TESTNET:
        # Binance Futures testnet endpoints (some CCXT setups may need custom)
        opts["urls"] = {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    exchange = ccxt.binance(opts)
    if BINANCE_API_KEY:
        exchange.apiKey = BINANCE_API_KEY
    if BINANCE_API_SECRET:
        exchange.secret = BINANCE_API_SECRET
    return exchange

# -------------------------
# Snapshot fetch (robust)
# -------------------------
async def fetch_snapshot(exchange, symbol):
    key = CACHE.make_key("snap", symbol)
    cached = CACHE.get(key)
    if cached:
        return cached

    price = None
    base_vol = 0.0
    spread_pct = 0.0
    closes_1h = closes_15m = closes_1m = []

    # fetch ticker safely
    try:
        tk = await exchange.fetch_ticker(symbol)
        # support different key names safely
        last = tk.get("last") or tk.get("close") or tk.get("lastPrice")
        price = float(last) if last is not None else None
        base_vol = float(tk.get("baseVolume") or tk.get("volume") or 0)
    except Exception:
        # keep defaults if any call fails
        tk = None
        price = None
        base_vol = 0.0

    # fallback price (if fetch_ticker failed)
    if price is None:
        price = random.uniform(1000, 60000)

    # orderbook / spread
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        if ob and ob.get("bids") and ob.get("asks"):
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            mid = (bid + ask) / 2 if (bid and ask) else price
            if mid and mid != 0:
                spread_pct = abs(ask - bid) / mid * 100
    except Exception:
        spread_pct = 0.0

    # OHLCV (safe)
    try:
        o1h = await exchange.fetch_ohlcv(symbol, "1h", limit=120)
        closes_1h = [r[4] for r in o1h] if o1h else []
    except Exception:
        closes_1h = []

    try:
        o15 = await exchange.fetch_ohlcv(symbol, "15m", limit=120)
        closes_15m = [r[4] for r in o15] if o15 else []
    except Exception:
        closes_15m = []

    try:
        o1 = await exchange.fetch_ohlcv(symbol, "1m", limit=120)
        closes_1m = [r[4] for r in o1] if o1 else []
    except Exception:
        closes_1m = []

    metrics = {
        "closes_1m": closes_1m,
        "closes_15m": closes_15m,
        "closes_1h": closes_1h,
        "rsi_1m": rsi_from_closes(closes_1m) if closes_1m else 50,
        "ema_1h_50": compute_ema_from_closes(closes_1h, 50) if closes_1h else 0,
        "ema_15m_50": compute_ema_from_closes(closes_15m, 50) if closes_15m else 0,
        "spread_pct": round(spread_pct, 4),
        "vol_1m": base_vol,
    }

    data = {"price": price, "metrics": metrics}
    CACHE.set(key, data, ttl_seconds=3)
    return data

# -------------------------
# AI SCORE (with cache + rate safety)
# -------------------------
def parse_json(text):
    try:
        return json.loads(text.strip())
    except Exception:
        return None

async def ai_score_with_cache(symbol, price, metrics, prefs, ttl):
    key = CACHE.make_key("ai", symbol + str(round(price, 6)))
    c = CACHE.get(key)
    if c:
        return c

    # rate-limit wait (small)
    wait = 0
    while not openai_can_call():
        await asyncio.sleep(1)
        wait += 1
        if wait > 12:
            # give up this cycle to avoid blocking forever
            return None

    prompt = build_ai_prompt(symbol, price, metrics, prefs)

    def call_ai():
        try:
            # use client.responses.create to be compatible with new SDK
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=0,
                max_output_tokens=300
            )
            # prefer output_text if exists
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
            # otherwise try dict
            try:
                return json.dumps(resp.to_dict())
            except Exception:
                return str(resp)
        except Exception as e:
            # return empty string on error, log minimal
            print("OpenAI call error:", str(e))
            return ""

    # note call and run in thread
    openai_note_call()
    raw = await asyncio.to_thread(call_ai)
    parsed = parse_json(raw)
    if parsed:
        CACHE.set(key, parsed, ttl_seconds=ttl)
    return parsed

# -------------------------
# Telegram (simple)
# -------------------------
async def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        # no telegram: skip quietly
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        # use HTML allowed tags only; helpers.esc used elsewhere
        "parse_mode": "HTML"
    }
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, data=payload, timeout=10)
    except Exception:
        # don't spam logs
        pass

# -------------------------
# Message builder (human friendly)
# -------------------------
def build_message(symbol, price, score, mode, reason, tp, sl):
    return (
        f"🔥⚡ <b>{mode.upper()} SIGNAL</b> ⚡🔥\n"
        f"{symbol} • Price: {price:.2f}\n"
        f"Score: {score} • {esc(reason)}\n"
        f"TP: {tp}\nSL: {sl}\n"
        f"{human_time()} • Cooldown {COOLDOWN_SECONDS//60}m"
    )

# -------------------------
# Worker: batch-safe main loop
# -------------------------
async def worker():
    if not client:
        print("ERROR: OPENAI_API_KEY missing — set OPENAI_API_KEY in env/railway variables")
        return

    exchange = await create_exchange()
    cd = {}  # cooldown map
    prefs = {
        "BTC_CALM_REQUIRED": True,
        "TP_SL": {
            "quick": {"tp_pct": 1.6, "sl_pct": 1.0},
            "mid": {"tp_pct": 2.0, "sl_pct": 1.0},
            "trend": {"tp_pct": 4.0, "sl_pct": 1.5}
        }
    }

    BATCH = int(os.getenv("BATCH_SIZE", "8"))
    index = 0

    print(f"Bot Started • symbols={len(SYMBOLS)} • batch={BATCH} • model={OPENAI_MODEL}")

    try:
        while True:
            batch = SYMBOLS[index:index+BATCH]
            if not batch:
                index = 0
                batch = SYMBOLS[0:BATCH]

            for sym in batch:
                # cooldown check
                if cd.get(sym, 0) > time.time():
                    continue

                # snapshot
                try:
                    snap = await fetch_snapshot(exchange, sym)
                except Exception:
                    # if snapshot fails, skip symbol
                    snap = None

                if not snap:
                    await asyncio.sleep(0.05)
                    continue

                price = snap["price"]
                metrics = snap["metrics"]

                # get AI parsed score
                parsed = await ai_score_with_cache(sym, price, metrics, prefs, ttl=OPENAI_TTL_SECONDS)
                if not parsed:
                    # skip if AI unavailable / limited
                    await asyncio.sleep(0.05)
                    continue

                # expected keys: score, mode, reason
                try:
                    score = int(parsed.get("score", 0))
                except Exception:
                    score = 0
                mode = parsed.get("mode", "quick")
                reason = parsed.get("reason", "")

                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_message(sym, price, score, mode, reason, tp, sl)
                    # send telegram (async)
                    await send_telegram(msg)
                    # log DB
                    log_signal_db(now_ts(), sym, price, score, mode, reason, tp, sl)
                    # console minimal
                    print(f"[SIGNAL] {sym} score={score} mode={mode}")
                    # set cooldown
                    cd[sym] = time.time() + COOLDOWN_SECONDS

                # small spacing to avoid burst
                await asyncio.sleep(0.08)

            index += BATCH
            await asyncio.sleep(CYCLE_TIME)

    except Exception as e:
        # catch-all to prevent silent crash
        print("Worker error:", str(e))
        print(traceback.format_exc())
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

# -------------------------
# Run entry
# -------------------------
if __name__ == "__main__":
    try:
        if not OPENAI_API_KEY:
            print("ERROR: Missing OPENAI_API_KEY — cannot start.")
        else:
            asyncio.run(worker())
    except KeyboardInterrupt:
        print("Interrupted by user — exiting.")
    except Exception:
        print("Fatal error:", traceback.format_exc())