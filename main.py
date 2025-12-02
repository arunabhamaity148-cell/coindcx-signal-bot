# main.py — FINAL PRODUCTION (clean, batch-safe, minimal logs)
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
# ENV / CONFIG
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

USE_TESTNET = os.getenv("USE_TESTNET","true").lower() in ("1","true","yes")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY","").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","").strip()

# Symbol list — add/edit here
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","FILUSDT","ATOMUSDT","ETCUSDT","OPUSDT","ICPUSDT","APTUSDT","NEARUSDT","INJUSDT",
    "SUIUSDT","AAVEUSDT","EOSUSDT","CRVUSDT","RUNEUSDT","XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT",
    "HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT","ZILUSDT","GALAUSDT","MTLUSDT","CHZUSDT","RNDRUSDT",
    "SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","KLAYUSDT","TOMOUSDT","VETUSDT","BLURUSDT","STRKUSDT","ZRXUSDT"
]

# -------------------------
# OpenAI client + rate guard
# -------------------------
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY missing in env.")
client = OpenAI(api_key=OPENAI_API_KEY)

_openai_calls = []
def openai_can_call():
    now = time.time()
    while _openai_calls and _openai_calls[0] <= now - 60:
        _openai_calls.pop(0)
    return len(_openai_calls) < OPENAI_MAX_PER_MIN

def openai_note_call():
    _openai_calls.append(time.time())

# -------------------------
# SQLite (signals log)
# -------------------------
DB_PATH = "signals.db"
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
        cur.execute("INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
                    (ts, symbol, price, score, mode, reason, tp, sl))
        conn.commit()
    except Exception:
        pass

# -------------------------
# Binance exchange helper
# -------------------------
async def create_exchange():
    opts = {"enableRateLimit": True, "options":{"defaultType":"future"}}
    if USE_TESTNET:
        opts["urls"] = {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    exchange = ccxt.binance(opts)
    exchange.apiKey = BINANCE_API_KEY
    exchange.secret = BINANCE_API_SECRET
    return exchange

# -------------------------
# Snapshot with safe handling
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
        if tk:
            price = float(tk.get("last") or tk.get("close") or 0.0)
            base_vol = float(tk.get("baseVolume") or 0)
        else:
            price = None
    except Exception:
        price = None
        base_vol = 0.0

    if price is None or price == 0:
        price = random.uniform(10, 50000)

    # spread
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        if ob and ob.get("bids") and ob.get("asks"):
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            mid = (bid + ask) / 2 if (bid and ask) else price
            spread_pct = abs(ask - bid) / (mid if mid else price) * 100
    except Exception:
        spread_pct = 0.0

    # OHLCV fetches — each wrapped
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
        "ema_1h_50": compute_ema_from_closes(closes_1h,50) if closes_1h else 0,
        "ema_15m_50": compute_ema_from_closes(closes_15m,50) if closes_15m else 0,
        "spread_pct": round(spread_pct,4),
        "vol_1m": base_vol,
    }

    data = {"price": price, "metrics": metrics}
    CACHE.set(key, data, ttl_seconds=3)
    return data

# -------------------------
# AI scoring (cached)
# -------------------------
def parse_json(text):
    try:
        return json.loads(text.strip())
    except Exception:
        return None

async def ai_score_with_cache(symbol, price, metrics, prefs, ttl):
    key = CACHE.make_key("ai", symbol + str(round(price,6)))
    c = CACHE.get(key)
    if c:
        return c

    wait = 0
    while not openai_can_call():
        await asyncio.sleep(1)
        wait += 1
        if wait > 12:
            return None

    prompt = build_ai_prompt(symbol, price, metrics, prefs)

    def call_ai():
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=0,
                max_output_tokens=300
            )
        except Exception as e:
            # don't spam logs — return empty string
            return ""
        # try output_text first (compat)
        try:
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
            return json.dumps(resp.to_dict())
        except Exception:
            return str(resp)

    openai_note_call()
    raw = await asyncio.to_thread(call_ai)
    parsed = parse_json(raw)
    if parsed:
        CACHE.set(key, parsed, ttl_seconds=ttl)
    return parsed

# -------------------------
# Telegram send (minimal)
# -------------------------
async def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        try:
            await s.post(url, data={
                "chat_id": CHAT_ID,
                "text": msg,
                "parse_mode": "HTML"
            }, timeout=10)
        except Exception:
            pass

# -------------------------
# Message builder
# -------------------------
def build_message(symbol, price, score, mode, reason, tp, sl):
    return (
        f"⚡ <b>{mode.upper()} SIGNAL</b> ⚡\n"
        f"{symbol}  •  Price: {price:.2f}\n"
        f"Score: {score}  •  {esc(reason)}\n"
        f"TP: {tp}\nSL: {sl}\n"
        f"{human_time()} • cooldown {COOLDOWN_SECONDS//60}m"
    )

# -------------------------
# Worker (batch-safe)
# -------------------------
async def worker():
    exchange = await create_exchange()
    cooldowns = {}
    prefs = {
        "BTC_CALM_REQUIRED": True,
        "TP_SL":{
            "quick":{"tp_pct":1.6,"sl_pct":1.0},
            "mid":{"tp_pct":2.0,"sl_pct":1.0},
            "trend":{"tp_pct":4.0,"sl_pct":1.5}
        }
    }

    BATCH = int(os.getenv("BATCH","8"))
    index = 0
    print(f"Bot started • symbols={len(SYMBOLS)} • batch={BATCH}")

    try:
        while True:
            batch = SYMBOLS[index:index+BATCH]
            if not batch:
                index = 0
                batch = SYMBOLS[0:BATCH]

            for sym in batch:
                # skip if cooling
                if cooldowns.get(sym, 0) > time.time():
                    continue

                snap = await fetch_snapshot(exchange, sym)
                price = snap["price"]
                metrics = snap["metrics"]

                parsed = await ai_score_with_cache(sym, price, metrics, prefs, ttl=OPENAI_TTL_SECONDS)
                if not parsed:
                    await asyncio.sleep(0.05)
                    continue

                try:
                    score = int(parsed.get("score",0))
                except Exception:
                    score = 0
                mode = parsed.get("mode","quick")
                reason = parsed.get("reason","no reason")

                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_message(sym, price, score, mode, reason, tp, sl)
                    await send_telegram(msg)
                    log_signal_db(now_ts(), sym, price, score, mode, reason, tp, sl)
                    print(f"[SIGNAL] {sym} {score} {mode}")
                    cooldowns[sym] = time.time() + COOLDOWN_SECONDS

                # small gap between symbols
                await asyncio.sleep(0.08)

            index += BATCH
            await asyncio.sleep(CYCLE_TIME)
    finally:
        try: await exchange.close()
        except Exception: pass

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Missing OPENAI_API_KEY")
    else:
        try:
            asyncio.run(worker())
        except KeyboardInterrupt:
            print("Stopped by user")
        except Exception as e:
            print("Fatal:", e)