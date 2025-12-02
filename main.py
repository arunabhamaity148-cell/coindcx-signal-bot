# main.py — FINAL (Single-file, minimal logs, IST 00:00-07:00 OFF, test endpoint)
import os, time, json, asyncio, random, hashlib, sqlite3, logging
from datetime import datetime
from dotenv import load_dotenv
from aiohttp import web
import aiohttp
import pytz

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
BOT_TOKEN        = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID          = os.getenv("CHAT_ID", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CYCLE_TIME       = int(os.getenv("CYCLE_TIME", "30"))    # seconds between batches
SCORE_THRESHOLD  = int(os.getenv("SCORE_THRESHOLD", "70"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))
USE_TESTNET      = os.getenv("USE_TESTNET", "true").lower() in ("1", "true", "yes")
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

# 45 high-volume coins (editable)
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","ATOMUSDT","ETCUSDT","OPUSDT","INJUSDT","SUIUSDT","AAVEUSDT","CRVUSDT","RUNEUSDT",
    "XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT","HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT",
    "GALAUSDT","CHZUSDT","RNDRUSDT","SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","TOMOUSDT","VETUSDT","BLURUSDT",
    "STRKUSDT","ZRXUSDT","APTUSDT","NEARUSDT","ICPUSDT"
]

# -------------------------
# OpenAI client & simple rate tracker (TTL / call limiter optional)
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

_openai_calls = []
OPENAI_MAX_PER_MIN = int(os.getenv("OPENAI_MAX_PER_MIN", "40"))

def openai_can_call():
    now = time.time()
    window = 60.0
    while _openai_calls and _openai_calls[0] <= now - window:
        _openai_calls.pop(0)
    return len(_openai_calls) < OPENAI_MAX_PER_MIN

def openai_note_call():
    _openai_calls.append(time.time())

# -------------------------
# SQLite (signals log)
# -------------------------
DB_PATH = os.getenv("SIGNAL_DB", "signals.db")
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

def log_signal(ts, sym, price, score, mode, reason, tp, sl):
    try:
        cur.execute("INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
                    (ts, sym, price, score, mode, reason, tp, sl))
        conn.commit()
    except Exception:
        pass

# -------------------------
# Exchange create (ccxt async)
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
# Snapshot (cached brief)
# -------------------------
async def fetch_snapshot(exchange, symbol):
    key = CACHE.make_key("snap", symbol)
    cached = CACHE.get(key)
    if cached:
        return cached

    price = 0.0
    base_vol = 0.0
    try:
        tk = await exchange.fetch_ticker(symbol)
        price = float(tk.get("last") or tk.get("close") or 0)
        base_vol = float(tk.get("baseVolume") or 0)
    except Exception:
        price = random.uniform(1, 50000)

    spread_pct = 0.0
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        if ob.get("bids") and ob.get("asks"):
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            mid = (bid + ask) / 2.0
            spread_pct = abs(ask - bid) / mid * 100.0
    except Exception:
        spread_pct = 0.0

    closes_1m = closes_15m = closes_1h = []
    try:
        o1 = await exchange.fetch_ohlcv(symbol, "1m", limit=60)
        closes_1m = [r[4] for r in o1]
    except Exception:
        closes_1m = []
    try:
        o15 = await exchange.fetch_ohlcv(symbol, "15m", limit=60)
        closes_15m = [r[4] for r in o15]
    except Exception:
        closes_15m = []
    try:
        o1h = await exchange.fetch_ohlcv(symbol, "1h", limit=60)
        closes_1h = [r[4] for r in o1h]
    except Exception:
        closes_1h = []

    metrics = {
        "closes_1m": closes_1m,
        "closes_15m": closes_15m,
        "closes_1h": closes_1h,
        "rsi_1m": rsi_from_closes(closes_1m) if closes_1m else 50,
        "ema_1h_50": compute_ema_from_closes(closes_1h, 50) if closes_1h else 0,
        "ema_15m_50": compute_ema_from_closes(closes_15m, 50) if closes_15m else 0,
        "spread_pct": round(spread_pct, 4),
        "vol_1m": base_vol,
        "price": price
    }

    out = {"price": price, "metrics": metrics}
    CACHE.set(key, out, ttl_seconds=3)
    return out

# -------------------------
# AI scoring (use responses API via thread to avoid blocking)
# -------------------------
def _call_openai_responses_sync(prompt: str):
    try:
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt, temperature=0, max_output_tokens=300)
    except Exception as e:
        return {"_error": str(e)}
    # Try to extract output_text (sdk compatibility)
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return {"text": resp.output_text}
        # fallback to dict
        return {"text": json.dumps(resp.to_dict())}
    except Exception:
        return {"_error": "no-output"}

async def ai_score_symbol_with_cache(symbol: str, price: float, metrics: dict, prefs: dict, ttl: int = 60):
    # small fingerprint cache key
    fingerprint = f"{symbol}|{round(price,6)}|{round(metrics.get('rsi_1m',50),2)}|{round(metrics.get('spread_pct',0),4)}"
    key = CACHE.make_key("ai", fingerprint)
    cached = CACHE.get(key)
    if cached:
        return cached

    # rate-limit guard
    wait = 0
    while not openai_can_call():
        await asyncio.sleep(1)
        wait += 1
        if wait > 12:
            return None

    prompt = build_ai_prompt(symbol, price, metrics, prefs)
    # if prompt is actual precomputed JSON (helpers may return immediate JSON string for filters), handle
    try:
        # if helpers returned JSON string (filter hit), parse and return directly
        maybe_json = json.loads(prompt)
        if isinstance(maybe_json, dict) and {"score","mode","reason"} <= set(maybe_json.keys()):
            CACHE.set(key, maybe_json, ttl_seconds=ttl)
            return maybe_json
    except Exception:
        pass

    openai_note_call()
    raw = await asyncio.to_thread(_call_openai_responses_sync, prompt)
    text = raw.get("text") or ""
    # extract JSON substring
    parsed = None
    try:
        parsed = json.loads(text.strip())
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    if parsed:
        CACHE.set(key, parsed, ttl_seconds=ttl)
    return parsed

# -------------------------
# Telegram send (plain text, no HTML if requested)
# -------------------------
async def send_telegram(msg: str, use_html: bool = False):
    if not BOT_TOKEN or not CHAT_ID:
        logging.info("Telegram not configured, skipping send.")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    if use_html:
        data["parse_mode"] = "HTML"
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.post(url, data=data, timeout=10) as r:
                return await r.json()
        except Exception:
            return None

# -------------------------
# Message builder (plain text)
# -------------------------
def build_message_plain(sym, price, score, mode, reason, tp, sl):
    return (
        f"{mode.upper()} SIGNAL\n"
        f"{sym}  Price: {price:.8f}\n"
        f"Score: {score}  Reason: {reason}\n"
        f"TP: {tp}\nSL: {sl}\n"
        f"{human_time()}  Cooldown: {COOLDOWN_SECONDS//60}m"
    )

# -------------------------
# Test endpoint (send a single test alert for BTC)
# GET /test
# -------------------------
async def handle_test(request):
    # Build a synthetic test alert using live snapshot if possible
    exchange = await create_exchange()
    try:
        snap = await fetch_snapshot(exchange, "BTCUSDT")
        price = snap["price"]
    except Exception:
        price = 0.0
    try:
        tp, sl = calc_tp_sl(price or 0.0, "quick")
        msg = build_message_plain("BTCUSDT", price, 99, "quick", "test-alert", tp, sl)
        await send_telegram(msg)
        return web.Response(text="Test alert sent")
    finally:
        try: await exchange.close()
        except: pass

# -------------------------
# Worker loop (batch safe, IST off 00:00-07:00)
# -------------------------
async def worker():
    exchange = await create_exchange()
    cd = {}
    prefs = {"BTC_CALM_REQUIRED": True}
    BATCH = int(os.getenv("BATCH_SIZE", "4"))
    idx = 0
    IST = pytz.timezone("Asia/Kolkata")
    logging.info(f"Bot started • symbols={len(SYMBOLS)} • cycle={CYCLE_TIME}s • threshold={SCORE_THRESHOLD}")

    try:
        while True:
            now_ist = datetime.now(IST)
            # IST night off between 00:00 and 07:00
            if 0 <= now_ist.hour < 7:
                logging.info("IST night window active — sleeping 30 minutes")
                await asyncio.sleep(1800)
                continue

            batch = SYMBOLS[idx:idx+BATCH] if idx+BATCH <= len(SYMBOLS) else SYMBOLS[idx:] + SYMBOLS[:(idx+BATCH)-len(SYMBOLS)]
            for sym in batch:
                if cd.get(sym, 0) > time.time():
                    continue
                snap = await fetch_snapshot(exchange, sym)
                price = snap["price"]
                metrics = snap["metrics"]
                parsed = await ai_score_symbol_with_cache(sym, price, metrics, prefs, ttl=int(os.getenv("OPENAI_TTL_SECONDS", "60")))
                if not parsed:
                    await asyncio.sleep(0.05)
                    continue
                score = int(parsed.get("score", 0))
                mode = parsed.get("mode", "quick")
                reason = parsed.get("reason", "")
                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_message_plain(sym, price, score, mode, reason, tp, sl)
                    await send_telegram(msg, use_html=False)
                    log_signal(now_ts(), sym, price, score, mode, reason, tp, sl)
                    cd[sym] = time.time() + COOLDOWN_SECONDS
                await asyncio.sleep(0.08)
            idx = (idx + BATCH) % len(SYMBOLS)
            await asyncio.sleep(CYCLE_TIME)
    finally:
        try: await exchange.close()
        except: pass

# -------------------------
# Health server + routes
# -------------------------
async def health_handler(_):
    return web.Response(text="ok")

async def start_health_app():
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/test", handle_test)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "7500"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Health & test server started on port {port}")

# -------------------------
# Main entry
# -------------------------
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY missing in environment")
        return
    await start_health_app()
    await worker()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Fatal error in main")