# main.py — FINAL (Stopping-proof, IST 00-07 OFF, 45 coins, 3/60s, 70 score)
import os, time, json, asyncio, random, hashlib, sqlite3, logging
from datetime import datetime
from dotenv import load_dotenv
from aiohttp import web
import aiohttp
import pytz
import ccxt.async_support as ccxt

load_dotenv()

from helpers import (
    now_ts, human_time, esc, calc_tp_sl, ensemble_score, CACHE,
    compute_ema_from_closes, atr, rsi_from_closes
)

# ------------------------- ENV -------------------------
BOT_TOKEN        = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID          = os.getenv("CHAT_ID", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CYCLE_TIME       = int(os.getenv("CYCLE_TIME", "60"))
SCORE_THRESHOLD  = int(os.getenv("SCORE_THRESHOLD", "70"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "1800"))
USE_TESTNET      = os.getenv("USE_TESTNET", "true").lower() in ("1", "true", "yes")
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "3"))

SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","ATOMUSDT","ETCUSDT","OPUSDT","INJUSDT","SUIUSDT","AAVEUSDT","CRVUSDT","RUNEUSDT",
    "XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT","HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT",
    "GALAUSDT","CHZUSDT","RNDRUSDT","SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","TOMOUSDT","VETUSDT","BLURUSDT",
    "STRKUSDT","ZRXUSDT","APTUSDT","NEARUSDT","ICPUSDT"
]

DB_PATH = os.getenv("SIGNAL_DB", "signals.db")

# ------------------------- SQLite -------------------------
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

# ------------------------- Exchange -------------------------
async def create_exchange():
    opts = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    if USE_TESTNET:
        opts["urls"] = {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    ex = ccxt.binance(opts)
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        ex.apiKey = BINANCE_API_KEY
        ex.secret = BINANCE_API_SECRET
    return ex

# ------------------------- Snapshot -------------------------
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
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
            mid = (bid + ask) / 2.0 if (bid + ask) else 1.0
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

# ------------------------- Telegram -------------------------
async def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": msg}
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.post(url, data=data, timeout=10) as r:
                await r.text()
        except Exception:
            pass

def build_message_plain(sym, price, score, mode, reason, tp, sl):
    return (
        f"{mode.upper()} SIGNAL\n"
        f"{sym}  Price: {price:.8f}\n"
        f"Score: {score}  Reason: {reason}\n"
        f"TP: {tp}\nSL: {sl}\n"
        f"{human_time()}  Cooldown: {COOLDOWN_SECONDS//60}m"
    )

# ------------------------- Health (SYNC + 0.0.0.0) -------------------------
def health_handler(request):
    return web.Response(text="ok")

async def start_health_app():
    app = web.Application()
    app.router.add_get("/", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "7500"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Health server on port {port}")

# ------------------------- Worker (minimal logs) -------------------------
async def worker():
    exchange = await create_exchange()
    cd = {}
    prefs = {"BTC_CALM_REQUIRED": True}
    idx = 0
    IST = pytz.timezone("Asia/Kolkata")
    logging.info("Bot started • IST 00-07 OFF • 45 coins • 3/60s • 70 score")

    try:
        while True:
            now_ist = datetime.now(IST)
            if 0 <= now_ist.hour < 7:          # IST night off
                logging.info("Night mode (IST 00-07) – sleeping 30 min")
                await asyncio.sleep(1800)
                continue

            batch = SYMBOLS[idx:idx+BATCH_SIZE] if idx+BATCH_SIZE <= len(SYMBOLS) else SYMBOLS[idx:] + SYMBOLS[:(idx+BATCH_SIZE)-len(SYMBOLS)]
            for sym in batch:
                if cd.get(sym, 0) > time.time():
                    continue
                snap = await fetch_snapshot(exchange, sym)
                price = snap["price"]
                metrics = snap["metrics"]
                parsed = await ensemble_score(sym, price, metrics, prefs, n=3)
                if not parsed:
                    continue
                score = int(parsed.get("score", 0))
                mode = parsed.get("mode", "quick")
                reason = parsed.get("reason", "")
                if score < SCORE_THRESHOLD:
                    continue
                tp, sl = calc_tp_sl(price, mode)
                msg = build_message_plain(sym, price, score, mode, reason, tp, sl)
                await send_telegram(msg)
                log_signal(now_ts(), sym, price, score, mode, reason, tp, sl)
                cd[sym] = time.time() + COOLDOWN_SECONDS
                await asyncio.sleep(0.08)
            idx = (idx + BATCH_SIZE) % len(SYMBOLS)
            await asyncio.sleep(CYCLE_TIME)
    finally:
        try: await exchange.close()
        except: pass

# ------------------------- Main (health FIRST) -------------------------
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    await start_health_app()   # ➜ আগে চালু
    await worker()             # ➜ পরে চালু

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception("Fatal error in main")
