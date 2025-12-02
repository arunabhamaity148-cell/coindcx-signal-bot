# ============================
# main.py — FINAL (45 coins, 4/30s, 70 score)
# ============================

import os, time, json, asyncio, random, hashlib, sqlite3, logging
from datetime import datetime
from dotenv import load_dotenv
from aiohttp import web

load_dotenv()

from openai import OpenAI
import ccxt.async_support as ccxt

from helpers import (
    now_ts, human_time, esc, calc_tp_sl, build_ai_prompt, CACHE,
    compute_ema_from_closes, atr, rsi_from_closes
)

# -------------------------
, rsi_from_closes
)

# -------------------------
# ENV
# -------------------------
BOT_TOKEN        = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID          = os.getenv("CHAT_ID", "").strip()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL     = "gpt-4o-mini"
CYCLE_TIME       = 30          # 30 sec
SCORE_THRESHOLD  = 70          # relaxed
COOLDOWN_SECONDS = 1800
USE_TESTNET      = True
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "").strip()

# 45 high-volume coins
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "LTCUSDT","LINKUSDT","ATOMUSDT","ETCUSDT","OPUSDT","INJUSDT","SUIUSDT","AAVEUSDT","CRVUSDT","RUNEUSDT",
    "XMRUSDT","FTMUSDT","SNXUSDT","DYDXUSDT","GMTUSDT","HBARUSDT","THETAUSDT","AXSUSDT","FLOWUSDT","KAVAUSDT",
    "GALAUSDT","CHZUSDT","RNDRUSDT","SANDUSDT","MANAUSDT","1INCHUSDT","COMPUSDT","TOMOUSDT","VETUSDT","BLURUSDT",
    "STRKUSDT","ZRXUSDT","APTUSDT","NEARUSDT","ICPUSDT"
]

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# SQLite
# -------------------------
conn = sqlite3.connect("signals.db", check_same_thread=False)
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
    cur.execute("INSERT INTO signals (ts,symbol,price,score,mode,reason,tp,sl) VALUES (?,?,?,?,?,?,?,?)",
                (ts, sym, price, score, mode, reason, tp, sl))
    conn.commit()

# -------------------------
# Exchange
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
    ex = ccxt.binance(opts)
    ex.apiKey = BINANCE_API_KEY
    ex.secret = BINANCE_API_SECRET
    return ex

# -------------------------
# Snapshot
# -------------------------
async def fetch_snapshot(exchange, symbol):
    key = CACHE.make_key("snap", symbol)
    if (c := CACHE.get(key)) is not None:
        return c

    price, base_vol = 0, 0
    try:
        tk = await exchange.fetch_ticker(symbol)
        price  = float(tk.get("last") or tk.get("close") or 0)
        base_vol = float(tk.get("baseVolume") or 0)
    except:
        price = random.uniform(1, 1000)
    if price == 0:
        price = random.uniform(1, 500)

    spread_pct = 0.0
    try:
        ob = await exchange.fetch_order_book(symbol, 5)
        if ob["bids"] and ob["asks"]:
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
            spread_pct = abs(ask - bid) / ((bid + ask) / 2) * 100
    except:
        pass

    closes_1m = closes_15m = closes_1h = []
    try: closes_1m   = [r[4] for r in await exchange.fetch_ohlcv(symbol, "1m",   limit=100)]
    except: pass
    try: closes_15m  = [r[4] for r in await exchange.fetch_ohlcv(symbol, "15m",  limit=100)]
    except: pass
    try: closes_1h   = [r[4] for r in await exchange.fetch_ohlcv(symbol, "1h",   limit=100)]
    except: pass

    metrics = {
        "closes_1m": closes_1m,
        "closes_15m": closes_15m,
        "closes_1h": closes_1h,
        "rsi_1m": rsi_from_closes(closes_1m) if closes_1m else 50,
        "ema_1h_50": compute_ema_from_closes(closes_1h, 50),
        "ema_15m_50": compute_ema_from_closes(closes_15m, 50),
        "spread_pct": round(spread_pct, 3),
        "vol_1m": base_vol
    }
    data = {"price": price, "metrics": metrics}
    CACHE.set(key, data, 3)
    return data

# -------------------------
# AI Score
# -------------------------
async def ai_score(symbol, price, metrics, prefs):
    prompt = build_ai_prompt(symbol, price, metrics, prefs)
    try:
        r = client.responses.create(model=OPENAI_MODEL, input=prompt, temperature=0)
        return json.loads(r.output_text)
    except:
        return None

# -------------------------
# Telegram
# -------------------------
async def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})

# -------------------------
# Signal Message
# -------------------------
def build_msg(sym, price, score, mode, reason, tp, sl):
    return (
        f"🔥 <b>{mode.upper()} SIGNAL</b>\n"
        f"{sym} • Price {price}\n"
        f"Score {score} • {esc(reason)}\n"
        f"TP: {tp}\nSL: {sl}\n"
        f"{human_time()} • {COOLDOWN_SECONDS//60}m CD"
    )

# -------------------------
# Worker
# -------------------------
async def worker():
    ex = await create_exchange()
    cd = {}
    prefs = {"BTC_CALM_REQUIRED": True}
    BATCH = 4
    idx = 0
    logging.info("Bot Started • 45 coins • 4/30s • 70 score • sleep 00-07 UTC")

    try:
        while True:
            utc_hour = datetime.utcnow().hour
            if 0 <= utc_hour < 7:          # night mode
                logging.info("Night mode – sleeping 30 min")
                await asyncio.sleep(1800)
                continue

            batch = SYMBOLS[idx:idx+BATCH] if idx+BATCH <= len(SYMBOLS) else SYMBOLS[idx:] + SYMBOLS[:idx+BATCH-len(SYMBOLS)]
            for sym in batch:
                if cd.get(sym, 0) > time.time():
                    continue
                snap  = await fetch_snapshot(ex, sym)
                price = snap["price"]
                parsed = await ai_score(sym, price, snap["metrics"], prefs)
                if not parsed:
                    continue
                score  = int(parsed.get("score", 0))
                mode   = parsed.get("mode", "quick")
                reason = parsed.get("reason", "")
                if score >= SCORE_THRESHOLD:
                    tp, sl = calc_tp_sl(price, mode)
                    msg = build_msg(sym, price, score, mode, reason, tp, sl)
                    await send_telegram(msg)
                    log_signal(now_ts(), sym, price, score, mode, reason, tp, sl)
                    cd[sym] = time.time() + COOLDOWN_SECONDS
                await asyncio.sleep(0.2)
            idx = (idx + BATCH) % len(SYMBOLS)
            await asyncio.sleep(CYCLE_TIME)
    except Exception as e:
        logging.exception("WORKER CRASHED")
        raise
    finally:
        try: await ex.close()
        except: pass

# -------------------------------------------------
# Health Server
# -------------------------------------------------
async def health_handler(_):
    return web.Response(text="ok")

async def start_health_app():
    app = web.Application()
    app.router.add_get("/", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 7500)))
    await site.start()
    logging.info("Health server on port %s", os.getenv("PORT", 7500))

# -------------------------------------------------
# Main Entry
# -------------------------------------------------
async def main():
    logging.basicConfig(level=logging.INFO)
    await start_health_app()
    await worker()

if __name__ == "__main__":
    asyncio.run(main())
