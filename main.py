# main.py – v2.2-clean (final, full file)
import os
import json
import asyncio
import logging
import time
import traceback
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

import aiohttp
import websockets
import uvicorn
from fastapi import FastAPI
import aiofiles

from helpers import redis, PAIRS
from scorer import compute_signal
from telegram_formatter import TelegramFormatter
from position_tracker import PositionTracker
from db import init_db, save_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- ENV ----------
TG_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INT   = int(os.getenv("SCAN_INTERVAL", 20))
COOLDOWN_M = int(os.getenv("COOLDOWN_MIN", 30))
BATCH      = int(os.getenv("WRITE_BATCH_SIZE", 50))
REDIS_SKIP = int(os.getenv("REDIS_ERROR_SKIP_SEC", 60))
SYNC_INT   = int(os.getenv("PERIODIC_SYNC_INTERVAL", 15))
ENABLE_SYNC= os.getenv("ENABLE_REDIS_SYNC", "true").lower() == "true"
CHUNK      = int(os.getenv("CHUNK_SIZE", 15))
WS_URL     = os.getenv("BINANCE_WS_BASE", "wss://stream.binance.com:9443/stream")

# ---------- STATE ----------
cooldown = {}
ACTIVE_ORDERS = {}

# in-memory buffers
TRADE_BUF = {s: deque(maxlen=500) for s in PAIRS}
# compatibility alias (old modules may import TRADE_BUFFER)
TRADE_BUFFER = TRADE_BUF

OB_CACHE = {}
TICKER_CACHE = {}
TK_CACHE = TICKER_CACHE  # alias

cntr = {s: 0 for s in PAIRS}
data_rx = 0
_last_err = 0

# helpers
formatter = TelegramFormatter()
pos_track = PositionTracker(storage_file=os.getenv("POSITION_FILE", "positions.json"))
db = None

# ---------- TELEGRAM ----------
async def tg_send(text: str):
    if not (TG_TOKEN and TG_CHAT):
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json=payload, timeout=10)
    except Exception as e:
        log.error(f"TG send error: {e}")

# ---------- FALLBACK (local) ----------
async def _persist(sym: str, items):
    fn_dir = "/tmp/trades"
    os.makedirs(fn_dir, exist_ok=True)
    fn = os.path.join(fn_dir, f"{sym}.jsonl")
    try:
        async with aiofiles.open(fn, "a") as f:
            for i in items:
                await f.write(json.dumps(i) + "\n")
    except Exception as e:
        log.error(f"fallback persist error for {sym}: {e}")

# ---------- SAFE REDIS HELPERS ----------
async def safe_lpush(key: str, items: list):
    global _last_err
    if time.time() - _last_err < REDIS_SKIP:
        return
    try:
        pipe = redis.pipeline()
        for i in items:
            pipe.lpush(key, json.dumps(i))
        await pipe.execute()
    except Exception as e:
        err = str(e).lower()
        log.error(f"Redis push error ({key}): {e}")
        if any(x in err for x in ["max requests", "rate limit", "too many requests"]):
            _last_err = time.time()
            log.warning("Redis rate-limit detected — entering backoff.")
        await _persist(key.split(":")[-1], items)

async def safe_setex(mapping: dict, ttl: int = 60):
    global _last_err
    if time.time() - _last_err < REDIS_SKIP:
        return
    try:
        pipe = redis.pipeline()
        for k, v in mapping.items():
            pipe.setex(k, ttl, json.dumps(v))
        await pipe.execute()
    except Exception as e:
        err = str(e).lower()
        log.error(f"Redis setex error: {e}")
        if any(x in err for x in ["max requests", "rate limit", "too many requests"]):
            _last_err = time.time()
            log.warning("Redis rate-limit detected — entering backoff.")

# ---------- DATA PUSH ----------
async def push_trade(sym: str, data: dict):
    global data_rx
    try:
        if not all(k in data for k in ("p", "q", "m", "t")):
            return
        trade = {"p": float(data["p"]), "q": float(data["q"]), "m": bool(data["m"]), "t": int(data["t"])}
        TRADE_BUF[sym].append(trade)
        cntr[sym] += 1
        data_rx += 1
        if cntr[sym] >= BATCH and ENABLE_SYNC:
            batch = list(TRADE_BUF[sym])[-BATCH:]
            await safe_lpush(f"tr:{sym}", batch)
            cntr[sym] = 0
    except Exception as e:
        log.error(f"push_trade error for {sym}: {e}")

async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        OB_CACHE[sym] = {"bid": float(bid), "ask": float(ask), "t": int(datetime.utcnow().timestamp() * 1000)}
    except Exception as e:
        log.error(f"push_orderbook error: {e}")

async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        TICKER_CACHE[sym] = {"last": float(last), "vol": float(vol), "t": int(ts)}
    except Exception as e:
        log.error(f"push_ticker error: {e}")

# ---------- PERIODIC SYNC ----------
async def periodic_sync():
    while True:
        try:
            await asyncio.sleep(SYNC_INT)
            if not ENABLE_SYNC:
                continue
            if OB_CACHE:
                await safe_setex({f"ob:{k}": v for k, v in OB_CACHE.items()})
            if TICKER_CACHE:
                await safe_setex({f"tk:{k}": v for k, v in TICKER_CACHE.items()})
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"periodic_sync error: {e}")

# ---------- WEBSOCKET WORKER ----------
def build_url(chunk):
    streams = []
    for p in chunk:
        p_low = p.lower()
        streams.append(f"{p_low}@aggTrade")
        streams.append(f"{p_low}@bookTicker")
    if "BTCUSDT" in chunk:
        streams.append("btcusdt@kline_1m")
    return f"{WS_URL}?streams={'/'.join(streams)}"

ws_ok = False

async def ws_worker(pairs_chunk, worker_id):
    global ws_ok
    url = build_url(pairs_chunk)
    backoff = 1
    log.info(f"WS-{worker_id} start ({len(pairs_chunk)} pairs)")
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=10) as ws:
                backoff = 1
                ws_ok = True
                log.info(f"WS-{worker_id} connected")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        # aggTrade
                        if stream.endswith("@aggTrade"):
                            sym = payload.get("s")
                            if sym and sym in PAIRS:
                                await push_trade(sym, {"p": payload.get("p"), "q": payload.get("q"), "m": payload.get("m"), "t": payload.get("T")})
                        # bookTicker
                        elif stream.endswith("@bookTicker"):
                            sym = payload.get("s")
                            if sym and sym in PAIRS:
                                bid = float(payload.get("b", 0))
                                ask = float(payload.get("a", 0))
                                if bid > 0 and ask > 0:
                                    await push_orderbook(sym, bid, ask)
                        # kline
                        elif "@kline" in stream:
                            k = payload.get("k", {})
                            if k.get("x"):
                                sym = payload.get("s", "BTCUSDT")
                                close = float(k.get("c", 0))
                                vol = float(k.get("q", 0))
                                ts = int(k.get("T", 0))
                                if close > 0:
                                    await push_ticker(sym, close, vol, ts)
                    except Exception:
                        continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            ws_ok = False
            log.warning(f"WS-{worker_id} error: {e}. reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

async def start_ws():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK] for i in range(0, len(pairs), CHUNK)]
    tasks = [asyncio.create_task(ws_worker(chunk, idx + 1)) for idx, chunk in enumerate(chunks)]
    tasks.append(asyncio.create_task(periodic_sync()))
    log.info(f"Started {len(tasks)-1} WS workers + sync task")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()

# ---------- COOLDOWN ----------
def cooldown_ok(sym: str, strat: str) -> bool:
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    diff = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return diff >= COOLDOWN_M

def set_cooldown(sym: str, strat: str):
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()

# ---------- SCANNER ----------
async def scanner():
    global db
    log.info("Scanner waiting for data...")
    await tg_send("🚀 <b>Scanner Starting...</b>\n⏳ Waiting for data stream")

    # Wait for in-memory buffer ready
    for i in range(20):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUF.get("BTCUSDT", []))
        if btc_count >= 20:
            log.info(f"✅ Data ready! BTC trades: {btc_count}")
            await tg_send(f"✅ <b>Data Stream Active!</b>\n📊 {btc_count} BTC trades loaded")
            break
    else:
        log.warning("⚠️ Data stream slow")

    while True:
        try:
            results = []
            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue

                    # <-- UPDATED: pass buffers to compute_signal
                    sig = await compute_signal(sym, strat, TRADE_BUF, OB_CACHE)
                    if sig:
                        results.append(sig)

            # sort and pick top results
            results.sort(key=lambda x: x["score"], reverse=True)

            if results:
                best = results[:3]
                for sig in best:
                    validated = sig.get("validated", True)
                    reason = sig.get("validation_reason", "")
                    if not validated:
                        log.info(f"✖ Signal filtered: {sig['symbol']} {sig['strategy']} -> {reason}")
                        continue

                    try:
                        msg = formatter.format_signal_alert(sig, sig.get("levels"), sig.get("volume"))
                        await tg_send(msg)
                    except Exception as e:
                        log.error(f"Formatter error: {e}. Falling back to quick format.")
                        await tg_send(f"🎯 <b>{sig['symbol']} {sig['side'].upper()}</b> [{sig['strategy']}]\n💰 <code>{sig['last']:.6f}</code>\n📊 Score: <b>{sig['score']:.1f}</b>\n")

                    # Save to DB
                    try:
                        if db:
                            await save_signal(db, sig)
                    except Exception as e:
                        log.warning(f"DB save failed: {e}")

                    # mark and cooldown
                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])
                    log.info(f"✔️ SIGNAL: {sig['symbol']} {sig['strategy']} = {sig['score']:.1f}")

            await asyncio.sleep(SCAN_INT)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            await asyncio.sleep(5)

# ---------- FASTAPI / LIFECYCLE ----------
scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task, db
    log.info("🚀 v2.2-clean starting")
    # start ws + sync
    ws_task = asyncio.create_task(start_ws())
    await asyncio.sleep(3)

    # load positions
    try:
        await pos_track.load()
        log.info("✅ Position tracker loaded")
    except Exception as e:
        log.warning(f"Position tracker load failed: {e}")

    # init db
    try:
        db = await init_db()
        log.info("✅ DB connected")
    except Exception as e:
        db = None
        log.warning(f"DB connection failed: {e}")

    scan_task = asyncio.create_task(scanner())

    yield

    log.info("🛑 Shutting down...")
    if scan_task:
        scan_task.cancel()
    if ws_task:
        ws_task.cancel()

    try:
        await asyncio.gather(scan_task, ws_task, return_exceptions=True)
    except:
        pass

    # save positions
    try:
        await pos_track.save()
        log.info("✅ Position tracker saved")
    except Exception as e:
        log.warning(f"Position tracker save failed: {e}")

    # close db
    try:
        if db:
            await db.close()
            log.info("✅ DB closed")
    except Exception as e:
        log.warning(f"DB close failed: {e}")

    try:
        await redis.aclose()
    except:
        pass

    log.info("🔴 Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "running",
        "ws_connected": ws_ok,
        "data_received": data_rx,
        "btc_trades_memory": len(TRADE_BUF.get("BTCUSDT", [])),
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "time": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    btc_trades = len(TRADE_BUF.get("BTCUSDT", []))
    return {
        "status": "ok" if btc_trades > 0 else "starting",
        "ws_connected": ws_ok,
        "btc_trades": btc_trades,
        "data_received": data_rx,
        "redis_usage": "optimized"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))