# main.py — Redis OFF, filesystem fallback only
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
from fastapi import FastAPI
import uvicorn
import aiofiles

from helpers import PAIRS
from scorer import compute_signal
from telegram_formatter import TelegramFormatter
from position_tracker import PositionTracker
from db import init_db, save_signal  # optional db: keep if you use DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- ENV ----------
TG_TOKEN    = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT     = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INT    = int(os.getenv("SCAN_INTERVAL", 20))
COOLDOWN_M  = int(os.getenv("COOLDOWN_MIN", 30))
BATCH       = int(os.getenv("WRITE_BATCH_SIZE", 50))
SYNC_INT    = int(os.getenv("PERIODIC_SYNC_INTERVAL", 15))

# ---------- STATE ----------
cooldown = {}
ACTIVE_ORDERS = {}
TRADE_BUF = {s: deque(maxlen=500) for s in PAIRS}
OB_CACHE = {}
TK_CACHE = {}
cntr = {s: 0 for s in PAIRS}
data_rx = 0

# ---------- FILE PATHS ----------
TRADE_DIR = os.getenv("TRADE_FALLBACK_DIR", "/tmp/trades")
OB_DIR = os.getenv("OB_FALLBACK_DIR", "/tmp/ob")
TK_DIR = os.getenv("TK_FALLBACK_DIR", "/tmp/tk")

os.makedirs(TRADE_DIR, exist_ok=True)
os.makedirs(OB_DIR, exist_ok=True)
os.makedirs(TK_DIR, exist_ok=True)

# ---------- HELPERS ----------
formatter = TelegramFormatter()
pos_track = PositionTracker(storage_file=os.getenv("POSITION_FILE", "positions.json"))
db = None  # set in lifespan if DB used

async def tg_send(text: str):
    if not (TG_TOKEN and TG_CHAT):
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=10)
    except Exception as e:
        log.error(f"TG send error: {e}")

# ---------- FILE PERSIST (trade batch) ----------
async def persist_trades_to_file(sym: str, items: list):
    """Append JSONL trades to /tmp/trades/{sym}.jsonl"""
    if not items:
        return
    fn = os.path.join(TRADE_DIR, f"{sym}.jsonl")
    try:
        async with aiofiles.open(fn, "a") as f:
            for t in items:
                await f.write(json.dumps(t) + "\n")
    except Exception as e:
        log.error(f"persist_trades error for {sym}: {e}")

async def persist_ob_to_file(mapping: dict):
    """Write each OB entry as small json file (overwrites)"""
    try:
        for sym, data in mapping.items():
            fn = os.path.join(OB_DIR, f"{sym}.json")
            async with aiofiles.open(fn, "w") as f:
                await f.write(json.dumps(data))
    except Exception as e:
        log.error(f"persist_ob error: {e}")

async def persist_tk_to_file(mapping: dict):
    """Write each ticker entry as small json file (overwrites)"""
    try:
        for sym, data in mapping.items():
            fn = os.path.join(TK_DIR, f"{sym}.json")
            async with aiofiles.open(fn, "w") as f:
                await f.write(json.dumps(data))
    except Exception as e:
        log.error(f"persist_tk error: {e}")

# ---------- DATA PUSH ----------
async def push_trade(sym: str, data: dict):
    """Keep in-memory, write to disk every BATCH trades"""
    global data_rx
    try:
        if not all(k in data for k in ("p","q","m","t")):
            return

        trade = {"p": float(data["p"]), "q": float(data["q"]), "m": bool(data["m"]), "t": int(data["t"])}
        TRADE_BUF[sym].append(trade)
        cntr[sym] += 1
        data_rx += 1

        # Write batch to file
        if cntr[sym] >= BATCH:
            batch = list(TRADE_BUF[sym])[-BATCH:]
            await persist_trades_to_file(sym, batch)
            cntr[sym] = 0

    except Exception as e:
        log.error(f"push_trade error for {sym}: {e}")

async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        OB_CACHE[sym] = {"bid": float(bid), "ask": float(ask), "t": int(time.time()*1000)}
    except Exception as e:
        log.error(f"push_orderbook error: {e}")

async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        TK_CACHE[sym] = {"last": float(last), "vol": float(vol), "t": int(ts)}
    except Exception as e:
        log.error(f"push_ticker error: {e}")

# ---------- PERIODIC SYNC (write OB/TK to files) ----------
async def periodic_sync():
    """Every SYNC_INT seconds write OB/TK caches to files"""
    while True:
        try:
            await asyncio.sleep(SYNC_INT)
            if OB_CACHE:
                # flush snapshot and keep memory
                await persist_ob_to_file(OB_CACHE)
            if TK_CACHE:
                await persist_tk_to_file(TK_CACHE)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"periodic_sync error: {e}")

# ---------- WEBSOCKET WORKERS ----------
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 15))
ws_connected = False

def build_stream_url(pairs_chunk):
    streams = []
    for p in pairs_chunk:
        p_low = p.lower()
        streams.append(f"{p_low}@aggTrade")
        streams.append(f"{p_low}@bookTicker")
    if "BTCUSDT" in pairs_chunk:
        streams.append("btcusdt@kline_1m")
    return f"{BINANCE_WS_BASE}?streams={'/'.join(streams)}"

async def ws_worker(pairs_chunk, worker_id):
    global ws_connected
    url = build_stream_url(pairs_chunk)
    backoff = 1
    log.info(f"WS Worker {worker_id} starting for {len(pairs_chunk)} pairs")
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=10) as ws:
                backoff = 1
                ws_connected = True
                log.info(f"WS Worker {worker_id} connected")
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        # aggTrade
                        if stream.endswith("@aggTrade"):
                            sym = payload.get("s")
                            if sym and sym in PAIRS:
                                await push_trade(sym, {
                                    "p": payload.get("p"),
                                    "q": payload.get("q"),
                                    "m": payload.get("m"),
                                    "t": payload.get("T")
                                })

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

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        log.error(f"WS parse error: {e}")
                        continue

        except asyncio.CancelledError:
            break
        except Exception as e:
            ws_connected = False
            log.warning(f"WS Worker {worker_id} error: {e}. Reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

async def start_all_ws():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    tasks = [asyncio.create_task(ws_worker(chunk, idx + 1)) for idx, chunk in enumerate(chunks)]
    tasks.append(asyncio.create_task(periodic_sync()))
    log.info(f"Started {len(tasks)-1} WS workers + sync task")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()

# ---------- COOLDOWN ----------
def cooldown_ok(sym, strat):
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    diff = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return diff >= COOLDOWN_M

def set_cooldown(sym, strat):
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()

# ---------- SIGNAL FORMATTER (fallback) ----------
def quick_format(sig):
    return f"🎯 {sig['symbol']} {sig['side'].upper()} | {sig['last']:.6f} | Score {sig['score']:.1f}"

# ---------- SCANNER ----------
async def scanner():
    log.info("Scanner waiting for data...")
    await tg_send("🚀 <b>Scanner Starting...</b>\n⏳ Waiting for data stream")
    for i in range(20):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUF.get("BTCUSDT", []))
        if btc_count >= 20:
            log.info(f"Data ready! BTC trades: {btc_count}")
            await tg_send(f"✅ <b>Data Stream Active!</b>\n📊 {btc_count} BTC trades loaded")
            break
    else:
        log.warning("Data stream slow")

    while True:
        try:
            results = []
            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue
                    # compute_signal expects helpers that use buffer dicts — pass TRADE_BUF and OB_CACHE
                    sig = await compute_signal(sym, strat, trade_buffer=TRADE_BUF, ob_cache=OB_CACHE)
                    if sig:
                        results.append(sig)

            results.sort(key=lambda x: x["score"], reverse=True)

            if results:
                best = results[:3]
                for sig in best:
                    validated = sig.get("validated", True)
                    reason = sig.get("validation_reason", "")
                    if not validated:
                        log.info(f"Signal filtered: {sig['symbol']} {sig.get('strategy','')} -> {reason}")
                        continue

                    try:
                        msg = formatter.format_signal_alert(sig, sig.get("levels"), sig.get("volume"))
                    except Exception:
                        msg = quick_format(sig)

                    await tg_send(msg)

                    # DB save (optional)
                    try:
                        if db:
                            await save_signal(db, sig)
                    except Exception as e:
                        log.warning(f"DB save failed: {e}")

                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])
                    log.info(f"✔️ SIGNAL: {sig['symbol']} {sig['strategy']} = {sig['score']:.1f}")

            await asyncio.sleep(SCAN_INT)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            await asyncio.sleep(5)

# ---------- FASTAPI LIFECYCLE ----------
scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task, db
    log.info("🚀 Bot Starting (Redis OFF)")
    log.info(f"✓ {len(PAIRS)} pairs loaded")

    # start ws and sync
    ws_task = asyncio.create_task(start_all_ws())
    await asyncio.sleep(3)

    # position tracker
    try:
        await pos_track.load()
        log.info("✅ Position tracker loaded")
    except Exception as e:
        log.warning(f"Position tracker load failed: {e}")

    # init db if needed
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

    try:
        await pos_track.save()
        log.info("✅ Position tracker saved")
    except Exception as e:
        log.warning(f"Position tracker save failed: {e}")

    try:
        if db:
            await db.close()
            log.info("✅ DB closed")
    except Exception as e:
        log.warning(f"DB close failed: {e}")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    btc_trades = len(TRADE_BUF.get("BTCUSDT", []))
    return {
        "status": "running",
        "ws_connected": ws_connected,
        "data_received": data_rx,
        "btc_trades_memory": btc_trades,
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "time": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    btc_trades = len(TRADE_BUF.get("BTCUSDT", []))
    return {
        "status": "ok" if btc_trades > 0 else "starting",
        "ws_connected": ws_connected,
        "btc_trades": btc_trades,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))