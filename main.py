# ================================================================
# main.py — OPTIMIZED (Redis Usage 95% Reduced)  -- UPDATED (rate-limit safe)
# ================================================================

import os
import json
import asyncio
import logging
import traceback
import time
from datetime import datetime
from contextlib import asynccontextmanager
from collections import deque

import aiohttp
import websockets
from fastapi import FastAPI
import uvicorn

# async file IO (ensure aiofiles in requirements.txt)
import aiofiles

from helpers import redis, PAIRS
from scorer import compute_signal

# NEW imports (non-destructive)
from telegram_formatter import TelegramFormatter
from position_tracker import PositionTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# -----------------------------------------------------------
# ENV
# -----------------------------------------------------------
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 20))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))

cooldown = {}
ACTIVE_ORDERS = {}

# WebSocket config
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 15))
TRADES_TTL_SEC = int(os.getenv("TRADES_TTL_SEC", 1800))

ws_connected = False
data_received = 0

# ⭐ IN-MEMORY BUFFERS (Reduces Redis writes by 95%)
TRADE_BUFFER = {sym: deque(maxlen=500) for sym in PAIRS}
OB_CACHE = {}
TICKER_CACHE = {}
WRITE_BATCH_SIZE = int(os.getenv("WRITE_BATCH_SIZE", 50))  # Write to Redis every N trades
write_counters = {sym: 0 for sym in PAIRS}

# NEW: Formatter & PositionTracker instances
formatter = TelegramFormatter()
position_tracker = PositionTracker(storage_file=os.getenv("POSITION_FILE", "positions.json"))

# -----------------------------------------------------------
# Fallback & Rate-limit state (Upstash protection)
# -----------------------------------------------------------
LOCAL_FALLBACK_DIR = os.getenv("FALLBACK_DIR", "/tmp/trade_fallback")
os.makedirs(LOCAL_FALLBACK_DIR, exist_ok=True)

_redis_rate_limited_until = 0.0
_redis_backoff_seconds = int(os.getenv("REDIS_BACKOFF_START", 30))  # exponential backoff start


# -----------------------------------------------------------
# TELEGRAM SENDER
# -----------------------------------------------------------
async def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return

    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"}

    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=10)
    except Exception as e:
        log.error(f"Telegram error: {e}")


# -----------------------------------------------------------
# ⭐ OPTIMIZED DATA PUSHER (Batched Redis Writes) — rate-limit safe
# -----------------------------------------------------------
async def _persist_fallback(sym: str, items: list):
    """Persist a list of trades to local fallback file (jsonl)"""
    if not items:
        return
    filename = os.path.join(LOCAL_FALLBACK_DIR, f"{sym}_fallback.jsonl")
    try:
        async with aiofiles.open(filename, "a") as f:
            for t in items:
                await f.write(json.dumps(t) + "\n")
    except Exception as e:
        log.error(f"Fallback persist error for {sym}: {e}")


async def push_trade(sym: str, data: dict):
    """Store in memory first, batch write to Redis with rate-limit/backoff handling"""
    global _redis_rate_limited_until, _redis_backoff_seconds, data_received

    try:
        if not all(k in data for k in ["p", "q", "m", "t"]):
            return

        trade = {
            "p": float(data["p"]),
            "q": float(data["q"]),
            "m": bool(data["m"]),
            "t": int(data["t"])
        }

        # Store in memory
        TRADE_BUFFER[sym].append(trade)
        write_counters[sym] += 1

        # If currently rate-limited, persist this trade to fallback and skip Redis
        if time.time() < _redis_rate_limited_until:
            await _persist_fallback(sym, [trade])
            data_received += 1
            return

        # Only write to Redis every WRITE_BATCH_SIZE trades
        if write_counters[sym] >= WRITE_BATCH_SIZE:
            batch = list(TRADE_BUFFER[sym])[-WRITE_BATCH_SIZE:]
            batch_json = [json.dumps(t) for t in batch]

            try:
                await redis.lpush(f"tr:{sym}", *batch_json)
                await redis.ltrim(f"tr:{sym}", 0, 499)
                await redis.expire(f"tr:{sym}", TRADES_TTL_SEC)
                # success — reset backoff
                _redis_backoff_seconds = int(os.getenv("REDIS_BACKOFF_START", 30))
            except Exception as e:
                err_str = str(e).lower()
                log.error(f"Push trade error for {sym}: {e}")

                # Detect Upstash max-requests/rate-limit textual hints
                if "max requests limit" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                    # Set rate-limited window and backoff
                    _redis_rate_limited_until = time.time() + _redis_backoff_seconds
                    _redis_backoff_seconds = min(_redis_backoff_seconds * 2, 3600)  # cap 1 hour
                    # persist batch to fallback for later ingestion
                    await _persist_fallback(sym, batch)
                else:
                    # For other Redis errors, also persist batch to be safe
                    await _persist_fallback(sym, batch)
            finally:
                write_counters[sym] = 0

        data_received += 1

    except Exception as e:
        log.error(f"Push trade error for {sym}: {e}")


async def push_orderbook(sym: str, bid: float, ask: float):
    """Store in memory, write to Redis every periodic sync"""
    try:
        OB_CACHE[sym] = {
            "bid": float(bid),
            "ask": float(ask),
            "t": int(datetime.utcnow().timestamp() * 1000)
        }
    except Exception as e:
        log.error(f"OB cache error: {e}")


async def push_ticker(sym: str, last: float, vol: float, ts: int):
    """Store in memory only"""
    try:
        TICKER_CACHE[sym] = {
            "last": float(last),
            "vol": float(vol),
            "t": int(ts)
        }
    except Exception as e:
        log.error(f"Ticker cache error: {e}")


# -----------------------------------------------------------
# ⭐ PERIODIC REDIS SYNC (Every 5 seconds)
# -----------------------------------------------------------
async def periodic_redis_sync():
    """Write orderbook & ticker cache to Redis periodically"""
    while True:
        try:
            await asyncio.sleep(5)

            # Sync orderbooks (batch write)
            if OB_CACHE:
                try:
                    pipe = redis.pipeline()
                    for sym, data in OB_CACHE.items():
                        pipe.setex(f"ob:{sym}", 60, json.dumps(data))
                    await pipe.execute()
                except Exception as e:
                    log.error(f"OB sync redis error: {e}")

            # Sync tickers (batch write)
            if TICKER_CACHE:
                try:
                    pipe = redis.pipeline()
                    for sym, data in TICKER_CACHE.items():
                        pipe.setex(f"tk:{sym}", 60, json.dumps(data))
                    await pipe.execute()
                except Exception as e:
                    log.error(f"Ticker sync redis error: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Sync error: {e}")


# -----------------------------------------------------------
# WEBSOCKET WORKER
# -----------------------------------------------------------
def build_stream_url(pairs_chunk):
    streams = []
    for p in pairs_chunk:
        p_low = p.lower()
        streams.append(f"{p_low}@aggTrade")
        streams.append(f"{p_low}@bookTicker")

    if "BTCUSDT" in pairs_chunk:
        streams.append("btcusdt@kline_1m")

    stream_path = "/".join(streams)
    return f"{BINANCE_WS_BASE}?streams={stream_path}"


async def ws_worker(pairs_chunk, worker_id):
    global ws_connected
    url = build_stream_url(pairs_chunk)
    backoff = 1

    log.info(f"🔌 WS Worker {worker_id} starting for {len(pairs_chunk)} pairs")

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                backoff = 1
                ws_connected = True
                log.info(f"✅ WS Worker {worker_id} connected")

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
            log.warning(f"⚠️ WS Worker {worker_id} error: {e}. Reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def start_all_ws():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]

    tasks = [asyncio.create_task(ws_worker(chunk, idx + 1)) for idx, chunk in enumerate(chunks)]
    tasks.append(asyncio.create_task(periodic_redis_sync()))

    log.info(f"🔌 Started {len(tasks)-1} WS workers + sync task")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()


# -----------------------------------------------------------
# COOLDOWN
# -----------------------------------------------------------
def cooldown_ok(sym, strat):
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    diff = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return diff >= COOLDOWN_MIN


def set_cooldown(sym, strat):
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()


# -----------------------------------------------------------
# SIGNAL FORMATTER (legacy quick format kept)
# -----------------------------------------------------------
def format_signal(sig):
    sym = sig["symbol"]
    side = sig["side"]
    score = sig["score"]
    last = sig["last"]
    strat = sig["strategy"]
    passed = " | ".join(sig["passed"][:4])

    return (
        f"🎯 <b>{sym} {side.upper()}</b> [{strat}]\n"
        f"💰 <code>{last:.6f}</code>\n"
        f"📊 Score: <b>{score:.1f}</b>\n"
        f"🔍 {passed}\n"
    )


# -----------------------------------------------------------
# SCANNER
# -----------------------------------------------------------
async def scanner():
    log.info("🔍 Scanner waiting for data...")

    await send_telegram("🚀 <b>Scanner Starting...</b>\n⏳ Waiting for data stream")

    # Wait for in-memory buffer
    for i in range(20):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUFFER.get("BTCUSDT", []))
        if btc_count >= 20:
            log.info(f"✅ Data ready! BTC trades: {btc_count}")
            await send_telegram(f"✅ <b>Data Stream Active!</b>\n📊 {btc_count} BTC trades loaded")
            break
    else:
        log.warning("⚠️ Data stream slow")

    scan_count = 0

    while True:
        try:
            scan_count += 1
            results = []

            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue

                    sig = await compute_signal(sym, strat)
                    if sig:
                        results.append(sig)

            results.sort(key=lambda x: x["score"], reverse=True)

            if results:
                best = results[:3]
                for sig in best:
                    # Use enhanced validation flag if present (enhance_signal sets "validated")
                    validated = sig.get("validated", True)
                    reason = sig.get("validation_reason", "")

                    if not validated:
                        log.info(f"✖ Signal filtered: {sig['symbol']} {sig['strategy']} -> {reason}")
                        continue

                    # Prefer full-feature Telegram message using formatter (levels, volume included)
                    try:
                        msg = formatter.format_signal_alert(sig, sig.get("levels"), sig.get("volume"))
                        await send_telegram(msg)
                    except Exception as e:
                        log.error(f"Formatter error: {e}. Falling back to quick format.")
                        await send_telegram(format_signal(sig))

                    # Mark active order + set cooldown
                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])
                    log.info(f"✔️ SIGNAL: {sig['symbol']} {sig['strategy']} = {sig['score']:.1f}")

            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            await asyncio.sleep(5)


# -----------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------
scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task

    log.info("🚀 Bot Starting (v2.2 - Optimized)")
    log.info(f"✓ {len(PAIRS)} pairs loaded")

    # Start WS + sync
    ws_task = asyncio.create_task(start_all_ws())
    await asyncio.sleep(3)

    # Load positions (if any)
    try:
        await position_tracker.load()
        log.info("✅ Position tracker loaded")
    except Exception as e:
        log.warning(f"Position tracker load failed: {e}")

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

    # Save positions on shutdown
    try:
        await position_tracker.save()
        log.info("✅ Position tracker saved")
    except Exception as e:
        log.warning(f"Position tracker save failed: {e}")

    try:
        await redis.aclose()
    except:
        pass

    log.info("🔴 Shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))

    return {
        "status": "running",
        "ws_connected": ws_connected,
        "data_received": data_received,
        "btc_trades_memory": btc_trades,
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "time": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))

    return {
        "status": "ok" if btc_trades > 0 else "starting",
        "ws_connected": ws_connected,
        "btc_trades": btc_trades,
        "data_received": data_received,
        "redis_usage": "optimized"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))