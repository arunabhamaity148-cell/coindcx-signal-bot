# ================================================================
# main.py — FULL BOT ENGINE (WebSocket + Scanner Combined)
# ================================================================

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import aiohttp
import websockets
from fastapi import FastAPI
import uvicorn

from helpers import redis, PAIRS, get_last_price
from scorer import compute_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# -----------------------------------------------------------
# ENV
# -----------------------------------------------------------
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 10))
COOLDOWN_MIN = 30

cooldown = {}
ACTIVE_ORDERS = {}

# WebSocket config
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
CHUNK_SIZE = 18
TRADES_TTL_SEC = 1800
OB_TTL_SEC = 120
TK_TTL_SEC = 120


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
# WEBSOCKET DATA PUSHER
# -----------------------------------------------------------
async def push_trade(sym: str, price: float, qty: float, is_sell: bool, ts: int):
    try:
        await redis.lpush(f"tr:{sym}", json.dumps({"p": price, "q": qty, "m": is_sell, "t": ts}))
        await redis.ltrim(f"tr:{sym}", 0, 499)
        await redis.expire(f"tr:{sym}", TRADES_TTL_SEC)
    except Exception:
        pass


async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        await redis.setex(f"ob:{sym}", OB_TTL_SEC, json.dumps({"bid": bid, "ask": ask, "t": int(datetime.utcnow().timestamp() * 1000)}))
    except Exception:
        pass


async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        await redis.setex(f"tk:{sym}", TK_TTL_SEC, json.dumps({"last": last, "vol": vol, "t": ts}))
    except Exception:
        pass


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


async def ws_worker(pairs_chunk):
    url = build_stream_url(pairs_chunk)
    backoff = 1
    
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1
                log.info(f"✅ WS connected for {len(pairs_chunk)} pairs")
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        if stream.endswith("@aggTrade"):
                            sym = payload.get("s")
                            if not sym:
                                continue
                            price = float(payload.get("p", 0))
                            qty = float(payload.get("q", 0))
                            is_sell = bool(payload.get("m", False))
                            ts = int(payload.get("T", int(datetime.utcnow().timestamp() * 1000)))
                            await push_trade(sym, price, qty, is_sell, ts)

                        elif stream.endswith("@bookTicker"):
                            sym = payload.get("s")
                            if not sym:
                                continue
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            await push_orderbook(sym, bid, ask)

                        elif "@kline" in stream:
                            k = payload.get("k", {})
                            sym = payload.get("s", "BTCUSDT")
                            close = float(k.get("c", 0))
                            vol = float(k.get("q", 0))
                            ts = int(k.get("T", int(datetime.utcnow().timestamp() * 1000)))
                            await push_ticker(sym, close, vol, ts)

                    except Exception as e:
                        continue

        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Exception as e:
            log.error(f"WS error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def start_all_ws():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    tasks = [asyncio.create_task(ws_worker(chunk)) for chunk in chunks]
    log.info(f"🔌 Started {len(tasks)} WS workers for {len(pairs)} pairs")
    await asyncio.gather(*tasks)


# -----------------------------------------------------------
# COOLDOWN CHECK
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
# SIGNAL FORMATTER
# -----------------------------------------------------------
def format_signal(sig):
    sym = sig["symbol"]
    side = sig["side"]
    score = sig["score"]
    last = sig["last"]
    strat = sig["strategy"]
    passed = " | ".join(sig["passed"][:5])  # top 5 only

    return (
        f"🎯 <b>{sym} {side.upper()} — [{strat}]</b>\n"
        f"💰 Price: <code>{last:.6f}</code>\n"
        f"📊 Score: <b>{score:.1f}</b>\n"
        f"🔍 Passed: {passed}\n"
        f"⏳ Cooldown: {COOLDOWN_MIN}m\n"
    )


# -----------------------------------------------------------
# MAIN SCANNER LOOP
# -----------------------------------------------------------
async def scanner():
    log.info("🔍 Scanner started")
    
    try:
        await redis.ping()
        log.info("✅ Redis connected successfully")
    except Exception as e:
        log.error(f"❌ Redis connection failed: {e}")
        await send_telegram("⚠️ <b>Bot Started but Redis NOT connected!</b>")
        return

    await send_telegram("🚀 <b>Binance Scanner LIVE</b>\n📡 Waiting for data stream...")

    # Wait 15 seconds for WebSocket to populate data
    await asyncio.sleep(15)

    while True:
        try:
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
                    msg = format_signal(sig)
                    await send_telegram(msg)

                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])

                    log.info(f"✔ SIGNAL: {sig['symbol']} {sig['strategy']} score={sig['score']:.1f}")

            else:
                log.info("📊 No valid signals")

            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}", exc_info=True)
            await asyncio.sleep(5)


# -----------------------------------------------------------
# FASTAPI + LIFESPAN
# -----------------------------------------------------------
scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task
    
    log.info("🚀 App starting (Production Bot v2.0)")
    log.info(f"✓ Loaded {len(PAIRS)} pairs")
    
    # Start WebSocket first
    ws_task = asyncio.create_task(start_all_ws())
    
    # Then start scanner
    scan_task = asyncio.create_task(scanner())
    
    yield
    
    # Cleanup
    if scan_task:
        scan_task.cancel()
    if ws_task:
        ws_task.cancel()
    
    try:
        await asyncio.gather(scan_task, ws_task, return_exceptions=True)
    except:
        pass
    
    await redis.close()
    log.info("🛑 App shutdown complete")


app = FastAPI(lifespan=lifespan)


# -----------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------
@app.get("/")
async def root():
    # Check Redis data availability
    sample_check = await redis.llen("tr:BTCUSDT")
    
    return {
        "status": "running",
        "pairs": len(PAIRS),
        "cooldown_active": len(cooldown),
        "active_signals": len(ACTIVE_ORDERS),
        "redis_sample_data": sample_check,
        "time": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    try:
        await redis.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    
    # Check data availability
    btc_trades = await redis.llen("tr:BTCUSDT")
    
    return {
        "status": "ok",
        "redis": redis_status,
        "pairs": len(PAIRS),
        "btc_trades_count": btc_trades
    }


@app.get("/signals")
async def get_signals():
    return {
        "active": ACTIVE_ORDERS,
        "count": len(ACTIVE_ORDERS)
    }


# -----------------------------------------------------------
# RUN SERVER
# -----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))