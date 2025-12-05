# ================================================================
# main.py — FULL BOT ENGINE (WebSocket + Scanner + Debug)
# ================================================================

import os
import json
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import aiohttp
import websockets
from fastapi import FastAPI
import uvicorn

from helpers import redis, PAIRS
from scorer import compute_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# -----------------------------------------------------------
# ENV
# -----------------------------------------------------------
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
COOLDOWN_MIN = 30

cooldown = {}
ACTIVE_ORDERS = {}

# WebSocket config
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
CHUNK_SIZE = 15  # Reduced for stability
TRADES_TTL_SEC = 1800
OB_TTL_SEC = 120
TK_TTL_SEC = 120

ws_connected = False
data_received = 0


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
# WEBSOCKET DATA PUSHER (WITH VALIDATION)
# -----------------------------------------------------------
async def push_trade(sym: str, data: dict):
    try:
        # Validate required fields
        if not all(k in data for k in ["p", "q", "m", "t"]):
            log.warning(f"Invalid trade data for {sym}: {data}")
            return
            
        trade_json = json.dumps({
            "p": float(data["p"]),
            "q": float(data["q"]),
            "m": bool(data["m"]),
            "t": int(data["t"])
        })
        
        await redis.lpush(f"tr:{sym}", trade_json)
        await redis.ltrim(f"tr:{sym}", 0, 499)
        await redis.expire(f"tr:{sym}", TRADES_TTL_SEC)
        
        global data_received
        data_received += 1
        
    except Exception as e:
        log.error(f"Push trade error for {sym}: {e}")


async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        await redis.setex(
            f"ob:{sym}", 
            OB_TTL_SEC, 
            json.dumps({
                "bid": float(bid), 
                "ask": float(ask), 
                "t": int(datetime.utcnow().timestamp() * 1000)
            })
        )
    except Exception as e:
        log.error(f"Push orderbook error: {e}")


async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        await redis.setex(
            f"tk:{sym}", 
            TK_TTL_SEC, 
            json.dumps({
                "last": float(last), 
                "vol": float(vol), 
                "t": int(ts)
            })
        )
    except Exception as e:
        log.error(f"Push ticker error: {e}")


# -----------------------------------------------------------
# WEBSOCKET WORKER (IMPROVED)
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
                                trade_data = {
                                    "p": payload.get("p"),
                                    "q": payload.get("q"),
                                    "m": payload.get("m"),
                                    "t": payload.get("T")
                                }
                                await push_trade(sym, trade_data)

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
                            if k.get("x"):  # only closed candles
                                sym = payload.get("s", "BTCUSDT")
                                close = float(k.get("c", 0))
                                vol = float(k.get("q", 0))
                                ts = int(k.get("T", 0))
                                if close > 0:
                                    await push_ticker(sym, close, vol, ts)

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        log.error(f"WS message parse error: {e}")
                        continue

        except asyncio.CancelledError:
            log.info(f"WS Worker {worker_id} cancelled")
            break
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
            ws_connected = False
            log.warning(f"⚠️ WS Worker {worker_id} closed: {e}. Reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except Exception as e:
            ws_connected = False
            log.error(f"❌ WS Worker {worker_id} error: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def start_all_ws():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    
    tasks = []
    for idx, chunk in enumerate(chunks):
        tasks.append(asyncio.create_task(ws_worker(chunk, idx + 1)))
    
    log.info(f"🔌 Started {len(tasks)} WS workers for {len(pairs)} pairs")
    
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
# SIGNAL FORMATTER
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
    
    try:
        await redis.ping()
        log.info("✅ Redis connected")
    except Exception as e:
        log.error(f"❌ Redis failed: {e}")
        await send_telegram("⚠️ <b>Redis connection failed!</b>")
        return

    await send_telegram("🚀 <b>Scanner Starting...</b>\n⏳ Waiting for data stream")

    # Wait for WebSocket data
    for i in range(30):
        await asyncio.sleep(2)
        btc_count = await redis.llen("tr:BTCUSDT")
        if btc_count >= 10:
            log.info(f"✅ Data ready! BTC trades: {btc_count}")
            await send_telegram(f"✅ <b>Data Stream Active!</b>\n📊 {btc_count} BTC trades loaded")
            break
        log.info(f"⏳ Waiting for data... ({i+1}/30) - BTC trades: {btc_count}")
    else:
        log.warning("⚠️ Data stream timeout!")
        await send_telegram("⚠️ <b>Warning:</b> Data stream slow. Continuing anyway...")

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
                    msg = format_signal(sig)
                    await send_telegram(msg)

                    ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                    set_cooldown(sig["symbol"], sig["strategy"])

                    log.info(f"✔️ SIGNAL: {sig['symbol']} {sig['strategy']} = {sig['score']:.1f}")
            else:
                if scan_count % 10 == 0:  # Log every 10th scan
                    log.info(f"📊 Scan #{scan_count}: No signals")

            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            log.error(traceback.format_exc())
            await asyncio.sleep(5)


# -----------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------
scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task
    
    log.info("🚀 Bot Starting (v2.1)")
    log.info(f"✓ {len(PAIRS)} pairs loaded")
    
    ws_task = asyncio.create_task(start_all_ws())
    await asyncio.sleep(2)  # Let WS start
    
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
        await redis.aclose()  # Use aclose() instead of close()
    except:
        pass
    
    log.info("🔴 App shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    btc_trades = await redis.llen("tr:BTCUSDT")
    
    return {
        "status": "running",
        "ws_connected": ws_connected,
        "data_received": data_received,
        "btc_trades": btc_trades,
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "time": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health():
    try:
        await redis.ping()
        redis_ok = True
    except:
        redis_ok = False
    
    btc_trades = await redis.llen("tr:BTCUSDT") if redis_ok else 0
    
    return {
        "status": "ok" if redis_ok and btc_trades > 0 else "degraded",
        "redis": redis_ok,
        "ws_connected": ws_connected,
        "btc_trades": btc_trades,
        "data_received": data_received
    }


@app.get("/debug")
async def debug():
    """Debug endpoint to check data"""
    debug_info = {}
    
    for sym in PAIRS[:5]:  # Check first 5 pairs
        count = await redis.llen(f"tr:{sym}")
        debug_info[sym] = count
    
    return debug_info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))