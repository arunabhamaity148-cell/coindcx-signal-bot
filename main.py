# ================================================================
# main.py – v3.0 PRODUCTION (Error-Free, Railway Optimized)
# ================================================================

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
from fastapi import FastAPI, Response
import aiofiles

from helpers import PAIRS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("main")

# ============================================================
# CONFIGURATION
# ============================================================

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 20))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))
BATCH_SIZE = int(os.getenv("WRITE_BATCH_SIZE", 50))
SYNC_INTERVAL = int(os.getenv("PERIODIC_SYNC_INTERVAL", 15))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 15))

# ============================================================
# GLOBAL STATE
# ============================================================

# In-memory data
TRADE_BUFFER = {sym: deque(maxlen=500) for sym in PAIRS}
OB_CACHE = {}
TK_CACHE = {}

# Counters
write_counters = {sym: 0 for sym in PAIRS}
data_received = 0
cooldown = {}
ACTIVE_ORDERS = {}

# Status flags
ws_connected = False
app_ready = False
shutdown_flag = False

# Storage directories
TRADE_DIR = "/tmp/trades"
OB_DIR = "/tmp/ob"
TK_DIR = "/tmp/tk"

for directory in (TRADE_DIR, OB_DIR, TK_DIR):
    os.makedirs(directory, exist_ok=True)

# ============================================================
# TELEGRAM
# ============================================================

async def send_telegram(message: str):
    """Send Telegram notification"""
    if not TG_TOKEN or not TG_CHAT:
        return
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    log.warning(f"Telegram error: {resp.status}")
    except asyncio.TimeoutError:
        log.warning("Telegram timeout")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")

# ============================================================
# FILE PERSISTENCE
# ============================================================

async def persist_trades(sym: str, trades: list):
    """Append trades to file"""
    if not trades:
        return
    
    filepath = os.path.join(TRADE_DIR, f"{sym}.jsonl")
    try:
        async with aiofiles.open(filepath, "a") as f:
            for trade in trades:
                await f.write(json.dumps(trade) + "\n")
    except Exception as e:
        log.error(f"Failed to persist trades for {sym}: {e}")

async def persist_orderbooks(ob_dict: dict):
    """Save orderbook snapshots"""
    if not ob_dict:
        return
    
    try:
        for sym, data in ob_dict.items():
            filepath = os.path.join(OB_DIR, f"{sym}.json")
            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(data))
    except Exception as e:
        log.error(f"Failed to persist orderbooks: {e}")

async def persist_tickers(tk_dict: dict):
    """Save ticker data"""
    if not tk_dict:
        return
    
    try:
        for sym, data in tk_dict.items():
            filepath = os.path.join(TK_DIR, f"{sym}.json")
            async with aiofiles.open(filepath, "w") as f:
                await f.write(json.dumps(data))
    except Exception as e:
        log.error(f"Failed to persist tickers: {e}")

# ============================================================
# DATA INGESTION
# ============================================================

async def push_trade(sym: str, data: dict):
    """Store trade in memory buffer"""
    global data_received
    
    try:
        if not all(k in data for k in ("p", "q", "m", "t")):
            return
        
        trade = {
            "p": float(data["p"]),
            "q": float(data["q"]),
            "m": bool(data["m"]),
            "t": int(data["t"])
        }
        
        TRADE_BUFFER[sym].append(trade)
        write_counters[sym] += 1
        data_received += 1
        
        # Batch write to disk
        if write_counters[sym] >= BATCH_SIZE:
            batch = list(TRADE_BUFFER[sym])[-BATCH_SIZE:]
            await persist_trades(sym, batch)
            write_counters[sym] = 0
    
    except Exception as e:
        log.error(f"push_trade error for {sym}: {e}")

async def push_orderbook(sym: str, bid: float, ask: float):
    """Update orderbook cache"""
    try:
        OB_CACHE[sym] = {
            "bid": float(bid),
            "ask": float(ask),
            "t": int(time.time() * 1000)
        }
    except Exception as e:
        log.error(f"push_orderbook error: {e}")

async def push_ticker(sym: str, last: float, vol: float, ts: int):
    """Update ticker cache"""
    try:
        TK_CACHE[sym] = {
            "last": float(last),
            "vol": float(vol),
            "t": int(ts)
        }
    except Exception as e:
        log.error(f"push_ticker error: {e}")

# ============================================================
# PERIODIC SYNC
# ============================================================

async def periodic_sync():
    """Periodically write cache to disk"""
    while not shutdown_flag:
        try:
            await asyncio.sleep(SYNC_INTERVAL)
            
            if OB_CACHE:
                await persist_orderbooks(dict(OB_CACHE))
            
            if TK_CACHE:
                await persist_tickers(dict(TK_CACHE))
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"periodic_sync error: {e}")

# ============================================================
# WEBSOCKET
# ============================================================

WS_BASE = "wss://stream.binance.com:9443/stream"

def build_stream_url(pairs_chunk: list) -> str:
    """Build WebSocket URL for pair chunk"""
    streams = []
    
    for pair in pairs_chunk:
        p_lower = pair.lower()
        streams.append(f"{p_lower}@aggTrade")
        streams.append(f"{p_lower}@bookTicker")
    
    if "BTCUSDT" in pairs_chunk:
        streams.append("btcusdt@kline_1m")
    
    stream_path = "/".join(streams)
    return f"{WS_BASE}?streams={stream_path}"

async def ws_worker(pairs_chunk: list, worker_id: int):
    """WebSocket worker for a chunk of pairs"""
    global ws_connected
    
    url = build_stream_url(pairs_chunk)
    backoff = 1
    
    log.info(f"WS-{worker_id} starting ({len(pairs_chunk)} pairs)")
    
    while not shutdown_flag:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                backoff = 1
                ws_connected = True
                log.info(f"✅ WS-{worker_id} connected")
                
                async for message in ws:
                    if shutdown_flag:
                        break
                    
                    try:
                        data = json.loads(message)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        
                        sym = payload.get("s")
                        if not sym or sym not in PAIRS:
                            continue
                        
                        # Handle aggTrade
                        if stream.endswith("@aggTrade"):
                            await push_trade(sym, {
                                "p": payload.get("p"),
                                "q": payload.get("q"),
                                "m": payload.get("m"),
                                "t": payload.get("T")
                            })
                        
                        # Handle bookTicker
                        elif stream.endswith("@bookTicker"):
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            if bid > 0 and ask > 0:
                                await push_orderbook(sym, bid, ask)
                        
                        # Handle kline
                        elif "@kline" in stream:
                            k = payload.get("k", {})
                            if k.get("x"):  # Closed candle
                                close = float(k.get("c", 0))
                                vol = float(k.get("q", 0))
                                ts = int(k.get("T", 0))
                                if close > 0:
                                    await push_ticker(sym, close, vol, ts)
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        log.warning(f"WS-{worker_id} message error: {e}")
                        continue
        
        except asyncio.CancelledError:
            log.info(f"WS-{worker_id} cancelled")
            break
        
        except Exception as e:
            ws_connected = False
            log.warning(f"⚠️ WS-{worker_id} disconnected: {e}. Reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
    
    log.info(f"WS-{worker_id} stopped")

async def start_websockets():
    """Start all WebSocket workers"""
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    
    tasks = []
    for idx, chunk in enumerate(chunks):
        task = asyncio.create_task(ws_worker(chunk, idx + 1))
        tasks.append(task)
    
    # Add periodic sync task
    tasks.append(asyncio.create_task(periodic_sync()))
    
    log.info(f"🔌 Started {len(tasks)-1} WS workers + sync task")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()

# ============================================================
# COOLDOWN
# ============================================================

def cooldown_ok(sym: str, strat: str) -> bool:
    """Check if cooldown period has passed"""
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    
    elapsed_min = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return elapsed_min >= COOLDOWN_MIN

def set_cooldown(sym: str, strat: str):
    """Set cooldown timestamp"""
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()

# ============================================================
# SCANNER
# ============================================================

async def scanner():
    """Main signal scanning loop"""
    log.info("🔍 Scanner starting...")
    
    await send_telegram("🚀 <b>Bot Started</b>\n⏳ Waiting for data stream...")
    
    # Wait for data
    for i in range(30):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUFFER.get("BTCUSDT", []))
        
        if btc_count >= 20:
            log.info(f"✅ Data ready! BTC trades: {btc_count}")
            await send_telegram(f"✅ <b>Data Stream Active!</b>\n📊 {btc_count} BTC trades loaded")
            break
        
        if i % 5 == 0:
            log.info(f"Waiting for data... ({i+1}/30) - BTC: {btc_count}")
    else:
        log.warning("⚠️ Data stream slow, continuing anyway")
    
    scan_count = 0
    
    while not shutdown_flag:
        try:
            scan_count += 1
            results = []
            
            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue
                    
                    try:
                        # Import here to avoid circular dependency
                        from scorer import compute_signal
                        
                        sig = await compute_signal(sym, strat, TRADE_BUFFER, OB_CACHE)
                        
                        if sig and sig.get("validated", True):
                            results.append(sig)
                    
                    except Exception as e:
                        log.error(f"Signal error {sym}/{strat}: {e}")
                        continue
            
            # Send top signals
            if results:
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                best = results[:3]
                
                for sig in best:
                    try:
                        # Import formatter here
                        from telegram_formatter import TelegramFormatter
                        formatter = TelegramFormatter()
                        
                        msg = formatter.format_signal_alert(
                            sig,
                            sig.get("levels"),
                            sig.get("volume")
                        )
                        
                        await send_telegram(msg)
                        
                        ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                        set_cooldown(sig["symbol"], sig["strategy"])
                        
                        log.info(f"✔️ SIGNAL: {sig['symbol']} {sig['strategy']} = {sig['score']:.1f}")
                    
                    except Exception as e:
                        log.error(f"Failed to send signal: {e}")
            
            else:
                if scan_count % 20 == 0:
                    log.info(f"📊 Scan #{scan_count}: No signals")
            
            await asyncio.sleep(SCAN_INTERVAL)
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            log.error(traceback.format_exc())
            await asyncio.sleep(5)
    
    log.info("Scanner stopped")

# ============================================================
# FASTAPI APPLICATION
# ============================================================

scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global scan_task, ws_task, app_ready, shutdown_flag
    
    log.info("=" * 60)
    log.info("🚀 Bot Starting (v3.0 Production)")
    log.info(f"✓ {len(PAIRS)} pairs configured")
    log.info("=" * 60)
    
    # Start WebSocket workers
    ws_task = asyncio.create_task(start_websockets())
    await asyncio.sleep(3)
    
    # Start scanner
    scan_task = asyncio.create_task(scanner())
    
    # Mark app as ready
    app_ready = True
    log.info("✅ Application ready")
    
    yield
    
    # Shutdown
    log.info("🛑 Shutting down...")
    shutdown_flag = True
    
    if scan_task:
        scan_task.cancel()
    if ws_task:
        ws_task.cancel()
    
    try:
        await asyncio.gather(scan_task, ws_task, return_exceptions=True)
    except Exception as e:
        log.error(f"Shutdown error: {e}")
    
    log.info("🔴 Shutdown complete")

app = FastAPI(lifespan=lifespan)

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))
    
    return {
        "status": "running" if app_ready else "starting",
        "ws_connected": ws_connected,
        "data_received": data_received,
        "btc_trades": btc_trades,
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "time": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint for Railway"""
    if not app_ready:
        return Response(
            content=json.dumps({"status": "starting"}),
            status_code=503,
            media_type="application/json"
        )
    
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))
    
    # Return 200 only if truly healthy
    if btc_trades > 0:
        return {
            "status": "healthy",
            "btc_trades": btc_trades,
            "ws_connected": ws_connected,
            "data_received": data_received
        }
    else:
        return Response(
            content=json.dumps({
                "status": "degraded",
                "btc_trades": btc_trades,
                "ws_connected": ws_connected
            }),
            status_code=503,
            media_type="application/json"
        )

@app.get("/debug")
async def debug():
    """Debug endpoint"""
    debug_data = {}
    
    for sym in list(PAIRS)[:10]:
        debug_data[sym] = len(TRADE_BUFFER.get(sym, []))
    
    return {
        "trade_counts": debug_data,
        "ob_cache_size": len(OB_CACHE),
        "ticker_cache_size": len(TK_CACHE),
        "ws_connected": ws_connected,
        "app_ready": app_ready
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )