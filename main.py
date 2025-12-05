# ================================================================
# main.py – v3.1 RAILWAY OPTIMIZED (Ultra Stable)
# ================================================================

import os
import json
import asyncio
import logging
import signal
import sys
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

import aiohttp
import websockets
import uvicorn
from fastapi import FastAPI

from helpers import PAIRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("main")

# ============================================================
# CONFIGURATION
# ============================================================

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 25))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 12))  # Reduced for stability

# Reduce pairs for stability
ACTIVE_PAIRS = PAIRS[:24]  # Only use first 24 pairs

# ============================================================
# GLOBAL STATE
# ============================================================

TRADE_BUFFER = {sym: deque(maxlen=300) for sym in ACTIVE_PAIRS}
OB_CACHE = {}
data_received = 0
cooldown = {}
ws_connected = False
app_ready = False
shutdown_requested = False

# ============================================================
# SIGNAL HANDLERS
# ============================================================

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    global shutdown_requested
    log.info(f"⚠️ Received signal {signum}, initiating shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ============================================================
# TELEGRAM
# ============================================================

async def send_telegram(message: str):
    """Send Telegram message with error suppression"""
    if not TG_TOKEN or not TG_CHAT:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(
                url,
                json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=5)
            )
    except:
        pass  # Silently fail

# ============================================================
# DATA INGESTION
# ============================================================

async def push_trade(sym: str, data: dict):
    """Store trade in buffer"""
    global data_received
    
    try:
        if not all(k in data for k in ("p", "q", "m", "t")):
            return
        
        TRADE_BUFFER[sym].append({
            "p": float(data["p"]),
            "q": float(data["q"]),
            "m": bool(data["m"]),
            "t": int(data["t"])
        })
        data_received += 1
    except:
        pass

async def push_orderbook(sym: str, bid: float, ask: float):
    """Update orderbook"""
    try:
        OB_CACHE[sym] = {
            "bid": float(bid),
            "ask": float(ask),
            "t": int(datetime.utcnow().timestamp() * 1000)
        }
    except:
        pass

# ============================================================
# WEBSOCKET
# ============================================================

WS_BASE = "wss://stream.binance.com:9443/stream"

def build_stream_url(pairs: list) -> str:
    """Build WebSocket URL"""
    streams = []
    for p in pairs:
        pl = p.lower()
        streams.append(f"{pl}@aggTrade")
        streams.append(f"{pl}@bookTicker")
    
    if "BTCUSDT" in pairs:
        streams.append("btcusdt@kline_1m")
    
    return f"{WS_BASE}?streams={'/'.join(streams)}"

async def ws_worker(pairs: list, wid: int):
    """WebSocket worker"""
    global ws_connected
    
    url = build_stream_url(pairs)
    backoff = 1
    
    log.info(f"WS-{wid} starting ({len(pairs)} pairs)")
    
    while not shutdown_requested:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            ) as ws:
                backoff = 1
                ws_connected = True
                log.info(f"✅ WS-{wid} connected")
                
                async for msg in ws:
                    if shutdown_requested:
                        break
                    
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        sym = payload.get("s")
                        
                        if not sym or sym not in ACTIVE_PAIRS:
                            continue
                        
                        if stream.endswith("@aggTrade"):
                            await push_trade(sym, {
                                "p": payload["p"],
                                "q": payload["q"],
                                "m": payload["m"],
                                "t": payload["T"]
                            })
                        
                        elif stream.endswith("@bookTicker"):
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            if bid > 0 and ask > 0:
                                await push_orderbook(sym, bid, ask)
                    
                    except:
                        continue
        
        except asyncio.CancelledError:
            break
        except:
            ws_connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
    
    log.info(f"WS-{wid} stopped")

async def start_websockets():
    """Start WebSocket workers"""
    chunks = [ACTIVE_PAIRS[i:i+CHUNK_SIZE] for i in range(0, len(ACTIVE_PAIRS), CHUNK_SIZE)]
    
    tasks = [
        asyncio.create_task(ws_worker(chunk, i+1))
        for i, chunk in enumerate(chunks)
    ]
    
    log.info(f"🔌 Started {len(tasks)} WS workers")
    
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except:
        pass

# ============================================================
# COOLDOWN
# ============================================================

def cooldown_ok(sym: str, strat: str) -> bool:
    """Check cooldown"""
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    elapsed = (datetime.utcnow() - cooldown[key]).total_seconds() / 60
    return elapsed >= COOLDOWN_MIN

def set_cooldown(sym: str, strat: str):
    """Set cooldown"""
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()

# ============================================================
# SCANNER
# ============================================================

async def scanner():
    """Signal scanner"""
    log.info("🔍 Scanner starting...")
    
    # Wait for data
    for i in range(20):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUFFER.get("BTCUSDT", []))
        if btc_count >= 15:
            log.info(f"✅ Data ready: {btc_count} BTC trades")
            await send_telegram(f"✅ Bot Active\n📊 {btc_count} trades loaded")
            break
    
    scan_count = 0
    
    while not shutdown_requested:
        try:
            scan_count += 1
            
            # Simple scan
            for sym in ACTIVE_PAIRS:
                if shutdown_requested:
                    break
                
                for strat in ["QUICK", "MID"]:  # Only 2 strategies
                    if not cooldown_ok(sym, strat):
                        continue
                    
                    try:
                        from scorer import compute_signal
                        sig = await compute_signal(sym, strat, TRADE_BUFFER, OB_CACHE)
                        
                        if sig and sig.get("score", 0) >= 7.5:
                            # Simple notification
                            msg = f"🎯 {sym} {sig['side'].upper()}\n💰 ${sig['last']:.6f}\n📊 Score: {sig['score']:.1f}"
                            await send_telegram(msg)
                            set_cooldown(sym, strat)
                            log.info(f"✔️ {sym} {strat} = {sig['score']:.1f}")
                    except:
                        continue
            
            if scan_count % 10 == 0:
                log.info(f"📊 Scan #{scan_count} complete")
            
            await asyncio.sleep(SCAN_INTERVAL)
        
        except asyncio.CancelledError:
            break
        except:
            await asyncio.sleep(5)
    
    log.info("Scanner stopped")

# ============================================================
# FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan"""
    global app_ready
    
    log.info("=" * 50)
    log.info("🚀 Bot Starting (v3.1 Railway)")
    log.info(f"✓ {len(ACTIVE_PAIRS)} pairs")
    log.info("=" * 50)
    
    # Start tasks
    ws_task = asyncio.create_task(start_websockets())
    await asyncio.sleep(2)
    
    scan_task = asyncio.create_task(scanner())
    
    app_ready = True
    log.info("✅ App ready")
    
    yield
    
    log.info("🛑 Shutting down...")
    ws_task.cancel()
    scan_task.cancel()
    
    try:
        await asyncio.gather(ws_task, scan_task, return_exceptions=True)
    except:
        pass
    
    log.info("✅ Shutdown complete")

app = FastAPI(lifespan=lifespan)

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "ready": app_ready,
        "ws": ws_connected,
        "btc": len(TRADE_BUFFER.get("BTCUSDT", [])),
        "pairs": len(ACTIVE_PAIRS),
        "data_rx": data_received,
        "time": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Health check - always return 200"""
    return {
        "status": "healthy",
        "ready": app_ready,
        "ws": ws_connected,
        "uptime": "ok"
    }

@app.get("/ping")
async def ping():
    """Simple ping"""
    return {"pong": True}

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    # Railway-optimized settings
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,  # Disable access logs
        timeout_keep_alive=120,
        limit_concurrency=100,
        backlog=128
    )
    
    server = uvicorn.Server(config)
    server.run()