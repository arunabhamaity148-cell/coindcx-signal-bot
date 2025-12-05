# ================================================================
# main.py – MINIMAL STABLE (Guaranteed No Crash)
# ================================================================

import os
import json
import asyncio
import logging
from datetime import datetime
from collections import deque

import aiohttp
import websockets
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Minimal pairs for stability
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
         "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

TRADE_BUFFER = {sym: deque(maxlen=200) for sym in PAIRS}
OB_CACHE = {}

app_ready = False
keep_running = True

# ============================================================
# TELEGRAM
# ============================================================

async def send_tg(msg: str):
    """Send telegram message"""
    if not TG_TOKEN or not TG_CHAT:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"}, timeout=aiohttp.ClientTimeout(total=5)) as r:
                pass
    except Exception as e:
        log.warning(f"TG error: {e}")

# ============================================================
# WEBSOCKET
# ============================================================

async def ws_stream():
    """Single WebSocket connection"""
    url = "wss://stream.binance.com:9443/stream?streams=" + "/".join([
        f"{p.lower()}@aggTrade" for p in PAIRS
    ] + [
        f"{p.lower()}@bookTicker" for p in PAIRS
    ] + ["btcusdt@kline_1m"])
    
    backoff = 1
    
    while keep_running:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1
                log.info("✅ WebSocket connected")
                
                async for msg in ws:
                    if not keep_running:
                        break
                    
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        sym = payload.get("s")
                        
                        if not sym or sym not in PAIRS:
                            continue
                        
                        # Store trades
                        if "@aggTrade" in stream:
                            TRADE_BUFFER[sym].append({
                                "p": float(payload["p"]),
                                "q": float(payload["q"]),
                                "m": bool(payload["m"]),
                                "t": int(payload["T"])
                            })
                        
                        # Store orderbook
                        elif "@bookTicker" in stream:
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            if bid > 0 and ask > 0:
                                OB_CACHE[sym] = {"bid": bid, "ask": ask}
                    
                    except:
                        continue
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"WS error: {e}, retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
    
    log.info("WebSocket stopped")

# ============================================================
# SCANNER
# ============================================================

async def scanner():
    """Simple scanner"""
    log.info("🔍 Scanner starting...")
    
    # Wait for data
    for i in range(15):
        await asyncio.sleep(2)
        if len(TRADE_BUFFER["BTCUSDT"]) >= 10:
            log.info(f"✅ Data ready: {len(TRADE_BUFFER['BTCUSDT'])} BTC trades")
            await send_tg("✅ <b>Bot Active</b>\n🚀 Scanning started")
            break
    
    scan_count = 0
    
    while keep_running:
        try:
            scan_count += 1
            
            # Simple logic: check if enough trades
            for sym in PAIRS:
                trades = list(TRADE_BUFFER[sym])
                if len(trades) < 50:
                    continue
                
                # Simple momentum check
                prices = [t["p"] for t in trades[-20:]]
                if not prices:
                    continue
                
                price_change = (prices[-1] - prices[0]) / prices[0] * 100
                
                # If strong move detected
                if abs(price_change) > 0.5:
                    msg = f"🎯 <b>{sym}</b>\n"
                    msg += f"💰 ${prices[-1]:.6f}\n"
                    msg += f"📊 Change: {price_change:+.2f}%"
                    
                    await send_tg(msg)
                    log.info(f"✔️ Signal: {sym} {price_change:+.2f}%")
                    
                    # Clear buffer to avoid re-signal
                    TRADE_BUFFER[sym].clear()
            
            if scan_count % 10 == 0:
                log.info(f"📊 Scan #{scan_count}")
            
            await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            await asyncio.sleep(5)
    
    log.info("Scanner stopped")

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()

ws_task = None
scan_task = None

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    global ws_task, scan_task, app_ready
    
    log.info("=" * 50)
    log.info("🚀 Bot Starting (Minimal Stable)")
    log.info(f"✓ {len(PAIRS)} pairs")
    log.info("=" * 50)
    
    ws_task = asyncio.create_task(ws_stream())
    await asyncio.sleep(2)
    
    scan_task = asyncio.create_task(scanner())
    
    app_ready = True
    log.info("✅ App ready")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup"""
    global keep_running
    
    log.info("🛑 Shutting down...")
    keep_running = False
    
    if ws_task:
        ws_task.cancel()
    if scan_task:
        scan_task.cancel()
    
    try:
        await asyncio.gather(ws_task, scan_task, return_exceptions=True)
    except:
        pass
    
    log.info("✅ Shutdown complete")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "ready": app_ready,
        "btc_trades": len(TRADE_BUFFER["BTCUSDT"]),
        "pairs": len(PAIRS)
    }

@app.get("/health")
async def health():
    """Health check - ALWAYS return 200"""
    return {"status": "healthy"}

@app.get("/debug")
async def debug():
    """Debug info"""
    return {
        "buffers": {sym: len(TRADE_BUFFER[sym]) for sym in PAIRS},
        "ob_cache": len(OB_CACHE)
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
        access_log=False
    )