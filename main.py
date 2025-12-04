"""
main.py — crash-proof + 60-s health-ping
Keeps Railway free-tier awake 24×7
"""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import aiohttp
from helpers import run, WS, Exchange, features, calc_tp_sl, position_size, send_telegram, ai_review_ml, regime, CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- Global tasks ----------
bot_task = None
ping_task = None

# ---------- 60-s keep-alive ----------
async def keep_alive():
    url = f"http://0.0.0.0:{os.getenv('PORT', 8080)}/health"
    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        log.info("✓ Health ping OK")
                    else:
                        log.warning(f"⚠ Health ping failed: {r.status}")
        except Exception as e:
            log.error(f"Keep-alive error: {e}")

# ---------- bot loop ----------
async def bot_loop():
    log.info("🤖 Bot loop starting...")
    try:
        ws = WS()
        asyncio.create_task(ws.run())
        await asyncio.sleep(3)
        ex = Exchange()
        
        while True:
            for sym in CFG["pairs"]:
                try:
                    f = await features(sym)
                    if not f: 
                        continue
                    
                    regime_val = await regime(sym)
                    score = round((max(f["rsi"], 50) - 50) / 5 + (f["imb"] + 1) * 2 + f["sweep"], 1)
                    side = "long" if score >= 7.5 else "short" if score <= 3.0 else "none"
                    
                    if side == "none": 
                        continue
                    
                    atr = 0.4
                    tp, sl = calc_tp_sl(f["last"], atr, side)
                    ai = await ai_review_ml(
                        sym, "QUICK", side, score, 
                        f"rsi={f['rsi']:.1f},imb={f['imb']:.2f}", 
                        f["spread"] < 0.1, f["depth_usd"] > 1e6, True
                    )
                    
                    if not ai["allow"]: 
                        continue
                    
                    qty = position_size(CFG["equity"], f["last"], sl)
                    msg = f"🎯 <b>{sym}</b> {side.upper()}  score={score}  qty={qty:.3f}  TP={tp}  SL={sl}  conf={ai['confidence']}%"
                    log.info(msg)
                    await send_telegram(msg)
                    
                except Exception as e:
                    log.error(f"Error processing {sym}: {e}")
                    continue
            
            await asyncio.sleep(5)
            
    except asyncio.CancelledError:
        log.info("🛑 Bot loop cancelled")
        raise
    except Exception as e:
        log.exception(f"💥 Bot loop crashed: {e}")
        raise

# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task, ping_task
    
    log.info("🚀 Starting bot + keep-alive...")
    bot_task = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())
    
    try:
        yield  # FastAPI runs here
    finally:
        log.info("🔄 Shutting down tasks...")
        if bot_task:
            bot_task.cancel()
        if ping_task:
            ping_task.cancel()
        
        # Wait for clean shutdown
        await asyncio.gather(bot_task, ping_task, return_exceptions=True)
        log.info("✅ All tasks stopped")

app = FastAPI(lifespan=lifespan)

# ---------- health check ----------
@app.get("/health")
def health():
    return {
        "status": "ok", 
        "bot": "coindcx-signal", 
        "time": datetime.utcnow().isoformat(),
        "bot_running": bot_task and not bot_task.done() if bot_task else False,
        "ping_running": ping_task and not ping_task.done() if ping_task else False
    }

# ---------- main ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")