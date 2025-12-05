# main.py
import os
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI
import uvicorn

from helpers import (
    PAIRS, poll_ticker, poll_orderbook, get_score, close
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

app = FastAPI()
_tasks = []

SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL", 5.0))   # seconds
TOPN = int(os.getenv("TOPN", 5))

async def scanner_loop():
    log.info("🔍 scanner started")
    while True:
        try:
            results = []
            for p in PAIRS:
                try:
                    s = await get_score(p)
                    if s:
                        results.append(s)
                except Exception as e:
                    log.debug("score error %s: %s", p, e)
            results.sort(key=lambda x: x["score"], reverse=True)
            if results:
                top = results[:TOPN]
                log.info("📊 Top signals:")
                for t in top:
                    log.info(" - %s score=%.2f price=%.6f rsi=%.1f imb=%.3f spread=%.3f",
                             t["pair"], t["score"], t["price"], t["rsi"], t["imbalance"], t["spread"])
            else:
                log.info("📊 Found 0 signals")
            await asyncio.sleep(SCAN_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("scanner_loop error: %s", e)
            await asyncio.sleep(3)

# Lifespan: start pollers + scanner
@app.on_event("startup")
async def startup_event():
    global _tasks
    log.info("🚀 App starting (CoinDCX REST poller)")
    # start pollers per pair
    for p in PAIRS:
        t1 = asyncio.create_task(poll_ticker(p))
        t2 = asyncio.create_task(poll_orderbook(p))
        _tasks += [t1, t2]
    # start scanner
    _tasks.append(asyncio.create_task(scanner_loop()))
    log.info("✅ All tasks started")

@app.on_event("shutdown")
async def shutdown_event():
    log.info("🔄 Shutting down tasks")
    for t in _tasks:
        t.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    await close()
    log.info("✅ Shutdown complete")

@app.get("/")
def root():
    return {"status": "running", "time": datetime.utcnow().isoformat(), "pairs": len(PAIRS)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info")