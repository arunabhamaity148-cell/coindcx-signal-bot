# main.py
import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from helpers import (
    PAIRS as UI_PAIRS,
    normalize_all_pairs,
    poll_ticker,
    poll_orderbook,
    get_score,
    pair_candidates,
    close
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL", 5.0))   # seconds
TOPN = int(os.getenv("TOPN", 5))

# tasks and runtime state
_tasks = []
NORMALIZED_PAIRS = []  # will be filled at startup

async def scanner_loop():
    log.info("🔍 scanner started")
    while True:
        try:
            results = []
            for p in NORMALIZED_PAIRS:
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

# lifespan to manage startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tasks, NORMALIZED_PAIRS

    log.info("🚀 App starting (CoinDCX REST poller w/ discovery)")

    # Normalize UI pairs -> API instruments
    try:
        NORMALIZED_PAIRS = await normalize_all_pairs()
        log.info("Normalized pairs: %s", NORMALIZED_PAIRS)
    except Exception as e:
        log.warning("normalize_all_pairs failed: %s", e)
        # fallback to UI_PAIRS if normalization fails
        NORMALIZED_PAIRS = UI_PAIRS

    # start pollers per pair
    for p in NORMALIZED_PAIRS:
        t1 = asyncio.create_task(poll_ticker(p))
        t2 = asyncio.create_task(poll_orderbook(p))
        _tasks += [t1, t2]

    # start scanner
    _tasks.append(asyncio.create_task(scanner_loop()))

    log.info("✅ All tasks started")
    try:
        yield
    finally:
        log.info("🔄 Shutting down tasks")
        for t in _tasks:
            t.cancel()
        await asyncio.gather(*_tasks, return_exceptions=True)
        await close()
        log.info("✅ Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "running",
        "time": datetime.utcnow().isoformat(),
        "ui_pairs": UI_PAIRS,
        "normalized_pairs": NORMALIZED_PAIRS
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "tasks_running": len([t for t in _tasks if not t.done()]) if _tasks else 0,
        "pairs": len(NORMALIZED_PAIRS)
    }

@app.get("/candidates/{ui_pair}")
async def candidates(ui_pair: str):
    """
    Return candidate matches from /symbols for a given UI pair (debug helper).
    """
    matches = await pair_candidates(ui_pair)
    return {"ui_pair": ui_pair, "candidates_count": len(matches), "candidates": matches[:20]}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info")