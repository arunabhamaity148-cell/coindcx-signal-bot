"""
main.py — health check + background bot
Railway keeps container alive
"""
import os
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI
import uvicorn
from helpers import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- health check ----------
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "bot": "coindcx-signal", "time": datetime.utcnow().isoformat()}

# ---------- background bot ----------
async def bot_loop():
    try:
        await run()
    except asyncio.CancelledError:
        log.info("Bot loop cancelled")
    except Exception as e:
        log.exception("Bot crashed: %s", e)

# ---------- startup ----------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(bot_loop())
    log.info("Background bot started")

# ---------- main ----------
def run_fastapi():
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_fastapi()
