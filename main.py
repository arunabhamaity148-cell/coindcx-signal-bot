"""
main.py — final crash-proof version
Lifespan events + lightweight health
Railway keeps pod alive
"""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn
from helpers import run, WS, Exchange, features, calc_tp_sl, position_size, send_telegram, ai_review_ml, regime, CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- lifespan (crash-proof) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting bot...")
    bot_task = asyncio.create_task(bot_loop())
    yield
    bot_task.cancel()
    await bot_task
    log.info("Bot stopped.")

app = FastAPI(lifespan=lifespan)

# ---------- health check ----------
@app.get("/health")
def health():
    return {"status": "ok", "bot": "coindcx-signal", "time": datetime.utcnow().isoformat()}

# ---------- bot loop ----------
async def bot_loop():
    try:
        ws = WS()
        asyncio.create_task(ws.run())          # websocket feed
        await asyncio.sleep(3)                 # let data fill
        ex = Exchange()
        while True:
            for sym in CFG["pairs"]:
                f = await features(sym)
                if not f: continue
                regime_val = await regime(sym)
                score = round((max(f["rsi"], 50) - 50) / 5 + (f["imb"] + 1) * 2 + f["sweep"], 1)
                side = "long" if score >= 7.5 else "short" if score <= 3.0 else "none"
                if side == "none": continue
                atr = 0.4  # dummy ATR (replace with redis ohlcv)
                tp, sl = calc_tp_sl(f["last"], atr, side)
                ai = await ai_review_ml(sym, "QUICK", side, score, f"rsi={f['rsi']:.1f},imb={f['imb']:.2f}", f["spread"] < 0.1, f["depth_usd"] > 1e6, True)
                if not ai["allow"]: continue
                qty = position_size(CFG["equity"], f["last"], sl)
                msg = f"🎯 <b>{sym}</b> {side.upper()}  score={score}  qty={qty:.3f}  TP={tp}  SL={sl}  conf={ai['confidence']}%"
                logging.info(msg); await send_telegram(msg)
                # paper-limit order (uncomment for live)
                # await ex.limit(sym, side, qty, f["last"])
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        log.info("Bot loop cancelled")
    except Exception as e:
        log.exception("Bot loop crashed: %s", e)

# ---------- main ----------
def run_fastapi():
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run_fastapi()
