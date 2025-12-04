"""
main.py — Production FastAPI + Trading Bot  (final)
Keeps Railway alive 24/7, runs ML-powered trading bot
"""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import aiohttp
from helpers_part2 import (   # helpers.py part-2 এখানে import
    WS, Exchange, calculate_advanced_score, ai_review_ensemble,
    calc_smart_tp_sl, position_size, send_telegram,
    check_risk_limits, update_daily_pnl, cleanup, CFG
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("main")

# ---------- global tasks ----------
bot_task = ping_task = ws_task = None

# ---------- 60-s keep-alive ----------
async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url  = f"http://0.0.0.0:{port}/health"
    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.info("✓ Health ping OK")
                    else:
                        log.warning(f"⚠ Health ping {resp.status}")
        except Exception as e:
            log.error(f"Keep-alive error: {e}")

# ---------- trading loop ----------
async def bot_loop():
    log.info("🤖 Starting trading bot...")
    ex = Exchange()
    open_positions = {}
    last_trade_time = {}

    await asyncio.sleep(8)                       # wait for WS data
    log.info("✓ Bot loop started")

    while True:
        try:
            for sym in CFG["pairs"]:
                # --- cooldown ---
                if sym in last_trade_time:
                    if (datetime.utcnow() - last_trade_time[sym]).seconds < 300:
                        continue

                # --- position already open ---
                if sym in open_positions:
                    pos = await ex.get_position(sym)
                    if pos:
                        open_positions[sym] = pos
                    else:                                               # position closed
                        pnl = open_positions[sym].get("unrealizedPnl", 0)
                        del open_positions[sym]
                        await update_daily_pnl(pnl)
                        log.info(f"✓ {sym} closed | PnL ${pnl:.2f}")
                    continue

                # --- risk limits ---
                ok, reason = await check_risk_limits(CFG["equity"], open_positions)
                if not ok:
                    log.info(f"⚠ Risk limit: {reason}")
                    continue

                # --- signal ---
                signal = await calculate_advanced_score(sym)
                if not signal or signal["side"] == "none":
                    continue

                side, score, last = signal["side"], signal["score"], signal["last"]

                # --- thresholds ---
                if side == "long"  and score < CFG["min_score"]:
                    continue
                if side == "short" and score > (10 - CFG["min_score"]):
                    continue

                # --- AI review ---
                ai = await ai_review_ensemble(sym, side, score)
                if not ai["allow"]:
                    log.info(f"❌ {sym} blocked: {ai['reason']} (conf {ai['confidence']}%)")
                    continue

                # --- TP/SL ---
                tp, sl = await calc_smart_tp_sl(sym, side, last)
                risk       = abs(last - sl)
                reward     = abs(tp - last)
                rrr        = reward / risk if risk else 0
                if rrr < CFG["min_rrr"]:
                    continue

                # --- sizing ---
                qty = position_size(CFG["equity"], last, sl, CFG["risk_perc"])
                if qty < 0.001:
                    continue
                leverage = min(int(1 / (risk / last)), CFG["max_lev"])
                await ex.set_leverage(sym, leverage)

                # --- SIGNAL LOG ---
                log.info(f"🎯 SIGNAL: {sym} {side.upper()} | Entry {last:.6f} | TP {tp:.6f} | SL {sl:.6f}")
                log.info(f"   Qty {qty:.4f} | Lev {leverage}x | RRR {rrr:.2f} | ML conf {ai['confidence']}%")

                # --- Telegram ---
                msg = (
                    f"🎯 <b>{sym}</b> {side.upper()}\n"
                    f"Entry <code>{last:.6f}</code>\n"
                    f"TP <code>{tp:.6f}</code> | SL <code>{sl:.6f}</code>\n"
                    f"Qty <b>{qty:.4f}</b> | Lev <b>{leverage}x</b>\n"
                    f"Score {score:.2f}/10 | RRR {rrr:.2f} | ML {ai['confidence']}%"
                )
                await send_telegram(msg)

                # --- LIVE ORDER (Paper=False hard-coded) ---
                order = await ex.limit(sym, side, qty, last, post_only=True)
                if order:
                    await ex.set_sl_tp(sym, side, sl, tp)
                    open_positions[sym] = await ex.get_position(sym)
                    log.info(f"✅ {sym} order {order['id']}")

                last_trade_time[sym] = datetime.utcnow()
                await asyncio.sleep(0.5)

                            await asyncio.sleep(2)          # next symbol
            await asyncio.sleep(1)              # next loop

    except asyncio.CancelledError:
        log.info("🛑 Bot loop cancelled")
        await ex.close()
        raise
    except Exception as e:
        log.exception(f"💥 Bot loop crashed: {e}")
        raise


# ---------- lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task, ping_task, ws_task
    log.info("🚀 Starting application...")

    ws   = WS()
    ws_task   = asyncio.create_task(ws.run())
    bot_task  = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())

    log.info("✓ All tasks started")
    try:
        yield
    finally:
        log.info("🔄 Shutting down...")
        for t in [ws_task, bot_task, ping_task]:
            if t: t.cancel()
        await asyncio.gather(ws_task, bot_task, ping_task, return_exceptions=True)
        await cleanup()
        log.info("✅ Shutdown complete")

# ---------- FastAPI ----------
app = FastAPI(lifespan=lifespan, title="ML-Trader", version="1.1")

@app.get("/")
def root():
    return {"status": "running", "bot": "ML-Trader v1.1", "time": datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "ws": bool(ws_task and not ws_task.done()),
        "bot": bool(bot_task and not bot_task.done()),
        "ping": bool(ping_task and not ping_task.done())
    }

@app.get("/stats")
async def stats():
    from helpers_part1 import redis
    r = await redis()
    daily = float(await r.get("daily_pnl") or 0)
    return {"daily_pnl": daily, "equity": CFG["equity"], "pairs": CFG["pairs"],
            "min_score": CFG["min_score"], "min_rrr": CFG["min_rrr"]}

# ---------- run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)), log_level="info")
