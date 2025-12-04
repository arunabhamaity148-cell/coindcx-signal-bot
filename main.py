# main.py — FINAL (Production FastAPI + Trading Bot)
# Auto Telegram alert on deploy + full trading loop + /debug endpoint
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
import uvicorn
import aiohttp

from helpers import (   # single helpers.py (part1/part2 should be present)
    WS, Exchange, calculate_advanced_score, ai_review_ensemble,
    calc_smart_tp_sl, position_size, send_telegram,
    check_risk_limits, update_daily_pnl, cleanup, CFG, redis
)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- global tasks ----------
bot_task = ping_task = ws_task = None

# ---------- 60-s keep-alive ----------
async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url  = f"http://127.0.0.1:{port}/health"
    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.debug("✓ Health ping OK")
                    else:
                        log.warning(f"⚠ Health ping {resp.status}")
        except Exception as e:
            log.warning(f"Keep-alive error: {e}")
            await asyncio.sleep(5)

# ---------- trading loop ----------
async def bot_loop():
    log.info("🤖 Starting trading bot...")
    ex = Exchange()
    open_positions = {}
    last_trade_time = {}

    # small warmup to let WS populate Redis
    await asyncio.sleep(int(os.getenv("STARTUP_DELAY", "8")))
    log.info("✓ Bot loop started")

    try:
        while True:
            for sym in CFG["pairs"]:
                try:
                    # cooldown (per-symbol)
                    if sym in last_trade_time:
                        if (datetime.utcnow() - last_trade_time[sym]).seconds < int(os.getenv("SYMBOL_COOLDOWN", "300")):
                            continue

                    # position already open?
                    if sym in open_positions:
                        pos = await ex.get_position(sym)
                        if pos:
                            open_positions[sym] = pos
                        else:
                            # position closed => update pnl and remove
                            pnl = open_positions[sym].get("unrealizedPnl", 0)
                            del open_positions[sym]
                            await update_daily_pnl(pnl)
                            log.info(f"✓ {sym} closed | PnL ${pnl:.2f}")
                        continue

                    # global risk limits
                    ok, reason = await check_risk_limits(CFG["equity"], open_positions)
                    if not ok:
                        log.info(f"⚠ Risk limit: {reason}")
                        continue

                    # compute advanced score
                    signal = await calculate_advanced_score(sym)
                    if not signal or signal.get("side") == "none":
                        continue

                    side, score, last = signal["side"], signal["score"], signal["last"]

                    # thresholds
                    if side == "long" and score < CFG["min_score"]:
                        continue
                    if side == "short" and score > (10 - CFG["min_score"]):
                        continue

                    # AI review ensemble
                    ai = await ai_review_ensemble(sym, side, score)
                    if not ai.get("allow", False):
                        log.info(f"❌ {sym} blocked: {ai.get('reason')} (conf {ai.get('confidence')}%)")
                        continue

                    # TP/SL + RRR check
                    tp, sl = await calc_smart_tp_sl(sym, side, last)
                    risk       = abs(last - sl)
                    reward     = abs(tp - last)
                    rrr        = (reward / risk) if risk else 0.0
                    if rrr < CFG["min_rrr"]:
                        log.debug(f"SKIP {sym} rrr {rrr:.2f} < {CFG['min_rrr']}")
                        continue

                    # sizing
                    qty = position_size(CFG["equity"], last, sl, CFG["risk_perc"])
                    if qty < float(os.getenv("MIN_QTY", "0.0001")):
                        log.debug(f"SKIP {sym} qty too small {qty}")
                        continue

                    leverage = min(int(max(1, 1 / (risk / last))) if risk else CFG["max_lev"], CFG["max_lev"])
                    await ex.set_leverage(sym, leverage)

                    # log + telegram
                    log.info(f"🎯 SIGNAL: {sym} {side.upper()} | Entry {last:.6f} | TP {tp:.6f} | SL {sl:.6f}")
                    log.info(f"   Qty {qty:.4f} | Lev {leverage}x | RRR {rrr:.2f} | ML conf {ai.get('confidence')}%")

                    msg = (
                        f"🎯 <b>{sym}</b> {side.upper()}\n"
                        f"Entry <code>{last:.6f}</code>\n"
                        f"TP <code>{tp:.6f}</code> | SL <code>{sl:.6f}</code>\n"
                        f"Qty <b>{qty:.4f}</b> | Lev <b>{leverage}x</b>\n"
                        f"Score {score:.2f}/10 | RRR {rrr:.2f} | ML {ai.get('confidence')}%"
                    )
                    await send_telegram(msg)

                    # place limit (paper/live depends on helpers.Exchange config)
                    order = await ex.limit(sym, side, qty, last, post_only=True)
                    if order:
                        await ex.set_sl_tp(sym, side, sl, tp)
                        open_positions[sym] = await ex.get_position(sym)
                        log.info(f"✅ {sym} order placed")
                    else:
                        log.warning(f"⚠ {sym} order placement failed")

                    last_trade_time[sym] = datetime.utcnow()
                    await asyncio.sleep(0.5)

                except Exception as e:
                    log.error(f"Bot sym {sym} error: {e}")
                    await asyncio.sleep(1)

            await asyncio.sleep(float(os.getenv("SYMBOL_LOOP_DELAY", "2")))
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

    # start websocket feeder and tasks
    ws   = WS()
    ws_task   = asyncio.create_task(ws.run())
    bot_task  = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())

    log.info("✓ All tasks started")

    # deploy notification
    try:
        await send_telegram("🟢 <b>Bot Deployed & Online</b>\n"
                            f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
                            f"📊 Pairs: {', '.join(CFG['pairs'])}\n"
                            f"💰 Equity: ${CFG['equity']}\n"
                            f"⚙️ Min Score: {CFG['min_score']}\n"
                            "✅ Ready to trade!")
    except Exception as e:
        log.warning(f"Deploy telegram failed: {e}")

    try:
        yield
    finally:
        log.info("🔄 Shutting down...")
        for t in [ws_task, bot_task, ping_task]:
            if t:
                t.cancel()
        await asyncio.gather(ws_task, bot_task, ping_task, return_exceptions=True)
        await cleanup()
        log.info("✅ Shutdown complete")

# ---------- FastAPI ----------
app = FastAPI(lifespan=lifespan, title="ML-Trader", version="1.2")

@app.get("/")
def root():
    return {"status": "running", "bot": "ML-Trader v1.2", "time": datetime.utcnow().isoformat()}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "ws": bool(ws_task and not ws_task.done()),
        "bot": bool(bot_task and not bot_task.done()),
        "ping": bool(ping_task and not ping_task.done())
    }

@app.get("/stats")
async def stats():
    try:
        r = await redis()
        daily = float(await r.get("daily_pnl") or 0)
    except Exception:
        daily = 0.0
    return {"daily_pnl": daily, "equity": CFG["equity"], "pairs": CFG["pairs"],
            "min_score": CFG["min_score"], "min_rrr": CFG["min_rrr"]}

# debug endpoint (safe: only enabled when DEBUG env true)
@app.get("/debug")
async def debug(request: Request):
    if os.getenv("DEBUG", "false").lower() not in ("1", "true", "yes"):
        return {"error": "debug disabled"}
    try:
        r = await redis()
        info = {
            "cfg": {k: v for k, v in CFG.items() if k not in ("key", "secret", "openai_key")},
            "tasks": {
                "ws": bool(ws_task and not ws_task.done()),
                "bot": bool(bot_task and not bot_task.done()),
                "ping": bool(ping_task and not ping_task.done())
            },
            "time": datetime.utcnow().isoformat()
        }
        # small Redis sample
        sample = {}
        for p in CFG["pairs"][:5]:
            try:
                sample[p] = {
                    "ticker": await r.hgetall(f"t:{p}") if r else None,
                    "kline_1m_len": await r.llen(f"kline_1m:{p}") if r else None
                }
            except Exception:
                sample[p] = "redis-error"
        info["redis_sample"] = sample
        return info
    except Exception as e:
        return {"error": str(e)}

# ---------- run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))
