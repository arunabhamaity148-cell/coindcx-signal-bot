# main.py — Final fixed (WS polling + Bot loop + Health)
import os
import json
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

# helpers (must be the big helpers.py you already added)
from helpers import (
    CFG,
    STRATEGY_CONFIG,
    redis,
    calculate_advanced_score,
    calc_tp_sl,
    iceberg_size,
    send_telegram,
    get_exchange
)

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ------------------------
# small in-process cooldown (fast)
# ------------------------
_cooldown = {}  # key -> datetime

def cooldown_ok(sym: str, strategy: str):
    key = f"{sym}:{strategy}"
    last = _cooldown.get(key)
    if not last:
        return True
    elapsed = (datetime.utcnow() - last).total_seconds() / 60.0
    return elapsed >= CFG.get("cooldown_min", 30)

def set_cooldown(sym: str, strategy: str):
    key = f"{sym}:{strategy}"
    _cooldown[key] = datetime.utcnow()

# ------------------------
# WS Polling (fixed)
# ------------------------
async def ws_polling():
    log.info("🔌 WS polling started...")
    r = await redis()

    # use helpers' get_exchange factory (so it picks up keys from ENV)
    ex = get_exchange({"enableRateLimit": True})  # if helpers.get_exchange wraps ccxt.coindcx
    # if get_exchange returned an instance factory-like, we ensure .fetch_* exist
    # ENSURE we don't crash here if exchange doesn't support some calls
    while True:
        try:
            for sym in CFG.get("pairs", []):
                try:
                    # safe ticker
                    try:
                        ticker = await ex.fetch_ticker(sym)
                        last = float(ticker.get("last", 0) or 0)
                        await r.hset(f"t:{sym}", mapping={"last": last, "E": int(datetime.utcnow().timestamp() * 1000)})
                    except Exception as e:
                        log.debug(f"ticker fetch {sym}: {e}")

                    # safe orderbook
                    try:
                        ob = await ex.fetch_order_book(sym, limit=20)
                        bids = ob.get("bids", []) if isinstance(ob, dict) else []
                        asks = ob.get("asks", []) if isinstance(ob, dict) else []
                        await r.hset(f"d:{sym}", mapping={"bids": json.dumps(bids[:20]), "asks": json.dumps(asks[:20]), "E": int(datetime.utcnow().timestamp() * 1000)})
                    except Exception as e:
                        log.debug(f"orderbook fetch {sym}: {e}")

                    # safe trades
                    try:
                        trades = await ex.fetch_trades(sym, limit=100)
                        if trades and len(trades) > 0:
                            pushed = 0
                            for t in trades[-100:]:
                                try:
                                    # normalize fields leniently
                                    p = t.get("price") if isinstance(t, dict) else getattr(t, "price", None)
                                    q = t.get("amount") if isinstance(t, dict) else getattr(t, "amount", None)
                                    side = t.get("side") if isinstance(t, dict) else getattr(t, "side", None)
                                    ts = None
                                    for k in ("timestamp", "time", "ts", "trade_timestamp"):
                                        if isinstance(t, dict) and k in t and t[k]:
                                            ts = t[k]; break
                                    if ts is None:
                                        ts = int(datetime.utcnow().timestamp() * 1000)
                                    p = float(p or 0.0); q = float(q or 0.0); is_sell = True if str(side).lower() == "sell" else False
                                    await r.lpush(f"tr:{sym}", json.dumps({"p": p, "q": q, "m": is_sell, "t": int(ts)}))
                                    pushed += 1
                                except Exception:
                                    continue
                            if pushed:
                                await r.ltrim(f"tr:{sym}", 0, 499)
                        else:
                            # no trades returned — that's okay, log debug
                            log.debug(f"no trades from exchange for {sym}")
                    except Exception as e:
                        log.debug(f"trades fetch {sym}: {e}")

                except Exception as inner:
                    log.debug(f"polling inner error {sym}: {inner}")
                    continue

            # throttle loop
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            log.info("WS polling cancelled")
            break
        except Exception as e:
            log.error(f"WS loop crash: {e}")
            await asyncio.sleep(3)

# ------------------------
# Bot loop
# ------------------------
async def bot_loop():
    log.info("🤖 Signal bot started...")
    # send deploy notice
    try:
        await send_telegram(f"🚀 <b>Bot deployed</b> — pairs: {len(CFG.get('pairs',[]))} — Mode: Aggressive (No ML)")
    except Exception as e:
        log.debug(f"deploy notify failed: {e}")

    await asyncio.sleep(2)

    while True:
        try:
            for sym in CFG.get("pairs", []):
                for strategy in STRATEGY_CONFIG.keys():
                    try:
                        if not cooldown_ok(sym, strategy):
                            log.debug(f"{sym} {strategy} cooldown active")
                            continue

                        # calculate advanced score (helpers)
                        sig = await calculate_advanced_score(sym, strategy)
                        if not sig:
                            log.debug(f"{sym} no-signal")
                            continue
                        if sig.get("side") == "none":
                            log.debug(f"{sym} blocked by score reason")
                            continue

                        side = sig.get("side")
                        last = float(sig.get("last", 0))
                        score = float(sig.get("score", 0))

                        # tp/sl calc
                        tp1, tp2, sl, lev, liq_price, liq_dist = await calc_tp_sl(sym, side, last, strategy)

                        # liq safety check
                        liq_alert_pct = float(os.getenv("LIQ_ALERT_PCT", 0.7))
                        if liq_dist is not None and liq_dist < liq_alert_pct:
                            log.warning(f"LIQ CLOSE {sym} {strategy} {side} — liq_dist={liq_dist:.2f}% -> skip")
                            await send_telegram(f"⚠️ <b>LIQ CLOSE:</b> {sym} {strategy} {side} | Entry={last} | LiqDist={liq_dist:.2f}%")
                            continue

                        # iceberg sizing
                        ice = iceberg_size(CFG.get("equity", 30000), last, sl, lev)
                        if not ice or ice.get("total", 0) <= 0:
                            log.info(f"{sym} blocked: tiny position")
                            continue

                        # final message
                        msg = (
                            f"🎯 <b>[{strategy}] {sym} {side.upper()}</b>\n"
                            f"Entry: <code>{last:.8f}</code>\n"
                            f"TP1: <code>{tp1:.8f}</code>\n"
                            f"TP2: <code>{tp2:.8f}</code>\n"
                            f"SL: <code>{sl:.8f}</code>\n"
                            f"Lev: <b>{lev}x</b>\n"
                            f"Score: {score:.2f}\n"
                            f"LiqPrice: <code>{liq_price:.8f}</code> | LiqDist: {liq_dist:.2f}%\n\n"
                            f"💰 Iceberg → Total: {ice['total']:.6f} | {ice['orders']}×{ice['each']:.6f}\n\n"
                            f"📝 Steps:\n1) Set Lev {lev}x\n2) Place {ice['orders']} limit orders @ {last:.8f}\n3) SL → {sl:.8f}\n4) Exit {STRATEGY_CONFIG[strategy]['tp1_mult']}× at TP1 {tp1:.8f}\n5) Full exit TP2 {tp2:.8f}\n"
                        )
                        await send_telegram(msg)
                        log.info(f"✔ SIGNAL SENT {sym} {strategy} {side} score={score:.2f}")

                        # set cooldown and small sleep
                        set_cooldown(sym, strategy)
                        await asyncio.sleep(0.6)

                    except Exception as inner_e:
                        log.error(f"symbol loop error {sym} {strategy}: {inner_e}")
                        continue

            await asyncio.sleep(1)
        except asyncio.CancelledError:
            log.info("Bot loop cancelled")
            break
        except Exception as e:
            log.error(f"Bot loop crash: {e}")
            await asyncio.sleep(5)

# ------------------------
# Keep-alive (internal hitting /health)
# ------------------------
async def keep_alive():
    port = int(os.getenv("PORT", "8080"))
    url = f"http://0.0.0.0:{port}/health"
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.debug("✓ Health ping OK")
        except Exception:
            pass
        await asyncio.sleep(60)

# ------------------------
# FastAPI app + lifespan
# ------------------------
app = FastAPI()
ws_task = None
bot_task = None
ping_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_task, bot_task, ping_task
    log.info("🚀 App starting — launching WS & Bot")
    ws_task = asyncio.create_task(ws_polling())
    # small delay to gather some trades first
    await asyncio.sleep(2)
    bot_task = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())
    try:
        yield
    finally:
        log.info("🔄 App shutting down — cancelling tasks")
        for t in [ws_task, bot_task, ping_task]:
            if t:
                t.cancel()
        await asyncio.gather(*[t for t in [ws_task, bot_task, ping_task] if t], return_exceptions=True)
        log.info("Shutdown done")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running", "pairs": len(CFG.get("pairs", [])), "time": datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "ws_running": ws_task is not None and not ws_task.done(),
        "bot_running": bot_task is not None and not bot_task.done()
    }

# ------------------------
# Run uvicorn
# ------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))