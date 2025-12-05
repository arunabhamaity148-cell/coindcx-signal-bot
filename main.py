import os, json, asyncio, logging
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn, aiohttp
from fastapi import FastAPI
from helpers import (
    CFG, STRATEGY_CONFIG, redis, filtered_pairs,
    calculate_advanced_score, calc_tp_sl, iceberg_size,
    send_telegram, get_exchange
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# --------------------  COOLDOWN  --------------------
cooldown_map = {}
ACTIVE_ORDERS = {}

def cooldown_ok(sym: str, strategy: str) -> bool:
    key = f"{sym}:{strategy}"
    last = cooldown_map.get(key)
    if not last: return True
    elapsed = (datetime.utcnow() - last).total_seconds() / 60
    return elapsed >= CFG["cooldown_min"]

def set_cooldown(sym: str, strategy: str):
    cooldown_map[f"{sym}:{strategy}"] = datetime.utcnow()

# --------------------  WS POLLING  --------------------
async def ws_polling():
    log.info("🔌 WS polling started (CoinDCX REST 1s)")
    ex = get_exchange({"enableRateLimit": True})
    while True:
        try:
            pairs = await filtered_pairs()
            for sym in pairs:
                try:
                    ticker = await ex.fetch_ticker(sym)
                    last = float(ticker.get("last", 0))
                    vol24 = float(ticker.get("quoteVolume", 0))
                    r = await redis()
                    await r.hset(f"t:{sym}", mapping={"last": last, "quoteVolume": vol24, "E": int(datetime.utcnow().timestamp() * 1000)})
                    ob = await ex.fetch_order_book(sym, limit=20)
                    bids = ob.get("bids", []); asks = ob.get("asks", [])
                    await r.hset(f"d:{sym}", mapping={"bids": json.dumps(bids[:20]), "asks": json.dumps(asks[:20]), "E": int(datetime.utcnow().timestamp() * 1000)})
                    trades = await ex.fetch_trades(sym, limit=100)
                    if trades:
                        pushed = 0
                        for t in trades[-100:]:
                            p = float(t.get("price", 0)); q = float(t.get("amount", 0))
                            side_t = t.get("side", ""); is_sell = side_t == "sell"
                            ts = t.get("timestamp") or int(datetime.utcnow().timestamp() * 1000)
                            await r.lpush(f"tr:{sym}", json.dumps({"p": p, "q": q, "m": is_sell, "t": int(ts)}))
                            pushed += 1
                        if pushed: await r.ltrim(f"tr:{sym}", 0, 499)
                except Exception as inner:
                    log.debug(f"polling inner {sym}: {inner}")
                    continue
            await asyncio.sleep(1)
        except asyncio.CancelledError: break
        except Exception as e:
            log.error(f"WS loop crash: {e}")
            await asyncio.sleep(3)

# --------------------  BOT LOOP  --------------------
async def bot_loop():
    await send_telegram("🚀 <b>Bot LIVE</b> – CoinDCX real data – News-Guard ON – max 30 pair")
    await asyncio.sleep(3)
    while True:
        try:
            pairs = await filtered_pairs()
            for sym in pairs:
                for strat in STRATEGY_CONFIG.keys():
                    if not cooldown_ok(sym, strat): continue
                    sig = await calculate_advanced_score(sym, strat)
                    if not sig or sig["side"] == "none": continue
                    side, last, score = sig["side"], sig["last"], sig["score"]
                    tp1, tp2, sl, lev, liq, liq_dist = await calc_tp_sl(sym, side, last, strat)
                    if liq_dist < 0.7:
                        await send_telegram(f"⚠️ <b>LiqClose</b> {sym} {strat} {side} dist={liq_dist:.2f}%")
                        continue
                    ice = iceberg_size(CFG["equity"], last, sl, lev)
                    if ice["total"] <= 0: continue
                    ACTIVE_ORDERS[f"{sym}:{strat}"] = {"entry": last, "sl": sl, "side": side, "tp1_hit": False}
                    msg = (f"🎯 <b>[{strat}] {sym} {side.upper()}</b>\n"
                           f"Entry: <code>{last:.8f}</code>\n"
                           f"TP1: <code>{tp1:.8f}</code>\n"
                           f"TP2: <code>{tp2:.8f}</code>\n"
                           f"SL: <code>{sl:.8f}</code>\n"
                           f"Leverage: <b>{lev}x</b>\n"
                           f"Score: {score:.2f}\n"
                           f"LiqPrice: <code>{liq:.8f}</code>\n"
                           f"LiqDist: {liq_dist:.2f}%\n"
                           f"💰 Iceberg: {ice['orders']}×{ice['each']:.6f}\n\n"
                           f"📌 Steps:\n1) Set Lev {lev}x\n2) Place {ice['orders']} limit @ {last:.8f}\n3) SL {sl:.8f}\n4) Exit {STRATEGY_CONFIG[strat]['tp1_exit']}× at TP1\n5) Full exit TP2")
                    await send_telegram(msg)
                    log.info(f"✔ SIGNAL SENT {sym} {strat} {side} score={score:.2f}")
                    set_cooldown(sym, strat)
                    await asyncio.sleep(0.6)
            await asyncio.sleep(1)
        except asyncio.CancelledError: break
        except Exception as e:
            log.error(f"bot-loop crash: {e}")
            await asyncio.sleep(5)

# --------------------  TRAILING SL  --------------------
async def trailing_task():
    while True:
        try:
            for key, data in list(ACTIVE_ORDERS.items()):
                sym, strat = key.split(":")
                side = data["side"]; entry = data["entry"]; tp1_hit = data["tp1_hit"]
                r = await redis(); last = float(await r.hget(f"t:{sym}", "last") or 0)
                if not last: continue
                tp1, *_ = await calc_tp_sl(sym, side, entry, strat)
                if not tp1_hit:
                    if (side == "long" and last >= tp1) or (side == "short" and last <= tp1):
                        ACTIVE_ORDERS[key]["tp1_hit"] = True; new_sl = entry * (1 - 0.001) if side == "long" else entry * (1 + 0.001)
                        ACTIVE_ORDERS[key]["sl"] = new_sl
                        await send_telegram(f"✅ TP1 hit {sym} {strat} → SL trailed to entry")
            await asyncio.sleep(60)
        except: await asyncio.sleep(60)

# --------------------  KEEP-ALIVE  --------------------
async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url = f"http://0.0.0.0:{port}/health"
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200: log.debug("✓ Health ping OK")
        except: pass
        await asyncio.sleep(60)

# --------------------  FASTAPI  --------------------
ws_task = bot_task = trail_task = ping_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_task, bot_task, trail_task, ping_task
    log.info("🚀 App starting – launching WS & Bot")
    ws_task   = asyncio.create_task(ws_polling())
    await asyncio.sleep(2)
    bot_task  = asyncio.create_task(bot_loop())
    trail_task = asyncio.create_task(trailing_task())
    ping_task = asyncio.create_task(keep_alive())
    try: yield
    finally:
        log.info("🔄 App shutting down – cancelling tasks")
        for t in [ws_task, bot_task, trail_task, ping_task]:
            if t: t.cancel()
        await asyncio.gather(*[t for t in [ws_task, bot_task, trail_task, ping_task] if t], return_exceptions=True)
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

# --------------------  RUN  --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
