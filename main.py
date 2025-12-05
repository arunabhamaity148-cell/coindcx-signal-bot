import os, asyncio, logging, aiohttp
from datetime import datetime
from fastapi import FastAPI
import uvicorn

from helpers import (
    CFG, STRATEGY_CONFIG, Exchange,
    calculate_advanced_score, calc_tp_sl,
    iceberg_size, send_telegram, redis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")


# ===================================================================
# WS POLLING TASK – FETCH MARKET DATA FROM REDIS STREAM
# ===================================================================
async def ws_polling():
    log.info("🔌 WS polling started...")
    r = await redis()

    import ccxt.async_support as ccxt
    ex = ccxt.coindcx({"enableRateLimit": True})

    while True:
        try:
            for sym in CFG["pairs"]:
                try:
                    # ticker
                    ticker = await ex.fetch_ticker(sym)
                    await r.hset(f"t:{sym}", mapping={
                        "last": ticker["last"], 
                        "E": int(datetime.utcnow().timestamp()*1000)
                    })

                    # orderbook
                    ob = await ex.fetch_order_book(sym, limit=20)
                    await r.hset(f"d:{sym}", mapping={
                        "bids": json.dumps(ob["bids"]),
                        "asks": json.dumps(ob["asks"]),
                        "E": int(datetime.utcnow().timestamp()*1000),
                    })

                    # trades
                    trades = await ex.fetch_trades(sym, limit=100)
                    for t in trades:
                        await r.lpush(f"tr:{sym}", json.dumps({
                            "p": t["price"], 
                            "q": t["amount"],
                            "m": t["side"] == "sell",
                            "t": t["timestamp"]
                        }))
                    await r.ltrim(f"tr:{sym}", 0, 400)

                except Exception as e:
                    log.warning(f"poll error {sym}: {e}")
                    continue

            await asyncio.sleep(2)

        except Exception as e:
            log.error(f"WS loop crash: {e}")
            await asyncio.sleep(5)


# ===================================================================
# SIGNAL BOT LOOP
# ===================================================================
cooldown_cache = {}

def cooldown_ok(sym, strategy):
    key = f"{sym}:{strategy}"
    if key not in cooldown_cache: 
        return True
    t = (datetime.utcnow() - cooldown_cache[key]).total_seconds()/60
    return t >= CFG["cooldown_min"]

def set_cooldown(sym, strategy):
    cooldown_cache[f"{sym}:{strategy}"] = datetime.utcnow()


async def bot_loop():
    log.info("🤖 Signal bot started...")

    while True:
        try:
            for sym in CFG["pairs"]:
                for strategy in STRATEGY_CONFIG.keys():

                    if not cooldown_ok(sym, strategy):
                        continue

                    sig = await calculate_advanced_score(sym, strategy)
                    if not sig or sig["side"] == "none":
                        continue

                    side = sig["side"]
                    entry = sig["last"]
                    score = sig["score"]

                    tp1, tp2, sl, lev, liq, liq_dist = await calc_tp_sl(sym, side, entry, strategy)

                    # dangerous SL ?
                    if liq_dist < 0.7:
                        await send_telegram(
                            f"⚠️ <b>LIQ WARNING</b>\n{sym} {strategy} {side}\n"
                            f"Entry: {entry}\nSL-LIQ Distance: {liq_dist:.2f}% (too close)"
                        )
                        continue

                    # iceberg sizing
                    ice = iceberg_size(CFG["equity"], entry, sl, lev)

                    msg = (
                        f"🎯 <b>[{strategy}] {sym} {side.upper()}</b>\n"
                        f"Entry: <code>{entry:.6f}</code>\n"
                        f"TP1: <code>{tp1:.6f}</code>\n"
                        f"TP2: <code>{tp2:.6f}</code>\n"
                        f"SL: <code>{sl:.6f}</code>\n"
                        f"Leverage: <b>{lev}x</b>\n"
                        f"Score: {score:.2f}\n"
                        f"Liq Distance: {liq_dist:.2f}%\n\n"
                        f"💰 <b>Iceberg:</b>\n"
                        f"Total Qty: {ice['total']:.6f}\n"
                        f"Split: {ice['orders']} × {ice['each']:.6f}\n\n"
                        f"📝 Steps:\n"
                        f"1️⃣ Set leverage {lev}x\n"
                        f"2️⃣ Place {ice['orders']} limit orders at {entry:.6f}\n"
                        f"3️⃣ SL → {sl:.6f}\n"
                        f"4️⃣ TP1 → {tp1:.6f} ({STRATEGY_CONFIG[strategy]['tp1_exit']*100:.0f}% exit)\n"
                        f"5️⃣ TP2 → {tp2:.6f}\n"
                        f"⏳ Cooldown: {CFG['cooldown_min']} min"
                    )

                    await send_telegram(msg)
                    log.info(f"✔ Sent signal {sym} {strategy} {side}")

                    set_cooldown(sym, strategy)

            await asyncio.sleep(2)

        except Exception as e:
            log.error(f"BOT LOOP ERROR: {e}")
            await asyncio.sleep(5)



# ===================================================================
# KEEP ALIVE PING
# ===================================================================
async def keep_alive():
    url = "http://0.0.0.0:8080/health"
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url) as r:
                    if r.status == 200:
                        log.info("✓ Keep-alive OK")
        except:
            pass
        await asyncio.sleep(60)


# ===================================================================
# FASTAPI + LIFESPAN
# ===================================================================
from contextlib import asynccontextmanager

bot_task = None
ws_task = None
ping_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task, ws_task, ping_task
    log.info("🚀 Bot Booting...")

    ws_task = asyncio.create_task(ws_polling())
    bot_task = asyncio.create_task(bot_loop())
    ping_task = asyncio.create_task(keep_alive())

    yield

    for t in [ws_task, bot_task, ping_task]:
        if t: t.cancel()

    await asyncio.gather(*[t for t in [ws_task, bot_task, ping_task] if t], return_exceptions=True)
    log.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "status": "running",
        "pairs": len(CFG["pairs"]),
        "strategies": list(STRATEGY_CONFIG.keys()),
        "time": datetime.utcnow().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ws": ws_task is not None and not ws_task.done(),
        "bot": bot_task is not None and not bot_task.done(),
        "time": datetime.utcnow().isoformat()
    }


# ===================================================================
# UVICORN START
# ===================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))