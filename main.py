import os, json, asyncio, logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uvicorn, aiohttp
from fastapi import FastAPI

from helpers import (
    CFG, STRATEGY_CONFIG, redis, filtered_pairs,
    calculate_advanced_score, calc_tp_sl, iceberg_size,
    send_telegram, get_exchange, DAILY_STATS, send_daily_summary
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

cooldown_map = {}
ACTIVE_ORDERS = {}
OPEN_CNT = {"QUICK": 0, "MID": 0, "TREND": 0}
MAX_CON = 3


def cooldown_ok(sym: str, strategy: str) -> bool:
    key = f"{sym}:{strategy}"
    last = cooldown_map.get(key)
    if not last:
        return True
    elapsed = (datetime.utcnow() - last).total_seconds() / 60
    return elapsed >= CFG["cooldown_min"]


def set_cooldown(sym: str, strategy: str):
    cooldown_map[f"{sym}:{strategy}"] = datetime.utcnow()


# ---------------------------------------------------
# 🔥 REST POLLING ONLY  (WebSocket OFF)
# ---------------------------------------------------
async def rest_fallback():
    log.info("🔄 REST fallback active (WS disabled)")
    ex = get_exchange(enableRateLimit=True)

    while True:
        try:
            pairs = await filtered_pairs()

            for sym in pairs[:20]:
                try:
                    ticker = await ex.fetch_ticker(sym)
                    last = float(ticker.get("last", 0))
                    vol24 = float(ticker.get("quoteVolume", 0))

                    r = await redis()
                    await r.setex(
                        f"t:{sym}",
                        3600,
                        json.dumps({
                            "last": last,
                            "quoteVolume": vol24,
                            "E": int(datetime.utcnow().timestamp() * 1000)
                        })
                    )

                    trades = await ex.fetch_trades(sym, limit=80)
                    if trades:
                        for t in trades[-80:]:
                            p = float(t.get("price", 0))
                            q = float(t.get("amount", 0))

                            # FIXED TIMESTAMP ISSUE
                            ts = t.get("timestamp") or int(datetime.utcnow().timestamp() * 1000)
                            is_sell = t.get("side", "") == "sell"

                            await r.lpush(
                                f"tr:{sym}",
                                json.dumps({"p": p, "q": q, "m": is_sell, "t": int(ts)})
                            )
                        await r.expire(f"tr:{sym}", 1800)
                        await r.ltrim(f"tr:{sym}", 0, 499)

                except Exception as inner:
                    log.error(f"REST error {sym}: {inner}")
                    continue

            await asyncio.sleep(2)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"REST fallback error: {e}")
            await asyncio.sleep(5)

    try:
        await ex.close()
    except:
        pass


# ---------------------------------------------------
# BOT LOOP
# ---------------------------------------------------
async def bot_loop():
    await send_telegram("🚀 <b>Premium Signal Bot LIVE (REST Mode)</b>")
    log.info("🤖 Bot loop started - waiting 10s for data...")
    await asyncio.sleep(10)

    scan_count = 0
    while True:
        try:
            scan_count += 1
            log.info(f"🔍 Scan #{scan_count} starting...")

            pairs = await filtered_pairs()
            all_signals = []

            for sym in pairs:
                for strat in STRATEGY_CONFIG.keys():

                    if not cooldown_ok(sym, strat):
                        continue

                    sig = await calculate_advanced_score(sym, strat)
                    if not sig or sig.get("side") == "none":
                        continue

                    side = sig["side"]
                    last = sig["last"]

                    tp1, tp2, sl, lev, liq, liq_dist = await calc_tp_sl(sym, side, last, strat)

                    if OPEN_CNT[strat] >= MAX_CON:
                        continue

                    iceberg = iceberg_size(CFG["equity"], last, sl, lev)
                    if iceberg["total"] <= 0:
                        continue

                    all_signals.append({
                        "sym": sym,
                        "strat": strat,
                        "side": side,
                        "score": sig["score"],
                        "last": last,
                        "tp1": tp1,
                        "tp2": tp2,
                        "sl": sl,
                        "lev": lev,
                        "liq_dist": liq_dist,
                        "done": sig.get("done", ""),
                        "iceberg": iceberg
                    })

            all_signals.sort(key=lambda x: x["score"], reverse=True)
            top_signals = all_signals[:5]

            log.info(f"📊 Found {len(all_signals)} signals, sending top {len(top_signals)}")

            for sig in top_signals:
                OPEN_CNT[sig["strat"]] += 1
                ACTIVE_ORDERS[f"{sig['sym']}:{sig['strat']}"] = {
                    "entry": sig["last"],
                    "sl": sig["sl"],
                    "side": sig["side"],
                    "tp1_hit": False
                }

                msg = (
                    f"🎯 <b>[{sig['strat']}] {sig['sym']} {sig['side'].upper()}</b>\n"
                    f"💠 Entry: <code>{sig['last']:.8f}</code>\n"
                    f"🔸 TP1: <code>{sig['tp1']:.8f}</code>\n"
                    f"🔸 TP2: <code>{sig['tp2']:.8f}</code>\n"
                    f"🛑 SL: <code>{sig['sl']:.8f}</code>\n"
                    f"⚡ Lev: <b>{sig['lev']}x</b>\n"
                    f"📊 Score: {sig['score']:.2f}\n"
                    f"✅ {sig['done']}\n"
                )

                await send_telegram(msg)
                set_cooldown(sig["sym"], sig["strat"])
                DAILY_STATS["signals_sent"] += 1
                await asyncio.sleep(2)

            log.info(f"✓ Scan #{scan_count} complete")
            await asyncio.sleep(30)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Bot loop error: {e}")
            await asyncio.sleep(10)


# ---------------------------------------------------
# TRAILING SL
# ---------------------------------------------------
async def trailing_task():
    log.info("🎯 Trailing SL task started")

    while True:
        try:
            for key, data in list(ACTIVE_ORDERS.items()):
                sym, strat = key.split(":")
                side = data["side"]
                entry = data["entry"]
                tp1_hit = data["tp1_hit"]

                r = await redis()
                ticker_str = await r.get(f"t:{sym}")
                if not ticker_str:
                    continue

                ticker = json.loads(ticker_str)
                last = float(ticker.get("last", 0))

                tp1, *_ = await calc_tp_sl(sym, side, entry, strat)

                if not tp1_hit:
                    if (side == "long" and last >= tp1) or (side == "short" and last <= tp1):
                        ACTIVE_ORDERS[key]["tp1_hit"] = True
                        ACTIVE_ORDERS[key]["sl"] = entry
                        await send_telegram(f"✅ TP1 hit → SL moved to entry ({sym} {strat})")
                        OPEN_CNT[strat] = max(0, OPEN_CNT[strat] - 1)
                        DAILY_STATS["tp1_hits"] += 1

            await asyncio.sleep(60)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Trailing error: {e}")
            await asyncio.sleep(60)


# ---------------------------------------------------
# DAILY SUMMARY
# ---------------------------------------------------
async def daily_summary_task():
    log.info("📊 Daily summary task started")

    while True:
        try:
            now = datetime.utcnow()
            next_report = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            wait_seconds = (next_report - now).total_seconds()

            await asyncio.sleep(wait_seconds)
            await send_daily_summary()

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Daily summary error: {e}")
            await asyncio.sleep(3600)


# ---------------------------------------------------
# FASTAPI + LIFESPAN
# ---------------------------------------------------
rest_task = bot_task = trail_task = summary_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rest_task, bot_task, trail_task, summary_task

    log.info("🚀 App starting (REST ONLY mode)")

    rest_task = asyncio.create_task(rest_fallback())
    bot_task = asyncio.create_task(bot_loop())
    trail_task = asyncio.create_task(trailing_task())
    summary_task = asyncio.create_task(daily_summary_task())

    log.info("✅ All tasks started")

    try:
        yield
    finally:
        for t in [rest_task, bot_task, trail_task, summary_task]:
            if t:
                t.cancel()
        await asyncio.gather(
            *[t for t in [rest_task, bot_task, trail_task, summary_task] if t],
            return_exceptions=True
        )


app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running", "mode": "REST ONLY", "time": datetime.utcnow()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))