import os, json, asyncio, logging, websockets
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import uvicorn, aiohttp
from fastapi import FastAPI
from helpers import (
    CFG, STRATEGY_CONFIG, redis, filtered_pairs,
    calculate_advanced_score, calc_tp_sl, iceberg_size,
    send_telegram, get_exchange, log_block, DAILY_STATS, send_daily_summary
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- COOLDOWN ----------
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

# ---------- WEBSOCKET DATA FEED ----------
async def ws_feed():
    """CoinDCX WebSocket for real-time data"""
    log.info("🔌 WebSocket feed starting...")
    
    # CoinDCX WebSocket URL (adjust if needed)
    ws_url = "wss://stream.coindcx.com"
    
    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                # Subscribe to all pairs
                pairs = await filtered_pairs()
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": [f"{sym.lower()}@ticker" for sym in pairs[:20]],  # First 20 pairs
                    "id": 1
                }
                await ws.send(json.dumps(subscribe_msg))
                log.info(f"✓ Subscribed to {len(pairs[:20])} pairs")
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        # Parse ticker data
                        if "e" in data and data["e"] == "24hrTicker":
                            sym = data["s"].upper()
                            last = float(data["c"])
                            vol = float(data["q"])
                            
                            # Store in Redis with TTL
                            r = await redis()
                            await r.setex(
                                f"t:{sym}", 
                                3600,  # 1hr expiry
                                json.dumps({"last": last, "quoteVolume": vol, "E": int(datetime.utcnow().timestamp() * 1000)})
                            )
                            
                        # Parse trade data
                        elif "e" in data and data["e"] == "trade":
                            sym = data["s"].upper()
                            p = float(data["p"])
                            q = float(data["q"])
                            is_sell = data["m"]
                            ts = data["T"]
                            
                            r = await redis()
                            await r.lpush(
                                f"tr:{sym}", 
                                json.dumps({"p": p, "q": q, "m": is_sell, "t": int(ts)})
                            )
                            await r.expire(f"tr:{sym}", 1800)  # 30min expiry
                            await r.ltrim(f"tr:{sym}", 0, 499)
                            
                    except asyncio.TimeoutError:
                        log.warning("WebSocket timeout, sending ping...")
                        await ws.ping()
                        
        except asyncio.CancelledError:
            log.info("WebSocket feed cancelled")
            break
        except Exception as e:
            log.error(f"WebSocket error: {e}, reconnecting in 5s...")
            await asyncio.sleep(5)

# ---------- FALLBACK REST POLLING ----------
async def rest_fallback():
    """Fallback to REST if WebSocket fails"""
    log.info("🔄 REST fallback active")
    ex = get_exchange({"enableRateLimit": True})
    
    while True:
        try:
            pairs = await filtered_pairs()
            
            for sym in pairs[:20]:  # Limit to 20 to avoid rate limit
                try:
                    ticker = await ex.fetch_ticker(sym)
                    last = float(ticker.get("last", 0))
                    vol24 = float(ticker.get("quoteVolume", 0))
                    
                    r = await redis()
                    await r.setex(
                        f"t:{sym}", 
                        3600,
                        json.dumps({"last": last, "quoteVolume": vol24, "E": int(datetime.utcnow().timestamp() * 1000)})
                    )
                    
                    trades = await ex.fetch_trades(sym, limit=100)
                    if trades:
                        for t in trades[-100:]:
                            p = float(t.get("price", 0))
                            q = float(t.get("amount", 0))
                            is_sell = t.get("side", "") == "sell"
                            ts = t.get("timestamp") or int(datetime.utcnow().timestamp() * 1000)
                            
                            await r.lpush(f"tr:{sym}", json.dumps({"p": p, "q": q, "m": is_sell, "t": int(ts)}))
                        
                        await r.expire(f"tr:{sym}", 1800)
                        await r.ltrim(f"tr:{sym}", 0, 499)
                    
                except Exception as inner:
                    log.error(f"REST error {sym}: {inner}")
                    continue
            
            await asyncio.sleep(2)  # 2s interval to avoid rate limit
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"REST fallback error: {e}")
            await asyncio.sleep(5)
    
    try:
        await ex.close()
    except:
        pass

# ---------- BOT LOOP (TOP 5 SIGNALS ONLY) ----------
async def bot_loop():
    await send_telegram("🚀 <b>Premium Signal Bot LIVE (v8.5)</b>\n✅ Top 5 signals/hour | 60min cooldown")
    log.info("🤖 Bot loop started - waiting 10s for data...")
    await asyncio.sleep(10)
    
    scan_count = 0
    while True:
        try:
            scan_count += 1
            log.info(f"🔍 Scan #{scan_count} starting...")
            
            pairs = await filtered_pairs()
            all_signals = []
            
            # Collect all valid signals
            for sym in pairs:
                for strat in STRATEGY_CONFIG.keys():
                    if not cooldown_ok(sym, strat):
                        continue
                    
                    sig = await calculate_advanced_score(sym, strat)
                    if not sig or sig.get("side") == "none":
                        continue
                    
                    side = sig["side"]
                    last = sig["last"]
                    score = sig["score"]
                    done = sig.get("done", "")
                    
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
                        "score": score,
                        "last": last,
                        "tp1": tp1,
                        "tp2": tp2,
                        "sl": sl,
                        "lev": lev,
                        "liq_dist": liq_dist,
                        "done": done,
                        "iceberg": iceberg
                    })
            
            # Sort by score and take top 5
            all_signals.sort(key=lambda x: x["score"], reverse=True)
            top_signals = all_signals[:5]
            
            log.info(f"📊 Found {len(all_signals)} signals, sending top {len(top_signals)}")
            
            # Send top 5 signals
            for sig in top_signals:
                win_chance = min(65 + int(sig["score"] * 3), 85)
                
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
                    f"🔸 TP1: <code>{sig['tp1']:.8f}</code> (60%)\n"
                    f"🔸 TP2: <code>{sig['tp2']:.8f}</code> (40%)\n"
                    f"🛑 SL: <code>{sig['sl']:.8f}</code>\n"
                    f"⚡ Leverage: <b>{sig['lev']}x</b> (Liq-dist {sig['liq_dist']:.1f}%)\n"
                    f"📊 Score: {sig['score']:.1f} | Win: ~{win_chance}%\n"
                    f"✅ {sig['done']}\n"
                    f"💰 Iceberg: {sig['iceberg']['orders']}×{sig['iceberg']['each']:.6f}\n\n"
                    f"📌 Manual: Set {sig['lev']}x → Place orders @ {sig['last']:.8f}"
                )
                
                await send_telegram(msg)
                log.info(f"✔ SIGNAL #{scan_count}: {sig['sym']} {sig['strat']} score={sig['score']:.1f}")
                
                DAILY_STATS["signals_sent"] += 1
                set_cooldown(sig["sym"], sig["strat"])
                await asyncio.sleep(2)
            
            log.info(f"✓ Scan #{scan_count} complete")
            await asyncio.sleep(30)  # 30s scan interval (was 5s)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Bot loop error: {e}")
            await asyncio.sleep(10)

# ---------- TRAILING SL ----------
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
                        new_sl = entry * (1 - 0.001) if side == "long" else entry * (1 + 0.001)
                        ACTIVE_ORDERS[key]["sl"] = new_sl
                        await send_telegram(f"✅ TP1 hit {sym} {strat} → SL → entry")
                        OPEN_CNT[strat] = max(0, OPEN_CNT[strat] - 1)
                        DAILY_STATS["tp1_hits"] += 1
            
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Trailing error: {e}")
            await asyncio.sleep(60)

# ---------- DAILY SUMMARY TASK ----------
async def daily_summary_task():
    log.info("📊 Daily summary task started")
    while True:
        try:
            now = datetime.utcnow()
            next_report = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait_seconds = (next_report - now).total_seconds()
            
            log.info(f"⏰ Next daily report in {wait_seconds/3600:.1f}h")
            await asyncio.sleep(wait_seconds)
            
            await send_daily_summary()
            log.info("✅ Daily summary sent")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Daily summary error: {e}")
            await asyncio.sleep(3600)

# ---------- FASTAPI ----------
ws_task = bot_task = trail_task = summary_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_task, bot_task, trail_task, summary_task
    
    log.info("🚀 App starting (v8.5/10)")
    
    # Try WebSocket first, fallback to REST
    try:
        ws_task = asyncio.create_task(ws_feed())
        await asyncio.sleep(3)
        
        # Check if WebSocket connected
        if ws_task.done() or ws_task.cancelled():
            log.warning("⚠️ WebSocket failed, using REST fallback")
            ws_task = asyncio.create_task(rest_fallback())
    except Exception as e:
        log.warning(f"⚠️ WebSocket not available: {e}, using REST")
        ws_task = asyncio.create_task(rest_fallback())
    
    bot_task = asyncio.create_task(bot_loop())
    trail_task = asyncio.create_task(trailing_task())
    summary_task = asyncio.create_task(daily_summary_task())
    
    log.info("✅ All tasks started")
    
    try:
        yield
    finally:
        log.info("🔄 Shutting down...")
        for t in [ws_task, bot_task, trail_task, summary_task]:
            if t:
                t.cancel()
        await asyncio.gather(
            *[t for t in [ws_task, bot_task, trail_task, summary_task] if t], 
            return_exceptions=True
        )

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "running",
        "version": "8.5/10",
        "pairs": len(CFG.get("pairs", [])),
        "time": datetime.utcnow().isoformat(),
        "active_orders": len(ACTIVE_ORDERS),
        "open_positions": OPEN_CNT,
        "daily_stats": DAILY_STATS
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "ws_running": ws_task is not None and not ws_task.done(),
        "bot_running": bot_task is not None and not bot_task.done(),
        "trail_running": trail_task is not None and not trail_task.done(),
        "summary_running": summary_task is not None and not summary_task.done()
    }

@app.get("/stats")
def stats():
    return {
        "cooldown_map": {k: v.isoformat() for k, v in cooldown_map.items()},
        "active_orders": ACTIVE_ORDERS,
        "open_count": OPEN_CNT,
        "daily_stats": DAILY_STATS,
        "total_active": len(ACTIVE_ORDERS)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))