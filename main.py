# main.py
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import aiohttp

from helpers import (
    WS, Exchange, calculate_advanced_score, ai_review_ensemble,
    calc_tp1_tp2_sl_liq, position_size_iceberg, send_telegram,
    check_cooldown, set_cooldown, check_daily_signal_limit,
    increment_signal_count, cleanup, CFG, STRATEGY_CONFIG
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

bot_task = None
ping_task = None
ws_task = None
deploy_notify_task = None

async def keep_alive():
    port = int(os.getenv("PORT", 8080))
    url = f"http://0.0.0.0:{port}/health"
    while True:
        try:
            await asyncio.sleep(60)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        log.info("✓ Health ping OK")
        except Exception as e:
            log.error(f"Keep-alive: {e}")

# ----------------- bot_loop (improved with block-logs) -----------------
async def bot_loop():
    log.info("🤖 Starting CoinDCX signal generator...")
    try:
        ex = Exchange()
        signal_count = 0
        await asyncio.sleep(8)
        log.info(f"✓ Bot initialized - Monitoring {len(CFG['pairs'])} pairs")
        while True:
            try:
                # daily limit guard
                limit_ok, limit_reason = await check_daily_signal_limit()
                if not limit_ok:
                    log.warning(f"⚠️ Daily limit reached: {limit_reason} - sleeping 1h")
                    await asyncio.sleep(3600)
                    continue

                for sym in CFG["pairs"]:
                    try:
                        for strategy in ["QUICK", "MID", "TREND"]:
                            try:
                                # cooldown check
                                cooldown_ok, cd_reason = await check_cooldown(sym, strategy)
                                if not cooldown_ok:
                                    log.debug(f"⏳ Cooldown active [{strategy}] {sym}: {cd_reason}")
                                    continue

                                # calculate heuristic signal
                                signal = await calculate_advanced_score(sym, strategy)
                                if not signal:
                                    log.debug(f"🚫 No heuristic signal data for {sym} [{strategy}] (signal None)")
                                    continue
                                if signal.get("side") == "none":
                                    log.debug(f"⛔ Heuristic returned none for {sym} [{strategy}] (score={signal.get('score')})")
                                    continue

                                side = signal["side"]
                                score = signal["score"]
                                last = signal["last"]

                                # AI / ML review
                                ai = await ai_review_ensemble(sym, side, score, strategy)
                                if not ai.get("allow", False):
                                    reason = ai.get("reason", "blocked")
                                    conf = ai.get("confidence", 0)
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — Reason: {reason} | Conf={conf}% | Score={score:.2f}")
                                    continue
                                confidence = ai.get("confidence", 0)

                                # TP/SL calc
                                tp1, tp2, sl, leverage, liq_price = await calc_tp1_tp2_sl_liq(sym, side, last, confidence, strategy)

                                # risk / RRR checks
                                risk = abs(last - sl)
                                if risk == 0:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — zero risk (entry==sl?)")
                                    continue
                                reward1 = abs(tp1 - last)
                                reward2 = abs(tp2 - last)
                                rrr1 = reward1 / risk if risk > 0 else 0
                                rrr2 = reward2 / risk if risk > 0 else 0
                                if rrr1 < 1.0:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — RRR too low (TP1_RRR={rrr1:.2f}) | Score={score:.2f} | Conf={confidence}%")
                                    continue

                                # liquidity / flow check (again for display + extra guard)
                                from helpers import orderflow_analysis
                                flow = await orderflow_analysis(sym)
                                if not flow:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — no orderflow data")
                                    continue
                                if flow.get("spread", 999) > 0.25:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — spread too wide ({flow.get('spread'):.3f}%)")
                                    continue
                                if flow.get("depth_usd", 0) < 3e5:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — low liquidity ({flow.get('depth_usd'):.0f}$)")
                                    continue

                                # position sizing
                                pos_info = position_size_iceberg(CFG["equity"], last, sl, leverage)
                                total_qty = pos_info.get("total_qty", 0)
                                iceberg_qty = pos_info.get("iceberg_qty", 0)
                                num_orders = pos_info.get("num_orders", 0)
                                if total_qty < 0.001:
                                    log.info(f"🚫 BLOCKED [{strategy}] {sym} — position too small ({total_qty})")
                                    continue

                                # all checks passed -> emit signal
                                signal_count += 1
                                await increment_signal_count()
                                log.info(f"✅ SIGNAL [{strategy}] {sym} {side.upper()} #{signal_count} | Entry={last:.6f} | TP1={tp1:.6f} | SL={sl:.6f} | Lev={leverage}x | Conf={confidence}% | Score={score:.2f}")
                                sl_to_liq = abs(sl - liq_price) / (liq_price if liq_price else 1) * 100
                                tp1_exit_pct = STRATEGY_CONFIG[strategy]["tp1_exit"] * 100

                                # prepare telegram message
                                try:
                                    flow_display = flow or {}
                                    spread_txt = f"{flow_display.get('spread',0):.2f}%" if flow_display else "N/A"
                                    depth_txt = f"{(flow_display.get('depth_usd',0)/1e3):.0f}k" if flow_display else "N/A"
                                    msg = (
                                        f"🎯 <b>[{strategy}] {sym} {side.upper()} #{signal_count}</b>\n"
                                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                        f"📊 <b>Entry:</b> <code>{last:,.2f}</code>\n\n"
                                        f"🎯 <b>TP1:</b> <code>{tp1:,.2f}</code> ({tp1_exit_pct:.0f}% exit)\n"
                                        f"🎯 <b>TP2:</b> <code>{tp2:,.2f}</code>\n"
                                        f"🛑 <b>SL:</b> <code>{sl:,.2f}</code>\n"
                                        f"⚠️ <b>Liquidation:</b> <code>{liq_price:,.2f}</code>\n\n"
                                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                        f"✅ <b>Spread:</b> {spread_txt} | <b>Depth:</b> {depth_txt} $\n"
                                        f"💰 <b>Position Size:</b>\n"
                                        f"   Total: <b>{total_qty:.6f} {sym.replace('USDT','')}</b>\n"
                                        f"   Iceberg: <b>{num_orders} orders × {iceberg_qty:.6f}</b>\n\n"
                                        f"⚡ <b>Leverage:</b> {leverage}x\n"
                                        f"📈 <b>Score:</b> {score:.2f}/10\n"
                                        f"🤖 <b>ML Confidence:</b> {confidence}%\n"
                                        f"📊 <b>RRR:</b> TP1={rrr1:.2f} | TP2={rrr2:.2f}\n"
                                        f"✅ <b>SL Buffer:</b> {sl_to_liq:.1f}% from liq\n\n"
                                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                        f"📋 <b>Manual Steps:</b>\n"
                                        f"1️⃣ Set leverage: <b>{leverage}x</b>\n"
                                        f"2️⃣ Place <b>{num_orders}</b> limit orders @ <code>{last:,.2f}</code>\n"
                                        f"   (Each: {iceberg_qty:.6f} {sym.replace('USDT','')})\n"
                                        f"3️⃣ Set SL @ <code>{sl:,.2f}</code>\n"
                                        f"4️⃣ Exit <b>{tp1_exit_pct:.0f}%</b> @ TP1: <code>{tp1:,.2f}</code>\n"
                                        f"5️⃣ Exit remaining @ TP2: <code>{tp2:,.2f}</code>\n\n"
                                        f"⏰ <b>Strategy:</b> {strategy} scalp\n"
                                        f"🕐 <b>Signal time:</b> {datetime.utcnow().strftime('%H:%M UTC')}\n"
                                        f"🔄 <b>Cooldown:</b> {CFG['cooldown_min']} min"
                                    )
                                    await send_telegram(msg)
                                    log.info(f"📨 Telegram sent for {sym} [{strategy}] #{signal_count}")
                                except Exception as e:
                                    log.error(f"Telegram send error for {sym} [{strategy}]: {e}")

                                # set cooldown and break this strategy loop for the symbol
                                await set_cooldown(sym, strategy)
                                break

                            except Exception as e:
                                log.error(f"Strategy {strategy} error {sym}: {e}")
                                continue

                    except Exception as e:
                        log.error(f"Symbol processing {sym}: {e}")
                        continue

                # short rest between full pair scans
                await asyncio.sleep(3)

            except Exception as e:
                log.error(f"Bot loop iteration: {e}")
                await asyncio.sleep(10)

    except asyncio.CancelledError:
        log.info("🛑 Bot cancelled")
        try:
            await ex.close()
        except Exception:
            pass
        raise
    except Exception as e:
        log.exception(f"💥 Bot crashed: {e}")
        raise

# ----------------- deploy notify helper -----------------
async def send_deploy_telegram():
    # small delay so services fully initialize
    await asyncio.sleep(5)
    try:
        pairs_n = len(CFG.get("pairs", []))
        msg = (
            f"🚀 <b>Bot Deployed</b>\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Pairs monitored: {pairs_n}\n"
            f"Mode: MANUAL_TRADING\n"
            f"Environment: {os.getenv('RAILWAY_ENVIRONMENT','production')}\n"
        )
        await send_telegram(msg)
        log.info("📨 Deploy Telegram sent.")
    except Exception as e:
        log.error(f"Deploy Telegram error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot_task, ping_task, ws_task, deploy_notify_task
    log.info("🚀 Starting CoinDCX Signal Bot...")
    ws = WS()
    ws_task = asyncio.create_task(ws.run())
    log.info("✓ WS task started")
    bot_task = asyncio.create_task(bot_loop())
    log.info("✓ Bot task started")
    ping_task = asyncio.create_task(keep_alive())
    log.info("✓ Keep-alive started")

    # start background deploy-notify (sends telegram once after startup)
    try:
        deploy_notify_task = asyncio.create_task(send_deploy_telegram())
    except Exception as e:
        log.error(f"Deploy notify task error: {e}")

    try:
        yield
    finally:
        log.info("🔄 Shutting down...")
        for task in [ws_task, bot_task, ping_task, deploy_notify_task]:
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in [ws_task, bot_task, ping_task, deploy_notify_task] if t), return_exceptions=True)
        await cleanup()
        log.info("✅ Shutdown complete")

app = FastAPI(lifespan=lifespan, title="CoinDCX Signal Bot", version="2.0")

@app.get("/")
def root():
    return {
        "status": "running",
        "bot": "CoinDCX Manual Trading Signal Generator",
        "version": "2.0",
        "mode": "MANUAL_TRADING",
        "time": datetime.utcnow().isoformat(),
        "pairs": len(CFG["pairs"]),
        "strategies": ["QUICK", "MID", "TREND"],
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "bot": "running",
        "time": datetime.utcnow().isoformat(),
        "ws_running": ws_task and not ws_task.done() if ws_task else False,
        "bot_running": bot_task and not bot_task.done() if bot_task else False
    }

@app.get("/stats")
async def stats():
    from helpers import redis
    r = await redis()
    signal_count = await r.get("daily_signal_count") or 0
    return {
        "daily_signals": int(signal_count),
        "max_daily": 30,
        "equity": CFG["equity"],
        "pairs": len(CFG["pairs"]),
        "leverage_range": f"{CFG['min_lev']}-{CFG['max_lev']}x",
        "cooldown_minutes": CFG['cooldown_min'],
        "strategies": STRATEGY_CONFIG,
        "mode": "MANUAL_TRADING"
    }

@app.get("/config")
def config():
    return {
        "strategies": {
            "QUICK": {
                "score_range": f"{STRATEGY_CONFIG['QUICK']['min_score']}-{STRATEGY_CONFIG['QUICK']['max_score']}",
                "tp1_mult": STRATEGY_CONFIG['QUICK']['tp1_mult'],
                "tp2_mult": STRATEGY_CONFIG['QUICK']['tp2_mult'],
                "tp1_exit": f"{STRATEGY_CONFIG['QUICK']['tp1_exit']*100}%",
                "min_confidence": f"{STRATEGY_CONFIG['QUICK']['min_conf']}%"
            },
            "MID": {
                "score_range": f"{STRATEGY_CONFIG['MID']['min_score']}-{STRATEGY_CONFIG['MID']['max_score']}",
                "tp1_mult": STRATEGY_CONFIG['MID']['tp1_mult'],
                "tp2_mult': STRATEGY_CONFIG['MID']['tp2_mult'],
                "tp1_exit": f"{STRATEGY_CONFIG['MID']['tp1_exit']*100}%",
                "min_confidence": f"{STRATEGY_CONFIG['MID']['min_conf']}%"
            },
            "TREND": {
                "score_range": f"{STRATEGY_CONFIG['TREND']['min_score']}-{STRATEGY_CONFIG['TREND']['max_score']}",
                "tp1_mult": STRATEGY_CONFIG['TREND']['tp1_mult'],
                "tp2_mult": STRATEGY_CONFIG['TREND']['tp2_mult'],
                "tp1_exit": f"{STRATEGY_CONFIG['TREND']['tp1_exit']*100}%",
                "min_confidence": f"{STRATEGY_CONFIG['TREND']['min_conf']}%"
            }
        },
        "risk_management": {
            "equity": CFG["equity"],
            "risk_per_trade": f"{CFG['risk_perc']}%",
            "leverage_range": f"{CFG['min_lev']}-{CFG['max_lev']}x",
            "liquidation_buffer": f"{CFG['liq_buffer']*100}%",
            "max_daily_signals": 30,
            "cooldown": f"{CFG['cooldown_min']} minutes"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")