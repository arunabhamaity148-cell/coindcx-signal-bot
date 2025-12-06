import os
import json
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from collections import deque
from contextlib import asynccontextmanager

import aiohttp
import websockets
from fastapi import FastAPI, Response
import uvicorn

from helpers import PAIRS
from scorer import compute_signal
from telegram_formatter import TelegramFormatter
from position_tracker import PositionTracker
from signal_confidence import SignalHistory
from chart_image import generate_chart_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("main")

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 25))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 30))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 15))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 60.0))
SEND_CHARTS = os.getenv("SEND_CHARTS", "true").lower() == "true"

TRADE_BUFFER = {sym: deque(maxlen=500) for sym in PAIRS}
OB_CACHE = {}
TK_CACHE = {}
cooldown = {}
ACTIVE_ORDERS = {}
data_received = 0
ws_connected = False
app_ready = False
shutdown_requested = False

formatter = TelegramFormatter()
pos_tracker = PositionTracker()
signal_history = SignalHistory(max_history=200)

def handle_shutdown(signum, frame):
    global shutdown_requested
    log.info(f"⚠️ Received signal {signum}, shutting down...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

async def send_telegram(message: str, chart_url: str = None, retry: int = 3):
    if not TG_TOKEN or not TG_CHAT:
        return False
    url_base = f"https://api.telegram.org/bot{TG_TOKEN}"
    if chart_url and SEND_CHARTS:
        url = f"{url_base}/sendPhoto"
        payload = {"chat_id": TG_CHAT, "photo": chart_url, "caption": message, "parse_mode": "HTML"}
    else:
        url = f"{url_base}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(retry):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return True
        except Exception as e:
            log.warning(f"Telegram error: {e}")
        if attempt < retry - 1:
            await asyncio.sleep(2 ** attempt)
    return False

async def push_trade(sym: str, data: dict):
    global data_received
    try:
        if not all(k in data for k in ("p", "q", "m", "t")):
            return
        trade = {
            "p": float(data["p"]),
            "q": float(data["q"]),
            "m": bool(data["m"]),
            "t": int(data["t"])
        }
        TRADE_BUFFER[sym].append(trade)
        data_received += 1
    except Exception as e:
        log.error(f"push_trade error for {sym}: {e}")

async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        OB_CACHE[sym] = {
            "bid": float(bid),
            "ask": float(ask),
            "t": int(datetime.now(timezone.utc).timestamp() * 1000)
        }
    except Exception as e:
        log.error(f"push_orderbook error: {e}")

async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        TK_CACHE[sym] = {
            "last": float(last),
            "vol": float(vol),
            "t": int(ts)
        }
    except Exception as e:
        log.error(f"push_ticker error: {e}")

WS_BASE = "wss://stream.binance.com:9443/stream"

def build_stream_url(pairs_chunk: list) -> str:
    streams = []
    for pair in pairs_chunk:
        p_lower = pair.lower()
        streams.append(f"{p_lower}@aggTrade")
        streams.append(f"{p_lower}@bookTicker")
    if "BTCUSDT" in pairs_chunk:
        streams.append("btcusdt@kline_1m")
    return f"{WS_BASE}?streams={'/'.join(streams)}"

async def ws_worker(pairs_chunk: list, worker_id: int):
    global ws_connected
    url = build_stream_url(pairs_chunk)
    backoff = 1
    log.info(f"WS-{worker_id} starting ({len(pairs_chunk)} pairs)")
    while not shutdown_requested:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=10) as ws:
                backoff = 1
                ws_connected = True
                log.info(f"✅ WS-{worker_id} connected")
                async for message in ws:
                    if shutdown_requested:
                        break
                    try:
                        data = json.loads(message)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        sym = payload.get("s")
                        if not sym or sym not in PAIRS:
                            continue
                        if stream.endswith("@aggTrade"):
                            await push_trade(sym, {
                                "p": payload.get("p"),
                                "q": payload.get("q"),
                                "m": payload.get("m"),
                                "t": payload.get("T")
                            })
                        elif stream.endswith("@bookTicker"):
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            if bid > 0 and ask > 0:
                                await push_orderbook(sym, bid, ask)
                        elif "@kline" in stream:
                            k = payload.get("k", {})
                            if k.get("x"):
                                close = float(k.get("c", 0))
                                vol = float(k.get("q", 0))
                                ts = int(k.get("T", 0))
                                if close > 0:
                                    await push_ticker(sym, close, vol, ts)
                    except Exception as e:
                        log.warning(f"WS-{worker_id} message error: {e}")
        except Exception as e:
            ws_connected = False
            log.warning(f"⚠️ WS-{worker_id} disconnected: {e}. Reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
    log.info(f"WS-{worker_id} stopped")

async def start_websockets():
    pairs = list(PAIRS)
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    tasks = [asyncio.create_task(ws_worker(chunk, idx + 1)) for idx, chunk in enumerate(chunks)]
    log.info(f"🔌 Started {len(tasks)} WebSocket workers")
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()

def cooldown_ok(sym: str, strat: str) -> bool:
    key = f"{sym}:{strat}"
    if key not in cooldown:
        return True
    elapsed_min = (datetime.now(timezone.utc) - cooldown[key]).total_seconds() / 60
    return elapsed_min >= COOLDOWN_MIN

def set_cooldown(sym: str, strat: str):
    cooldown[f"{sym}:{strat}"] = datetime.now(timezone.utc)

async def update_positions():
    if not pos_tracker.open_positions:
        return
    price_dict = {sym: TK_CACHE[sym]["last"] for sym in PAIRS if sym in TK_CACHE}
    if price_dict:
        alerts = await pos_tracker.update_all(price_dict)
        for alert in alerts:
            alert_type = alert["type"]
            pos = alert["position"]
            msg = f"🛑 <b>STOP LOSS HIT</b>\n" if alert_type == "STOP_LOSS" else f"✅ <b>TAKE PROFIT HIT</b>\n"
            msg += f"📊 {pos['symbol']} {pos['side']}\n"
            msg += f"💰 Entry: ${pos['entry_price']:.6f}\n"
            msg += f"🎯 Exit: ${pos.get('exit_price', 0):.6f}\n"
            msg += f"📈 P&L: ${pos.get('pnl', 0):.2f}"
            await send_telegram(msg)

async def scanner():
    log.info("🔍 Scanner starting...")
    await send_telegram("🚀 <b>Bot Started v5.0</b>\n⏳ Waiting for data stream...")
    for i in range(30):
        await asyncio.sleep(2)
        btc_count = len(TRADE_BUFFER.get("BTCUSDT", []))
        if btc_count >= 20:
            log.info(f"✅ Data ready! BTC trades: {btc_count}")
            await send_telegram(
                f"✅ <b>Data Stream Active!</b>\n"
                f"📊 {btc_count} BTC trades\n"
                f"🎯 {len(PAIRS)} pairs\n"
                f"📈 Charts: {'Enabled' if SEND_CHARTS else 'Disabled'}"
            )
            break
        if i % 5 == 0:
            log.info(f"Waiting for data... ({i+1}/30) - BTC: {btc_count}")
    else:
        log.warning("⚠️ Data stream slow, continuing anyway")

    scan_count = 0
    last_summary = datetime.now(timezone.utc)

    while not shutdown_requested:
        try:
            scan_count += 1
            results = []
            for sym in PAIRS:
                if shutdown_requested:
                    break
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat):
                        continue
                    try:
                        sig = await compute_signal(sym, strat, TRADE_BUFFER, OB_CACHE)
                        if sig and sig.get("validated", True):
                            confidence = sig.get("confidence", 0)
                            if confidence >= MIN_CONFIDENCE:
                                results.append(sig)
                    except Exception as e:
                        log.error(f"Signal error {sym}/{strat}: {e}")
                        continue

            if results:
                results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                best = results[:3]
                for sig in best:
                    try:
                        msg, chart_url = await formatter.format_signal_alert(
                            sig,
                            sig.get("levels"),
                            sig.get("volume"),
                            include_chart=SEND_CHARTS
                        )
                        success = await send_telegram(msg, chart_url)
                        if success:
                            signal_history.add_signal(sig)
                            ACTIVE_ORDERS[f"{sig['symbol']}:{sig['strategy']}"] = sig
                            set_cooldown(sig["symbol"], sig["strategy"])
                            log.info(
                                f"✔️ {sig['symbol']} {sig['strategy']} "
                                f"Score: {sig['score']:.1f} "
                                f"Conf: {sig.get('confidence', 0):.1f}% "
                                f"Corr: {sig.get('correlation', {}).get('price_corr', 0):+.2f}"
                            )
                    except Exception as e:
                        log.error(f"Failed to send signal: {e}")
            else:
                if scan_count % 20 == 0:
                    log.info(f"📊 Scan #{scan_count}: No signals")

            await update_positions()

            if (datetime.now(timezone.utc) - last_summary).total_seconds() >= 3600:
                recent_signals = signal_history.get_recent_signals(24)
                if recent_signals:
                    summary = formatter.format_summary_report(recent_signals, 1)
                    await send_telegram(summary)
                last_summary = datetime.now(timezone.utc)

            await asyncio.sleep(SCAN_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Scanner error: {e}")
            import traceback
            log.error(traceback.format_exc())
            await asyncio.sleep(5)

    log.info("Scanner stopped")

scan_task = None
ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task, app_ready
    log.info("=" * 60)
    log.info("🚀 Bot Starting (v5.0 Ultimate)")
    log.info(f"✓ {len(PAIRS)} pairs")
    log.info(f"✓ Min confidence: {MIN_CONFIDENCE}%")
    log.info(f"✓ Scan interval: {SCAN_INTERVAL}s")
    log.info(f"✓ Cooldown: {COOLDOWN_MIN}min")
    log.info(f"✓ Charts: {'Enabled' if SEND_CHARTS else 'Disabled'}")
    log.info("=" * 60)

    try:
        await pos_tracker.load()
        log.info("✓ Position tracker loaded")
    except Exception as e:
        log.warning(f"Position tracker load failed: {e}")

    ws_task = asyncio.create_task(start_websockets())
    await asyncio.sleep(3)
    scan_task = asyncio.create_task(scanner())
    app_ready = True
    log.info("✅ Application ready")

    yield

    log.info("🛑 Shutting down...")
    shutdown_requested = True

    if scan_task:
        scan_task.cancel()
    if ws_task:
        ws_task.cancel()

    try:
        await asyncio.gather(scan_task, ws_task, return_exceptions=True)
    except Exception as e:
        log.error(f"Shutdown error: {e}")

    try:
        await pos_tracker.save()
        log.info("✓ Position tracker saved")
    except Exception as e:
        log.warning(f"Position tracker save failed: {e}")

    log.info("🔴 Shutdown complete")

app = FastAPI(
    lifespan=lifespan,
    title="Crypto Trading Bot",
    version="5.0",
    description="Advanced trading signal generator with charts and correlation"
)

@app.get("/")
async def root():
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))
    return {
        "status": "running" if app_ready else "starting",
        "version": "5.0",
        "ws_connected": ws_connected,
        "data_received": data_received,
        "btc_trades": btc_trades,
        "pairs": len(PAIRS),
        "active_signals": len(ACTIVE_ORDERS),
        "open_positions": len(pos_tracker.open_positions),
        "features": {
            "charts": SEND_CHARTS,
            "correlation": True,
            "volume_profile": True,
            "smart_money": True
        },
        "time": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health():
    if not app_ready:
        return Response(content=json.dumps({"status": "starting"}), status_code=503, media_type="application/json")
    btc_trades = len(TRADE_BUFFER.get("BTCUSDT", []))
    return {
        "status": "healthy" if btc_trades > 0 else "degraded",
        "btc_trades": btc_trades,
        "ws_connected": ws_connected,
        "data_received": data_received
    }

@app.get("/stats")
async def stats():
    return {
        "trading_stats": pos_tracker.get_statistics(),
        "open_positions": pos_tracker.get_open_positions_summary(),
        "recent_signals": len(signal_history.get_recent_signals(10)),
        "signal_history": signal_history.get_recent_signals(20)
    }

@app.get("/signals")
async def signals():
    recent = signal_history.get_recent_signals(20)
    return {"count": len(recent), "signals": recent}

@app.get("/debug")
async def debug():
    debug_data = {}
    for sym in list(PAIRS)[:10]:
        debug_data[sym] = {
            "trades": len(TRADE_BUFFER.get(sym, [])),
            "has_ob": sym in OB_CACHE,
            "has_ticker": sym in TK_CACHE
        }
    return {
        "buffers": debug_data,
        "ob_cache_size": len(OB_CACHE),
        "ticker_cache_size": len(TK_CACHE),
        "ws_connected": ws_connected,
        "app_ready": app_ready,
        "shutdown_requested": shutdown_requested,
        "config": {
            "scan_interval": SCAN_INTERVAL,
            "cooldown_min": COOLDOWN_MIN,
            "min_confidence": MIN_CONFIDENCE,
            "send_charts": SEND_CHARTS
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        timeout_keep_alive=120
    )
