# main.py – v2.2-final (Redis OFF, restart-safe, Ping 200 OK log)
import os, json, asyncio, logging, time
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
import aiohttp, websockets, uvicorn
from fastapi import FastAPI
import aiofiles
from helpers import PAIRS
from scorer import compute_signal
from telegram_formatter import TelegramFormatter
from position_tracker import PositionTracker
from db import init_db, save_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# ---------- ENV ----------
TG_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT    = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INT   = int(os.getenv("SCAN_INTERVAL", 20))
COOLDOWN_M = int(os.getenv("COOLDOWN_MIN", 30))
BATCH      = int(os.getenv("WRITE_BATCH_SIZE", 50))
SYNC_INT   = int(os.getenv("PERIODIC_SYNC_INTERVAL", 15))

# ---------- STATE ----------
cooldown, ACTIVE_ORDERS = {}, {}
TRADE_BUF = {s: deque(maxlen=500) for s in PAIRS}
OB_CACHE, TK_CACHE = {}, {}
cntr, data_rx = {s: 0 for s in PAIRS}, 0
ws_connected = True

# ---------- DIRS ----------
TRADE_DIR = "/tmp/trades"
OB_DIR    = "/tmp/ob"
TK_DIR    = "/tmp/tk"
for d in (TRADE_DIR, OB_DIR, TK_DIR): os.makedirs(d, exist_ok=True)

# ---------- HELPERS ----------
formatter = TelegramFormatter()
pos_track = PositionTracker(storage_file=os.getenv("POSITION_FILE", "positions.json"))
db = None

async def tg_send(text: str):
    if not (TG_TOKEN and TG_CHAT): return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e: log.error(f"TG: {e}")

# ---------- FILE PERSIST ----------
async def persist_trades(sym: str, items: list):
    if not items: return
    fn = os.path.join(TRADE_DIR, f"{sym}.jsonl")
    try:
        async with aiofiles.open(fn, "a") as f:
            for i in items: await f.write(json.dumps(i) + "\n")
    except Exception as e: log.error(f"persist_trades: {e}")

async def persist_ob(mapping: dict):
    try:
        for sym, data in mapping.items():
            fn = os.path.join(OB_DIR, f"{sym}.json")
            async with aiofiles.open(fn, "w") as f: await f.write(json.dumps(data))
    except Exception as e: log.error(f"persist_ob: {e}")

async def persist_tk(mapping: dict):
    try:
        for sym, data in mapping.items():
            fn = os.path.join(TK_DIR, f"{sym}.json")
            async with aiofiles.open(fn, "w") as f: await f.write(json.dumps(data))
    except Exception as e: log.error(f"persist_tk: {e}")

# ---------- DATA PUSH ----------
async def push_trade(sym: str, data: dict):
    global data_rx
    if not all(k in data for k in ("p","q","m","t")): return
    TRADE_BUF[sym].append({"p": float(data["p"]), "q": float(data["q"]),
                           "m": bool(data["m"]), "t": int(data["t"])})
    cntr[sym] += 1; data_rx += 1
    if cntr[sym] >= BATCH:
        await persist_trades(sym, list(TRADE_BUF[sym])[-BATCH:])
        cntr[sym] = 0

async def push_ob(sym: str, bid: float, ask: float):
    OB_CACHE[sym] = {"bid": float(bid), "ask": float(ask), "t": int(time.time()*1000)}

async def push_tk(sym: str, last: float, vol: float, ts: int):
    TK_CACHE[sym] = {"last": float(last), "vol": float(vol), "t": int(ts)}

# ---------- PERIODIC SYNC ----------
async def periodic_sync():
    while True:
        await asyncio.sleep(SYNC_INT)
        if OB_CACHE: await persist_ob(OB_CACHE)
        if TK_CACHE: await persist_tk(TK_CACHE)

# ---------- WS ----------
WS_URL = "wss://stream.binance.com:9443/stream"
CHUNK  = int(os.getenv("CHUNK_SIZE", 15))

def build_url(chunk):
    s = "/".join([f"{p.lower()}@aggTrade" for p in chunk] +
                 [f"{p.lower()}@bookTicker" for p in chunk])
    if "BTCUSDT" in chunk: s += "/btcusdt@kline_1m"
    return f"{WS_URL}?streams={s}"

async def ws_worker(chunk, wid):
    url, backoff = build_url(chunk), 1
    log.info(f"WS-{wid} start ({len(chunk)} pairs)")
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1
                async for msg in ws:
                    try:
                        d = json.loads(msg)
                        st, pay = d.get("stream", ""), d.get("data", {})
                        sym = pay.get("s")
                        if not sym: continue
                        if st.endswith("@aggTrade"):
                            await push_trade(sym, {"p": pay["p"], "q": pay["q"], "m": pay["m"], "t": pay["T"]})
                        elif st.endswith("@bookTicker") and float(pay.get("b",0))>0:
                            await push_ob(sym, pay["b"], pay["a"])
                        elif "@kline" in st and pay.get("k",{}).get("x"):
                            k = pay["k"]
                            await push_tk(sym, k["c"], k["q"], k["T"])
                    except: pass
        except asyncio.CancelledError: break
        except Exception as e:
            log.warning(f"WS-{wid} {e} → retry {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 30)

async def start_ws():
    tasks = [asyncio.create_task(ws_worker(chunk, i+1))
             for i, chunk in enumerate([PAIRS[i:i+CHUNK] for i in range(0, len(PAIRS), CHUNK)])]
    tasks.append(asyncio.create_task(periodic_sync()))
    await asyncio.gather(*tasks, return_exceptions=True)

# ---------- SCANNER ----------
def cooldown_ok(sym, strat):
    key = f"{sym}:{strat}"
    return (datetime.utcnow() - cooldown.get(key, datetime.min)).total_seconds()/60 >= COOLDOWN_M

def set_cooldown(sym, strat):
    cooldown[f"{sym}:{strat}"] = datetime.utcnow()

async def scanner():
    await tg_send("🚀 Scanner starting…")
    for _ in range(20):
        await asyncio.sleep(2)
        if len(TRADE_BUF["BTCUSDT"]) > 20: break
    else: log.warning("data stream slow")
    while True:
        try:
            for sym in PAIRS:
                for strat in ["QUICK", "MID", "TREND"]:
                    if not cooldown_ok(sym, strat): continue
                    sig = await compute_signal(sym, strat, TRADE_BUF, OB_CACHE)
                    if sig and sig.get("validated", True):
                        msg = formatter.format_signal_alert(sig, sig.get("levels"), sig.get("volume"))
                        await tg_send(msg)
                        if db: await save_signal(db, sig)
                        set_cooldown(sym, strat)
                        log.info(f"SIGNAL {sym} {strat} {sig['score']:.1f}")
            await asyncio.sleep(SCAN_INT)
        except asyncio.CancelledError: break
        except Exception as e: log.error(f"scanner: {e}")

# ---------- FASTAPI ----------
scan_task = ws_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scan_task, ws_task, db
    log.info("🚀 Bot Starting (Redis OFF)")
    ws_task = asyncio.create_task(start_ws())
    await asyncio.sleep(3)
    await pos_track.load()
    try: db = await init_db(); log.info("DB ok")
    except: db = None; log.warning("DB skip")
    scan_task = asyncio.create_task(scanner())
    yield
    log.info("🛑 Shutting down…")
    for t in [scan_task, ws_task]: t.cancel()
    await asyncio.gather(scan_task, ws_task, return_exceptions=True)
    await pos_track.save()
    if db: await db.close()

app = FastAPI(lifespan=lifespan, timeout=120)  # ✅ restart-safe

@app.get("/")
def root():
    return {"status": "running", "ws": True, "btc_trades": len(TRADE_BUF.get("BTCUSDT", [])),
            "pairs": len(PAIRS), "signals": len(ACTIVE_ORDERS), "time": datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    btc_trades = len(TRADE_BUF.get("BTCUSDT", []))
    log.info("Ping 200 OK – health check")   # ✅ Ping 200 OK in log
    return {"status": "ok" if btc_trades > 0 else "starting", "btc_trades": btc_trades}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
