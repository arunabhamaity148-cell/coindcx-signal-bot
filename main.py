import os
import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from helpers import (
    final_score, calc_tp_sl, cooldown_ok, update_cd,
    trade_side, format_signal, auto_leverage, ema,
    calc_exhaustion, calc_fvg, calc_ob
)

TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT    = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("BINANCE_API_KEY")
API_SEC = os.getenv("BINANCE_SECRET")
INTERVAL = 60

WATCHLIST = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
"LINKUSDT","DOTUSDT","MATICUSDT","TRXUSDT","LTCUSDT","ATOMUSDT","FILUSDT","INJUSDT",
"OPUSDT","ARBUSDT","AAVEUSDT","SANDUSDT","DYDXUSDT","NEARUSDT","FTMUSDT","RUNEUSDT",
"PYTHUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","WLDUSDT","MINAUSDT","JTOUSDT","GALAUSDT",
"SUIUSDT","FLOWUSDT","CHZUSDT","CTSIUSDT","EGLDUSDT","APTUSDT","LDOUSDT","MKRUSDT",
"CRVUSDT","SKLUSDT","ENSUSDT","XLMUSDT","XTZUSDT","CFXUSDT","KAVAUSDT","PEPEUSDT","SHIBUSDT"
]

cache = {}
CACHE_TTL = 60

def sign(qs):
    return hmac.new(API_SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()

async def retry_get(session, url, params=None, headers=None, max_retry=5):
    for i in range(max_retry):
        try:
            async with session.get(url, params=params, headers=headers, timeout=15) as r:
                data = await r.json()
                if isinstance(data, dict) and data.get("code"):
                    print("Binance err", data, "retry", i+1)
                    await asyncio.sleep(2 ** i)
                    continue
                return data
        except Exception as e:
            print("Retry ex", e, i+1)
            await asyncio.sleep(2 ** i)
    return None

# ---------- klines batch ----------
async def fetch_klines_batch(session, symbols, interval="1m", limit=30):
    url  = "https://api.binance.com/api/v3/klines"
    symbols = [s.upper() for s in symbols]
    if not symbols:
        print("Empty symbols – skipping klines")
        return {}
    params = {"symbols": json.dumps(symbols, separators=(",", ":")), "interval": interval, "limit": limit}
    print("klines params", params)   # ➜ debug
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    data = await retry_get(session, url, params, headers)
    if not isinstance(data, list) or len(data) != len(symbols):
        return {}
    out = {}
    for idx, klines in enumerate(data):
        sym = symbols[idx]
        if isinstance(klines, list) and len(klines) >= 3:
            try:
                price = float(klines[-1][4])
                prev  = float(klines[-2][4])
                vol_1m= float(klines[-1][5])
                out[sym] = {"kl": klines, "price": price, "prev": prev, "vol_1m": vol_1m}
            except Exception as e:
                print("parse err", sym, e)
    return out

# ---------- spread batch ----------
async def fetch_spread_batch(session, symbols):
    url  = "https://api.binance.com/api/v3/ticker/bookTicker"
    symbols = [s.upper() for s in symbols]
    if not symbols:
        print("Empty symbols – skipping spread")
        return {}
    params = {"symbols": json.dumps(symbols, separators=(",", ":"))}
    print("spread params", params)   # ➜ debug
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    data = await retry_get(session, url, params, headers)
    out = {}
    if isinstance(data, list):
        for d in data:
            try:
                sym = d["symbol"]
                bid = float(d["bidPrice"])
                ask = float(d["askPrice"])
                out[sym] = (ask - bid) / ask if ask else 0
            except:
                pass
    return out

# ---------- ema batch ----------
async def fetch_ema_batch(session, symbols, interval, length=20):
    url  = "https://api.binance.com/api/v3/klines"
    symbols = [s.upper() for s in symbols]
    if not symbols:
        print("Empty symbols – skipping ema")
        return {}
    params = {"symbols": json.dumps(symbols, separators=(",", ":")), "interval": interval, "limit": length}
    print("ema params", params)   # ➜ debug
    headers = {"X-MBX-APIKEY": API_KEY} if API_KEY else {}
    data = await retry_get(session, url, params, headers)
    out = {}
    if isinstance(data, list) and len(data) == len(symbols):
        for idx, klines in enumerate(data):
            sym = symbols[idx]
            try:
                closes = [float(x[4]) for x in klines]
                out[sym] = ema(closes, length)
            except:
                pass
    return out

# ---------- telegram ----------
async def send(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, json={"chat_id": CHAT, "text": msg, "parse_mode": "HTML"})

# ---------- symbol handler ----------
MODE_THRESH = {"quick": 55, "mid": 62, "trend": 70}

async def handle_symbol(session, symbol, kdata, spreads, emas15, emas1h, emas4h, emas8h):
    if symbol not in kdata:
        return
    kd = kdata[symbol]
    price, prev, vol_1m, klines = kd["price"], kd["prev"], kd["vol_1m"], kd["kl"]
    spread = spreads.get(symbol, 0)
    ema15m = emas15.get(symbol, price)
    ema1h  = emas1h.get(symbol, price)
    ema4h  = emas4h.get(symbol, price)
    ema8h  = emas8h.get(symbol, price)

    if spread > 0.05:
        print("Risk skip", symbol)
        return

    side = trade_side(price, prev, ema1h)
    if not side:
        print("No dir", symbol)
        return

    for mode in ("quick", "mid", "trend"):
        score = final_score({
            "price": price, "ema_15m": ema15m, "ema_1h": ema1h, "ema_4h": ema4h, "ema_8h": ema8h,
            "trend": "bull_break" if price >= prev else "reject", "trend_strength": abs(price - prev) / price,
            "micro_pb": price > prev and price > ema15m,
            "exhaustion": calc_exhaustion(klines[-1]), "fvg": calc_fvg(klines), "orderblock": calc_ob(klines),
            "vol_1m": vol_1m, "vol_5m": sum(float(x[5]) for x in klines),
            "adr_ok": True, "atr_expanding": True,
            "spread": spread, "btc_calm": True, "kill_primary": False,
            "wick_ratio": 1.0, "liq_wall": False, "liq_bend": False,
            "wall_shift": False, "speed_imbalance": False, "absorption": False,
            "liquidation_sweep": False, "slippage": False, "liq_dist": 0.4, "taker_pressure": False
        })
        if score >= MODE_THRESH[mode] and cooldown_ok(symbol):
            tp, sl = calc_tp_sl(price, mode)
            lev = auto_leverage(score, mode)
            update_cd(symbol)
            await send(format_signal(symbol, side, mode, price, score, tp, sl, 0.4, lev))
            print("Signal", symbol, side, mode, score, lev)

# ---------- scanner ----------
async def scanner():
    print("🚀 BATCH-48 COIN BOT RUNNING")
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        while True:
            tasks = []
            # ➜ 20 করে batch করো (min 2 guarantee)
            for i in range(0, len(WATCHLIST), 20):
                batch = WATCHLIST[i:i+20]
                tasks.append(process_batch(session, batch, sem))
            await asyncio.gather(*tasks)
            await asyncio.sleep(INTERVAL)


async def process_batch(session, batch, sem):
    async with sem:
        if not batch:
            print("Empty batch – skipping")
            return
        print("Processing batch", batch)   # ➜ debug
        now = time.time()
        if cache.get("kl_time", 0) > now - CACHE_TTL:
            kdata   = cache.get("kl", {})
            spreads = cache.get("spr", {})
            em15    = cache.get("em15", {})
            em1h    = cache.get("em1h", {})
            em4h    = cache.get("em4h", {})
            em8h    = cache.get("em8h", {})
        else:
            kdata   = await fetch_klines_batch(session, batch)
            spreads = await fetch_spread_batch(session, batch)
            em15    = await fetch_ema_batch(session, batch, "15m", 20)
            em1h    = await fetch_ema_batch(session, batch, "1h", 20)
            em4h    = await fetch_ema_batch(session, batch, "4h", 20)
            em8h    = await fetch_ema_batch(session, batch, "4h", 40)
            cache.update({"kl": kdata, "spr": spreads, "em15": em15, "em1h": em1h, "em4h": em4h, "em8h": em8h, "kl_time": now})

        await asyncio.gather(*[handle_symbol(session, sym, kdata, spreads, em15, em1h, em4h, em8h) for sym in batch])

if __name__ == "__main__":
    asyncio.run(scanner())
