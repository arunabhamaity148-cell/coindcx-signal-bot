import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from helpers import (
    final_score, calc_tp_sl, cooldown_ok, update_cd,
    trade_side, apply_be_sl, format_signal
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT  = os.getenv("TELEGRAM_CHAT_ID")
INTERVAL = 7

WATCHLIST = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
"LINKUSDT","DOTUSDT","MATICUSDT","TRXUSDT","LTCUSDT","ATOMUSDT","FILUSDT","INJUSDT",
"OPUSDT","ARBUSDT","AAVEUSDT","SANDUSDT","DYDXUSDT","NEARUSDT","FTMUSDT","RUNEUSDT",
"PYTHUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","WLDUSDT","MINAUSDT","JTOUSDT","GALAUSDT",
"SUIUSDT","FLOWUSDT","CHZUSDT","CTSIUSDT","EGLDUSDT","APTUSDT","LDOUSDT","MKRUSDT",
"CRVUSDT","SKLUSDT","ENSUSDT","XLMUSDT","XTZUSDT","CFXUSDT","KAVAUSDT","PEPEUSDT","SHIBUSDT"
]

async def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, json={"chat_id": CHAT, "text": msg, "parse_mode": "HTML"})

async def get_ema(session, symbol, interval, length=20):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={length}"
    async with session.get(url, timeout=5) as r:
        kl = await r.json()
    if not isinstance(kl, list) or len(kl) < length:
        return None
    closes = [float(x[4]) for x in kl]
    return sum(closes[-length:]) / length

async def get_data(session, symbol):
    try:
        k_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=30"
        async with session.get(k_url, timeout=5) as r:
            kl = await r.json()
        if not isinstance(kl, list) or len(kl) < 3:
            return None

        price = float(kl[-1][4])
        prev  = float(kl[-2][4])
        volume_1m = float(kl[-1][5])

        if volume_1m < 5:
            return None

        spread = abs(float(kl[-1][2]) - float(kl[-1][3])) / price

        ema_15m = await get_ema(session, symbol, "15m", 20) or price
        ema_1h  = await get_ema(session, symbol, "1h", 20) or price
        ema_4h  = await get_ema(session, symbol, "4h", 20) or price
        ema_8h  = await get_ema(session, symbol, "4h", 40) or price  # fallback

        live = {
            "price": price,
            "ema_15m": ema_15m,
            "ema_1h": ema_1h,
            "ema_4h": ema_4h,
            "ema_8h": ema_8h,

            "trend": "bull_break" if price >= prev else "reject",
            "trend_strength": abs(price - prev) / price,
            "micro_pb": price > prev,
            "exhaustion": False,
            "fvg": False,
            "orderblock": False,

            "vol_1m": volume_1m,
            "vol_5m": sum(float(x[5]) for x in kl),

            "adr_ok": True,
            "atr_expanding": True,

            "spread": spread,
            "btc_calm": True,
            "kill_primary": False,

            "wick_ratio": 1.0,
            "liq_wall": False,
            "liq_bend": False,
            "wall_shift": False,
            "speed_imbalance": False,
            "absorption": False,
            "liquidation_sweep": False,
            "slippage": False,
            "liq_dist": 0.4,
            "taker_pressure": False
        }

        return live, price, prev

    except Exception as e:
        print("Error in get_data:", e)
        return None

async def handle_symbol(session, symbol):
    data = await get_data(session, symbol)
    if not data:
        print("Skipping (no data):", symbol)
        return

    live, price, prev = data

    side = trade_side(price, prev, live["ema_1h"])
    if not side:
        print("No direction:", symbol)
        return

    for mode in ["quick", "mid", "trend"]:
        score = final_score(live)
        threshold = {"quick":55, "mid":62, "trend":70}[mode]

        if score >= threshold and cooldown_ok(symbol):
            tp, sl = calc_tp_sl(price, mode)
            update_cd(symbol)

            msg = format_signal(
                symbol, side, mode, price, score, tp, sl, live["liq_dist"]
            )
            await send_telegram(msg)

            print("Signal --->", symbol, side, mode, score)
        else:
            print("Skip:", symbol, score)

async def scanner():
    print("🚀 FIXED BOT RUNNING…")
    async with aiohttp.ClientSession() as session:
        while True:
            await asyncio.gather(*[handle_symbol(session, s) for s in WATCHLIST])
            await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(scanner())
