# main.py — 50 Coins + Quick/Mid/Trend + Buy/Sell + Binance Scanner

import os
import asyncio
import aiohttp
import time
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


async def get_data(session, symbol):
    # ---- Binance klines ----
    k_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=30"
    ob_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"

    async with session.get(k_url) as r:
        kl = await r.json()

    async with session.get(ob_url) as r:
        ob = await r.json()

    price = float(kl[-1][4])
    prev  = float(kl[-2][4])

    spread = abs(float(ob["asks"][0][0]) - float(ob["bids"][0][0]))

    live = {
        "price": price,
        "ema_15m": price,
        "ema_1h": price,
        "ema_4h": price,
        "ema_8h": price,

        "trend": "bull_break" if price > prev else "reject",
        "trend_strength": abs(price - prev) / price,
        "micro_pb": price > prev,
        "exhaustion": False,
        "fvg": False,
        "orderblock": False,

        "vol_1m": sum(float(x[5]) for x in kl[-5:]),
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


async def handle_symbol(session, symbol):
    try:
        live, price, prev = await get_data(session, symbol)

        side = trade_side(price, prev, live['ema_1h'])
        if not side:
            print("No direction:", symbol)
            return

        for mode in ["quick", "mid", "trend"]:
            score = final_score(live)

            thresh = {"quick":55, "mid":62, "trend":70}[mode]

            if score >= thresh and cooldown_ok(symbol):
                tp, sl = calc_tp_sl(price, mode)
                update_cd(symbol)

                msg = format_signal(symbol, side, mode, price, score, tp, sl, live['liq_dist'])
                await send_telegram(msg)

                print("Signal ---->", symbol, side, mode, score)
            else:
                print("Skip:", symbol, score)

    except Exception as e:
        print("Error:", symbol, e)


async def scanner():
    print("🚀 30-Logic | Quick/Mid/Trend | Buy/Sell | Scanner Running...")
    async with aiohttp.ClientSession() as session:
        while True:
            tasks = [handle_symbol(session, s) for s in WATCHLIST]
            await asyncio.gather(*tasks)
            await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(scanner())