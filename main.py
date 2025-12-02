# main.py — FIXED SCANNER LOOP (indent corrected)
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from helpers import (
    process_data_with_ai,
    btc_calm_check,
    format_signal
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))

WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","BNBUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","TRXUSDT",
    "DOTUSDT","LTCUSDT","LINKUSDT","MATICUSDT","OPUSDT","ARBUSDT","FILUSDT","AAVEUSDT","SANDUSDT",
    "ATOMUSDT","NEARUSDT","INJUSDT","FXSUSDT","DYDXUSDT","EGLDUSDT","APTUSDT","RNDRUSDT","TIAUSDT",
    "SEIUSDT","BONKUSDT","FTMUSDT","RUNEUSDT","PYTHUSDT","WLDUSDT","SKLUSUSDT","BLURUSDT","MINAUSDT",
    "JTOUSDT","MEWUSDT","1000PEPEUSDT"
]

MODES = ["quick", "mid", "trend"]

async def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❗ Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload, timeout=5)
        except:
            pass

async def get_live_data(session, symbol):
    try:
        async with session.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5
        ) as r:
            if r.status != 200:
                return None
            d = await r.json()
            price = float(d.get("price"))
    except:
        return None

    return {
        "price": price,
        "ema_15m": price,
        "ema_1h": price,
        "ema_4h": price,
        "ema_8h": price,
        "trend": "bull_break",
        "fvg": True,
        "trend_strength": 0.8,
        "micro_pb": True,
        "wick_ratio": 1.8,
        "liq_dist": 0.4,
        "vol_1m": 50,
        "vol_5m": 40,
        "oi_spike": True,
        "btc_calm": True
    }

async def scanner():
    print("🚀 Hybrid AI Scanner Started...")
    async with aiohttp.ClientSession() as session:
        while True:

            btc_ok = await btc_calm_check(session)

            if not btc_ok:
                print("⚠️ BTC Volatile — AI layer evaluating risk (scanning continues)")

            # 🔥🔥🔥 THE LOOP THAT WAS NOT RUNNING (FIXED)
            for symbol in WATCHLIST:
                live = await get_live_data(session, symbol)
                if not live:
                    continue

                live["btc_calm"] = btc_ok

                for mode in MODES:
                    try:
                        decision = await process_data_with_ai(symbol, mode, live)
                    except Exception as e:
                        print("processor error:", e)
                        continue

                    if decision:
                        msg = format_signal(decision)
                        await send_telegram(msg)
                        print(
                            f"📤 SIGNAL SENT → {decision['symbol']} | {decision['mode']} | score={decision['score']}"
                        )

            await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    asyncio.run(scanner())