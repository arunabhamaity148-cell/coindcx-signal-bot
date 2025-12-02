# main.py — FULL REAL BINANCE AUTO-SIGNAL BOT (FINAL)
import os
import time
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

from helpers import (
    final_process,
    format_signal,
    cooldown_ok,
    update_cd
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT  = os.getenv("TELEGRAM_CHAT_ID")

INTERVAL = 8   # seconds

WATCHLIST = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
"LINKUSDT","DOTUSDT","MATICUSDT","TRXUSDT","LTCUSDT","ATOMUSDT","FILUSDT","INJUSDT",
"OPUSDT","ARBUSDT","AAVEUSDT","SANDUSDT","DYDXUSDT","NEARUSDT","FTMUSDT","RUNEUSDT",
"PYTHUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","WLDUSDT","MINAUSDT","JTOUSDT","GALAUSDT",
"SUIUSDT","FLOWUSDT","CHZUSDT","CTSIUSDT","EGLDUSDT","APTUSDT","LDOUSDT","MKRUSDT",
"CRVUSDT","SKLUSDT","ENSUSDT","XLMUSDT","XTZUSDT","CFXUSDT","KAVAUSDT","PEPEUSDT","SHIBUSDT"
]

MODES = ["quick", "mid", "trend"]

# ----------------------------------------------------------
# TELEGRAM
# ----------------------------------------------------------
async def send_telegram(msg):
    if not TOKEN:
        print("❌ Telegram Token Missing")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={
                "chat_id": CHAT,
                "text": msg,
                "parse_mode": "HTML"
            })
    except Exception as e:
        print("Telegram error:", e)

# ----------------------------------------------------------
# PROCESS EACH SYMBOL
# ----------------------------------------------------------
async def handle_symbol(session, symbol):
    for mode in MODES:
        try:
            res = await final_process(session, symbol, mode)
            if not res:
                continue

            if not cooldown_ok(symbol):
                return

            msg = format_signal(res)
            await send_telegram(msg)
            update_cd(symbol)

            print(f"✅ SIGNAL SENT | {symbol} | {mode} | {res['direction']} | Score={res['score']}")
            return

        except Exception as e:
            print("ERR:", symbol, mode, e)
            return

# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------
async def scanner():
    print("🚀 Real Binance AI Bot Started — AUTO-DECISION ACTIVE")

    async with aiohttp.ClientSession() as session:
        while True:
            start = time.time()

            tasks = [handle_symbol(session, s) for s in WATCHLIST]

            # Run in chunks to avoid overload
            for i in range(0, len(tasks), 10):
                batch = tasks[i:i+10]
                await asyncio.gather(*batch)

            t = time.time() - start
            sleep_time = max(1, INTERVAL - t)
            print(f"Cycle: {t:.2f}s | Sleep: {sleep_time}s")
            await asyncio.sleep(sleep_time)

# ----------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(scanner())
    except KeyboardInterrupt:
        print("Bot stopped manually.")