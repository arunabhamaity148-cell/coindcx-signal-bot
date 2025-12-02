# main.py — FULL AUTONOMOUS SIGNAL BOT (NO MANUAL APPROVAL)
import os
import asyncio
import aiohttp
import time
from dotenv import load_dotenv

load_dotenv()
from helpers import final_process, format_signal, cooldown_ok, update_cd

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT  = os.getenv("TELEGRAM_CHAT_ID")
INTERVAL = 8

WATCHLIST = [
"BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
"LINKUSDT","DOTUSDT","MATICUSDT","TRXUSDT","LTCUSDT","ATOMUSDT","FILUSDT","INJUSDT",
"OPUSDT","ARBUSDT","AAVEUSDT","SANDUSDT","DYDXUSDT","NEARUSDT","FTMUSDT","RUNEUSDT",
"PYTHUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","WLDUSDT","MINAUSDT","JTOUSDT","GALAUSDT",
"SUIUSDT","FLOWUSDT","CHZUSDT","CTSIUSDT","EGLDUSDT","APTUSDT","LDOUSDT","MKRUSDT",
"CRVUSDT","SKLUSDT","ENSUSDT","XLMUSDT","XTZUSDT","CFXUSDT","KAVAUSDT","PEPEUSDT","SHIBUSDT"
]

MODES = ["quick", "mid", "trend"]

async def send_telegram(msg):
    if not TOKEN:
        print("Telegram token missing")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={"chat_id": CHAT, "text": msg, "parse_mode": "HTML"})
    except Exception as e:
        print("Telegram error:", e)

async def handle_symbol(session, symbol):
    for mode in MODES:
        try:
            sig = await final_process(session, symbol, mode)
            if not sig:
                continue

            if not cooldown_ok(symbol):
                return

            msg = format_signal(sig)
            await send_telegram(msg)
            update_cd(symbol)

            print(f"SIGNAL SENT → {symbol} | {mode} | {sig['direction']} | score={sig['score']}")
            return
        except Exception as e:
            print("ERR", symbol, mode, e)
            return

async def scanner():
    print("🚀 Bot Started — AUTO DECISION MODE — Hybrid AI v3")
    async with aiohttp.ClientSession() as session:
        while True:
            t0 = time.time()
            tasks = [handle_symbol(session, s) for s in WATCHLIST]

            for i in range(0, len(tasks), 10):
                chunk = tasks[i:i+10]
                await asyncio.gather(*chunk)

            dt = time.time() - t0
            slp = max(1, INTERVAL - dt)
            print(f"Cycle = {dt:.2f}s | sleep {slp}s")
            await asyncio.sleep(slp)

if __name__ == "__main__":
    asyncio.run(scanner())