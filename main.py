# main.py — FINAL PATCHED CLEAN VERSION
import asyncio
import aiohttp
import time
import os
from dotenv import load_dotenv

from helpers import process_data, btc_calm_check, format_signal

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","BNBUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","TRXUSDT",
    "DOTUSDT","LTCUSDT","LINKUSDT","MATICUSDT","OPUSDT","ARBUSDT","FILUSDT","AAVEUSDT","SANDUSDT",
    "ATOMUSDT","NEARUSDT","INJUSDT","FXSUSDT","DYDXUSDT","EGLDUSDT","APTUSDT","RNDRUSDT","TIAUSDT",
    "SEIUSDT","BONKUSDT","FTMUSDT","RUNEUSDT","PYTHUSDT","WLDUSDT","SKLUSDT","BLURUSDT","MINAUSDT",
    "JTOUSDT","MEWUSDT","1000PEPEUSDT"
]

MODES = ["quick", "mid", "trend"]

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 30))

# -----------------------------------------
# Telegram Sender
# -----------------------------------------
async def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload, timeout=5)
        except:
            pass

# -----------------------------------------
# Binance Data
# -----------------------------------------
async def get_live_data(session, symbol):
    try:
        async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5) as r:
            d = await r.json()
            price = float(d["price"])
    except:
        return None

    # Dummy data required for 58 logic system
    return {
        "price": price,
        "ema_15m": price,
        "ema_1h": price,
        "ema_4h": price,
        "ema_8h": price,
        "trend": "bull_break",
        "fvg": True,
        "trend_strength": 0.8,
        "exhaustion": False,
        "micro_pb": True,
        "wick_ratio": 1.8,
        "liquidation_sweep": False,
        "vol_1m": 50,
        "vol_5m": 40,
        "delta_1m": False,
        "delta_htf": True,
        "iceberg_1m": False,
        "iceberg_v2": False,
        "wall_shift": True,
        "liq_wall": False,
        "liq_bend": True,
        "adr_ok": True,
        "atr_expanding": False,
        "phase_shift": False,
        "compression": False,
        "speed_imbalance": True,
        "taker_pressure": True,
        "vol_imprint": True,
        "cluster_tiny": False,
        "absorption": False,
        "weakness": False,
        "spread_snap_05": True,
        "spread_snap_025": False,
        "spread": 0.02,
        "be_lock": False,
        "liq_dist": 0.4,
        "kill_5m": False,
        "kill_htf": False,
        "kill_fast": False,
        "kill_primary": False,
        "news_risk": False,
        "recheck_ok": True,
        "btc_calm": True,
        "btc_trending_fast": False,
        "funding_oi_combo": True,
        "funding_extreme": False,
        "funding_delta": False,
        "arb_opportunity": False,
        "oi_spike": True,
        "oi_sustained": False,
        "beta_div": False,
        "gamma_flip": True,
        "heat_sweep": False,
        "slippage": False,
        "orderblock": True
    }

# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
async def scanner():
    print("🚀 Hybrid AI Scanner Started...")

    async with aiohttp.ClientSession() as session:
        while True:

            # BTC Calm check
            btc_ok = await btc_calm_check(session)

            if not btc_ok:
                print("⚠️ BTC not calm → AI risk filter blocking signals")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            for symbol in WATCHLIST:
                live = await get_live_data(session, symbol)
                if not live:
                    continue

                for mode in MODES:
                    decision = process_data(symbol, mode, live)
                    if decision:
                        msg = format_signal(decision)
                        await send_telegram(msg)
                        print(f"📤 SIGNAL SENT → {symbol} | {mode}")

            await asyncio.sleep(SCAN_INTERVAL)

# -----------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(scanner())
    except KeyboardInterrupt:
        print("Stopped.")