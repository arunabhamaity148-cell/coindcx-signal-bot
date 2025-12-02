# main.py — FINAL (scanner + telegram + uses helpers.process_data_with_ai)
import os
import asyncio
import aiohttp
import time
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
        print("❗ Telegram not configured in .env")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=5)
    except Exception as e:
        # swallow to keep scanner alive
        print("Telegram send error:", str(e))

async def get_live_data(session, symbol):
    # Minimal REST price fetch (replace/extend with OB/OI/funding endpoints if available)
    try:
        async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5) as r:
            if r.status != 200:
                return None
            d = await r.json()
            price = float(d.get("price"))
    except Exception:
        return None

    # Dummy enrichment — this is required shape for 58 logic functions.
    # Later you should replace with real klines/orderbook/OI/funding fetches.
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

async def scanner():
    print("🚀 Hybrid AI Scanner Started...")
    async with aiohttp.ClientSession() as session:
        while True:
            # Check BTC calm (robust)
            btc_ok = await btc_calm_check(session)
            if not btc_ok:
                # Do NOT skip scanning — mark risk and continue scanning.
                print("⚠️ BTC Volatile — AI layer to evaluate risk (scanning continues)")

            for symbol in WATCHLIST:
                live = await get_live_data(session, symbol)
                if not live:
                    continue

                # attach btc_calm flag for logic functions
                live["btc_calm"] = btc_ok

                for mode in MODES:
                    try:
                        decision = await process_data_with_ai(symbol, mode, live)
                    except Exception as e:
                        # protect loop
                        print("Processor error:", str(e))
                        decision = None

                    if decision:
                        msg = format_signal(decision)
                        await send_telegram(msg)
                        print(f"📤 SIGNAL SENT → {decision['symbol']} | {decision['mode']} | score={decision.get('score')}")

            await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(scanner())
    except KeyboardInterrupt:
        print("Stopped.")