# main.py — PATCHED to use OpenAI verifier (process_data_with_ai)
import asyncio
import aiohttp
import time
import os
from dotenv import load_dotenv

load_dotenv()

from helpers import format_signal, btc_calm_check, process_data_with_ai

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

WATCHLIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","BNBUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","TRXUSDT",
    "DOTUSDT","LTCUSDT","LINKUSDT","MATICUSDT","OPUSDT","ARBUSDT","FILUSDT","AAVEUSDT","SANDUSDT",
    "ATOMUSDT","NEARUSDT","INJUSDT","FXSUSDT","DYDXUSDT","EGLDUSDT","APTUSDT","RNDRUSDT","TIAUSDT",
    "SEIUSDT","BONKUSDT","FTMUSDT","RUNEUSDT","PYTHUSDT","WLDUSDT","SKLUSUT","BLURUSDT","MINAUSDT",
    "JTOUSDT","MEWUSDT","1000PEPEUSDT"
]

MODES = ["quick", "mid", "trend"]
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "30"))

async def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    async with aiohttp.ClientSession() as session:
        try:
            await session.post(url, json=payload, timeout=5)
        except:
            pass

async def get_live_data(session, symbol):
    try:
        async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5) as r:
            d = await r.json()
            price = float(d["price"])
    except:
        return None

    # dummy structure — replace with real OB/OI/funding fetch later
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
                    # use async processor with OpenAI verifier
                    try:
                        decision = await process_data_with_ai(symbol, mode, live)
                    except Exception as e:
                        # protect scanner loop from processor errors
                        decision = None

                    if decision:
                        msg = format_signal(decision)
                        await send_telegram(msg)
                        print(f"📤 SIGNAL SENT → {symbol} | {mode} | score={decision.get('score')}")
            await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(scanner())
    except KeyboardInterrupt:
        print("Stopped.")