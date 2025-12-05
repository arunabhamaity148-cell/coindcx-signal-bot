# ============================================================
#  binance_ws.py — REALTIME BINANCE DATA ENGINE (PRO BUILD)
# ============================================================

import asyncio
import json
import websockets
from datetime import datetime
from redis.asyncio import Redis

PAIR_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "MATICUSDT", "BNBUSDT", "DOGEUSDT", "AVAXUSDT"
]

WS_URL = "wss://stream.binance.com:9443/stream"

redis = Redis.from_url("redis://localhost:6379", decode_responses=True)


# -----------------------------
# FORMATTER — unified storage
# -----------------------------
async def push_trade(sym, p, q, m, ts):
    await redis.lpush(f"tr:{sym}", json.dumps({
        "p": p,
        "q": q,
        "m": m,
        "t": ts
    }))
    await redis.ltrim(f"tr:{sym}", 0, 499)
    await redis.expire(f"tr:{sym}", 1800)


async def push_ticker(sym, last, vol, ts):
    await redis.setex(
        f"tk:{sym}", 120,
        json.dumps({"last": last, "vol": vol, "t": ts})
    )


async def push_orderbook(sym, bid, ask, ts):
    await redis.setex(
        f"ob:{sym}", 120,
        json.dumps({"bid": bid, "ask": ask, "t": ts})
    )


# -----------------------------
# MAIN WS CONNECTOR
# -----------------------------
async def start_binance_ws():
    streams = []

    for s in PAIR_LIST:
        streams.append(f"{s.lower()}@aggTrade")
        streams.append(f"{s.lower()}@bookTicker")

    # 1m candle for global market
    streams.append("btcusdt@kline_1m")

    url = WS_URL + "?streams=" + "/".join(streams)

    print("🔌 Connecting Binance WS…")

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print("✅ Binance WebSocket Connected")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    stream = data.get("stream")
                    payload = data.get("data", {})

                    # ------------------ TRADES ------------------
                    if "aggTrade" in stream:
                        sym = payload["s"]
                        p = float(payload["p"])
                        q = float(payload["q"])
                        m = payload["m"]
                        ts = int(payload["T"])
                        await push_trade(sym, p, q, m, ts)

                    # ------------------ ORDERBOOK TOP ------------------
                    elif "bookTicker" in stream:
                        sym = payload["s"]
                        bid = float(payload["b"])
                        ask = float(payload["a"])
                        ts = int(datetime.utcnow().timestamp() * 1000)
                        await push_orderbook(sym, bid, ask, ts)

                    # ------------------ BTC KLINE 1m ------------------
                    elif "kline" in stream:
                        k = payload["k"]
                        close = float(k["c"])
                        vol = float(k["q"])
                        ts = int(k["T"])
                        await push_ticker("BTCUSDT", close, vol, ts)

        except Exception as e:
            print("⚠️ Binance WS Error:", e)
            print("Reconnecting in 5 seconds…")
            await asyncio.sleep(5)