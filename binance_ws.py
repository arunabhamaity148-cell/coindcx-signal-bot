# binance_ws.py — Final production Binance WebSocket streamer (Railway Ready)

import asyncio
import json
import traceback
from datetime import datetime
import websockets

from helpers import redis, PAIRS

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
CHUNK_SIZE = 18

TRADES_TTL_SEC = 1800
OB_TTL_SEC = 120
TK_TTL_SEC = 120


async def push_trade(sym: str, price: float, qty: float, is_sell: bool, ts: int):
    try:
        await redis.lpush(f"tr:{sym}", json.dumps({"p": price, "q": qty, "m": is_sell, "t": ts}))
        await redis.ltrim(f"tr:{sym}", 0, 499)
        await redis.expire(f"tr:{sym}", TRADES_TTL_SEC)
    except Exception:
        pass


async def push_orderbook(sym: str, bid: float, ask: float):
    try:
        await redis.setex(f"ob:{sym}", OB_TTL_SEC, json.dumps({"bid": bid, "ask": ask, "t": int(datetime.utcnow().timestamp() * 1000)}))
    except Exception:
        pass


async def push_ticker(sym: str, last: float, vol: float, ts: int):
    try:
        await redis.setex(f"tk:{sym}", TK_TTL_SEC, json.dumps({"last": last, "vol": vol, "t": ts}))
    except Exception:
        pass


def build_stream_url(pairs_chunk):
    streams = []
    for p in pairs_chunk:
        p_low = p.lower()
        streams.append(f"{p_low}@aggTrade")
        streams.append(f"{p_low}@bookTicker")
    if "BTCUSDT" in pairs_chunk:
        streams.append("btcusdt@kline_1m")
    stream_path = "/".join(streams)
    return f"{BINANCE_WS_BASE}?streams={stream_path}"


async def ws_worker(pairs_chunk):
    url = build_stream_url(pairs_chunk)
    backoff = 1
    
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1
                print(f"✅ WS connected for {len(pairs_chunk)} pairs")
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        if stream.endswith("@aggTrade"):
                            sym = payload.get("s")
                            if not sym:
                                continue
                            price = float(payload.get("p", 0))
                            qty = float(payload.get("q", 0))
                            is_sell = bool(payload.get("m", False))
                            ts = int(payload.get("T", int(datetime.utcnow().timestamp() * 1000)))
                            await push_trade(sym, price, qty, is_sell, ts)

                        elif stream.endswith("@bookTicker"):
                            sym = payload.get("s")
                            if not sym:
                                continue
                            bid = float(payload.get("b", 0))
                            ask = float(payload.get("a", 0))
                            await push_orderbook(sym, bid, ask)

                        elif "@kline" in stream:
                            k = payload.get("k", {})
                            sym = payload.get("s", "BTCUSDT")
                            close = float(k.get("c", 0))
                            vol = float(k.get("q", 0))
                            ts = int(k.get("T", int(datetime.utcnow().timestamp() * 1000)))
                            await push_ticker(sym, close, vol, ts)

                    except Exception:
                        continue

        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
            print(f"⚠️ WS closed ({e}). Reconnecting in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        except Exception as e:
            print("⚠️ WS worker exception:", e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue


async def start_all_ws():
    # Test Redis first
    try:
        await redis.ping()
        print("✅ Redis connected in WS module")
    except Exception as e:
        print(f"❌ Redis connection failed in WS: {e}")
        return
    
    pairs = list(PAIRS)
    if not pairs:
        raise RuntimeError("PAIRS list empty in helpers.py")
    
    chunks = [pairs[i:i + CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    tasks = []
    
    for chunk in chunks:
        tasks.append(asyncio.create_task(ws_worker(chunk)))
    
    print(f"Started {len(tasks)} WS workers for {len(pairs)} pairs ({len(chunks)} chunks)")
    
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(start_all_ws())
    except KeyboardInterrupt:
        print("Interrupted, exiting...")