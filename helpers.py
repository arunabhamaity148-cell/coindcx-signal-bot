import os, json, asyncio, logging, joblib, aiohttp, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import ccxt.async_support as ccxt
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

CFG = {
    "key": os.getenv("COINDCX_KEY"),
    "secret": os.getenv("COINDCX_SECRET"),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "tg_token": os.getenv("TELEGRAM_TOKEN"),
    "tg_chat": os.getenv("TELEGRAM_CHAT_ID"),
    "equity": float(os.getenv("EQUITY_USD", 100000)),
    "risk_perc": float(os.getenv("RISK_PERC", 1.0)),
    "min_lev": int(os.getenv("MIN_LEV", 15)),
    "max_lev": int(os.getenv("MAX_LEV", 30)),
    "liq_buffer": float(os.getenv("LIQ_BUFFER", 0.15)),
    "cooldown_min": int(os.getenv("COOLDOWN_MIN", 45)),
    "pairs": json.loads(os.getenv("TOP_PAIRS", '["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","LTCUSDT","UNIUSDT","ETCUSDT","FILUSDT","TRXUSDT","NEARUSDT","ARBUSDT","APTUSDT","INJUSDT","STXUSDT","TIAUSDT","SEIUSDT","OPUSDT","SUIUSDT","FETUSDT","RENDERUSDT","IMXUSDT","RUNEUSDT","XLMUSDT","ALGOUSDT","SANDUSDT","ICPUSDT","GRTUSDT","AAVEUSDT","LDOUSDT","HBARUSDT","FTMUSDT","VETUSDT","MANAUSDT","AXSUSDT","THETAUSDT","FLOWUSDT","SNXUSDT","CHZUSDT","ENJUSDT","MKRUSDT","COMPUSDT","KSMUSDT","XTZUSDT","ZECUSDT","DASHUSDT","BATUSDT","ZILUSDT","ONTUSDT","IOSTUSDT","IOTAUSDT","QTUMUSDT","WAVESUSDT","ZRXUSDT","OMGUSDT","CRVUSDT","YFIUSDT","BALUSDT","1INCHUSDT","RLCUSDT","KAVAUSDT","SUSHIUSDT","OCEANUSDT","RSRUSDT","CELOUSDT","BANDUSDT","STORJUSDT","CELRUSDT","SKLUSDT","ANKRUSDT","BLZUSDT","ARPAUSDT","NMRUSDT"]')),
}

STRATEGY_CONFIG = {
    "QUICK": {"min_score": 7.0, "max_score": 8.5, "tp1_mult": 1.2, "tp2_mult": 1.8, "sl_mult": 0.8, "min_conf": 60, "tp1_exit": 0.5},
    "MID": {"min_score": 7.2, "max_score": 8.8, "tp1_mult": 1.5, "tp2_mult": 2.5, "sl_mult": 1.0, "min_conf": 65, "tp1_exit": 0.4},
    "TREND": {"min_score": 7.5, "max_score": 10.0, "tp1_mult": 2.0, "tp2_mult": 3.5, "sl_mult": 1.2, "min_conf": 70, "tp1_exit": 0.3}
}

redis_client: Redis = None
async def redis():
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(CFG["redis_url"], decode_responses=True, socket_connect_timeout=5)
        await redis_client.ping()
        log.info("✓ Redis connected")
    return redis_client
async def redis_close():
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
# ---------- CoinDCX-only exchange ----------
def get_exchange(**kwargs):
    exchange_class = getattr(ccxt, 'coincdcx', None)
    if exchange_class:
        return exchange_class({**kwargs, "enableRateLimit": True, "timeout": 30000})
    # fallback REST
    return ccxt.Exchange({
        "id": "coincdcx", "name": "CoinCDCX", "countries": ["IN"],
        "urls": {"api": {"public": "https://api.coindcx.com/exchange/v1", "private": "https://api.coindcx.com/exchange/v1"}},
        "api": {"public": {"get": ["markets", "tickers", "orderbook", "trades"]},
                "private": {"get": ["orders", "balances"], "post": ["order/new", "order/cancel"]}},
        "markets": None, **kwargs, "enableRateLimit": True, "timeout": 30000
    })

class WS:
    def __init__(self):
        self.running = False
        self.ex = get_exchange()
    async def run(self):
        self.running = True
        log.info("🔌 Starting CoinDCX data polling...")
        while self.running:
            try:
                for sym in CFG["pairs"]:
                    try:
                        ticker = await self.ex.fetch_ticker(sym)
                        orderbook = await self.ex.fetch_order_book(sym, limit=20)
                        trades = await self.ex.fetch_trades(sym, limit=100)
                        r = await redis()
                        await r.hset(f"t:{sym}", mapping={"last": ticker['last'], "vol": ticker.get('baseVolume', 0), "E": int(datetime.utcnow().timestamp() * 1000)})
                        await r.hset(f"d:{sym}", mapping={"bids": json.dumps(orderbook['bids'][:20]), "asks": json.dumps(orderbook['asks'][:20]), "E": int(datetime.utcnow().timestamp() * 1000)})
                        for t in trades[-100:]:
                            await r.lpush(f"tr:{sym}", json.dumps({"p": t['price'], "q": t['amount'], "m": t['side'] == 'sell', "t": t['timestamp']}))
                        await r.ltrim(f"tr:{sym}", 0, 499)
                    except Exception as e:
                        log.debug(f"Poll {sym}: {e}")
                        continue
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                self.running = False
                break
            except Exception as e:
                log.error(f"WS polling: {e}")
                await asyncio.sleep(5)

# ---------- OHLCV from CoinDCX trades ----------
async def build_ohlcv_from_trades(sym, timeframe='1m', bars=100):
    r = await redis()
    trades_raw = await r.lrange(f"tr:{sym}", 0, bars * 70)
    if not trades_raw or len(trades_raw) < 50:
        return None
    trades = [json.loads(t) for t in trades_raw]
    df_trades = pd.DataFrame(trades)
    df_trades['p'] = df_trades['p'].astype(float)
    df_trades['q'] = df_trades['q'].astype(float)
    df_trades['t'] = pd.to_datetime(df_trades['t'], unit='ms', utc=True)
    df_trades = df_trades.set_index('t').sort_index()
    ohlc = df_trades['p'].resample(timeframe).ohlc()
    vol = df_trades['q'].resample(timeframe).sum()
    df = ohlc.join(vol.rename('v')).dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}, inplace=True)
    return df.tail(bars) if len(df) >= bars else None

async def get_ohlcv(sym, tf="5m", limit=100):
    df = await build_ohlcv_from_trades(sym, timeframe=tf, bars=limit)
    if df is None or len(df) < 50:
        r = await redis()
        last = float(await r.hget(f"t:{sym}", "last") or 0)
        if not last:
            return None
        now = datetime.utcnow()
        flat = pd.DataFrame([{"t": now, "o": last, "h": last, "l": last, "c": last, "v": 0}])
        return pd.concat([flat] * limit, ignore_index=True)
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

async def mtf_trend(sym):
    trends = []
    for tf in ["1m", "5m"]:
        df = await get_ohlcv(sym, tf, 50)
        if df is None or len(df) < 50:
            continue
        ema20 = df["c"].ewm(span=20).mean().iloc[-1]
        ema50 = df["c"].ewm(span=50).mean().iloc[-1]
        trends.append(1 if ema20 > ema50 else -1)
    if not trends:
        return 0
    avg = sum(trends) / len(trends)
    return 1 if avg > 0.5 else -1 if avg < -0.5 else 0

async def orderflow_analysis(sym):
    try:
        r = await redis()
        trades_raw = await r.lrange(f"tr:{sym}", 0, 499)
        if not trades_raw:
            return None
        trades = [json.loads(t) for t in trades_raw]
        delta = sum(float(t["q"]) if not t["m"] else -float(t["q"]) for t in trades)
        recent_delta = sum(float(t["q"]) if not t["m"] else -float(t["q"]) for t in trades[:50])
        volumes = [float(t["q"]) for t in trades]
        avg_vol, std_vol = np.mean(volumes), np.std(volumes)
        large_buys = sum(1 for t in trades[:20] if not t["m"] and float(t["q"]) > avg_vol + 2 * std_vol)
        large_sells = sum(1 for t in trades[:20] if t["m"] and float(t["q"]) > avg_vol + 2 * std_vol)
        depth_raw = await r.hgetall(f"d:{sym}")
        if not depth_raw:
            return None
        bids = json.loads(depth_raw.get("bids", "[]"))
        asks = json.loads(depth_raw.get("asks", "[]"))
        if not bids or not asks:
            return None
        bid_vol = sum(float(b[1]) for b in bids[:5])
        ask_vol = sum(float(a[1]) for a in asks[:5])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 100
        depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:10]) + sum(float(a[0]) * float(a[1]) for a in asks[:10])
        return {"delta": delta, "recent_delta": recent_delta, "large_buys": large_buys, "large_sells": large_sells, "imbalance": imbalance, "spread": spread, "depth_usd": depth_usd, "delta_momentum": recent_delta / (abs(delta) + 1e-6)}
    except Exception as e:
        log.error(f"Orderflow {sym}: {e}")
        return None
async def regime(sym):
    try:
        df = await get_ohlcv(sym, "15m", 50)
        if df is None or len(df) < 30:
            return "NORMAL"
        high, low, close = df["h"].values, df["l"].values, df["c"].values
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        volatility = pd.Series(tr).rolling(14).mean().iloc[-1] / close[-1]
        return "TREND" if volatility > 0.015 else "CHOP" if volatility < 0.005 else "NORMAL"
    except:
        return "NORMAL"

FEATURE_COLS = ["rsi_1m", "rsi_5m", "macd_hist", "mom_1m", "mom_5m", "vol_ratio", "atr_1m", "atr_5m", "delta_norm", "delta_momentum", "imbalance", "large_buys", "large_sells", "spread", "depth", "mtf_trend", "reg_trend", "reg_chop", "funding", "poc_dist"]

# ---------- alias for main.py ----------
async def calculate_advanced_score(sym, strategy):
    return await calc_advanced_score(sym, strategy)


# ---------- Exchange wrapper ----------
class Exchange:
    def __init__(self):
        self.ex = get_exchange(apiKey=CFG["key"], secret=CFG["secret"], options={"defaultType": "swap"})
        log.info("✓ CoinDCX initialized (manual trading mode)")
    async def close(self):
        await self.ex.close()

# ---------- rest of helpers ----------
