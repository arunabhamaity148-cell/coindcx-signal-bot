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
    "pairs": json.loads(os.getenv("TOP_PAIRS", '["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]')),
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

def get_exchange(**kwargs):
    cls = getattr(ccxt, 'coincdcx', None)
    if cls:
        return cls({**kwargs, "enableRateLimit": True, "timeout": 30000})
    return ccxt.Exchange({
        "id": "coincdcx", "name": "CoinCDCX",
        "urls": {"api": {"public": "https://api.coindcx.com/exchange/v1", "private": "https://api.coindcx.com/exchange/v1"}},
        "api": {"public": {"get": ["markets", "tickers", "orderbook", "trades"]}, "private": {"get": ["orders", "balances"], "post": ["order/new", "order/cancel"]}},
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

async def extract_features(sym):
    try:
        df_1m = await get_ohlcv(sym, "1m", 100)
        df_5m = await get_ohlcv(sym, "5m", 100)
        if df_1m is None or len(df_1m) < 50:
            return None
        last = float(df_1m["c"].iloc[-1])
        rsi_1m = calc_rsi(df_1m["c"], 14).iloc[-1]
        rsi_5m = calc_rsi(df_5m["c"], 14).iloc[-1] if df_5m is not None and len(df_5m) > 14 else 50
        ema12 = df_1m["c"].ewm(span=12).mean()
        ema26 = df_1m["c"].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
        mom_1m = (df_1m["c"].iloc[-1] - df_1m["c"].iloc[-10]) / df_1m["c"].iloc[-10]
        mom_5m = (df_5m["c"].iloc[-1] - df_5m["c"].iloc[-5]) / df_5m["c"].iloc[-5] if df_5m is not None else 0
        vol_ratio = df_1m["v"].iloc[-10:].mean() / df_1m["v"].iloc[-50:].mean()
        atr_1m = await get_real_atr(sym, "1m", 14)
        atr_5m = await get_real_atr(sym, "5m", 14)
        flow = await orderflow_analysis(sym)
        if not flow:
            return None
        mtf = await mtf_trend(sym)
        reg = await regime(sym)
        return {"rsi_1m": rsi_1m / 100, "rsi_5m": rsi_5m / 100, "macd_hist": np.tanh(macd_hist / last), "mom_1m": np.tanh(mom_1m * 100), "mom_5m": np.tanh(mom_5m * 100), "vol_ratio": min(vol_ratio, 3) / 3, "atr_1m": min(atr_1m, 0.02) / 0.02, "atr_5m": min(atr_5m, 0.02) / 0.02, "delta_norm": np.tanh(flow["delta"] / 1e6), "delta_momentum": flow["delta_momentum"], "imbalance": (flow["imbalance"] + 1) / 2, "large_buys": min(flow["large_buys"], 5) / 5, "large_sells": min(flow["large_sells"], 5) / 5, "spread": min(flow["spread"], 0.2) / 0.2, "depth": np.tanh(flow["depth_usd"] / 5e6), "mtf_trend": (mtf + 1) / 2, "reg_trend": 1 if reg == "TREND" else 0, "reg_chop": 1 if reg == "CHOP" else 0, "funding": 0.0, "poc_dist": 0.0, "last": last}
    except Exception as e:
        log.error(f"Feature {sym}: {e}")
        return None

def load_ensemble():
    models = []
    for name in ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]:
        if os.path.exists(name):
            models.append(joblib.load(name))
    return models if len(models) == 3 else None

async def ai_review_ensemble(sym, side, score, strategy):
    try:
        features = await extract_features(sym)
        if not features:
            return {"allow": False, "confidence": 30, "reason": "no-features"}
        last = features.pop("last")
        X = np.array([features[c] for c in FEATURE_COLS]).reshape(1, -1)
        models = load_ensemble()
        if not models:
            return {"allow": True, "confidence": 55, "reason": "no-ml"}
        preds = []
        for m in models:
            try:
                proba = m.predict_proba(X)[0]
                preds.append(proba[1] if len(proba) > 1 else 0.5)
            except:
                preds.append(0.5)
        avg_prob = np.mean(preds)
        confidence = int(abs(avg_prob - 0.5) * 200)
        ml_side = "long" if avg_prob > 0.52 else "short" if avg_prob < 0.48 else "none"
        if ml_side == "long" and side == "short":
            return {"allow": False, "confidence": confidence, "reason": "ml-opposite"}
        if ml_side == "short" and side == "long":
            return {"allow": False, "confidence": confidence, "reason": "ml-opposite"}
        flow = await orderflow_analysis(sym)
        if not flow:
            return {"allow": False, "confidence": 40, "reason": "no-flow"}
        if flow["spread"] > 0.25:
            return {"allow": False, "confidence": confidence, "reason": "spread-wide"}
        if flow["depth_usd"] < 3e5:
            return {"allow": False, "confidence": confidence, "reason": "low-liquidity"}
        min_conf = STRATEGY_CONFIG[strategy]["min_conf"]
        if confidence < min_conf:
            return {"allow": False, "confidence": confidence, "reason": f"low-conf"}
        return {"allow": True, "confidence": confidence, "reason": f"{strategy.lower()}-approved"}
    except Exception as e:
        log.error(f"AI review {sym}: {e}")
        return {"allow": False, "confidence": 30, "reason": "error"}

async def calc_tp1_tp2_sl_liq(sym, side, entry, confidence, strategy):
    try:
        atr = await get_real_atr(sym, "5m", 14)
        entry = float(entry)
        cfg = STRATEGY_CONFIG[strategy]
        leverage = CFG["min_lev"] + int((confidence - 60) / 40 * (CFG["max_lev"] - CFG["min_lev"]))
        leverage = max(CFG["min_lev"], min(leverage, CFG["max_lev"]))
        liq_dist_pct = 1 / leverage
        max_sl_dist = entry * liq_dist_pct * (1 - CFG["liq_buffer"])
        base_sl_dist = atr * entry * cfg["sl_mult"]
        sl_dist = min(base_sl_dist, max_sl_dist)
        if side == "long":
            sl = entry - sl_dist
            tp1 = entry + (sl_dist * cfg["tp1_mult"])
            tp2 = entry + (sl_dist * cfg["tp2_mult"])
        else:
            sl = entry + sl_dist
            tp1 = entry - (sl_dist * cfg["tp1_mult"])
            tp2 = entry - (sl_dist * cfg["tp2_mult"])
        liq_price = entry * (1 - 1/leverage) if side == "long" else entry * (1 + 1/leverage)
        if side == "long" and sl <= liq_price:
            sl = liq_price * 1.02
        elif side == "short" and sl >= liq_price:
            sl = liq_price * 0.98
        return round(tp1, 8), round(tp2, 8), round(sl, 8), leverage, round(liq_price, 8)
    except Exception as e:
        log.error(f"TP/SL calc {sym}: {e}")
        if side == "long":
            return entry * 1.012, entry * 1.018, entry * 0.992, 15, entry * 0.93
        return entry * 0.988, entry * 0.982, entry * 1.008, 15, entry * 1.07

def position_size_iceberg(equity, entry, sl, leverage):
    try:
        risk_usd = equity * CFG["risk_perc"] / 100
        entry, sl = float(entry), float(sl)
        price_risk = abs(entry - sl)
        qty = (risk_usd / price_risk) / leverage
        max_qty = (equity * 0.3) / entry
        total_qty = round(min(qty, max_qty), 6)
        num_orders = 4
        iceberg_qty = round(total_qty / num_orders, 6)
        return {"total_qty": total_qty, "iceberg_qty": iceberg_qty, "num_orders": num_orders}
    except Exception as e:
        log.error(f"Position size: {e}")
        return {"total_qty": 0.001, "iceberg_qty": 0.0003, "num_orders": 3}

async def check_cooldown(sym, strategy):
    try:
        r = await redis()
        key = f"cooldown:{sym}:{strategy}"
        last_trade = await r.get(key)
        if last_trade:
            last_time = datetime.fromisoformat(last_trade)
            elapsed = (datetime.utcnow() - last_time).total_seconds() / 60
            if elapsed < CFG["cooldown_min"]:
                return False, f"cooldown-{int(CFG['cooldown_min'] - elapsed)}min"
        return True, "ok"
    except Exception as e:
        return True, "ok"

async def set_cooldown(sym, strategy):
    try:
        r = await redis()
        key = f"cooldown:{sym}:{strategy}"
        await r.set(key, datetime.utcnow().isoformat(), ex=CFG["cooldown_min"] * 60)
    except Exception as e:
        log.error(f"Cooldown set: {e}")

async def check_daily_signal_limit():
    try:
        r = await redis()
        count = await r.get("daily_signal_count")
        if count and int(count) >= 30:
            return False, "daily-limit-30"
        return True, "ok"
    except:
        return True, "ok"

async def increment_signal_count():
    try:
        r = await redis()
        await r.incr("daily_signal_count")
        await r.expire("daily_signal_count", 86400)
    except:
        pass

async def send_telegram(txt):
    if not CFG["tg_token"] or not CFG["tg_chat"]:
        return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(url, json={"chat_id": CFG["tg_chat"], "text": txt, "parse_mode": "HTML", "disable_web_page_preview": True}) as resp:
                    if resp.status == 200:
                        log.info("✓ Telegram sent")
                        return
        except Exception as e:
            await asyncio.sleep(1)

class Exchange:
    def __init__(self):
        self.ex = get_exchange(apiKey=CFG["key"], secret=CFG["secret"], options={"defaultType": "swap"})
        log.info("✓ CoinDCX initialized (manual trading mode)")
    async def close(self):
        await self.ex.close()

async def cleanup():
    await redis_close()
    log.info("✓ Cleanup complete")
