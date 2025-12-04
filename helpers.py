"""
helpers.py  part-1  (Redis / WS / data / features)
"""
import os, json, asyncio, logging, joblib, aiohttp
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import ccxt.async_support as ccxt
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

# ---------- Config ----------
CFG = {
    "key": os.getenv("BINANCE_KEY"),
    "secret": os.getenv("BINANCE_SECRET"),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "testnet": os.getenv("USE_TESTNET", "false").lower() == "true",
    "tg_token": os.getenv("TELEGRAM_TOKEN"),
    "tg_chat": os.getenv("TELEGRAM_CHAT_ID"),
    "equity": float(os.getenv("EQUITY_USD", 6000)),
    "risk_perc": float(os.getenv("RISK_PERC", 0.8)),
    "max_lev": int(os.getenv("MAX_LEV", 15)),
    "pairs": json.loads(os.getenv("TOP_PAIRS", '["BTCUSDT","ETHUSDT","BNBUSDT"]')),
    "openai_key": os.getenv("OPENAI_KEY"),
    "min_score": float(os.getenv("MIN_SCORE", 7.8)),
    "min_rrr": float(os.getenv("MIN_RRR", 2.0)),
}

# ---------- Redis ----------
redis_client: Redis = None

async def redis():
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(
            CFG["redis_url"],
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        await redis_client.ping()
        log.info("✓ Redis connected")
    return redis_client

async def redis_close():
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None

# ---------- WebSocket ----------
class WS:
    def __init__(self):
        self.url = "wss://fstream.binance.com/stream?streams="
        self.running = False

    def build(self):
        streams = []
        for p in CFG["pairs"]:
            pl = p.lower()
            streams += [f"{pl}@ticker", f"{pl}@depth20@100ms", f"{pl}@trade",
                        f"{pl}@kline_1m", f"{pl}@kline_5m", f"{pl}@kline_15m"]
        self.url += "/".join(streams)

    async def run(self):
        self.build(); self.running = True; retry = 0
        while self.running and retry < 10:
            try:
                log.info(f"🔌 WS connect attempt {retry+1}")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.url, heartbeat=30) as ws:
                        log.info("✓ WebSocket connected"); retry = 0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT.TEXT:
                                await self._handle(json.loads(msg.data))
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                self.running = False; break
            except Exception as e:
                retry += 1; wait = min(retry * 2, 30)
                log.error(f"WS error: {e} → retry {wait}s")
                await asyncio.sleep(wait)

    async def _handle(self, data):
        try:
            r = await redis()
            stream = data.get("stream", "")
            d = data.get("data", {})
            s = d.get("s")
            if not s:
                return
            if "@ticker" in stream:
                await r.hset(f"t:{s}", mapping={
                    "last": d.get("c", 0), "vol": d.get("v", 0),
                    "qvol": d.get("q", 0), "trades": d.get("n", 0), "E": d.get("E", 0)
                })
            elif "@depth20" in stream:
                await r.hset(f"d:{s}", mapping={
                    "bids": json.dumps(d.get("bids", [])),
                    "asks": json.dumps(d.get("asks", [])), "E": d.get("E", 0)
                })
            elif "@trade" in stream:
                trade = json.dumps({"p": d.get("p"), "q": d.get("q"), "m": d.get("m"), "t": d.get("T")})
                await r.lpush(f"tr:{s}", trade); await r.ltrim(f"tr:{s}", 0, 499)
            elif "@kline" in stream:
                k = d.get("k", {})
                if k.get("x"):
                    kline = json.dumps({
                        "t": k.get("t"), "o": k.get("o"), "h": k.get("h"),
                        "l": k.get("l"), "c": k.get("c"), "v": k.get("v")
                    })
                    await r.lpush(f"kline_{k.get('i')}:{s}", kline)
                    await r.ltrim(f"kline_{k.get('i')}:{s}", 0, 199)
        except Exception as e:
            log.error(f"WS handler error: {e}")

# ---------- OHLCV ----------
async def get_ohlcv(sym, tf="5m", limit=100):
    try:
        r = await redis()
        klines_raw = await r.lrange(f"kline_{tf}:{sym}", 0, limit - 1)
        if not klines_raw:
            return None
        klines = [json.loads(k) for k in reversed(klines_raw)]
        df = pd.DataFrame(klines)
        df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
        df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    except Exception as e:
        log.error(f"OHLCV error {sym} {tf}: {e}")
        return None

async def get_real_atr(sym, tf="5m", period=14):
    try:
        df = await get_ohlcv(sym, tf, limit=period + 20)
        if df is None or len(df) < period:
            return 0.005
        tr = np.maximum(df["h"] - df["l"],
                        np.maximum(np.abs(df["h"] - df["c"].shift()),
                                   np.abs(df["l"] - df["c"].shift())))
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr / df["c"].iloc[-1])
    except Exception as e:
        log.warning(f"ATR error {sym}: {e}")
        return 0.005

# ---------- Indicators ----------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

async def mtf_trend(sym):
    try:
        trends = []
        for tf in ["1m", "5m", "15m"]:
            df = await get_ohlcv(sym, tf, 50)
            if df is None or len(df) < 50:
                continue
            ema20 = df["c"].ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = df["c"].ewm(span=50, adjust=False).mean().iloc[-1]
            trends.append(1 if ema20 > ema50 else -1)
        if not trends:
            return 0
        avg = sum(trends) / len(trends)
        return 1 if avg > 0.6 else -1 if avg < -0.6 else 0
    except Exception as e:
        log.error(f"MTF error {sym}: {e}")
        return 0

# ---------- Orderflow ----------
async def orderflow_analysis(sym):
    try:
        r = await redis()
        trades_raw = await r.lrange(f"tr:{sym}", 0, 499)
        if not trades_raw:
            return None
        trades = [json.loads(t) for t in trades_raw]

        delta = sum(float(t["q"]) if t["m"] == "false" else -float(t["q"]) for t in trades)
        recent_delta = sum(float(t["q"]) if t["m"] == "false" else -float(t["q"]) for t in trades[:50])

        volumes = [float(t["q"]) for t in trades]
        avg_vol, std_vol = np.mean(volumes), np.std(volumes)
        large_buys = sum(1 for t in trades[:20] if t["m"] == "false" and float(t["q"]) > avg_vol + 2 * std_vol)
        large_sells = sum(1 for t in trades[:20] if t["m"] == "true" and float(t["q"]) > avg_vol + 2 * std_vol)

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

        return {
            "delta": delta, "recent_delta": recent_delta,
            "large_buys": large_buys, "large_sells": large_sells,
            "imbalance": imbalance, "spread": spread,
            "depth_usd": depth_usd, "delta_momentum": recent_delta / (abs(delta) + 1e-6)
        }
    except Exception as e:
        log.error(f"Orderflow error {sym}: {e}")
        return None

# ---------- Regime ----------
async def regime(sym):
    try:
        df = await get_ohlcv(sym, "15m", 50)
        if df is None or len(df) < 30:
            return "CHOP"
        atr_pct = await get_real_atr(sym, "15m", 14)
        high, low, close = df["h"].values, df["l"].values, df["c"].values
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr_smooth = pd.Series(tr).rolling(14).mean().iloc[-1]
        volatility = atr_smooth / close[-1]
        if volatility > 0.015:
            return "TREND"
        return "CHOP" if volatility < 0.008 else "VOLATILE"
    except Exception as e:
        log.warning(f"Regime error {sym}: {e}")
        return "CHOP"

# ---------- Feature Extract ----------
FEATURE_COLS = [
    "rsi_1m", "rsi_5m", "rsi_div", "macd_hist", "mom_1m", "mom_5m",
    "vol_ratio", "atr_1m", "atr_5m", "delta_norm", "delta_momentum",
    "imbalance", "large_buys", "large_sells", "spread", "depth",
    "mtf_trend", "reg_trend", "reg_chop", "funding", "poc_dist"
]

async def extract_features(sym):
    try:
        print(f"DEBUG_FEATURE: started for {sym}")
        df_1m = await get_ohlcv(sym, "1m", 100)
        df_5m = await get_ohlcv(sym, "5m", 100)
        if df_1m is None or len(df_1m) < 50:
            print(f"DEBUG_FEATURE: no 1m data for {sym}")
            return None
        print(f"DEBUG_FEATURE: df_1m rows={len(df_1m)}")
        last = float(df_1m["c"].iloc[-1])

        rsi_1m = calc_rsi(df_1m["c"], 14).iloc[-1]
        rsi_5m = calc_rsi(df_5m["c"], 14).iloc[-1] if df_5m is not None and len(df_5m) > 14 else 50

        ema12 = df_1m["c"].ewm(span=12).mean()
        ema26 = df_1m["c"].ewm(span=26).mean()
        macd_hist = (ema12 - ema26).iloc[-1] - (ema12 - ema26).ewm(span=9).mean().iloc[-1]

        mom_1m = (df_1m["c"].iloc[-1] - df_1m["c"].iloc[-10]) / df_1m["c"].iloc[-10]
        mom_5m = (df_5m["c"].iloc[-1] - df_5m["c"].iloc[-5]) / df_5m["c"].iloc[-5] if df_5m is not None else 0

        vol_ratio = df_1m["v"].iloc[-10:].mean() / df_1m["v"].iloc[-50:].mean()

        atr_1m = await get_real_atr(sym, "1m", 14)
        atr_5m = await get_real_atr(sym, "5m", 14)

        flow = await orderflow_analysis(sym)
        if not flow:
            print(f"DEBUG_FEATURE: no orderflow for {sym}")
            return None

        mtf = await mtf_trend(sym)
        reg = await regime(sym)

        return {
            "rsi_1m": rsi_1m / 100, "rsi_5m": rsi_5m / 100, "rsi_div": 0.0,
            "macd_hist": np.tanh(macd_hist / last),
            "mom_1m": np.tanh(mom_1m * 100), "mom_5m": np.tanh(mom_5m * 100),
            "vol_ratio": min(vol_ratio, 3) / 3,
            "atr_1m": min(atr_1m, 0.02) / 0.02, "atr_5m": min(atr_5m, 0.02) / 0.02,
            "delta_norm": np.tanh(flow["delta"] / 1e6),
            "delta_momentum": flow["delta_momentum"],
            "imbalance": (flow["imbalance"] + 1) / 2,
            "large_buys": min(flow["large_buys"], 5) / 5,
            "large_sells": min(flow["large_sells"], 5) / 5,
            "spread": min(flow["spread"], 0.2) / 0.2,
            "depth": np.tanh(flow["depth_usd"] / 5e6),
            "mtf_trend": (mtf + 1) / 2,
            "reg_trend": 1 if reg == "TREND" else 0,
            "reg_chop": 1 if reg == "CHOP" else 0,
            "funding": 0.0, "poc_dist": 0.0, "last": last
        }
    except Exception as e:
        log.error(f"Feature extraction error {sym}: {e}")
        return None
"""
helpers.py  part-2  (ML / score / TP-SL / risk / Telegram / exchange)
"""
import os, json, joblib, aiohttp
import numpy as np
from datetime import datetime
from helpers_part1 import CFG, redis, redis_close, get_real_atr, regime, mtf_trend, orderflow_analysis, extract_features, FEATURE_COLS

log = logging.getLogger("helpers")

# ---------- ML Ensemble ----------
def load_ensemble():
    models = []
    for name in ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]:
        if os.path.exists(name):
            models.append(joblib.load(name))
            log.info(f"✓ Loaded {name}")
    return models if len(models) == 3 else None

async def ai_review_ensemble(sym, side, score):
    try:
        features = await extract_features(sym)
        if not features:
            return {"allow": False, "confidence": 30, "reason": "no-features"}
        last = features.pop("last")
        X = np.array([features[c] for c in FEATURE_COLS]).reshape(1, -1)

        models = load_ensemble()
        if not models:
            return {"allow": True, "confidence": 60, "reason": "no-ml-fallback"}

        preds = []
        for m in models:
            try:
                proba = m.predict_proba(X)[0]
                preds.append(proba[1] if len(proba) > 1 else 0.5)
            except:
                preds.append(0.5)
        avg_prob = np.mean(preds)
        confidence = int(abs(avg_prob - 0.5) * 200)
        ml_side = "long" if avg_prob > 0.58 else "short" if avg_prob < 0.42 else "none"
        if ml_side != side:
            return {"allow": False, "confidence": confidence, "reason": "ml-disagree"}

        flow = await orderflow_analysis(sym)
        if not flow:
            return {"allow": False, "confidence": 40, "reason": "no-flow"}
        if flow["spread"] > 0.12:
            return {"allow": False, "confidence": confidence, "reason": "spread-wide"}
        if flow["depth_usd"] < 8e5:
            return {"allow": False, "confidence": confidence, "reason": "low-liquidity"}

        mtf = await mtf_trend(sym)
        if side == "long" and mtf < 0:
            return {"allow": False, "confidence": confidence, "reason": "mtf-bearish"}
        if side == "short" and mtf > 0:
            return {"allow": False, "confidence": confidence, "reason": "mtf-bullish"}

        if await regime(sym) == "CHOP":
            return {"allow": False, "confidence": confidence, "reason": "choppy"}

        if side == "long" and flow["recent_delta"] < 0:
            return {"allow": False, "confidence": confidence, "reason": "delta-bearish"}
        if side == "short" and flow["recent_delta"] > 0:
            return {"allow": False, "confidence": confidence, "reason": "delta-bullish"}

        if confidence < 65:
            return {"allow": False, "confidence": confidence, "reason": "low-confidence"}

        return {"allow": True, "confidence": confidence, "reason": "all-green"}
    except Exception as e:
        log.error(f"AI review error {sym}: {e}")
        return {"allow": False, "confidence": 30, "reason": f"error:{str(e)[:20]}"}

# ---------- Scoring ----------
async def calculate_advanced_score(sym):
    try:
        features = await extract_features(sym)
        if not features:
            return None
        last = features.get("last", 0)
        scores = {}

        rsi_1m = features["rsi_1m"] * 100
        scores["rsi"] = 3.0 if rsi_1m > 70 else 7.0 if rsi_1m < 30 else 5.0

        mom_1m, mom_5m = features["mom_1m"], features["mom_5m"]
        if mom_1m > 0.5 and mom_5m > 0.5:
            scores["momentum"] = 8.5
        elif mom_1m < -0.5 and mom_5m < -0.5:
            scores["momentum"] = 1.5
        else:
            scores["momentum"] = 5.0

        delta_mom, imbalance = features["delta_momentum"], (features["imbalance"] - 0.5) * 2
        if delta_mom > 0.6 and imbalance > 0.3:
            scores["orderflow"] = 9.0
        elif delta_mom < -0.6 and imbalance < -0.3:
            scores["orderflow"] = 1.0
        else:
            scores["orderflow"] = 5.0

        mtf = features["mtf_trend"] * 2 - 1
        scores["mtf"] = 9.0 if mtf > 0.8 else 1.0 if mtf < -0.8 else 5.0

        if features["reg_trend"] == 1:
            scores["regime"] = 8.0
        elif features["reg_chop"] == 1:
            scores["regime"] = 2.0
        else:
            scores["regime"] = 6.0

        vol_ratio = features["vol_ratio"] * 3
        scores["volume"] = 8.0 if vol_ratio > 1.5 else 4.0 if vol_ratio < 0.7 else 6.0

        weights = {"rsi": 2.5, "momentum": 2.0, "orderflow": 3.0, "mtf": 2.5, "regime": 1.5, "volume": 1.5}
        total = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())

        side = "long" if total >= 7.8 else "short" if total <= 2.2 else "none"
        return {"score": round(total, 2), "side": side, "components": scores, "last": last, "features": features}
    except Exception as e:
        log.error(f"Scoring error {sym}: {e}")
        return None

# ---------- TP / SL ----------
async def calc_smart_tp_sl(sym, side, entry):
    try:
        atr_1m = await get_real_atr(sym, "1m", 14)
        atr_5m = await get_real_atr(sym, "5m", 14)
        atr_avg = (atr_1m + atr_5m) / 2
        reg = await regime(sym)

        if reg == "TREND":
            sl_mult, tp_mult = 1.0, 2.5
        elif reg == "VOLATILE":
            sl_mult, tp_mult = 1.2, 2.0
        else:
            sl_mult, tp_mult = 0.8, 1.8

        entry = float(entry)
        if side == "long":
            sl = entry - (atr_avg * entry * sl_mult)
            tp = entry + (entry - sl) * tp_mult
        else:
            sl = entry + (atr_avg * entry * sl_mult)
            tp = entry - (sl - entry) * tp_mult
        return round(tp, 8), round(sl, 8)
    except Exception as e:
        log.error(f"TP/SL error {sym}: {e}")
        return (entry * 1.015, entry * 0.992) if side == "long" else (entry * 0.985, entry * 1.008)

# ---------- Position Size ----------
def position_size(equity, entry, sl, risk_pct=0.8):
    try:
        risk_usd = equity * risk_pct / 100
        entry, sl = float(entry), float(sl)
        price_risk = abs(entry - sl) or entry * 0.008
        qty = risk_usd / price_risk
        max_qty = (equity * 0.2) / entry
        return round(min(qty, max_qty), 6)
    except Exception as e:
        log.error(f"Position size error: {e}")
        return 0.001

# ---------- Risk ----------
async def check_risk_limits(equity, open_positions):
    try:
        if len(open_positions) >= 3:
            return False, "max-positions"
        exposure = sum(abs(p["size"]) * p["entry"] for p in open_positions.values())
        if exposure > equity * 0.6:
            return False, "max-exposure"
        r = await redis()
        daily = float(await r.get("daily_pnl") or 0)
        if daily < -equity * 0.05:
            return False, "daily-loss-limit"
        return True, "ok"
    except Exception as e:
        log.error(f"Risk check error: {e}")
        return False, "error"

async def update_daily_pnl(pnl):
    try:
        r = await redis()
        current = float(await r.get("daily_pnl") or 0)
        await r.set("daily_pnl", current + pnl)
        now = datetime.utcnow()
        if now.hour == 0 and now.minute < 5:
            await r.set("daily_pnl", 0)
    except Exception as e:
        log.error(f"PnL update error: {e}")

# ---------- Telegram ----------
async def send_telegram(txt):
    if not CFG["tg_token"] or not CFG["tg_chat"]:
        return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.post(url, json={"chat_id": CFG["tg_chat"], "text": txt, "parse_mode": "HTML"}) as r:
                    if r.status == 200:
                        return
        except Exception as e:
            log.warning(f"Telegram error (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)

# ---------- Exchange ----------
class Exchange:
    def __init__(self):
        self.ex = ccxt.binance({
            "apiKey": CFG["key"],
            "secret": CFG["secret"],
            "options": {"defaultType": "future", "testnet": CFG["testnet"]},
            "enableRateLimit": True,
            "timeout": 30000,
        })

    async def limit(self, sym, side, qty, price, post_only=True):
        try:
            order = await self.ex.create_order(sym, "LIMIT", side, qty, price, params={"postOnly": post_only})
            log.info(f"✓ Limit order: {sym} {side} {qty} @ {price}")
            return order
        except Exception as e:
            log.error(f"❌ Limit order failed: {e}")
            return None

    async def market(self, sym, side, qty):
        try:
            order = await self.ex.create_order(sym, "MARKET", side, qty)
            log.info(f"✓ Market order: {sym} {side} {qty}")
            return order
        except Exception as e:
            log.error(f"❌ Market order failed: {e}")
            return None

    async def set_leverage(self, sym, leverage):
        try:
            await self.ex.fapiPrivate_post_leverage({"symbol": sym.replace("/", ""), "leverage": leverage})
            log.info(f"✓ Leverage: {sym} {leverage}x")
        except Exception as e:
            log.warning(f"Leverage error {sym}: {e}")

    async def set_sl_tp(self, sym, side, sl, tp):
        try:
            sl_side = "SELL" if side == "long" else "BUY"
            await self.ex.create_order(sym, "STOP_MARKET", sl_side, None, params={"stopPrice": sl, "closePosition": True})
            tp_side = "SELL" if side == "long" else "BUY"
            await self.ex.create_order(sym, "TAKE_PROFIT_MARKET", tp_side, None, params={"stopPrice": tp, "closePosition": True})
            log.info(f"✓ SL/TP: {sym} SL={sl} TP={tp}")
        except Exception as e:
            log.error(f"❌ SL/TP error {sym}: {e}")

    async def get_position(self, sym):
        try:
            positions = await self.ex.fapiPrivate_get_positionrisk({"symbol": sym.replace("/", "")})
            for p in positions:
                if float(p["positionAmt"]) != 0:
                    return {
                        "size": float(p["positionAmt"]),
                        "entry": float(p["entryPrice"]),
                        "unrealizedPnl": float(p["unRealizedProfit"]),
                        "leverage": int(p["leverage"])
                    }
            return None
        except Exception as e:
            log.error(f"Position check error {sym}: {e}")
            return None

    async def close(self):
        await self.ex.close()

# ---------- Cleanup ----------
async def cleanup():
    await redis_close()
    log.info("✓ Cleanup complete")
