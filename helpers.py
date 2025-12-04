"""
helpers.py  –  COMBINED  (part-1 + part-2)
Redis / WS / data / ML / score / TP-SL / risk / Telegram / exchange
"""
import os, json, asyncio, logging, joblib, aiohttp, pandas as pd, numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import ccxt.async_support as ccxt
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

# ---------- CONFIG ----------
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
    "pairs": json.loads(os.getenv("TOP_PAIRS", '["BTCUSDT","ETHUSDT"]')),
    "min_score": float(os.getenv("MIN_SCORE", 5.8)),
    "min_rrr": float(os.getenv("MIN_RRR", 1.0)),
}

# ---------- REDIS ----------
redis_client: Redis = None
async def redis():
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(CFG["redis_url"], decode_responses=True, socket_connect_timeout=5)
        await redis_client.ping()
        log.info("✓ Redis connected")
    return redis_client
async def redis_close():
    if redis_client: await redis_client.close()

# ---------- WEBSOCKET ----------
class WS:
    def __init__(self): self.url = "wss://fstream.binance.com/stream?streams="; self.running = False
    def build(self):
        self.url += "/".join([f"{p.lower()}@ticker", f"{p.lower()}@depth20@100ms", f"{p.lower()}@trade", f"{p.lower()}@kline_1m", f"{p.lower()}@kline_5m"] for p in CFG["pairs"])
    async def run(self):
        self.build(); self.running = True; retry = 0
        while self.running and retry < 10:
            try:
                log.info("🔌 WS connect"); async with aiohttp.ClientSession() as s, s.ws_connect(self.url, heartbeat=30) as ws:
                    log.info("✓ WS connected"); retry = 0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT: await self._handle(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR): break
            except asyncio.CancelledError: self.running = False; break
            except Exception as e: retry += 1; await asyncio.sleep(min(retry * 2, 30))
    async def _handle(self, data):
        try:
            r, stream, d, s = await redis(), data.get("stream", ""), data.get("data", {}), d.get("s")
            if not s: return
            if "@ticker" in stream: await r.hset(f"t:{s}", mapping={"last": d.get("c"), "vol": d.get("v")})
            elif "@depth20" in stream: await r.hset(f"d:{s}", mapping={"bids": json.dumps(d.get("bids")), "asks": json.dumps(d.get("asks"))})
            elif "@trade" in stream: trade = json.dumps({"p": d.get("p"), "q": d.get("q"), "m": d.get("m")}); await r.lpush(f"tr:{s}", trade); await r.ltrim(f"tr:{s}", 0, 499)
            elif "@kline" in stream and d.get("k", {}).get("x"): k = d["k"]; kline = json.dumps({"t": k["t"], "o": k["o"], "h": k["h"], "l": k["l"], "c": k["c"], "v": k["v"]}); await r.lpush(f"kline_{k['i']}:{s}", kline); await r.ltrim(f"kline_{k['i']}:{s}", 0, 199)
        except Exception as e: log.error(f"WS handler: {e}")

# ---------- OHLCV ----------
async def get_ohlcv(sym, tf="5m", limit=100):
    try:
        r, klines_raw = await redis(), await r.lrange(f"kline_{tf}:{sym}", 0, limit - 1)
        if not klines_raw: return None
        klines = [json.loads(k) for k in reversed(klines_raw)]
        df = pd.DataFrame(klines); df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float); df["t"] = pd.to_datetime(df["t"], unit="ms")
        return df
    except Exception as e: log.error(f"OHLCV {sym}: {e}"); return None

# ---------- ATR / RSI / MTF / ORDERFLOW / REGIME ----------
async def get_real_atr(sym, tf="5m", period=14):
    df = await get_ohlcv(sym, tf, limit=period + 20)
    if df is None or len(df) < period: return 0.005
    tr = np.maximum(df["h"] - df["l"], np.maximum(np.abs(df["h"] - df["c"].shift()), np.abs(df["l"] - df["c"].shift())))
    return float(tr.rolling(period).mean().iloc[-1] / df["c"].iloc[-1])

def calc_rsi(series, period=14):
    delta = series.diff(); gain = delta.clip(lower=0).rolling(period).mean(); loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10); return 100 - (100 / (1 + rs))

async def mtf_trend(sym):
    trends = []
    for tf in ["1m", "5m", "15m"]:
        df = await get_ohlcv(sym, tf, 50)
        if df is None or len(df) < 50: continue
        ema20, ema50 = df["c"].ewm(span=20).mean().iloc[-1], df["c"].ewm(span=50).mean().iloc[-1]
        trends.append(1 if ema20 > ema50 else -1)
    if not trends: return 0
    avg = sum(trends) / len(trends)
    return 1 if avg > 0.6 else -1 if avg < -0.6 else 0

async def orderflow_analysis(sym):
    try:
        r, trades_raw = await redis(), await r.lrange(f"tr:{sym}", 0, 499)
        if not trades_raw: return None
        trades = [json.loads(t) for t in trades_raw]
        delta = sum(float(t["q"]) if t["m"] == "false" else -float(t["q"]) for t in trades)
        recent_delta = sum(float(t["q"]) if t["m"] == "false" else -float(t["q"]) for t in trades[:50])
        volumes = [float(t["q"]) for t in trades]; avg_vol, std_vol = np.mean(volumes), np.std(volumes)
        large_buys = sum(1 for t in trades[:20] if t["m"] == "false" and float(t["q"]) > avg_vol + 2 * std_vol)
        large_sells = sum(1 for t in trades[:20] if t["m"] == "true" and float(t["q"]) > avg_vol + 2 * std_vol)
        depth_raw = await r.hgetall(f"d:{sym}")
        if not depth_raw: return None
        bids, asks = json.loads(depth_raw.get("bids", "[]")), json.loads(depth_raw.get("asks", "[]"))
        if not bids or not asks: return None
        bid_vol, ask_vol = sum(float(b[1]) for b in bids[:5]), sum(float(a[1]) for a in asks[:5])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 100
        depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:10]) + sum(float(a[0]) * float(a[1]) for a in asks[:10])
        return {"delta": delta, "recent_delta": recent_delta, "large_buys": large_buys, "large_sells": large_sells, "imbalance": imbalance, "spread": spread, "depth_usd": depth_usd, "delta_momentum": recent_delta / (abs(delta) + 1e-6)}
    except Exception as e: log.error(f"Orderflow {sym}: {e}"); return None

async def regime(sym):
    df = await get_ohlcv(sym, "15m", 50)
    if df is None or len(df) < 30: return "CHOP"
    atr_pct = await get_real_atr(sym, "15m", 14)
    high, low, close = df["h"].values, df["l"].values, df["c"].values
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    volatility = pd.Series(tr).rolling(14).mean().iloc[-1] / close[-1]
    return "TREND" if volatility > 0.015 else "CHOP" if volatility < 0.008 else "VOLATILE"

# ---------- FEATURE EXTRACT ----------
FEATURE_COLS = ["rsi_1m", "rsi_5m", "macd_hist", "mom_1m", "mom_5m", "vol_ratio", "atr_1m", "atr_5m", "delta_norm", "delta_momentum", "imbalance", "large_buys", "large_sells", "spread", "depth", "mtf_trend", "reg_trend", "reg_chop", "funding", "poc_dist"]
async def extract_features(sym):
    try:
        df_1m = await get_ohlcv(sym, "1m", 100)
        df_5m = await get_ohlcv(sym, "5m", 100)
        if df_1m is None or len(df_1m) < 50: return None
        last = float(df_1m["c"].iloc[-1])
        rsi_1m = calc_rsi(df_1m["c"], 14).iloc[-1]
        rsi_5m = calc_rsi(df_5m["c"], 14).iloc[-1] if df_5m is not None and len(df_5m) > 14 else 50
        ema12, ema26 = df_1m["c"].ewm(span=12).mean(), df_1m["c"].ewm(span=26).mean()
        macd_hist = (ema12 - ema26).iloc[-1] - (ema12 - ema26).ewm(span=9).mean().iloc[-1]
        mom_1m = (df_1m["c"].iloc[-1] - df_1m["c"].iloc[-10]) / df_1m["c"].iloc[-10]
        mom_5m = (df_5m["c"].iloc[-1] - df_5m["c"].iloc[-5]) / df_5m["c"].iloc[-5] if df_5m is not None else 0
        vol_ratio = df_1m["v"].iloc[-10:].mean() / df_1m["v"].iloc[-50:].mean()
        atr_1m, atr_5m = await get_real_atr(sym, "1m", 14), await get_real_atr(sym, "5m", 14)
        flow = await orderflow_analysis(sym)
        if not flow: return None
        mtf, reg = await mtf_trend(sym), await regime(sym)
        return {
            "rsi_1m": rsi_1m / 100, "rsi_5m": rsi_5m / 100, "macd_hist": np.tanh(macd_hist / last),
            "mom_1m": np.tanh(mom_1m * 100), "mom_5m": np.tanh(mom_5m * 100), "vol_ratio": min(vol_ratio, 3) / 3,
            "atr_1m": min(atr_1m, 0.02) / 0.02, "atr_5m": min(atr_5m, 0.02) / 0.02,
            "delta_norm": np.tanh(flow["delta"] / 1e6), "delta_momentum": flow["delta_momentum"],
            "imbalance": (flow["imbalance"] + 1) / 2, "large_buys": min(flow["large_buys"], 5) / 5,
            "large_sells": min(flow["large_sells"], 5) / 5, "spread": min(flow["spread"], 0.2) / 0.2,
            "depth": np.tanh(flow["depth_usd"] / 5e6), "mtf_trend": (mtf + 1) / 2,
            "reg_trend": 1 if reg == "TREND" else 0, "reg_chop": 1 if reg == "CHOP" else 0,
            "funding": 0.0, "poc_dist": 0.0, "last": last
        }
    except Exception as e: log.error(f"Feature {sym}: {e}"); return None

# ---------- ML ----------
def load_ensemble():
    models = []
    for name in ["gb_model.pkl", "rf_model.pkl", "lr_model.pkl"]:
        if os.path.exists(name): models.append(joblib.load(name)); log.info(f"✓ Loaded {name}")
    return models if len(models) == 3 else None

async def ai_review_ensemble(sym, side, score):
    try:
        features = await extract_features(sym)
        if not features: return {"allow": False, "confidence": 30, "reason": "no-features"}
        last = features.pop("last")
        X = np.array([features[c] for c in FEATURE_COLS]).reshape(1, -1)
        models = load_ensemble()
        if not models: return {"allow": True, "confidence": 60, "reason": "no-ml-fallback"}
        preds = []
        for m in models:
            try: proba = m.predict_proba(X)[0]; preds.append(proba[1] if len(proba) > 1 else 0.5)
            except: preds.append(0.5)
        avg_prob = np.mean(preds)
        confidence = int(abs(avg_prob - 0.5) * 200)
        ml_side = "long" if avg_prob > 0.58 else "short" if avg_prob < 0.42 else "none"
        if ml_side != side: return {"allow": False, "confidence": confidence, "reason": "ml-disagree"}
        flow = await orderflow_analysis(sym)
        if not flow: return {"allow": False, "confidence": 40, "reason": "no-flow"}
        if flow["spread"] > 0.12: return {"allow": False, "confidence": confidence, "reason": "spread-wide"}
        if flow["depth_usd"] < 8e5: return {"allow": False, "confidence": confidence, "reason": "low-liquidity"}
        mtf = await mtf_trend(sym)
        if side == "long" and mtf < 0: return {"allow": False, "confidence": confidence, "reason": "mtf-bearish"}
        if side == "short" and mtf > 0: return {"allow": False, "confidence": confidence, "reason": "mtf-bullish"}
        if await regime(sym) == "CHOP": return {"allow": False, "confidence": confidence, "reason": "choppy"}
        if side == "long" and flow["recent_delta"] < 0: return {"allow": False, "confidence": confidence, "reason": "delta-bearish"}
        if side == "short" and flow["recent_delta"] > 0: return {"allow": False, "confidence": confidence, "reason": "delta-bullish"}
        if confidence < 65: return {"allow": False, "confidence": confidence, "reason": "low-confidence"}
        return {"allow": True, "confidence": confidence, "reason": "all-green"}
    except Exception as e: log.error(f"AI review {sym}: {e}"); return {"allow": False, "confidence": 30, "reason": f"error:{str(e)[:20]}"}

# ---------- SCORING ----------
async def calculate_advanced_score(sym):
    try:
        features = await extract_features(sym)
        if not features: return None
        last = features.get("last", 0)
        rsi_1m = features["rsi_1m"] * 100
        scores = {
            "rsi": 3.0 if rsi_1m > 70 else 7.0 if rsi_1m < 30 else 5.0,
            "momentum": 8.5 if features["mom_1m"] > 0.5 and features["mom_5m"] > 0.5 else 1.5 if features["mom_1m"] < -0.5 and features["mom_5m"] < -0.5 else 5.0,
            "orderflow": 9.0 if features["delta_momentum"] > 0.6 and (features["imbalance"] - 0.5) * 2 > 0.3 else 1.0 if features["delta_momentum"] < -0.6 and (features["imbalance"] - 0.5) * 2 < -0.3 else 5.0,
            "mtf": 9.0 if (features["mtf_trend"] * 2 - 1) > 0.8 else 1.0 if (features["mtf_trend"] * 2 - 1) < -0.8 else 5.0,
            "regime": 8.0 if features["reg_trend"] == 1 else 2.0 if features["reg_chop"] == 1 else 6.0,
            "volume": 8.0 if features["vol_ratio"] * 3 > 1.5 else 4.0 if features["vol_ratio"] * 3 < 0.7 else 6.0,
        }
        weights = {"rsi": 2.5, "momentum": 2.0, "orderflow": 3.0, "mtf": 2.5, "regime": 1.5, "volume": 1.5}
        total = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        side = "long" if total >= 7.8 else "short" if total <= 2.2 else "none"
        return {"score": round(total, 2), "side": side, "components": scores, "last": last, "features": features}
    except Exception as e: log.error(f"Scoring {sym}: {e}"); return None

# ---------- TP / SL ----------
async def calc_smart_tp_sl(sym, side, entry):
    try:
        atr_1m, atr_5m = await get_real_atr(sym, "1m", 14), await get_real_atr(sym, "5m", 14)
        atr_avg = (atr_1m + atr_5m) / 2
        reg = await regime(sym)
        sl_mult, tp_mult = (1.0, 2.5) if reg == "TREND" else (1.2, 2.0) if reg == "VOLATILE" else (0.8, 1.8)
        entry = float(entry)
        if side == "long":
            sl = entry - (atr_avg * entry * sl_mult)
            tp = entry + (entry - sl) * tp_mult
        else:
            sl = entry + (atr_avg * entry * sl_mult)
            tp = entry - (sl - entry) * tp_mult
        return round(tp, 8), round(sl, 8)
    except Exception as e: log.error(f"TP/SL {sym}: {e}"); return (entry * 1.015, entry * 0.992) if side == "long" else (entry * 0.985, entry * 1.008)

# ---------- POSITION SIZE ----------
def position_size(equity, entry, sl, risk_pct=0.8):
    try:
        risk_usd = equity * risk_pct / 100
        entry, sl = float(entry), float(sl)
        price_risk = abs(entry - sl) or entry * 0.008
        qty = risk_usd / price_risk
        max_qty = (equity * 0.2) / entry
        return round(min(qty, max_qty), 6)
    except Exception as e: log.error(f"Position size: {e}"); return 0.001

# ---------- RISK ----------
async def check_risk_limits(equity, open_positions):
    try:
        if len(open_positions) >= 3: return False, "max-positions"
        exposure = sum(abs(p["size"]) * p["entry"] for p in open_positions.values())
        if exposure > equity * 0.6: return False, "max-exposure"
        r = await redis()
        daily = float(await r.get("daily_pnl") or 0)
        if daily < -equity * 0.05: return False, "daily-loss-limit"
        return True, "ok"
    except Exception as e: log.error(f"Risk check: {e}"); return False, "error"

async def update_daily_pnl(pnl):
    try:
        r = await redis()
        current = float(await r.get("daily_pnl") or 0)
        await r.set("daily_pnl", current + pnl)
        now = datetime.utcnow()
        if now.hour == 0 and now.minute < 5: await r.set("daily_pnl", 0)
    except Exception as e: log.error(f"PnL update: {e}")

# ---------- TELEGRAM ----------
async def send_telegram(txt):
    if not CFG["tg_token"] or not CFG["tg_chat"]: return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.post(url, json={"chat_id": CFG["tg_chat"], "text": txt, "parse_mode": "HTML"}) as r:
                    if r.status == 200: return
        except Exception as e: log.warning(f"Telegram attempt {attempt+1}: {e}"); await asyncio.sleep(1)

# ---------- EXCHANGE ----------
class Exchange:
    def __init__(self):
        self.ex = ccxt.binance({
            "apiKey": CFG["key"], "secret": CFG["secret"],
            "options": {"defaultType": "future", "testnet": CFG["testnet"]},
            "enableRateLimit": True, "timeout": 30000,
        })
    async def limit(self, sym, side, qty, price, post_only=True):
        try: order = await self.ex.create_order(sym, "LIMIT", side, qty, price, params={"postOnly": post_only}); log.info(f"✓ Limit: {sym} {side} {qty} @ {price}"); return order
        except Exception as e: log.error(f"❌ Limit {sym}: {e}"); return None
    async def market(self, sym, side, qty):
        try: order = await self.ex.create_order(sym, "MARKET", side, qty); log.info(f"✓ Market: {sym} {side} {qty}"); return order
        except Exception as e: log.error(f"❌ Market {sym}: {e}"); return None
    async def set_leverage(self, sym, leverage):
        try: await self.ex.fapiPrivate_post_leverage({"symbol": sym.replace("/", ""), "leverage": leverage}); log.info(f"✓ Leverage: {sym} {leverage}x")
        except Exception as e: log.warning(f"Leverage {sym}: {e}")
    async def set_sl_tp(self, sym, side, sl, tp):
        try:
            sl_side = "SELL" if side == "long" else "BUY"
            await self.ex.create_order(sym, "STOP_MARKET", sl_side, None, params={"stopPrice": sl, "closePosition": True})
            tp_side = "SELL" if side == "long" else "BUY"
            await self.ex.create_order(sym, "TAKE_PROFIT_MARKET", tp_side, None, params={"stopPrice": tp, "closePosition": True})
            log.info(f"✓ SL/TP: {sym} SL={sl} TP={tp}")
        except Exception as e: log.error(f"❌ SL/TP {sym}: {e}")
    async def get_position(self, sym):
        try:
            positions = await self.ex.fapiPrivate_get_positionrisk({"symbol": sym.replace("/", "")})
            for p in positions:
                if float(p["positionAmt"]) != 0: return {"size": float(p["positionAmt"]), "entry": float(p["entryPrice"]), "unrealizedPnl": float(p["unRealizedProfit"]), "leverage": int(p["leverage"])}
            return None
        except Exception as e: log.error(f"Position {sym}: {e}"); return None
    async def close(self): await self.ex.close()

# ---------- CLEANUP ----------
async def cleanup(): await redis_close(); log.info("✓ Cleanup complete")
