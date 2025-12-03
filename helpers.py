"""
helpers.py — ML + LLM AI layer, regime, orderflow, dynamic TP/SL, position size
Redis-py 4.3.4 asyncio (aioredis crash fix)
"""
import os, json, asyncio, hmac, hashlib, math, logging, joblib, aiohttp
import pandas as pd, numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
import openai, ccxt.async_support as ccxt
# ---------- redis-py 4.3.4 async ----------
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

# ---------- config ----------
CFG = {
    "key": os.getenv("BINANCE_KEY"),
    "secret": os.getenv("BINANCE_SECRET"),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "testnet": os.getenv("USE_TESTNET", "false").lower() == "true",
    "tg_token": os.getenv("TELEGRAM_TOKEN"),
    "tg_chat": os.getenv("TELEGRAM_CHAT_ID"),
    "equity": float(os.getenv("EQUITY_USD", 6000)),
    "risk_perc": float(os.getenv("RISK_PERC", 1.0)),
    "max_lev": int(os.getenv("MAX_LEV", 20)),
    "pairs": json.loads(os.getenv("TOP_PAIRS", '["BTCUSDT"]')),
    "openai_key": os.getenv("OPENAI_KEY"),
}
openai.api_key = CFG["openai_key"]

# ---------- redis ----------
redis_client: Redis = None
async def redis():
    global redis_client
    if redis_client is None:
        redis_client = Redis.from_url(CFG["redis_url"], decode_responses=True)
    return redis_client

# ---------- websocket ----------
class WS:
    def __init__(self): self.url = "wss://fstream.binance.com/stream?streams="
    def build(self):
        streams = [f"{p.lower()}@ticker/{p.lower()}@depth20@100ms/{p.lower()}@trade" for p in CFG["pairs"]]
        self.url += "/".join(streams)
    async def run(self):
        self.build()
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(self.url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await self._handle(data)
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
    async def _handle(self, data):
        r = await redis()
        stream = data.get("stream", "")
        payload = data.get("data", {})
        if "@ticker" in stream:
            s = payload.get("s")
            if s: await r.hset(f"t:{s}", mapping={"last": payload.get("c"), "vol": payload.get("v"), "E": payload.get("E")})
        elif "@depth20" in stream:
            s = payload.get("s")
            bids = payload.get("bids", [])
            asks = payload.get("asks", [])
            if s and (bids or asks): await r.hset(f"d:{s}", mapping={"bids": json.dumps(bids), "asks": json.dumps(asks)})
        elif "@trade" in stream:
            s = payload.get("s")
            if s: await r.lpush(f"tr:{s}", json.dumps({"p": payload.get("p"), "q": payload.get("q"), "m": payload.get("m"), "t": payload.get("T")}))
            await r.ltrim(f"tr:{s}", 0, 199)

# ---------- regime ----------
def adx_np(close, high, low, n=14):
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    dmplus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dmminus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    tr_smooth = pd.Series(tr).rolling(n).sum(); dmplus_smooth = pd.Series(dmplus).rolling(n).sum(); dmminus_smooth = pd.Series(dmminus).rolling(n).sum()
    dx = 100 * np.abs(dmplus_smooth - dmminus_smooth) / (dmplus_smooth + dmminus_smooth + 1e-6)
    adx = dx.rolling(n).mean()
    return adx.iloc[-1] if len(adx) > 0 else 0
def atr_np(df, n=14):
    tr = np.maximum(df["h"] - df["l"], np.maximum(np.abs(df["h"] - df["c"].shift()), np.abs(df["l"] - df["c"].shift())))
    return tr.rolling(n).mean().iloc[-1] if len(tr) > n else 0.4
async def regime(sym):
    r = await redis()
    h = await r.hget(f"ohlcv_1h:{sym}", "data")
    if not h: return "CHOP"
    df = pd.read_json(h)
    adx = adx_np(df["c"], df["h"], df["l"])
    atr = atr_np(df)
    if adx > 25 and atr > np.median(df["atr"].dropna()[-24:]): return "TREND"
    if adx < 20: return "CHOP"
    return "VOLATILE"

# ---------- features ----------
async def features(sym):
    r = await redis()
    last = float(await r.hget(f"t:{sym}", "last") or 0)
    trades = [json.loads(x) for x in await r.lrange(f"tr:{sym}", 0, 199)]
    depth_raw = await r.hgetall(f"d:{sym}")
    if not depth_raw: return None
    bids = json.loads(depth_raw.get("bids", "[]"))
    asks = json.loads(depth_raw.get("asks", "[]"))
    # 1-s ohlcv
    df_trades = pd.DataFrame(trades, columns=["p","q","m","t"])
    df_trades["p"] = df_trades["p"].astype(float)
    ohlc = df_trades["p"].resample("1s").ohlc().ffill()
    close = ohlc["close"].dropna().values
    # indicators
    rsi = 50 if len(close) < 14 else 100 - 100/(1+(close[-14:].diff().clip(lower=0).mean()/(close[-14:].diff().clip(upper=0).abs().mean()+1e-6)))
    macd, macdsig = (0,0) if len(close) < 26 else (
        close.ewm(span=12).mean().iloc[-1] - close.ewm(span=26).mean().iloc[-1],
        (close.ewm(span=12).mean() - close.ewm(span=26).mean()).ewm(span=9).mean().iloc[-1]
    )
    # orderflow
    imb = (float(bids[0][1]) - float(asks[0][1])) / (float(bids[0][1]) + float(asks[0][1]) + 1e-6) if bids and asks else 0
    delta = sum(float(t["q"]) if t["m"] == "false" else -float(t["q"]) for t in trades[-50:])
    sweep = int(any(float(t["q"]) > 50 and float(t["q"]) > 2 * np.mean([float(x["q"]) for x in trades[-100:]]) for t in trades[-10:]))
    # micro
    spread = (float(asks[0][0]) - float(bids[0][0])) / last * 100 if bids and asks else 0.1
    depth_usd = sum(float(b[0]) * float(b[1]) for b in bids[:5]) + sum(float(a[0]) * float(a[1]) for a in asks[:5])
    return dict(last=last, rsi=rsi, macd=macd, macdsig=macdsig, imb=imb, delta=delta, sweep=sweep, spread=spread, depth_usd=depth_usd)

# ---------- ML model ----------
MODEL_PATH = "ml_model.pkl"
FEATURE_COLS = ["rsi", "macd_slope", "imb", "delta_norm", "sweep", "spread_atr", "depth_norm", "btc_1m", "reg_trend", "reg_chop"]
def build_vector(f, btc_change, regime):
    macd_slope = np.tanh(f["macd"] - f["macdsig"])
    delta_norm = np.tanh(f["delta"] / 1e6)
    spread_atr = f["spread"] / 0.5
    depth_norm = np.tanh(f["depth_usd"] / 5e6)
    reg_trend = 1 if regime == "TREND" else 0
    reg_chop  = 1 if regime == "CHOP" else 0
    return np.array([f["rsi"]/100, macd_slope, (f["imb"]+1)/2, delta_norm, float(f["sweep"]),
                     spread_atr, depth_norm, btc_change, reg_trend, reg_chop]).reshape(1, -1)
async def ai_review_ml(sym, mode, direction, score, triggers, spread_ok, depth_ok, btc_calm):
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if not model: return {"allow": True, "confidence": 80, "reason": "no-ml"}
    r = await redis()
    f = await features(sym)
    if not f: return {"allow": False, "confidence": 40, "reason": "no-features"}
    btc_change = float(await r.get("btc_1m_change") or 0)
    regime_val = await regime(sym)
    X = build_vector(f, btc_change, regime_val)
    prob_long, prob_short = model.predict_proba(X)[0]
    confidence = int(max(prob_long, prob_short) * 100)
    ai_side = "long" if prob_long > 0.58 else "short" if prob_short > 0.58 else "none"
    if ai_side != direction: return {"allow": False, "confidence": confidence, "reason": "ml-disagree"}
    if not CFG["openai_key"]: return {"allow": True, "confidence": confidence, "reason": "ml-ok"}
    try:
        prompt = (f"Crypto quant AI. Return JSON:{{allow(bool),confidence(0-100),reason(str)}}.\n"
                  f"Symbol:{sym} Mode:{mode} Direction:{direction} Score:{score}/10 MLconf:{confidence}\n"
                  f"SpreadOK:{spread_ok} DepthOK:{depth_ok} BTCcalm:{btc_calm}\nTriggers:\n{triggers}\n"
                  f"Approve only if MLconf>=60,SpreadOK,DepthOK,BTCcalm.")
        res = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}],
                                                  temperature=0, max_tokens=60)
        llm = json.loads(res.choices[0].message.content.strip())
        return {"allow": llm["allow"] and confidence >= 60, "confidence": min(confidence, llm["confidence"]),
                "reason": f"ml:{confidence} llm:{llm['reason']}"}
    except: return {"allow": True, "confidence": confidence, "reason": "ml-ok"}

# ---------- TP/SL + size ----------
def calc_tp_sl(entry, atr, side):
    entry = float(entry)
    if side == "long":
        sl = entry - 0.8 * atr
        tp = entry + 1.5 * (entry - sl)
    else:
        sl = entry + 0.8 * atr
        tp = entry - 1.5 * (sl - entry)
    return round(tp, 6), round(sl, 6)
def position_size(equity, entry, sl):
    risk_usd = equity * CFG["risk_perc"] / 100
    qty = risk_usd / abs(entry - sl)
    return qty

# ---------- telegram ----------
async def send_telegram(txt):
    if not CFG["tg_token"]: return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, json={"chat_id": CFG["tg_chat"], "text": txt, "parse_mode": "HTML"})

# ---------- exchange ----------
class Exchange:
    def __init__(self):
        self.ex = ccxt.binance({
            "apiKey": CFG["key"], "secret": CFG["secret"],
            "options": {"defaultType": "future", "testnet": CFG["testnet"]},
            "enableRateLimit": True,
        })
    async def limit(self, sym, side, qty, price, post_only=True):
        return await self.ex.create_order(sym, "LIMIT", side, qty, price, params={"postOnly": post_only})
    async def market(self, sym, side, qty):
        return await self.ex.create_order(sym, "MARKET", side, qty)
    async def close(self): await self.ex.close()

# ---------- bot loop (exported) ----------
async def run():
    """background bot loop (called by main.py)"""
    ws = WS()
    asyncio.create_task(ws.run())          # websocket feed
    await asyncio.sleep(3)                 # let data fill
    ex = Exchange()
    while True:
        for sym in CFG["pairs"]:
            f = await features(sym)
            if not f: continue
            regime_val = await regime(sym)
            score = round((max(f["rsi"], 50) - 50) / 5 + (f["imb"] + 1) * 2 + f["sweep"], 1)
            side = "long" if score >= 7.5 else "short" if score <= 3.0 else "none"
            if side == "none": continue
            atr = 0.4  # dummy ATR (replace with redis ohlcv)
            tp, sl = calc_tp_sl(f["last"], atr, side)
            ai = await ai_review_ml(sym, "QUICK", side, score, f"rsi={f['rsi']:.1f},imb={f['imb']:.2f}", f["spread"] < 0.1, f["depth_usd"] > 1e6, True)
            if not ai["allow"]: continue
            qty = position_size(CFG["equity"], f["last"], sl)
            msg = f"🎯 <b>{sym}</b> {side.upper()}  score={score}  qty={qty:.3f}  TP={tp}  SL={sl}  conf={ai['confidence']}%"
            logging.info(msg); await send_telegram(msg)
            # paper-limit order (uncomment for live)
            # await ex.limit(sym, side, qty, f["last"])
        await asyncio.sleep(5)
