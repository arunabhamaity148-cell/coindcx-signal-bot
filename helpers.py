import os, json, asyncio, logging, aiohttp, joblib
import pandas as pd, numpy as np
from datetime import datetime, timedelta
from redis.asyncio import Redis
import ccxt.async_support as ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

# --------------------  80 PAIR  --------------------
TOP_PAIRS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","MATICUSDT",
    "DOTUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","LTCUSDT","UNIUSDT","ETCUSDT","FILUSDT",
    "TRXUSDT","NEARUSDT","ARBUSDT","APTUSDT","INJUSDT","STXUSDT","TIAUSDT","SEIUSDT",
    "OPUSDT","SUIUSDT","FETUSDT","RENDERUSDT","IMXUSDT","RUNEUSDT","XLMUSDT","ALGOUSDT",
    "SANDUSDT","ICPUSDT","GRTUSDT","AAVEUSDT","LDOUSDT","HBARUSDT","FTMUSDT","VETUSDT",
    "MANAUSDT","AXSUSDT","THETAUSDT","FLOWUSDT","SNXUSDT","CHZUSDT","ENJUSDT","MKRUSDT",
    "COMPUSDT","KSMUSDT","XTZUSDT","ZECUSDT","DASHUSDT","BATUSDT","ZILUSDT","ONTUSDT",
    "IOSTUSDT","IOTAUSDT","QTUMUSDT","WAVESUSDT","ZRXUSDT","OMGUSDT","CRVUSDT","YFIUSDT",
    "BALUSDT","1INCHUSDT","RLCUSDT","KAVAUSDT","SUSHIUSDT","OCEANUSDT","RSRUSDT","CELOUSDT",
    "BANDUSDT","STORJUSDT","CELRUSDT","SKLUSDT","ANKRUSDT","BLZUSDT","ARPAUSDT","NMRUSDT"
]

# --------------------  CONFIG  --------------------
CFG = {
    "key": os.getenv("COINDCX_KEY", ""),
    "secret": os.getenv("COINDCX_SECRET", ""),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "tg_token": os.getenv("TELEGRAM_TOKEN"),
    "tg_chat": os.getenv("TELEGRAM_CHAT_ID"),
    "equity": float(os.getenv("EQUITY_USD", 30000)),
    "risk_perc": float(os.getenv("RISK_PERC", 0.7)),
    "min_lev": int(os.getenv("MIN_LEV", 15)),
    "max_lev": int(os.getenv("MAX_LEV", 30)),
    "liq_buffer": float(os.getenv("LIQ_BUFFER", 0.15)),
    "cooldown_min": int(os.getenv("COOLDOWN_MIN", 30)),
    "pairs": TOP_PAIRS,
}

# --------------------  STRATEGY CONFIG  --------------------
STRATEGY_CONFIG = {
    "QUICK":  {"min_score": 6.2, "tp1_mult": 1.1, "tp2_mult": 1.6, "sl_mult": 1.0, "tp1_exit": 0.6},
    "MID":    {"min_score": 6.5, "tp1_mult": 1.4, "tp2_mult": 2.2, "sl_mult": 1.0, "tp1_exit": 0.5},
    "TREND":  {"min_score": 7.0, "tp1_mult": 1.8, "tp2_mult": 2.8, "sl_mult": 1.2, "tp1_exit": 0.4}
}

# --------------------  REDIS  --------------------
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

# --------------------  EXCHANGE  --------------------
def get_exchange(**kwargs):
    cls = getattr(ccxt, "coindcx", None)
    if cls: return cls({"enableRateLimit": True, **kwargs})
    return ccxt.Exchange({
        "id": "coindcx", "name": "CoinDCX", "enableRateLimit": True, "timeout": 30000, **kwargs
    })
class Exchange:
    def __init__(self):
        self.ex = get_exchange(apiKey=CFG["key"], secret=CFG["secret"])
        log.info("✓ CoinDCX Exchange Auth Initialized")
    async def fetch_balance(self):
        try: return await self.ex.fetch_balance()
        except Exception as e: log.error(f"balance error: {e}"); return None

# --------------------  BLOCK  --------------------
def dbg(reason, sym, strategy):
    log.warning(f"⛔ BLOCK [{sym}] [{strategy}] → {reason}")

# --------------------  OHLC  --------------------
async def build_ohlcv(sym, tf="1m", bars=100):
    r = await redis()
    raw = await r.lrange(f"tr:{sym}", 0, 600)
    if not raw: return None
    trades = [json.loads(x) for x in raw]
    df = pd.DataFrame(trades)
    df["p"] = df["p"].astype(float); df["q"] = df["q"].astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").sort_index()
    ohlc = df["p"].resample(tf).ohlc(); vol = df["q"].resample(tf).sum()
    o = ohlc.join(vol.rename("v")).dropna()
    return o.tail(bars)
async def get_ohlcv(sym, tf="1m", limit=100):
    df = await build_ohlcv(sym, tf, limit)
    if df is None or len(df) < 20: return None
    df = df.reset_index(); df.rename(columns={"open":"o","high":"h","low":"l","close":"c"}, inplace=True)
    return df
def calc_rsi(series, period=14):
    delta = series.diff(); gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean(); rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))
async def get_atr(sym, tf="5m", period=14):
    df = await get_ohlcv(sym, tf, 100)
    if df is None: return 0.002
    h, l, c = df["h"], df["l"], df["c"]; prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum(abs(h - prev_c), abs(l - prev_c)))
    return tr.rolling(period).mean().iloc[-1]

# --------------------  CVD + VWAP  --------------------
async def cvd_vwap(sym):
    r = await redis()
    raw = await r.lrange(f"tr:{sym}", 0, 200)
    if not raw: return None
    trades = [json.loads(x) for x in raw]
    df = pd.DataFrame(trades)
    df["p"] = df["p"].astype(float); df["q"] = df["q"].astype(float)
    df["side"] = np.where(df["m"], -1, 1)
    df["vol"] = df["q"] * df["p"]; df["pv"] = df["p"] * df["vol"]
    vwap = df["pv"].sum() / (df["vol"].sum() + 1e-9)
    cvd  = (df["q"] * df["side"]).sum()
    return {"cvd": cvd, "vwap": vwap, "price": df["p"].iloc[-1]}

# --------------------  ORDERFLOW  --------------------
async def orderflow(sym):
    try:
        r = await redis()
        raw = await r.lrange(f"tr:{sym}", 0, 200)
        if not raw: return None
        t = [json.loads(x) for x in raw]
        delta = sum(float(x["q"]) if not x["m"] else -float(x["q"]) for x in t)
        recent = sum(float(x["q"]) if not x["m"] else -float(x["q"]) for x in t[:40])
        volumes = [float(x["q"]) for x in t]; avg, std = np.mean(volumes), np.std(volumes)
        buys = sum(1 for x in t[:20] if not x["m"] and float(x["q"]) > avg + 2*std)
        sells = sum(1 for x in t[:20] if x["m"] and float(x["q"]) > avg + 2*std)
        depth_raw = await r.hgetall(f"d:{sym}")
        bids = json.loads(depth_raw.get("bids", "[]")) if depth_raw else []
        asks = json.loads(depth_raw.get("asks", "[]")) if depth_raw else []
        if not bids or not asks: return None
        bid_vol = sum(float(b[1]) for b in bids[:5]); ask_vol = sum(float(a[1]) for a in asks[:5])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 100
        depth_usd = sum(float(b[0])*float(b[1]) for b in bids[:10]) + sum(float(a[0])*float(a[1]) for a in asks[:10])
        return {"delta": delta, "recent_delta": recent, "imbalance": imbalance, "spread": spread,
                "buys": buys, "sells": sells, "depth_usd": depth_usd, "momentum": recent/(abs(delta)+1e-6)}
    except Exception as e: log.error(e); return None

# --------------------  MTF  --------------------
async def mtf(sym):
    t = []
    for tf in ["1m", "5m"]:
        df = await get_ohlcv(sym, tf, 60)
        if df is None: continue
        ema20 = df["c"].ewm(span=20).mean().iloc[-1]; ema50 = df["c"].ewm(span=50).mean().iloc[-1]
        t.append(1 if ema20 > ema50 else -1)
    if not t: return 0
    x = sum(t)/len(t); return 1 if x>0.4 else -1 if x<-0.4 else 0

# --------------------  NEWS / SENTIMENT  --------------------
async def news_guard(sym):
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://api.alternative.me/fng/?limit=1") as r:
                fg_val = int((await r.json())["data"][0]["value"])
        fg_score = 0
        if fg_val <= 20: fg_score = +1
        elif fg_val >= 80: fg_score = -1
        lc_key = os.getenv("LUNARCRUSH_KEY", "")
        lc_score = 0
        if lc_key and sym.replace("USDT", "") in {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX"}:
            url = f"https://lunarcrush.com/api4/coin/{sym.replace('USDT', '')}/metrics"
            headers = {"Authorization": f"Bearer {lc_key}"}
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        sent = (await resp.json()).get("data", {}).get("sentiment", 0)
                        if sent > 0.6: lc_score = +1
                        elif sent < -0.6: lc_score = -1
        total = fg_score + lc_score
        return {"score": np.clip(total, -2, 2)}
    except Exception as e:
        log.debug(f"news-guard err {e}"); return {"score": 0}

# --------------------  DYNAMIC PAIR FILTER  --------------------
async def filtered_pairs():
    r = await redis(); good = []
    for sym in TOP_PAIRS:
        try:
            ticker = await r.hgetall(f"t:{sym}")
            if not ticker: continue
            last = float(ticker.get("last", 0))
            vol24 = float(ticker.get("quoteVolume", 0))
            depth = await r.hgetall(f"d:{sym}")
            if not depth: continue
            bids = json.loads(depth.get("bids", "[]")); asks = json.loads(depth.get("asks", "[]"))
            if not bids or not asks: continue
            spread = (float(asks[0][0]) - float(bids[0][0])) / last * 100
            if vol24 >= 1_500_000 and spread <= 1.0: good.append(sym)
        except: continue
    return good[:30]

# --------------------  SCORE  --------------------
async def calculate_advanced_score(sym, strategy):
    try:
        df = await get_ohlcv(sym, "1m", 120)
        if df is None: return None
        last = df["c"].iloc[-1]; rsi = calc_rsi(df["c"]).iloc[-1]
        mom = (last - df["c"].iloc[-6]) / df["c"].iloc[-6]
        flow = await orderflow(sym)
        if not flow: return None
        cv = await cvd_vwap(sym)
        if not cv: return None
        trend = await mtf(sym); ng = await news_guard(sym)
        score = 0
        score += (50 - abs(rsi - 50)) / 25 * 2
        score += mom * 4
        score += flow["imbalance"] * 5
        score += flow["momentum"] * 3
        score += trend * 4
        score += ng["score"] * 3
        score += (1 if cv["price"] > cv["vwap"] * 0.999 else -1) * 2
        score += (1 if cv["cvd"] > 0 else -1) * 2
        if flow["spread"] > 0.3: return None
        if flow["depth_usd"] < 150000: return None
        side = "long" if score >= STRATEGY_CONFIG[strategy]["min_score"] else "none"
        return {"side": side, "score": score, "last": float(last)}
    except Exception as e:
        log.error(f"score error {sym}: {e}"); return None

# --------------------  TP/SL  --------------------
async def calc_tp_sl(sym, side, entry, strategy):
    atr = await get_atr(sym, "5m", 14)
    cfg = STRATEGY_CONFIG[strategy]
    lev = CFG["min_lev"]
    sl_dist = atr * entry * cfg["sl_mult"]
    liq = entry * (1 - 1 / lev) if side == "long" else entry * (1 + 1 / lev)
    if side == "long":
        sl = entry - sl_dist; tp1 = entry + sl_dist * cfg["tp1_mult"]; tp2 = entry + sl_dist * cfg["tp2_mult"]
    else:
        sl = entry + sl_dist; tp1 = entry - sl_dist * cfg["tp1_mult"]; tp2 = entry - sl_dist * cfg["tp2_mult"]
    liq_dist = abs(sl - liq) / liq * 100
    return tp1, tp2, sl, lev, liq, liq_dist

# --------------------  ICEBERG  --------------------
def iceberg_size(equity, entry, sl, lev):
    risk = equity * (CFG["risk_perc"] / 100)
    dist = abs(entry - sl)
    qty = (risk / dist) / lev
    iq = qty / 4
    return {"total": qty, "each": iq, "orders": 4}

# --------------------  TELEGRAM  --------------------
async def send_telegram(text):
    if not CFG["tg_token"] or not CFG["tg_chat"]: return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    async with aiohttp.ClientSession() as s:
        await s.post(url, json={"chat_id": CFG["tg_chat"], "text": text, "parse_mode": "HTML"})

log.info("✓ helpers loaded (FULL)")
