# helpers.py — Auth + Polling + Logic + Risk + Telegram helpers
import os, json, asyncio, logging, aiohttp
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("helpers")

# ----------------------
# CONFIG from ENV
# ----------------------
CFG = {
    "key": os.getenv("COINDCX_KEY", ""),
    "secret": os.getenv("COINDCX_SECRET", ""),
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "tg_token": os.getenv("TELEGRAM_TOKEN", ""),
    "tg_chat": os.getenv("TELEGRAM_CHAT_ID", ""),
    "equity": float(os.getenv("EQUITY_USD", 30000)),
    "min_lev": int(os.getenv("MIN_LEV", 15)),
    "max_lev": int(os.getenv("MAX_LEV", 30)),
    "liq_buffer": float(os.getenv("LIQ_BUFFER", 0.15)),  # 15% safety from theoretical liq distance
    "cooldown_min": int(os.getenv("COOLDOWN_MIN", 30)),
    "top_pairs"= [
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
 json.loads(os.getenv("TOP_PAIRS", "[]")),
    "liq_alert_pct": float(os.getenv("LIQ_ALERT_PCT", 0.7)),  # alert if SL->LIQ distance % < this
    "daily_max_signals": int(os.getenv("DAILY_MAX_SIGNALS", 30)),
}

# ----------------------
# Redis client
# ----------------------
_redis = None
async def redis():
    global _redis
    if _redis is None:
        _redis = Redis.from_url(CFG["redis_url"], decode_responses=True)
        await _redis.ping()
        log.info("✓ Redis connected")
    return _redis

async def redis_close():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None

# ----------------------
# CCXT Exchange factory (CoinDCX)
# ----------------------
def get_exchange(apiKey=None, secret=None):
    # uses ccxt.coindcx if available, else generic Exchange with custom urls (simple)
    try:
        cls = getattr(ccxt, "coindcx", None)
        if cls:
            ex = cls({
                "apiKey": apiKey or CFG["key"],
                "secret": secret or CFG["secret"],
                "enableRateLimit": True,
                "timeout": 30000
            })
            return ex
    except Exception:
        pass
    # fallback basic Exchange
    ex = ccxt.Exchange({
        "id": "coindcx",
        "name": "coindcx",
        "urls": {"api": {"public": "https://api.coindcx.com/exchange/v1", "private": "https://api.coindcx.com/exchange/v1"}},
        "api": {"public": {"get": ["markets", "tickers", "orderbook", "trades"]}, "private": {"get": ["orders", "balances"], "post": ["order/new", "order/cancel"]}},
        "enableRateLimit": True,
        "timeout": 30000
    })
    # attach keys if provided
    if apiKey:
        ex.apiKey = apiKey
    if secret:
        ex.secret = secret
    return ex

# ----------------------
# Authenticated Exchange wrapper for create/cancel/fetch_balance
# ----------------------
class Exchange:
    def __init__(self):
        self.ex = get_exchange(CFG["key"], CFG["secret"])
        log.info("✓ Exchange wrapper initialized")

    async def close(self):
        try:
            await self.ex.close()
        except Exception:
            pass

    async def fetch_balance(self):
        try:
            return await self.ex.fetch_balance()
        except Exception as e:
            log.error(f"fetch_balance error: {e}")
            return None

    async def create_limit_order(self, symbol, side, amount, price, params=None):
        try:
            return await self.ex.create_limit_order(symbol, side, amount, price, params or {})
        except Exception as e:
            log.error(f"create_limit_order error {symbol} {side} {price} {amount}: {e}")
            return None

    async def create_market_order(self, symbol, side, amount, params=None):
        try:
            return await self.ex.create_market_order(symbol, side, amount, params or {})
        except Exception as e:
            log.error(f"create_market_order error {symbol} {side} {amount}: {e}")
            return None

    async def cancel_order(self, id, symbol=None):
        try:
            return await self.ex.cancel_order(id, symbol)
        except Exception as e:
            log.error(f"cancel_order error: {e}")
            return None

# ----------------------
# WS Poller (fetch ticker/orderbook/trades -> redis)
# ----------------------
class WS:
    def __init__(self):
        self.running = False
        self.ex = get_exchange()

    async def run(self):
        self.running = True
        log.info("🔌 Starting CoinDCX data polling...")
        r = await redis()
        while self.running:
            try:
                for sym in CFG["top_pairs"]:
                    try:
                        ticker = await self.ex.fetch_ticker(sym)
                        orderbook = await self.ex.fetch_order_book(sym, limit=20)
                        trades = await self.ex.fetch_trades(sym, limit=200)
                        # safe writes
                        last = float(ticker.get("last") or 0)
                        base_vol = ticker.get("baseVolume", 0)
                        await r.hset(f"t:{sym}", mapping={"last": last, "vol": base_vol, "E": int(datetime.utcnow().timestamp()*1000)})
                        bids = orderbook.get("bids", []) if isinstance(orderbook, dict) else []
                        asks = orderbook.get("asks", []) if isinstance(orderbook, dict) else []
                        await r.hset(f"d:{sym}", mapping={"bids": json.dumps(bids[:20]), "asks": json.dumps(asks[:20]), "E": int(datetime.utcnow().timestamp()*1000)})
                        # normalize and push trades
                        if trades:
                            for t in trades[-200:]:
                                try:
                                    p = t.get("price") or t.get("p") or 0.0
                                    q = t.get("amount") or t.get("q") or 0.0
                                    maker = t.get("maker_side") or t.get("side") or ""
                                    is_sell = True if str(maker).lower() in ["sell", "ask", "true", "maker"] else False
                                    ts = None
                                    for k in ("timestamp", "time", "ts", "trade_timestamp"):
                                        if t.get(k) is not None:
                                            ts = t.get(k); break
                                    if ts is None:
                                        ts = int(datetime.utcnow().timestamp()*1000)
                                    p = float(p); q = float(q); ts = int(ts)
                                    await r.lpush(f"tr:{sym}", json.dumps({"p": p, "q": q, "m": is_sell, "t": ts}))
                                except Exception:
                                    continue
                            await r.ltrim(f"tr:{sym}", 0, 499)
                    except Exception as e:
                        log.debug(f"Poll {sym}: {e}")
                        continue
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.running = False
                break
            except Exception as e:
                log.error(f"WS loop error: {e}")
                await asyncio.sleep(3)

# ----------------------
# Build OHLCV from trades
# ----------------------
async def build_ohlcv_from_trades(sym, timeframe='1m', bars=100):
    r = await redis()
    trades_raw = await r.lrange(f"tr:{sym}", 0, bars*80)
    if not trades_raw:
        return None
    trades = []
    for raw in trades_raw:
        try:
            t = json.loads(raw)
            if "p" not in t or "q" not in t or "t" not in t:
                continue
            trades.append(t)
        except Exception:
            continue
    if not trades or len(trades) < 20:
        return None
    df = pd.DataFrame(trades)
    df['p'] = pd.to_numeric(df['p'], errors='coerce').fillna(0.0)
    df['q'] = pd.to_numeric(df['q'], errors='coerce').fillna(0.0)
    # normalize timestamps (ms)
    def to_ms(x):
        try:
            xi = int(x)
            if xi < 1e11:
                xi = xi * 1000
            return xi
        except:
            return int(datetime.utcnow().timestamp()*1000)
    df['t'] = df['t'].apply(to_ms)
    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df = df.set_index('t').sort_index()
    ohlc = df['p'].resample(timeframe).ohlc()
    vol = df['q'].resample(timeframe).sum()
    out = ohlc.join(vol.rename('v')).dropna()
    if out.empty:
        return None
    out.reset_index(inplace=True)
    out.rename(columns={'open':'o','high':'h','low':'l','close':'c','v':'v'}, inplace=True)
    return out.tail(bars) if len(out) >= bars else None

# ----------------------
# RSI helper
# ----------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# ----------------------
# Orderflow analysis
# ----------------------
async def orderflow_analysis(sym):
    try:
        r = await redis()
        trades_raw = await r.lrange(f"tr:{sym}", 0, 300)
        if not trades_raw:
            return None
        trades = [json.loads(t) for t in trades_raw]
        delta = sum(float(t["q"]) if not t["m"] else -float(t["q"]) for t in trades)
        recent_delta = sum(float(t["q"]) if not t["m"] else -float(t["q"]) for t in trades[:50])
        volumes = [float(t["q"]) for t in trades]
        avg_vol = np.mean(volumes) if volumes else 0
        std_vol = np.std(volumes) if volumes else 0
        large_buys = sum(1 for t in trades[:50] if not t["m"] and float(t["q"]) > avg_vol + 2*std_vol)
        large_sells = sum(1 for t in trades[:50] if t["m"] and float(t["q"]) > avg_vol + 2*std_vol)
        depth_raw = await r.hgetall(f"d:{sym}") or {}
        bids = json.loads(depth_raw.get("bids", "[]"))
        asks = json.loads(depth_raw.get("asks", "[]"))
        if not bids or not asks:
            return None
        bid_vol = sum(float(b[1]) for b in bids[:5])
        ask_vol = sum(float(a[1]) for a in asks[:5])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0]) * 100
        depth_usd = sum(float(b[0])*float(b[1]) for b in bids[:10]) + sum(float(a[0])*float(a[1]) for a in asks[:10])
        return {"delta": delta, "recent_delta": recent_delta, "large_buys": large_buys, "large_sells": large_sells, "imbalance": imbalance, "spread": spread, "depth_usd": depth_usd}
    except Exception as e:
        log.error(f"Orderflow error {sym}: {e}")
        return None

# ----------------------
# Pure-logic scoring (Aggressive thresholds)
# ----------------------
async def calc_score(sym):
    """
    returns dict: {"side":"long"/"short"/"none","score":float,"last":price,"reason":str}
    """
    try:
        df1 = await build_ohlcv_from_trades(sym, "1m", 80)
        df5 = await build_ohlcv_from_trades(sym, "5m", 80)
        if df1 is None or df5 is None or len(df1) < 20 or len(df5) < 20:
            return {"side":"none","reason":"no-ohlcv"}
        last = float(df1['c'].iloc[-1])
        rsi1 = calc_rsi(df1['c'], 14).iloc[-1]
        ema20 = df1['c'].ewm(span=20).mean().iloc[-1]
        ema50 = df1['c'].ewm(span=50).mean().iloc[-1]
        trend = "long" if ema20 > ema50 else "short"
        flow = await orderflow_analysis(sym)
        if not flow:
            return {"side":"none","reason":"no-flow"}
        score = 0.0
        # trend weight
        score += 1.5
        # RSI mid-range bonus
        if 42 < rsi1 < 68:
            score += 1.2
        # recent delta large
        if abs(flow["recent_delta"]) > 20:
            score += 1.2
        # imbalance
        if abs(flow["imbalance"]) > 0.10:
            score += 0.9
        # spread
        if flow["spread"] < 0.25:
            score += 1.0
        else:
            return {"side":"none","reason":f"spread-high-{flow['spread']:.2f}"}
        # depth guard
        if flow["depth_usd"] < 150000:
            return {"side":"none","reason":"low-depth"}
        # final threshold for aggressive mode ~6.0
        if score < 6.0:
            return {"side":"none","reason":f"low-score-{score:.2f}"}
        return {"side":trend,"score":round(score,2),"last":last,"reason":"ok"}
    except Exception as e:
        log.error(f"calc_score error {sym}: {e}")
        return {"side":"none","reason":"error"}

# ----------------------
# TP/TP2/SL/LIQ calculation (safe)
# ----------------------
async def calc_tp1_tp2_sl_liq(sym, side, entry, confidence=60, strategy="QUICK"):
    try:
        entry = float(entry)
        # approximate leverage suggestion from confidence
        lev = CFG["min_lev"] + int(max(0, min( (confidence-50)/50, 1.0)) * (CFG["max_lev"] - CFG["min_lev"]))
        lev = max(CFG["min_lev"], min(lev, CFG["max_lev"]))
        # use ATR if available fallback
        atr = 0.002
        try:
            df5 = await build_ohlcv_from_trades(sym, "5m", 50)
            if df5 is not None and len(df5) > 20:
                tr = (df5['h'] - df5['l']).abs()
                atr_val = tr.rolling(14).mean().iloc[-1]
                if atr_val and atr_val > 0:
                    atr = min(max(0.0005, atr_val/df5['c'].iloc[-1]), 0.05)
        except Exception:
            pass
        # strategy multipliers (sane defaults)
        s = {"QUICK":{"sl_mult":0.8,"tp1_mult":1.2,"tp2_mult":1.8},
             "MID":{"sl_mult":1.0,"tp1_mult":1.5,"tp2_mult":2.5},
             "TREND":{"sl_mult":1.2,"tp1_mult":2.0,"tp2_mult":3.5}}
        cfg = s.get(strategy, s["MID"])
        base_sl_dist = atr * entry * cfg["sl_mult"]
        liq_pct = 1.0/lev
        max_sl_dist = entry * liq_pct * (1 - CFG["liq_buffer"])
        sl_dist = min(base_sl_dist, max_sl_dist)
        if side == "long":
            sl = entry - sl_dist
            tp1 = entry + sl_dist * cfg["tp1_mult"]
            tp2 = entry + sl_dist * cfg["tp2_mult"]
            liq_price = entry * (1 - 1.0/lev)
        else:
            sl = entry + sl_dist
            tp1 = entry - sl_dist * cfg["tp1_mult"]
            tp2 = entry - sl_dist * cfg["tp2_mult"]
            liq_price = entry * (1 + 1.0/lev)
        # ensure SL not closer than liq (adjust)
        if side == "long" and sl <= liq_price:
            sl = liq_price * (1 + 0.02)
        if side == "short" and sl >= liq_price:
            sl = liq_price * (1 - 0.02)
        liq_dist_pct_from_sl = abs((sl - liq_price) / liq_price) * 100
        return round(tp1,8), round(tp2,8), round(sl,8), lev, round(liq_price,8), round(liq_dist_pct_from_sl,3)
    except Exception as e:
        log.error(f"calc_tp1_tp2_sl_liq error {e}")
        # fallback conservative
        if side=="long":
            return entry*1.01, entry*1.02, entry*0.99, CFG["min_lev"], entry*0.9, 10.0
        return entry*0.99, entry*0.98, entry*1.01, CFG["min_lev"], entry*1.1, 10.0

# ----------------------
# Position sizing (iceberg)
# ----------------------
def position_size_iceberg(equity, entry, sl, leverage):
    try:
        risk_usd = float(equity) * 0.01  # 1% default risk per trade
        price_risk = abs(float(entry) - float(sl))
        if price_risk <= 0:
            return {"total_qty":0, "iceberg_qty":0, "num_orders":0}
        qty = (risk_usd / price_risk) / leverage
        max_qty = (float(equity) * 0.3) / float(entry)
        total_qty = min(qty, max_qty)
        num_orders = 4
        iceberg_qty = round(max(0.0001, float(total_qty) / num_orders), 6)
        return {"total_qty": round(total_qty,6), "iceberg_qty": iceberg_qty, "num_orders": num_orders}
    except Exception as e:
        log.error(f"position_size_iceberg error: {e}")
        return {"total_qty":0.0005, "iceberg_qty":0.0002, "num_orders":3}

# ----------------------
# Cooldown & daily counters
# ----------------------
async def check_cooldown(sym, strategy="GLOBAL"):
    r = await redis()
    key = f"cd:{sym}:{strategy}"
    last = await r.get(key)
    if last:
        elapsed = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds()/60
        if elapsed < CFG["cooldown_min"]:
            return False, f"cooldown-{int(CFG['cooldown_min']-elapsed)}min"
    return True, "ok"

async def set_cooldown(sym, strategy="GLOBAL"):
    r = await redis()
    await r.set(f"cd:{sym}:{strategy}", datetime.utcnow().isoformat(), ex=CFG["cooldown_min"]*60)

async def check_daily_signal_limit():
    try:
        r = await redis()
        count = await r.get("daily_signal_count")
        if count and int(count) >= CFG["daily_max_signals"]:
            return False, "daily-limit"
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

# ----------------------
# Telegram helpers
# ----------------------
async def send_telegram(txt):
    if not CFG["tg_token"] or not CFG["tg_chat"]:
        log.debug("Telegram not configured")
        return
    url = f"https://api.telegram.org/bot{CFG['tg_token']}/sendMessage"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            await s.post(url, json={"chat_id": CFG["tg_chat"], "text": txt, "parse_mode":"HTML", "disable_web_page_preview": True})
    except Exception as e:
        log.error(f"Telegram send error: {e}")

async def send_signal_telegram(sym, strategy, side, entry, tp1, tp2, sl, lev, iceberg_info=None, liq_dist_pct=None, extra=""):
    be_sl = entry if side=="long" else entry
    iceberg_txt = ""
    if iceberg_info:
        iceberg_txt = f"Iceberg: {iceberg_info['num_orders']} × {iceberg_info['iceberg_qty']:.6f} {sym.replace('USDT','')}\n"
    liq_warn = ""
    if liq_dist_pct is not None and liq_dist_pct < CFG["liq_alert_pct"]:
        liq_warn = f"⚠️ <b>LIQ DISTANCE CLOSE:</b> {liq_dist_pct:.2f}% — consider widen SL or skip\n"
    msg = (
        f"🎯 <b>[{strategy}] {sym} {side.upper()}</b>\n"
        f"Entry: <code>{entry}</code>\n"
        f"TP1: <code>{tp1}</code>\n"
        f"TP2: <code>{tp2}</code>\n"
        f"SL: <code>{sl}</code>\n"
        f"Lev: {lev}x\n"
        f"{iceberg_txt}"
        f"BE-SL suggestion: <code>{be_sl:.8f}</code>\n"
        f"{liq_warn}"
        f"{extra}"
    )
    await send_telegram(msg)

# ----------------------
# Cleanup
# ----------------------
async def cleanup():
    try:
        await redis_close()
    except:
        pass
    log.info("✓ Helpers cleanup done")