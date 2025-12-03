# ==========================================
# helpers.py — BLOCK 1 (imports, utils, cooldown)
# ==========================================

import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt

from config import *

logger = logging.getLogger("signal_bot")

# ======================================================
# SYMBOL NORMALIZATION
# ======================================================
def normalize_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym

# ======================================================
# SCORE SMOOTHING (EMA)
# ======================================================
score_history = {}

def smooth_score(key, new_score, alpha=0.35):
    old = score_history.get(key)
    if old is None:
        score_history[key] = new_score
    else:
        score_history[key] = (old * (1 - alpha)) + (new_score * alpha)
    return score_history[key]

# ======================================================
# COOLDOWN + DEDUPLICATION ENGINE
# ======================================================
class CooldownManager:
    def __init__(self):
        self.last_sent = {}
        self.signal_hashes = {}

    def can_send(self, symbol, mode, cooldown_seconds):
        key = f"{symbol}_{mode}"
        now = datetime.utcnow()
        last = self.last_sent.get(key)

        if last:
            if (now - last).total_seconds() < cooldown_seconds:
                return False

        self.last_sent[key] = now
        return True

    def generate_hash(self, rule_key, triggers, price, mode):
        try:
            if isinstance(triggers, str):
                t = triggers.split("\n")
            else:
                t = triggers
            t_sorted = sorted([x.strip() for x in t if x.strip()])
            data = json.dumps({
                "rule": rule_key,
                "triggers": t_sorted,
                "price": float(price),
                "mode": mode
            }, separators=(",", ":"))
            return hashlib.md5(data.encode()).hexdigest()
        except:
            return hashlib.md5(str(time.time()).encode()).hexdigest()

    def ensure_single_alert(self, rule_key, triggers, price, mode):
        h = self.generate_hash(rule_key, triggers, price, mode)
        now = datetime.utcnow()

        if h in self.signal_hashes:
            if (now - self.signal_hashes[h]).total_seconds() < 1800:
                return False

        self.signal_hashes[h] = now
        return True

# create global instance
cooldown_manager = CooldownManager()
# ==========================================
# helpers.py — BLOCK 2 (market wrapper + validation + scoring)
# ==========================================

# -----------------------
# EXCHANGE MARKET WRAPPER
# -----------------------
class MarketData:
    def __init__(self, api_key, secret):
        self.rate_sema = asyncio.Semaphore(6)
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "test": USE_TESTNET
            }
        })

    async def fetch_with_retry(self, async_fn, retries=3):
        for attempt in range(retries):
            try:
                async with self.rate_sema:
                    return await async_fn()
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Fetch failed permanently: {e}")
                    return None
                wait = 2 ** attempt
                logger.warning(f"Retry {attempt+1} after {wait}s: {e}")
                await asyncio.sleep(wait)

    async def get_all_data(self, symbol):
        try:
            sym = normalize_symbol(symbol)

            ticker = await self.fetch_with_retry(lambda: self.exchange.fetch_ticker(sym))
            if not ticker:
                return None

            o1 = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1m", limit=200))
            o5 = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "5m", limit=200))
            o15 = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "15m", limit=200))
            o1h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1h", limit=200))
            o4h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "4h", limit=200))
            ob = await self.fetch_with_retry(lambda: self.exchange.fetch_order_book(sym, limit=20))

            if not ob or not ob.get("bids") or not ob.get("asks"):
                logger.error(f"Orderbook empty for {symbol}")
                return None

            bid = ob["bids"][0][0] if ob["bids"] else None
            ask = ob["asks"][0][0] if ob["asks"] else None

            spread = (ask - bid) / ((ask + bid) / 2) * 100 if bid and ask else float("inf")

            return {
                "symbol": symbol,
                "price": ticker["last"],
                "volume": ticker.get("baseVolume", ticker.get("volume", 0)),
                "ohlcv_1m": o1,
                "ohlcv_5m": o5,
                "ohlcv_15m": o15,
                "ohlcv_1h": o1h,
                "ohlcv_4h": o4h,
                "orderbook": ob,
                "spread": spread,
            }

        except Exception as e:
            logger.error(f"Market fetch error for {symbol}: {e}")
            return None

    async def close(self):
        try:
            await self.exchange.close()
        except:
            pass

# -----------------------
# OHLCV VALIDATION + FORMAT
# -----------------------
def validate_ohlcv(ohlcv, req):
    return bool(ohlcv and len(ohlcv) >= req)

def format_price_usd(p):
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:.2f}"
    return f"${p:.4f}"

# -----------------------
# SCORING ENGINE (weighted + normalize)
# -----------------------
def get_weight_safe(name, weights, cap=3.0):
    w = weights.get(name, 1.0)
    try:
        w = float(w)
    except:
        w = 1.0
    return max(0.0, min(w, cap))

def compute_weighted_score(triggers, weights):
    raw = 0.0
    maxp = 0.0
    for name, val in triggers.items():
        w = get_weight_safe(name, weights)
        v = 1.0 if val else 0.0
        raw += v * w
        maxp += w
    if maxp <= 0:
        maxp = 1.0
    return raw, maxp

def conflict_penalty(names, raw, pct=0.15):
    long_like = sum(1 for t in names if "long" in t.lower() or "bull" in t.lower())
    short_like = sum(1 for t in names if "short" in t.lower() or "bear" in t.lower())
    return raw * (1 - pct) if long_like and short_like else raw

def normalize_score(raw, maxp):
    score = (raw / maxp) * 10.0
    return max(0.0, min(score, 10.0))
# ==========================================
# helpers.py — BLOCK 3 (Safety + Indicators)
# ==========================================

# -----------------------
# LIQUIDATION SAFETY
# -----------------------
def calc_liquidation(entry, sl, lev, side):
    if side == "long":
        liq = entry * (1 - 1/lev)
    else:
        liq = entry * (1 + 1/lev)

    dist = abs(liq - sl) / entry * 100
    return {"liq": liq, "dist_pct": dist}

def safety_sl_check(entry, sl, lev, side):
    info = calc_liquidation(entry, sl, lev, side)
    return info["dist_pct"] >= MIN_SAFE_LIQ_DISTANCE_PCT


# -----------------------
# SPREAD / DEPTH CHECK
# -----------------------
def spread_ok(data):
    return data["spread"] <= MAX_SPREAD_PCT

def depth_ok(orderbook):
    try:
        bd = sum(b[0] * b[1] for b in orderbook["bids"][:5])
        ad = sum(a[0] * a[1] for a in orderbook["asks"][:5])
        return bd >= MIN_ORDERBOOK_DEPTH and ad >= MIN_ORDERBOOK_DEPTH
    except:
        return False


# -----------------------
# SAFE INDICATORS
# -----------------------

def safe_rsi(o, period=14):
    if not validate_ohlcv(o, period+10):
        return 50
    closes = np.array([c[4] for c in o], float)
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return 70

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def safe_macd(o):
    if not validate_ohlcv(o, 35):
        return (0, 0, 0, 0)

    closes = pd.Series([c[4] for c in o])
    e12 = closes.ewm(span=12, adjust=False).mean()
    e26 = closes.ewm(span=26, adjust=False).mean()

    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()

    return macd.iloc[-1], sig.iloc[-1], macd.iloc[-2], sig.iloc[-2]


def safe_ema(o, p):
    if not validate_ohlcv(o, p+10):
        return 0
    closes = pd.Series([c[4] for c in o])
    return closes.ewm(span=p, adjust=False).mean().iloc[-1]


def safe_vwap(o):
    if not validate_ohlcv(o, 10):
        return o[-1][4]
    tp = [(c[2] + c[3] + c[4]) / 3 for c in o]
    vol = [c[5] for c in o]
    tot = sum(vol)
    if tot == 0:
        return o[-1][4]
    return sum([t*v for t, v in zip(tp, vol)]) / tot


def safe_adx(o, period=14):
    if not validate_ohlcv(o, period+10):
        return 20

    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
    df["+dm"] = (df["high"].diff()).clip(lower=0)
    df["-dm"] = (-df["low"].diff()).clip(lower=0)
    df["tr"] = (df["high"] - df["low"]).abs()

    atr = df["tr"].rolling(period).mean()
    di_plus = 100 * (df["+dm"].rolling(period).mean() / atr)
    di_minus = 100 * (df["-dm"].rolling(period).mean() / atr)

    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(period).mean()

    v = adx.iloc[-1]
    return 20 if pd.isna(v) else v


def safe_atr(o, period=14):
    if not validate_ohlcv(o, period+10):
        return 0

    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
    df["tr"] = df[["high","low","close"]].apply(
        lambda x: max(
            x["high"] - x["low"],
            abs(x["high"] - x["close"]),
            abs(x["low"] - x["close"])
        ),
        axis=1
    )

    atr = df["tr"].rolling(period).mean().iloc[-1]
    return 0 if pd.isna(atr) else atr


# -----------------------
# BTC CALM CHECK
# -----------------------
async def btc_calm_check(market):
    try:
        data = await market.get_all_data("BTCUSDT")
        if not data:
            return True

        o1 = data["ohlcv_1m"]
        o5 = data["ohlcv_5m"]

        if not validate_ohlcv(o1, 3) or not validate_ohlcv(o5, 3):
            return True

        c1_now = o1[-1][4]
        c1_prev = o1[-2][4]
        change_1m = abs(c1_now - c1_prev) / c1_prev * 100

        c5_now = o5[-1][4]
        c5_prev = o5[-2][4]
        change_5m = abs(c5_now - c5_prev) / c5_prev * 100

        if change_1m > BTC_VOLATILITY_THRESHOLDS["1m"]:
            return False
        if change_5m > BTC_VOLATILITY_THRESHOLDS["5m"]:
            return False

        return True
    except:
        return True
# ==========================================
# helpers.py — BLOCK 4 (TP/SL + Telegram Senders)
# ==========================================

# -----------------------
# TP/SL ENGINE (Safe Version)
# -----------------------
def calc_tp_sl(entry, side, mode):
    cfg = TP_SL_CONFIG.get(mode, TP_SL_CONFIG["MID"])
    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]

    # Long / Short logic
    if side == "long":
        tp = entry * (1 + tp_pct / 100)
        sl = entry * (1 - sl_pct / 100)
    else:
        tp = entry * (1 - tp_pct / 100)
        sl = entry * (1 + sl_pct / 100)

    lev = SUGGESTED_LEVERAGE.get(mode, 20)

    # CLEAN, SAFE CODE BLOCK
    code = f'''```python
ENTRY = {entry}
TP = {tp}
SL = {sl}
LEVERAGE = {lev}
```'''

    return tp, sl, lev, code


# -----------------------
# AI-Compatible TP/SL Engine
# -----------------------
def calc_single_tp_sl(entry_price, direction, mode, confidence_pct=None):
    cfg = TP_SL_CONFIG.get(mode, TP_SL_CONFIG["MID"])
    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]

    # Dynamic adjustment based on AI confidence
    if confidence_pct is not None:
        try:
            boost = max(0, min(float(confidence_pct), 100)) / 100
        except:
            boost = 0

        tp_pct = tp_pct * (1 + 0.30 * boost)     # up to +30%
        sl_pct = sl_pct * (1 - 0.20 * boost)     # up to -20%

    # Direction logic
    if direction == "long":
        tp = entry_price * (1 + tp_pct/100)
        sl = entry_price * (1 - sl_pct/100)
    else:
        tp = entry_price * (1 - tp_pct/100)
        sl = entry_price * (1 + sl_pct/100)

    lev = SUGGESTED_LEVERAGE.get(mode, 20)

    return {
        "tp": tp,
        "sl": sl,
        "leverage": lev
    }


# -----------------------
# TELEGRAM MESSAGE SENDER
# -----------------------
async def send_telegram_message(token, chat_id, message, retries=3):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, timeout=10) as r:
                    out = await r.json()
                    if out.get("ok"):
                        return out
        except Exception as e:
            await asyncio.sleep(2 ** attempt)

    return None


async def send_copy_block(token, chat_id, code):
    await send_telegram_message(token, chat_id, code)
# ==========================================
# helpers.py — BLOCK 5 (Signal Engine + Formatter)
# ==========================================

# -----------------------
# QUICK SIGNAL ENGINE
# -----------------------
def calculate_quick_signals(data):
    price = data["price"]
    o1 = data["ohlcv_1m"]
    o5 = data["ohlcv_5m"]

    triggers = {}
    labels = []

    # 1) RSI
    r = safe_rsi(o1)
    if 25 < r < 35:
        triggers["RSI_long"] = 1
        labels.append("RSI Long Zone")
    elif 65 < r < 75:
        triggers["RSI_short"] = 1
        labels.append("RSI Short Zone")

    # 2) MACD
    m, s, pm, ps = safe_macd(o5)
    if m > s and pm <= ps:
        triggers["MACD_bull"] = 1
        labels.append("MACD Bullish Cross")
    elif m < s and pm >= ps:
        triggers["MACD_bear"] = 1
        labels.append("MACD Bearish Cross")

    # 3) VWAP
    vw = safe_vwap(o5)
    if price > vw:
        triggers["VWAP_long"] = 1
        labels.append("VWAP Reclaim")
    else:
        triggers["VWAP_short"] = 1
        labels.append("VWAP Reject")

    # 4) EMA
    e9 = safe_ema(o5, 9)
    e21 = safe_ema(o5, 21)
    if price > e9 > e21:
        triggers["EMA_bull"] = 1
        labels.append("EMA Bull Structure")
    elif price < e9 < e21:
        triggers["EMA_bear"] = 1
        labels.append("EMA Bear Structure")

    # -----------------------
    # SCORING
    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_QUICK", score)

    # Direction
    long_count = sum(1 for k in triggers if "long" in k or "bull" in k)
    short_count = sum(1 for k in triggers if "short" in k or "bear" in k)
    side = "long" if long_count > short_count else "short" if short_count > long_count else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels) if labels else "No triggers",
        "direction": side
    }


# -----------------------
# MID SIGNAL ENGINE
# -----------------------
def calculate_mid_signals(data):
    price = data["price"]
    o15 = data["ohlcv_15m"]
    o1h = data["ohlcv_1h"]

    triggers = {}
    labels = []

    # 1) RSI
    r = safe_rsi(o15)
    if r < 30:
        triggers["RSI_os"] = 1
        labels.append("RSI Oversold")
    elif r > 70:
        triggers["RSI_ob"] = 1
        labels.append("RSI Overbought")

    # 2) MACD
    m, s, _, _ = safe_macd(o15)
    if m > s:
        triggers["MACD_bull"] = 1
        labels.append("MACD Up")

    # 3) EMA50 Reaction
    e50 = safe_ema(o1h, 50)
    if e50 > 0 and abs(price - e50) / e50 < 0.01:
        triggers["EMA50_touch"] = 1
        labels.append("EMA 50 Reaction")

    # -----------------------
    # SCORING
    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_MID", score)

    long_count = sum(1 for k in triggers if "os" in k or "bull" in k)
    short_count = sum(1 for k in triggers if "ob" in k)
    side = "long" if long_count > short_count else "short" if short_count > long_count else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels) if labels else "No triggers",
        "direction": side
    }


# -----------------------
# TREND SIGNAL ENGINE
# -----------------------
def calculate_trend_signals(data):
    price = data["price"]
    o4h = data["ohlcv_4h"]

    triggers = {}
    labels = []

    # 1) Supertrend-like condition
    e50 = safe_ema(o4h, 50)
    e200 = safe_ema(o4h, 200)

    if e50 > e200 and price > e50:
        triggers["Trend_long"] = 1
        labels.append("Trend Bullish (50>200)")
    elif e50 < e200 and price < e50:
        triggers["Trend_short"] = 1
        labels.append("Trend Bearish (50<200)")

    # 2) ATR confirmation
    atr = safe_atr(o4h)
    if atr > 0:
        triggers["ATR_ok"] = 1
        labels.append("ATR Stable")

    # -----------------------
    # SCORING
    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_TREND", score, alpha=0.25)

    long_count = sum(1 for k in triggers if "long" in k)
    short_count = sum(1 for k in triggers if "short" in k)
    side = "long" if long_count > short_count else "short" if short_count > long_count else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels) if labels else "No triggers",
        "direction": side
    }


# -----------------------
# PREMIUM TELEGRAM FORMATTER (STYLE C)
# -----------------------
def telegram_formatter_style_c(symbol, mode, direction, result, data):
    emoji_map = {
        "QUICK": "⚡",
        "MID": "🔵",
        "TREND": "🟣"
    }

    emoji = emoji_map.get(mode, "📊")
    dir_text = "🟢 LONG" if direction == "long" else "🔴 SHORT"

    entry = data["price"]
    tp, sl, lev, code = calc_tp_sl(entry, direction, mode)

    # SL safety check
    if not safety_sl_check(entry, sl, lev, direction):
        return None, None

    msg = f"""
{emoji} <b>{mode} {dir_text} SIGNAL</b>
━━━━━━━━━━━━━━━━━━
<b>Pair:</b> {symbol}
<b>Entry:</b> {format_price_usd(entry)}
<b>TP:</b> {format_price_usd(tp)}
<b>SL:</b> {format_price_usd(sl)}
<b>Leverage:</b> {lev}x

<b>Score:</b> {result['score']}/10
<b>Triggers:</b>
{result['triggers']}

<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━
"""

    return msg, code
# ==========================================
# helpers.py — BLOCK 6 (Final Utilities + Polish)
# ==========================================

# -----------------------
# SPREAD CHECK
# -----------------------
def spread_ok(data):
    spread = data.get("spread", 0)
    symbol = data.get("symbol", "UNKNOWN")
    if spread > MAX_SPREAD_PCT:
        logger.warning(f"❌ Spread too wide on {symbol}: {spread:.3f}%")
        return False
    return True


# -----------------------
# ORDERBOOK DEPTH CHECK
# -----------------------
def depth_ok(orderbook):
    try:
        bid_depth = sum(bid[0] * bid[1] for bid in orderbook["bids"][:5])
        ask_depth = sum(ask[0] * ask[1] for ask in orderbook["asks"][:5])
        return bid_depth >= MIN_ORDERBOOK_DEPTH and ask_depth >= MIN_ORDERBOOK_DEPTH
    except:
        return False


# -----------------------
# BTC CALM CHECK (already linked to bot)
# -----------------------
async def btc_calm_check(market):
    try:
        data = await market.get_all_data("BTCUSDT")
        if not data:
            return True

        o1 = data["ohlcv_1m"]
        o5 = data["ohlcv_5m"]

        if not validate_ohlcv(o1, 3) or not validate_ohlcv(o5, 3):
            return True

        # 1m volatility
        c1_now = o1[-1][4]
        c1_prev = o1[-2][4]
        ch1 = abs(c1_now - c1_prev) / c1_prev * 100

        # 5m volatility
        c5_now = o5[-1][4]
        c5_prev = o5[-2][4]
        ch5 = abs(c5_now - c5_prev) / c5_prev * 100

        if ch1 > BTC_VOLATILITY_THRESHOLDS["1m"]:
            logger.warning(f"⚠️ BTC volatile (1m {ch1:.2f}%)")
            return False

        if ch5 > BTC_VOLATILITY_THRESHOLDS["5m"]:
            logger.warning(f"⚠️ BTC volatile (5m {ch5:.2f}%)")
            return False

        return True

    except Exception as e:
        logger.error(f"BTC Calm Error: {e}")
        return True


# -----------------------
# SAFE IMPORT CHECK (optional)
# -----------------------
def safe_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default


# -----------------------
# LOG HELPER FOR CLEAN OUTPUT
# -----------------------
def log_score(symbol, mode, score, dir_text):
    logger.info(
        f"📊 {symbol} | {mode} | Score: {score}/10 | Direction: {dir_text}"
    )