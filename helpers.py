# ==========================================
# helpers.py — PART 1
# ==========================================

import aiohttp
import asyncio
import os
import time
import hmac
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt

from config import *

logger = logging.getLogger("signal_bot")

# ==============================
# SAFE FLOAT (CRITICAL PATCH)
# ==============================
def safe_float(v, default=0.0):
    """Convert API/string values safely to float"""
    try:
        return float(v)
    except:
        try:
            return float(str(v).replace(",", "").strip())
        except:
            return default

# ==============================
# SYMBOL NORMALIZATION
# ==============================
def normalize_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym

# ==============================
# SCORE SMOOTHING (EMA)
# ==============================
score_history = {}

def smooth_score(key, new_score, alpha=0.35):
    old = score_history.get(key)
    if old is None:
        score_history[key] = new_score
    else:
        score_history[key] = (old * (1 - alpha)) + (new_score * alpha)
    return score_history[key]

# ==============================
# COOLDOWN + DEDUPE MANAGER
# ==============================
class CooldownManager:
    def __init__(self):
        self.last_sent = {}
        self.signal_hashes = {}

    def can_send(self, symbol, mode, cooldown_seconds):
        key = f"{symbol}_{mode}"
        now = datetime.utcnow()
        last = self.last_sent.get(key)

        if last and (now - last).total_seconds() < cooldown_seconds:
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
                "price": safe_float(price),
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

cooldown_manager = CooldownManager()

# ==============================
# MARKET DATA WRAPPER
# ==============================
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

    async def fetch_with_retry(self, fn, retries=3):
        for attempt in range(retries):
            try:
                async with self.rate_sema:
                    return await fn()
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Fetch failed permanently: {e}")
                    return None
                backoff = 2 ** attempt
                logger.warning(f"Retry {attempt+1} after {backoff}s: {e}")
                await asyncio.sleep(backoff)

    async def get_all_data(self, symbol):
        try:
            sym = normalize_symbol(symbol)

            ticker = await self.fetch_with_retry(lambda: self.exchange.fetch_ticker(sym))
            if not ticker:
                return None

            # -------------------------
            # SAFE FLOAT PATCH
            # -------------------------
            last = safe_float(ticker.get("last", 0))
            vol = safe_float(ticker.get("baseVolume", ticker.get("volume", 0)))

            ohlcv_1m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1m", limit=200))
            ohlcv_5m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "5m", limit=200))
            ohlcv_15m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "15m", limit=200))
            ohlcv_1h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1h", limit=200))
            ohlcv_4h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "4h", limit=200))

            orderbook = await self.fetch_with_retry(lambda: self.exchange.fetch_order_book(sym, limit=20))
            if not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
                logger.error(f"Orderbook empty for {symbol}")
                return None

            bid = safe_float(orderbook["bids"][0][0]) if orderbook["bids"] else None
            ask = safe_float(orderbook["asks"][0][0]) if orderbook["asks"] else None

            if bid is None or ask is None or bid == 0:
                spread = 999
            else:
                spread = (ask - bid) / ((ask + bid) / 2) * 100

            return {
                "symbol": symbol,
                "price": last,
                "volume": vol,
                "ohlcv_1m": ohlcv_1m,
                "ohlcv_5m": ohlcv_5m,
                "ohlcv_15m": ohlcv_15m,
                "ohlcv_1h": ohlcv_1h,
                "ohlcv_4h": ohlcv_4h,
                "orderbook": orderbook,
                "spread": spread
            }

        except Exception as e:
            logger.error(f"Market fetch error for {symbol}: {e}")
            return None

    async def close(self):
        try:
            await self.exchange.close()
        except:
            pass

# ==============================
# VALIDATION / FORMATTING
# ==============================
def validate_ohlcv(ohlcv, required):
    return bool(ohlcv and len(ohlcv) >= required)

def format_price_usd(p):
    p = safe_float(p)
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:.2f}"
    return f"${p:.4f}"
# ==========================================
# helpers.py — PART 2
# ==========================================

# ==============================
# INDICATORS (Stable Versions)
# ==============================
def safe_rsi(ohlcv, period=14):
    if not validate_ohlcv(ohlcv, period + 10):
        return 50
    closes = np.array([c[4] for c in ohlcv], dtype=float)
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 70
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def safe_macd(ohlcv):
    if not validate_ohlcv(ohlcv, 35):
        return (0, 0, 0, 0)
    closes = pd.Series([c[4] for c in ohlcv])
    exp1 = closes.ewm(span=12, adjust=False).mean()
    exp2 = closes.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1], macd.iloc[-2], signal.iloc[-2]


def safe_ema(ohlcv, period):
    if not validate_ohlcv(ohlcv, period + 10):
        return 0
    closes = pd.Series([c[4] for c in ohlcv])
    return closes.ewm(span=period, adjust=False).mean().iloc[-1]


def safe_atr(ohlcv, period=14):
    if not validate_ohlcv(ohlcv, period + 10):
        return 0
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    df["tr"] = df[["high", "low", "close"]].apply(
        lambda x: max(
            x["high"] - x["low"],
            abs(x["high"] - x["close"]),
            abs(x["low"] - x["close"])
        ), axis=1
    )
    atr = df["tr"].rolling(period).mean().iloc[-1]
    return 0 if pd.isna(atr) else atr


def safe_vwap(ohlcv):
    if not validate_ohlcv(ohlcv, 10):
        return safe_float(ohlcv[-1][4])
    typical = [(c[2] + c[3] + c[4]) / 3 for c in ohlcv]
    vol = [c[5] for c in ohlcv]
    total = sum(vol)
    if total == 0:
        return safe_float(ohlcv[-1][4])
    return sum([t * v for t, v in zip(typical, vol)]) / total


# ==============================
# BTC CALM CHECK
# ==============================
async def btc_calm_check(market):
    try:
        data = await market.get_all_data("BTCUSDT")
        if not data:
            return True

        o1 = data["ohlcv_1m"]
        o5 = data["ohlcv_5m"]

        if not validate_ohlcv(o1, 3) or not validate_ohlcv(o5, 3):
            return True

        c1_now = safe_float(o1[-1][4])
        c1_prev = safe_float(o1[-2][4])
        change_1m = abs(c1_now - c1_prev) / c1_prev * 100

        c5_now = safe_float(o5[-1][4])
        c5_prev = safe_float(o5[-2][4])
        change_5m = abs(c5_now - c5_prev) / c5_prev * 100

        if change_1m > BTC_VOLATILITY_THRESHOLDS["1m"]:
            logger.warning(f"BTC volatile (1m): {change_1m:.2f}%")
            return False
        if change_5m > BTC_VOLATILITY_THRESHOLDS["5m"]:
            logger.warning(f"BTC volatile (5m): {change_5m:.2f}%")
            return False

        return True

    except Exception as e:
        logger.error(f"BTC calm error: {e}")
        return True


# ==============================
# SPREAD / DEPTH CHECK
# ==============================
def spread_ok(data):
    if data["spread"] > MAX_SPREAD_PCT:
        logger.warning(f"❌ Spread too wide on {data['symbol']}: {data['spread']:.3f}%")
        return False
    return True


def depth_ok(orderbook):
    try:
        bid_depth = sum(b[0] * b[1] for b in orderbook["bids"][:5])
        ask_depth = sum(a[0] * a[1] for a in orderbook["asks"][:5])
        return bid_depth >= MIN_ORDERBOOK_DEPTH and ask_depth >= MIN_ORDERBOOK_DEPTH
    except:
        return False


# ==============================
# SCORE ENGINE
# ==============================
def get_weight_safe(name, weights, cap=3.0):
    try:
        return max(0.0, min(float(weights.get(name, 1.0)), cap))
    except:
        return 1.0


def compute_weighted_score(triggers, weights):
    raw = 0
    max_poss = 0
    for name, active in triggers.items():
        w = get_weight_safe(name, weights)
        if active:
            raw += w
        max_poss += w
    if max_poss <= 0:
        max_poss = 1
    return raw, max_poss


def conflict_penalty(trigger_names, raw, pct=0.15):
    long_like = sum(1 for t in trigger_names if "long" in t or "bull" in t)
    short_like = sum(1 for t in trigger_names if "short" in t or "bear" in t)
    if long_like and short_like:
        return raw * (1 - pct)
    return raw


def normalize_score(raw, max_poss):
    score = (raw / max_poss) * 10
    return max(0, min(score, 10))


# ==============================
# TP/SL ENGINE (SAFE)
# ==============================
def calc_tp_sl(entry, side, mode):
    entry = safe_float(entry)

    cfg = TP_SL_CONFIG.get(mode, TP_SL_CONFIG["MID"])
    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]

    if side == "long":
        tp = entry * (1 + tp_pct / 100)
        sl = entry * (1 - sl_pct / 100)
    else:
        tp = entry * (1 - tp_pct / 100)
        sl = entry * (1 + sl_pct / 100)

    leverage = SUGGESTED_LEVERAGE.get(mode, 20)

    code_block = f"""```python
ENTRY = {entry}
TP = {tp}
SL = {sl}
LEVERAGE = {leverage}
```"""

    return tp, sl, leverage, code_block
# ======================================================
# CONFIDENCE-BASED TP/SL (Used by AI Layer)
# ======================================================
def calc_single_tp_sl(entry_price, direction, mode, confidence_pct=None):
    entry_price = safe_float(entry_price)

    cfg = TP_SL_CONFIG.get(mode, TP_SL_CONFIG["MID"])
    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]

    # AI-based leverage boost
    if confidence_pct is not None and confidence_pct >= 90:
        lev = 50
    else:
        lev = SUGGESTED_LEVERAGE.get(mode, 20)

    if direction == "long":
        tp = entry_price * (1 + tp_pct / 100)
        sl = entry_price * (1 - sl_pct / 100)
    else:
        tp = entry_price * (1 - tp_pct / 100)
        sl = entry_price * (1 + sl_pct / 100)

    code = f"""```python
ENTRY = {entry_price}
TP = {tp}
SL = {sl}
LEVERAGE = {lev}
```"""

    return {
        "tp": tp,
        "sl": sl,
        "leverage": lev,
        "code": code
    }


# ==============================
# TELEGRAM SENDER
# ==============================
async def send_telegram_message(token, chat_id, msg, retries=3):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, timeout=10) as r:
                    out = await r.json()
                    if out.get("ok"):
                        return out
        except Exception as e:
            logger.warning(f"Telegram retry {attempt+1}: {e}")
            await asyncio.sleep(2 ** attempt)

    logger.error("Telegram failed after retries")
    return None


async def send_copy_block(token, chat_id, code):
    await send_telegram_message(token, chat_id, code)


# ==============================
# SIGNAL BUILDERS
# ==============================
def calculate_quick_signals(data):
    price = safe_float(data["price"])
    o1 = data["ohlcv_1m"]
    o5 = data["ohlcv_5m"]

    triggers = {}
    labels = []

    r = safe_rsi(o1)
    if 25 < r < 35:
        triggers["RSI_long"] = 1
        labels.append("RSI Long Zone")
    elif 65 < r < 75:
        triggers["RSI_short"] = 1
        labels.append("RSI Short Zone")

    m, s, pm, ps = safe_macd(o5)
    if m > s and pm <= ps:
        triggers["MACD_bull"] = 1
        labels.append("MACD Bull Cross")
    elif m < s and pm >= ps:
        triggers["MACD_bear"] = 1
        labels.append("MACD Bear Cross")

    vw = safe_vwap(o5)
    if price > vw:
        triggers["VWAP_long"] = 1
        labels.append("VWAP Reclaim")
    else:
        triggers["VWAP_short"] = 1
        labels.append("VWAP Reject")

    e9 = safe_ema(o5, 9)
    e21 = safe_ema(o5, 21)
    if price > e9 > e21:
        triggers["EMA_bull"] = 1
        labels.append("EMA Bull Structure")
    elif price < e9 < e21:
        triggers["EMA_bear"] = 1
        labels.append("EMA Bear Structure")

    # score
    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_QUICK", score)

    longs = sum(1 for k in triggers if "long" in k or "bull" in k)
    shorts = sum(1 for k in triggers if "short" in k or "bear" in k)
    side = "long" if longs > shorts else "short" if shorts > longs else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels) if labels else "No triggers",
        "direction": side
    }


def calculate_mid_signals(data):
    price = safe_float(data["price"])
    o15 = data["ohlcv_15m"]
    o1h = data["ohlcv_1h"]

    triggers = {}
    labels = []

    r = safe_rsi(o15)
    if r < 30:
        triggers["RSI_os"] = 1
        labels.append("RSI Oversold")
    elif r > 70:
        triggers["RSI_ob"] = 1
        labels.append("RSI Overbought")

    m, s, _, _ = safe_macd(o15)
    if m > s:
        triggers["MACD_bull"] = 1
        labels.append("MACD Up")

    e50 = safe_ema(o1h, 50)
    if abs(price - e50) / e50 < 0.01:
        triggers["EMA50_touch"] = 1
        labels.append("EMA50 Reaction")

    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_MID", score)

    longs = sum(1 for k in triggers if "os" in k or "bull" in k)
    shorts = sum(1 for k in triggers if "ob" in k)
    side = "long" if longs > shorts else "short" if shorts > longs else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels),
        "direction": side
    }


def calculate_trend_signals(data):
    price = safe_float(data["price"])
    o4h = data["ohlcv_4h"]

    triggers = {}
    labels = []

    e50 = safe_ema(o4h, 50)
    e200 = safe_ema(o4h, 200)

    if e50 > e200 and price > e50:
        triggers["ST_long"] = 1
        labels.append("Supertrend Bull")
    elif e50 < e200 and price < e50:
        triggers["ST_short"] = 1
        labels.append("Supertrend Bear")

    atr = safe_atr(o4h)
    if atr:
        triggers["ATR_ok"] = 1

    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_TREND", score, alpha=0.25)

    longs = sum(1 for k in triggers if "long" in k)
    shorts = sum(1 for k in triggers if "short" in k)
    side = "long" if longs > shorts else "short" if shorts > longs else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels),
        "direction": side
    }


# ==============================
# TELEGRAM FINAL FORMATTER
# ==============================
def telegram_formatter_style_c(symbol, mode, direction, result, data):

    emoji_map = {
        "QUICK": "⚡",
        "MID": "🔵",
        "TREND": "🟣",
    }
    emoji = emoji_map.get(mode, "📊")

    dir_txt = "🟢 LONG" if direction == "long" else "🔴 SHORT"

    entry = safe_float(data["price"])

    tp, sl, lev, code = calc_tp_sl(entry, direction, mode)

    msg = f"""
{emoji} <b>{mode} {dir_txt} SIGNAL</b>
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
# PENDING SIGNAL MEMORY (robust + file-backed)
# ==========================================

import json
import os
from threading import Lock

_pending_memory = {}
_pending_lock = Lock()
_PENDING_FILE = "/tmp/pending_signals.json"

def _load_file_to_memory():
    try:
        if os.path.exists(_PENDING_FILE):
            with open(_PENDING_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _pending_memory.update(data)
    except Exception as e:
        logger.warning(f"Failed to load pending file: {e}")

def _dump_memory_to_file():
    try:
        with open(_PENDING_FILE, "w") as f:
            json.dump(_pending_memory, f)
        return True
    except Exception as e:
        logger.error(f"Failed to write pending file: {e}")
        return False

# init load (safe)
_load_file_to_memory()

def save_pending_signal(key, data):
    """
    Save last signal snapshot for dedupe and reference.
    Returns True on success, False on failure.
    key = "BTCUSDT_QUICK"
    data = {"triggers": "...", "price": 1234, "mode": "QUICK", "time": "2025-12-03T..."}
    """
    try:
        with _pending_lock:
            _pending_memory[key] = data
            ok = _dump_memory_to_file()
            return ok
    except Exception as e:
        logger.error(f"save_pending_signal error: {e}")
        return False

def load_pending_signals():
    """Return a shallow copy of full memory dictionary"""
    with _pending_lock:
        return dict(_pending_memory)

def get_pending_signal(key):
    """Return single pending signal or None"""
    with _pending_lock:
        return _pending_memory.get(key)

def clear_pending_signal(key):
    """Delete one pending key"""
    try:
        with _pending_lock:
            if key in _pending_memory:
                del _pending_memory[key]
                _dump_memory_to_file()
                return True
    except Exception as e:
        logger.error(f"clear_pending_signal error: {e}")
    return False