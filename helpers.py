# ==========================================
# helpers.py — PRO VERSION (Part 1 of 2)
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

# ======================================================
# SYMBOL NORMALIZATION
# ======================================================
def normalize_symbol(sym: str) -> str:
    """Convert BTCUSDT → BTC/USDT automatically."""
    if "/" in sym:
        return sym
    if sym.endswith("USDT"):
        return f"{sym[:-4]}/USDT"
    return sym


# ======================================================
# GLOBAL SCORE HISTORY (for smoothing)
# ======================================================
score_history = {}

def smooth_score(key, new_score, alpha=0.35):
    """EMA smoothing to avoid jumpy scores."""
    old = score_history.get(key)
    if old is None:
        score_history[key] = new_score
    else:
        score_history[key] = (old * (1 - alpha)) + (new_score * alpha)
    return score_history[key]


# ======================================================
# COOLDOWN + DEDUPE ENGINE (Upgraded)
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
        """Deterministic sorted trigger hash."""
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
        """Avoid sending identical alerts within cooldown."""
        h = self.generate_hash(rule_key, triggers, price, mode)
        now = datetime.utcnow()

        if h in self.signal_hashes:
            if (now - self.signal_hashes[h]).total_seconds() < 1800:
                return False

        self.signal_hashes[h] = now
        return True


cooldown_manager = CooldownManager()


# ======================================================
# EXCHANGE MARKET DATA WRAPPER
# ======================================================
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
        """Retry with exponential backoff."""
        for attempt in range(retries):
            try:
                async with self.rate_sema:
                    return await async_fn()
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Fetch failed permanently: {e}")
                    return None
                backoff = 2 ** attempt
                logger.warning(f"Retry {attempt+1} after {backoff}s: {e}")
                await asyncio.sleep(backoff)

    async def get_all_data(self, symbol):
        """Fetch ticker, OHLCVs & orderbook safely."""
        try:
            sym = normalize_symbol(symbol)

            ticker = await self.fetch_with_retry(lambda: self.exchange.fetch_ticker(sym))
            if not ticker:
                return None

            ohlcv_1m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1m", limit=200))
            ohlcv_5m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "5m", limit=200))
            ohlcv_15m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "15m", limit=200))
            ohlcv_1h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "1h", limit=200))
            ohlcv_4h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(sym, "4h", limit=200))
            orderbook = await self.fetch_with_retry(lambda: self.exchange.fetch_order_book(sym, limit=20))

            if not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
                logger.error(f"Orderbook empty for {symbol}")
                return None

            bid = orderbook["bids"][0][0] if orderbook["bids"] else None
            ask = orderbook["asks"][0][0] if orderbook["asks"] else None

            if bid is None or ask is None or bid == 0:
                spread = float("inf")
            else:
                spread = (ask - bid) / ((ask + bid) / 2) * 100

            return {
                "symbol": symbol,
                "price": ticker["last"],
                "volume": ticker.get("baseVolume", ticker.get("volume", 0)),
                "ohlcv_1m": ohlcv_1m,
                "ohlcv_5m": ohlcv_5m,
                "ohlcv_15m": ohlcv_15m,
                "ohlcv_1h": ohlcv_1h,
                "ohlcv_4h": ohlcv_4h,
                "orderbook": orderbook,
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


# ======================================================
# OHLCV VALIDATION
# ======================================================
def validate_ohlcv(ohlcv, required):
    return bool(ohlcv and len(ohlcv) >= required)


# ======================================================
# PRICE FORMATTING
# ======================================================
def format_price_usd(p):
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:.2f}"
    return f"${p:.4f}"


# ======================================================
# SCORING ENGINE v2 — WEIGHTING + NORMALIZATION + PENALTY
# ======================================================
def get_weight_safe(name, weights, cap=3.0):
    w = weights.get(name, 1.0)
    try:
        w = float(w)
    except:
        w = 1.0
    return max(0.0, min(w, cap))


def compute_weighted_score(triggers_dict, weights):
    raw = 0
    max_poss = 0

    for name, val in triggers_dict.items():
        w = get_weight_safe(name, weights)
        v = 1 if val else 0
        raw += v * w
        max_poss += w

    if max_poss <= 0:
        max_poss = 1

    return raw, max_poss


def conflict_penalty(trigger_names, raw, pct=0.15):
    long_like = sum(1 for t in trigger_names if "long" in t.lower() or "bull" in t.lower())
    short_like = sum(1 for t in trigger_names if "short" in t.lower() or "bear" in t.lower())

    if long_like and short_like:
        return raw * (1 - pct)
    return raw


def normalize_score(raw, max_poss):
    score = (raw / max_poss) * 10
    return max(0, min(score, 10))


# ======================================================
# LIQUIDATION SAFETY
# ======================================================
def calc_liquidation(entry, sl, leverage, side):
    if side == "long":
        liq = entry * (1 - 1/leverage)
    else:
        liq = entry * (1 + 1/leverage)

    dist_pct = abs(liq - sl) / entry * 100

    return {
        "liq": liq,
        "dist_pct": dist_pct
    }


def safety_sl_check(entry, sl, lev, side):
    info = calc_liquidation(entry, sl, lev, side)
    return info["dist_pct"] >= MIN_SAFE_LIQ_DISTANCE_PCT


# ======================================================
# SPREAD & DEPTH CHECK
# ======================================================
def spread_ok(data):
    if data["spread"] > MAX_SPREAD_PCT:
        logger.warning(f"Spread too wide: {data['symbol']} {data['spread']:.3f}%")
        return False
    return True


def depth_ok(orderbook):
    try:
        bid_depth = sum(b[0] * b[1] for b in orderbook["bids"][:5])
        ask_depth = sum(a[0] * a[1] for a in orderbook["asks"][:5])
        return bid_depth >= MIN_ORDERBOOK_DEPTH and ask_depth >= MIN_ORDERBOOK_DEPTH
    except:
        return False
# ==========================================
# helpers.py — PRO VERSION (Part 2 of 2)
# ==========================================

# ======================================================
# SAFE INDICATORS (Optimized)
# ======================================================

def safe_rsi(ohlcv, period=14):
    if not validate_ohlcv(ohlcv, period+10):
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
    if not validate_ohlcv(ohlcv, period+10):
        return 0
    closes = pd.Series([c[4] for c in ohlcv])
    return closes.ewm(span=period, adjust=False).mean().iloc[-1]


def safe_adx(ohlcv, period=14):
    if not validate_ohlcv(ohlcv, period+10):
        return 20
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    df["tr"] = (
        df["high"] - df["low"]
    ).to_numpy()
    df["+dm"] = (df["high"].diff()).clip(lower=0)
    df["-dm"] = (-df["low"].diff()).clip(lower=0)
    atr = df["tr"].rolling(period).mean()
    di_plus = 100 * (df["+dm"].rolling(period).mean() / atr)
    di_minus = 100 * (df["-dm"].rolling(period).mean() / atr)
    dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = dx.rolling(period).mean()
    v = adx.iloc[-1]
    return 20 if pd.isna(v) else v


def safe_bbands(ohlcv, period=20, sd=2):
    if not validate_ohlcv(ohlcv, period+10):
        px = ohlcv[-1][4]
        return px, px, px
    closes = pd.Series([c[4] for c in ohlcv])
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = mid + sd * std
    lower = mid - sd * std
    return upper.iloc[-1], mid.iloc[-1], lower.iloc[-1]


def safe_atr(ohlcv, period=14):
    if not validate_ohlcv(ohlcv, period+10):
        return 0
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["tr"] = df[["high","low","close"]].apply(
        lambda x: max(
            x["high"] - x["low"],
            abs(x["high"] - x["close"]),
            abs(x["low"] - x["close"])
        ), axis=1)
    atr = df["tr"].rolling(period).mean().iloc[-1]
    return 0 if pd.isna(atr) else atr


def safe_vwap(ohlcv):
    if not validate_ohlcv(ohlcv, 10):
        return ohlcv[-1][4]
    typical = [(c[2] + c[3] + c[4]) / 3 for c in ohlcv]
    vol = [c[5] for c in ohlcv]
    tot = sum(vol)
    if tot == 0:
        return ohlcv[-1][4]
    return sum([typ * v for typ, v in zip(typical, vol)]) / tot


# ======================================================
# BTC CALM CHECK
# ======================================================
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
            logger.warning(f"BTC too volatile (1m): {change_1m:.2f}%")
            return False
        if change_5m > BTC_VOLATILITY_THRESHOLDS["5m"]:
            logger.warning(f"BTC too volatile (5m): {change_5m:.2f}%")
            return False

        return True
    except Exception as e:
        logger.error(f"BTC calm check error: {e}")
        return True


# ======================================================
# TP/SL ENGINE WITH SAFETY
# ======================================================
def calc_tp_sl(entry, side, mode):
    cfg = TP_SL_CONFIG.get(mode, TP_SL_CONFIG["MID"])
    tp_pct = cfg["tp"]
    sl_pct = cfg["sl"]

    if side == "long":
        tp = entry * (1 + tp_pct / 100)
        sl = entry * (1 - sl_pct / 100)
    else:
        tp = entry * (1 - tp_pct / 100)
        sl = entry * (1 + sl_pct / 100)

    lev = SUGGESTED_LEVERAGE.get(mode, 30)

    code = f"""```python
ENTRY = {entry}
TP = {tp}
SL = {sl}
LEVERAGE = {lev}
```"""

    return tp, sl, lev, code


# ======================================================
# TELEGRAM SENDER WITH RETRY
# ======================================================
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
                    logger.warning(f"Telegram error: {out}")
        except Exception as e:
            logger.warning(f"Telegram retry {attempt+1}: {e}")
            await asyncio.sleep(2 ** attempt)
    logger.error("Telegram failed after retries")
    return None


async def send_copy_block(token, chat_id, code):
    await send_telegram_message(token, chat_id, code)


# ======================================================
# SIGNAL BUILDERS (QUICK / MID / TREND)
# (Optimized Stable Version)
# ======================================================

def calculate_quick_signals(data):
    price = data["price"]
    o1 = data["ohlcv_1m"]
    o5 = data["ohlcv_5m"]

    triggers = {}
    labels = []

    # --------------------------------------------------
    # 1) RSI zone
    r = safe_rsi(o1)
    if 25 < r < 35:
        triggers["RSI_long"] = 1
        labels.append("RSI Long Zone")
    elif 65 < r < 75:
        triggers["RSI_short"] = 1
        labels.append("RSI Short Zone")

    # --------------------------------------------------
    # 2) MACD
    m, s, pm, ps = safe_macd(o5)
    if m > s and pm <= ps:
        triggers["MACD_bull"] = 1
        labels.append("MACD Bull Cross")
    elif m < s and pm >= ps:
        triggers["MACD_bear"] = 1
        labels.append("MACD Bear Cross")

    # --------------------------------------------------
    # 3) VWAP
    vwag = safe_vwap(o5)
    if price > vwag:
        triggers["VWAP_long"] = 1
        labels.append("VWAP Reclaim")
    else:
        triggers["VWAP_short"] = 1
        labels.append("VWAP Reject")

    # --------------------------------------------------
    # 4) EMA
    e9 = safe_ema(o5, 9)
    e21 = safe_ema(o5, 21)
    if price > e9 > e21:
        triggers["EMA_bull"] = 1
        labels.append("EMA Bull Structure")
    elif price < e9 < e21:
        triggers["EMA_bear"] = 1
        labels.append("EMA Bear Structure")

    # SCORING
    raw, max_poss = compute_weighted_score(triggers, LOGIC_WEIGHTS)
    raw = conflict_penalty(list(triggers.keys()), raw)
    score = normalize_score(raw, max_poss)
    score = smooth_score(f"{data['symbol']}_QUICK", score)

    # Direction
    longs = sum(1 for k in triggers if "long" in k or "bull" in k)
    shorts = sum(1 for k in triggers if "short" in k or "bear" in k)
    side = "long" if longs > shorts else "short" if shorts > longs else "none"

    return {
        "score": round(score, 1),
        "triggers": "\n".join(labels) if labels else "No triggers",
        "direction": side
    }


def calculate_mid_signals(data):
    price = data["price"]
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

    # SCORING
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
    price = data["price"]
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

    # SCORING
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


# ======================================================
# TELEGRAM FORMATTER (PREMIUM STYLE C)
# ======================================================
def telegram_formatter_style_c(symbol, mode, direction, result, data):
    emoji_map = {
        "QUICK": "⚡",
        "MID": "🔵",
        "TREND": "🟣"
    }
    emoji = emoji_map.get(mode, "📊")

    dir_txt = "🟢 LONG" if direction == "long" else "🔴 SHORT"

    entry = data["price"]
    tp, sl, lev, code = calc_tp_sl(entry, direction, mode)

    # Safety check
    if not safety_sl_check(entry, sl, lev, direction):
        return None, None

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