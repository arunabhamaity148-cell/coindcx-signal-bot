# ============================================
# helpers.py — FINAL PRO SETUP (PART 1/4)
# Core, Exchange, Fetch, Indicators
# ============================================

from __future__ import annotations
import os, time, json, math, logging, requests
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import ccxt
except Exception:
    ccxt = None

# -----------------------------
# ENV CONFIG
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")

EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "92"))
THRESH_MID = float(os.getenv("THRESH_MID", "85"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "72"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

# same coin cooldown (all 30 mins except trend)
TTL_QUICK = 1800
TTL_MID = 1800
TTL_TREND = 3600

EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

# -----------------------------
# LOCAL JSON COOLDOWN
# -----------------------------
class CooldownManager:
    def __init__(self):
        self.map = {}
        self._load()

    def _load(self):
        try:
            with open(COOLDOWN_JSON, "r") as f:
                self.map = {k:int(v) for k,v in json.load(f).items()}
        except Exception:
            self.map = {}

    def _save(self):
        try:
            with open(COOLDOWN_JSON + ".tmp","w") as f:
                json.dump(self.map, f)
            os.replace(COOLDOWN_JSON + ".tmp", COOLDOWN_JSON)
        except:
            pass

    def ttl_for_mode(self, mode: str) -> int:
        if mode == "QUICK": return TTL_QUICK
        if mode == "MID": return TTL_MID
        return TTL_TREND

    def set(self, pair: str, mode: str):
        ttl = self.ttl_for_mode(mode)
        now = int(time.time())
        self.map[pair] = now + ttl
        self._save()

    def cooled(self, pair: str) -> bool:
        now = int(time.time())
        exp = self.map.get(pair, 0)
        if exp <= now:
            if pair in self.map:
                self.map.pop(pair); self._save()
            return False
        return True

_cd = CooldownManager()

def cooldown_key_for(pair: str, mode: str) -> str:
    return f"{pair.upper()}::{mode.upper()}"

# -----------------------------
# EXCHANGE
# -----------------------------
def get_exchange() -> "ccxt.Exchange":
    if ccxt is None:
        raise RuntimeError("ccxt missing")

    try:
        ex_cls = getattr(ccxt, EXCHANGE_NAME, ccxt.binance)
        ex = ex_cls({'enableRateLimit': True})
    except:
        ex = ccxt.binance({'enableRateLimit': True})

    try:
        ex.load_markets()
    except:
        pass

    return ex

def normalize_symbol(s: str) -> str:
    s = s.upper().strip()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# -----------------------------
# OHLCV FETCH
# -----------------------------
def df_from_ohlcv(ohlcv):
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=[
        "ts","open","high","low","close","volume"
    ])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df if not df.empty else None

def fetch_ohlcv(ex, symbol, tf="1m", limit=200):
    symbol = normalize_symbol(symbol)
    last = None
    for i in range(3):
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            time.sleep(0.2)
            return df_from_ohlcv(data)
        except Exception as e:
            last = e
            time.sleep(0.3*(i+1))
    LOG.debug(f"OHLCV failed {symbol}: {last}")
    return None

# -----------------------------
# INDICATORS
# -----------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

def atr(df, length=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([
        (high-low),
        (high-prev).abs(),
        (low-prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()
# ============================================
# helpers.py — FINAL PRO SETUP (PART 2/4)
# Liquidity, Orderbook, Spread, SR Filter, Gainers
# ============================================

# -----------------------------
# ORDERBOOK
# -----------------------------
def fetch_orderbook_safe(ex, symbol, limit=50):
    try:
        ob = ex.fetch_order_book(normalize_symbol(symbol), limit=limit)
        time.sleep(0.15)
        return ob or {}
    except:
        return {}

def orderbook_imbalance(ob, depth=12) -> float:
    bids = ob.get('bids', [])
    asks = ob.get('asks', [])
    bid_vol = sum(float(b[1]) for b in bids[:depth]) if bids else 0.0
    ask_vol = sum(float(a[1]) for a in asks[:depth]) if asks else 0.0
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

# -----------------------------
# LIQUIDITY SCORE (PRO)
# -----------------------------
def liquidity_score(ob) -> float:
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    top5 = (
        sum(float(b[1]) for b in bids[:5]) +
        sum(float(a[1]) for a in asks[:5])
    ) / 2 if (bids or asks) else 0

    deep20 = (
        sum(float(b[1]) for b in bids[:20]) +
        sum(float(a[1]) for a in asks[:20])
    ) / 2 if (bids or asks) else 1

    if deep20 == 0:
        return 0.0

    score = min(1.0, top5 / (deep20 + 1e-12))
    return round(score, 3)

# -----------------------------
# SPREAD CALC
# -----------------------------
def calc_spread(ob):
    try:
        bid = float(ob['bids'][0][0])
        ask = float(ob['asks'][0][0])
        return abs(ask - bid) / ask
    except:
        return 999.0

# -----------------------------
# S/R FAKE SIGNAL FILTER
# Reject signals touching local levels
# -----------------------------
def sr_rejection_filter(df):
    close = df['close']
    high = df['high']
    low = df['low']

    recent_high = high.rolling(25).max().iloc[-2]
    recent_low = low.rolling(25).min().iloc[-2]

    last_close = close.iloc[-1]

    # too close to resistance → risky for BUY
    if abs(last_close - recent_high) / last_close < 0.0025:
        return False, "near_resistance"

    # too close to support → risky for SELL
    if abs(last_close - recent_low) / last_close < 0.0025:
        return False, "near_support"

    return True, ""

# -----------------------------
# TOP GAINERS / STRONG TREND FILTER
# Skip dying / flat coins
# -----------------------------
def top_gainer_filter(df1m):
    try:
        close = df1m['close']
        pct = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]  # last 6 min momentum
        if pct > 0.004:     # strong pump → ok
            return True
        if pct < -0.004:    # dump → skip
            return False
        return True
    except:
        return True
# ============================================
# helpers.py — FINAL PRO SETUP (PART 3/4)
# Scoring + Direction + TP/SL + Mode Logic
# ============================================

# -----------------------------
# SCORING ENGINE (MTF + Liquidity + Volume + OBI)
# -----------------------------
def compute_score(df1, df5, ob, spread):
    reasons = []
    score = 40.0

    # ===== EMA =====
    e20_1 = ema(df1['close'], EMA_FAST).iloc[-1]
    e50_1 = ema(df1['close'], EMA_SLOW).iloc[-1]
    e20_5 = ema(df5['close'], EMA_FAST).iloc[-1] if df5 is not None else e20_1
    e50_5 = ema(df5['close'], EMA_SLOW).iloc[-1] if df5 is not None else e50_1

    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    else:
        reasons.append("EMA20<50_1m")

    if e20_5 > e50_5:
        score += 8; reasons.append("EMA20>50_5m")
    else:
        reasons.append("EMA20<50_5m")

    # ===== MACD =====
    macd_val = (ema(df1['close'], 12) - ema(df1['close'], 26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # ===== RSI =====
    r = float(rsi(df1['close']).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")
    else:
        score += 1; reasons.append("RSI_high")

    # ===== Volume Spike =====
    vol = df1['volume']
    vol_avg = vol.rolling(20).mean().iloc[-1]
    if vol.iloc[-1] > vol_avg * 1.6:
        score += 12; reasons.append("VOL_spike")
    else:
        reasons.append("VOL_ok")

    # ===== OBI =====
    obi = orderbook_imbalance(ob)
    if obi > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    elif obi < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    # ===== Liquidity =====
    liq = liquidity_score(ob)
    if liq > 0.55:
        score += 8; reasons.append("Liquidity_ok")
    else:
        reasons.append("Liquidity_low")

    # ===== Spread =====
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = max(0, min(score, 100))
    return round(score, 1), reasons


# -----------------------------
# BUY/SELL DIRECTION ENGINE
# -----------------------------
def detect_direction(df1, ob):
    e20 = ema(df1['close'], EMA_FAST).iloc[-1]
    e50 = ema(df1['close'], EMA_SLOW).iloc[-1]
    obi = orderbook_imbalance(ob)

    # SELL super strong when EMA down + OBI negative
    if e20 < e50 or obi < -0.45:
        return "SELL"
    return "BUY"


# -----------------------------
# MODE: QUICK | MID | TREND
# -----------------------------
def detect_mode(score):
    if score >= THRESH_QUICK:
        return "QUICK"
    if score >= THRESH_MID:
        return "MID"
    return "TREND"


# -----------------------------
# TP / SL via ATR (Symmetric BUY/SELL)
# -----------------------------
def build_signal(symbol, df1, df5, ob, score, reasons):
    entry = float(df1['close'].iloc[-1])
    atr_val = float(atr(df1).iloc[-1])

    direction = detect_direction(df1, ob)
    mode = detect_mode(score)

    # TP SL multipliers
    if mode == "QUICK":
        tp_m = 1.5
        sl_m = 1.0
    elif mode == "TREND":
        tp_m = 3.0
        sl_m = 2.0
    else:
        tp_m = 2.0
        sl_m = 1.4

    if direction == "BUY":
        tp = entry + atr_val * tp_m
        sl = entry - atr_val * sl_m
    else:
        tp = entry - atr_val * tp_m
        sl = entry + atr_val * sl_m

    # suggested leverage
    lev = "50x" if mode == "QUICK" else ("25x" if mode == "MID" else "20x")

    danger = (
        round(entry - atr_val * 1.1, 8),
        round(entry + atr_val * 1.1, 8)
    )

    return {
        "pair": normalize_symbol(symbol),
        "direction": direction,
        "mode": mode,
        "score": score,
        "entry": round(entry, 8),
        "tp": round(tp, 8),
        "sl": round(sl, 8),
        "atr": round(atr_val, 8),
        "lev": lev,
        "danger": danger,
        "reasons": reasons,
        "ts": int(time.time())
    }
# ============================================
# helpers.py — FINAL PRO SETUP (PART 4/4)
# Telegram + Analyzer + Final Scanner Logic
# ============================================

# -----------------------------
# TELEGRAM FORMATTER
# -----------------------------
def format_signal_message(sig):
    mode_emoji = "⚡" if sig["mode"]=="QUICK" else ("🔥" if sig["mode"]=="MID" else "🚀")
    dir_emoji = "🟢BUY" if sig["direction"]=="BUY" else "🔴SELL"

    dz_low, dz_high = sig["danger"]
    reason = ", ".join(sig["reasons"])

    return (
        f"{mode_emoji} <b>{dir_emoji} — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code> | SL: <code>{sig['sl']}</code>\n"
        f"Leverage: <b>{sig['lev']}</b>\n"
        f"Score: <b>{sig['score']}</b> | ATR: {sig['atr']}\n"
        f"⚠️ Danger Zone: <code>{dz_low}</code> → <code>{dz_high}</code>\n"
        f"Reason: {reason}\n"
        f"⏰ {pd.to_datetime(sig['ts'], unit='s').strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )


# -----------------------------
# TELEGRAM SEND
# -----------------------------
def send_telegram_message(msg):
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("\n--- TELEGRAM PREVIEW (NOT SENT) ---\n" + msg)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        LOG.warning("Telegram send failed: %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        LOG.warning("Telegram exception: %s", e)
        return False


# -----------------------------
# FINAL ANALYZER
# -----------------------------
def analyze_coin(ex, symbol):
    norm = normalize_symbol(symbol)

    # EXISTS?
    if norm not in ex.markets:
        return None

    # Fetch OHLCV
    df1 = fetch_ohlcv_sync(ex, norm, "1m", 200)
    df5 = fetch_ohlcv_sync(ex, norm, "5m", 200)

    if df1 is None or df1.empty:
        return None

    # Volume filter
    last_vol = float(df1['volume'].iloc[-1])
    if last_vol < MIN_24H_VOLUME:
        return None

    # Orderbook + spread
    ob = fetch_orderbook_safe(ex, norm, 50)
    spread = calc_spread(ob)
    if spread > MAX_SPREAD_PCT:
        return None

    # S/R Filter
    ok_sr, reason_sr = sr_rejection_filter(df1)
    if not ok_sr:
        LOG.debug(f"{symbol} rejected by SR ({reason_sr})")
        return None

    # Top Gainer Filter (momentum check)
    if not top_gainer_filter(df1):
        LOG.debug(f"{symbol} rejected by momentum")
        return None

    # SCORE
    score, reasons = compute_score(df1, df5, ob, spread)
    if score < MIN_SIGNAL_SCORE:
        return None

    # BUILD SIGNAL
    sig = build_signal(symbol, df1, df5, ob, score, reasons)

    # COOL DOWN CHECK
    key = cooldown_key_for(sig["pair"], sig["mode"])
    if _cd_mgr.is_cooled(key):
        return None

    # SET COOLDOWN
    _cd_mgr.set_cooldown(key, sig["mode"])

    return sig


# -----------------------------
# FINAL SCAN LOOP (used by main.py)
# -----------------------------
def scan_batch(ex, batch_symbols):
    emits = 0
    for sym in batch_symbols:
        sig = analyze_coin(ex, sym)
        if sig:
            msg = format_signal_message(sig)
            send_telegram_message(msg)
            LOG.info(f"EMIT {sig['pair']} | {sig['direction']} | {sig['mode']} | score={sig['score']}")
            emits += 1
            if emits >= MAX_EMITS_PER_LOOP:
                break
    return emits