# ============================================================
# helpers.py — FINAL CLEAN SAFE PRO SETUP (NO REDIS VERSION)
# ============================================================

import os, time, math, logging, json
import pandas as pd
from typing import Dict, Any, Optional, List
import ccxt
import requests

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel("INFO")

# ------------------------------------------------------------
# ENV CONFIG
# ------------------------------------------------------------
EXCHANGE_NAME = "binance"
QUOTE = "USDT"

MIN_SCORE = 82
TH_QUICK = 90
TH_MID   = 82
TH_TREND = 70

MIN_VOL = 120000          # volume filter
MAX_SPREAD = 0.004        # 0.4%
COOLDOWN_SEC = 1800       # 30 min same coin block

NOTIFY_ONLY = True
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# cooldown file
COOLDOWN_FILE = "cooldown.json"

# ------------------------------------------------------------
# COOL DOWN — JSON fallback only
# ------------------------------------------------------------
def load_cooldown():
    try:
        return json.load(open(COOLDOWN_FILE, "r"))
    except:
        return {}

def save_cooldown(data):
    try:
        json.dump(data, open(COOLDOWN_FILE, "w"))
    except:
        pass

cool_db = load_cooldown()

def cooldown_key(symbol, mode):
    return f"{symbol}::{mode}"

def check_and_set_cooldown(symbol, mode):
    key = cooldown_key(symbol, mode)
    now = int(time.time())
    exp = cool_db.get(key, 0)
    if exp > now:
        return False
    cool_db[key] = now + COOLDOWN_SEC
    save_cooldown(cool_db)
    return True

# ------------------------------------------------------------
# Exchange
# ------------------------------------------------------------
def get_ex():
    ex = ccxt.binance({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()
    return ex

def norm(symbol):
    s = symbol.upper()
    if "/" not in s:
        s = f"{s}/{QUOTE}"
    return s

# ------------------------------------------------------------
# Fetch OHLCV safe
# ------------------------------------------------------------
def fetch_ohlcv(ex, symbol, tf="1m", limit=200):
    try:
        data = ex.fetch_ohlcv(norm(symbol), timeframe=tf, limit=limit)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except:
        return None

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n).mean()
    rs = up/(dn+1e-12)
    return 100 - 100/(1+rs)

def atr(df, n=14):
    high = df["high"]; low = df["low"]; close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().bfill()

# ------------------------------------------------------------
# Spread
# ------------------------------------------------------------
def fetch_orderbook(ex, symbol):
    try:
        return ex.fetch_order_book(norm(symbol))
    except:
        return {}

def calc_spread(ob):
    try:
        bid = ob["bids"][0][0]
        ask = ob["asks"][0][0]
        return abs(ask - bid) / ask
    except:
        return 999

def orderbook_imb(ob):
    try:
        bids = sum([b[1] for b in ob["bids"][:10]])
        asks = sum([a[1] for a in ob["asks"][:10]])
        return (bids - asks) / (bids + asks + 1e-9)
    except:
        return 0

# ------------------------------------------------------------
# BTC Stability (1m)
# ------------------------------------------------------------
def is_btc_stable(ex):
    df = fetch_ohlcv(ex, "BTC/USDT", "1m", 20)
    if df is None: return False
    v = df["close"].pct_change().abs().tail(5).mean()
    return v < 0.0018     # stable condition

# ------------------------------------------------------------
# Scoring — (Volume + EMA + MACD + RSI + Spread + OBI)
# ------------------------------------------------------------
def compute_score(df, ob_imb, spread):
    close = df["close"]
    score = 40
    reasons = []

    # EMA
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]

    if e20 > e50: score += 12; reasons.append("EMA20>50")
    if e20 > e200: score += 8; reasons.append("EMA20>200")

    # MACD
    macd = (ema(close,12)-ema(close,26)).iloc[-1]
    if macd > 0: score += 12; reasons.append("MACD_pos")

    # RSI
    R = rsi(close).iloc[-1]
    if 40 < R < 70:
        score += 8; reasons.append("RSI_ok")

    # Volume Spike
    vol = df["volume"]
    if vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.5:
        score += 10; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # Spread
    if spread < MAX_SPREAD:
        score += 6; reasons.append("Spread_ok")

    # OBI
    if ob_imb > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    if ob_imb < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    return round(min(score,100),1), reasons

# ------------------------------------------------------------
# Build Signal
# ------------------------------------------------------------
def build_signal(symbol, df, score, reasons):
    last = df.iloc[-1]
    entry = float(last["close"])
    at = float(atr(df).iloc[-1] or 0.0)

    # Mode
    mode = "MID"
    if score >= TH_QUICK: mode = "QUICK"
    elif score >= TH_MID: mode = "MID"
    else: mode = "TREND"

    # Direction
    direction = "BUY"
    if ema(df["close"],20).iloc[-1] < ema(df["close"],50).iloc[-1]:
        direction = "SELL"

    # TP/SL by ATR
    if mode == "QUICK":
        rr_tp, rr_sl = (1.5, 1.0)
    elif mode == "TREND":
        rr_tp, rr_sl = (3.0, 2.0)
    else:
        rr_tp, rr_sl = (2.0, 1.4)

    if direction == "BUY":
        tp = entry + at * rr_tp
        sl = entry - at * rr_sl
    else:
        tp = entry - at * rr_tp
        sl = entry + at * rr_sl

    danger = (round(entry - at*1.1,8), round(entry + at*1.1,8))

    return {
        "symbol": norm(symbol),
        "entry": round(entry,8),
        "tp": round(tp,8),
        "sl": round(sl,8),
        "atr": round(at,8),
        "mode": mode,
        "direction": direction,
        "score": score,
        "reasons": reasons,
        "danger": danger,
        "ts": int(time.time())
    }

# ------------------------------------------------------------
# Telegram
# ------------------------------------------------------------
def tg_format(sig):
    em = "⚡" if sig["mode"]=="QUICK" else ("🔥" if sig["mode"]=="MID" else "🚀")
    dz_low, dz_high = sig["danger"]
    R = ", ".join(sig["reasons"])

    return (
        f"{em} <b>{sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['symbol']}</b>  Score: <b>{sig['score']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Leverage: 50x (manual)\n"
        f"⚠️ Danger Zone: <code>{dz_low}</code> - <code>{dz_high}</code>\n"
        f"Reason: {R}\n"
        f"Time: {pd.to_datetime(sig['ts'], unit='s').strftime('%H:%M:%S')} UTC"
    )

def tg_send(msg):
    if NOTIFY_ONLY:
        LOG.info("PREVIEW:\n%s", msg)
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram disabled")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

# ------------------------------------------------------------
# MAIN ANALYZE
# ------------------------------------------------------------
def analyze(ex, symbol):
    df = fetch_ohlcv(ex, symbol, "1m", 200)
    if df is None or len(df) < 100:
        return None

    # volume filter
    if df["volume"].iloc[-1] < MIN_VOL:
        return None

    # spread
    ob = fetch_orderbook(ex, symbol)
    spread = calc_spread(ob)
    if spread > MAX_SPREAD:
        return None

    # score
    obi = orderbook_imb(ob)
    score, reasons = compute_score(df, obi, spread)
    if score < MIN_SCORE:
        return None

    sig = build_signal(symbol, df, score, reasons)

    # cooldown check
    if not check_and_set_cooldown(symbol, sig["mode"]):
        return None

    return sig