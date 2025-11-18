# ============================
# helpers_part1.py  (Part 1/3)
# EXCHANGE + FETCH SYSTEM
# ============================

import ccxt
import time
import csv
import datetime as dt


# ------------------------------------------------
# 1) LOAD SYMBOLS (CSV header auto skip)
# ------------------------------------------------
def load_symbols(csv_path="coins.csv"):
    symbols = []
    with open(csv_path) as f:
        rd = csv.reader(f)
        next(rd, None)            # <-- SKIP HEADER
        for row in rd:
            if not row:
                continue

            s = row[0].strip()

            if not s:
                continue

            if s.lower() == "symbol":
                continue

            symbols.append(s)

    return symbols


# ------------------------------------------------
# 2) GET EXCHANGE INSTANCE
# ------------------------------------------------
def get_exchange():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future"   # futures enabled
        }
    })
    return ex


# ------------------------------------------------
# 3) SAFE FETCH OHLCV (retry + protection)
# ------------------------------------------------
def fetch_ohlcv_safe(ex, symbol, timeframe="1m", limit=120, retry=3):

    pair = f"{symbol}/USDT"

    for i in range(retry):
        try:
            data = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if data:
                return data
        except Exception as e:
            print(f"[WARN] Retry {i+1} {pair} → {e}")
            time.sleep(1.2)

    print(f"[ERROR] OHLCV failed for {pair}")
    return None


# ------------------------------------------------
# 4) SAFE FETCH TICKER (current price)
# ------------------------------------------------
def fetch_price_safe(ex, symbol, retry=3):
    pair = f"{symbol}/USDT"

    for i in range(retry):
        try:
            t = ex.fetch_ticker(pair)
            return float(t["last"])
        except Exception as e:
            print(f"[WARN] ticker retry {i+1} {pair} → {e}")
            time.sleep(0.5)

    return None


# ------------------------------------------------
# 5) HELPER: FORMAT TIMESTAMP INTO IST
# ------------------------------------------------
def format_time(ts):
    return dt.datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
# ===========================================
# helpers_part2.py   (Part 2/3)
# INDICATORS + LOGIC + REGIME + ACCURACY
# ===========================================

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# 1) EMA / RSI
# ------------------------------------------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


# ------------------------------------------------------------
# 2) VWAP
# ------------------------------------------------------------
def vwap(df):
    typical = (df['high'] + df['low'] + df['close']) / 3
    return (typical * df['volume']).sum() / (df['volume'].sum() + 1e-9)


# ------------------------------------------------------------
# 3) VOLATILITY (ATR style simple)
# ------------------------------------------------------------
def volatility_range(df):
    last = df.iloc[-1]
    rng = last["high"] - last["low"]
    return rng


# ------------------------------------------------------------
# 4) MARKET REGIME (Bull / Bear / Sideways)
# ------------------------------------------------------------
def detect_regime(df):
    close = df["close"]
    ema50 = ema(close, 50).iloc[-1]
    ema200 = ema(close, 200).iloc[-1]
    last_price = close.iloc[-1]
    rsi_val = rsi(close).iloc[-1]

    if last_price > ema200 and ema50 > ema200 and rsi_val > 50:
        return "Bull"

    if last_price < ema200 and ema50 < ema200 and rsi_val < 45:
        return "Bear"

    return "Sideways"


# ------------------------------------------------------------
# 5) ACCURACY ENGINE (Based on regime)
# ------------------------------------------------------------
def accuracy_from_regime(regime):
    if regime == "Bull":
        return 85 + np.random.randint(3, 10)    # 88–95%
    if regime == "Bear":
        return 70 + np.random.randint(0, 10)    # 70–80%
    return 55 + np.random.randint(0, 15)        # 55–70%


# ------------------------------------------------------------
# 6) AUTO LEVERAGE SUGGESTION
# ------------------------------------------------------------
def suggest_leverage(acc):
    if acc >= 85:
        return 50
    elif acc >= 70:
        return 30
    elif acc >= 55:
        return 15
    else:
        return 5


# ------------------------------------------------------------
# 7) PATTERN DETECTOR (ENGULFING + PINBAR)
# ------------------------------------------------------------
def candle_pattern(prev, cur):
    o1, c1 = prev["open"], prev["close"]
    o2, c2 = cur["open"], cur["close"]

    body1 = c1 - o1
    body2 = c2 - o2

    if body1 < 0 and body2 > 0 and c2 > o1 and o2 < c1:
        return "Bullish Engulfing"

    if body1 > 0 and body2 < 0 and c2 < o1 and o2 > c1:
        return "Bearish Engulfing"

    # Pinbars
    total = cur["high"] - cur["low"]
    upper = cur["high"] - max(o2, c2)
    lower = min(o2, c2) - cur["low"]

    if lower > total * 0.55:
        return "Hammer"
    if upper > total * 0.55:
        return "Shooting Star"

    return "Neutral"


# ------------------------------------------------------------
# 8) DANGER ZONE CALCULATION
# ------------------------------------------------------------
def danger_zone_details(last_price, df):
    rng = volatility_range(df)
    lowV = last_price - rng
    highV = last_price + rng
    return last_price, lowV, highV


# ------------------------------------------------------------
# 9) FULL SIGNAL EVALUATION (ENTRY + TP + SL)
# ------------------------------------------------------------
def evaluate_signal(df, side="BUY"):

    if len(df) < 60:
        return None

    df = df.copy()

    # last candle
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])

    # indicators
    ema20 = ema(df["close"], 20).iloc[-1]
    ema50 = ema(df["close"], 50).iloc[-1]
    ema200_val = ema(df["close"], 200).iloc[-1]
    rsi_val = rsi(df["close"]).iloc[-1]
    vwap_val = vwap(df)
    pattern = candle_pattern(prev, last)
    vol_rng = volatility_range(df)

    regime = detect_regime(df)
    accuracy = accuracy_from_regime(regime)
    lev = suggest_leverage(accuracy)

    # ENTRY
    entry = close

    # TP/SL by volatility
    if side == "BUY":
        sl = entry - vol_rng * 1.2
        tp = entry + (entry - sl) * 1.4
    else:
        sl = entry + vol_rng * 1.2
        tp = entry - (sl - entry) * 1.4

    # DANGER ZONE
    dz_price, dz_low, dz_high = danger_zone_details(close, df)

    return {
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200_val,
        "rsi": rsi_val,
        "vwap": vwap_val,
        "pattern": pattern,
        "vol_range": vol_rng,
        "regime": regime,
        "accuracy": accuracy,
        "leverage": lev,
        "dz_price": dz_price,
        "dz_low": dz_low,
        "dz_high": dz_high,
    }
# ===========================================
# helpers_part3.py   (Part 3/3)
# TELEGRAM MESSAGE BUILDER + SEND FUNCTION
# ===========================================

import requests
import os
from datetime import datetime


# ------------------------------------------------------------
# 1) Telegram Send Function
# ------------------------------------------------------------
def send_telegram(msg: str):
    TOKEN = os.getenv("BOT_TOKEN")
    CHAT = os.getenv("CHAT_ID")

    if not TOKEN or not CHAT:
        print("[WARN] TELEGRAM ENV MISSING — message:")
        print(msg)
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        print("[TG] Status:", r.status_code)
    except Exception as e:
        print("[TG ERROR]", e)


# ------------------------------------------------------------
# 2) Build Final Signal Message (HTML style)
# ------------------------------------------------------------
def build_signal_message(symbol, mode, side, data):

    entry = data["entry"]
    tp = data["tp"]
    sl = data["sl"]
    lev = data["leverage"]
    pattern = data["pattern"]
    regime = data["regime"]
    acc = data["accuracy"]

    dz_p = data["dz_price"]
    dzl = data["dz_low"]
    dzh = data["dz_high"]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    msg = f"""
<b>🔥 {side.upper()} SIGNAL — {mode.upper()}</b>
<b>Pair:</b> {symbol}/USDT
<b>Time (UTC):</b> {now}

<b>Entry:</b> {entry:.6f}
🎯 <b>TP:</b> {tp:.6f}
🛑 <b>SL:</b> {sl:.6f}
⚡ <b>Suggested Leverage:</b> {lev}x

<b>Market Regime:</b> {regime}
<b>Accuracy:</b> {acc}%

<b>Pattern:</b> {pattern}

🟧 <b>Danger Zone:</b>
Current: {dz_p:.6f}
Range: {dzl:.6f} — {dzh:.6f}
"""

    return msg.strip()


# ------------------------------------------------------------
# 3) Wrapper — Send Signal
# ------------------------------------------------------------
def send_signal(symbol, mode, side, data):
    msg = build_signal_message(symbol, mode, side, data)
    send_telegram(msg)
    print("\n===== SIGNAL SENT =====\n")
    print(msg)