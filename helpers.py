# ============================================
# helpers.py — PART 1 / 4
# PRO MAX SETUP (MTF + BUY/SELL + S/R + BTC CALM)
# ============================================

from __future__ import annotations
import os, time, json, math, logging, requests
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import ccxt
except:
    ccxt = None

# -----------------------------
# ENV + GLOBAL CONFIG
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")

EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET  = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE  = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "86"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "93"))
THRESH_MID   = float(os.getenv("THRESH_MID", "86"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "74"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

TTL_QUICK = 1800   # 30m
TTL_MID   = 1800   # 30m
TTL_TREND = 3600   # 60m

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
# COOLDOWN (Local JSON)
# -----------------------------
class CooldownManager:
    def __init__(self):
        self.store = {}
        self._load()

    def _load(self):
        try:
            with open(COOLDOWN_JSON,"r") as f:
                self.store = {k:int(v) for k,v in json.load(f).items()}
        except:
            self.store = {}

    def _save(self):
        try:
            with open(COOLDOWN_JSON+".tmp","w") as f:
                json.dump(self.store, f)
            os.replace(COOLDOWN_JSON+".tmp", COOLDOWN_JSON)
        except:
            pass

    def ttl_for_mode(self, m):
        m = m.lower()
        if m=="quick": return TTL_QUICK
        if m=="mid":   return TTL_MID
        return TTL_TREND

    def is_active(self, key):
        now = int(time.time())
        exp = self.store.get(key)
        if not exp: return False
        if exp <= now:
            self.store.pop(key,None); self._save()
            return False
        return True

    def set(self, key, mode):
        now = int(time.time())
        ttl = self.ttl_for_mode(mode)
        exp = self.store.get(key,0)
        if exp <= now:
            self.store[key] = now+ttl
            self._save()
            return True
        return False

_cd = CooldownManager()

def cd_key(pair, mode):
    return f"{pair.upper()}::{mode.upper()}"

# -----------------------------
# EXCHANGE
# -----------------------------
def get_exchange():
    if ccxt is None:
        raise RuntimeError("CCXT Missing")
    cls = getattr(ccxt, EXCHANGE_NAME.lower(), ccxt.binance)
    ex = cls({"enableRateLimit": True})
    try: ex.load_markets()
    except: pass
    return ex

def normalize_symbol(s):
    s = s.upper().strip()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# -----------------------------
# OHLCV FETCH
# -----------------------------
def df_from_ohlcv(data):
    if not data: return None
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df if not df.empty else None

def fetch_df(ex, sym, tf="1m", limit=200):
    s = normalize_symbol(sym)
    for i in range(3):
        try:
            d = ex.fetch_ohlcv(s, timeframe=tf, limit=limit)
            time.sleep(0.2)
            return df_from_ohlcv(d)
        except:
            time.sleep(0.3*(i+1))
    return None

# -----------------------------
# INDICATORS
# -----------------------------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    dn = -d.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up/(dn+1e-12)
    return 100 - (100/(1+rs))

def atr(df, length=14):
    h = df["high"]; l=df["low"]; c=df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean().bfill()

# -----------------------------
# BTC CALM FILTER
# -----------------------------
def btc_is_calm(ex):
    df = fetch_df(ex, "BTC/USDT", "1m", 40)
    if df is None or len(df)<10: return True
    vol = df["close"].pct_change().abs().tail(8).mean()
    return vol < 0.0020  # strict btc calm
# -----------------------------------------------------------
# PART 1 END
# -----------------------------------------------------------
# ============================================
# helpers.py — PART 2 / 4
# Liquidity + Spread + S/R + Gainers Priority
# ============================================

# -----------------------------
# ORDERBOOK + LIQUIDITY
# -----------------------------
def fetch_orderbook_safe(ex, sym, limit=50):
    try:
        ob = ex.fetch_order_book(normalize_symbol(sym), limit=limit)
        time.sleep(0.15)
        return ob or {}
    except:
        return {}

def orderbook_imbalance(ob, depth=12):
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    b = sum([float(x[1]) for x in bids[:depth]]) if bids else 0.0
    a = sum([float(x[1]) for x in asks[:depth]]) if asks else 0.0
    total = b + a + 1e-12
    return (b - a) / total

def liquidity_score(ob):
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    if not bids and not asks:
        return 0.0

    top5  = (sum([float(x[1]) for x in bids[:5]]) +
             sum([float(x[1]) for x in asks[:5]])) / 2.0

    top20 = (sum([float(x[1]) for x in bids[:20]]) +
             sum([float(x[1]) for x in asks[:20]])) / 2.0

    if top20 <= 0:
        return 0.0

    score = top5 / (top20 + 1e-12)
    return round(min(1.0, score), 3)

def calc_spread(ob):
    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
        return abs(ask - bid) / ask
    except:
        return 999.0


# -----------------------------
# AUTO S/R (3-Pivot Level Scan)
# -----------------------------
def support_resistance_levels(df):
    closes = df["close"].values
    pivots = []
    for i in range(2, len(closes)-2):
        c = closes[i]
        if c < closes[i-1] and c < closes[i+1] and c < closes[i-2] and c < closes[i+2]:
            pivots.append(("S", c))
        if c > closes[i-1] and c > closes[i+1] and c > closes[i-2] and c > closes[i+2]:
            pivots.append(("R", c))
    return pivots[-6:] if len(pivots) > 6 else pivots

def s_r_conflict(entry, pivots):
    """
    Returns True if entry is too close to S or R (fake move probability high)
    """
    for t, lvl in pivots:
        if abs(entry - lvl) < entry * 0.003:   # within 0.3%
            return True
    return False


# -----------------------------
# TOP VOLUME + GAINERS FILTER
# -----------------------------
def get_top_volume_symbols(ex):
    try:
        tickers = ex.fetch_tickers()
    except:
        return []

    vols = []
    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"): continue
        vol = tk.get("quoteVolume", 0)
        vols.append((sym, vol))

    vols = sorted(vols, key=lambda x: x[1], reverse=True)
    return [v[0] for v in vols[:120]]  # top 120 only


def get_top_gainers(ex):
    try:
        tickers = ex.fetch_tickers()
    except:
        return []

    changes = []
    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"): continue
        ch = tk.get("percentage", 0)
        changes.append((sym, ch))

    changes = sorted(changes, key=lambda x: x[1], reverse=True)
    return [c[0] for c in changes[:50]]  # strongest movers only
# -----------------------------------------------------------
# PART 2 END
# -----------------------------------------------------------
# ============================================
# helpers.py — PART 3 / 4
# MTF scoring + BUY/SELL logic + TP/SL Model
# ============================================

# -----------------------------
# MULTI-TIMEFRAME SCORING (1m + 5m)
# -----------------------------
def compute_score(df1, df5, ob, spread):
    reasons = []
    score = 40.0

    close1 = df1["close"]
    e20_1 = ema(close1, 20).iloc[-1]
    e50_1 = ema(close1, 50).iloc[-1]

    # 5m fallback
    if df5 is not None and not df5.empty:
        close5 = df5["close"]
        e20_5 = ema(close5, 20).iloc[-1]
        e50_5 = ema(close5, 50).iloc[-1]
    else:
        e20_5, e50_5 = e20_1, e50_1

    # EMA alignment (1m + 5m)
    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    else:
        reasons.append("EMA20<=50_1m")

    if e20_5 > e50_5:
        score += 8; reasons.append("EMA20>50_5m")
    else:
        reasons.append("EMA20<=50_5m")

    # MACD-like
    macd_val = (ema(close1,12) - ema(close1,26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # RSI strength
    r = float(rsi(close1).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")
    else:
        score += 1; reasons.append("RSI_high")

    # Volume spike
    vol = df1["volume"]
    avg = vol.rolling(20).mean().iloc[-1]
    if vol.iloc[-1] > avg * 1.6:
        score += 12; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # Orderbook imbalance
    obi = orderbook_imbalance(ob)
    if obi > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    elif obi < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    # Liquidity score
    liq = liquidity_score(ob)
    if liq > 0.55:
        score += 8; reasons.append("Liquidity_ok")
    else:
        reasons.append("Liquidity_low")

    # Spread filter
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = max(0, min(100, score))
    return round(score, 1), reasons


# -----------------------------
# BUY/SELL DIRECTION MODEL PRO
# -----------------------------
def detect_direction(df1, ob):
    e20 = ema(df1["close"], 20).iloc[-1]
    e50 = ema(df1["close"], 50).iloc[-1]
    obi = orderbook_imbalance(ob)

    # Strong sell indication
    if e20 < e50 and obi < -0.40:
        return "SELL"

    # Balanced → BUY default
    return "BUY"


# -----------------------------
# TP/SL ADVANCED MODEL (by ATR)
# QUICK / MID / TREND
# -----------------------------
def calculate_tp_sl(entry, atr_val, mode, direction):
    if mode == "QUICK":
        tp_off = atr_val * 1.4
        sl_off = atr_val * 0.9
    elif mode == "TREND":
        tp_off = atr_val * 3.2
        sl_off = atr_val * 2.0
    else:  # MID
        tp_off = atr_val * 2.1
        sl_off = atr_val * 1.3

    if direction == "BUY":
        tp = entry + tp_off
        sl = entry - sl_off
    else:
        tp = entry - tp_off
        sl = entry + sl_off

    return round(tp,8), round(sl,8)


# -----------------------------
# DANGER ZONE FIX (BUY + SELL)
# -----------------------------
def compute_danger_zone(entry, atr_val):
    low = round(entry - atr_val*1.05, 8)
    high = round(entry + atr_val*1.05, 8)
    return (low, high)


# -----------------------------
# FINAL SIGNAL BUILDER
# -----------------------------
def build_signal(symbol, df1, df5, ob, score, reasons):
    entry = float(df1["close"].iloc[-1])
    atr_val = float(atr(df1).iloc[-1])

    # mode selection
    if score >= THRESH_QUICK:
        mode = "QUICK"
    elif score >= THRESH_MID:
        mode = "MID"
    else:
        mode = "TREND"

    direction = detect_direction(df1, ob)
    tp, sl = calculate_tp_sl(entry, atr_val, mode, direction)
    dz = compute_danger_zone(entry, atr_val)

    lev = "50x" if mode=="QUICK" else ("25x" if mode=="MID" else "15x")

    return {
        "pair": normalize_symbol(symbol),
        "mode": mode,
        "score": score,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "atr": atr_val,
        "direction": direction,
        "danger": dz,
        "reasons": reasons,
        "lev_suggest": lev,
        "ts": int(time.time())
    }
# -----------------------------------------------------------
# PART 3 END
# -----------------------------------------------------------
# ============================================
# helpers.py — PART 4 / 4
# Formatter + Telegram + Analyzer + Cooldown
# ============================================


# -----------------------------
# TELEGRAM FORMATTER (Emoji PRO)
# -----------------------------
def format_signal_message(sig):
    mode_emoji = "⚡" if sig["mode"]=="QUICK" else ("🔥" if sig["mode"]=="MID" else "🚀")
    dir_emoji  = "⬆️ BUY" if sig["direction"]=="BUY" else "⬇️ SELL"

    dz_low, dz_high = sig["danger"]
    reason_txt = ", ".join(sig["reasons"])

    msg = (
        f"{mode_emoji} <b>{dir_emoji} — {sig['mode']} MODE</b>\n"
        f"Pair: <b>{sig['pair']}</b>\n\n"
        f"🎯 Entry: <code>{sig['entry']}</code>\n"
        f"🏆 TP: <code>{sig['tp']}</code>\n"
        f"🛑 SL: <code>{sig['sl']}</code>\n"
        f"⚙️ Leverage: <b>{sig['lev_suggest']}</b>\n"
        f"📊 Score: <b>{sig['score']}</b>\n"
        f"ATR: {sig['atr']}\n\n"
        f"⚠️ Danger Zone:\n<code>{dz_low}</code> → <code>{dz_high}</code>\n\n"
        f"🔍 Reason: {reason_txt}\n"
        f"⏰ Time: {pd.to_datetime(sig['ts'], unit='s').strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    return msg


# -----------------------------
# TELEGRAM SENDER
# -----------------------------
def send_telegram_message(msg):
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Preview (NOT SENT):\n" + msg)
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
        LOG.warning(f"Telegram Error {r.status_code}: {r.text}")
        return False
    except Exception as e:
        LOG.warning(f"Telegram send exception: {e}")
        return False


# -----------------------------
# FINAL ANALYZER (with ALL filters)
# -----------------------------
def analyze_coin(ex, symbol):
    try:
        symbol_n = normalize_symbol(symbol)

        # check market exists
        if symbol_n not in ex.markets:
            return None

        # fetch OHLCV
        df1 = fetch_ohlcv_sync(ex, symbol_n, timeframe="1m", limit=200)
df5 = fetch_ohlcv_sync(ex, symbol_n, timeframe="5m", limit=200)

        if df1 is None or df1.empty:
            return None

        # volume filter
        vol_last = float(df1["volume"].iloc[-1])
        if vol_last < MIN_24H_VOLUME:
            return None

        # orderbook + spread
        ob = fetch_orderbook_safe(ex, symbol_n)
        spread = calc_spread(ob)
        if spread > MAX_SPREAD_PCT:
            return None

        # S/R filter
        pivots = support_resistance_levels(df1)
        entry = float(df1["close"].iloc[-1])
        if s_r_conflict(entry, pivots):
            return None

        # compute score
        score, reasons = compute_score(df1, df5, ob, spread)
        if score < MIN_SIGNAL_SCORE:
            return None

        # cooldown check
        # same coin 30–60 min block
        if score >= THRESH_QUICK:
            mode = "QUICK"
        elif score >= THRESH_MID:
            mode = "MID"
        else:
            mode = "TREND"

        key = cooldown_key_for(symbol_n, mode)
        if _cd_mgr.is_cooled(key):
            return None   # skip same coin too soon

        # build final signal
        sig = build_signal(symbol_n, df1, df5, ob, score, reasons)

        # set cooldown
        _cd_mgr.set_cooldown(key, mode)

        return sig

    except Exception as e:
        LOG.exception(f"analyze_coin error {symbol}: {e}")
        return None


# -----------------------------
# END OF HELPERS.PY
# -----------------------------