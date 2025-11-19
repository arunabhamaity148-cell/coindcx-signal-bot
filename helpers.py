# helpers_part1.py — PART 1/2
# (imports, config, cooldown, exchange, OHLCV, indicators, BTC calm,
#  orderbook/liquidity, S/R pivots, top-volume/gainers, scoring)

from __future__ import annotations
import os
import time
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

# optional dependency
try:
    import ccxt
except Exception:
    ccxt = None

# -----------------------------
# ENV + GLOBAL CONFIG
# -----------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1", "true", "yes")

EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "86"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "93"))
THRESH_MID = float(os.getenv("THRESH_MID", "86"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "74"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "1800"))  # 30m
TTL_MID = int(os.getenv("COOLDOWN_MID_S", "1800"))  # 30m
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))  # 60m

EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

# -----------------------------
# LOGGING
# -----------------------------
LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))

# -----------------------------
# Cooldown (local JSON)
# -----------------------------
class CooldownManager:
    def __init__(self, path: str = COOLDOWN_JSON):
        self.path = path
        self.store: Dict[str, int] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            self.store = {k: int(v) for k, v in data.items()}
        except Exception:
            self.store = {}

    def _save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.store, f)
            os.replace(tmp, self.path)
        except Exception:
            LOG.debug("CooldownManager: save failed", exc_info=True)

    def ttl_for_mode(self, mode: str) -> int:
        m = mode.lower()
        if m == "quick":
            return TTL_QUICK
        if m == "mid":
            return TTL_MID
        return TTL_TREND

    def is_active(self, key: str) -> bool:
        now = int(time.time())
        exp = self.store.get(key)
        if not exp:
            return False
        if exp <= now:
            self.store.pop(key, None)
            self._save()
            return False
        return True

    def set(self, key: str, mode: str) -> bool:
        now = int(time.time())
        ttl = self.ttl_for_mode(mode)
        exp = self.store.get(key, 0)
        if exp <= now:
            self.store[key] = now + ttl
            self._save()
            return True
        return False

_cd = CooldownManager(COOLDOWN_JSON)

def cd_key(pair: str, mode: str) -> str:
    return f"{pair.upper()}::{mode.upper()}"

# -----------------------------
# Exchange helpers
# -----------------------------
def get_exchange(api_keys: bool = False) -> "ccxt.Exchange":
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex_name = EXCHANGE_NAME.lower()
    try:
        ex_cls = getattr(ccxt, ex_name, None) or ccxt.binance
        ex = ex_cls({"enableRateLimit": True})
    except Exception:
        ex = ccxt.binance({"enableRateLimit": True})
    # optional keys
    if api_keys and os.getenv("EXCHANGE_API_KEY") and os.getenv("EXCHANGE_API_SECRET"):
        ex.apiKey = os.getenv("EXCHANGE_API_KEY")
        ex.secret = os.getenv("EXCHANGE_API_SECRET")
    try:
        ex.load_markets()
    except Exception:
        LOG.debug("exchange.load_markets failed/ignored")
    return ex

def normalize_symbol(s: str) -> str:
    s = s.strip().upper()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# -----------------------------
# OHLCV Fetch & Dataframe helpers
# -----------------------------
def df_from_ohlcv(data) -> Optional[pd.DataFrame]:
    if not data:
        return None
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df if not df.empty else None

def fetch_df(ex, sym: str, tf: str = "1m", limit: int = 200, retries: int = 3) -> Optional[pd.DataFrame]:
    s = normalize_symbol(sym)
    last_exc = None
    for attempt in range(retries):
        try:
            data = ex.fetch_ohlcv(s, timeframe=tf, limit=limit)
            time.sleep(0.18)
            return df_from_ohlcv(data)
        except Exception as e:
            last_exc = e
            time.sleep(0.25 * (attempt + 1))
    LOG.debug("fetch_df failed %s: %s", s, last_exc)
    return None

# compatibility wrapper expected by analyzer
def fetch_ohlcv_sync(ex, sym: str, timeframe: str = "1m", limit: int = 200, retries: int = 3) -> Optional[pd.DataFrame]:
    # prefer existing fetch_df
    try:
        return fetch_df(ex, sym, tf=timeframe, limit=limit, retries=retries)
    except Exception:
        # fallback direct fetch
        s = normalize_symbol(sym)
        last_exc = None
        for attempt in range(retries):
            try:
                data = ex.fetch_ohlcv(s, timeframe=timeframe, limit=limit)
                time.sleep(0.18)
                return df_from_ohlcv(data)
            except Exception as e:
                last_exc = e
                time.sleep(0.25 * (attempt + 1))
        LOG.debug("fetch_ohlcv_sync fallback failed %s: %s", s, last_exc)
        return None

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    dn = -d.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()

# -----------------------------
# BTC calm filter
# -----------------------------
def btc_is_calm(ex: "ccxt.Exchange") -> bool:
    try:
        df = fetch_ohlcv_sync(ex, "BTC/USDT", timeframe="1m", limit=40)
    except Exception:
        return True
    if df is None or len(df) < 10:
        return True
    vol = df["close"].pct_change().abs().tail(8).mean()
    return vol < 0.0020

# -----------------------------
# Orderbook / Liquidity / Spread
# -----------------------------
def fetch_orderbook_safe(ex: "ccxt.Exchange", sym: str, limit: int = 50) -> dict:
    try:
        ob = ex.fetch_order_book(normalize_symbol(sym), limit=limit)
        time.sleep(0.15)
        return ob or {}
    except Exception:
        return {}

def orderbook_imbalance(ob: dict, depth: int = 12) -> float:
    if not ob:
        return 0.0
    bids = ob.get("bids", []) or []
    asks = ob.get("asks", []) or []
    bid_vol = sum([float(x[1]) for x in bids[:depth]]) if bids else 0.0
    ask_vol = sum([float(x[1]) for x in asks[:depth]]) if asks else 0.0
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

def liquidity_score(ob: dict) -> float:
    bids = ob.get("bids", []) or []
    asks = ob.get("asks", []) or []
    if not bids and not asks:
        return 0.0
    top5 = (sum([float(x[1]) for x in bids[:5]]) + sum([float(x[1]) for x in asks[:5]])) / 2.0
    top20 = (sum([float(x[1]) for x in bids[:20]]) + sum([float(x[1]) for x in asks[:20]])) / 2.0
    if top20 <= 0:
        return 0.0
    score = top5 / (top20 + 1e-12)
    return round(min(1.0, score), 3)

def calc_spread_from_ob(ob: dict) -> float:
    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
        return abs(ask - bid) / ask
    except Exception:
        return 999.0

# -----------------------------
# Support/Resistance (simple pivot detection)
# -----------------------------
def support_resistance_levels(df: pd.DataFrame) -> List[Tuple[str, float]]:
    closes = df["close"].values
    pivots: List[Tuple[str, float]] = []
    for i in range(2, len(closes) - 2):
        c = closes[i]
        if c < closes[i - 1] and c < closes[i + 1] and c < closes[i - 2] and c < closes[i + 2]:
            pivots.append(("S", float(c)))
        if c > closes[i - 1] and c > closes[i + 1] and c > closes[i - 2] and c > closes[i + 2]:
            pivots.append(("R", float(c)))
    if len(pivots) > 6:
        return pivots[-6:]
    return pivots

def s_r_conflict(entry: float, pivots: List[Tuple[str, float]]) -> bool:
    for t, lvl in pivots:
        if abs(entry - lvl) < entry * 0.003:  # within 0.3%
            return True
    return False

# -----------------------------
# Top volume & gainers helpers
# -----------------------------
def get_top_volume_symbols(ex: "ccxt.Exchange") -> List[str]:
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        return []
    vols = []
    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        vol = tk.get("quoteVolume", 0) or tk.get("baseVolume", 0) or 0
        vols.append((sym, float(vol)))
    vols = sorted(vols, key=lambda x: x[1], reverse=True)
    return [v[0] for v in vols[:120]]

def get_top_gainers(ex: "ccxt.Exchange") -> List[str]:
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        return []
    changes = []
    for sym, tk in tickers.items():
        if not sym.endswith("/USDT"):
            continue
        ch = tk.get("percentage", 0) or tk.get("change", 0) or 0
        try:
            chf = float(ch)
        except Exception:
            chf = 0.0
        changes.append((sym, chf))
    changes = sorted(changes, key=lambda x: x[1], reverse=True)
    return [c[0] for c in changes[:50]]

# -----------------------------
# Scoring engine (MTF + liquidity + OBI + spread + vol)
# -----------------------------
def compute_score(df1: pd.DataFrame, df5: Optional[pd.DataFrame], ob: dict, spread: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 40.0

    close1 = df1["close"]
    e20_1 = ema(close1, EMA_FAST).iloc[-1]
    e50_1 = ema(close1, EMA_SLOW).iloc[-1]

    if df5 is not None and not df5.empty:
        close5 = df5["close"]
        e20_5 = ema(close5, EMA_FAST).iloc[-1]
        e50_5 = ema(close5, EMA_SLOW).iloc[-1]
    else:
        e20_5, e50_5 = e20_1, e50_1

    # EMA alignment
    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    else:
        reasons.append("EMA20<=50_1m")

    if e20_5 > e50_5:
        score += 8; reasons.append("EMA20>50_5m")
    else:
        reasons.append("EMA20<=50_5m")

    # MACD-like
    macd_val = (ema(close1, 12) - ema(close1, 26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # RSI
    r = float(rsi(close1).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")
    else:
        score += 1; reasons.append("RSI_high")

    # Volume spike
    vol = df1["volume"]
    vol_avg = vol.rolling(20, min_periods=1).mean().iloc[-1] or 1.0
    if vol.iloc[-1] > vol_avg * 1.6:
        score += 12; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # Orderbook imbalance
    obi = orderbook_imbalance(ob)
    if obi > 0.55:
        score += 6; reasons.append("OB_buy_pressure")
    elif obi < -0.55:
        score += 6; reasons.append("OB_sell_pressure")

    # Liquidity
    liq = liquidity_score(ob)
    if liq > 0.55:
        score += 8; reasons.append("Liquidity_ok")
    else:
        reasons.append("Liquidity_low")

    # Spread
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = max(0.0, min(100.0, score))
    return round(score, 1), reasons

# alias for compatibility
def compute_score_and_reasons(df1, df5, ob, spread):
    return compute_score(df1, df5, ob, spread)
# helpers_part2.py — PART 2/2
# (direction, TP/SL, signal builder, telegram, passes filters, analyzer)

from typing import Any, Dict, Tuple, Optional, List
import pandas as pd
import time
import logging

# re-use functions/vars from part1: normalize_symbol, atr, ema, orderbook_imbalance,
# liquidity_score, calc_spread_from_ob, support_resistance_levels, s_r_conflict,
# get_top_volume_symbols, get_top_gainers, compute_score_and_reasons, fetch_ohlcv_sync,
# fetch_orderbook_safe, _cd, cd_key

# -----------------------------
# Direction + TP/SL + danger
# -----------------------------
def detect_direction(df1: pd.DataFrame, ob: dict) -> str:
    e20 = ema(df1["close"], EMA_FAST).iloc[-1]
    e50 = ema(df1["close"], EMA_SLOW).iloc[-1]
    obi = orderbook_imbalance(ob)
    if e20 < e50 and obi < -0.40:
        return "SELL"
    return "BUY"

def calculate_tp_sl(entry: float, atr_val: float, mode: str, direction: str) -> Tuple[float, float]:
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

    return round(tp, 8), round(sl, 8)

def compute_danger_zone(entry: float, atr_val: float) -> Tuple[float, float]:
    low = round(entry - atr_val * 1.05, 8)
    high = round(entry + atr_val * 1.05, 8)
    return (low, high)

def build_signal(symbol: str, df1: pd.DataFrame, df5: Optional[pd.DataFrame], ob: dict, score: float, reasons: List[str]) -> Dict[str, Any]:
    entry = float(df1["close"].iloc[-1])
    atr_val = float(atr(df1).iloc[-1])
    if score >= THRESH_QUICK:
        mode = "QUICK"
    elif score >= THRESH_MID:
        mode = "MID"
    else:
        mode = "TREND"

    direction = detect_direction(df1, ob)
    tp, sl = calculate_tp_sl(entry, atr_val, mode, direction)
    dz = compute_danger_zone(entry, atr_val)
    lev = "50x" if mode == "QUICK" else ("25x" if mode == "MID" else "15x")

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

# compatibility builder
def build_signal_from_df(symbol, df1, df5, ob, score, reasons):
    return build_signal(symbol, df1, df5, ob, score, reasons)

# -----------------------------
# Telegram formatter + sender
# -----------------------------
def format_signal_message(sig: Dict[str, Any]) -> str:
    mode_emoji = "⚡" if sig["mode"] == "QUICK" else ("🔥" if sig["mode"] == "MID" else "🚀")
    dir_emoji = "⬆️ BUY" if sig["direction"] == "BUY" else "⬇️ SELL"
    dz_low, dz_high = sig["danger"]
    reason_txt = ", ".join(sig.get("reasons", []))
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

def send_telegram_message(msg: str) -> bool:
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram preview (NOT SENT):\n%s", msg)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        LOG.warning("Telegram send failed: %s %s", r.status_code, r.text)
        return False
    except Exception as e:
        LOG.warning("Telegram send exception: %s", e)
        return False

# -----------------------------
# S/R & GAINERS filters used by analyzer
# -----------------------------
def passes_sr_filter(df1: pd.DataFrame) -> bool:
    try:
        pivots = support_resistance_levels(df1)
        entry = float(df1["close"].iloc[-1])
        return not s_r_conflict(entry, pivots)
    except Exception:
        return True

def passes_gainers_filter(ex: "ccxt.Exchange", symbol: str) -> bool:
    try:
        top_vol = get_top_volume_symbols(ex)
        top_gain = get_top_gainers(ex)
        s = normalize_symbol(symbol)
        if not top_vol and not top_gain:
            return True
        if s in top_vol or s in top_gain:
            return True
        return False
    except Exception:
        return True

# -----------------------------
# FINAL ANALYZER (ALL FILTERS)
# -----------------------------
def analyze_coin(ex: "ccxt.Exchange", symbol: str) -> Optional[Dict[str, Any]]:
    try:
        symbol_n = normalize_symbol(symbol)
        try:
            ex.load_markets()
        except Exception:
            pass

        if symbol_n not in ex.markets:
            LOG.debug("analyze_coin skip: not in markets %s", symbol_n)
            return None

        # BTC calm pre-check
        if not btc_is_calm(ex):
            LOG.debug("BTC not calm — skipping signals")
            return None

        # fetch ohlcv
        df1 = fetch_ohlcv_sync(ex, symbol_n, timeframe="1m", limit=200)
        df5 = fetch_ohlcv_sync(ex, symbol_n, timeframe="5m", limit=200)

        if df1 is None or df1.empty:
            LOG.debug("analyze_coin skip: no df1 %s", symbol_n)
            return None

        # volume filter (last candle)
        last_vol = float(df1["volume"].iloc[-1]) if "volume" in df1.columns and not df1["volume"].empty else 0.0
        if last_vol < MIN_24H_VOLUME:
            LOG.debug("analyze_coin skip %s: low vol %.1f", symbol_n, last_vol)
            return None

        # orderbook + spread
        ob = fetch_orderbook_safe(ex, symbol_n, limit=50)
        spread = calc_spread_from_ob(ob)
        if spread > MAX_SPREAD_PCT:
            LOG.debug("analyze_coin skip %s: spread %.6f", symbol_n, spread)
            return None

        # s/r filter
        if not passes_sr_filter(df1):
            LOG.debug("analyze_coin skip %s: SR conflict", symbol_n)
            return None

        # gainers / top-volume filter
        if not passes_gainers_filter(ex, symbol_n):
            LOG.debug("analyze_coin skip %s: not in top lists", symbol_n)
            return None

        # scoring
        score, reasons = compute_score_and_reasons(df1, df5, ob, spread)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("analyze_coin skip %s: score %.1f < min %.1f", symbol_n, score, MIN_SIGNAL_SCORE)
            return None

        # build signal candidate
        sig_candidate = build_signal_from_df(symbol_n, df1, df5, ob, score, reasons)
        if not sig_candidate:
            return None

        # cooldown check
        key = cd_key(sig_candidate["pair"], sig_candidate["mode"])
        if _cd.is_active(key):
            LOG.debug("analyze_coin skip %s: cooldown active %s", symbol_n, key)
            return None

        # set cooldown and return
        _cd.set(key, sig_candidate["mode"])
        LOG.info("→ SIGNAL READY %s | mode=%s score=%.1f", sig_candidate["pair"], sig_candidate["mode"], sig_candidate["score"])
        return sig_candidate

    except Exception as e:
        LOG.exception("analyze_coin error %s: %s", symbol, e)
        return None