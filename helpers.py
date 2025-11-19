# helpers.py — FINAL CLEAN (redis-free) for ArunBot
from __future__ import annotations
import os, time, json, math, logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import ccxt
except Exception:
    ccxt = None

import requests
from dotenv import load_dotenv
load_dotenv()

LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))

# -----------------------------
# Config (env)
# -----------------------------
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1","true","yes")
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "False").lower() in ("1","true","yes")
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "90"))   # stricter default
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "95"))
THRESH_MID = float(os.getenv("THRESH_MID", "90"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "82"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))   # base filter
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "2.2"))  # last candle vs avg*mult

MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.002"))  # 0.2% default stricter

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

OHLCV_MAX_RETRIES = int(os.getenv("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "0.8"))
RATE_LIMIT_PAUSE = float(os.getenv("RATE_LIMIT_PAUSE", "0.6"))

# Indicator params
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
EMA_LONG = int(os.getenv("EMA_LONG", "200"))

# TTL defaults (seconds) — same-coin cooldown per mode
TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "900"))   # 15m
TTL_MID   = int(os.getenv("COOLDOWN_MID_S", "1800"))    # 30m
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))  # 60m

# -----------------------------
# Local cooldown manager (JSON only)
# -----------------------------
class CooldownManager:
    def __init__(self, path: str = COOLDOWN_JSON):
        self.path = path
        self.map: Dict[str,int] = {}
        self._load()
        LOG.info("CooldownManager: using local JSON fallback (%s)", self.path)

    def _load(self):
        try:
            with open(self.path,"r") as f:
                self.map = {k:int(v) for k,v in json.load(f).items()}
        except Exception:
            self.map = {}

    def _save(self):
        try:
            with open(self.path + ".tmp","w") as f:
                json.dump(self.map, f)
            os.replace(self.path + ".tmp", self.path)
        except Exception as e:
            LOG.warning("CooldownManager save failed: %s", e)

    def _ttl_for_mode(self, mode: str) -> int:
        m = (mode or "").lower()
        if m == "quick": return TTL_QUICK
        if m == "mid": return TTL_MID
        return TTL_TREND

    def set_cooldown(self, key: str, mode: str) -> bool:
        ttl = self._ttl_for_mode(mode)
        now = int(time.time())
        exp = self.map.get(key, 0)
        if exp <= now:
            self.map[key] = now + ttl
            self._save()
            return True
        return False

    def is_cooled(self, key: str) -> bool:
        now = int(time.time())
        exp = self.map.get(key)
        if not exp: return False
        if exp <= now:
            self.map.pop(key, None)
            self._save()
            return False
        return True

    def clear(self, key: str):
        self.map.pop(key, None)
        self._save()

_cd = CooldownManager()

# -----------------------------
# Exchange helpers
# -----------------------------
def get_exchange(api_keys: bool = False) -> "ccxt.Exchange":
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex_name = EXCHANGE_NAME.lower()
    try:
        if ex_name == "binance":
            ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        else:
            ex_cls = getattr(ccxt, ex_name, None)
            ex = ex_cls({'enableRateLimit': True}) if ex_cls else ccxt.binance({'enableRateLimit': True})
    except Exception:
        ex = ccxt.binance({'enableRateLimit': True})
    if api_keys and os.getenv("EXCHANGE_API_KEY") and os.getenv("EXCHANGE_API_SECRET"):
        ex.apiKey = os.getenv("EXCHANGE_API_KEY"); ex.secret = os.getenv("EXCHANGE_API_SECRET")
    try:
        ex.load_markets()
    except Exception:
        LOG.debug("exchange.load_markets failed/ignored")
    return ex

def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# -----------------------------
# OHLCV fetch (sync) with retries
# -----------------------------
def df_from_ohlcv(ohlcv) -> Optional[pd.DataFrame]:
    if not ohlcv: return None
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    if df.empty: return None
    return df

def fetch_ohlcv_with_retry_sync(ex: "ccxt.Exchange", symbol: str, timeframe: str = "1m", limit: int = 200) -> Optional[pd.DataFrame]:
    s = normalize_symbol(symbol)
    last_exc = None
    for attempt in range(OHLCV_MAX_RETRIES):
        try:
            data = ex.fetch_ohlcv(s, timeframe=timeframe, limit=limit)
            time.sleep(RATE_LIMIT_PAUSE)
            return df_from_ohlcv(data)
        except Exception as e:
            last_exc = e
            LOG.debug("fetch_ohlcv attempt %d failed for %s: %s", attempt+1, s, e)
            time.sleep(OHLCV_RETRY_DELAY * (attempt+1))
    LOG.warning("fetch_ohlcv failed for %s after %d attempts: %s", s, OHLCV_MAX_RETRIES, last_exc)
    return None

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/length, adjust=False).mean()
    rs = up/(down+1e-12)
    return 100 - (100/(1+rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean().bfill()

# -----------------------------
# Orderbook & spread
# -----------------------------
def fetch_orderbook_safe_sync(ex: "ccxt.Exchange", symbol: str, limit: int = 50) -> dict:
    try:
        ob = ex.fetch_order_book(normalize_symbol(symbol), limit=limit)
        time.sleep(RATE_LIMIT_PAUSE)
        return ob or {}
    except Exception:
        LOG.debug("fetch_orderbook failed for %s", symbol)
        return {}

def orderbook_imbalance_from_ob(ob: dict, depth_levels: int = 12) -> float:
    bids = ob.get('bids', []); asks = ob.get('asks', [])
    bid_vol = sum([float(x[1]) for x in bids[:depth_levels]])
    ask_vol = sum([float(x[1]) for x in asks[:depth_levels]])
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

def calc_spread_from_orderbook(ob: dict) -> float:
    try:
        bid = float(ob['bids'][0][0])
        ask = float(ob['asks'][0][0])
        return abs(ask - bid)/ask
    except Exception:
        return 999.0

# -----------------------------
# Filters & helpers (MTF)
# -----------------------------
def btc_stable_from_df(df: pd.DataFrame) -> bool:
    p = df['close'].pct_change().abs().tail(6).mean()
    return p < 0.0018

def danger_zone_check(entry: float, sl: float, atr_val: float) -> bool:
    return abs(entry - sl) < (0.6 * atr_val)

# -----------------------------
# Scoring engine (MTF: uses 1m + 5m)
# -----------------------------
def compute_score_and_reasons(df1m: pd.DataFrame, df5m: pd.DataFrame, ob_imb: float, spread: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if df1m is None or df5m is None or len(df1m) < 30 or len(df5m) < 20:
        return 0.0, ["insufficient_data"]
    close = df1m['close']
    score = 40.0

    # EMA alignment (use 1m and 5m)
    e20_1 = ema(close, EMA_FAST).iloc[-1]
    e50_1 = ema(close, EMA_SLOW).iloc[-1]
    e200_1 = ema(close, EMA_LONG).iloc[-1]
    if e20_1 > e50_1:
        score += 10; reasons.append("EMA20>50_1m")
    if e20_1 > e200_1:
        score += 6; reasons.append("EMA20>200_1m")

    # 5m confirmation (trend higher weight)
    c5 = df5m['close']
    e20_5 = ema(c5, EMA_FAST).iloc[-1]
    e50_5 = ema(c5, EMA_SLOW).iloc[-1]
    if e20_5 > e50_5:
        score += 12; reasons.append("EMA20>50_5m")

    # MACD-like
    macd_val = (ema(close, 12) - ema(close, 26)).iloc[-1]
    if macd_val > 0:
        score += 10; reasons.append("MACD_pos")

    # RSI (1m)
    r = float(rsi(close).iloc[-1])
    if 40 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 40:
        score += 2; reasons.append("RSI_low")

    # volume spike (1m vs 20-avg)
    vol = df1m['volume']
    vol_avg = vol.rolling(20, min_periods=1).mean().iloc[-1] or 1.0
    if vol.iloc[-1] > vol_avg * VOLUME_MULTIPLIER:
        score += 12; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # orderbook & spread
    if ob_imb > 0.55:
        score += 8; reasons.append("OB_buy_pressure")
    if spread <= MAX_SPREAD_PCT:
        score += 8; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    score = max(0.0, min(100.0, score))
    return round(score,1), reasons

# -----------------------------
# Signal builder
# -----------------------------
def build_signal_from_df(symbol: str, df1m: pd.DataFrame, df5m: pd.DataFrame, ob: dict, score: float, reasons: List[str]) -> Dict[str,Any]:
    last = df1m.iloc[-1]
    entry = float(last['close'])
    atr_val = float(atr(df1m).iloc[-1]) if not atr(df1m).empty else 0.0

    mode = "MID"
    if score >= THRESH_QUICK: mode = "QUICK"
    elif score >= THRESH_MID: mode = "MID"
    else: mode = "TREND"

    if mode == "QUICK":
        tp = entry + atr_val*1.2
        sl = entry - atr_val*0.9
    elif mode == "TREND":
        tp = entry + atr_val*3.5
        sl = entry - atr_val*2.2
    else:
        tp = entry + atr_val*2.2
        sl = entry - atr_val*1.4

    direction = "BUY"
    if ema(df1m['close'], EMA_FAST).iloc[-1] < ema(df1m['close'], EMA_SLOW).iloc[-1]:
        direction = "SELL"
        tp = entry - (tp - entry)
        sl = entry + (entry - sl)

    danger = (round(entry - atr_val*1.1,8), round(entry + atr_val*1.1,8))
    return {
        "pair": normalize_symbol(symbol),
        "mode": mode,
        "score": score,
        "entry": round(entry,8),
        "tp": round(tp,8),
        "sl": round(sl,8),
        "atr": round(atr_val,8),
        "direction": direction,
        "reasons": reasons,
        "danger": danger,
        "ts": int(time.time())
    }

# -----------------------------
# Telegram formatting & sender
# -----------------------------
def format_signal_message(sig: Dict[str,Any]) -> str:
    em = "🔥" if sig['mode']=="MID" else ("⚡" if sig['mode']=="QUICK" else "🚀")
    dz_low, dz_high = sig.get("danger", (0,0))
    reason = ", ".join(sig.get("reasons", []))
    lev_sugg = os.getenv("AUTO_LEVERAGE_SUGGEST", "25x")
    return (
        f"{em} <b>{sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Leverage: <b>{lev_sugg}</b>\n"
        f"Score: <b>{sig['score']}</b>\n"
        f"ATR: {sig['atr']}\n"
        f"Reason: {reason}\n"
        f"⚠️ Danger Zone: <code>{dz_low}</code> - <code>{dz_high}</code>\n"
        f"Time: {pd.to_datetime(sig['ts'], unit='s').strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

def send_telegram_message(msg: str, preview: bool = False) -> bool:
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
# Top-level analyze (sync) - name analyze_coin (used by main)
# -----------------------------
def analyze_coin(ex: "ccxt.Exchange", symbol: str) -> Optional[Dict[str,Any]]:
    try:
        norm = normalize_symbol(symbol)
        try:
            ex.load_markets()
            if norm not in ex.markets:
                LOG.debug("Symbol not in markets: %s", norm)
                return None
        except Exception:
            pass

        # fetch 1m + 5m (MTF)
        df1 = fetch_ohlcv_with_retry_sync(ex, norm, timeframe="1m", limit=200)
        df5 = fetch_ohlcv_with_retry_sync(ex, norm, timeframe="5m", limit=200)
        if df1 is None or df5 is None:
            return None

        # quick volume filter (use last candle vs avg * multiplier)
        last_vol = float(df1['volume'].iloc[-1]) if 'volume' in df1.columns and not df1['volume'].empty else 0.0
        vol_avg_20 = float(df1['volume'].rolling(20, min_periods=1).mean().iloc[-1] or 1.0)
        if last_vol < MIN_24H_VOLUME and last_vol < (vol_avg_20 * VOLUME_MULTIPLIER):
            LOG.debug("%s skip: low vol last %.1f avg20*mult %.1f", norm, last_vol, vol_avg_20 * VOLUME_MULTIPLIER)
            return None

        # orderbook & spread
        ob = fetch_orderbook_safe_sync(ex, norm, limit=30)
        ob_imb = orderbook_imbalance_from_ob(ob)
        spread = calc_spread_from_orderbook(ob)
        if spread > MAX_SPREAD_PCT:
            LOG.debug("%s skip: spread %.6f > max %.6f", norm, spread, MAX_SPREAD_PCT)
            return None

        # score
        score, reasons = compute_score_and_reasons(df1, df5, ob_imb, spread)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("%s skip: score %.1f < min %.1f", norm, score, MIN_SIGNAL_SCORE)
            return None

        sig = build_signal_from_df(norm, df1, df5, ob, score, reasons)
        return sig
    except Exception as e:
        LOG.exception("analyze_coin error %s: %s", symbol, e)
        return None

def cooldown_key_for(sig_or_pair: Any, mode: Optional[str]=None) -> str:
    if isinstance(sig_or_pair, dict):
        pair = sig_or_pair.get("pair") or sig_or_pair.get("symbol")
    else:
        pair = sig_or_pair
    m = mode or (sig_or_pair.get("mode") if isinstance(sig_or_pair, dict) else "MID")
    return f"{normalize_symbol(pair)}::{m.upper()}"

def preview_signal_log(sig: Dict[str,Any]):
    try:
        LOG.info("→ SIGNAL %s | mode=%s score=%.1f entry=%s tp=%s sl=%s",
                 sig.get("pair"), sig.get("mode"), sig.get("score"),
                 sig.get("entry"), sig.get("tp"), sig.get("sl"))
    except Exception:
        LOG.info("→ SIGNAL (preview failed)")