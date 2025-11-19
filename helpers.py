# helpers.py
# ArunBot - helpers (Redis cooldown, indicators, scoring, signal builder, telegram)
# Copy-paste করে helpers.py নামে save করো।
# Author: ChatGPT for Arun
# Note: Requires ccxt, pandas, numpy, redis (asyncio), requests, python-dotenv

from __future__ import annotations
import os
import time
import json
import math
import logging
import asyncio
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
import numpy as np

# external libs
try:
    import ccxt
except Exception:
    ccxt = None

try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

import requests
from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# Logging
# ------------------------------
LOG = logging.getLogger("helpers")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(os.getenv("HELPERS_LOG_LEVEL", "INFO"))

# ------------------------------
# Config (env defaults)
# ------------------------------
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1", "true", "yes")
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "False").lower() in ("1", "true", "yes")
SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
OHLCV_MAX_RETRIES = int(os.getenv("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "0.8"))
RATE_LIMIT_PAUSE = float(os.getenv("RATE_LIMIT_PAUSE", "0.6"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "92"))
THRESH_MID = float(os.getenv("THRESH_MID", "85"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "72"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.4"))

EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")
REDIS_URL = os.getenv("REDIS_URL", "")

# Cooldown TTLs (seconds)
TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "1800"))   # 30 min
TTL_MID   = int(os.getenv("COOLDOWN_MID_S", "900"))      # 15 min
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))   # 60 min

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "") or os.getenv("CHAT_ID", "")

# ------------------------------
# Cooldown Manager (Redis async + JSON fallback)
# ------------------------------
class CooldownManager:
    """
    Async cooldown manager. Use Redis if REDIS_URL provided and aioredis available,
    otherwise use local JSON file (COOLDOWN_JSON).
    Methods are async to allow Redis usage.
    Key convention: f"{symbol}::{mode}" e.g. "BTC/USDT::QUICK"
    """

    def __init__(self):
        self.use_redis = aioredis is not None and bool(REDIS_URL)
        self.redis = None
        self._local_map: Dict[str, int] = {}
        if not self.use_redis:
            self._load_local()
            LOG.info("CooldownManager: using local JSON fallback (%s)", COOLDOWN_JSON)
        else:
            LOG.info("CooldownManager: configured to use Redis (REDIS_URL present)")

    async def init(self):
        if self.use_redis:
            try:
                self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                await self.redis.ping()
                LOG.info("CooldownManager: connected to Redis")
            except Exception as e:
                LOG.warning("CooldownManager: Redis init failed, falling back to local JSON: %s", e)
                self.use_redis = False
                self._load_local()

    def _load_local(self):
        try:
            with open(COOLDOWN_JSON, "r") as f:
                data = json.load(f)
                self._local_map = {k: int(v) for k, v in data.items()}
        except FileNotFoundError:
            self._local_map = {}
        except Exception as e:
            LOG.warning("CooldownManager: failed to load local json: %s", e)
            self._local_map = {}

    def _save_local(self):
        try:
            with open(COOLDOWN_JSON + ".tmp", "w") as f:
                json.dump(self._local_map, f)
            os.replace(COOLDOWN_JSON + ".tmp", COOLDOWN_JSON)
        except Exception as e:
            LOG.warning("CooldownManager: failed to save local json: %s", e)

    def _ttl_for_mode(self, mode: str) -> int:
        m = mode.lower()
        if m == "quick":
            return TTL_QUICK
        if m == "mid":
            return TTL_MID
        return TTL_TREND

    async def set_cooldown(self, key: str, mode: str) -> bool:
        """
        Try to set cooldown. Returns True if set (i.e. no active cooldown existed).
        """
        ttl = self._ttl_for_mode(mode)
        if self.use_redis and self.redis:
            try:
                # NX = set only if not exists
                ok = await self.redis.set(name=key, value="1", ex=ttl, nx=True)
                return bool(ok)
            except Exception as e:
                LOG.warning("CooldownManager: Redis set failed, fallback local: %s", e)
                self.use_redis = False

        # local fallback
        now = int(time.time())
        exp = self._local_map.get(key, 0)
        if exp <= now:
            self._local_map[key] = now + ttl
            self._save_local()
            return True
        return False

    async def is_cooled(self, key: str) -> bool:
        """
        Return True if key is currently in cooldown (blocked).
        """
        if self.use_redis and self.redis:
            try:
                ttl = await self.redis.ttl(key)
                return bool(ttl and int(ttl) > 0)
            except Exception as e:
                LOG.warning("CooldownManager: Redis ttl failed, fallback local: %s", e)
                self.use_redis = False

        now = int(time.time())
        exp = self._local_map.get(key)
        if not exp:
            return False
        if exp <= now:
            # expired -> remove
            self._local_map.pop(key, None)
            self._save_local()
            return False
        return True

    async def clear(self, key: str):
        if self.use_redis and self.redis:
            try:
                await self.redis.delete(key)
                return
            except Exception as e:
                LOG.warning("CooldownManager: Redis delete failed: %s", e)
                self.use_redis = False
        self._local_map.pop(key, None)
        self._save_local()

# singleton cooldown manager (to import from main)
_cd_manager: Optional[CooldownManager] = None

async def init_cooldown_manager() -> CooldownManager:
    global _cd_manager
    if _cd_manager is None:
        _cd_manager = CooldownManager()
        await _cd_manager.init()
    return _cd_manager

# ------------------------------
# Exchange factory & helpers
# ------------------------------
def get_exchange(api_keys: bool = False) -> ccxt.Exchange:
    """
    Returns ccxt exchange instance.
    If api_keys True or env has keys, sets apiKey/secret.
    """
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")

    ex_name = EXCHANGE_NAME.lower()
    if ex_name == "binance":
        ex = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    else:
        ex_cls = getattr(ccxt, ex_name, None)
        if ex_cls is None:
            ex = ccxt.binance({'enableRateLimit': True})
        else:
            ex = ex_cls({'enableRateLimit': True})
    if api_keys or (os.getenv("EXCHANGE_API_KEY") and os.getenv("EXCHANGE_API_SECRET")):
        try:
            ex.apiKey = os.getenv("EXCHANGE_API_KEY")
            ex.secret = os.getenv("EXCHANGE_API_SECRET")
        except Exception:
            LOG.debug("Exchange API keys not set or failed to assign")
    try:
        ex.load_markets()
    except Exception:
        LOG.debug("Exchange.load_markets() failed/skipped")
    return ex

def normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" not in s:
        s = f"{s}/{QUOTE_ASSET}"
    return s

# ------------------------------
# OHLCV fetch with retry
# ------------------------------
def df_from_ohlcv(ohlcv) -> Optional[pd.DataFrame]:
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    return df

def fetch_ohlcv_with_retry_sync(ex: ccxt.Exchange, symbol: str, timeframe: str = "1m", limit: int = 200) -> Optional[pd.DataFrame]:
    symbol = normalize_symbol(symbol)
    last_exc = None
    for attempt in range(OHLCV_MAX_RETRIES):
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            time.sleep(RATE_LIMIT_PAUSE)
            return df_from_ohlcv(data)
        except ccxt.BaseError as e:
            last_exc = e
            LOG.debug("fetch_ohlcv attempt %d failed for %s: %s", attempt+1, symbol, e)
            time.sleep(OHLCV_RETRY_DELAY * (attempt+1))
        except Exception as e:
            last_exc = e
            LOG.debug("fetch_ohlcv attempt %d failed for %s: %s", attempt+1, symbol, e)
            time.sleep(OHLCV_RETRY_DELAY * (attempt+1))
    LOG.warning("fetch_ohlcv failed for %s after %d attempts: %s", symbol, OHLCV_MAX_RETRIES, last_exc)
    return None

# ------------------------------
# Indicators
# ------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, sig=9):
    efast = ema(series, fast)
    eslow = ema(series, slow)
    macd_line = efast - eslow
    sig_line = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean().fillna(method="backfill")

# ------------------------------
# Score engine
# ------------------------------
def compute_score(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    return (score 0-100, reasons[])
    """
    reasons: List[str] = []
    if df is None or len(df) < 30:
        return 0.0, ["insufficient_data"]

    close = df['close']
    score = 40.0  # base

    # EMA trend
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    if ema21 > ema50:
        score += 12; reasons.append("EMA_up")
    else:
        score -= 8; reasons.append("EMA_down")

    # MACD hist
    _, _, hist = macd(close)
    if hist.iloc[-1] > 0:
        score += 10; reasons.append("MACD_pos")
    else:
        reasons.append("MACD_neg")

    # RSI
    r = float(rsi(close).iloc[-1])
    if 30 < r < 70:
        score += 6; reasons.append("RSI_ok")
    elif r <= 30:
        score += 3; reasons.append("RSI_oversold")
    else:
        score -= 6; reasons.append("RSI_overbought")

    # volume
    vol = df['vol']
    vol_avg = vol.rolling(20, min_periods=1).mean().iloc[-1] or 1.0
    vol_ratio = float(vol.iloc[-1]) / (vol_avg + 1e-12)
    if vol_ratio > 1.8:
        score += 10; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # ATR normalized penalty
    last_atr = float(atr(df).iloc[-1] if not atr(df).empty else 0.0)
    close_now = float(close.iloc[-1])
    if last_atr > 0:
        rel = last_atr / (close_now + 1e-9)
        if rel > 0.01:
            score -= 4; reasons.append("High_ATR")
        else:
            reasons.append("ATR_ok")

    # price vs sma50
    sma50 = sma(close, 50).iloc[-1]
    if close_now > sma50:
        score += 6; reasons.append("Price_above_SMA50")

    # clamp
    score = max(0.0, min(100.0, score))
    return round(score, 1), reasons

# ------------------------------
# Signal builder
# ------------------------------
def build_signal(df: pd.DataFrame, symbol: str, score: float, reasons: List[str], mode: str = "MID") -> Dict[str, Any]:
    close_now = float(df['close'].iloc[-1])
    last_atr = float(atr(df).iloc[-1] if not atr(df).empty else 0.0)

    entry = close_now
    if mode.upper() == "QUICK":
        tp = entry + last_atr * 1.5
        sl = entry - last_atr * 1.2
    elif mode.upper() == "TREND":
        tp = entry + last_atr * 3.0
        sl = entry - last_atr * 2.0
    else:  # MID
        tp = entry + last_atr * 2.0
        sl = entry - last_atr * 1.5

    danger_low = entry - last_atr * 1.1
    danger_high = entry + last_atr * 1.1

    return {
        "pair": normalize_symbol(symbol),
        "mode": mode.upper(),
        "score": score,
        "entry": round(entry, 8),
        "tp": round(tp, 8),
        "sl": round(sl, 8),
        "danger": (round(danger_low, 8), round(danger_high, 8)),
        "atr": round(last_atr, 8),
        "reasons": reasons,
        "ts": int(time.time())
    }

# ------------------------------
# Telegram message
# ------------------------------
def format_signal_message(sig: Dict[str, Any]) -> str:
    mode = sig.get("mode", "MID")
    emoji = "⚡" if mode == "QUICK" else ("🔥" if mode == "MID" else "🚀")
    danger_low, danger_high = sig.get("danger", (0,0))
    reason_text = ", ".join(sig.get("reasons", []))
    msg = (
        f"{emoji} <b>BUY SIGNAL — {mode}</b>\n"
        f"Pair: <b>{sig['pair']}</b>  Score: <b>{sig['score']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Qty: (manual)   Size: (manual)   Lev: (manual)\n"
        f"⚠️ Danger Zone: <code>{danger_low}</code> - <code>{danger_high}</code> (ATR={sig['atr']})\n"
        f"Reason: {reason_text}\n"
        f"Mode: {sig['mode']} | Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(sig['ts']))} UTC"
    )
    return msg

def send_telegram_message(msg: str, preview: bool = False) -> bool:
    """
    Send Telegram message. If NOTIFY_ONLY True, skip actual send (preview logged).
    """
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        LOG.info("Telegram preview (NOT SENT):\n%s", msg)
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
        LOG.warning("Telegram send exception: %s", e)
        return False

# ------------------------------
# Top-level analyze_symbol (sync)
# ------------------------------
def analyze_symbol_sync(ex: ccxt.Exchange, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Analyze one symbol and return signal dict or None.
    This is synchronous to be used in main.py (sync loop).
    """
    try:
        s = normalize_symbol(symbol)
        # quick existence check
        try:
            ex.load_markets()
            if s not in ex.markets:
                LOG.debug("Symbol not in markets: %s", s)
                return None
        except Exception:
            # ignore load_markets issues
            pass

        df = fetch_ohlcv_with_retry_sync(ex, s, timeframe="1m", limit=200)
        if df is None or df.empty:
            return None

        # quick filters
        # volume check (use last candle volume as proxy)
        last_vol = float(df['vol'].iloc[-1]) if 'vol' in df.columns and not df['vol'].empty else 0.0
        if last_vol < MIN_24H_VOLUME:
            LOG.debug("%s skipped: vol %.1f < MIN_24H_VOLUME %.1f", s, last_vol, MIN_24H_VOLUME)
            return None

        # compute score
        score, reasons = compute_score(df)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("%s skipped: score %.1f < MIN_SIGNAL_SCORE %.1f", s, score, MIN_SIGNAL_SCORE)
            return None

        # decide mode
        mode = "MID"
        if score >= THRESH_QUICK:
            mode = "QUICK"
        elif score >= THRESH_MID:
            mode = "MID"
        elif score >= THRESH_TREND:
            mode = "TREND"
        else:
            return None

        sig = build_signal(df, s, score, reasons, mode=mode)
        return sig
    except Exception as e:
        LOG.exception("analyze_symbol_sync error for %s: %s", symbol, e)
        return None

# ------------------------------
# Simple helper: key name builder
# ------------------------------
def cooldown_key_for(sig_or_pair: Any, mode: Optional[str] = None) -> str:
    if isinstance(sig_or_pair, dict):
        pair = sig_or_pair.get("pair") or sig_or_pair.get("symbol")
    else:
        pair = sig_or_pair
    pair = normalize_symbol(pair)
    m = mode or (sig_or_pair.get("mode") if isinstance(sig_or_pair, dict) else "MID")
    return f"{pair}::{m.upper()}"

# ------------------------------
# Backward-compatible local cooldown helpers (sync)
# ------------------------------
def load_local_cooldown_map(path: str = COOLDOWN_JSON) -> Dict[str, int]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items()}
    except Exception:
        return {}

def save_local_cooldown_map(m: Dict[str, int], path: str = COOLDOWN_JSON):
    try:
        with open(path + ".tmp", "w") as f:
            json.dump(m, f)
        os.replace(path + ".tmp", path)
    except Exception as e:
        LOG.warning("save_local_cooldown_map failed: %s", e)

# ------------------------------
# Small util: pretty print signal (for logs)
# ------------------------------
def preview_signal_log(sig: Dict[str, Any]):
    try:
        LOG.info("→ SIGNAL %s | mode=%s score=%.1f entry=%s tp=%s sl=%s",
                 sig.get("pair"), sig.get("mode"), sig.get("score"),
                 sig.get("entry"), sig.get("tp"), sig.get("sl"))
    except Exception:
        LOG.info("→ SIGNAL (unable to pretty print)")

# ------------------------------
# End of helpers.py
# ------------------------------
