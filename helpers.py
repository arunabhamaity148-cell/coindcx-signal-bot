# helpers.py — SAFE PRO SETUP (Option A)
# - Redis cooldown (async) with local JSON fallback
# - OHLCV fetch (ccxt sync), indicators (EMA/MACD/RSI/ATR), score engine
# - Filters: BTC stability, spread, orderbook imbalance, volume, danger zone
# - Signal builder (QUICK / MID / TREND) with ATR TP/SL
# - Telegram sender (HTML) preview via NOTIFY_ONLY by default
# COPY → save as helpers.py

from __future__ import annotations
import os, time, json, math, logging, asyncio
from typing import Dict, Any, List, Optional, Tuple
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

# -----------------------------
# Logging
# -----------------------------
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

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85"))
THRESH_QUICK = float(os.getenv("THRESH_QUICK", "92"))
THRESH_MID = float(os.getenv("THRESH_MID", "85"))
THRESH_TREND = float(os.getenv("THRESH_TREND", "72"))

MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.004"))  # 0.4%

COOLDOWN_JSON = os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json")
REDIS_URL = os.getenv("REDIS_URL", "")

# TTL defaults (seconds)
TTL_QUICK = int(os.getenv("COOLDOWN_QUICK_S", "1800"))   # 30m
TTL_MID   = int(os.getenv("COOLDOWN_MID_S", "900"))      # 15m
TTL_TREND = int(os.getenv("COOLDOWN_TREND_S", "3600"))   # 60m

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

OHLCV_MAX_RETRIES = int(os.getenv("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "0.8"))
RATE_LIMIT_PAUSE = float(os.getenv("RATE_LIMIT_PAUSE", "0.6"))

# Indicator params
EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

# -----------------------------
# Cooldown manager (Redis async + JSON fallback)
# -----------------------------
class CooldownManager:
    def __init__(self):
        self.use_redis = aioredis is not None and bool(REDIS_URL)
        self.redis = None
        self._local_map: Dict[str,int] = {}
        if not self.use_redis:
            self._load_local()
            LOG.info("CooldownManager: using local JSON fallback (%s)", COOLDOWN_JSON)
        else:
            LOG.info("CooldownManager: Redis configured")

    async def init(self):
        if self.use_redis:
            try:
                self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                await self.redis.ping()
                LOG.info("CooldownManager: connected to Redis")
            except Exception as e:
                LOG.warning("CooldownManager: Redis init failed, fallback to local JSON: %s", e)
                self.use_redis = False
                self._load_local()

    def _load_local(self):
        try:
            with open(COOLDOWN_JSON, "r") as f:
                data = json.load(f)
            self._local_map = {k:int(v) for k,v in data.items()}
        except Exception:
            self._local_map = {}

    def _save_local(self):
        try:
            with open(COOLDOWN_JSON + ".tmp","w") as f:
                json.dump(self._local_map, f)
            os.replace(COOLDOWN_JSON + ".tmp", COOLDOWN_JSON)
        except Exception as e:
            LOG.warning("CooldownManager: failed to save local json: %s", e)

    def _ttl_for_mode(self, mode: str) -> int:
        m = mode.lower()
        if m == "quick": return TTL_QUICK
        if m == "mid": return TTL_MID
        return TTL_TREND

    async def set_cooldown(self, key: str, mode: str) -> bool:
        ttl = self._ttl_for_mode(mode)
        if self.use_redis and self.redis:
            try:
                ok = await self.redis.set(name=key, value="1", ex=ttl, nx=True)
                return bool(ok)
            except Exception as e:
                LOG.warning("CooldownManager: redis set failed, fallback local: %s", e)
                self.use_redis = False
        now = int(time.time())
        exp = self._local_map.get(key, 0)
        if exp <= now:
            self._local_map[key] = now + ttl
            self._save_local()
            return True
        return False

    async def is_cooled(self, key: str) -> bool:
        if self.use_redis and self.redis:
            try:
                ttl = await self.redis.ttl(key)
                return bool(ttl and int(ttl) > 0)
            except Exception as e:
                LOG.warning("CooldownManager: redis ttl failed, fallback local: %s", e)
                self.use_redis = False
        now = int(time.time())
        exp = self._local_map.get(key)
        if not exp: return False
        if exp <= now:
            self._local_map.pop(key, None)
            self._save_local()
            return False
        return True

    async def clear(self, key: str):
        if self.use_redis and self.redis:
            try:
                await self.redis.delete(key); return
            except Exception as e:
                LOG.warning("CooldownManager: redis delete failed: %s", e)
                self.use_redis = False
        self._local_map.pop(key, None); self._save_local()

_cd_manager: Optional[CooldownManager] = None
async def init_cooldown_manager() -> CooldownManager:
    global _cd_manager
    if _cd_manager is None:
        _cd_manager = CooldownManager()
        await _cd_manager.init()
    return _cd_manager

# -----------------------------
# Exchange helpers
# -----------------------------
def get_exchange(api_keys: bool = False) -> ccxt.Exchange:
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
    # optional keys
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
# OHLCV fetch (sync via ccxt) with retries
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

def fetch_ohlcv_with_retry_sync(ex: ccxt.Exchange, symbol: str, timeframe: str = "1m", limit: int = 200) -> Optional[pd.DataFrame]:
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
# Indicators (pandas-safe)
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()

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
# Orderbook Imbalance (sync fetch)
# -----------------------------
def fetch_orderbook_safe_sync(ex: ccxt.Exchange, symbol: str, limit: int = 50) -> dict:
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

# -----------------------------
# Filters & helpers
# -----------------------------
def calc_spread_from_orderbook(ob: dict) -> float:
    try:
        bid = float(ob['bids'][0][0])
        ask = float(ob['asks'][0][0])
        return abs(ask - bid)/ask
    except Exception:
        return 999.0

def btc_stable_from_df(df: pd.DataFrame) -> bool:
    # check short-term percentage change volatility
    p = df['close'].pct_change().abs().tail(6).mean()
    return p < 0.0018  # tuned threshold: small movement -> stable

def danger_zone_check(entry: float, sl: float, atr_val: float) -> bool:
    # return True if entry is dangerously close to SL
    # Danger if distance < 0.6 * atr (tunable)
    return abs(entry - sl) < (0.6 * atr_val)

# -----------------------------
# Scoring engine
# -----------------------------
def compute_score_and_reasons(df: pd.DataFrame, ob_imb: float, spread: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if df is None or len(df) < 30:
        return 0.0, ["insufficient_data"]
    close = df['close']
    score = 40.0

    # EMA alignment
    e20 = ema(close, EMA_FAST).iloc[-1]
    e50 = ema(close, EMA_SLOW).iloc[-1]
    e200 = ema(close, EMA_LONG).iloc[-1]
    if e20 > e50:
        score += 12; reasons.append("EMA20>50")
    else:
        reasons.append("EMA20<=50")
    if e20 > e200:
        score += 8; reasons.append("EMA20>200")

    # MACD-like (ema diff)
    macd_val = (ema(close, 12) - ema(close, 26)).iloc[-1]
    if macd_val > 0:
        score += 12; reasons.append("MACD_pos")

    # RSI
    r = float(rsi(close).iloc[-1])
    if 40 < r < 70:
        score += 8; reasons.append("RSI_ok")
    elif r <= 40:
        score += 4; reasons.append("RSI_low")

    # volume spike
    vol = df['volume']; vol_avg = vol.rolling(20, min_periods=1).mean().iloc[-1] or 1.0
    if vol.iloc[-1] > vol_avg*1.6:
        score += 10; reasons.append("Vol_spike")
    else:
        reasons.append("Vol_ok")

    # orderbook imbalance and spread penalties/bonuses
    if ob_imb > 0.55:
        score += 8; reasons.append("OB_buy_pressure")
    if spread <= MAX_SPREAD_PCT:
        score += 6; reasons.append("Spread_ok")
    else:
        reasons.append("Spread_high")

    # clamp
    score = max(0.0, min(100.0, score))
    return round(score,1), reasons

# -----------------------------
# Signal builder
# -----------------------------
def build_signal_from_df(symbol: str, df: pd.DataFrame, ob: dict, score: float, reasons: List[str]) -> Dict[str,Any]:
    last = df.iloc[-1]
    entry = float(last['close'])
    atr_val = float(atr(df).iloc[-1]) if not atr(df).empty else 0.0

    # default mid mode rr
    mode = "MID"
    if score >= THRESH_QUICK: mode = "QUICK"
    elif score >= THRESH_MID: mode = "MID"
    else: mode = "TREND"

    if mode == "QUICK":
        tp = entry + atr_val*1.5
        sl = entry - atr_val*1.0
    elif mode == "TREND":
        tp = entry + atr_val*3.0
        sl = entry - atr_val*2.0
    else:
        tp = entry + atr_val*2.0
        sl = entry - atr_val*1.4

    # detect possible sell if ema20 < ema50 (trend down)
    direction = "BUY"
    if ema(df['close'], EMA_FAST).iloc[-1] < ema(df['close'], EMA_SLOW).iloc[-1]:
        direction = "SELL"
        # swap TP/SL for sell
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
# Telegram formatter & sender
# -----------------------------
def format_signal_message(sig: Dict[str,Any]) -> str:
    em = "🔥" if sig['mode']=="MID" else ("⚡" if sig['mode']=="QUICK" else "🚀")
    dz_low, dz_high = sig.get("danger", (0,0))
    reason = ", ".join(sig.get("reasons", []))
    return (
        f"{em} <b>{sig['direction']} SIGNAL — {sig['mode']}</b>\n"
        f"Pair: <b>{sig['pair']}</b>  Score: <b>{sig['score']}</b>\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"TP: <code>{sig['tp']}</code>   SL: <code>{sig['sl']}</code>\n"
        f"Qty: (manual)   Size: (manual)   Lev: (manual)\n"
        f"⚠️ Danger Zone: <code>{dz_low}</code> - <code>{dz_high}</code> (ATR={sig['atr']})\n"
        f"Reason: {reason}\n"
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
# Top-level analyze (sync)
# -----------------------------
def analyze_symbol_sync(ex: ccxt.Exchange, symbol: str) -> Optional[Dict[str,Any]]:
    try:
        norm = normalize_symbol(symbol)
        # quick market check
        try:
            ex.load_markets()
            if norm not in ex.markets:
                LOG.debug("Symbol not in markets: %s", norm)
                return None
        except Exception:
            pass

        df = fetch_ohlcv_with_retry_sync(ex, norm, timeframe="1m", limit=200)
        if df is None or df.empty:
            return None

        # volume filter (last candle)
        last_vol = float(df['volume'].iloc[-1]) if 'volume' in df.columns and not df['volume'].empty else 0.0
        if last_vol < MIN_24H_VOLUME:
            LOG.debug("%s skip: low vol %.1f", norm, last_vol); return None

        # BTC stability check (fetch BTC 1m quick inside caller ideally)
        # we'll assume caller checked BTC separately; skip here

        # orderbook & spread
        ob = fetch_orderbook_safe_sync(ex, norm, limit=20)
        ob_imb = orderbook_imbalance_from_ob(ob)
        spread = calc_spread_from_orderbook(ob)
        if spread > MAX_SPREAD_PCT:
            LOG.debug("%s skip: spread %.6f > max %.6f", norm, spread, MAX_SPREAD_PCT); return None

        # compute score
        score, reasons = compute_score_and_reasons(df, ob_imb, spread)
        if score < MIN_SIGNAL_SCORE:
            LOG.debug("%s skip: score %.1f < min %.1f", norm, score, MIN_SIGNAL_SCORE); return None

        # build signal
        sig = build_signal_from_df(norm, df, ob, score, reasons)
        return sig
    except Exception as e:
        LOG.exception("analyze_symbol_sync error %s: %s", symbol, e)
        return None

# -----------------------------
# Utility: cooldown key
# -----------------------------
def cooldown_key_for(sig_or_pair: Any, mode: Optional[str]=None) -> str:
    if isinstance(sig_or_pair, dict):
        pair = sig_or_pair.get("pair") or sig_or_pair.get("symbol")
    else:
        pair = sig_or_pair
    m = mode or (sig_or_pair.get("mode") if isinstance(sig_or_pair, dict) else "MID")
    return f"{normalize_symbol(pair)}::{m.upper()}"

# -----------------------------
# Small preview log helper
# -----------------------------
def preview_signal_log(sig: Dict[str,Any]):
    try:
        LOG.info("→ SIGNAL %s | mode=%s score=%.1f entry=%s tp=%s sl=%s",
                 sig.get("pair"), sig.get("mode"), sig.get("score"),
                 sig.get("entry"), sig.get("tp"), sig.get("sl"))
    except Exception:
        LOG.info("→ SIGNAL (preview failed)")

# End of helpers.py
