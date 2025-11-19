# helpers.py — FINAL UPGRADED (replace your existing helpers.py with this file)
# Purpose:
# - Merge/upgrade your existing helpers into one production-ready helpers.py
# - Mode-aware TP/SL for QUICK/MID/TREND
# - Same-coin cooldown (30 minutes) with Redis optional + JSON fallback
# - BTC calm check (1m + 5m)
# - Telegram emoji-rich HTML message builder and sender (NOTIFY_ONLY safe mode)
# - Orderbook / spread / volume / OB imbalance filters
# - Score engine, ATR usage, entry/TP/SL calculations
# - Suggest leverage per mode/score (suggestion only)
# - Safe ccxt async fetchers (or sync fallback) with retries
# - Designed to be a drop-in helpers module for your main.py
#
# NOTE:
# - This file will NOT auto-execute orders (AUTO_EXECUTE must be explicitly enabled elsewhere).
# - Set ENV variables (.env or host) to control behavior (see defaults below).
# - Keep TELEGRAM_TOKEN/CHAT_ID out of Git — use secrets or platform env vars.
#
# Minimal ENV variables used (defaults included):
# NOTIFY_ONLY=True
# AUTO_EXECUTE=False
# EXCHANGE=binance
# QUOTE_ASSET=USDT
# COOLDOWN_MINUTES=30
# SCAN_BATCH_SIZE=20
# LOOP_SLEEP_SECONDS=5
# MAX_EMITS_PER_LOOP=1
# MIN_SIGNAL_SCORE=85
# MIN_24H_VOLUME=250000
# USE_REDIS=False
# REDIS_URL=
# TELEGRAM_BOT_TOKEN=
# TELEGRAM_CHAT_ID=
# OHLCV_MAX_RETRIES=3
# OHLCV_RETRY_DELAY=0.8
# OHLCV_CONCURRENCY=6

from __future__ import annotations
import os
import time
import json
import math
import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# pandas for indicators
import pandas as pd

# Attempt async ccxt, otherwise user should adapt to sync ccxt in main
try:
    import ccxt.async_support as ccxt_async
except Exception:
    ccxt_async = None

# optional aiohttp for telegram
try:
    import aiohttp
except Exception:
    aiohttp = None

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("helpers")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("HELPERS_LOGLEVEL", "INFO"))

# ---------------------------
# Config / ENV
# ---------------------------
NOTIFY_ONLY = os.getenv("NOTIFY_ONLY", "True").lower() in ("1", "true", "yes")
AUTO_EXECUTE = os.getenv("AUTO_EXECUTE", "False").lower() in ("1", "true", "yes")
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance")
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")

COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))
COOLDOWN_PERSIST_PATH = Path(os.getenv("COOLDOWN_PERSIST_PATH", "cooldown.json"))
USE_REDIS = os.getenv("USE_REDIS", "False").lower() in ("1", "true", "yes")
REDIS_URL = os.getenv("REDIS_URL", "")

SCAN_BATCH_SIZE = int(os.getenv("SCAN_BATCH_SIZE", "20"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "5"))
MAX_EMITS_PER_LOOP = int(os.getenv("MAX_EMITS_PER_LOOP", "1"))

MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "85.0"))
MIN_24H_VOLUME = float(os.getenv("MIN_24H_VOLUME", "250000"))

OHLCV_MAX_RETRIES = int(os.getenv("OHLCV_MAX_RETRIES", "3"))
OHLCV_RETRY_DELAY = float(os.getenv("OHLCV_RETRY_DELAY", "0.8"))
OHLCV_CONCURRENCY = int(os.getenv("OHLCV_CONCURRENCY", "6"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# BTC calm thresholds
BTC_CALM_STD_PCT = float(os.getenv("BTC_CALM_STD_PCT", "0.0018"))  # 0.18% stdev
BTC_CALM_VOL_PCT = float(os.getenv("BTC_CALM_VOL_PCT", "0.4"))

# TP/SL percent ranges per mode
TP_SL_SETTINGS = {
    "quick": {"tp_pct_min": 0.008, "tp_pct_max": 0.013, "sl_pct_min": 0.004, "sl_pct_max": 0.006},
    "mid":   {"tp_pct_min": 0.018, "tp_pct_max": 0.024, "sl_pct_min": 0.009, "sl_pct_max": 0.012},
    "trend": {"tp_pct_min": 0.02,  "tp_pct_max": 0.035, "sl_pct_min": 0.01,  "sl_pct_max": 0.016},
}

# leverage suggestion table (score thresholds -> leverage)
LEVERAGE_SUGGEST = {
    "quick": [(95, 50), (90, 45), (80, 40), (0, 30)],
    "mid":   [(95, 40), (90, 35), (80, 30), (0, 20)],
    "trend": [(95, 30), (90, 25), (80, 20), (0, 10)],
}

# concurrency semaphore for ohlcv fetches
_ohlcv_semaphore = asyncio.Semaphore(OHLCV_CONCURRENCY)

# ---------------------------
# Cooldown manager (Redis optional, JSON fallback)
# ---------------------------
class CooldownManager:
    def __init__(self, persist_path: Path = COOLDOWN_PERSIST_PATH, use_redis: bool = False, redis_url: str = ""):
        self.persist_path = persist_path
        self.lock = threading.Lock()
        self.map: Dict[str, int] = {}
        self.use_redis = use_redis and bool(redis_url)
        self.redis_client = None
        if self.use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("CooldownManager: connected to redis")
            except Exception as e:
                logger.warning("CooldownManager: redis init failed, fallback to JSON: %s", e)
                self.use_redis = False
        self._load()

    def _load(self):
        if not self.use_redis and self.persist_path.exists():
            try:
                with self.lock:
                    with open(self.persist_path, "r") as f:
                        data = json.load(f)
                        self.map = {k: int(v) for k, v in data.items()}
            except Exception:
                self.map = {}

    def _save(self):
        if not self.use_redis:
            try:
                tmp = self.persist_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(self.map, f)
                tmp.replace(self.persist_path)
            except Exception as e:
                logger.debug("CooldownManager: save failed: %s", e)

    def _key(self, coin: str) -> str:
        return coin.upper()

    def is_blocked(self, coin: str) -> bool:
        now = int(time.time())
        key = self._key(coin)
        if self.use_redis and self.redis_client:
            try:
                val = self.redis_client.get(key)
                return (val is not None) and (int(val) > now)
            except Exception:
                pass
        val = self.map.get(key)
        return (val is not None) and (int(val) > now)

    def set(self, coin: str, minutes: int = COOLDOWN_MINUTES) -> None:
        until = int(time.time() + minutes * 60)
        key = self._key(coin)
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.set(key, int(until), ex=minutes * 60)
                return
            except Exception:
                pass
        with self.lock:
            self.map[key] = int(until)
            self._save()

    def clear(self, coin: str) -> None:
        key = self._key(coin)
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
                return
            except Exception:
                pass
        with self.lock:
            if key in self.map:
                del self.map[key]
                self._save()

_cd_manager: Optional[CooldownManager] = None

def get_cooldown_manager() -> CooldownManager:
    global _cd_manager
    if _cd_manager is None:
        _cd_manager = CooldownManager(persist_path=COOLDOWN_PERSIST_PATH, use_redis=USE_REDIS, redis_url=REDIS_URL)
    return _cd_manager

# ---------------------------
# Exchange & OHLCV helpers (async)
# ---------------------------
async def load_client(exchange_name: str = EXCHANGE_NAME, **kwargs):
    if ccxt_async is None:
        raise RuntimeError("ccxt.async_support not installed. Please install ccxt for async support.")
    cls = getattr(ccxt_async, exchange_name, None)
    if cls is None:
        cls = getattr(ccxt_async, "binance")
    opts = {"enableRateLimit": True, "timeout": 30000}
    opts.update(kwargs or {})
    client = cls(opts)
    try:
        await client.load_markets()
    except Exception:
        logger.debug("load_markets failed/ignored")
    return client

async def fetch_ohlcv_df(client, symbol: str, timeframe: str = "1m", limit: int = 200) -> Optional[pd.DataFrame]:
    sym = symbol if "/" in symbol else f"{symbol}/{QUOTE_ASSET}"
    last_exc = None
    for attempt in range(1, OHLCV_MAX_RETRIES + 1):
        try:
            async with _ohlcv_semaphore:
                data = await client.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not data:
                return None
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(inplace=True)
            if df.empty:
                return None
            return df
        except Exception as e:
            last_exc = e
            logger.debug("fetch_ohlcv_df %s attempt %d failed: %s", sym, attempt, e)
            await asyncio.sleep(OHLCV_RETRY_DELAY * attempt)
    logger.warning("fetch_ohlcv_df failed for %s after %d attempts: %s", sym, OHLCV_MAX_RETRIES, last_exc)
    return None

async def fetch_orderbook_safe(client, symbol: str, limit: int = 50) -> dict:
    try:
        sym = symbol if "/" in symbol else f"{symbol}/{QUOTE_ASSET}"
        ob = await client.fetch_order_book(sym, limit=limit)
        return ob or {}
    except Exception:
        logger.debug("fetch_orderbook failed for %s", symbol)
        return {}

# ---------------------------
# Indicators
# ---------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    if ohlcv is None or len(ohlcv) < 2:
        return 0.0
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    prev = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return float(atr)

# ---------------------------
# BTC calm detector
# ---------------------------
async def is_btc_calm(client, tf_short: str = "1m", tf_mid: str = "5m", lookback: int = 10) -> bool:
    try:
        df1 = await fetch_ohlcv_df(client, "BTC/USDT", timeframe=tf_short, limit=lookback + 1)
        df5 = await fetch_ohlcv_df(client, "BTC/USDT", timeframe=tf_mid, limit=int(lookback/2) + 1)
        if df1 is None or df5 is None:
            return False
        for df, std_t in ((df1, BTC_CALM_STD_PCT), (df5, BTC_CALM_STD_PCT * 1.3)):
            ret = df["close"].pct_change().fillna(0)
            std = ret.tail(lookback).std()
            if std is None or std >= std_t:
                return False
            med_vol = df["volume"].median()
            last_vol = df["volume"].iloc[-1]
            vol_change = abs(last_vol - med_vol) / (med_vol + 1e-12)
            if vol_change > BTC_CALM_VOL_PCT:
                return False
        return True
    except Exception as e:
        logger.debug("is_btc_calm error: %s", e)
        return False

# ---------------------------
# TP/SL calculation (mode-aware)
# ---------------------------
def calc_tp_sl(entry: float, mode: str = "mid", atr: Optional[float] = None) -> Tuple[float, float]:
    m = mode.lower()
    if m not in TP_SL_SETTINGS:
        m = "mid"
    s = TP_SL_SETTINGS[m]
    tp_pct = (s["tp_pct_min"] + s["tp_pct_max"]) / 2.0
    sl_pct = (s["sl_pct_min"] + s["sl_pct_max"]) / 2.0
    tp = entry * (1.0 + tp_pct)
    sl = entry * (1.0 - sl_pct)
    if atr and atr > 0:
        atr_sl = entry - (1.05 * atr)
        if atr_sl < sl:
            sl = max(0.0, atr_sl)
        atr_tp = entry + (1.8 * atr)
        if atr_tp > tp:
            tp = atr_tp
    tp = round(tp, 8)
    sl = round(sl, 8)
    return tp, sl

# ---------------------------
# Leverage suggestion
# ---------------------------
def suggest_leverage(mode: str, score: float) -> int:
    m = mode.lower()
    table = LEVERAGE_SUGGEST.get(m, LEVERAGE_SUGGEST["mid"])
    for threshold, lev in table:
        if score >= threshold:
            return lev
    return table[-1][1]

# ---------------------------
# Orderbook imbalance
# ---------------------------
def orderbook_imbalance(orderbook: dict, depth_levels: int = 12) -> float:
    if not orderbook:
        return 0.0
    bids = orderbook.get("bids", [])[:depth_levels]
    asks = orderbook.get("asks", [])[:depth_levels]
    bid_vol = sum([float(x[1]) for x in bids]) if bids else 0.0
    ask_vol = sum([float(x[1]) for x in asks]) if asks else 0.0
    total = bid_vol + ask_vol + 1e-12
    return (bid_vol - ask_vol) / total

# ---------------------------
# Score engine
# ---------------------------
def score_signal(metrics: Dict[str, Any]) -> float:
    score = 0.0
    rsi = metrics.get("rsi", 50.0)
    if 40 < rsi < 70:
        score += (rsi - 40) / 30.0 * 20.0
    macd_hist = metrics.get("macd_hist", 0.0)
    if macd_hist > 0:
        score += min(20.0, macd_hist * 50.0)
    vol_ratio = metrics.get("vol_ratio", 1.0)
    if vol_ratio > 1.0:
        score += min(20.0, (vol_ratio - 1.0) * 10.0)
    vwap_gap = metrics.get("vwap_gap", 0.0)
    if vwap_gap < 0.01:
        score += max(0.0, (0.01 - vwap_gap)/0.01 * 20.0)
    return float(max(0.0, min(100.0, score)))

# ---------------------------
# Message builder
# ---------------------------
def build_signal_message(pair: str,
                         side: str,
                         mode: str,
                         entry: float,
                         tp: float,
                         sl: float,
                         score: float,
                         reason: str,
                         metrics: Dict[str, Any],
                         suggested_leverage: Optional[int] = None) -> str:
    dz_low = sl * (1 - 0.0015)
    dz_high = sl * (1 + 0.0015)
    atr = metrics.get("atr", 0.0)
    mode_icon = "⚡" if mode.lower() == "quick" else ("🔥" if mode.lower() == "mid" else "🟢")
    lev_text = f"{suggested_leverage}x" if suggested_leverage else "(manual)"
    ob = metrics.get("orderbook_imbalance")
    ob_str = f" | OB:{ob:+.2f}" if ob is not None else ""
    msg = (
        f"{mode_icon} <b>{side.upper()} SIGNAL — {mode.upper()}</b>\n"
        f"<b>Pair:</b> {pair}   <b>Score:</b> {score:.1f}%\n\n"
        f"<b>Entry:</b> <code>{entry:.8f}</code>\n"
        f"🎯 <b>TP:</b> <code>{tp:.8f}</code>   🛑 <b>SL:</b> <code>{sl:.8f}</code>\n"
        f"⚡ <b>Lev (suggest):</b> {lev_text}   <b>Qty:</b> (manual)   <b>Size:</b> (manual)\n\n"
        f"⚠️ <b>Danger Zone:</b> <code>{dz_low:.8f}</code> — <code>{dz_high:.8f}</code> (ATR={atr})\n"
        f"💡 <b>Reason:</b> {reason}{ob_str}\n"
        f"🕒 <b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    )
    return msg

# ---------------------------
# Telegram sender
# ---------------------------
async def send_telegram(message: str) -> bool:
    if NOTIFY_ONLY or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or aiohttp is None:
        logger.info("Telegram preview (NOT SENT):\n%s", message)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                text = await resp.text()
                if resp.status == 200:
                    logger.info("Telegram send ok")
                    return True
                else:
                    logger.warning("Telegram send failed status=%s text=%s", resp.status, text)
                    return False
    except Exception as e:
        logger.exception("send_telegram exception: %s", e)
        return False

# ---------------------------
# Evaluate signal (single symbol)
# ---------------------------
async def evaluate_signal(client, symbol: str, mode: str = "mid") -> Optional[Dict[str, Any]]:
    try:
        df1 = await fetch_ohlcv_df(client, symbol, timeframe="1m", limit=200)
        df5 = await fetch_ohlcv_df(client, symbol, timeframe="5m", limit=50)
        if df1 is None or df5 is None:
            return None
        c = float(df1["close"].iloc[-1])
        vol = float(df1["volume"].iloc[-1])
        vol_med = float(df1["volume"].rolling(20, min_periods=1).mean().iloc[-1] or 1.0)
        vol_ratio = vol / (vol_med + 1e-12)
        rsi = float(compute_rsi(df1["close"]).iloc[-1])
        ema20 = float(ema(df1["close"], 20).iloc[-1])
        ema50 = float(ema(df1["close"], 50).iloc[-1])
        macd_val = float((ema(df1["close"], 12) - ema(df1["close"], 26)).iloc[-1])
        vwap = ((df1["high"] + df1["low"] + df1["close"]) / 3 * df1["volume"]).sum() / (df1["volume"].sum() + 1e-12)
        vwap_gap = abs(c - vwap) / (vwap if vwap else 1.0)
        atr_val = compute_atr(df1)
        ob = await fetch_orderbook_safe(client, symbol, limit=20)
        ob_imb = orderbook_imbalance(ob)

        metrics = {
            "close": c, "vol": vol, "vol_ratio": vol_ratio, "rsi": rsi,
            "ema20": ema20, "ema50": ema50, "macd_hist": macd_val, "vwap_gap": vwap_gap,
            "atr": atr_val, "orderbook_imbalance": ob_imb
        }

        # quick filters
        if vol < max(1, vol_med * 0.5):
            return None
        spread = 0.0
        try:
            bid = float(ob["bids"][0][0])
            ask = float(ob["asks"][0][0])
            spread = abs(ask - bid) / (ask + 1e-12)
        except Exception:
            spread = 0.99
        if spread > 0.008:  # skip high spread markets
            return None

        # compute score and require MIN_SIGNAL_SCORE
        score = score_signal({"rsi": rsi, "macd_hist": macd_val, "vol_ratio": vol_ratio, "vwap_gap": vwap_gap})
        if score < MIN_SIGNAL_SCORE:
            return None

        side = "buy" if (ema20 > ema50 and macd_val > 0) else "sell" if (ema20 < ema50 and macd_val < 0) else "buy"
        tp, sl = calc_tp_sl(entry=c, mode=mode, atr=atr_val)
        # for sell invert tp/sl around entry
        if side == "sell":
            tp = round(c - (tp - c), 8)
            sl = round(c + (c - sl), 8)

        reason_parts = []
        reason_parts.append("EMA20>50" if ema20 > ema50 else "EMA20<=50")
        if macd_val > 0:
            reason_parts.append("MACD_pos")
        if vol_ratio > 1.2:
            reason_parts.append("Vol_spike")
        if spread < 0.002:
            reason_parts.append("Spread_ok")

        result = {
            "symbol": symbol if "/" in symbol else f"{symbol}/{QUOTE_ASSET}",
            "side": side,
            "mode": mode,
            "entry": round(c, 8),
            "tp": tp,
            "sl": sl,
            "score": round(score, 1),
            "reason": ", ".join(reason_parts),
            "metrics": metrics
        }
        return result
    except Exception as e:
        logger.exception("evaluate_signal error %s %s", symbol, e)
        return None

# ---------------------------
# Notify wrapper with cooldown + BTC calm
# ---------------------------
async def notify_and_log(client, diag: Dict[str, Any]) -> bool:
    try:
        sym = diag["symbol"]
        base = (sym.split("/")[0] if "/" in sym else sym).upper()
        cd = get_cooldown_manager()
        if cd.is_blocked(base):
            logger.info("notify suppressed: cooldown active for %s", base)
            return False

        btc_ok = await is_btc_calm(client)
        if not btc_ok:
            logger.info("BTC not calm — skipping emit")
            return False

        suggested_lev = suggest_leverage(diag.get("mode", "mid"), diag.get("score", 0.0))
        msg = build_signal_message(
            pair=diag["symbol"],
            side=diag["side"],
            mode=diag["mode"],
            entry=diag["entry"],
            tp=diag["tp"],
            sl=diag["sl"],
            score=diag["score"],
            reason=diag["reason"],
            metrics=diag.get("metrics", {}),
            suggested_leverage=suggested_lev
        )
        sent = await send_telegram(msg)
        if sent:
            logger.info("[NOTIFY] %s %s score=%.1f", diag["symbol"], diag["side"], diag["score"])
        else:
            logger.info("[PREVIEW] %s %s score=%.1f", diag["symbol"], diag["side"], diag["score"])
        cd.set(base, minutes=COOLDOWN_MINUTES)
        return True
    except Exception as e:
        logger.exception("notify_and_log exception: %s", e)
        return False

# ---------------------------
# Load coins CSV helper
# ---------------------------
def load_coins(path: str = "coins.csv", limit: Optional[int] = None) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    res = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.lower().startswith("symbol"):
                    continue
                # normalize to SYMBOL/QUOTE
                if "/" not in s:
                    s = f"{s}/{QUOTE_ASSET}"
                res.append(s.upper())
    except Exception:
        logger.exception("load_coins error")
    return res[:limit] if limit else res

# ---------------------------
# Cleanup
# ---------------------------
async def close_helpers():
    try:
        cd = get_cooldown_manager()
        cd._save()
        logger.info("helpers cleanup done")
    except Exception:
        pass

# ---------------------------
# Demo entrypoint (not for production) — shows usage pattern
# ---------------------------
if __name__ == "__main__":
    print("helpers.py loaded — import into main loop.")
    print("This helpers module supports async operation with ccxt.async_support.")
