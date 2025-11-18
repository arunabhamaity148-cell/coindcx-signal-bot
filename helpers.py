# helpers.py (PART 1)
# Required packages: ccxt, pandas, numpy, requests
# Make sure requirements.txt has: ccxt, pandas, numpy, requests

import os
import json
import time
import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import numpy as np
import requests

logger = logging.getLogger("helpers")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# -------------------------
# Exchange helper
# -------------------------
def create_exchange(name: str = "binance", api_key: Optional[str] = None, secret: Optional[str] = None, enable_rate_limit: bool = True):
    """Create and return a ccxt exchange object. Pass API keys via args or env."""
    exch_class = getattr(ccxt, name)
    params = {"enableRateLimit": enable_rate_limit}
    api_key = api_key or os.getenv(f"{name.upper()}_API_KEY")
    secret = secret or os.getenv(f"{name.upper()}_API_SECRET")
    cfg = {**params}
    if api_key and secret:
        cfg.update({"apiKey": api_key, "secret": secret})
    try:
        ex = exch_class(cfg)
        # set timeout a bit high
        ex.timeout = 20000
        logger.info(f"Created exchange: {name}")
        return ex
    except Exception as e:
        logger.exception("Failed to create exchange %s: %s", name, e)
        raise


# -------------------------
# Coins loader
# -------------------------
def load_coins(path: str = "coins.csv") -> List[str]:
    """
    Load coins from a simple CSV containing one column 'symbol' or no header lines like:
    BTC
    ETH
    Or with header:
    symbol
    BTC
    """
    if not os.path.exists(path):
        logger.warning("coins file not found: %s. Returning default ['BTC','ETH','XRP']", path)
        return ["BTC", "ETH", "XRP"]
    try:
        df = pd.read_csv(path, header=0 if pd.read_csv(path, nrows=1).columns[0].lower() == "symbol" else None)
        if df.shape[1] == 1:
            col = df.columns[0]
            symbols = df[col].astype(str).str.strip().tolist()
        else:
            # If header present with column name 'symbol'
            if "symbol" in df.columns:
                symbols = df["symbol"].astype(str).str.strip().tolist()
            else:
                # fallback: first column
                symbols = df.iloc[:, 0].astype(str).str.strip().tolist()
        # remove empties and duplicates
        symbols = [s for s in symbols if s and str(s).upper() != "SYMBOL"]
        symbols = list(dict.fromkeys([s.upper() for s in symbols]))
        logger.info("Loaded %d coins from %s", len(symbols), path)
        return symbols
    except Exception as e:
        logger.exception("Failed load_coins: %s", e)
        return []


# -------------------------
# OHLCV fetcher
# -------------------------
def get_ohlcv_sample(exchange, symbol: str, timeframe: str = "1m", limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV and return pandas DataFrame with columns: ts, open, high, low, close, volume
    symbol should be like 'BTC/USDT'
    """
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except ccxt.BadSymbol as e:
        logger.warning("BadSymbol for %s: %s", symbol, e)
        return None
    except Exception as e:
        logger.exception("Error fetch_ohlcv for %s: %s", symbol, e)
        return None


# -------------------------
# Indicators (pandas / numpy)
# -------------------------
def sma(series: pd.Series, period: int) -> float:
    return float(series.rolling(period).mean().iloc[-1])


def ema(series: pd.Series, period: int) -> float:
    """Exponential moving average using pandas"""
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-10)
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = true_range(high, low, close)
    return float(tr.rolling(period).mean().iloc[-1])


# ADX implementation
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = true_range(high, low, close)
    atr_series = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr_series + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr_series + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx_series = dx.rolling(period).mean()
    return float(adx_series.iloc[-1])


# -------------------------
# Cooldown manager (persisted)
# -------------------------
class CooldownManager:
    def __init__(self, filepath: str = "cooldown.json"):
        self.filepath = filepath
        self._data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self._data, f, default=str)
        except Exception as e:
            logger.exception("Cooldown save failed: %s", e)

    def allowed(self, symbol: str, mode: str = "ANY") -> bool:
        """
        Returns True if allowed to signal for this symbol.
        mode may be QUICK/MID/TREND/ANY
        """
        key = symbol.upper()
        now = datetime.utcnow().timestamp()
        rec = self._data.get(key)
        if not rec:
            return True
        # rec: {"mode": "QUICK", "until": timestamp}
        until = rec.get("until", 0)
        if now >= until:
            return True
        return False

    def mark(self, symbol: str, mode: str, ttl_seconds: int = 1800):
        key = symbol.upper()
        until = datetime.utcnow().timestamp() + ttl_seconds
        self._data[key] = {"mode": mode, "until": until}
        self._save()
        logger.debug("Cooldown marked %s until %s", key, until)

    def clear(self, symbol: str):
        key = symbol.upper()
        if key in self._data:
            del self._data[key]
            self._save()


# -------------------------
# Coin limiter
# -------------------------
class CoinLimiter:
    def __init__(self, max_coins: int = 12):
        self.max_coins = max_coins
        self.active = set()  # symbols currently considered 'used' in timeframe

    def allow_new(self) -> bool:
        return len(self.active) < self.max_coins

    def mark_used(self, symbol: str):
        self.active.add(symbol.upper())

    def unmark(self, symbol: str):
        self.active.discard(symbol.upper())

    def reset(self):
        self.active.clear()


# -------------------------
# Telegram sender (simple requests)
# -------------------------
def send_telegram_signal(token: str, chat_id: str, signal: Dict[str, Any]) -> bool:
    """
    signal: dict with fields: mode, side, entry, tp, sl, leverage, reason, danger_zone (optional), accuracy (optional)
    This function formats a message and sends to Telegram via bot API.
    """
    if not token or not chat_id:
        logger.warning("Telegram token/chat_id not set. Skipping telegram send.")
        return False
    try:
        text_lines = []
        mode = signal.get("mode", "QUICK")
        side = signal.get("side", "BUY")
        symbol = signal.get("symbol", "")
        entry = signal.get("entry")
        tp = signal.get("tp")
        sl = signal.get("sl")
        lev = signal.get("leverage", "")
        reason = signal.get("reason", "")
        accuracy = signal.get("accuracy", None)
        danger = signal.get("danger_zone", None)

        header = f"🔥 {side} SIGNAL — {mode}\nSymbol: {symbol}"
        text_lines.append(header)
        if entry:
            text_lines.append(f"Entry: {entry}")
        if tp:
            text_lines.append(f"🎯 TP: {tp}")
        if sl:
            text_lines.append(f"🛑 SL: {sl}")
        if lev:
            text_lines.append(f"⚡ Leverage: {lev}x")
        if reason:
            text_lines.append(f"\n📌 Reason:\n{reason}")
        if danger:
            # danger can be dict {'price':..., 'note':...}
            dz = f"\n🟧 Danger Zone: price ~ {danger.get('price')} — {danger.get('note')}"
            text_lines.append(dz)
        if accuracy is not None:
            text_lines.append(f"\n📊 Accuracy: {accuracy}%")
        # join
        text = "\n".join(text_lines)
        # urlencode & post
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            logger.info("Telegram message sent for %s", symbol)
            return True
        else:
            logger.warning("Telegram send failed: %s %s", r.status_code, r.text)
            return False
    except Exception as e:
        logger.exception("Telegram send error: %s", e)
        return False
# -------------------------
# helpers.py (PART 2)
# Append this to the end of PART 1
# -------------------------

import math
from collections import deque

# ---------- Configurable defaults ----------
DEFAULT_TTL_QUICK = 60 * 5        # 5 minutes quick cooldown (example)
DEFAULT_TTL_MID = 60 * 30         # 30 minutes mid
DEFAULT_TTL_TREND = 60 * 60 * 4   # 4 hours trend
SAME_COIN_BLOCK_SECONDS = 60 * 30  # 30 minutes block for same coin alerts
HISTORY_FILE = "signal_history.json"
LAST_SIGNALS_FILE = "last_signals.json"

# ---------- Persistence helpers ----------
def _load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json_file(path: str, obj: Dict[str, Any]):
    try:
        with open(path, "w") as f:
            json.dump(obj, f, default=str, indent=2)
    except Exception as e:
        logger.exception("Save json failed %s: %s", path, e)


# ---------- Signal utils ----------
def tp_sl_calc(entry: float, atr_value: Optional[float] = None, pct_tp: float = 0.3, pct_sl: float = 0.2,
               side: str = "BUY", leverage: int = 50) -> Tuple[float, float]:
    """
    Calculate TP and SL.
    - If atr_value provided, use ATR multiples (preferred for volatile coins).
    - Otherwise use pct_tp and pct_sl percentages.
    Returns (tp, sl)
    """
    if entry is None:
        raise ValueError("entry is required")
    if atr_value and atr_value > 0:
        # use ATR multiples: TP = entry +/- 3*ATR, SL = entry +/- 1*ATR (example)
        if side.upper() == "BUY":
            tp = entry + 3 * atr_value
            sl = max(0.0, entry - 1 * atr_value)
        else:
            tp = max(0.0, entry - 3 * atr_value)
            sl = entry + 1 * atr_value
    else:
        if side.upper() == "BUY":
            tp = entry * (1 + pct_tp / 100)
            sl = entry * (1 - pct_sl / 100)
        else:
            tp = entry * (1 - pct_tp / 100)
            sl = entry * (1 + pct_sl / 100)
    # adjust for leverage -> keep absolute price TP/SL (leverage only for position sizing)
    return round(tp, 8), round(sl, 8)


def danger_zone_calc(df: pd.DataFrame, threshold_atr_mult: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    Build a simple danger-zone dict using ATR amplitude: if current ATR * mult > recent price range => danger.
    Returns: {"price": last_close, "note": "..."} or None
    """
    try:
        last = df.iloc[-1]
        atr_val = atr(df["high"], df["low"], df["close"], period=14)
        range_ = last["high"] - last["low"]
        if atr_val * threshold_atr_mult > range_:
            return {"price": float(last["close"]), "note": f"High ATR ({atr_val:.4f}) vs range {range_:.4f}"}
        return None
    except Exception:
        return None


def estimate_accuracy_and_regime(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Very simple accuracy/regime estimator:
    - regime = 'bull' if EMA(50) > EMA(200) else 'bear'
    - accuracy score (0-100) based on volatility + trend strength heuristics (rough)
    """
    try:
        close = df["close"]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        regime = "bull" if ema50 > ema200 else "bear"
        # volatility = ATR / price
        atr_val = atr(df["high"], df["low"], df["close"], period=14)
        vol = atr_val / (close.iloc[-1] + 1e-10)
        # trend_strength via MACD hist magnitude normalized
        _, _, macdh = macd(close, 12, 26, 9)
        macd_score = min(1.0, abs(macdh) / (abs(close.iloc[-1]) + 1e-10))
        # combine into score
        score = max(20, min(95, int((1 - vol) * 60 + macd_score * 40)))
        return score, regime
    except Exception:
        return 50, "unknown"


# ---------- Full evaluator ----------
class SignalEvaluator:
    def __init__(self, exchange, telegram_token: Optional[str] = None, telegram_chat: Optional[str] = None,
                 cooldown: Optional[CooldownManager] = None, limiter: Optional[CoinLimiter] = None):
        self.exchange = exchange
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat = telegram_chat or os.getenv("TELEGRAM_CHAT_ID")
        self.cooldown = cooldown or CooldownManager()
        self.limiter = limiter or CoinLimiter(max_coins=12)
        self._last_signals = _load_json_file(LAST_SIGNALS_FILE)
        self._history = _load_json_file(HISTORY_FILE)

    def _save_last_signal(self, sig: Dict[str, Any]):
        key = f"{sig.get('symbol')}_{sig.get('mode')}"
        self._last_signals[key] = {**sig, "ts": datetime.utcnow().isoformat()}
        _save_json_file(LAST_SIGNALS_FILE, self._last_signals)

    def _append_history(self, sig: Dict[str, Any]):
        arr = self._history.get("signals", [])
        arr.append({**sig, "ts": datetime.utcnow().isoformat()})
        # keep only recent 1000
        if len(arr) > 1000:
            arr = arr[-1000:]
        self._history["signals"] = arr
        _save_json_file(HISTORY_FILE, self._history)

    def _blocked_by_recent(self, symbol: str) -> bool:
        """Block same coin within SAME_COIN_BLOCK_SECONDS"""
        now = datetime.utcnow().timestamp()
        key = symbol.upper()
        rec = self._last_signals.get(key)
        if not rec:
            return False
        last_ts = rec.get("timestamp") or datetime.fromisoformat(rec.get("ts")).timestamp() if rec.get("ts") else 0
        return (now - last_ts) < SAME_COIN_BLOCK_SECONDS

    def evaluate_symbol_all_modes(self, base_symbol: str, quote: str = "USDT", timeframe: str = "1m") -> Optional[Dict[str, Any]]:
        """
        Evaluate symbol and possibly return a full signal dict.
        Modes: QUICK, MID, TREND
        This is a basic rule-based example — tune thresholds to your strategy.
        """
        symbol_pair = f"{base_symbol}/{quote}"
        df = get_ohlcv_sample(self.exchange, symbol_pair, timeframe=timeframe, limit=300)
        if df is None or df.empty:
            logger.debug("No ohlcv for %s", symbol_pair)
            return None

        # basic indicators
        close = df["close"]
        last_close = float(close.iloc[-1])
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        rsi14 = rsi(close, 14)
        macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
        atr_val = atr(df["high"], df["low"], df["close"], period=14)
        adx_val = adx(df["high"], df["low"], df["close"], period=14)

        # regime & accuracy
        accuracy, regime = estimate_accuracy_and_regime(df)

        # quick rule (scalping): ema9 > ema21, rsi between 40-70, small adx threshold
        if ema9 > ema21 and rsi14 > 40 and macd_hist > 0 and adx_val > 15:
            mode = "QUICK"
            side = "BUY"
            entry = last_close
            tp, sl = tp_sl_calc(entry, atr_value=atr_val, side=side)
            danger = danger_zone_calc(df)
            # check cooldowns and same coin block
            if not self.cooldown.allowed(base_symbol, mode):
                logger.info("Cooldown blocks %s for mode %s", base_symbol, mode)
                return None
            if self._blocked_by_recent(base_symbol):
                logger.info("Same coin block active for %s", base_symbol)
                return None
            # build signal
            sig = {
                "mode": mode,
                "symbol": symbol_pair,
                "side": side,
                "entry": round(entry, 8),
                "tp": tp,
                "sl": sl,
                "leverage": 50,
                "reason": f"EMA9({ema9:.2f})>EMA21({ema21:.2f}), RSI14={rsi14:.1f}, MACDh={macd_hist:.6f}, ADX={adx_val:.2f}",
                "accuracy": accuracy,
                "regime": regime,
                "danger_zone": danger,
                "timestamp": datetime.utcnow().isoformat()
            }
            # mark cooldowns & save
            self.cooldown.mark(base_symbol, mode, ttl_seconds=DEFAULT_TTL_QUICK)
            self._save_last_signal(sig)
            self._append_history(sig)
            # send telegram
            send_telegram_signal(self.telegram_token, self.telegram_chat, sig)
            logger.info("SIGNAL SENT for %s", symbol_pair)
            return sig

        # mid rule: trend confirmation on higher timeframe (use EMA cross + ADX strong)
        # (demonstrative; user should extend with multi-timeframe)
        if ema21 > ema9 and adx_val > 25 and abs(macd_hist) > 0.5:
            mode = "MID"
            side = "SELL" if macd_hist < 0 else "BUY"
            entry = last_close
            tp, sl = tp_sl_calc(entry, atr_value=atr_val, side=side)
            danger = danger_zone_calc(df)
            if not self.cooldown.allowed(base_symbol, mode):
                return None
            if self._blocked_by_recent(base_symbol):
                return None
            sig = {
                "mode": mode,
                "symbol": symbol_pair,
                "side": side,
                "entry": round(entry, 8),
                "tp": tp,
                "sl": sl,
                "leverage": 20,
                "reason": f"MID rule ADX {adx_val:.2f}, MACDh {macd_hist:.4f}",
                "accuracy": max(50, accuracy - 5),
                "regime": regime,
                "danger_zone": danger,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.cooldown.mark(base_symbol, mode, ttl_seconds=DEFAULT_TTL_MID)
            self._save_last_signal(sig)
            self._append_history(sig)
            send_telegram_signal(self.telegram_token, self.telegram_chat, sig)
            logger.info("MID SIGNAL SENT for %s", symbol_pair)
            return sig

        # trend rule: longer timeframe confirmation (user can call evaluate on higher tf)
        # Example: require ema50 > ema200 and ADX > 30
        # We'll compute coarse ema50/200 from close (works but prefer higher tf)
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        if ema50 > ema200 and adx_val > 30:
            mode = "TREND"
            side = "BUY"
            entry = last_close
            tp, sl = tp_sl_calc(entry, atr_value=atr_val, side=side, pct_tp=1.0, pct_sl=0.5)
            danger = danger_zone_calc(df)
            if not self.cooldown.allowed(base_symbol, mode):
                return None
            if self._blocked_by_recent(base_symbol):
                return None
            sig = {
                "mode": mode,
                "symbol": symbol_pair,
                "side": side,
                "entry": round(entry, 8),
                "tp": tp,
                "sl": sl,
                "leverage": 10,
                "reason": f"TREND rule EMA50>{ema200:.2f} ADX{adx_val:.2f}",
                "accuracy": max(60, accuracy),
                "regime": regime,
                "danger_zone": danger,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.cooldown.mark(base_symbol, mode, ttl_seconds=DEFAULT_TTL_TREND)
            self._save_last_signal(sig)
            self._append_history(sig)
            send_telegram_signal(self.telegram_token, self.telegram_chat, sig)
            logger.info("TREND SIGNAL SENT for %s", symbol_pair)
            return sig

        # no signal
        return None


# ---------- Small example runner ----------
if __name__ == "__main__":
    # quick local test (requires API key or public market access)
    ex = create_exchange("binance")
    cd = CooldownManager()
    cl = CoinLimiter(max_coins=8)
    ev = SignalEvaluator(ex, telegram_token=None, telegram_chat=None, cooldown=cd, limiter=cl)

    coins = load_coins("coins.csv")[:6]
    for c in coins:
        try:
            sig = ev.evaluate_symbol_all_modes(c, quote="USDT", timeframe="1m")
            if sig:
                print("SIGNAL:", sig)
            else:
                print("No signal for", c)
            time.sleep(1)  # avoid rate limit in quick loop
        except Exception as e:
            print("Error for", c, e)