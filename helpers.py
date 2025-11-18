# helpers.py
"""
Helpers for coindcx-signal-bot
- load_coins()
- get_ohlcv()
- indicators (EMA, RSI, MACD, ADX, ATR, Bollinger)
- signal engines: quick/mid/trend
- TP/SL calculators
- Cooldown / Rate limiter / Coin limit manager
- Telegram formatter + sender
Requirements:
  - ccxt
  - pandas
  - numpy
  - python-telegram-bot
  - ta (optional, but included simple implementations)
Environment variables needed:
  - TELEGRAM_TOKEN
  - TELEGRAM_CHAT_ID
  - EXCHANGE_API keys if private calls needed
"""

import os
import time
import math
import csv
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
from requests import Session

# Optional: from telegram import Bot   # python-telegram-bot v13/20 usage (we'll use simple requests if not installed)

# ----------------------
# CONFIG (editable)
# ----------------------
COINS_CSV = "coins.csv"
DEFAULT_EXCHANGE = "binance"
OHLCV_LIMIT = 100
TIMEFRAME = "1m"  # primary timeframe for Quick; main.py can override
COIN_LIMIT = 8  # maximum coins to check per loop (increase as requested)
SAME_COIN_BLOCK_MINUTES = 30  # block repeated alerts for same coin
MODERATE_ALERT_MIN = 15  # moderate alert min cooldown
MODERATE_ALERT_MAX = 30  # moderate alert max cooldown
GLOBAL_LOOP_DELAY = 10  # seconds between cycles (tune in main)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Indicator params
EMA_PERIODS = [9, 20, 50, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0

# Confidence thresholds
CONF_HIGH = 80
CONF_MED = 60
CONF_LOW = 40

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----------------------
# Utility & state classes
# ----------------------
class CooldownManager:
    """Manage cooldowns and blocking for coins and alert types."""

    def __init__(self):
        # last_alerts: {coin: datetime}
        self.last_alerts: Dict[str, datetime] = {}
        # moderate alerts: separate last times for moderate alerts to throttle them
        self.last_moderate: Dict[str, datetime] = {}
        # global counters (per-run)
        self.alert_counters: Dict[str, int] = {}

        # lock for thread-safety
        self._lock = threading.Lock()

    def can_alert(self, coin: str, mode: str) -> bool:
        """Return True if allowed to send alert for this coin+mode now."""
        now = datetime.utcnow()
        with self._lock:
            last = self.last_alerts.get(coin)
            if last:
                delta = (now - last).total_seconds() / 60.0
                if delta < SAME_COIN_BLOCK_MINUTES:
                    return False

            if mode.lower() == "moderate":
                last_mod = self.last_moderate.get(coin)
                if last_mod:
                    delta = (now - last_mod).total_seconds() / 60.0
                    # throttle moderate alerts into window [MODERATE_ALERT_MIN, MODERATE_ALERT_MAX]
                    if delta < MODERATE_ALERT_MIN:
                        return False

            # coin limit check (per-run)
            cnt = self.alert_counters.get(coin, 0)
            if cnt >= 3:  # per-coin-per-run soft limit (tunable)
                return False

            return True

    def register_alert(self, coin: str, mode: str):
        now = datetime.utcnow()
        with self._lock:
            self.last_alerts[coin] = now
            if mode.lower() == "moderate":
                self.last_moderate[coin] = now
            self.alert_counters[coin] = self.alert_counters.get(coin, 0) + 1

    def reset_counters(self):
        with self._lock:
            self.alert_counters = {}


class CoinLimiter:
    """Manage how many coins to process and enforce global coin limit per loop."""

    def __init__(self, limit: int = COIN_LIMIT):
        self.limit = limit

    def slice_coins(self, coins: List[str]) -> List[str]:
        return coins[: self.limit]


# ----------------------
# Load coins
# ----------------------
def load_coins(csv_path: str = COINS_CSV) -> List[str]:
    """Load coin symbols from coins.csv, header 'symbol' expected.
    Returns upper-case list of symbols (no /USDT suffix).
    """
    coins = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # try header detection
            start = 0
            if rows and rows[0] and rows[0][0].strip().lower() == "symbol":
                start = 1
            for r in rows[start:]:
                if not r:
                    continue
                sym = r[0].strip()
                if not sym:
                    continue
                coins.append(sym.upper())
    except FileNotFoundError:
        logging.error("coins.csv not found. Using default small list.")
        coins = ["BTC", "ETH", "XRP"]
    return coins


# ----------------------
# Exchange / OHLCV
# ----------------------
def create_exchange(name: str = DEFAULT_EXCHANGE, rate_limit: bool = True) -> ccxt.Exchange:
    ex_cls = getattr(ccxt, name)
    ex = ex_cls({"enableRateLimit": rate_limit})
    # If you need API keys for some endpoints, set them via env and uncomment:
    # ex.apiKey = os.getenv("EXCHANGE_API_KEY")
    # ex.secret = os.getenv("EXCHANGE_API_SECRET")
    return ex


def get_ohlcv(
    ex: ccxt.Exchange, symbol: str, pair_suffix: str = "USDT", timeframe: str = TIMEFRAME, limit: int = OHLCV_LIMIT, retry: int = 3, backoff: int = 2
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV and return pandas DataFrame with columns: ts, open, high, low, close, volume"""
    pair = f"{symbol}/{pair_suffix}"
    for attempt in range(retry):
        try:
            raw = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            if not raw:
                logging.warning(f"Empty OHLCV for {pair}")
                return None
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            return df
        except ccxt.BadSymbol as e:
            logging.error(f"BadSymbol: {pair} not found on exchange ({e})")
            return None
        except Exception as e:
            logging.warning(f"get_ohlcv attempt {attempt+1}/{retry} failed for {pair}: {e}")
            time.sleep(backoff * (attempt + 1))
            continue
    return None


# ----------------------
# Indicators (pandas)
# ----------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")


def adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    # Simple ADX implementation
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr_series + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr_series + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx_series = dx.rolling(period).mean()
    return adx_series.fillna(method="bfill")


def bollinger_bands(series: pd.Series, period: int = BB_PERIOD, std: float = BB_STD) -> pd.DataFrame:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + std * sd
    lower = ma - std * sd
    return pd.DataFrame({"ma": ma, "upper": upper, "lower": lower})


# ----------------------
# Signal & calculators
# ----------------------
def calculate_tp_sl(
    last_price: float, atr_value: float, mode: str = "quick", risk_reward: float = 1.5
) -> Tuple[float, float]:
    """Return TP, SL based on ATR and mode (quick: tighter, trend: wider)."""
    if mode.lower() == "quick":
        sl = last_price - 1.0 * atr_value
        tp = last_price + risk_reward * (last_price - sl)
    elif mode.lower() == "mid":
        sl = last_price - 1.5 * atr_value
        tp = last_price + risk_reward * (last_price - sl)
    else:  # trend
        sl = last_price - 2.5 * atr_value
        tp = last_price + risk_reward * (last_price - sl)
    # ensure positive
    sl = max(0.0, sl)
    tp = max(tp, last_price * 1.001)
    return tp, sl


def compute_confidence(scores: Dict[str, float]) -> int:
    """Combine several sub-scores into an integer confidence 0-100."""
    # example weighting
    w = {"trend": 0.35, "momentum": 0.25, "volume": 0.2, "volatility": 0.2}
    total = 0.0
    for k, weight in w.items():
        total += scores.get(k, 50) * weight
    return int(max(0, min(100, round(total))))


def reason_from_flags(flags: Dict[str, bool]) -> str:
    parts = []
    for k, v in flags.items():
        if v:
            parts.append(k.replace("_", " ").upper())
    return " + ".join(parts) if parts else "No strong reason"


def evaluate_quick(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick mode signal heuristics. Returns dict with side, score, flags."""
    close = df["close"]
    last = float(close.iloc[-1])
    ema9 = ema(close, 9).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]
    r = float(rsi(close).iloc[-1])
    mac = macd(close).iloc[-1]
    atr_v = float(atr(df).iloc[-1])
    vol = float(df["volume"].iloc[-1])

    flags = {
        "ema9_above_ema20": ema9 > ema20,
        "rsi_okay": r < 70 and r > 30,
        "macd_hist_positive": mac["hist"] > 0,
        "volume_spike": vol > df["volume"].rolling(20).mean().iloc[-2] * 1.8 if df["volume"].rolling(20).mean().iloc[-2] > 0 else False,
    }
    # simple score composition
    trend_score = 80 if flags["ema9_above_ema20"] else 30
    momentum_score = 80 if flags["macd_hist_positive"] else 30
    volume_score = 90 if flags["volume_spike"] else 40
    vol_score = 70 if atr_v / last < 0.02 else 50

    conf = compute_confidence({"trend": trend_score, "momentum": momentum_score, "volume": volume_score, "volatility": vol_score})
    side = "BUY" if flags["ema9_above_ema20"] and flags["macd_hist_positive"] and r < 70 else ("SELL" if not flags["ema9_above_ema20"] and not flags["macd_hist_positive"] and r > 30 else "NEUTRAL")
    return {"side": side, "confidence": conf, "last": last, "atr": atr_v, "flags": flags}


def evaluate_mid(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["close"]
    last = float(close.iloc[-1])
    ema20v = ema(close, 20).iloc[-1]
    ema50v = ema(close, 50).iloc[-1]
    r = float(rsi(close).iloc[-1])
    m = macd(close)
    adx_v = float(adx(df).iloc[-1])
    atr_v = float(atr(df).iloc[-1])
    flags = {
        "ema20_above_50": ema20v > ema50v,
        "macd_hist_pos": m["hist"].iloc[-1] > 0,
        "adx_strong": adx_v > 22,
    }
    trend_score = 90 if flags["ema20_above_50"] else 30
    momentum_score = 80 if flags["macd_hist_pos"] else 35
    volume_score = 60
    vol_score = 80 if atr_v / last < 0.03 else 50
    conf = compute_confidence({"trend": trend_score, "momentum": momentum_score, "volume": volume_score, "volatility": vol_score})
    side = "BUY" if flags["ema20_above_50"] and flags["macd_hist_pos"] and adx_v > 20 else ("SELL" if not flags["ema20_above_50"] and not flags["macd_hist_pos"] and adx_v > 20 else "NEUTRAL")
    return {"side": side, "confidence": conf, "last": last, "atr": atr_v, "flags": flags}


def evaluate_trend(df: pd.DataFrame, higher_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    close = df["close"]
    last = float(close.iloc[-1])
    ema200 = ema(close, 200).iloc[-1] if len(close) >= 200 else ema(close, 50).iloc[-1]
    adx_v = float(adx(df).iloc[-1])
    atr_v = float(atr(df).iloc[-1])
    flags = {"above_ema200": last > ema200, "adx_strong": adx_v > 25}
    trend_score = 95 if flags["above_ema200"] else 40
    momentum_score = 80
    vol_score = 60
    conf = compute_confidence({"trend": trend_score, "momentum": momentum_score, "volume": vol_score, "volatility": 70})
    side = "BUY" if flags["above_ema200"] and flags["adx_strong"] else ("SELL" if not flags["above_ema200"] and flags["adx_strong"] else "NEUTRAL")
    return {"side": side, "confidence": conf, "last": last, "atr": atr_v, "flags": flags}


# ----------------------
# Telegram helper (simple)
# ----------------------
def send_telegram_message(text: str, token: Optional[str] = TELEGRAM_TOKEN, chat_id: Optional[str] = TELEGRAM_CHAT_ID) -> bool:
    if not token or not chat_id:
        logging.warning("Telegram token/chat_id not set; skipping send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        s = Session()
        r = s.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logging.error("Failed to send telegram message: %s", e)
        return False


def format_signal_message(symbol: str, mode: str, side: str, entry: float, tp: float, sl: float, leverage: int, confidence: int, reason: str, danger_zone: Optional[str] = None) -> str:
    msg = []
    emoji = "🔥" if side.upper() == "BUY" else "❗" if side.upper() == "SELL" else "ℹ️"
    msg.append(f"{emoji} *{side.upper()} SIGNAL — {mode.upper()}*")
    msg.append(f"*Symbol:* {symbol}/USDT")
    msg.append(f"*Entry:* {entry}")
    msg.append(f"*TP:* {tp}")
    msg.append(f"*SL:* {sl}")
    msg.append(f"*Leverage:* {leverage}x")
    msg.append("")
    msg.append(f"*Reason:* {reason}")
    if danger_zone:
        msg.append(f"*Danger Zone:* {danger_zone}")
    msg.append(f"*Confidence:* {confidence}%")
    msg.append(f"_Timestamp:_ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return "\n".join(msg)


# ----------------------
# High-level wrapper: evaluate symbol for all modes
# ----------------------
def evaluate_symbol_all_modes(ex: ccxt.Exchange, symbol: str, pair_suffix: str = "USDT", timeframe: str = TIMEFRAME) -> Optional[Dict[str, Any]]:
    df = get_ohlcv(ex, symbol, pair_suffix=pair_suffix, timeframe=timeframe, limit=300)
    if df is None or df.empty:
        return None
    # compute quick/mid/trend
    quick = evaluate_quick(df)
    mid = evaluate_mid(df)
    trend = evaluate_trend(df)
    out = {"symbol": symbol, "quick": quick, "mid": mid, "trend": trend}
    return out


# ----------------------
# Danger zone builder
# ----------------------
def compute_danger_zone(df: pd.DataFrame) -> str:
    """Return a simple price range string describing danger zone (ATR-based)."""
    last = float(df["close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    lower = last - 1.5 * a
    upper = last + 1.5 * a
    return f"${lower:.4f} - ${upper:.4f}"


# ----------------------
# Example usage snippet (to be called from main.py)
# ----------------------
if __name__ == "__main__":
    # quick local test
    coins = load_coins()
    coins = coins[: COIN_LIMIT]
    ex = create_exchange()
    cd = CooldownManager()
    limiter = CoinLimiter(limit=COIN_LIMIT)
    slice_coins = limiter.slice_coins(coins)
    logging.info("Loaded coins: %s", slice_coins)
    for c in slice_coins:
        res = evaluate_symbol_all_modes(ex, c)
        if not res:
            logging.info("No data for %s", c)
            continue
        # example: if quick says BUY and confidence high -> build message and send
        q = res["quick"]
        if q["side"] == "BUY" and q["confidence"] >= CONF_MED and cd.can_alert(c, "quick"):
            tp, sl = calculate_tp_sl(q["last"], q["atr"], mode="quick")
            reason = reason_from_flags(q["flags"])
            danger = compute_danger_zone(get_ohlcv(ex, c, limit=100))
            msg = format_signal_message(c, "quick", "BUY", q["last"], tp, sl, leverage=50, confidence=q["confidence"], reason=reason, danger_zone=danger)
            logging.info("SIGNAL: %s", msg)
            send_telegram_message(msg)
            cd.register_alert(c, "quick")