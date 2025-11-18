import os
import ccxt
import time
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------
# ENV + LOGGING SETUP
# -----------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# -----------------------------
# LOAD COINS FROM CSV
# -----------------------------
def load_coins(file_path="coins.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("coins.csv NOT FOUND!")

    df = pd.read_csv(file_path)
    coins = df["symbol"].astype(str).tolist()
    logger.info(f"Loaded {len(coins)} coins from {file_path}")
    return coins


# -----------------------------
# CREATE EXCHANGE (BINANCE)
# -----------------------------
def create_exchange():
    try:
        ex = ccxt.binance({
            "enableRateLimit": True
        })
        logger.info("Exchange created successfully")
        return ex
    except Exception as e:
        logger.error(f"Exchange create failed: {e}")
        return None


# -----------------------------
# CHECK IF MARKET EXISTS
# -----------------------------
def market_exists(ex, symbol):
    try:
        markets = ex.load_markets()
        return symbol in markets
    except:
        return False


# -----------------------------
# FETCH OHLCV (ROBUST)
# -----------------------------
def fetch_ohlcv_sample(ex, symbol, timeframe="1m", limit=200):
    """
    Returns clean OHLCV DataFrame.
    Handles:
        - retries
        - dead candles
        - fillna fix
    """
    if not market_exists(ex, symbol):
        logger.warning(f"Market not found: {symbol}")
        return None

    for attempt in range(4):
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )

            df["time"] = pd.to_datetime(df["time"], unit="ms")

            # Clean candles
            df = df.ffill().bfill()

            return df

        except Exception as e:
            logger.warning(f"[{symbol}] OHLCV fetch error ({attempt+1}/4): {e}")
            time.sleep(1.2)

    logger.error(f"[{symbol}] Failed to fetch OHLCV after retries")
    return None


# -----------------------------
# SAMPLE TEST
# -----------------------------
if __name__ == "__main__":
    ex = create_exchange()
    df = fetch_ohlcv_sample(ex, "BTC/USDT")
    print(df.tail())
import os
import json
import math
import time
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# -----------------------------
# COOLDOWN (PERSISTED)
# -----------------------------
COOLDOWN_FILE = "cooldown.json"

def _load_cooldown() -> Dict[str, Dict]:
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Couldn't read cooldown file: {e}")
    return {}

def _save_cooldown(state: Dict[str, Dict]):
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"Couldn't write cooldown file: {e}")

def is_on_cooldown(symbol: str, mode: str) -> bool:
    """
    mode: "QUICK", "MID", "TREND"
    cooldowns:
      QUICK -> 30 min
      MID   -> 45 min
      TREND -> 90 min
    """
    mapping = {"QUICK": 30, "MID": 45, "TREND": 90}  # minutes
    state = _load_cooldown()
    key = f"{symbol}:{mode}"
    if key not in state:
        return False
    expires_at = datetime.fromisoformat(state[key]["expires_at"])
    if datetime.utcnow() >= expires_at:
        # expired -> remove
        del state[key]
        _save_cooldown(state)
        return False
    return True

def set_cooldown(symbol: str, mode: str):
    mapping = {"QUICK": 30, "MID": 45, "TREND": 90}
    minutes = mapping.get(mode.upper(), 30)
    expires_at = (datetime.utcnow() + timedelta(minutes=minutes)).isoformat()
    state = _load_cooldown()
    state[f"{symbol}:{mode}"] = {"expires_at": expires_at}
    _save_cooldown(state)
    logger.info(f"Cooldown set for {symbol} mode={mode} until {expires_at}")

# -----------------------------
# TECHNICAL INDICATORS (pandas)
# -----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def adx(df: pd.DataFrame, length=14) -> pd.Series:
    # basic ADX implementation
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = true_range(df)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

# -----------------------------
# FEATURE BUILDER
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA20/50/200, RSI14, MACD, ADX, vol_ma and returns df
    """
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["adx14"] = adx(df, 14)
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["vol_rel"] = df["volume"] / (df["vol_ma20"].replace(0, 1))
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["candle_body_rel"] = df["candle_body"] / df["close"]
    return df

# -----------------------------
# SIGNAL RULES
# -----------------------------
def detect_regime(df: pd.DataFrame) -> str:
    """
    Simple regime detection: compare short EMA vs long EMA
    returns: 'bull', 'bear', 'side'
    """
    latest = df.iloc[-1]
    if latest["ema20"] > latest["ema50"] > latest["ema200"]:
        return "bull"
    if latest["ema20"] < latest["ema50"] < latest["ema200"]:
        return "bear"
    return "side"

def quick_condition(df: pd.DataFrame) -> Optional[str]:
    """
    Quick: fast scalp signals using EMA crossover + volume spike + MACD hist > 0
    Return 'BUY' or 'SELL' or None
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # EMA cross (20 over 50)
    ema_cross_buy = (prev["ema20"] <= prev["ema50"]) and (latest["ema20"] > latest["ema50"])
    ema_cross_sell = (prev["ema20"] >= prev["ema50"]) and (latest["ema20"] < latest["ema50"])

    vol_spike = latest["vol_rel"] > 2.0  # volume at least 2x MA
    macd_momentum = latest["macd_hist"] > 0.0
    rsi_ok_buy = latest["rsi14"] < 75
    rsi_ok_sell = latest["rsi14"] > 25

    if ema_cross_buy and vol_spike and macd_momentum and rsi_ok_buy:
        return "BUY"
    if ema_cross_sell and vol_spike and (latest["macd_hist"] < 0) and rsi_ok_sell:
        return "SELL"
    return None

def mid_condition(df: pd.DataFrame) -> Optional[str]:
    """
    Mid: slower signals using EMA50/200 alignment + ADX trend strength
    """
    latest = df.iloc[-1]
    # trend strength
    strong_trend = latest["adx14"] > 20
    bull = (latest["ema50"] > latest["ema200"])
    bear = (latest["ema50"] < latest["ema200"])

    if bull and strong_trend and latest["rsi14"] < 85:
        # require MACD positive
        if latest["macd"] > latest["macd_signal"]:
            return "BUY"
    if bear and strong_trend and latest["rsi14"] > 15:
        if latest["macd"] < latest["macd_signal"]:
            return "SELL"
    return None

def trend_condition(df: pd.DataFrame) -> Optional[str]:
    """
    Trend: take only strong alignment and sustained momentum
    """
    latest = df.iloc[-1]
    # strong alignment across EMAs
    if latest["ema20"] > latest["ema50"] > latest["ema200"] and latest["adx14"] > 25 and latest["macd_hist"] > 0:
        return "BUY"
    if latest["ema20"] < latest["ema50"] < latest["ema200"] and latest["adx14"] > 25 and latest["macd_hist"] < 0:
        return "SELL"
    return None

# -----------------------------
# TP/SL & LEVERAGE SUGGESTION
# -----------------------------
def calc_tp_sl(entry: float, direction: str, atr: Optional[float] = None) -> Tuple[float, float]:
    """
    Simple TP/SL:
    if no ATR -> use percentage rules
    direction: 'BUY' or 'SELL'
    returns (tp, sl)
    """
    if atr is None or math.isnan(atr):
        pct_sl = 0.004  # 0.4% default SL
        pct_tp = 0.003  # 0.3 TP default (scalp)
    else:
        # SL = 1.5 * ATR, TP = 1.0 * ATR (scalp-like)
        sl_price_move = 1.5 * atr
        tp_price_move = 1.0 * atr
        if direction == "BUY":
            sl = entry - sl_price_move
            tp = entry + tp_price_move
            return round(tp, 6), round(sl, 6)
        else:
            sl = entry + sl_price_move
            tp = entry - tp_price_move
            return round(tp, 6), round(sl, 6)

    if direction == "BUY":
        sl = entry * (1 - pct_sl)
        tp = entry * (1 + pct_tp)
    else:
        sl = entry * (1 + pct_sl)
        tp = entry * (1 - pct_tp)
    return round(tp, 6), round(sl, 6)

def suggest_leverage(regime: str, mode: str) -> int:
    """
    Simple leverage suggestion:
      - Quick: higher (50) if bull and low volatility
      - Mid: moderate
      - Trend: lower (10-20) safer
    """
    mode = mode.upper()
    if mode == "QUICK":
        return 50 if regime == "bull" else 25
    if mode == "MID":
        return 20 if regime == "bull" else 10
    if mode == "TREND":
        return 15 if regime == "bull" else 8
    return 10

# -----------------------------
# DANGER ZONE (price extremes)
# -----------------------------
def danger_zone(df: pd.DataFrame) -> Dict[str, float]:
    """
    Return near-high/low range for danger zone
    for example: last candle high/low +/- small percent
    """
    latest = df.iloc[-1]
    p = latest["close"]
    hi = latest["high"]
    lo = latest["low"]
    # danger zone threshold - if price within 0.5% of 24h high/low
    return {
        "last_close": float(p),
        "last_high": float(hi),
        "last_low": float(lo),
        "price_to_high_pct": round((hi - p) / p * 100, 3),
        "price_to_low_pct": round((p - lo) / p * 100, 3),
    }

# -----------------------------
# TELEGRAM SENDER (simple)
# -----------------------------
def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram token/chat_id not set. Skipping telegram send.")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=8)
        if r.status_code == 200:
            return True
        logger.warning(f"Telegram send failed: {r.status_code} {r.text}")
        return False
    except Exception as e:
        logger.warning(f"Telegram exception: {e}")
        return False

# -----------------------------
# BUILD SIGNAL PAYLOAD
# -----------------------------
def build_signal_payload(symbol: str, mode: str, direction: str, entry: float, tp: float, sl: float, leverage: int, reason: str, accuracy: float, dz: dict) -> str:
    """
    Build a clean telegram message (HTML)
    """
    tztime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        f"🔥 <b>{direction} SIGNAL — {mode}</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Entry:</b> {entry}\n\n"
        f"🎯 <b>TP:</b> {tp}\n"
        f"🛑 <b>SL:</b> {sl}\n"
        f"⚡ <b>Leverage:</b> {leverage}x\n\n"
        f"📌 <b>Reason:</b> {reason}\n"
        f"🟧 <b>Danger Zone:</b> close={dz['last_close']} hi={dz['last_high']} lo={dz['last_low']} (to_high={dz['price_to_high_pct']}%, to_low={dz['price_to_low_pct']}%)\n\n"
        f"📊 <b>Accuracy (est):</b> {accuracy}%\n"
        f"⏱ <b>Time:</b> {tztime}\n"
    )
    return msg

# -----------------------------
# SIGNAL ENGINE (one function to call)
# -----------------------------
def analyze_and_prepare(df: pd.DataFrame, symbol: str, mode: str) -> Optional[Dict]:
    """
    mode: QUICK/MID/TREND
    Returns dict with keys: direction, entry, tp, sl, leverage, reason, accuracy, danger_zone
    """
    df = add_indicators(df)
    if is_on_cooldown(symbol, mode):
        logger.info(f"Cooldown blocks {symbol} for mode {mode}")
        return None

    regime = detect_regime(df)
    reason = ""
    direction = None

    if mode.upper() == "QUICK":
        direction = quick_condition(df)
        reason = "EMA20/50 cross + vol spike + MACD momentum"
        acc = 65  # placeholder
    elif mode.upper() == "MID":
        direction = mid_condition(df)
        reason = "EMA50/200 alignment + ADX trend"
        acc = 72
    elif mode.upper() == "TREND":
        direction = trend_condition(df)
        reason = "EMA all aligned + ADX strong"
        acc = 80
    else:
        logger.warning("Unknown mode")
        return None

    if not direction:
        return None

    entry = float(df.iloc[-1]["close"])
    # ATR approx: use TR EMA 14
    tr = true_range(df).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
    tp, sl = calc_tp_sl(entry, direction, atr=tr)
    leverage = suggest_leverage(regime, mode)
    dz = danger_zone(df)

    payload = {
        "symbol": symbol,
        "mode": mode,
        "direction": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "leverage": leverage,
        "reason": reason,
        "accuracy": acc,
        "danger_zone": dz,
    }

    # set cooldown after preparing
    set_cooldown(symbol, mode)

    return payload

# -----------------------------
# MODULE QUICK TEST
# -----------------------------
if __name__ == "__main__":
    # quick local test: load sample CSV and sample OHLC from file if available
    test_df = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.utcnow(), periods=120, freq="T"),
        "open": pd.Series(range(120)).astype(float) + 100,
        "high": pd.Series(range(120)).astype(float) + 101,
        "low": pd.Series(range(120)).astype(float) + 99,
        "close": pd.Series(range(120)).astype(float) + 100,
        "volume": pd.Series([100 + (i % 5) * 50 for i in range(120)])
    })
    test_df = test_df.set_index("time").reset_index()
    test_df = add_indicators(test_df)
    payload = analyze_and_prepare(test_df, "BTC/USDT", "QUICK")
    print("TEST PAYLOAD:", payload)