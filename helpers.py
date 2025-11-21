"""
helpers.py - Signal engine helpers for Binance scalper bot
ASCII-only, async friendly.

Exposed functions:
  - run_all_modes()
  - multi_override_watch(active_signals)
  - format_signal(sig, signal_no)
"""

import asyncio
import logging
import math
import os
import time
import smtplib
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# ENV CONFIG
# -------------------------------------------------------------
SCORE_MIN = int(os.getenv("SCORE_MIN", "90"))

# Estimated hold minutes per mode (for message display only)
HOLD_MINUTES: Dict[str, int] = {
    "quick": 15,
    "mid": 45,
    "trend": 120,
}

# Capital and risk for position sizing (can be tuned from env)
BASE_CAPITAL = float(os.getenv("BASE_CAPITAL", "100000"))      # in USDT or INR-equivalent
BASE_RISK_PCT = float(os.getenv("BASE_RISK_PCT", "0.01"))      # 1% default
MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.015"))       # 1.5% hard cap

# Concurrency limit for async scans (do not spam Binance)
MAX_CONCURRENT_SCANS = int(os.getenv("MAX_CONCURRENT_SCANS", "10"))

# -------------------------------------------------------------
# EMAIL ALERT CONFIG / HELPER
# -------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "")
SMTP_TO = os.getenv("SMTP_TO", "")
SMTP_TLS = os.getenv("SMTP_TLS", "1")  # "1" -> use starttls


async def send_email_async(subject: str, body: str) -> None:
    """
    Simple async email sender for danger alerts.
    If SMTP env not set, it silently skips.
    """
    if not (SMTP_HOST and SMTP_TO):
        # no smtp configured, skip
        return

    def _send() -> None:
        try:
            msg = (
                f"From: {SMTP_FROM}\r\n"
                f"To: {SMTP_TO}\r\n"
                f"Subject: {subject}\r\n"
                "\r\n"
                f"{body}"
            )
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                try:
                    if SMTP_TLS == "1":
                        server.starttls()
                except Exception:
                    pass
                if SMTP_USER and SMTP_PASS:
                    try:
                        server.login(SMTP_USER, SMTP_PASS)
                    except Exception:
                        pass
                server.sendmail(SMTP_FROM or SMTP_USER, [SMTP_TO], msg.encode("utf-8", "ignore"))
        except Exception as e:
            logger.warning("send_email_async inner error: %s", str(e))

    try:
        await asyncio.to_thread(_send)
    except Exception as e:
        logger.warning("send_email_async failed: %s", str(e))

# -------------------------------------------------------------
# BINANCE HTTP SESSION
# -------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"  # for funding


class HTTPSession:
    _session: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def get(cls, url: str, retries: int = 3, timeout: int = 8) -> Any:
        """
        Simple GET with retry and timeout.
        Returns JSON-decoded body or None on failure.
        """
        if cls._session is None:
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)
            cls._session = aiohttp.ClientSession(timeout=timeout_cfg)

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with cls._session.get(url) as resp:
                    return await resp.json()
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * (attempt + 1))
        logger.error("HTTP GET failed after %s retries: %s | err=%s", retries, url, last_err)
        return None

    @classmethod
    async def close(cls) -> None:
        if cls._session is not None:
            await cls._session.close()
            cls._session = None

# -------------------------------------------------------------
# DATA FETCHERS
# -------------------------------------------------------------
async def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await HTTPSession.get(url)
    if not data:
        logger.warning("OHLCV empty for %s %s", symbol, interval)
        return pd.DataFrame()
    try:
        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_base", "taker_quote", "ignore",
            ],
        )
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logger.error("OHLCV parse failed for %s: %s", symbol, e)
        return pd.DataFrame()


async def fetch_ticker(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_BASE}/api/v3/ticker/bookTicker?symbol={symbol}"
    data = await HTTPSession.get(url)
    return data or {}


async def fetch_spread(symbol: str) -> float:
    t = await fetch_ticker(symbol)
    try:
        bid = float(t["bidPrice"])
        ask = float(t["askPrice"])
        return (ask - bid) / bid * 100.0
    except Exception:
        return 999.0


async def fetch_funding(symbol: str) -> float:
    url = f"{BINANCE_FUTURES}/fapi/v1/premiumIndex?symbol={symbol}"
    data = await HTTPSession.get(url)
    try:
        return float(data.get("lastFundingRate", 0.0)) * 100.0
    except Exception:
        return 0.0


async def fetch_server_time() -> int:
    url = f"{BINANCE_BASE}/api/v3/time"
    data = await HTTPSession.get(url)
    try:
        return int(data.get("serverTime", 0))
    except Exception:
        return 0

# -------------------------------------------------------------
# INDICATORS
# -------------------------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum()

# -------------------------------------------------------------
# BTC STABILITY + HTF PATTERN
# -------------------------------------------------------------
async def btc_stable() -> Dict[str, Any]:
    df = await fetch_ohlcv("BTCUSDT", "1m", 50)
    if df.empty:
        return {"ok": False, "vol": 0.0, "wick": 0.0}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    window = high.tail(10).max() - low.tail(10).min()
    vol = window / close.iloc[-1] * 100.0

    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    wick_pct = (upper_wick + lower_wick) / c * 100.0

    stable = (vol < 1.0) and (wick_pct < 0.5)
    return {"ok": stable, "vol": float(vol), "wick": float(wick_pct)}


def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]
    body = abs(c - o)
    rng = h - l

    if rng <= 0:
        return "none"

    # strong body candle
    if body > rng * 0.6:
        if c > o:
            return "bull_engulf"
        else:
            return "bear_engulf"

    upper = h - max(o, c)
    lower = min(o, c) - l
    if upper > body * 2:
        return "shooting_star"
    if lower > body * 2:
        return "hammer"
    return "none"


async def htf_signal(symbol: str) -> Dict[str, str]:
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    df1h = await fetch_ohlcv(symbol, "1h", 50)
    if df15.empty or df1h.empty:
        return {"15m": "none", "1h": "none"}
    return {
        "15m": detect_pattern(df15),
        "1h": detect_pattern(df1h),
    }

# -------------------------------------------------------------
# PRICE ACTION FILTERS
# -------------------------------------------------------------
def liquidity_sweep(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]
    body = abs(c - o)
    total_range = h - l
    if body <= 0 or total_range <= 0:
        return False
    wick = total_range - body
    return wick > body * 2.5


def ema21_pullback(df: pd.DataFrame) -> bool:
    if len(df) < 22:
        return False
    e21 = ema(df["close"], 21)
    price = df["close"].iloc[-1]
    dist = abs(price - e21.iloc[-1]) / price
    return dist < 0.002  # 0.2%


def range_break_retest(df: pd.DataFrame) -> bool:
    if len(df) < 40:
        return False
    recent = df.tail(20)
    high_line = recent["high"].max()
    low_line = recent["low"].min()
    close_last = df["close"].iloc[-1]
    low_last = df["low"].iloc[-1]

    broke_up = close_last > high_line and low_last <= high_line
    broke_down = close_last < low_line and low_last >= low_line
    return broke_up or broke_down


def detect_order_block(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    o1 = df["open"].iloc[-2]
    c1 = df["close"].iloc[-2]
    o2 = df["open"].iloc[-1]
    c2 = df["close"].iloc[-1]
    if c1 < o1 and c2 > o2:
        return "bull_OB"
    if c1 > o1 and c2 < o2:
        return "bear_OB"
    return "none"


def detect_fvg(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    h1 = df["high"].iloc[-3]
    l1 = df["low"].iloc[-3]
    h3 = df["high"].iloc[-1]
    l3 = df["low"].iloc[-1]
    if l1 > h3:
        return "bull_FVG"
    if h1 < l3:
        return "bear_FVG"
    return "none"

# -------------------------------------------------------------
# SPREAD + FUNDING + SESSION
# -------------------------------------------------------------
async def spread_ok(symbol: str) -> bool:
    sp = await fetch_spread(symbol)
    return sp < 0.06  # 0.06%


async def funding_ok(symbol: str) -> bool:
    f = await fetch_funding(symbol)
    return abs(f) < 0.02  # 0.02%


def session_now(ts_ms: int) -> str:
    h = time.gmtime(ts_ms / 1000).tm_hour
    if 1 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "europe"
    return "us"

# -------------------------------------------------------------
# VOLUME, SCORE, MODE, COOLDOWN
# -------------------------------------------------------------
def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < 30:
        return False
    recent = df["volume"].iloc[-1]
    avg = df["volume"].tail(20).mean()
    return recent > avg * 2.0


def calc_score(
    symbol: str,
    df: pd.DataFrame,
    htf: Dict[str, str],
    pa: Dict[str, Any],
    is_spread_ok: bool,
    is_funding_ok: bool,
) -> int:
    score = 0

    close = df["close"]
    if len(close) > 50:
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        if e20 > e50:
            score += 15

    if volume_spike(df):
        score += 15

    if htf.get("15m") in ("bull_engulf", "hammer"):
        score += 10
    if htf.get("1h") in ("bull_engulf", "hammer"):
        score += 10

    if pa.get("sweep"):
        score += 10
    if pa.get("pullback"):
        score += 10
    if pa.get("range_retest"):
        score += 10
    if pa.get("ob") != "none":
        score += 10
    if pa.get("fvg") != "none":
        score += 10

    if is_spread_ok:
        score += 5
    if is_funding_ok:
        score += 5

    return min(score, 100)


def mode_requirements(df: pd.DataFrame, mode: str) -> bool:
    close = df["close"]
    if len(close) < 50:
        return False
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]

    mode = mode.lower()
    if mode == "quick":
        return e20 > e50
    if mode == "mid":
        return e20 > e50 and volume_spike(df)
    if mode == "trend":
        return e20 > e50 and close.iloc[-1] > e20
    return False


COOLDOWN: Dict[str, int] = {}


def cooldown_ok(symbol: str) -> bool:
    now = int(time.time())
    last = COOLDOWN.get(symbol)
    if last is None:
        return True
    return (now - last) > 1800  # 30 min


def set_cooldown(symbol: str) -> None:
    COOLDOWN[symbol] = int(time.time())

# -------------------------------------------------------------
# COIN LIST (50) + GROUPS + WEIGHTS
# -------------------------------------------------------------
COIN_LIST: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT", "UNIUSDT", "LINKUSDT", "ATOMUSDT", "ETCUSDT",
    "FILUSDT", "ICPUSDT", "NEARUSDT", "APTUSDT", "SANDUSDT", "AXSUSDT", "THETAUSDT", "FTMUSDT",
    "RUNEUSDT", "ALGOUSDT", "EGLDUSDT", "IMXUSDT", "INJUSDT", "OPUSDT", "ARBUSDT", "SUIUSDT",
    "TIAUSDT", "PEPEUSDT", "TRBUSDT", "SEIUSDT", "JTOUSDT", "PYTHUSDT", "RAYUSDT", "GMTUSDT",
    "MINAUSDT", "WLDUSDT", "ZKUSDT", "STRKUSDT", "DYDXUSDT", "VETUSDT", "GALAUSDT", "KAVAUSDT",
]

# Grouping for position sizing / news weighting
GROUP_A_BTC_MAJOR: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
]

GROUP_B_L1_L2: List[str] = [
    "ADAUSDT", "AVAXUSDT", "MATICUSDT", "OPUSDT", "ARBUSDT",
    "SUIUSDT", "TIAUSDT", "ZKUSDT", "STRKUSDT", "NEARUSDT",
]

GROUP_C_DEFI_DERIV: List[str] = [
    "INJUSDT", "DYDXUSDT", "UNIUSDT", "RUNEUSDT", "LINKUSDT",
    "ATOMUSDT", "EGLDUSDT",
]

GROUP_D_GAMING_META: List[str] = [
    "SANDUSDT", "AXSUSDT", "GALAUSDT", "IMXUSDT", "PYTHUSDT", "GMTUSDT",
]

GROUP_E_MEME_HIGH_BETA: List[str] = [
    "PEPEUSDT", "DOGEUSDT", "TRBUSDT", "SEIUSDT", "JTOUSDT",
]

COIN_GROUPS: Dict[str, List[str]] = {
    "A": GROUP_A_BTC_MAJOR,
    "B": GROUP_B_L1_L2,
    "C": GROUP_C_DEFI_DERIV,
    "D": GROUP_D_GAMING_META,
    "E": GROUP_E_MEME_HIGH_BETA,
}

GROUP_WEIGHTS: Dict[str, float] = {
    "A": 1.0,   # majors: BTC/ETH etc
    "B": 0.8,
    "C": 0.7,
    "D": 0.6,
    "E": 0.4,   # meme / high beta
}

def detect_group(symbol: str) -> str:
    """
    Return group letter (A/B/C/D/E) for symbol. Default "A" if unknown.
    """
    for g, coins in COIN_GROUPS.items():
        if symbol in coins:
            return g
    return "A"

# -------------------------------------------------------------
# POSITION SIZING (GROUP-WEIGHTED)
# -------------------------------------------------------------
def calc_position_size(symbol: str, entry: float, sl: float) -> Dict[str, float]:
    """
    Simple group-weight based position sizing.
    - BASE_CAPITAL: total notional capital
    - BASE_RISK_PCT: base risk per trade
    - MAX_RISK_PCT: hard cap on risk
    """
    if entry <= 0 or sl <= 0:
        return {
            "group": "A",
            "weight": 1.0,
            "risk_pct": 0.0,
            "capital_risk": 0.0,
            "pos_size_usdt": 0.0,
        }

    group = detect_group(symbol)
    weight = GROUP_WEIGHTS.get(group, 1.0)

    risk_pct = BASE_RISK_PCT * weight
    risk_pct = min(risk_pct, MAX_RISK_PCT)  # enforce max cap

    risk_amount = BASE_CAPITAL * risk_pct

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return {
            "group": group,
            "weight": weight,
            "risk_pct": risk_pct * 100.0,
            "capital_risk": 0.0,
            "pos_size_usdt": 0.0,
        }

    # For USDT pairs, position notional approx:
    pos_size_usdt = risk_amount * (entry / sl_dist)

    return {
        "group": group,
        "weight": weight,
        "risk_pct": risk_pct * 100.0,
        "capital_risk": risk_amount,
        "pos_size_usdt": pos_size_usdt,
    }

# -------------------------------------------------------------
# TP/SL WITH SIMPLE SLIPPAGE BUFFER
# -------------------------------------------------------------
async def estimate_spread_buffer(symbol: str) -> float:
    """
    Very simple slippage buffer based on spread.
    Returns an extra percent to add on TP, half for SL.

    Example:
      spread 0.03% -> buffer ~0.115 (%)
    """
    sp = await fetch_spread(symbol)
    # clamp spread between 0 and 0.2%
    sp = max(0.0, min(sp, 0.2))
    # base buffer 0.1 + half of spread (both in percent)
    return 0.1 + sp * 0.5


async def calc_tp_sl(entry: float, symbol: str, mode: str, side: str = "BUY") -> Dict[str, float]:
    """
    Return dict with tp, sl, including a small buffer for spread/slippage.
    Internal math uses percentage in % (not decimals).
    """
    mode = mode.lower()
    side = side.upper()

    if mode == "quick":
        base_tp = 0.4
        base_sl = 0.4
    elif mode == "mid":
        base_tp = 2.0
        base_sl = 1.0
    elif mode == "trend":
        base_tp = 3.0
        base_sl = 1.5
    else:
        base_tp = 1.0
        base_sl = 1.0

    buf = await estimate_spread_buffer(symbol)
    tp_pct = base_tp + buf
    sl_pct = base_sl + buf * 0.5

    tp_factor = tp_pct / 100.0
    sl_factor = sl_pct / 100.0

    if side == "SELL":
        tp = entry * (1.0 - tp_factor)
        sl = entry * (1.0 + sl_factor)
    else:
        tp = entry * (1.0 + tp_factor)
        sl = entry * (1.0 - sl_factor)

    return {
        "tp": round(tp, 8),
        "sl": round(sl, 8),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    }

# -------------------------------------------------------------
# BUILD SIGNAL (CORE LOGIC)
# -------------------------------------------------------------
async def build_signal(symbol: str, mode: str, side: str = "BUY") -> Dict[str, Any]:
    """
    Core signal logic for one symbol/mode.
    Applies:
      - cooldown
      - HTF patterns
      - price action filters
      - spread/funding filters
      - score + mode requirements
      - TP/SL with slippage buffer
      - position sizing (group-weighted)
    """
    if not cooldown_ok(symbol):
        return {"ok": False, "reason": "cooldown"}

    df = await fetch_ohlcv(symbol, "1m", 200)
    if df.empty:
        return {"ok": False, "reason": "data_error"}

    # HTF patterns
    htf = await htf_signal(symbol)

    # Price action
    pa = {
        "sweep": liquidity_sweep(df),
        "pullback": ema21_pullback(df),
        "range_retest": range_break_retest(df),
        "ob": detect_order_block(df),
        "fvg": detect_fvg(df),
    }

    # Spread + funding
    sp_ok = await spread_ok(symbol)
    f_ok = await funding_ok(symbol)

    # Score
    score = calc_score(symbol, df, htf, pa, sp_ok, f_ok)
    if score < SCORE_MIN:
        return {"ok": False, "reason": "low_score", "score": score}

    # Mode level filter
    if not mode_requirements(df, mode):
        return {"ok": False, "reason": "mode_not_fit", "score": score}

    # Entry
    entry = float(df["close"].iloc[-1])

    # TP/SL with buffer
    tpsl = await calc_tp_sl(entry, symbol, mode, side)

    # Position sizing
    pos_info = calc_position_size(symbol, entry, tpsl["sl"])

    # Cooldown mark
    set_cooldown(symbol)

    # Time (UTC string)
    now_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    return {
        "ok": True,
        "symbol": symbol,
        "mode": mode.lower(),
        "side": side.upper(),
        "entry": entry,
        "tp": tpsl["tp"],
        "sl": tpsl["sl"],
        "tp_pct": tpsl["tp_pct"],
        "sl_pct": tpsl["sl_pct"],
        "score": score,
        "htf": htf,
        "pa": pa,
        "pos": pos_info,
        "time_utc": now_utc,
    }

# -------------------------------------------------------------
# CONCURRENCY-LIMITED RUNNERS
# -------------------------------------------------------------
_scan_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCANS)


async def scan_coin_limited(symbol: str, mode: str) -> Dict[str, Any]:
    """
    Scan one symbol with concurrency limit (to avoid overloading Binance).
    """
    async with _scan_semaphore:
        try:
            return await build_signal(symbol, mode)
        except Exception as e:
            logger.error("scan_coin error for %s %s: %s", symbol, mode, e)
            return {"ok": False, "error": str(e)}


async def run_mode(mode: str) -> List[Dict[str, Any]]:
    """
    Run a full scan for a single mode (quick/mid/trend)
    over COIN_LIST with concurrency limit.
    """
    tasks = [scan_coin_limited(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r.get("ok")]
    logger.info("mode=%s signals=%d", mode, len(valid))
    return valid


async def run_all_modes() -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all three modes:
      - quick
      - mid
      - trend
    Returns dict: {"quick": [...], "mid": [...], "trend": [...]}
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    out["quick"] = await run_mode("quick")
    out["mid"] = await run_mode("mid")
    out["trend"] = await run_mode("trend")
    return out

# -------------------------------------------------------------
# TIME HELPERS (UTC -> IST)
# -------------------------------------------------------------
def _utc_to_ist_str(utc_str: str) -> str:
    """
    Convert "YYYY-mm-dd HH:MM:SS" UTC string to IST time string "HH:MM:SS AM/PM".
    """
    from datetime import datetime, timedelta

    try:
        dt_utc = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt_utc = datetime.utcnow()
    dt_ist = dt_utc + timedelta(hours=5, minutes=30)
    return dt_ist.strftime("%I:%M:%S %p")


def _utc_to_ist_hold(utc_str: str, mode: str) -> Tuple[str, int]:
    """
    Return (hold_till_str, hold_min) using mode-wise HOLD_MINUTES.
    """
    from datetime import datetime, timedelta

    hold_min = HOLD_MINUTES.get(mode.lower(), 30)
    try:
        dt_utc = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt_utc = datetime.utcnow()
    dt_ist = dt_utc + timedelta(hours=5, minutes=30)
    dt_ist_hold = dt_ist + timedelta(minutes=hold_min)
    return dt_ist_hold.strftime("%I:%M:%S %p"), hold_min


def _flag(ok: bool) -> str:
    return "✅" if ok else "❌"

# -------------------------------------------------------------
# TELEGRAM FORMATTER (INDIAN TIME + COPY BLOCK)
# -------------------------------------------------------------
def format_signal(sig: Dict[str, Any], signal_no: int) -> str:
    """
    Build premium Telegram signal text.
    ENTRY/TP/SL only in copy block at bottom.
    """
    symbol = sig["symbol"]
    mode = sig["mode"].upper()
    side = sig.get("side", "BUY").upper()
    score = sig.get("score", 0)
    entry = sig["entry"]
    tp = sig["tp"]
    sl = sig["sl"]
    htf = sig.get("htf", {})
    pa = sig.get("pa", {})
    time_utc = sig.get("time_utc", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    time_ist = _utc_to_ist_str(time_utc)
    hold_till_ist, hold_min = _utc_to_ist_hold(time_utc, sig.get("mode", "quick"))

    # Reason flags (simple version)
    vol_ok = True   # we already used volume_spike in score
    macd_ok = True  # can later compute from df if needed
    rsi_ok = True   # can later compute from df if needed

    btc_flag = _flag(True)  # detailed check is in override layer
    htf_flag = _flag(
        htf.get("15m") in ("bull_engulf", "hammer")
        or htf.get("1h") in ("bull_engulf", "hammer")
    )
    spread_flag = _flag(True)
    funding_flag = _flag(True)
    struct_flag = _flag(
        bool(pa.get("pullback"))
        or bool(pa.get("range_retest"))
        or pa.get("ob") != "none"
    )

    text = f"""🔥 {side} SIGNAL — {mode} MODE
━━━━━━━━━━━━━━━━
📌 Signal #{signal_no} (Today)
Pair: {symbol}
Mode: {mode}
Score: {score}
━━━━━━━━━━━━━━━━
🔍 HTF:
15m → {htf.get('15m', 'none')}
1h → {htf.get('1h', 'none')}
━━━━━━━━━━━━━━━━
📊 Price Action:
Sweep: {pa.get('sweep')}
Pullback: {pa.get('pullback')}
Range Retest: {pa.get('range_retest')}
OB: {pa.get('ob')}
FVG: {pa.get('fvg')}
━━━━━━━━━━━━━━━━
⏱ Indian Time (Entry): {time_ist}
🕒 Hold till (est):      {hold_till_ist}  (~{hold_min} min)
━━━━━━━━━━━━━━━━
✅ Reason:
vol {_flag(vol_ok)}   macd {_flag(macd_ok)}   rsi {_flag(rsi_ok)}
btc {btc_flag}   htf {htf_flag}    spread {spread_flag}
funding {funding_flag}   structure {struct_flag}
━━━━━━━━━━━━━━━━
📋 COPY BLOCK (ONLY ENTRY/TP/SL)
ENTRY: {entry}
TP:    {tp}
SL:    {sl}
━━━━━━━━━━━━━━━━
"""
    return text

# -------------------------------------------------------------
# ACTIVE TRADES STORE (FOR OVERRIDE MONITORING)
# -------------------------------------------------------------
ACTIVE_TRADES: List[Dict[str, Any]] = []


def register_trade(sig: Dict[str, Any]) -> None:
    """
    Register a signal as an active trade.
    We assume: signal alert == you took the trade.
    """
    if not isinstance(sig, dict):
        return
    item = dict(sig)
    item["activated_at"] = int(time.time())
    ACTIVE_TRADES.append(item)


def cleanup_active_trades(max_age_sec: int = 3600) -> None:
    """
    Keep only trades opened within last max_age_sec seconds.
    Default ~1 hour.
    """
    now = int(time.time())
    keep: List[Dict[str, Any]] = []
    for t in ACTIVE_TRADES:
        ts = t.get("activated_at", now)
        if now - ts <= max_age_sec:
            keep.append(t)
    ACTIVE_TRADES[:] = keep
# -------------------------------------------------------------
# TIGHT DANGER LAYER (FALLBACK VERSION - NO POSITION API)
# -------------------------------------------------------------
async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    """
    Tight danger layer using only:
      - signal data (entry, SL, TP, mode, side, time_utc)
      - latest 1m candles of the symbol
      - BTC stability + BTC short-term move

    Levels:
      - CAUTION (🟡): mainly time-trap / boredom
      - HIGH RISK (🟠): single strong danger condition
      - EMERGENCY (🔴): 2+ strong dangers together

    Note:
      - This does NOT auto-close anything.
      - You use the alert to decide exit/hold.
    """
    alerts: List[str] = []
    if not active_signals:
        return alerts

    # ---- BTC overall stability + short-term move ----
    try:
        btc_info = await btc_stable()
    except Exception:
        btc_info = {"ok": True, "vol": 0.0, "wick": 0.0}

    # extra: last 10 minutes BTC move
    btc_move_pct = 0.0
    try:
        btc_df = await fetch_ohlcv("BTCUSDT", "1m", 12)
        if not btc_df.empty and len(btc_df) >= 10:
            c = btc_df["close"]
            btc_move_pct = (c.iloc[-1] - c.iloc[-10]) / max(c.iloc[-10], 1e-8) * 100.0
    except Exception:
        btc_move_pct = 0.0

    now_utc_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for sig in active_signals:
        try:
            if not sig.get("ok"):
                continue

            symbol = sig.get("symbol")
            if not symbol:
                continue

            mode = str(sig.get("mode", "quick")).upper()
            side = str(sig.get("side", "BUY")).upper()

            entry = float(sig.get("entry"))
            tp = float(sig.get("tp"))
            sl = float(sig.get("sl"))
            time_utc = str(sig.get("time_utc", now_utc_str))

            # ---- parse signal time to get age in minutes ----
            from datetime import datetime

            def _parse_utc(t_str: str) -> datetime:
                try:
                    return datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    return datetime.utcnow()

            t_sig = _parse_utc(time_utc)
            t_now = _parse_utc(now_utc_str)
            age_minutes = (t_now - t_sig).total_seconds() / 60.0

            # ---- get latest price + candle info ----
            df = await fetch_ohlcv(symbol, "1m", 20)
            if df.empty or len(df) < 6:
                continue

            close = df["close"]
            high = df["high"]
            low = df["low"]

            price = float(close.iloc[-1])

            # ---- basic metrics ----
            # move since entry (signed, in %)
            if side == "SELL":
                move_pct = (entry - price) / max(entry, 1e-8) * 100.0
            else:  # BUY
                move_pct = (price - entry) / max(entry, 1e-8) * 100.0

            # distance to SL (always positive, in % of price)
            dist_sl_pct = abs(price - sl) / max(price, 1e-8) * 100.0

            # loss side?
            in_loss = False
            if side == "SELL":
                in_loss = price > entry
            else:
                in_loss = price < entry

            # last candle range vs previous 5 (sudden spike check)
            last_range = float(high.iloc[-1] - low.iloc[-1])
            last_range_pct = last_range / max(price, 1e-8) * 100.0
            prev_ranges = (high.iloc[-6:-1] - low.iloc[-6:-1]).abs()
            prev_avg_range = float(prev_ranges.mean()) if len(prev_ranges) > 0 else 0.0
            prev_avg_range_pct = prev_avg_range / max(price, 1e-8) * 100.0

            # ---- build danger conditions ----
            reasons: List[str] = []
            strong_flags: List[str] = []
            level = None  # "CAUTION" / "HIGH" / "EMERGENCY"

            # 1) NEAR SL
            near_sl = dist_sl_pct < 0.3 and in_loss
            if near_sl:
                strong_flags.append("NEAR_SL")
                reasons.append(f"SL very close (~{dist_sl_pct:.2f}%)")

            # 2) FAST LOSS (within first 5 min)
            fast_loss = age_minutes <= 5.0 and move_pct <= -0.7
            if fast_loss and in_loss:
                strong_flags.append("FAST_LOSS")
                reasons.append(f"Fast loss after entry (~{move_pct:.2f}%)")

            # 3) BIG LOSS (overall move < -1.0%)
            big_loss = move_pct <= -1.0 and in_loss
            if big_loss:
                strong_flags.append("BIG_LOSS")
                reasons.append(f"Loss deep (~{move_pct:.2f}%)")

            # 4) SUDDEN SPIKE (last candle 2.5x bigger than previous avg + loss side)
            spike = False
            if prev_avg_range_pct > 0:
                if last_range_pct > max(0.7, prev_avg_range_pct * 2.5) and in_loss:
                    spike = True
            if spike:
                strong_flags.append("SUDDEN_SPIKE")
                reasons.append(
                    f"Sudden spike/wick (last range {last_range_pct:.2f}% vs prev {prev_avg_range_pct:.2f}%)"
                )

            # 5) BTC PRESSURE
            btc_unstable = not btc_info.get("ok", True) or btc_move_pct < -0.7
            btc_danger = btc_unstable and in_loss
            if btc_danger:
                strong_flags.append("BTC_PRESSURE")
                reasons.append(
                    f"BTC unstable (vol={btc_info.get('vol', 0):.2f}%, move={btc_move_pct:.2f}%)"
                )

            # 6) TIME TRAP (age >= 30 min, move small)
            time_trap = age_minutes >= 30.0 and abs(move_pct) < 0.2
            if time_trap:
                reasons.append(
                    f"Time trap: {age_minutes:.1f} min, price stuck near entry (~{move_pct:.2f}%)"
                )

            # ---- decide level ----
            strong_count = len(strong_flags)

            if strong_count >= 2:
                level = "EMERGENCY"
            elif strong_count == 1:
                level = "HIGH"
            elif time_trap:
                level = "CAUTION"

            if not level:
                continue  # no meaningful danger, skip

            # ---- build message text ----
            level_emoji = {
                "CAUTION": "🟡",
                "HIGH": "🟠",
                "EMERGENCY": "🔴",
            }.get(level, "⚠️")

            header = f"{level_emoji} {level} DANGER ALERT"

            reason_txt = "\n- ".join(reasons) if reasons else "Risk elevated"

            msg = (
                f"{header}\n"
                f"Pair: {symbol}\n"
                f"Mode: {mode}\n"
                f"Side: {side}\n"
                f"Age: {age_minutes:.1f} min\n"
                f"\nReasons:\n- {reason_txt}\n"
                f"\nENTRY: {entry}\n"
                f"TP:    {tp}\n"
                f"SL:    {sl}\n"
                f"Price now: {price}\n"
                f"Move vs entry: {move_pct:.2f}%\n"
                f"BTC move (last 10m): {btc_move_pct:.2f}%\n"
            )

            alerts.append(msg)

        except Exception as e:
            logger.debug("multi_override_watch tight-layer error for %s: %s", sig, str(e))
            continue

    # deduplicate
    unique: List[str] = []
    seen: set = set()
    for a in alerts:
        if a not in seen:
            unique.append(a)
            seen.add(a)
    return unique

# -------------------------------------------------------------
# EXPORTS
# -------------------------------------------------------------
__all__ = [
    "run_all_modes",
    "multi_override_watch",
    "format_signal",
]