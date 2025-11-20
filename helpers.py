# helpers.py - unified version with Telegram + override watcher

import os
import asyncio
import aiohttp
import logging
import time
import math
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# TELEGRAM CONFIG
# -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

async def send_telegram(text: str) -> None:
    """
    Simple Telegram sender.
    Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM CONFIG MISSING – skipping send")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                body = await resp.text()
                if resp.status != 200:
                    logger.error("TELEGRAM SEND FAILED %s – %s", resp.status, body)
                else:
                    logger.info("TELEGRAM: send OK")
    except Exception as e:
        logger.error("TELEGRAM SEND EXCEPTION: %s", e)

# -------------------------------------------------------------
# BINANCE API URL
# -------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"

# -------------------------------------------------------------
# ASYNC HTTP SESSION
# -------------------------------------------------------------
class HTTPSession:
    _session = None

    @classmethod
    async def get(cls, url):
        if cls._session is None:
            cls._session = aiohttp.ClientSession()
        async with cls._session.get(url) as resp:
            return await resp.json()

    @classmethod
    async def close(cls):
        if cls._session:
            await cls._session.close()
            cls._session = None

# -------------------------------------------------------------
# FETCH OHLCV
# -------------------------------------------------------------
async def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        data = await HTTPSession.get(url)
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_base","taker_quote","ignore"
        ])
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        logger.error("OHLCV fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()

# -------------------------------------------------------------
# FETCH TICKER + SPREAD
# -------------------------------------------------------------
async def fetch_ticker(symbol: str) -> Dict[str, Any]:
    url = f"{BINANCE_BASE}/api/v3/ticker/bookTicker?symbol={symbol}"
    try:
        return await HTTPSession.get(url)
    except Exception:
        return {}

async def fetch_spread(symbol: str) -> float:
    t = await fetch_ticker(symbol)
    try:
        bid = float(t['bidPrice'])
        ask = float(t['askPrice'])
        return ((ask - bid) / bid) * 100
    except Exception:
        return 999.0

# -------------------------------------------------------------
# FETCH FUNDING RATE
# -------------------------------------------------------------
async def fetch_funding(symbol: str) -> float:
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex?symbol={symbol}"
    try:
        data = await HTTPSession.get(url)
        return float(data.get("lastFundingRate", 0)) * 100
    except Exception:
        return 0.0

# -------------------------------------------------------------
# SERVER TIME
# -------------------------------------------------------------
async def fetch_server_time() -> int:
    url = f"{BINANCE_BASE}/api/v3/time"
    try:
        data = await HTTPSession.get(url)
        return int(data.get("serverTime", 0))
    except Exception:
        return 0

# -------------------------------------------------------------
# INDICATORS: EMA, RSI, MACD, ATR, VWAP
# -------------------------------------------------------------
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal, macd_line - signal

def atr(df: pd.DataFrame, length: int = 14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def vwap(df: pd.DataFrame):
    typical = (df['high'] + df['low'] + df['close']) / 3
    return (typical * df['volume']).cumsum() / df['volume'].cumsum()

# -------------------------------------------------------------
# BTC STABILITY CHECK
# -------------------------------------------------------------
async def btc_stable() -> Dict[str, Any]:
    df = await fetch_ohlcv("BTCUSDT", "1m", 50)
    if df.empty:
        return {"ok": False, "reason": "BTC data error"}

    close = df['close']
    high = df['high']
    low = df['low']

    vol = (high.tail(10).max() - low.tail(10).min()) / close.iloc[-1] * 100

    last_high = high.iloc[-1]
    last_low = low.iloc[-1]
    last_close = close.iloc[-1]
    last_open = df['open'].iloc[-1]
    upper_wick = last_high - max(last_close, last_open)
    lower_wick = min(last_close, last_open) - last_low
    wick_pct = (upper_wick + lower_wick) / last_close * 100

    stable = vol < 1 and wick_pct < 0.5

    return {"ok": stable, "vol": vol, "wick": wick_pct}

# -------------------------------------------------------------
# HTF CANDLE PATTERN DETECTION (15m + 1h)
# -------------------------------------------------------------
def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"

    o = df['open'].iloc[-1]
    c = df['close'].iloc[-1]
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]

    if c > o and (c - o) > (h - l) * 0.6:
        return "bull_engulf"
    if o > c and (o - c) > (h - l) * 0.6:
        return "bear_engulf"
    if (h - max(o, c)) > (c - o) * 2:
        return "shooting_star"
    if (min(o, c) - l) > (c - o) * 2:
        return "hammer"
    return "none"

async def htf_signal(symbol: str) -> Dict[str, str]:
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    df1h = await fetch_ohlcv(symbol, "1h", 50)

    if df15.empty or df1h.empty:
        return {"15m": "none", "1h": "none"}

    p15 = detect_pattern(df15)
    p1h = detect_pattern(df1h)
    return {"15m": p15, "1h": p1h}

# -------------------------------------------------------------
# PRICE ACTION FILTERS
# -------------------------------------------------------------
def liquidity_sweep(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    w = (df['high'].iloc[-1] - df['low'].iloc[-1])
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    return w > body * 3

def ema21_pullback(df: pd.DataFrame) -> bool:
    e21 = ema(df['close'], 21)
    if len(df) < 2:
        return False
    return abs(df['close'].iloc[-1] - e21.iloc[-1]) / df['close'].iloc[-1] < 0.002

def range_break_retest(df: pd.DataFrame) -> bool:
    if len(df) < 20:
        return False
    high_line = df['high'].tail(20).max()
    broke = df['close'].iloc[-1] > high_line
    retest = df['low'].iloc[-1] <= high_line
    return broke and retest

def detect_order_block(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    if df['close'].iloc[-2] < df['open'].iloc[-2] and df['close'].iloc[-1] > df['open'].iloc[-1]:
        return "bull_OB"
    if df['close'].iloc[-2] > df['open'].iloc[-2] and df['close'].iloc[-1] < df['open'].iloc[-1]:
        return "bear_OB"
    return "none"

def detect_fvg(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    h1 = df['high'].iloc[-3]
    l1 = df['low'].iloc[-3]
    h2 = df['high'].iloc[-1]
    l2 = df['low'].iloc[-1]

    if l1 > h2:
        return "bull_FVG"
    if h1 < l2:
        return "bear_FVG"
    return "none"

# -------------------------------------------------------------
# SPREAD + FUNDING + SESSION
# -------------------------------------------------------------
async def spread_ok(symbol: str) -> bool:
    sp = await fetch_spread(symbol)
    return sp < 0.06

async def funding_ok(symbol: str) -> bool:
    f = await fetch_funding(symbol)
    return abs(f) < 0.02

def session_now(ts: int) -> str:
    h = time.gmtime(ts/1000).tm_hour
    if 1 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "europe"
    return "us"

# -------------------------------------------------------------
# VOLUME FILTERS • SCORE SYSTEM • MODES (QUICK/MID/TREND)
# COOLDOWN SYSTEM • COIN LIST (50 COINS)
# -------------------------------------------------------------
COOLDOWN: Dict[str, int] = {}

def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < 30:
        return False
    recent = df['volume'].iloc[-1]
    avg = df['volume'].tail(20).mean()
    return recent > avg * 2

def calc_score(symbol: str, df: pd.DataFrame, htf: Dict[str, str],
               pa: Dict[str, Any], spread_ok_flag: bool, funding_ok_flag: bool) -> int:
    score = 0

    close = df['close']
    if len(close) > 50:
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        if e20 > e50:
            score += 15

    if volume_spike(df):
        score += 15

    if htf['15m'] in ["bull_engulf", "hammer"]:
        score += 10
    if htf['1h'] in ["bull_engulf", "hammer"]:
        score += 10

    if pa.get('sweep'):
        score += 10
    if pa.get('pullback'):
        score += 10
    if pa.get('range_retest'):
        score += 10
    if pa.get('ob') != "none":
        score += 10
    if pa.get('fvg') != "none":
        score += 10

    if spread_ok_flag:
        score += 5
    if funding_ok_flag:
        score += 5

    return min(score, 100)

def mode_requirements(df: pd.DataFrame, mode: str) -> bool:
    close = df['close']
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]

    if mode == "quick":
        return e20 > e50

    if mode == "mid":
        return e20 > e50 and volume_spike(df)

    if mode == "trend":
        return e20 > e50 and close.iloc[-1] > e20

    return False

def cooldown_ok(symbol: str) -> bool:
    now = int(time.time())
    if symbol not in COOLDOWN:
        return True
    return (now - COOLDOWN[symbol]) > 1800

def set_cooldown(symbol: str) -> None:
    COOLDOWN[symbol] = int(time.time())

COIN_LIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","MATICUSDT",
    "DOTUSDT","LTCUSDT","BCHUSDT","AVAXUSDT","UNIUSDT","LINKUSDT","ATOMUSDT","ETCUSDT",
    "FILUSDT","ICPUSDT","NEARUSDT","APTUSDT","SANDUSDT","AXSUSDT","THETAUSDT","FTMUSDT",
    "RUNEUSDT","ALGOUSDT","EGLDUSDT","IMXUSDT","INJUSDT","OPUSDT","ARBUSDT","SUIUSDT",
    "TIAUSDT","PEPEUSDT","TRBUSDT","SEIUSDT","JTOUSDT","PYTHUSDT","RAYUSDT","GMTUSDT",
    "MINAUSDT","WLDUSDT","ZKUSDT","STRKUSDT","DYDXUSDT","VETUSDT","GALAUSDT","KAVAUSDT",
    "FLOWUSDT","SKLUSDT"
]

# -------------------------------------------------------------
# SIGNAL BUILDER • TP/SL ENGINE • PREMIUM TELEGRAM FORMATTER
# -------------------------------------------------------------
def calc_tp_sl(entry: float, mode: str) -> Dict[str, float]:
    if mode == "quick":
        tp = entry * 1.004
        sl = entry * 0.996
    elif mode == "mid":
        tp = entry * 1.020
        sl = entry * 0.990
    elif mode == "trend":
        tp = entry * 1.030
        sl = entry * 0.985
    else:
        tp = entry * 1.010
        sl = entry * 0.990

    return {"tp": round(tp, 8), "sl": round(sl, 8)}

async def build_signal(symbol: str, mode: str) -> Dict[str, Any]:
    if not cooldown_ok(symbol):
        return {"ok": False, "reason": "cooldown"}

    df = await fetch_ohlcv(symbol, "1m", 200)
    if df.empty:
        return {"ok": False, "reason": "data_error"}

    htf = await htf_signal(symbol)

    pa = {
        "sweep": liquidity_sweep(df),
        "pullback": ema21_pullback(df),
        "range_retest": range_break_retest(df),
        "ob": detect_order_block(df),
        "fvg": detect_fvg(df)
    }

    sp_ok = await spread_ok(symbol)
    f_ok = await funding_ok(symbol)

    score = calc_score(symbol, df, htf, pa, sp_ok, f_ok)
    if score < 90:
        return {"ok": False, "reason": "low_score", "score": score}

    if not mode_requirements(df, mode):
        return {"ok": False, "reason": "mode_not_fit"}

    entry = float(df['close'].iloc[-1])
    tpsl = calc_tp_sl(entry, mode)

    set_cooldown(symbol)

    return {
        "ok": True,
        "symbol": symbol,
        "mode": mode,
        "entry": entry,
        "tp": tpsl['tp'],
        "sl": tpsl['sl'],
        "score": score,
        "htf": htf,
        "pa": pa,
        "time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }

def format_signal(sig: Dict[str, Any]) -> str:
    return f"""
🔥 BUY SIGNAL — {sig['mode'].upper()} MODE
━━━━━━━━━━━━━━━━
Pair: {sig['symbol']}
━━━━━━━━━━━━━━━━
🎯 Entry: {sig['entry']}
🏆 TP: {sig['tp']}
🛑 SL: {sig['sl']}
📈 Score: {sig['score']}
━━━━━━━━━━━━━━━━
🔍 HTF:
15m → {sig['htf']['15m']}
1h → {sig['htf']['1h']}
━━━━━━━━━━━━━━━━
📊 Price Action:
Sweep: {sig['pa']['sweep']}
Pullback: {sig['pa']['pullback']}
Range Retest: {sig['pa']['range_retest']}
OB: {sig['pa']['ob']}
FVG: {sig['pa']['fvg']}
━━━━━━━━━━━━━━━━
⏱ Time: {sig['time']}
"""

def format_exit_alert(symbol: str, reason_main: str, extra: str = "") -> str:
    return f"""
🛑 EMERGENCY EXIT — {symbol}
━━━━━━━━━━━━━━━━━━━━
⚠️ DANGER DETECTED
Reason: {reason_main}
{extra}
━━━━━━━━━━━━━━━━━━━━
🚨 ACTION: EXIT NOW — PROTECT CAPITAL
━━━━━━━━━━━━━━━━━━━━
⏱ Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""

# -------------------------------------------------------------
# MAIN SCAN LOOP • MULTI-MODE RUNNER
# -------------------------------------------------------------
async def scan_coin(symbol: str, mode: str) -> Dict[str, Any]:
    try:
        sig = await build_signal(symbol, mode)
        return sig
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def run_mode(mode: str) -> List[Dict[str, Any]]:
    tasks = [scan_coin(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r.get("ok")]

async def run_all_modes() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "quick": await run_mode("quick"),
        "mid": await run_mode("mid"),
        "trend": await run_mode("trend"),
    }

# -------------------------------------------------------------
# OVERRIDE UPGRADE HELPERS
# -------------------------------------------------------------
def _to_pd(df_like):
    if isinstance(df_like, pd.DataFrame):
        return df_like
    try:
        return pd.DataFrame(df_like)
    except Exception:
        return None

def htf_alignment(ohlcv_15m, ohlcv_1h, ema_short: int = 20, ema_long: int = 50) -> bool:
    try:
        df15 = _to_pd(ohlcv_15m)
        df60 = _to_pd(ohlcv_1h)
        if df15 is None or df60 is None or len(df15) < ema_long or len(df60) < ema_long:
            return True

        s15 = ema(df15["close"], ema_short).iloc[-1]
        l15 = ema(df15["close"], ema_long).iloc[-1]
        s60 = ema(df60["close"], ema_short).iloc[-1]
        l60 = ema(df60["close"], ema_long).iloc[-1]

        dir15 = 1 if s15 > l15 else -1
        dir60 = 1 if s60 > l60 else -1
        return dir15 == dir60
    except Exception as e:
        logger.warning("htf_alignment failed: %s", e)
        return True

def detect_liquidity_sweep_adv(ohlcv, wick_ratio_threshold: float = 2.5,
                               vol_mult: float = 1.5) -> Dict[str, Any]:
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 3:
            return {"sweep": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        if body <= 0:
            return {"sweep": False, "reason": "zero_body"}

        upper_wick = last["high"] - max(last["open"], last["close"])
        lower_wick = min(last["open"], last["close"]) - last["low"]
        wick_ratio = max(upper_wick, lower_wick) / (body + 1e-12)

        avg_vol = df["volume"].rolling(10, min_periods=1).mean().iloc[-2]
        vol = last["volume"]
        vol_spike = vol > max(avg_vol * vol_mult, 1e-6)

        if wick_ratio >= wick_ratio_threshold and vol_spike:
            direction = "upper" if upper_wick > lower_wick else "lower"
            reason = f"wick_ratio={wick_ratio:.2f}, vol_spike={vol_spike}, dir={direction}"
            return {"sweep": True, "reason": reason, "direction": direction}
        return {"sweep": False, "reason": "none"}
    except Exception as e:
        logger.warning("detect_liquidity_sweep_adv failed: %s", e)
        return {"sweep": False, "reason": "error"}

def detect_sudden_spike(ohlcv, vol_multiplier: float = 3.0,
                        price_move_pct: float = 0.01) -> Dict[str, Any]:
    try:
        df = _to_pd(ohlcv)
        if df is None or len(df) < 10:
            return {"spike": False, "reason": "insufficient_data"}

        last = df.iloc[-1]
        prev_avg_vol = df["volume"].rolling(10, min_periods=5).mean().iloc[-2]
        vol_spike = last["volume"] > max(prev_avg_vol * vol_multiplier, 1e-6)
        prev_close = df["close"].iloc[-2]
        price_move = abs(last["close"] - prev_close) / (prev_close + 1e-12)

        if vol_spike and price_move >= price_move_pct:
            reason = f"vol_mult={(last['volume']/max(prev_avg_vol,1e-12)):.2f}, price_move={price_move:.3f}"
            return {
                "spike": True,
                "reason": reason,
                "price_move": price_move,
                "vol_mult": last["volume"] / max(prev_avg_vol, 1e-12),
            }
        return {"spike": False, "reason": "none"}
    except Exception as e:
        logger.warning("detect_sudden_spike failed: %s", e)
        return {"spike": False, "reason": "error"}

async def sl_before_touch_check(active_signals: List[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    if not active_signals:
        return warnings

    for sig in active_signals:
        try:
            symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
            if not symbol:
                continue
            sl = float(sig.get("sl", 0) or 0)
            entry = float(sig.get("entry", 0) or 0)
            if sl == 0 or entry == 0:
                continue

            dist_pct = abs(entry - sl) / (entry + 1e-12)
            if dist_pct <= 0.006:
                warnings.append(
                    f"⚠️ SL CLOSE: {symbol} entry={entry} sl={sl} dist={dist_pct:.4f}"
                )
        except Exception as e:
            logger.debug("sl_before_touch_check item fail: %s", e)
            continue
    return warnings

# -------------------------------------------------------------
# MULTI OVERRIDE WATCH (MAIN)
# -------------------------------------------------------------
async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    alerts: List[str] = []
    if not isinstance(active_signals, list):
        logger.warning("multi_override_watch: expected list, got %s", type(active_signals))
        return alerts

    # 1) SL-close warnings
    sl_warns = await sl_before_touch_check(active_signals)
    alerts.extend(sl_warns)

    for sig in active_signals:
        try:
            symbol = sig.get("symbol") or sig.get("pair") or sig.get("market")
            if not symbol:
                continue

            # fetch HTF/1m data
            ohlcv_1m = await fetch_ohlcv(symbol, "1m", 80)
            ohlcv_15m = await fetch_ohlcv(symbol, "15m", 80)
            ohlcv_1h = await fetch_ohlcv(symbol, "1h", 80)

            # HTF alignment
            critical_flags: List[str] = []
            if not htf_alignment(ohlcv_15m, ohlcv_1h):
                alerts.append(f"⚠️ HTF MISALIGN: {symbol} — skip or review (15m vs 1h conflict)")
                critical_flags.append("HTF_MISALIGN")

            sweep = detect_liquidity_sweep_adv(ohlcv_1m)
            if sweep.get("sweep"):
                alerts.append(f"⚠️ LIQ SWEEP: {symbol} — {sweep.get('reason')}")
                critical_flags.append("LIQ_SWEEP")

            spike = detect_sudden_spike(ohlcv_1m)
            if spike.get("spike"):
                alerts.append(f"⚠️ SUDDEN SPIKE: {symbol} — {spike.get('reason')}")
                critical_flags.append("SUDDEN_SPIKE")

            if critical_flags:
                parts = [
                    "🚨 ACTION: EXIT NOW — PROTECT CAPITAL",
                    f"Pair: {symbol}",
                    f"Issues: {', '.join(critical_flags)}",
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
                ]
                entry = sig.get("entry")
                tp = sig.get("tp") or sig.get("tps")
                sl = sig.get("sl")
                if entry:
                    parts.append(f"Entry: {entry}")
                if tp:
                    parts.append(f"TP: {tp}")
                if sl:
                    parts.append(f"SL: {sl}")
                alert_text = "\n".join(parts)
                alerts.append(alert_text)

                # try Telegram send
                try:
                    await send_telegram(alert_text)
                except Exception as e:
                    logger.error("Telegram send failed in override: %s", e)

        except Exception as e:
            logger.debug("multi_override per-sig fail: %s", e)
            continue

    # dedupe
    unique: List[str] = []
    seen = set()
    for a in alerts:
        if a not in seen:
            unique.append(a)
            seen.add(a)
    return unique

__all__ = [
    "scan_coin",
    "run_mode",
    "run_all_modes",
    "multi_override_watch",
    "format_signal",
    "format_exit_alert",
    "send_telegram",
]
