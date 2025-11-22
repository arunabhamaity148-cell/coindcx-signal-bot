"""
helpers.py – Sniper-grade Binance scalper engine
Live news, dynamic size, emoji TG, zero-error.
Exposed:
  run_all_modes()
  multi_override_watch()
  format_signal()
  send_telegram()
  log_signal()
"""

import asyncio
import logging
import os
import time
import csv
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# ----------------- LOGGING -----------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/signals.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------- ENV -----------------
SCORE_MIN            = int(os.getenv("SCORE_MIN", "90"))
BASE_CAPITAL         = float(os.getenv("BASE_CAPITAL", "100000"))
BASE_RISK_PCT        = float(os.getenv("BASE_RISK_PCT", "0.01"))
MAX_RISK_PCT         = float(os.getenv("MAX_RISK_PCT", "0.015"))
MAX_CONCURRENT_SCANS = int(os.getenv("MAX_CONCURRENT_SCANS", "10"))

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID")

# ----------------- CONST -----------------
BINANCE_BASE    = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"

KILLZONES = {
    "asia":   (0, 8),
    "london": (7, 11),
    "ny":     (12, 17),
}

MODE_RRR      = {"quick": 1.5, "mid": 2.0, "trend": 3.0}
MODE_ATR_MULT = {"quick": 1.0, "mid": 1.5, "trend": 2.0}

# ----------------- HTTP -----------------
class HTTPSession:
    _session: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def get(cls, url: str, retries: int = 3, timeout: int = 8) -> Any:
        if cls._session is None:
            cls._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
        for i in range(retries):
            try:
                async with cls._session.get(url) as r:
                    return await r.json()
            except Exception as e:
                logger.warning("HTTP error (%s): %s", url, e)
                await asyncio.sleep(0.5 * (i + 1))
        logger.error("HTTP fail %s", url)
        return None

    @classmethod
    async def close(cls) -> None:
        if cls._session:
            await cls._session.close()
            cls._session = None

# ----------------- DATA -----------------
async def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await HTTPSession.get(url)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb", "tq", "ignore",
        ],
    )
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


async def fetch_spread(symbol: str) -> float:
    url = f"{BINANCE_BASE}/api/v3/ticker/bookTicker?symbol={symbol}"
    data = await HTTPSession.get(url)
    try:
        bid = float(data["bidPrice"])
        ask = float(data["askPrice"])
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

# ----------------- INDICATORS -----------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


async def adr(symbol: str) -> float:
    df1d = await fetch_ohlcv(symbol, "1d", 15)
    if df1d.empty:
        return 3.0
    ranges = (df1d["high"] - df1d["low"]) / df1d["close"] * 100
    return ranges.iloc[-14:].mean()

# ----------------- ICT / SESSION -----------------
def session_now() -> str:
    h = datetime.utcnow().hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "london"
    return "ny"


def killzone_active(mode: str) -> bool:
    if mode == "quick":
        return True
    sess = session_now()
    start, end = KILLZONES[sess]
    return start <= datetime.utcnow().hour < end

# ----------------- LIVE NEWS (investpy) -----------------
def high_impact_now(minutes: int = 15) -> bool:
    try:
        import investpy

        today = datetime.utcnow().strftime("%d/%m/%Y")
        cal = investpy.economic_calendar(
            time_zone="GMT",
            countries=["united states", "euro zone", "united kingdom"],
            from_date=today,
            to_date=today,
        )
        if cal.empty:
            return False
        cal["time"] = pd.to_datetime(
            cal["time"], format="%H:%M:%S"
        ).dt.tz_localize(None)
        now = datetime.utcnow()
        for _, row in cal.iterrows():
            if (
                row["impact"] == "High"
                and abs((row["time"] - now).total_seconds()) < minutes * 60
            ):
                return True
    except Exception as e:
        logger.warning("news check fail: %s", e)
    return False

# ----------------- BIAS -----------------
async def ssl_hybrid_bias(symbol: str) -> str:
    df1h = await fetch_ohlcv(symbol, "1h", 50)
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    if df1h.empty or df15.empty:
        return "none"

    bias1h = "bull" if df1h["close"].iloc[-1] > ema(df1h["close"], 20).iloc[-1] else "bear"
    bias15 = "bull" if df15["close"].iloc[-1] > ema(df15["close"], 20).iloc[-1] else "bear"

    if bias1h == bias15 == "bull":
        return "bull"
    if bias1h == bias15 == "bear":
        return "bear"
    return "none"

# ----------------- SMART TP/SL -----------------
async def smart_tp_sl(
    symbol: str,
    entry: float,
    side: str,
    mode: str,
    df1m: pd.DataFrame,
) -> Dict[str, float]:
    atrv = atr(df1m, 14).iloc[-1]
    mult = MODE_ATR_MULT[mode]
    sl_dist = atrv * mult
    rrr = MODE_RRR[mode]
    side = side.upper()

    if side == "BUY":
        sl = entry - sl_dist
        tp = entry + sl_dist * rrr
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * rrr

    return {"tp": round(tp, 8), "sl": round(sl, 8), "atr": float(atrv)}

# ----------------- PATTERN -----------------
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

    if body > rng * 0.6:
        return "bull_engulf" if c > o else "bear_engulf"

    upper = h - max(o, c)
    lower = min(o, c) - l
    if upper > body * 2:
        return "shooting_star"
    if lower > body * 2:
        return "hammer"
    return "none"

# ----------------- PRICE ACTION -----------------
def liquidity_sweep(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    o = df["open"].iloc[-1]
    c = df["close"].iloc[-1]
    h = df["high"].iloc[-1]
    l = df["low"].iloc[-1]

    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    wick = rng - body
    return wick > body * 2.5


def ema21_pullback(df: pd.DataFrame) -> bool:
    if len(df) < 22:
        return False
    return (
        abs(df["close"].iloc[-1] - ema(df["close"], 21).iloc[-1])
        / df["close"].iloc[-1]
        < 0.002
    )


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

# ----------------- SCORE v2 -----------------
def calc_score_v2(
    symbol: str,
    df: pd.DataFrame,
    htf: Dict[str, str],
    pa: Dict[str, Any],
    spr_ok: bool,
    fund_ok: bool,
    bias: str,
) -> int:
    score = 0
    close = df["close"]

    if len(close) > 50:
        if ema(close, 20).iloc[-1] > ema(close, 50).iloc[-1]:
            score += 15

    if bias == "bull":
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

    if htf.get("15m") in ("bull_engulf", "hammer"):
        score += 10
    if htf.get("1h") in ("bull_engulf", "hammer"):
        score += 10

    if spr_ok:
        score += 5
    if fund_ok:
        score += 5

    return min(score, 100)

# ----------------- DYNAMIC POSITION SIZE -----------------
async def fetch_balance_usdt() -> float:
    """Binance futures USDT balance via ccxt (read-only API)."""
    try:
        import ccxt

        exchange = ccxt.binance(
            {
                "apiKey": os.getenv("BINANCE_KEY", ""),
                "secret": os.getenv("BINANCE_SECRET", ""),
                "options": {"defaultType": "future"},
            }
        )
        bal = exchange.fetch_balance()
        return float(bal["USDT"]["free"])
    except Exception as e:
        logger.warning("balance fetch fail: %s, using BASE_CAPITAL", e)
        return BASE_CAPITAL


async def calc_position_size(
    symbol: str,
    entry: float,
    sl: float,
    mode: str,
) -> Dict[str, float]:
    balance = await fetch_balance_usdt()
    risk_pct = BASE_RISK_PCT
    risk_amt = balance * risk_pct
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return {"pos_size_usdt": 0.0, "risk_usdt": 0.0}
    size = risk_amt / sl_dist * entry
    return {"pos_size_usdt": round(size, 2), "risk_usdt": round(risk_amt, 2)}

# ----------------- BUILD SIGNAL v2 -----------------
COOLDOWN: Dict[str, int] = {}


def cooldown_ok(sym: str) -> bool:
    return int(time.time()) - COOLDOWN.get(sym, 0) > 1800


def set_cooldown(sym: str) -> None:
    COOLDOWN[sym] = int(time.time())


async def build_signal_v2(symbol: str, mode: str) -> Dict[str, Any]:
    mode = mode.lower()

    if not cooldown_ok(symbol):
        return {"ok": False, "reason": "cooldown"}
    if high_impact_now():
        return {"ok": False, "reason": "news_block"}
    if not killzone_active(mode):
        return {"ok": False, "reason": "killzone"}

    df1m = await fetch_ohlcv(symbol, "1m", 200)
    if df1m.empty:
        return {"ok": False, "reason": "data"}

    df1d = await fetch_ohlcv(symbol, "1d", 15)
    if df1d.empty:
        return {"ok": False, "reason": "data_1d"}

    adr_val = await adr(symbol)
    today_range = (df1d["high"].iloc[-1] - df1d["low"].iloc[-1]) / df1d["close"].iloc[-1] * 100
    if today_range > adr_val * 0.7:
        return {"ok": False, "reason": "adr_exhaust"}

    bias = await ssl_hybrid_bias(symbol)
    if bias == "none":
        return {"ok": False, "reason": "bias_none"}

    htf = {
        "15m": detect_pattern(await fetch_ohlcv(symbol, "15m", 50)),
        "1h": detect_pattern(await fetch_ohlcv(symbol, "1h", 50)),
    }

    pa = {
        "sweep": liquidity_sweep(df1m),
        "pullback": ema21_pullback(df1m),
        "range_retest": range_break_retest(df1m),
        "ob": detect_order_block(df1m),
        "fvg": detect_fvg(df1m),
    }

    spr_ok = await fetch_spread(symbol) < 0.06
    fund_ok = abs(await fetch_funding(symbol)) < 0.02

    score = calc_score_v2(symbol, df1m, htf, pa, spr_ok, fund_ok, bias)
    if score < SCORE_MIN:
        return {"ok": False, "reason": "low_score"}

    entry = float(df1m["close"].iloc[-1])
    tpsl = await smart_tp_sl(symbol, entry, "BUY", mode, df1m)
    pos = await calc_position_size(symbol, entry, tpsl["sl"], mode)

    set_cooldown(symbol)
    return {
        "ok": True,
        "symbol": symbol,
        "mode": mode,
        "side": "BUY",
        "entry": entry,
        "tp": tpsl["tp"],
        "sl": tpsl["sl"],
        "score": score,
        "htf": htf,
        "pa": pa,
        "pos": pos,
        "time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
# ----------------- RUNNER -----------------
_scan_sem = asyncio.Semaphore(MAX_CONCURRENT_SCANS)


async def scan_coin(sym: str, mode: str) -> Dict[str, Any]:
    async with _scan_sem:
        try:
            return await build_signal_v2(sym, mode)
        except Exception as e:
            logger.error("scan %s %s: %s", sym, mode, e)
            return {"ok": False, "error": str(e)}


async def run_mode(mode: str) -> List[Dict[str, Any]]:
    tasks = [scan_coin(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r.get("ok")]
    logger.info("mode=%s signals=%d", mode, len(valid))
    return valid


async def run_all_modes() -> Dict[str, List[Dict[str, Any]]]:
    return {m: await run_mode(m) for m in ("quick", "mid", "trend")}

# ----------------- TG / LOG -----------------
async def send_telegram(text: str) -> None:
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json=payload)
    except Exception as e:
        logger.warning("tg send fail: %s", e)


def log_signal(sig: Dict[str, Any]) -> None:
    os.makedirs("logs", exist_ok=True)
    with open("logs/trades.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                sig["time_utc"],
                sig["symbol"],
                sig["side"],
                sig["entry"],
                sig["tp"],
                sig["sl"],
                sig["score"],
                sig["mode"],
            ]
        )

# ----------------- EMOJI RICH TG FORMATTER -----------------
def format_signal(sig: Dict[str, Any], no: int) -> str:
    symbol = sig["symbol"]
    mode = sig["mode"].upper()
    side = sig["side"]
    score = sig["score"]
    entry = sig["entry"]
    tp = sig["tp"]
    sl = sig["sl"]
    pos = sig.get("pos", {})
    time_utc = sig["time_utc"]
    htf = sig.get("htf", {})
    pa = sig.get("pa", {})

    ist_time = (
        datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S")
        + timedelta(hours=5, minutes=30)
    ).strftime("%I:%M %p")
    hold_till = (
        datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S")
        + timedelta(minutes={"quick": 15, "mid": 45, "trend": 120}[mode])
    ).strftime("%I:%M %p")

    emoji_side = "🟢" if side == "BUY" else "🔴"
    emoji_mode = {"QUICK": "⚡", "MID": "🎯", "TREND": "📈"}.get(mode, "🎯")

    return f"""
{emoji_side} <b>{side} {emoji_mode} {mode}</b>  #{no}
┏━━━━━━━━━━━━━━━━━━━━
┃ 📌 <b>{symbol}</b>  |  Score: <b>{score}/100</b>
┗━━━━━━━━━━━━━━━━━━━━
🔍 HTF 15m: <code>{htf.get('15m','none')}</code>  |  1h: <code>{htf.get('1h','none')}</code>
📊 PA: Sweep {pa.get('sweep')}  |  OB: {pa.get('ob')}  |  FVG: {pa.get('fvg')}
💰 Size: <b>{pos.get('pos_size_usdt',0)} USDT</b>  |  Risk: {pos.get('risk_usdt',0)} USDT
🕐 IST: {ist_time}  |  Hold till: {hold_till}
┏━━━━━━━━━━━━━━━━━━━━
┃ 💥 COPY BLOCK
┃ Entry: <code>{entry}</code>
┃ TP:    <code>{tp}</code>
┃ SL:    <code>{sl}</code>
┗━━━━━━━━━━━━━━━━━━━━
"""

# ----------------- OVERRIDE WATCH (SL + TP alerts) -----------------
ACTIVE_TRADES: List[Dict[str, Any]] = []


async def multi_override_watch(active_signals: List[Dict[str, Any]]) -> List[str]:
    alerts: List[str] = []
    if not active_signals:
        return alerts

    btc_df = await fetch_ohlcv("BTCUSDT", "1m", 12)
    btc_move = 0.0
    if not btc_df.empty and len(btc_df) >= 10:
        btc_move = (
            btc_df["close"].iloc[-1] - btc_df["close"].iloc[-10]
        ) / btc_df["close"].iloc[-10] * 100

    for sig in active_signals:
        if not sig.get("ok"):
            continue

        symbol = sig["symbol"]
        side = sig.get("side", "BUY")
        entry = float(sig["entry"])
        sl = float(sig["sl"])
        tp = float(sig["tp"])
        time_utc = sig["time_utc"]
        mode = sig.get("mode", "quick")

        df = await fetch_ohlcv(symbol, "1m", 20)
        if df.empty or len(df) < 6:
            continue

        price = float(df["close"].iloc[-1])
        age_min = (
            datetime.utcnow() - datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S")
        ).total_seconds() / 60

        if side == "BUY":
            move_pct = (price - entry) / entry * 100
            dist_sl = abs(price - sl) / price * 100
            dist_tp = abs(tp - price) / price * 100
            in_loss = price < entry
        else:
            move_pct = (entry - price) / entry * 100
            dist_sl = abs(price - sl) / price * 100
            dist_tp = abs(price - tp) / price * 100
            in_loss = price > entry

        reasons: List[str] = []
        strong = 0

        # ----- SL danger -----
        if dist_sl < 0.3 and in_loss:
            strong += 1
            reasons.append(f"SL very close (~{dist_sl:.2f}%)")
        if age_min <= 5 and move_pct <= -0.7:
            strong += 1
            reasons.append(f"Fast loss (~{move_pct:.2f}%)")
        if move_pct <= -1.0:
            strong += 1
            reasons.append(f"Big loss (~{move_pct:.2f}%)")
        if btc_move < -0.7 and in_loss:
            strong += 1
            reasons.append(f"BTC pressure (~{btc_move:.2f}%)")
        if age_min >= 30 and abs(move_pct) < 0.2:
            reasons.append(f"Time trap (~{age_min:.1f} min)")

        level = (
            "EMERGENCY"
            if strong >= 2
            else "HIGH"
            if strong == 1
            else "CAUTION"
            if strong
            else None
        )
        if level:
            emoji = {"EMERGENCY": "🔴", "HIGH": "🟠", "CAUTION": "🟡"}.get(level, "⚠️")
            alerts.append(
                f"{emoji} <b>{level} ALERT</b>\n"
                f"📌 <b>{symbol}</b>\n"
                "Reasons:\n- " + "\n- ".join(reasons) +
                f"\nEntry: <code>{entry}</code>\n"
                f"TP: <code>{tp}</code>\n"
                f"SL: <code>{sl}</code>\n"
                f"Now: <code>{price:.4f}</code> ({move_pct:+.2f}%)"
            )

        # ----- TP side (profit management) -----
        tp_reasons: List[str] = []
        tp_level = None

        if move_pct >= 0.8 and dist_tp < 0.3:
            tp_level = "TP_HIT_CHANCE"
            tp_reasons.append(f"Price very close to TP (~{dist_tp:.2f}%)")
        elif move_pct >= 0.6 and dist_tp < 0.6:
            tp_level = "TP_NEAR"
            tp_reasons.append(
                f"Good profit and close to TP (pnl ~{move_pct:.2f}%, dist ~{dist_tp:.2f}%)"
            )
        elif move_pct >= 1.2:
            tp_level = "TP_STRONG_PROFIT"
            tp_reasons.append(f"Strong profit (~{move_pct:.2f}%)")

        if mode == "trend" and age_min >= 90 and move_pct >= 0.5:
            if tp_level is None:
                tp_level = "TP_TRAIL_IDEA"
            tp_reasons.append(
                f"Old trend position (~{age_min:.1f} min) with profit (~{move_pct:.2f}%)"
            )

        if tp_level and tp_reasons:
            tp_emoji = "✅" if tp_level == "TP_HIT_CHANCE" else "🟢"
            alerts.append(
                f"{tp_emoji} <b>{tp_level} – TP zone</b>\n"
                f"📌 <b>{symbol}</b>\n"
                "Reasons:\n- " + "\n- ".join(tp_reasons) +
                f"\nEntry: <code>{entry}</code>\n"
                f"TP: <code>{tp}</code>\n"
                f"SL: <code>{sl}</code>\n"
                f"Now: <code>{price:.4f}</code> ({move_pct:+.2f}%)"
            )

    return alerts

# ----------------- COIN LIST -----------------
COIN_LIST: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "MATICUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "AVAXUSDT", "UNIUSDT", "LINKUSDT", "ATOMUSDT", "ETCUSDT",
    "FILUSDT", "ICPUSDT", "NEARUSDT", "APTUSDT", "SANDUSDT", "AXSUSDT", "THETAUSDT", "FTMUSDT",
    "RUNEUSDT", "ALGOUSDT", "EGLDUSDT", "IMXUSDT", "INJUSDT", "OPUSDT", "ARBUSDT", "SUIUSDT",
    "TIAUSDT", "PEPEUSDT", "TRBUSDT", "SEIUSDT", "JTOUSDT", "PYTHUSDT", "RAYUSDT", "GMTUSDT",
    "MINAUSDT", "WLDUSDT", "ZKUSDT", "STRKUSDT", "DYDXUSDT", "VETUSDT", "GALAUSDT", "KAVAUSDT",
]

# ----------------- EXPORT -----------------
__all__ = [
    "run_all_modes",
    "multi_override_watch",
    "format_signal",
    "send_telegram",
    "log_signal",
]