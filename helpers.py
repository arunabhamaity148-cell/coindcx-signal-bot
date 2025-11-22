"""
helpers.py – Sniper-grade Binance scalper engine
ASCII-only, async, ICT-aware, ADR-aware, news-aware.
Exposed:
  run_all_modes()
  multi_override_watch()
  format_signal()
"""

import asyncio, logging, math, os, time, smtplib, csv, aiohttp, pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="logs/signals.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------- ENV -----------------
SCORE_MIN               = int(os.getenv("SCORE_MIN", "90"))
BASE_CAPITAL            = float(os.getenv("BASE_CAPITAL", "100000"))
BASE_RISK_PCT           = float(os.getenv("BASE_RISK_PCT", "0.01"))
MAX_RISK_PCT            = float(os.getenv("MAX_RISK_PCT", "0.015"))
MAX_CONCURRENT_SCANS    = int(os.getenv("MAX_CONCURRENT_SCANS", "10"))
TG_BOT_TOKEN            = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID              = os.getenv("TG_CHAT_ID")

# ----------------- CONST -----------------
BINANCE_BASE   = "https://api.binance.com"
BINANCE_FUTURES= "https://fapi.binance.com"

# ICT Killzones (UTC)
KILLZONES = {
    "asia":  (0, 8),   # 00-08 UTC
    "london":(7, 11),  # 08-12 BST ≈ 07-11 UTC
    "ny":    (12, 17), # 08-13 EST ≈ 12-17 UTC
}

# Mode-wise RRR and ATR multiplier
MODE_RRR = {"quick": 1.5, "mid": 2.0, "trend": 3.0}
MODE_ATR_MULT = {"quick": 1.0, "mid": 1.5, "trend": 2.0}

# ----------------- HTTP -----------------
class HTTPSession:
    _session: Optional[aiohttp.ClientSession] = None
    @classmethod
    async def get(cls, url: str, retries: int = 3, timeout: int = 8) -> Any:
        if cls._session is None:
            cls._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
        for i in range(retries):
            try:
                async with cls._session.get(url) as r:
                    return await r.json()
            except Exception as e:
                await asyncio.sleep(0.5*(i+1))
        logger.error("HTTP fail %s", url)
        return None
    @classmethod
    async def close(cls):
        if cls._session:
            await cls._session.close()
            cls._session = None

# ----------------- DATA -----------------
async def fetch_ohlcv(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await HTTPSession.get(url)
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","tb","tq","ignore"])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

# ----------------- INDICATORS -----------------
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(df["high"]-df["low"]), (df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def adr(df: pd.DataFrame) -> float:
    """last 14 days (daily) range average in %"""
    df1d = await fetch_ohlcv(df["symbol"].iloc[0], "1d", 15)  # cached via caller
    if df1d.empty: return 3.0
    ranges = (df1d["high"] - df1d["low"]) / df1d["close"] * 100
    return ranges.iloc[-14:].mean()

# ----------------- ICT / SESSION -----------------
def utc_now(): return datetime.utcnow()
def session_now() -> str:
    h = utc_now().hour
    if 0 <= h < 8: return "asia"
    if 8 <= h < 16: return "london"
    return "ny"
def killzone_active(mode: str) -> bool:
    """only allow signals inside killzone for mid/trend"""
    if mode == "quick": return True  # quick always ok
    sess = session_now()
    h = utc_now().hour
    start, end = KILLZONES[sess]
    return start <= h < end

# ----------------- NEWS DUMMY FILTER -----------------
HIGH_IMPACT_TIMES_UTC = [
    (12, 30),  # US CPI
    (14, 30),  # US GDP/PCE
    (12, 0),   # FOMC
]
def news_block() -> bool:
    now = utc_now()
    for h, m in HIGH_IMPACT_TIMES_UTC:
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if abs((now - t).total_seconds()) < 900:  # 15 min block
            return True
    return False

# ----------------- SWEEP / BIAS -----------------
def pdh_pdl_sweep(symbol: str, df1d: pd.DataFrame) -> Dict[str, bool]:
    if df1d.empty: return {"pdh_sweep": False, "pdl_sweep": False}
    pdh = df1d["high"].iloc[-2]
    pdl = df1d["low"].iloc[-2]
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    if df15.empty: return {"pdh_sweep": False, "pdl_sweep": False}
    h15 = df15["high"].max()
    l15 = df15["low"].min()
    return {"pdh_sweep": h15 > pdh, "pdl_sweep": l15 < pdl}

def ssl_hybrid_bias(symbol: str) -> str:
    """1h+15m SSL-like bias"""
    df1h = await fetch_ohlcv(symbol, "1h", 50)
    df15 = await fetch_ohlcv(symbol, "15m", 50)
    if df1h.empty or df15.empty: return "none"
    # simple: close above 1h ema20 = bullish
    bias1h = "bull" if df1h["close"].iloc[-1] > ema(df1h["close"], 20).iloc[-1] else "bear"
    bias15 = "bull" if df15["close"].iloc[-1] > ema(df15["close"], 20).iloc[-1] else "bear"
    if bias1h == bias15 == "bull": return "bull"
    if bias1h == bias15 == "bear": return "bear"
    return "none"

# ----------------- SMART TP/SL -----------------
async def smart_tp_sl(symbol: str, entry: float, side: str, mode: str, df1m: pd.DataFrame) -> Dict[str, float]:
    atrv = atr(df1m, 14).iloc[-1]
    mult = MODE_ATR_MULT[mode]
    sl_dist = atrv * mult
    rrr   = MODE_RRR[mode]
    if side.upper() == "BUY":
        sl = entry - sl_dist
        tp = entry + sl_dist * rrr
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * rrr
    return {"tp": round(tp, 8), "sl": round(sl, 8), "atr": atrv}

# ----------------- SCORE v2 -----------------
def calc_score_v2(symbol: str, df: pd.DataFrame, htf: Dict, pa: Dict, spr_ok: bool, fund_ok: bool, bias: str) -> int:
    score = 0
    close = df["close"]
    if len(close) > 50:
        if ema(close, 20).iloc[-1] > ema(close, 50).iloc[-1]: score += 15
    if bias == "bull": score += 10
    if pa.get("sweep"): score += 10
    if pa.get("pullback"): score += 10
    if pa.get("range_retest"): score += 10
    if pa.get("ob") != "none": score += 10
    if pa.get("fvg") != "none": score += 10
    if htf.get("15m") in ("bull_engulf", "hammer"): score += 10
    if htf.get("1h") in ("bull_engulf", "hammer"): score += 10
    if spr_ok: score += 5
    if fund_ok: score += 5
    return min(score, 100)

# ----------------- BUILD SIGNAL v2 -----------------
COOLDOWN: Dict[str, int] = {}
def cooldown_ok(sym: str) -> bool:
    return int(time.time()) - COOLDOWN.get(sym, 0) > 1800
def set_cooldown(sym: str): COOLDOWN[sym] = int(time.time())

async def build_signal_v2(symbol: str, mode: str) -> Dict[str, Any]:
    mode = mode.lower()
    if not cooldown_ok(symbol): return {"ok": False, "reason": "cooldown"}
    if news_block(): return {"ok": False, "reason": "news_block"}
    if not killzone_active(mode): return {"ok": False, "reason": "killzone"}

    df1m = await fetch_ohlcv(symbol, "1m", 200)
    if df1m.empty: return {"ok": False, "reason": "data"}
    df1d = await fetch_ohlcv(symbol, "1d", 15)
    adr_val = adr(df1d)

    # ADR filter
    today_range = (df1d["high"].iloc[-1] - df1d["low"].iloc[-1]) / df1d["close"].iloc[-1] * 100
    if today_range > adr_val * 0.7: return {"ok": False, "reason": "adr_exhaust"}

    bias = ssl_hybrid_bias(symbol)
    if bias == "none": return {"ok": False, "reason": "bias_none"}

    htf = {"15m": detect_pattern(await fetch_ohlcv(symbol, "15m", 50)),
           "1h": detect_pattern(await fetch_ohlcv(symbol, "1h", 50))}

    pa = {"sweep": liquidity_sweep(df1m),
          "pullback": ema21_pullback(df1m),
          "range_retest": range_break_retest(df1m),
          "ob": detect_order_block(df1m),
          "fvg": detect_fvg(df1m)}

    spr_ok = await fetch_spread(symbol) < 0.06
    fund_ok = abs(await fetch_funding(symbol)) < 0.02

    score = calc_score_v2(symbol, df1m, htf, pa, spr_ok, fund_ok, bias)
    if score < SCORE_MIN: return {"ok": False, "reason": "low_score"}

    entry = float(df1m["close"].iloc[-1])
    tpsl = await smart_tp_sl(symbol, entry, "BUY", mode, df1m)

    set_cooldown(symbol)
    return {"ok": True, "symbol": symbol, "mode": mode, "side": "BUY",
            "entry": entry, "tp": tpsl["tp"], "sl": tpsl["sl"],
            "score": score, "htf": htf, "pa": pa,
            "time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}

# ----------------- RUNNER -----------------
_scan_sem = asyncio.Semaphore(MAX_CONCURRENT_SCANS)
async def scan_coin(sym: str, mode: str) -> Dict:
    async with _scan_sem:
        try: return await build_signal_v2(sym, mode)
        except Exception as e:
            logger.error("scan %s %s: %s", sym, mode, e)
            return {"ok": False, "error": str(e)}

async def run_mode(mode: str) -> List[Dict]:
    tasks = [scan_coin(sym, mode) for sym in COIN_LIST]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r.get("ok")]
    logger.info("mode=%s signals=%d", mode, len(valid))
    return valid

async def run_all_modes() -> Dict[str, List[Dict]]:
    return {m: await run_mode(m) for m in ("quick", "mid", "trend")}

# ----------------- TG / LOG -----------------
async def send_tg_async(text: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID): return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(url, json=payload)
    except Exception as e: logger.warning("tg %s", e)

def log_signal(sig: Dict):
    os.makedirs("logs", exist_ok=True)
    with open("logs/trades.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sig["time_utc"], sig["symbol"], sig["side"], sig["entry"],
                         sig["tp"], sig["sl"], sig["score"], sig["mode"]])

# ----------------- FORMATTER -----------------
def format_signal(sig: Dict, no: int) -> str:
    return f"""🔥 {sig['side']} {sig['mode'].upper()} #{no}
Pair: {sig['symbol']}
Score: {sig['score']}
Entry: {sig['entry']}
TP: {sig['tp']}
SL: {sig['sl']}
UTC: {sig['time_utc']}"""

# ----------------- EXPORT -----------------
__all__ = ["run_all_modes", "multi_override_watch", "format_signal", "send_tg_async", "log_signal"]
