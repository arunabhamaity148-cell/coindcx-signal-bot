# scorer.py – final clean
import numpy as np, logging, asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Deque
from helpers import build_ohlcv_from_trades, calc_rsi, calc_atr, calc_vwap_from_trades, orderflow_metrics, btc_calm_check
log = logging.getLogger("scorer")

STRAT_MIN = {"QUICK": 6.8, "MID": 7.4, "TREND": 8.0}

# ---------- MTF ----------
def _ema_state(sr):
    if sr is None or len(sr) < 60: return None
    c = sr.ewm(span=20).mean().iloc[-1]
    l = sr.ewm(span=50).mean().iloc[-1]
    return 1 if c > l else -1

def _mtf(o1, o5, o15, o60):
    bull = bear = 0
    for df in (o1, o5, o15, o60):
        st = _ema_state(df["close"] if df is not None else None)
        if st == 1: bull += 1
        elif st == -1: bear += 1
    return {"bull": bull, "bear": bear}

# ---------- LOGIC ----------
def _logic(p: Dict[str, Any]) -> Dict[str, int]:
    last, vwap, rsi, atr, imb, depth, spread, mom1, mom5, mtf = (
        p["last"], p["vwap"], p["rsi"], p["atr"], p["imb"],
        p["depth"], p["spread"], p["mom1"], p["mom5"], p["mtf"]
    )
    return {
        "mtf_bull": int(mtf["bull"] >= 3),
        "mtf_bear": int(mtf["bear"] >= 3),
        "atr_stable": int(atr < last * 0.002),
        "vwap_close": int(abs(last - vwap) / vwap < 0.002),
        "mom1": int(abs(mom1) > 0.0008),
        "mom5": int(abs(mom5) > 0.0015),
        "imbalance": int(abs(imb) > 0.05),
        "spread_ok": int(spread < 0.32),
        "depth_ok": int(depth > 30000),
        "rsi_mid": int(45 <= rsi <= 55),
        "rsi_flip": int(rsi > 50 and mom1 > 0),
        "rsi_div": int((mom1 > 0 and rsi < 50) or (mom1 < 0 and rsi > 50)),
        "vwap_small_dev": int(abs(last - vwap) / vwap < 0.0025),
        "vwap_momentum": int(last > vwap and mom1 > 0),
        "imb_buy": int(imb > 0.05),
        "imb_sell": int(imb < -0.05),
        "low_atr": int(atr < last * 0.0018),
    }

def _score(l: Dict[str, int]) -> float:
    return float(sum(l.values()))

# ---------- ENHANCE ----------
def _enhance(sig: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder – add your confidence/level/volume modules here
    return sig

# ---------- PUBLIC ----------
async def compute_signal(
    sym: str,
    strat: str,
    trade_buffer: Dict[str, Deque[Dict[str, Any]]],
    ob_cache: Dict[str, Dict[str, float]]
) -> Optional[Dict[str, Any]]:
    if sym != "BTCUSDT":
        if not await btc_calm_check(trade_buffer):
            return None

    o1 = await build_ohlcv_from_trades(sym, "1min", 200, buffer_dict=trade_buffer)
    if o1 is None or len(o1) < 60:
        return None
    o5 = await build_ohlcv_from_trades(sym, "5min", 200, buffer_dict=trade_buffer)
    o15 = await build_ohlcv_from_trades(sym, "15min", 200, buffer_dict=trade_buffer)
    o60 = await build_ohlcv_from_trades(sym, "60min", 200, buffer_dict=trade_buffer)

    mtf = _mtf(o1, o5, o15, o60)
    last = float(o1["close"].iloc[-1])
    mom1 = (last - float(o1["close"].iloc[-2])) / last
    mom5 = (last - float(o5["close"].iloc[-2])) / last if o5 is not None and len(o5) > 2 else 0.0
    rsi = float(calc_rsi(o1["close"]).iloc[-1])
    atr = float(calc_atr(o1))
    vwap = await calc_vwap_from_trades(sym, buffer_dict=trade_buffer)
    if vwap is None:
        return None
    flow = await orderflow_metrics(sym, buffer_dict=trade_buffer, ob_cache=ob_cache)
    if not flow:
        return None

    params = {
        "last": last, "vwap": vwap, "rsi": rsi, "atr": atr,
        "imb": flow["imbalance"], "depth": flow["depth_usd"],
        "spread": flow["spread_pct"], "mom1": mom1, "mom5": mom5, "mtf": mtf
    }
    logic = _logic(params)
    score = _score(logic)
    if score < STRAT_MIN.get(strat, 0):
        return None

    side = "long" if params["imb"] > 0 else "short"
    sig = {
        "symbol": sym, "side": side, "score": score, "last": last,
        "strategy": strat, "logic": logic, "passed": [k for k, v in logic.items() if v],
        "timestamp": datetime.utcnow().isoformat()
    }
    return _enhance(sig, params)
