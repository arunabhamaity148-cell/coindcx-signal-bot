# helpers.py — FINAL (58 logic desc + strict filters + score builder)
import time, html, json, hashlib, logging
from typing import Dict, List
from datetime import datetime

# ---------- time ----------
def now_ts() -> int:
    return int(time.time())

def human_time(ts: int = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(x) -> str:
    return html.escape("" if x is None else str(x))

# ---------- TP/SL ----------
TP_SL_RULES = {
    "quick": {"tp_pct": 1.6, "sl_pct": 1.0},
    "mid":   {"tp_pct": 2.0, "sl_pct": 1.0},
    "trend": {"tp_pct": 4.0, "sl_pct": 1.5},
}

def calc_tp_sl(price: float, mode: str):
    cfg = TP_SL_RULES.get(mode, {"tp_pct": 2.0, "sl_pct": 1.0})
    tp = price * (1 + cfg["tp_pct"]/100.0)
    sl = price * (1 - cfg["sl_pct"]/100.0)
    return round(tp, 8), round(sl, 8)

# ---------- TTL cache ----------
class SimpleCache:
    def __init__(self):
        self.store = {}
    def get(self, key):
        rec = self.store.get(key)
        if not rec:
            return None
        exp, val = rec
        if exp < time.time():
            try: del self.store[key]
            except: pass
            return None
        return val
    def set(self, key, value, ttl_seconds: int):
        self.store[key] = (time.time() + ttl_seconds, value)
    def make_key(self, *parts) -> str:
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# ---------- indicators ----------
def sma(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return sum(values) / len(values)
    return sum(values[-period:]) / period

def compute_ema_from_closes(closes: List[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return sma(closes, period)
    seed = sum(closes[:period]) / period
    ema_val = seed
    k = 2 / (period + 1)
    for price in closes[period:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def atr(ohlc: List[List[float]], period: int = 14) -> float:
    if len(ohlc) < 2:
        return 0.0
    trs = []
    for i in range(1, len(ohlc)):
        high, low, prev_close = ohlc[i][2], ohlc[i][3], ohlc[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    seed = sum(trs[:period]) / period
    atr_val = seed
    k = 2 / (period + 1)
    for tr in trs[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val

def rsi_from_closes(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0)); losses.append(abs(min(ch, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ---------- 58 logic short desc ----------
LOGIC_DESC = {
    "HTF_EMA_1h_15m": "1h vs 15m EMA alignment and slope",
    "HTF_EMA_1h_4h": "1h vs 4h EMA confluence",
    "HTF_EMA_1h_8h": "1h vs 8h EMA confluence",
    "HTF_Structure_Break": "Breakout of HTF support/resistance",
    "HTF_Structure_Reject": "HTF rejection at structure",
    "Imbalance_FVG": "Fair-value gap / imbalance near price",
    "Trend_Continuation": "Current trend likely to continue",
    "Trend_Exhaustion": "Momentum dying out at push extremes",
    "Micro_Pullback": "Small retracement into value for entry",
    "Wick_Strength": "Strong wick rejection candle present",
    "Sweep_Reversal": "Liquidity sweep then reversal",
    "Vol_Sweep_1m": "1m volume spike vs baseline",
    "Vol_Sweep_5m": "5m volume spike vs baseline",
    "Delta_Divergence_1m": "Trade-delta divergence on 1m",
    "Delta_Divergence_HTF": "Delta divergence on HTFs",
    "Iceberg_1m": "Hidden large order signatures",
    "Iceberg_v2": "Enhanced iceberg detection",
    "Orderbook_Wall_Shift": "Large wall shift in orderbook",
    "Liquidity_Wall": "Liquidity wall near price",
    "Liquidity_Bend": "Liquidity bending/rotation",
    "ADR_DayRange": "Price vs daily range (ADR)",
    "ATR_Expansion": "ATR expansion confirms breakout",
    "Phase_Shift": "Market shifting phase (comp->exp)",
    "Price_Compression": "Tight range pre-breakout",
    "Speed_Imbalance": "Rapid move vs volume imbalance",
    "Taker_Pressure": "Taker dominance in trades",
    "HTF_Volume_Imprint": "HTF volume footprint",
    "Tiny_Cluster_Imprint": "Micro cluster volume imprint",
    "Absorption": "Opposing liquidity absorbing aggression",
    "Recent_Weakness": "Recent push weakness",
    "Spread_Snap_0_5s": "0.5s spread snapshot",
    "Spread_Snap_0_25s": "0.25s spread snapshot",
    "Tight_Spread_Filter": "Is spread tight enough",
    "Spread_Safety": "Spread vs baseline safety",
    "BE_SL_AutoLock": "Auto-lock BE/SL rules",
    "Liquidation_Distance": "Distance to liquidation clusters",
    "Kill_Zone_5m": "5m time-based killzone",
    "Kill_Zone_HTF": "HTF killzone (news/market open)",
    "Kill_Switch_Fast": "Immediate kill switch conditions",
    "Kill_Switch_Primary": "Primary safety kill switch",
    "News_Guard": "Major news guard",
    "30s_Recheck_Loop": "30s quick recheck rule",
    "Partial_Exit_Logic": "Partial take-profit logic",
    "BTC_Risk_Filter_L1": "BTC calm / volatility filter L1",
    "BTC_Risk_FILTER_L2": "Stricter BTC risk filter L2",
    "BTC_Funding_OI_Combo": "BTC funding & OI combination",
    "Funding_Extreme": "Extreme funding rate signal",
    "Funding_Delta_Speed": "Speed of funding change",
    "Funding_Arbitrage": "Funding arbitrage signal",
    "OI_Spike_5pct": "OI spike >5% detection",
    "OI_Spike_Sustained": "Sustained OI increase",
    "ETH_BTC_Beta_Divergence": "ETH/BTC beta divergence",
    "Options_Gamma_Flip": "Options gamma flip signal",
    "Heatmap_Sweep": "Heatmap sweep on orderflow",
    "Micro_Slip": "Micro execution slippage",
    "Order_Block": "Order block detection",
    "Score_Normalization": "Normalization step for scores",
    "Final_Signal_Score": "Final combined AI decision/confidence"
}

def logic_descriptions_text() -> str:
    return "\n".join([f"{k}: {v}" for k, v in LOGIC_DESC.items()])

# ---------- strict filters ----------
def btc_calm_filter() -> bool:
    snap = CACHE.get("btc_snapshot")
    if not snap:
        return False
    atr_pct = snap["metrics"].get("atr_1h_pct", 999)
    spr_pct = snap["metrics"].get("spread_5m_pct", 999)
    return atr_pct < 1.2 and spr_pct < 0.05

def spread_filter(metrics: Dict) -> bool:
    return metrics.get("spread_pct", 999) < 0.08

def volume_spike_ok(metrics: Dict) -> bool:
    vol = metrics.get("vol_1m", 0)
    base = metrics.get("vol_1m_avg_20", 0) or 1
    return vol > base * 1.5

def htf_alignment_score(metrics: Dict) -> int:
    ema1h = metrics.get("ema_1h_50", 0)
    ema15 = metrics.get("ema_15m_50", 0)
    price = metrics["price"]
    if price > ema15 > ema1h:
        return 20
    if price < ema15 < ema1h:
        return 20
    return 0

def funding_oi_filter(symbol: str) -> bool:
    funding = CACHE.get(f"funding_{symbol}")
    oi = CACHE.get(f"oi_change_{symbol}")
    if funding and abs(funding) > 0.05:
        return False
    if oi and oi > 10:
        return False
    return True

def kill_switch() -> bool:
    btc_ret = CACHE.get("btc_5m_return")
    if btc_ret and btc_ret < -2.0:
        return True
    return False

# ---------- prompt + score builder ----------
def build_ai_prompt(symbol: str, price: float, metrics: Dict, prefs: Dict) -> str:
    # ---- strict filters ----
    if not btc_calm_filter():
        return json.dumps({"score": 0, "mode": "quick", "reason": "BTC volatile"})
    if not spread_filter(metrics):
        return json.dumps({"score": 0, "mode": "quick", "reason": "Spread wide"})
    if kill_switch():
        return json.dumps({"score": 0, "mode": "quick", "reason": "Kill-switch"})

    base = 0
    base += htf_alignment_score(metrics)
    if not volume_spike_ok(metrics):
        base = min(base, 60)
    if not funding_oi_filter(symbol):
        base = min(base, 50)

    small = {k: v[-20:] if isinstance(v, list) else v for k, v in metrics.items()}
    metrics_text = json.dumps(small, default=str)
    prefs_text = json.dumps(prefs, default=str)
    logic_text = logic_descriptions_text()

    prompt = f"""
You are an expert crypto futures signal scorer. Evaluate the symbol using 58 trading logics.

Return EXACT single-line JSON only with keys:
- score: integer 0-100
- mode: "quick"|"mid"|"trend"
- reason: 4-12 words concise

Symbol: {symbol}
Price: {price}
Metrics: {metrics_text}
Preferences: {prefs_text}

Base filter score so far: {base}

Logic descriptions (short):
{logic_text}

Rules:
1) Add 0-50 more points based on HTF alignment, volume, ATR, orderflow, imbalance, liquidation distance, etc.
2) If total >85 → mode "trend", >70 → "mid", else "quick"
3) Penalize wide spread, low liquidity, extreme funding/OI
4) Output strict JSON only.
"""
    return prompt.strip()
