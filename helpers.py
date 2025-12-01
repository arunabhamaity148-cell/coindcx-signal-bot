# helpers.py — FINAL (indicators, prompt builder, TP/SL, cache)
import time, html, json, hashlib
from typing import Dict, List

# -------------------------
# Time / small utils
# -------------------------
def now_ts() -> int:
    return int(time.time())

def human_time(ts: int = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(x) -> str:
    return html.escape("" if x is None else str(x))

# -------------------------
# TP/SL (mode-based)
# -------------------------
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

# -------------------------
# Simple in-memory TTL cache
# -------------------------
class SimpleCache:
    def __init__(self):
        self.store = {}  # key -> (expiry, value)

    def get(self, key):
        rec = self.store.get(key)
        if not rec:
            return None
        exp, val = rec
        if exp < time.time():
            del self.store[key]
            return None
        return val

    def set(self, key, value, ttl_seconds: int):
        self.store[key] = (time.time() + ttl_seconds, value)

    def make_key(self, *parts) -> str:
        raw = "|".join([str(p) for p in parts])
        return hashlib.sha256(raw.encode()).hexdigest()

CACHE = SimpleCache()

# -------------------------
# Basic indicators (lightweight)
# -------------------------
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
        high = ohlc[i][2]
        low = ohlc[i][3]
        prev_close = ohlc[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return sum(trs) / len(trs)
    # EMA of TRs
    seed = sum(trs[:period]) / period
    atr_val = seed
    k = 2 / (period + 1)
    for tr in trs[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val

def rsi_from_closes(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        if ch > 0:
            gains.append(ch)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(ch))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------
# 58 logic short descriptions (for prompt)
# -------------------------
LOGIC_DETAILS = {
    "HTF_EMA_1h_15m": "1h vs 15m EMA alignment and slope",
    "HTF_EMA_1h_4h": "1h vs 4h EMA confluence",
    "HTF_EMA_1h_8h": "1h vs 8h EMA confluence",
    "HTF_Structure_Break": "Breakout of HTF support/resistance",
    "HTF_Structure_Reject": "HTF rejection at structure",
    "Imbalance_FVG": "Fair-value gap / imbalance presence near price",
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
    "BTC_Risk_Filter_L2": "Stricter BTC risk filter L2",
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
    parts = []
    for k, v in LOGIC_DETAILS.items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)

# -------------------------
# Build AI prompt
# -------------------------
def build_ai_prompt(symbol: str, price: float, metrics: Dict, prefs: Dict) -> str:
    # Keep arrays small in prompt (trim to last 20)
    small = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            small[k] = v[-20:]
        else:
            small[k] = v
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

Logic descriptions (short):
{logic_text}

Rules:
1) If BTC calm filter fails -> return score 0.
2) Use HTF alignment + volume + ATR to increase score.
3) Penalize wide spread / low liquidity.
4) Choose mode by horizon: quick(mid scalp), mid(momentum), trend(multi-hour).
5) Output strict JSON only.
"""
    return prompt.strip()