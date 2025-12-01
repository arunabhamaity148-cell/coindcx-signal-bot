# helpers.py — AI-only scoring helper (detailed prompt + utilities)

import time, html, json
from typing import Dict

def now_ts() -> int:
    return int(time.time())

def human_time(ts: int = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(s) -> str:
    return html.escape("" if s is None else str(s))

# TP/SL rules (percent)
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

# -----------------------------------------------------------------------------
# 58 logics short descriptions (these will be embedded into the AI prompt)
# -----------------------------------------------------------------------------
LOGIC_DETAILS = {
    "HTF_EMA_1h_15m": "Higher timeframe trend alignment: 1h EMA vs 15m EMA alignment and slope",
    "HTF_EMA_1h_4h": "Multi-hour EMA alignment: 1h EMA vs 4h EMA confluence",
    "HTF_EMA_1h_8h": "Longer HTF EMA alignment: 1h vs 8h EMA confirmation",
    "HTF_Structure_Break": "Clear breakout of recent HTF support or resistance with confirm",
    "HTF_Structure_Reject": "Rejection / false-break at HTF structure",
    "Imbalance_FVG": "Fair-value-gap / imbalance area presence near entry",
    "Trend_Continuation": "Signs that current trend is continuing (HTF + momentum)",
    "Trend_Exhaustion": "Signs of exhaustion at push extremes (weakening volume)",
    "Micro_Pullback": "Small retracement into value area for better entry",
    "Wick_Strength": "Presence of strong wick rejection candles",
    "Sweep_Reversal": "Liquidity sweep and reversal pattern",
    "Vol_Sweep_1m": "1-minute volume spike/sweep vs recent average",
    "Vol_Sweep_5m": "5-minute volume spike vs baseline",
    "Delta_Divergence_1m": "Trade delta divergence vs price on 1m",
    "Delta_Divergence_HTF": "Delta divergence on higher TFs (5m/15m/1h)",
    "Iceberg_1m": "Iceberg / hidden large order signatures on 1m",
    "Iceberg_v2": "Enhanced iceberg detection with orderbook changes",
    "Orderbook_Wall_Shift": "Significant orderbook wall shift (buy/sell imbalance)",
    "Liquidity_Wall": "Static or moving liquidity wall near price",
    "Liquidity_Bend": "Liquidity bending/flow indicating rotation",
    "ADR_DayRange": "Price relative to daily range (ADR)",
    "ATR_Expansion": "ATR expanding (volatility breakout confirmation)",
    "Phase_Shift": "Market phase shift from compression to expansion",
    "Price_Compression": "Compression of price (tight range) pre-breakout",
    "Speed_Imbalance": "Rapid price changes vs volume suggesting imbalance",
    "Taker_Pressure": "Taker-side dominance in recent trades",
    "HTF_Volume_Imprint": "Higher timeframe volume imprint indicating interest",
    "Tiny_Cluster_Imprint": "Micro cluster volume footprints under price",
    "Absorption": "Absorption of aggressive orders by opposing liquidity",
    "Recent_Weakness": "Recent inability to push higher/lower (momentum decay)",
    "Spread_Snap_0_5s": "Short snapshot spread observation (0.5s) - microstructure",
    "Spread_Snap_0_25s": "Shorter snapshot spread (0.25s) reliability check",
    "Tight_Spread_Filter": "Is spread sufficiently tight for safe execution",
    "Spread_Safety": "Spread vs typical baseline (safety filter)",
    "BE_SL_AutoLock": "Auto-adjust stop/BE logic to protect position",
    "Liquidation_Distance": "Distance to major liquidation clusters",
    "Kill_Zone_5m": "Time-based killzone short-term (5m)",
    "Kill_Zone_HTF": "Longer timeframe killzone (news/market open/close)",
    "Kill_Switch_Fast": "Immediate kill switch conditions (extreme events)",
    "Kill_Switch_Primary": "Primary safety kill switch rules",
    "News_Guard": "Major news/announcements guard",
    "30s_Recheck_Loop": "Short re-check before sending signal (30s)",
    "Partial_Exit_Logic": "Rules for partial profit taking",
    "BTC_Risk_Filter_L1": "BTC calm/volatility filter level 1",
    "BTC_Risk_FILTER_L2": "BTC risk filter level 2 (stricter)",
    "BTC_Funding_OI_Combo": "Combination of BTC funding and OI dynamics",
    "Funding_Extreme": "Extreme funding rate signals",
    "Funding_Delta_Speed": "Speed of change of funding rate",
    "Funding_Arbitrage": "Opportunities / risk from funding arbitrage",
    "OI_Spike_5pct": "Open interest spike >5% detection",
    "OI_Spike_Sustained": "Sustained OI increase over multiple intervals",
    "ETH_BTC_Beta_Divergence": "ETH/BTC relative beta divergence",
    "Options_Gamma_Flip": "Options-gamma flip signals (derivatives)",
    "Heatmap_Sweep": "Heatmap sweep detection on orderflow/volume",
    "Micro_Slip": "Micro execution slippage indicator",
    "Order_Block": "Order block detection area",
    "Score_Normalization": "Normalization step for combining logic scores",
    "Final_Signal_Score": "Final combined AI decision and confidence"
}

# Build a compact description mapping (list) for the prompt
def get_logic_descriptions_text() -> str:
    parts = []
    for k, v in LOGIC_DETAILS.items():
        parts.append(f"{k}: {v}")
    return "\\n".join(parts)


# Build AI prompt (full, with logic descriptions + metrics)
def build_ai_prompt(symbol: str, price: float, metrics: Dict, prefs: Dict) -> str:
    """
    Build a robust prompt that instructs the model to evaluate the 58 logics
    and return a strict JSON: {"score":int,"mode":"quick|mid|trend","reason":"..."}.
    """
    metrics_json = json.dumps(metrics, default=str)
    prefs_json = json.dumps(prefs, default=str)
    logic_text = get_logic_descriptions_text()

    prompt = f\"\"\"You are an expert crypto futures signal scorer.

Evaluate the following symbol using the 58 signal logics described below.
Return ONLY a single-line JSON object with keys:
- score: integer 0-100 (higher = better)
- mode: one of ["quick","mid","trend"]
- reason: 4-14 words explanation (concise)

Symbol: {symbol}
Price: {price}
Metrics: {metrics_json}

Preferences: {prefs_json}

Logic Descriptions (short):
{logic_text}

Rules:
1) If BTC is not calm (as per BTC_Risk_Filter) return score 0.
2) Use HTF alignment as primary filter; volume/ATR/oi support increases score.
3) Penalize high spread / low liquidity.
4) Favor quick for small, momentum scalp; mid for sustained momentum; trend for multi-hour alignment.
5) Output strict valid JSON only, nothing else.

Now evaluate and output a single-line JSON object.
\"\"\"
    return prompt