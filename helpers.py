# helpers.py — AI-Only Logic Version
import time, html, json

def now_ts():
    return int(time.time())

def human_time(ts=None):
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def esc(s):
    return html.escape("" if s is None else str(s))


# ----------------------------------------------------
# TP/SL RULES (mode-based)
# ----------------------------------------------------
TP_SL_RULES = {
    "quick": {"tp_pct": 1.6, "sl_pct": 1.0},
    "mid":   {"tp_pct": 2.0, "sl_pct": 1.0},
    "trend": {"tp_pct": 4.0, "sl_pct": 1.5},
}

def calc_tp_sl(price, mode):
    r = TP_SL_RULES.get(mode, {"tp_pct": 2.0, "sl_pct": 1.0})
    tp = price * (1 + r["tp_pct"]/100.0)
    sl = price * (1 - r["sl_pct"]/100.0)
    return round(tp, 8), round(sl, 8)


# ----------------------------------------------------
# 58 Logic Names → Prompt-এ যাবে (AI evaluation)
# ----------------------------------------------------
LOGIC_NAMES = [
    "HTF_EMA_1h_15m",
    "HTF_EMA_1h_4h",
    "HTF_EMA_1h_8h",
    "HTF_Structure_Break",
    "HTF_Structure_Reject",
    "Imbalance_FVG",
    "Trend_Continuation",
    "Trend_Exhaustion",
    "Micro_Pullback",
    "Wick_Strength",
    "Sweep_Reversal",
    "Vol_Sweep_1m",
    "Vol_Sweep_5m",
    "Delta_Divergence_1m",
    "Delta_Divergence_HTF",
    "Iceberg_1m",
    "Iceberg_v2",
    "Orderbook_Wall_Shift",
    "Liquidity_Wall",
    "Liquidity_Bend",
    "ADR_DayRange",
    "ATR_Expansion",
    "Phase_Shift",
    "Price_Compression",
    "Speed_Imbalance",
    "Taker_Pressure",
    "HTF_Volume_Imprint",
    "Tiny_Cluster_Imprint",
    "Absorption",
    "Recent_Weakness",
    "Spread_Snap_0_5s",
    "Spread_Snap_0_25s",
    "Tight_Spread_Filter",
    "Spread_Safety",
    "BE_SL_AutoLock",
    "Liquidation_Distance",
    "Kill_Zone_5m",
    "Kill_Zone_HTF",
    "Kill_Switch_Fast",
    "Kill_Switch_Primary",
    "News_Guard",
    "30s_Recheck_Loop",
    "Partial_Exit_Logic",
    "BTC_Risk_Filter_L1",
    "BTC_Risk_Filter_L2",
    "BTC_Funding_OI_Combo",
    "Funding_Extreme",
    "Funding_Delta_Speed",
    "Funding_Arbitrage",
    "OI_Spike_5pct",
    "OI_Spike_Sustained",
    "ETH_BTC_Beta_Divergence",
    "Options_Gamma_Flip",
    "Heatmap_Sweep",
    "Micro_Slip",
    "Order_Block",
    "Score_Normalization",
    "Final_Signal_Score"
]


# ----------------------------------------------------
# Build OpenAI Prompt for AI Scoring
# ----------------------------------------------------
def build_ai_prompt(symbol: str, price: float, metrics: dict, prefs: dict) -> str:
    """
    The AI will evaluate:
    - Trend (multi-TF)
    - Volume spikes
    - Volatility
    - Orderflow signs
    - Spread safety
    - BTC calm check
    - 58 logic total
    """

    logic_text = ", ".join(LOGIC_NAMES)
    metrics_json = json.dumps(metrics)
    prefs_json = json.dumps(prefs)

    prompt = f"""
You are an advanced crypto futures signal evaluator.
Evaluate the symbol using 58 trading logics:

Logic List = [{logic_text}]

Input Data:
symbol = {symbol}
price = {price}
metrics = {metrics_json}

Preferences:
{prefs_json}

Return STRICT JSON ONLY, example:
{{"score":85, "mode":"quick", "reason":"HTF aligned + volume surge"}}

Rules:
- score = 0 to 100
- mode ∈ ["quick","mid","trend"]
- reason = 4–12 words
- if data weak or risky → score < 78
- if strong HTF alignment + volume + volatility control → score > 78
- NEVER output anything outside JSON
"""
    return prompt.strip()