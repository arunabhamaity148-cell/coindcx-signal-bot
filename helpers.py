# helpers.py — PATCHED with OpenAI verifier integration
# (Includes 58-logic scoring + decision layer)
import asyncio
import math
import time
import statistics
from datetime import datetime
import os

# import OpenAI verifier
from helpers_openai import call_openai_decision  # make sure helpers_openai.py is in same dir

# -----------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------
MODES = ["quick", "mid", "trend"]
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "30"))
BTC_STABILITY_THRESHOLD = float(os.getenv("BTC_STABILITY_THRESHOLD", "1.0"))
RECHECK_DELAY = 30
MAX_SCORE = 100

MODE_THRESHOLDS = {
    "quick": 55,
    "mid": 62,
    "trend": 70,
}

TP_RULES = {
    "quick": (1.2, 1.6),
    "mid": (1.8, 2.4),
    "trend": (2.5, 3.5),
}

SL_RULES = {
    "quick": (0.5, 0.8),
    "mid": (0.9, 1.2),
    "trend": (1.2, 1.8),
}

last_signal_time = {}

def is_cooldown(symbol):
    if symbol not in last_signal_time:
        return False
    diff = time.time() - last_signal_time[symbol]
    return diff < (COOLDOWN_MINUTES * 60)

def update_cooldown(symbol):
    last_signal_time[symbol] = time.time()

def ema(values, length):
    if not values or len(values) < length:
        return None
    k = 2 / (length + 1)
    ema_val = float(values[0])
    for v in values[1:]:
        ema_val = (float(v) * k) + (ema_val * (1 - k))
    return ema_val

def get_closes(klines):
    return [float(k[4]) for k in klines]

# robust btc_calm_check (keeps fail-safe)
import aiohttp
async def btc_calm_check(session):
    url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
    try:
        async with session.get(url, timeout=5) as r:
            if r.status != 200:
                return True
            d = await r.json()
            if not d or "priceChangePercent" not in d:
                return True
            try:
                change = abs(float(d.get("priceChangePercent", 0.0)))
            except:
                return True
            return change < BTC_STABILITY_THRESHOLD
    except Exception:
        return True

# --------------------------
# (All 58 logic functions)
# For brevity here: include all previously provided logic functions exactly as before.
# I'll paste them in full so the file is self-contained.
# --------------------------

def HTF_EMA_1h_15m(data):
    try:
        return 2 if data.get("ema_15m", 0) > data.get("ema_1h", 0) else 0
    except:
        return 0

def HTF_EMA_1h_4h(data):
    try:
        return 2 if data.get("ema_1h", 0) > data.get("ema_4h", 0) else 0
    except:
        return 0

def HTF_EMA_1h_8h(data):
    try:
        return 2 if data.get("ema_4h", 0) > data.get("ema_8h", 0) else 0
    except:
        return 0

def HTF_Structure_Break(data):
    return 3 if data.get("trend") == "bull_break" else 0

def HTF_Structure_Reject(data):
    return 3 if data.get("trend") == "reject" else 0

def Imbalance_FVG(data):
    return 2 if data.get("fvg") else 0

def Trend_Continuation(data):
    return 3 if data.get("trend_strength", 0) > 0.7 else 0

def Trend_Exhaustion(data):
    return 1 if data.get("exhaustion") else 0

def Micro_Pullback(data):
    return 2 if data.get("micro_pb") else 0

def Wick_Strength(data):
    try:
        return 2 if float(data.get("wick_ratio", 0)) > 1.5 else 0
    except:
        return 0

def Sweep_Reversal(data):
    return 3 if data.get("liquidation_sweep") else 0

def Vol_Sweep_1m(data):
    try:
        return 2 if float(data.get("vol_1m", 0)) > float(data.get("vol_5m", 0)) else 0
    except:
        return 0

def Vol_Sweep_5m(data):
    try:
        return 2 if float(data.get("vol_5m", 0)) > float(data.get("vol_1m", 0)) else 0
    except:
        return 0

def Delta_Divergence_1m(data):
    return 2 if data.get("delta_1m") else 0

def Delta_Divergence_HTF(data):
    return 2 if data.get("delta_htf") else 0

def Iceberg_1m(data):
    return 3 if data.get("iceberg_1m") else 0

def Iceberg_v2(data):
    return 3 if data.get("iceberg_v2") else 0

def Orderbook_Wall_Shift(data):
    return 2 if data.get("wall_shift") else 0

def Liquidity_Wall(data):
    return 2 if data.get("liq_wall") else 0

def Liquidity_Bend(data):
    return 2 if data.get("liq_bend") else 0

def ADR_DayRange(data):
    return 1 if data.get("adr_ok") else 0

def ATR_Expansion(data):
    return 1 if data.get("atr_expanding") else 0

def Phase_Shift(data):
    return 2 if data.get("phase_shift") else 0

def Price_Compression(data):
    return 1 if data.get("compression") else 0

def Speed_Imbalance(data):
    return 2 if data.get("speed_imbalance") else 0

def Taker_Pressure(data):
    return 2 if data.get("taker_pressure") else 0

def HTF_Volume_Imprint(data):
    return 2 if data.get("vol_imprint") else 0

def Tiny_Cluster_Imprint(data):
    return 1 if data.get("cluster_tiny") else 0

def Absorption(data):
    return 2 if data.get("absorption") else 0

def Recent_Weakness(data):
    return 1 if data.get("weakness") else 0

def Spread_Snap_0_5s(data):
    return 2 if data.get("spread_snap_05") else 0

def Spread_Snap_0_25s(data):
    return 2 if data.get("spread_snap_025") else 0

def Tight_Spread_Filter(data):
    try:
        return 1 if float(data.get("spread", 999)) < 0.03 else 0
    except:
        return 0

def Spread_Safety(data):
    try:
        return 1 if float(data.get("spread", 999)) < 0.05 else 0
    except:
        return 0

def BE_SL_AutoLock(data):
    return 1 if data.get("be_lock") else 0

def Liquidation_Distance(data):
    try:
        return 2 if float(data.get("liq_dist", 999)) < 0.5 else 0
    except:
        return 0

def Kill_Zone_5m(data):
    return 1 if data.get("kill_5m") else 0

def Kill_Zone_HTF(data):
    return 1 if data.get("kill_htf") else 0

def Kill_Switch_Fast(data):
    return -10 if data.get("kill_fast") else 0

def Kill_Switch_Primary(data):
    return -25 if data.get("kill_primary") else 0

def News_Guard(data):
    return -5 if data.get("news_risk") else 0

def Recheck_30s(data):
    return 1 if data.get("recheck_ok") else 0

def Partial_Exit_Logic(data):
    return 1

def BTC_Risk_Filter_L1(data):
    return -8 if not data.get("btc_calm", True) else 2

def BTC_Risk_Filter_L2(data):
    return -15 if data.get("btc_trending_fast") else 2

def BTC_Funding_OI_Combo(data):
    return 2 if data.get("funding_oi_combo") else 0

def Funding_Extreme(data):
    return -2 if data.get("funding_extreme") else 0

def Funding_Delta_Speed(data):
    return 1 if data.get("funding_delta") else 0

def Funding_Arbitrage(data):
    return 2 if data.get("arb_opportunity") else 0

def OI_Spike_5pct(data):
    return 2 if data.get("oi_spike") else 0

def OI_Spike_Sustained(data):
    return 2 if data.get("oi_sustained") else 0

def ETH_BTC_Beta_Divergence(data):
    return 1 if data.get("beta_div") else 0

def Options_Gamma_Flip(data):
    return 3 if data.get("gamma_flip") else 0

def Heatmap_Sweep(data):
    return 2 if data.get("heat_sweep") else 0

def Micro_Slip(data):
    return -1 if data.get("slippage") else 0

def Order_Block(data):
    return 2 if data.get("orderblock") else 0

def Score_Normalization(score):
    if score < 0:
        return 0
    return max(0, min(int(score), MAX_SCORE))

def Final_Signal_Score(data):
    total = sum([
        HTF_EMA_1h_15m(data),
        HTF_EMA_1h_4h(data),
        HTF_EMA_1h_8h(data),
        HTF_Structure_Break(data),
        HTF_Structure_Reject(data),
        Imbalance_FVG(data),
        Trend_Continuation(data),
        Trend_Exhaustion(data),
        Micro_Pullback(data),
        Wick_Strength(data),
        Sweep_Reversal(data),
        Vol_Sweep_1m(data),
        Vol_Sweep_5m(data),
        Delta_Divergence_1m(data),
        Delta_Divergence_HTF(data),
        Iceberg_1m(data),
        Iceberg_v2(data),
        Orderbook_Wall_Shift(data),
        Liquidity_Wall(data),
        Liquidity_Bend(data),
        ADR_DayRange(data),
        ATR_Expansion(data),
        Phase_Shift(data),
        Price_Compression(data),
        Speed_Imbalance(data),
        Taker_Pressure(data),
        HTF_Volume_Imprint(data),
        Tiny_Cluster_Imprint(data),
        Absorption(data),
        Recent_Weakness(data),
        Spread_Snap_0_5s(data),
        Spread_Snap_0_25s(data),
        Tight_Spread_Filter(data),
        Spread_Safety(data),
        BE_SL_AutoLock(data),
        Liquidation_Distance(data),
        Kill_Zone_5m(data),
        Kill_Zone_HTF(data),
        Kill_Switch_Fast(data),
        Kill_Switch_Primary(data),
        News_Guard(data),
        Recheck_30s(data),
        Partial_Exit_Logic(data),
        BTC_Risk_Filter_L1(data),
        BTC_Risk_Filter_L2(data),
        BTC_Funding_OI_Combo(data),
        Funding_Extreme(data),
        Funding_Delta_Speed(data),
        Funding_Arbitrage(data),
        OI_Spike_5pct(data),
        OI_Spike_Sustained(data),
        ETH_BTC_Beta_Divergence(data),
        Options_Gamma_Flip(data),
        Heatmap_Sweep(data),
        Micro_Slip(data),
        Order_Block(data),
    ])
    return Score_Normalization(total)

def calc_tp_sl(price, mode):
    tp_min, _ = TP_RULES.get(mode, TP_RULES["mid"])
    sl_min, _ = SL_RULES.get(mode, SL_RULES["mid"])
    tp = price * (1 + tp_min/100)
    sl = price * (1 - sl_min/100)
    return round(tp, 6), round(sl, 6)

def decide_signal(symbol, mode, price, score):
    if mode not in MODE_THRESHOLDS:
        return None
    if score < MODE_THRESHOLDS[mode]:
        return None
    if is_cooldown(symbol):
        return None
    tp, sl = calc_tp_sl(price, mode)
    update_cooldown(symbol)
    return {
        "symbol": symbol,
        "mode": mode,
        "price": price,
        "score": score,
        "tp": tp,
        "sl": sl,
        "leverage": 50,
        "cooldown_min": COOLDOWN_MINUTES
    }

def format_signal(sig):
    return f"""
🔥 <b>{sig['mode'].upper()} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Price:</b> {sig['price']}
<b>Score:</b> {sig['score']} / {MAX_SCORE}

🎯 <b>TP:</b>
<code>{sig['tp']}</code>

🛑 <b>SL:</b>
<code>{sig['sl']}</code>

⚡ Leverage: {sig.get('leverage', 50)}x
⏱️ Cooldown: {sig.get('cooldown_min', COOLDOWN_MINUTES)} minutes
📊 Mode: {sig['mode'].upper()}

#HybridAI #ArunSystem
"""

# ---------- NEW: async process_data_with_ai ----------
async def process_data_with_ai(symbol, mode, live_data):
    """
    Asynchronous processor: local scoring -> local decision -> OpenAI verifier -> final decision
    Returns decision dict or None
    """
    # ensure btc_calm present
    if "btc_calm" not in live_data:
        live_data["btc_calm"] = True

    # local score + local decision
    score = Final_Signal_Score(live_data)
    local_decision = decide_signal(symbol, mode, live_data.get("price", 0.0), score)
    if not local_decision:
        return None

    # prepare compact summary for OpenAI (keep payload small)
    live_summary = {
        "trend": live_data.get("trend"),
        "vol_1m": live_data.get("vol_1m"),
        "vol_5m": live_data.get("vol_5m"),
        "oi_spike": live_data.get("oi_spike"),
        "funding_extreme": live_data.get("funding_extreme"),
        "btc_calm": live_data.get("btc_calm", True),
        "score": score
    }

    # call OpenAI verifier (async) -> fail-open default
    ai_resp = await call_openai_decision(symbol, mode, local_decision["price"], score, live_summary)

    # if OpenAI explicitly blocks -> do not send signal
    if not ai_resp.get("approve", True):
        # optionally: you can log ai_resp["note"]
        return None

    # override TP/SL if OpenAI supplied custom ones
    if ai_resp.get("tp") is not None:
        local_decision["tp"] = round(float(ai_resp["tp"]), 6)
    if ai_resp.get("sl") is not None:
        local_decision["sl"] = round(float(ai_resp["sl"]), 6)

    local_decision["ai_note"] = ai_resp.get("note", "")
    return local_decision

# keep old sync function for backwards-compat
def process_data(symbol, mode, live_data):
    return asyncio.get_event_loop().run_until_complete(process_data_with_ai(symbol, mode, live_data))