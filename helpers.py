# helpers.py — FINAL HYBRID MIND (Part 1/2)

import asyncio, math, time, statistics
from datetime import datetime
import aiohttp

# -----------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------
MODES = ["quick", "mid", "trend"]

COOLDOWN_MINUTES = 30
BTC_STABILITY_THRESHOLD = 0.35   # % change allowed
RECHECK_DELAY = 30               # seconds
MAX_SCORE = 100

# -----------------------------------------
# FETCH DATA (Binance)
# -----------------------------------------
async def fetch_binance(session, url):
    try:
        async with session.get(url, timeout=5) as r:
            return await r.json()
    except Exception:
        return None

async def get_price(session, symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    data = await fetch_binance(session, url)
    try:
        return float(data["price"])
    except:
        return None

async def get_klines(session, symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = await fetch_binance(session, url)
    return data or []

# -----------------------------------------
# BASE INDICATORS
# -----------------------------------------
def ema(values, length):
    if len(values) < length:
        return None
    k = 2 / (length + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = (v * k) + (ema_val * (1 - k))
    return ema_val

def get_closes(klines):
    return [float(k[4]) for k in klines]

# -----------------------------------------
# LOGIC ENGINE (58 LOGICS)
# -----------------------------------------

# Each logic returns 0–3 score.

def HTF_EMA_1h_15m(data):
    return 2 if data["ema_15m"] > data["ema_1h"] else 0

def HTF_EMA_1h_4h(data):
    return 2 if data["ema_1h"] > data["ema_4h"] else 0

def HTF_EMA_1h_8h(data):
    return 2 if data["ema_4h"] > data["ema_8h"] else 0

def HTF_Structure_Break(data):
    return 3 if data["trend"] == "bull_break" else 0

def HTF_Structure_Reject(data):
    return 3 if data["trend"] == "reject" else 0

def Imbalance_FVG(data):
    return 2 if data["fvg"] else 0

def Trend_Continuation(data):
    return 3 if data["trend_strength"] > 0.7 else 0

def Trend_Exhaustion(data):
    return 1 if data["exhaustion"] else 0

def Micro_Pullback(data):
    return 2 if data["micro_pb"] else 0

def Wick_Strength(data):
    return 2 if data["wick_ratio"] > 1.5 else 0

def Sweep_Reversal(data):
    return 3 if data["liquidation_sweep"] else 0

def Vol_Sweep_1m(data):
    return 2 if data["vol_1m"] > data["vol_5m"] else 0

def Vol_Sweep_5m(data):
    return 2 if data["vol_5m"] > data["vol_1m"] else 0

def Delta_Divergence_1m(data):
    return 2 if data["delta_1m"] else 0

def Delta_Divergence_HTF(data):
    return 2 if data["delta_htf"] else 0

def Iceberg_1m(data):
    return 3 if data["iceberg_1m"] else 0

def Iceberg_v2(data):
    return 3 if data["iceberg_v2"] else 0

def Orderbook_Wall_Shift(data):
    return 2 if data["wall_shift"] else 0

def Liquidity_Wall(data):
    return 2 if data["liq_wall"] else 0

def Liquidity_Bend(data):
    return 2 if data["liq_bend"] else 0

def ADR_DayRange(data):
    return 1 if data["adr_ok"] else 0

def ATR_Expansion(data):
    return 1 if data["atr_expanding"] else 0

def Phase_Shift(data):
    return 2 if data["phase_shift"] else 0

def Price_Compression(data):
    return 1 if data["compression"] else 0

def Speed_Imbalance(data):
    return 2 if data["speed_imbalance"] else 0

def Taker_Pressure(data):
    return 2 if data["taker_pressure"] else 0

def HTF_Volume_Imprint(data):
    return 2 if data["vol_imprint"] else 0

def Tiny_Cluster_Imprint(data):
    return 1 if data["cluster_tiny"] else 0

def Absorption(data):
    return 2 if data["absorption"] else 0

def Recent_Weakness(data):
    return 1 if data["weakness"] else 0

def Spread_Snap_0_5s(data):
    return 2 if data["spread_snap_05"] else 0

def Spread_Snap_0_25s(data):
    return 2 if data["spread_snap_025"] else 0

def Tight_Spread_Filter(data):
    return 1 if data["spread"] < 0.03 else 0

def Spread_Safety(data):
    return 1 if data["spread"] < 0.05 else 0

def BE_SL_AutoLock(data):
    return 1 if data["be_lock"] else 0

def Liquidation_Distance(data):
    return 2 if data["liq_dist"] < 0.5 else 0

def Kill_Zone_5m(data):
    return 1 if data["kill_5m"] else 0

def Kill_Zone_HTF(data):
    return 1 if data["kill_htf"] else 0

def Kill_Switch_Fast(data):
    return -10 if data["kill_fast"] else 0

def Kill_Switch_Primary(data):
    return -25 if data["kill_primary"] else 0

def News_Guard(data):
    return -5 if data["news_risk"] else 0

def Recheck_30s(data):
    return 1 if data["recheck_ok"] else 0

def Partial_Exit_Logic(data):
    return 1

def BTC_Risk_Filter_L1(data):
    return -8 if not data["btc_calm"] else 2

def BTC_Risk_Filter_L2(data):
    return -15 if data["btc_trending_fast"] else 2

def BTC_Funding_OI_Combo(data):
    return 2 if data["funding_oi_combo"] else 0

def Funding_Extreme(data):
    return -2 if data["funding_extreme"] else 0

def Funding_Delta_Speed(data):
    return 1 if data["funding_delta"] else 0

def Funding_Arbitrage(data):
    return 2 if data["arb_opportunity"] else 0

def OI_Spike_5pct(data):
    return 2 if data["oi_spike"] else 0

def OI_Spike_Sustained(data):
    return 2 if data["oi_sustained"] else 0

def ETH_BTC_Beta_Divergence(data):
    return 1 if data["beta_div"] else 0

def Options_Gamma_Flip(data):
    return 3 if data["gamma_flip"] else 0

def Heatmap_Sweep(data):
    return 2 if data["heat_sweep"] else 0

def Micro_Slip(data):
    return -1 if data["slippage"] else 0

def Order_Block(data):
    return 2 if data["orderblock"] else 0

def Score_Normalization(score):
    return max(0, min(score, MAX_SCORE))

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
# helpers.py — FINAL HYBRID MIND (Part 2/2)

import time
from datetime import datetime

# -----------------------------------------
# MODE SCORE THRESHOLDS (Arun's system)
# -----------------------------------------
MODE_THRESHOLDS = {
    "quick": 55,
    "mid": 62,
    "trend": 70,
}

# TP/SL rules (Arun preference)
TP_RULES = {
    "quick": (1.2, 1.6),   # %
    "mid": (1.8, 2.4),
    "trend": (2.5, 3.5),
}

SL_RULES = {
    "quick": (0.5, 0.8),
    "mid": (0.9, 1.2),
    "trend": (1.2, 1.8),
}

# -----------------------------------------
# BTC CALM CHECK
# -----------------------------------------
async def btc_calm_check(session):
    url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
    try:
        async with session.get(url) as r:
            d = await r.json()
            change = abs(float(d["priceChangePercent"]))
            return change < BTC_STABILITY_THRESHOLD
    except:
        return False

# -----------------------------------------
# COOLDOWN SYSTEM
# -----------------------------------------
last_signal_time = {}

def is_cooldown(symbol):
    if symbol not in last_signal_time:
        return False
    diff = time.time() - last_signal_time[symbol]
    return diff < (COOLDOWN_MINUTES * 60)

def update_cooldown(symbol):
    last_signal_time[symbol] = time.time()

# -----------------------------------------
# TP/SL AUTO CALCULATOR
# -----------------------------------------
def calc_tp_sl(price, mode):
    tp_min, tp_max = TP_RULES[mode]
    sl_min, sl_max = SL_RULES[mode]

    tp = price * (1 + tp_min/100)
    sl = price * (1 - sl_min/100)

    return round(tp, 4), round(sl, 4)

# -----------------------------------------
# FINAL DECISION ENGINE
# -----------------------------------------
def decide_signal(symbol, mode, price, score):
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
        "sl": sl
    }

# -----------------------------------------
# FORMAT TELEGRAM SIGNAL (Style C Premium)
# -----------------------------------------
def format_signal(sig):
    return f"""
🔥 <b>{sig['mode'].upper()} SIGNAL</b>

<b>Pair:</b> {sig['symbol']}
<b>Price:</b> {sig['price']}
<b>Score:</b> {sig['score']} / 100

🎯 <b>TP:</b>
<code>{sig['tp']}</code>

🛑 <b>SL:</b>
<code>{sig['sl']}</code>

⚡ Leverage: 50x
⏱️ Cooldown: 30 minutes
📊 Mode: {sig['mode'].upper()}

#HybridAI #ArunSystem
"""

# -----------------------------------------
# MASTER PROCESSOR (Used by main.py)
# -----------------------------------------
def process_data(symbol, mode, live_data):
    score = Final_Signal_Score(live_data)
    decision = decide_signal(symbol, mode, live_data["price"], score)
    return decision