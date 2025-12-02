import time
import math

# ================================
# COOLDOWN (30 minutes)
# ================================
COOLDOWN = 1800
last_signal = {}

def cooldown_ok(symbol):
    t = last_signal.get(symbol, 0)
    return (time.time() - t) > COOLDOWN

def update_cd(symbol):
    last_signal[symbol] = time.time()

# ================================
# MODE THRESHOLDS
# ================================
MODE_THRESH = {
    "quick": 55,
    "mid": 62,
    "trend": 70
}

TP_SL = {
    "quick": (1.2, 0.7),
    "mid":   (1.8, 1.0),
    "trend": (2.5, 1.2)
}

# ================================
# 30 LOGIC (18 + 12)
# ================================

# ---- HTF Trend ----
def HTF_EMA_1h_15m(d):  return 2 if d['ema_15m'] > d['ema_1h'] else 0
def HTF_EMA_1h_4h(d):   return 2 if d['ema_1h'] > d['ema_4h'] else 0
def HTF_EMA_1h_8h(d):   return 2 if d['ema_4h'] > d['ema_8h'] else 0

def HTF_Structure_Break(d):   return 3 if d['trend'] == "bull_break" else 0
def HTF_Structure_Reject(d):  return 3 if d['trend'] == "reject" else 0
def Trend_Continuation(d):    return 3 if d['trend_strength'] > 0.7 else 0

# ---- PA / Volume ----
def Micro_Pullback(d):        return 2 if d['micro_pb'] else 0
def Trend_Exhaustion(d):      return 1 if d['exhaustion'] else 0
def Imbalance_FVG(d):         return 2 if d['fvg'] else 0
def Order_Block(d):           return 2 if d['orderblock'] else 0

def Vol_Sweep_1m(d):          return 2 if d['vol_1m'] > d['vol_5m'] else 0
def Vol_Sweep_5m(d):          return 2 if d['vol_5m'] > d['vol_1m'] else 0

def ADR_DayRange(d):          return 1 if d['adr_ok'] else 0
def ATR_Expansion(d):         return 1 if d['atr_expanding'] else 0

# ---- Spread/Risk ----
def Tight_Spread_Filter(d):   return 1 if d['spread'] < 0.03 else 0
def Spread_Safety(d):         return 1 if d['spread'] < 0.05 else 0
def BTC_Risk_Filter_L1(d):    return -8 if not d['btc_calm'] else 2
def Kill_Switch_Primary(d):   return -25 if d['kill_primary'] else 0

# ---- PRO 12 (Orderflow) ----
def Wick_Strength(d):         return 2 if d['wick_ratio'] > 1.5 else 0
def Liquidity_Wall(d):        return 2 if d['liq_wall'] else 0
def Liquidity_Bend(d):        return 2 if d['liq_bend'] else 0
def Orderbook_Wall_Shift(d):  return 2 if d['wall_shift'] else 0
def Speed_Imbalance(d):       return 2 if d['speed_imbalance'] else 0
def Absorption(d):            return 2 if d['absorption'] else 0

def Sweep_Reversal(d):        return 2 if d['liquidation_sweep'] else 0
def Micro_Slip(d):            return -1 if d['slippage'] else 0
def Liquidation_Distance(d):  return 2 if d['liq_dist'] < 0.5 else 0

def Spread_Snap_025(d):       return 1 if d['spread'] < 0.0025 else 0
def Spread_Snap_05(d):        return 1 if d['spread'] < 0.005 else 0
def Taker_Pressure(d):        return 2 if d['taker_pressure'] else 0

# ================================
# FINAL SCORE
# ================================
def final_score(d):
    score = sum([
        HTF_EMA_1h_15m(d), HTF_EMA_1h_4h(d), HTF_EMA_1h_8h(d),
        HTF_Structure_Break(d), HTF_Structure_Reject(d),
        Trend_Continuation(d), Micro_Pullback(d), Trend_Exhaustion(d),
        Imbalance_FVG(d), Order_Block(d),
        Vol_Sweep_1m(d), Vol_Sweep_5m(d),
        ADR_DayRange(d), ATR_Expansion(d),
        Tight_Spread_Filter(d), Spread_Safety(d),
        BTC_Risk_Filter_L1(d), Kill_Switch_Primary(d),
        Wick_Strength(d), Liquidity_Wall(d), Liquidity_Bend(d),
        Orderbook_Wall_Shift(d), Speed_Imbalance(d), Absorption(d),
        Sweep_Reversal(d), Micro_Slip(d), Liquidation_Distance(d),
        Spread_Snap_025(d), Spread_Snap_05(d), Taker_Pressure(d)
    ])
    return max(0, min(score, 100))

# ================================
# BUY / SELL DIRECTION
# ================================
def trade_side(price, prev, ema1h):
    if price > prev and price > ema1h:
        return "BUY"
    elif price < prev and price < ema1h:
        return "SELL"
    return None

# ================================
# TP / SL PER MODE + BE-SL
# ================================
def calc_tp_sl(price, mode):
    tp_pct, sl_pct = TP_SL[mode]
    tp = price * (1 + tp_pct/100)
    sl = price * (1 - sl_pct/100)
    return round(tp, 6), round(sl, 6)

def apply_be_sl(entry_price):
    return round(entry_price, 6)

# ================================
# PREMIUM TELEGRAM STYLE
# ================================
def format_signal(symbol, side, mode, price, score, tp, sl, liq):
    return f"""
🔥 <b>{mode.upper()} {side} SIGNAL</b>

<b>Pair:</b> {symbol}
<b>Entry:</b> <code>{price}</code>
<b>Score:</b> <b>{score}</b> / 100

🎯 <b>TP:</b> <code>{tp}</code>
🛑 <b>SL:</b> <code>{sl}</code>

🟩 <b>BE-SL:</b> Enabled  
📉 <b>LiQ Distance:</b> {liq}

⚡ Leverage: 50x  
⏱ Cooldown: 30m  
📊 Mode: <b>{mode}</b>

#ArunSystem #HybridAI
"""
