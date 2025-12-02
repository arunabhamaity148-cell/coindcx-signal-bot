import time, hmac, hashlib, pandas as pd, numpy as np, os
from dotenv import load_dotenv
load_dotenv()

COOLDOWN = 1800
last_signal = {}
MODE_THRESH = {"quick": 55, "mid": 62, "trend": 70}
TP_SL = {"quick": (1.2, 0.7), "mid": (1.8, 1.0), "trend": (2.5, 1.2)}

def auto_leverage(score, mode):
    base = {"quick": 15, "mid": 20, "trend": 25}
    if score >= 90: return min(base[mode] + 5, 30)
    if score >= 80: return base[mode]
    if score >= 70: return base[mode] - 2
    return 15

def ema(closes, period):
    return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

def calc_exhaustion(k):
    o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
    rng = h - l
    if rng == 0: return False
    wick_top = h - max(o, c)
    wick_bot = min(o, c) - l
    return (wick_top / rng >= 0.7) or (wick_bot / rng >= 0.7)

def calc_fvg(klines):
    for i in range(1, len(klines) - 1):
        prev_high, prev_low = float(klines[i - 1][2]), float(klines[i - 1][3])
        curr_low, curr_high = float(klines[i][3]), float(klines[i][2])
        nxt_low = float(klines[i + 1][3])
        if curr_low > prev_high and nxt_low > prev_high: return True
        if curr_high < prev_low and float(klines[i + 1][2]) < prev_low: return True
    return False

def calc_ob(klines):
    o, h, l, c = float(klines[-2][1]), float(klines[-2][2]), float(klines[-2][3]), float(klines[-2][4])
    bull = (o > c) and (float(klines[-1][3]) < l)
    bear = (o < c) and (float(klines[-1][2]) > h)
    return bull or bear

# 30-logic
def HTF_EMA_1h_15m(d): return 2 if d['ema_15m'] > d['ema_1h'] else 0
def HTF_EMA_1h_4h(d): return 2 if d['ema_1h'] > d['ema_4h'] else 0
def HTF_EMA_1h_8h(d): return 2 if d['ema_4h'] > d['ema_8h'] else 0
def HTF_Structure_Break(d): return 3 if d['trend'] == "bull_break" else 0
def HTF_Structure_Reject(d): return 3 if d['trend'] == "reject" else 0
def Trend_Continuation(d): return 3 if d['trend_strength'] > 0.7 else 0
def Micro_Pullback(d): return 2 if d['micro_pb'] else 0
def Trend_Exhaustion(d): return 1 if d['exhaustion'] else 0
def Imbalance_FVG(d): return 2 if d['fvg'] else 0
def Order_Block(d): return 2 if d['orderblock'] else 0
def Vol_Sweep_1m(d): return 2 if d['vol_1m'] > d['vol_5m'] else 0
def Vol_Sweep_5m(d): return 2 if d['vol_5m'] > d['vol_1m'] else 0
def ADR_DayRange(d): return 1 if d['adr_ok'] else 0
def ATR_Expansion(d): return 1 if d['atr_expanding'] else 0
def Tight_Spread_Filter(d): return 1 if d['spread'] < 0.03 else 0
def Spread_Safety(d): return 1 if d['spread'] < 0.05 else 0
def BTC_Risk_Filter_L1(d): return -8 if not d['btc_calm'] else 2
def Kill_Switch_Primary(d): return -25 if d['kill_primary'] else 0
def Wick_Strength(d): return 2 if d['wick_ratio'] > 1.5 else 0
def Liquidity_Wall(d): return 2 if d['liq_wall'] else 0
def Liquidity_Bend(d): return 2 if d['liq_bend'] else 0
def Orderbook_Wall_Shift(d): return 2 if d['wall_shift'] else 0
def Speed_Imbalance(d): return 2 if d['speed_imbalance'] else 0
def Absorption(d): return 2 if d['absorption'] else 0
def Sweep_Reversal(d): return 2 if d['liquidation_sweep'] else 0
def Micro_Slip(d): return -1 if d['slippage'] else 0
def Liquidation_Distance(d): return 2 if d['liq_dist'] < 0.5 else 0
def Spread_Snap_025(d): return 1 if d['spread'] < 0.0025 else 0
def Spread_Snap_05(d): return 1 if d['spread'] < 0.005 else 0
def Taker_Pressure(d): return 2 if d['taker_pressure'] else 0

def final_score(d):
    return max(0, min(sum([
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
    ]), 100))

def trade_side(price, prev, ema1h):
    if price > prev and price > ema1h:
        return "BUY"
    elif price < prev and price < ema1h:
        return "SELL"
    return None

def calc_tp_sl(price, mode):
    tp_pct, sl_pct = TP_SL[mode]
    tp = price * (1 + tp_pct / 100)
    sl = price * (1 - sl_pct / 100)
    return round(tp, 6), round(sl, 6)

def format_signal(symbol, side, mode, price, score, tp, sl, liq, lev):
    return f"""🔥 <b>{mode.upper()} {side} SIGNAL</b>
<b>Pair:</b> {symbol}
<b>Entry:</b> <code>{price}</code>
<b>Score:</b> <b>{score}</b>/100
🎯 <b>TP:</b> <code>{tp}</code>
🛑 <b>SL:</b> <code>{sl}</code>
🟩 <b>BE-SL:</b> Enabled
📉 <b>LiQ Distance:</b> {liq}
⚙️ <b>Leverage:</b> {lev}x
⏱ Cooldown: 30m
📊 Mode: <b>{mode}</b>
#ArunSystem #HybridAI"""
