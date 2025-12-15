# signal_engine.py
import pandas as pd
import numpy as np
from datetime import datetime
from config import *

class SignalEngine:
    def __init__(self):
        self.last_signal_time = {}

    # ---------- UTILS ----------
    def ema(self, s, p): return s.ewm(span=p, adjust=False).mean()
    def rsi(self, s, p=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / l)

    # ---------- CORE FILTERS ----------
    def ema_trend(self, df):
        if len(df) < EMA_SLOW: return "NEUTRAL"
        c = df['close']
        e20, e50, e200 = self.ema(c, EMA_FAST), self.ema(c, EMA_MID), self.ema(c, EMA_SLOW)
        if e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]: return "BULL"
        if e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]: return "BEAR"
        return "NEUTRAL"

    def volume_spike(self, df):
        vol, ma = df['volume'], df['volume'].rolling(VOLUME_MA_PERIOD).mean()
        return "SPIKE" if vol.iloc[-1] > ma.iloc[-1] * 1.1 else "NORMAL"

    def rsi_cond(self, df):
        r = self.rsi(df['close'])
        if r.iloc[-1] < RSI_OVERSOLD: return "OVERSOLD"
        if r.iloc[-1] > RSI_OVERBOUGHT: return "OVERBOUGHT"
        return "NEUTRAL"

    def macd_momentum(self, df):
        c = df['close']
        macd = self.ema(c, MACD_FAST) - self.ema(c, MACD_SLOW)
        hist = macd - self.ema(macd, MACD_SIGNAL)
        return "BULL" if hist.iloc[-1] > hist.iloc[-2] else "BEAR"

    def structure(self, df):
        highs = df['high'].rolling(5).max()
        lows  = df['low'].rolling(5).min()
        hh = highs.iloc[-1] > highs.iloc[-3] and highs.iloc[-2] > highs.iloc[-4]
        ll = lows.iloc[-1] < lows.iloc[-3] and lows.iloc[-2] < lows.iloc[-4]
        return "BULL" if hh else "BEAR" if ll else "NEUTRAL"

    def manipulation_filter(self, df):
        c = df.iloc[-1]
        body = abs(c['close'] - c['open'])
        range_ = c['high'] - c['low']
        if range_ == 0: return True
        return (body / range_) > 0.05          # শিথিল

    def round_trap(self, price):
        return len(str(int(price)).rstrip('0')) <= 1 and str(int(price))[-4:] == '0000'

    def cooldown_ok(self, sym, mode):
        if sym not in self.last_signal_time: return True
        return (datetime.now() - self.last_signal_time[sym]).seconds / 60 >= SIGNAL_MODES[mode]['hold_time']

    # ---------- MAIN ----------
    def generate(self, data, mode='mid'):
        sym, df = data['symbol'], data['data'].copy()
        if len(df) < 50: return None

        if not self.cooldown_ok(sym, mode):
            print(f"  ❌ {sym} cooldown")
            return None
        if not self.manipulation_filter(df):
            print(f"  ❌ {sym} manipulation")
            return None
        if self.round_trap(df['close'].iloc[-1]):
            print(f"  ❌ {sym} round trap")
            return None

        trend   = self.ema_trend(df)
        vol     = self.volume_spike(df)
        rsi     = self.rsi_cond(df)
        macd    = self.macd_momentum(df)
        struct  = self.structure(df)

        score_long = score_short = 0
        if trend == "BULL":  score_long += 2
        if trend == "BEAR":  score_short += 2
        if vol == "SPIKE":   score_long += 1; score_short += 1
        if rsi == "OVERSOLD": score_long += 1
        if rsi == "OVERBOUGHT": score_short += 1
        if macd == "BULL":   score_long += 1
        if macd == "BEAR":   score_short += 1
        if struct == "BULL": score_long += 1
        if struct == "BEAR": score_short += 1

        price = df['close'].iloc[-1]
        if score_long >= MIN_SIGNAL_SCORE:
            self.last_signal_time[sym] = datetime.now()
            return dict(symbol=sym, direction='LONG', entry=price, score=score_long, mode=mode)
        if score_short >= MIN_SIGNAL_SCORE:
            self.last_signal_time[sym] = datetime.now()
            return dict(symbol=sym, direction='SHORT', entry=price, score=score_short, mode=mode)
        return None
