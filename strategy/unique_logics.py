"""
UNIQUE TRADING LOGICS - 45 Advanced Strategies  
(Part 1 of 3)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

class UniqueLogics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    # ==================== A) MARKET HEALTH FILTERS ====================

    def check_btc_calm(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < 20:
            return True
        recent = df.tail(20)
        vol = (recent['high'] - recent['low']) / recent['close'].replace(0, np.nan)
        v = float(vol.mean() or 0)
        return v < float(self.config.get("btc_calm_threshold", 0.015))

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        if df is None or len(df) < 50:
            return "unknown"
        adx = self._calculate_adx(df)
        atr = self._calculate_atr(df)
        last = df['close'].iloc[-1]
        atr_pct = float(atr.iloc[-1] / last)
        if adx > 25:
            return "trending"
        if atr_pct > 0.03:
            return "volatile"
        return "ranging"

    def check_funding_rate_normal(self, rate: float) -> bool:
        return abs(rate) < float(self.config.get("funding_rate_extreme", 0.0015))

    def check_fear_greed_index(self, v: int) -> bool:
        low = int(self.config.get("fear_greed_extreme_low", 20))
        high = int(self.config.get("fear_greed_extreme_high", 80))
        return low < v < high

    def fragile_btc_mode(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if df is None or len(df) < 30:
            return False, "normal"
        recent = df.tail(30)
        moves = (abs(recent['close'].pct_change()) > 0.02).sum()
        vol_avg = recent['volume'].mean()
        vol_last = recent['volume'].tail(5).mean()
        fragile = moves > 5 or vol_last > vol_avg * 2
        return fragile, ("conservative" if fragile else "normal")

    def check_news_filter(self, now: datetime, news_times: List[datetime]) -> bool:
        skip = int(self.config.get("news_skip_minutes", 30))
        for t in news_times:
            if abs((now - t).total_seconds()) / 60 < skip:
                return False
        return True

    def check_spread_slippage(self, ob: Dict[str, Any]) -> bool:
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return True
        mid = (bids[0][0] + asks[0][0]) / 2
        spread = ob.get("spread", 0)
        pct = spread / mid
        return pct < float(self.config.get("spread_max_percent", 0.001))

    def check_liquidity_window(self, now: datetime) -> bool:
        bad = self.config.get("low_liquidity_hours", [2,3,4,5])
        return now.hour not in bad

    # ==================== B) PRICE ACTION & STRUCTURE ====================

    def check_breakout_confirmation(self, c):
        body = abs(c['close'] - c['open'])
        rng = c['high'] - c['low']
        if rng == 0:
            return False
        return (body / rng) > 0.6

    def detect_market_structure_shift(self, df):
        if df is None or len(df) < 20:
            return "neutral"
        h = df['high'].tail(20)
        l = df['low'].tail(20)
        if h.iloc[-5:].max() > h.iloc[-15:-5].max() and \
           l.iloc[-5:].min() > l.iloc[-15:-5].min():
            return "bullish"
        if h.iloc[-5:].max() < h.iloc[-15:-5].max() and \
           l.iloc[-5:].min() < l.iloc[-15:-5].min():
            return "bearish"
        return "neutral"

    def check_orderblock_retest(self, df, price):
        if df is None or len(df) < 50:
            return {'found': False}
        x = df.copy()
        x['v'] = x['volume'].rank(pct=True)
        zones = x[x['v'] > 0.9].tail(10)
        for _, z in zones.iterrows():
            if abs(price - z['close']) / price < 0.005:
                return {
                    'found': True,
                    'price': float(z['close']),
                    'type': ("bullish" if z['close'] > z['open'] else "bearish")
                }
        return {'found': False}

    def detect_fair_value_gap(self, df):
        if len(df) < 3:
            return {'found': False}
        a, b, c = df.tail(3).to_dict('records')
        if a['high'] < c['low']:
            return {'found': True, 'type': 'bullish'}
        if a['low'] > c['high']:
            return {'found': True, 'type': 'bearish'}
        return {'found': False}

    def check_ema_alignment(self, df):
        if len(df) < 200:
            return "neutral"
        e20 = df['close'].ewm(span=20).mean().iloc[-1]
        e50 = df['close'].ewm(span=50).mean().iloc[-1]
        e200 = df['close'].ewm(span=200).mean().iloc[-1]
        if e20 > e50 > e200:
            return "bullish"
        if e20 < e50 < e200:
            return "bearish"
        return "neutral"

    def calculate_atr_filter(self, df):
        atr = self._calculate_atr(df)
        return float(atr.iloc[-1])

    def detect_bollinger_squeeze(self, df):
        if len(df) < 20:
            return False
        up, lo = self._calculate_bollinger_bands(df)
        w = (up - lo) / df['close']
        return w.iloc[-1] < w.rolling(50).mean().iloc[-1] * 0.8
# ==================== C) MOMENTUM ====================

    def check_rsi_conditions(self, df):
        rsi = self._calculate_rsi(df)
        v = float(rsi.iloc[-1])
        if v < 30:
            sig = "oversold"
        elif v > 70:
            sig = "overbought"
        else:
            sig = "neutral"
        return {"signal": sig, "value": v}

    def check_macd_cross(self, df):
        macd, sig, hist = self._calculate_macd(df)
        if len(hist) < 2:
            return {"signal": "neutral"}
        if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
            return {"signal": "bullish"}
        if hist.iloc[-2] > 0 and hist.iloc[-1] < 0:
            return {"signal": "bearish"}
        return {"signal": "neutral"}

    def check_stochastic(self, df):
        k, d = self._calculate_stochastic(df)
        if len(k) < 2:
            return {"signal": "neutral"}
        if k.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1]:
            return {"signal": "bullish"}
        if k.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1]:
            return {"signal": "bearish"}
        return {"signal": "neutral"}

    def check_obv_divergence(self, df):
        obv = self._calculate_obv(df)
        if len(obv) < 10:
            return "neutral"
        p_up = df['close'].iloc[-10:].is_monotonic_increasing
        o_up = obv.iloc[-10:].is_monotonic_increasing
        if p_up and not o_up:
            return "bearish_divergence"
        if not p_up and o_up:
            return "bullish_divergence"
        return "neutral"

    def check_mfi(self, df):
        mfi = self._calculate_mfi(df)
        v = float(mfi.iloc[-1])
        if v > 80:
            return {"signal": "overbought", "value": v}
        if v < 20:
            return {"signal": "oversold", "value": v}
        return {"signal": "neutral", "value": v}

    def check_roc(self, df):
        if len(df) < 20:
            return 0.0
        return float((df['close'].iloc[-1] - df['close'].iloc[-20]) /
                     df['close'].iloc[-20] * 100)

# ==================== D) ORDERFLOW ====================

    def check_orderbook_imbalance(self, ob):
        r = float(ob.get("imbalance", 1))
        th = float(self.config.get("orderbook_imbalance_threshold", 1.2))
        if r > th: return {"signal": "bullish", "ratio": r}
        if r < 1/th: return {"signal": "bearish", "ratio": r}
        return {"signal": "neutral", "ratio": r}

    def calculate_vwap(self, df):
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()

    def check_vwap_deviation(self, df):
        vwap = self.calculate_vwap(df)
        price = df['close'].iloc[-1]
        dv = (price - vwap.iloc[-1]) / vwap.iloc[-1]
        th = float(self.config.get("vwap_deviation_percent", 0.005))
        if abs(dv) > th:
            return {"signal": ("far" if dv > 0 else "near"), "deviation": dv}
        return {"signal": "normal", "deviation": dv}

    def check_vwap_logic(self, df):
        v = self.calculate_vwap(df)
        r = df.tail(5)
        c = 0
        for i in range(1, 5):
            if (r['close'].iloc[i] > v.iloc[-5+i]) != (r['close'].iloc[i-1] > v.iloc[-5+i-1]):
                c += 1
        if c >= 2:
            return "bounce"
        return "reclaim" if r['close'].iloc[-1] > v.iloc[-1] else "rejection"

    def calculate_cvd(self, trades):
        buy = sum(t['amount'] for t in trades if t['side']=="buy")
        sell = sum(t['amount'] for t in trades if t['side']=="sell")
        return float(buy - sell)

    def detect_large_order(self, ob):
        th = float(self.config.get("large_order_threshold", 100000))
        L=[]
        for b in ob.get("bids",[])[:20]:
            if b[0]*b[1] >= th: L.append({"side":"bid","price":b[0]})
        for a in ob.get("asks",[])[:20]:
            if a[0]*a[1] >= th: L.append({"side":"ask","price":a[0]})
        return L

    def detect_spoofing_wall(self, ob):
        th = float(self.config.get("spoofing_wall_threshold", 500000))
        bids = [b for b in ob.get("bids",[]) if b[0]*b[1] > th]
        asks = [a for a in ob.get("asks",[]) if a[0]*a[1] > th]
        return {"bid_walls":len(bids),"ask_walls":len(asks),"suspicious":bool(bids or asks)}

    def calculate_true_liquidity(self, ob):
        total=0
        for b in ob.get("bids",[])[:20]:
            total += b[0]*b[1]
        for a in ob.get("asks",[])[:20]:
            total += a[0]*a[1]
        return float(total)

    def calculate_aggression_ratio(self, trades):
        b = sum(1 for t in trades if t["side"]=="buy")
        s = sum(1 for t in trades if t["side"]=="sell")
        if s==0: return 999
        return float(b/s)

    def check_spread_velocity(self, hist):
        if len(hist)<2: return 0
        return float(hist[-1].get("spread",0) - hist[-2].get("spread",0))

# ==================== E) DERIVATIVES ====================

    def check_oi_trend(self, oi):
        if len(oi)<10: return "neutral"
        if np.mean(oi[-5:]) > np.mean(oi[-10:-5])*1.1: return "increasing"
        if np.mean(oi[-5:]) < np.mean(oi[-10:-5])*0.9: return "decreasing"
        return "stable"

    def check_oi_divergence(self, df, oi):
        if len(oi)<10 or len(df)<10: return "neutral"
        p_up = df['close'].iloc[-10:].is_monotonic_increasing
        oi_up = oi[-10] > oi[-20] if len(oi)>=20 else False
        if p_up and not oi_up: return "bearish_divergence"
        if not p_up and oi_up: return "bullish_divergence"
        return "neutral"

    def check_liquidation_proximity(self, price, clusters):
        out=[]
        th = float(self.config.get("liquidation_proximity_percent",0.02))
        for c in clusters:
            if abs(price - c['price'])/price < th:
                out.append(c)
        return {"nearby":bool(out),"clusters":out}

    def check_funding_arbitrage(self, fr):
        vals = list(fr.values())
        if len(vals)<2:
            return {"opportunity":False}
        sp = max(vals)-min(vals)
        return {"opportunity":sp>0.0005, "spread":sp}

    def check_gamma_exposure(self, opt):
        g = float(opt.get("gamma",0))
        th=float(self.config.get("gamma_exposure_threshold",0.5))
        if g>th: return "positive"
        if g<-th: return "negative"
        return "neutral"

    def gamma_adjusted_sizing(self, size, gamma):
        if gamma=="negative": return size*0.5
        if gamma=="positive": return size
        return size*0.75

# ==================== F) ANTI-TRAP ====================

    def avoid_round_numbers(self, price):
        r = round(price, -3)
        return abs(price-r)/price > float(self.config.get("round_number_avoid_distance",0.001))

    def avoid_obvious_sr(self, price, sr):
        for s in sr:
            if abs(price-s)/price < 0.005:
                return False
        return True

    def detect_sl_hunting_zone(self, df):
        if len(df)<20: return False
        c=0
        for _,x in df.tail(20).iterrows():
            body = abs(x['close']-x['open'])
            uw = x['high']-max(x['close'],x['open'])
            lw = min(x['close'],x['open'])-x['low']
            if uw>body*2 or lw>body*2: c+=1
        return c>3

    def check_odd_time_entry(self, now):
        bad=[9,10,17]
        return now.minute not in [0,15,30,45] or now.hour not in bad

    def filter_sudden_wick(self, df):
        if len(df)<2: return True
        x=df.iloc[-1]
        pct = (x['high']-x['low'])/x['close']
        return pct<0.01

    def avoid_bot_rush_time(self, now):
        return now.hour not in [9,10,17]

    def filter_manipulation_candle(self, c):
        body=abs(c['close']-c['open'])
        rng=c['high']-c['low']
        if rng==0: return False
        return (body/rng)>0.2

    def check_consecutive_losses(self, trades):
        c=0
        for t in reversed(trades):
            if t['pnl']<0: c+=1
            else: break
        return c<2, c
# ==================== HELPER INDICATORS ====================

    def _calculate_adx(self, df, period=14):
        h=df['high']; l=df['low']; c=df['close']
        pc=c.shift(1)
        tr = pd.concat([
            h-l,
            (h-pc).abs(),
            (l-pc).abs()
        ],axis=1).max(axis=1)
        atr=tr.rolling(period).mean()
        up=h.diff(); dn=l.shift(1)-l
        plus=np.where((up>dn)&(up>0),up,0)
        minus=np.where((dn>up)&(dn>0),dn,0)
        plus=pd.Series(plus).rolling(period).mean()
        minus=pd.Series(minus).rolling(period).mean()
        di_p = 100*(plus/atr)
        di_m = 100*(minus/atr)
        dx = 100*(abs(di_p-di_m)/(di_p+di_m))
        return float(dx.rolling(period).mean().iloc[-1])

    def _calculate_atr(self, df, period=14):
        h=df['high']; l=df['low']; c=df['close']
        pc=c.shift(1)
        tr = pd.concat([
            h-l,
            (h-pc).abs(),
            (l-pc).abs()
        ],axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_bollinger_bands(self, df, p=20, k=2):
        ma=df['close'].rolling(p).mean()
        sd=df['close'].rolling(p).std()
        return ma+k*sd, ma-k*sd

    def _calculate_rsi(self, df, p=14):
        d=df['close'].diff()
        g=d.where(d>0,0).rolling(p).mean()
        l=(-d.where(d<0,0)).rolling(p).mean()
        rs=g/l.replace(0,np.nan)
        return 100-(100/(1+rs))

    def _calculate_macd(self, df, f=12, s=26, sig=9):
        e1=df['close'].ewm(span=f).mean()
        e2=df['close'].ewm(span=s).mean()
        m=e1-e2
        sl=m.ewm(span=sig).mean()
        return m, sl, m-sl

    def _calculate_stochastic(self, df, kp=14, dp=3):
        lo=df['low'].rolling(kp).min()
        hi=df['high'].rolling(kp).max()
        k = 100*((df['close']-lo)/(hi-lo))
        d = k.rolling(dp).mean()
        return k, d

    def _calculate_obv(self, df):
        obv=[0]
        for i in range(1,len(df)):
            if df['close'].iloc[i]>df['close'].iloc[i-1]:
                obv.append(obv[-1]+df['volume'].iloc[i])
            elif df['close'].iloc[i]<df['close'].iloc[i-1]:
                obv.append(obv[-1]-df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv,index=df.index)

    def _calculate_mfi(self, df, p=14):
        tp=(df['high']+df['low']+df['close'])/3
        mf=tp*df['volume']
        pos=[0]; neg=[0]
        for i in range(1,len(df)):
            if tp.iloc[i]>tp.iloc[i-1]: pos.append(mf.iloc[i]); neg.append(0)
            else: neg.append(mf.iloc[i]); pos.append(0)
        pos=pd.Series(pos).rolling(p).sum()
        neg=pd.Series(neg).rolling(p).sum()
        return 100-(100/(1+(pos/neg)))

# ==================== LogicEvaluator Wrapper ====================

class LogicEvaluator:
    def __init__(self, config: Dict[str,Any]):
        self.engine = UniqueLogics(config)

    def evaluate_all_logics(self, df, orderbook, funding_rate, oi_history,
                            trades, fear=50, news=None, liq=None):

        news = news or []
        liq = liq or []

        # Market section
        checks = [
            self.engine.check_btc_calm(df),
            self.engine.check_funding_rate_normal(funding_rate),
            self.engine.check_fear_greed_index(fear),
            self.engine.check_liquidity_window(datetime.utcnow()),
            self.engine.check_spread_slippage(orderbook),
            self.engine.check_news_filter(datetime.utcnow(), news)
        ]
        base = (sum(checks)/len(checks))*100

        atr = self.engine.calculate_atr_filter(df)
        price = df['close'].iloc[-1]
        if (atr/price) > 0.05:
            base -= 20

        final = round(base,2)
        allowed = final >= 50 and all(checks)

        return {
            "final_score": final,
            "trade_allowed": allowed,
            "market_regime": self.engine.detect_market_regime(df),
            "ema": self.engine.check_ema_alignment(df),
            "atr": atr
        }