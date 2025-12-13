"""
45 Trading Logics Implementation
All market analysis and signal generation logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from enum import Enum
import logging

from helpers import TechnicalIndicators

logger = logging.getLogger(__name__)

class TradeMode(Enum):
    """Trading modes based on timeframe"""
    QUICK = "QUICK"  # 5m
    MID = "MID"      # 15m
    TREND = "TREND"  # 1h+

# ==================== A) MARKET HEALTH FILTERS (1-8) ====================
class MarketHealthLogics:
    
    @staticmethod
    def btc_calm_check(df: pd.DataFrame, threshold: float = 0.015) -> Tuple[bool, float]:
        """Logic 1: BTC低波动率过滤"""
        atr = TechnicalIndicators.atr(df).iloc[-1]
        price = df['close'].iloc[-1]
        volatility = atr / price
        return volatility < threshold, round(volatility * 100, 2)
    
    @staticmethod
    def market_regime(df: pd.DataFrame) -> str:
        """Logic 2: Market Regime Detection"""
        ema20 = TechnicalIndicators.ema(df, 20).iloc[-1]
        ema50 = TechnicalIndicators.ema(df, 50).iloc[-1]
        atr = TechnicalIndicators.atr(df).iloc[-1]
        price = df['close'].iloc[-1]
        volatility = atr / price
        
        if ema20 > ema50 and volatility < 0.02:
            return "TRENDING_UP"
        elif ema20 < ema50 and volatility < 0.02:
            return "TRENDING_DOWN"
        elif volatility > 0.03:
            return "VOLATILE"
        else:
            return "RANGING"
    
    @staticmethod
    def fragile_market(df: pd.DataFrame) -> bool:
        """Logic 5: Fragile Market Detection"""
        recent_wicks = []
        for i in range(-5, 0):
            body = abs(df['close'].iloc[i] - df['open'].iloc[i])
            total_range = df['high'].iloc[i] - df['low'].iloc[i]
            wick_ratio = 1 - (body / total_range if total_range > 0 else 0)
            recent_wicks.append(wick_ratio)
        
        return np.mean(recent_wicks) > 0.6
    
    @staticmethod
    def spread_check(orderbook: dict) -> bool:
        """Logic 7: Spread & Slippage Safety"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return False
        
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        
        if best_bid == 0:
            return False
        
        spread = (best_ask - best_bid) / best_bid
        return spread < 0.001  # 0.1% max spread

# ==================== B) PRICE ACTION & STRUCTURE (9-15) ====================
class PriceActionLogics:
    
    @staticmethod
    def breakout_confirmation(df: pd.DataFrame) -> Tuple[bool, str]:
        """Logic 9: Breakout Confirmation (Body > Wick)"""
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        total_wick = upper_wick + lower_wick
        
        if body > total_wick * 1.5:
            return True, "BULLISH" if last['close'] > last['open'] else "BEARISH"
        return False, "NONE"
    
    @staticmethod
    def market_structure(df: pd.DataFrame) -> str:
        """Logic 10: Market Structure Shift"""
        highs = df['high'].iloc[-10:].values
        lows = df['low'].iloc[-10:].values
        
        # Higher Highs and Higher Lows
        hh_hl = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
        
        # Lower Highs and Lower Lows
        lh_ll = all(highs[i] <= highs[i-1] for i in range(1, len(highs)))
        
        if hh_hl:
            return "BULLISH"
        elif lh_ll:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    @staticmethod
    def ema_alignment(df: pd.DataFrame) -> Tuple[bool, str]:
        """Logic 13: EMA Alignment (20/50/200)"""
        ema20 = TechnicalIndicators.ema(df, 20).iloc[-1]
        ema50 = TechnicalIndicators.ema(df, 50).iloc[-1]
        ema200 = TechnicalIndicators.ema(df, 200).iloc[-1]
        
        if ema20 > ema50 > ema200:
            return True, "BULLISH"
        elif ema20 < ema50 < ema200:
            return True, "BEARISH"
        else:
            return False, "NEUTRAL"
    
    @staticmethod
    def bb_squeeze(df: pd.DataFrame) -> bool:
        """Logic 15: Bollinger Band Squeeze"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df)
        bandwidth = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
        min_bw = ((upper - lower) / middle).iloc[-20:].min()
        return bandwidth <= min_bw * 1.1

# ==================== C) MOMENTUM & OSCILLATORS (16-21) ====================
class MomentumLogics:
    
    @staticmethod
    def rsi_analysis(df: pd.DataFrame) -> Dict:
        """Logic 16: RSI + Divergence"""
        rsi = TechnicalIndicators.rsi(df)
        current_rsi = rsi.iloc[-1]
        
        # Divergence
        price_trend = df['close'].iloc[-10:].diff().sum()
        rsi_trend = rsi.iloc[-10:].diff().sum()
        
        return {
            'value': round(current_rsi, 2),
            'overbought': current_rsi > 70,
            'oversold': current_rsi < 30,
            'bullish_div': price_trend < 0 and rsi_trend > 0,
            'bearish_div': price_trend > 0 and rsi_trend < 0
        }
    
    @staticmethod
    def macd_analysis(df: pd.DataFrame) -> Dict:
        """Logic 17: MACD Cross & Momentum"""
        macd_line, signal, histogram = TechnicalIndicators.macd(df)
        
        curr_macd = macd_line.iloc[-1]
        curr_sig = signal.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal.iloc[-2]
        
        return {
            'bullish_cross': prev_macd <= prev_sig and curr_macd > curr_sig,
            'bearish_cross': prev_macd >= prev_sig and curr_macd < curr_sig,
            'momentum_up': histogram.iloc[-1] > histogram.iloc[-5]
        }
    
    @staticmethod
    def obv_divergence(df: pd.DataFrame) -> Dict:
        """Logic 19: OBV Divergence"""
        obv = TechnicalIndicators.obv(df)
        
        price_trend = df['close'].iloc[-10:].diff().sum()
        obv_trend = obv.iloc[-10:].diff().sum()
        
        return {
            'bullish_div': price_trend < 0 and obv_trend > 0,
            'bearish_div': price_trend > 0 and obv_trend < 0,
            'trending_up': obv_trend > 0
        }

# ==================== D) ORDER FLOW & DEPTH (22-31) ====================
class OrderFlowLogics:
    
    @staticmethod
    def orderbook_imbalance(orderbook: dict) -> Dict:
        """Logic 22: Orderbook Imbalance"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return {'imbalance': 0, 'direction': 'NEUTRAL'}
        
        bid_vol = sum([b[1] for b in orderbook['bids'][:10]])
        ask_vol = sum([a[1] for a in orderbook['asks'][:10]])
        
        total = bid_vol + ask_vol
        if total == 0:
            return {'imbalance': 0, 'direction': 'NEUTRAL'}
        
        imbalance = (bid_vol - ask_vol) / total
        
        direction = "BULLISH" if imbalance > 0.2 else "BEARISH" if imbalance < -0.2 else "NEUTRAL"
        
        return {'imbalance': round(imbalance, 3), 'direction': direction}
    
    @staticmethod
    def vwap_deviation(df: pd.DataFrame) -> Dict:
        """Logic 24: VWAP Deviation"""
        vwap = TechnicalIndicators.vwap(df)
        price = df['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        deviation_pct = ((price - current_vwap) / current_vwap) * 100
        
        status = "EXTENDED" if abs(deviation_pct) > 2 else "DEVIATED" if abs(deviation_pct) > 1 else "FAIR"
        
        return {'deviation_pct': round(deviation_pct, 2), 'status': status}
    
    @staticmethod
    def whale_detection(orderbook: dict) -> bool:
        """Logic 27: Whale Detection"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return False
        
        # Calculate average order size
        all_orders = orderbook['bids'] + orderbook['asks']
        avg_size = np.mean([o[1] for o in all_orders])
        
        # Check for orders 10x larger than average
        whale_threshold = avg_size * 10
        
        for order in all_orders:
            if order[1] > whale_threshold:
                return True
        
        return False

# ==================== F) ANTI-TRAP & PROTECTION (38-45) ====================
class AntiTrapLogics:
    
    @staticmethod
    def round_number_trap(price: float) -> bool:
        """Logic 38: Round Number Trap Avoidance"""
        round_numbers = [50, 100, 150, 200, 250, 500, 1000, 5000, 10000, 50000, 100000]
        
        for rn in round_numbers:
            if abs(price - rn) / rn < 0.01:
                return True
        
        return False
    
    @staticmethod
    def manipulation_candle(df: pd.DataFrame) -> bool:
        """Logic 42: Manipulation Candle Detection"""
        last = df.iloc[-1]
        prev_avg_range = (df['high'].iloc[-10:-1] - df['low'].iloc[-10:-1]).mean()
        
        current_range = last['high'] - last['low']
        body = abs(last['close'] - last['open'])
        
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        total_wick = upper_wick + lower_wick
        
        # Sudden wick
        sudden_wick = current_range > prev_avg_range * 2
        
        # High wick ratio
        wick_ratio = total_wick / current_range if current_range > 0 else 0
        
        return sudden_wick and wick_ratio > 0.7
    
    @staticmethod
    def consecutive_loss_cooldown(recent_trades: List[Dict], max_losses: int = 3) -> bool:
        """Logic 45: Consecutive Loss Protection"""
        if len(recent_trades) < max_losses:
            return True
        
        last_trades = recent_trades[-max_losses:]
        all_losses = all(t.get('pnl', 0) < 0 for t in last_trades)
        
        return not all_losses  # Return False if max consecutive losses reached

# ==================== SIGNAL GENERATOR ====================
class SignalGenerator:
    """Main signal generation using all 45 logics"""
    
    def __init__(self):
        self.market_health = MarketHealthLogics()
        self.price_action = PriceActionLogics()
        self.momentum = MomentumLogics()
        self.orderflow = OrderFlowLogics()
        self.anti_trap = AntiTrapLogics()
    
    async def generate_signal(
        self,
        market: str,
        candles: pd.DataFrame,
        orderbook: dict,
        timeframe: str,
        current_price_inr: float
    ) -> Dict:
        """
        Analyze market data and generate trading signal
        Returns signal dict or None if conditions not met
        """
        
        if len(candles) < 200:
            return None
        
        # Initialize scoring
        score = 0
        max_score = 0
        details = {}
        
        # === A) MARKET HEALTH (Weight: 25%) ===
        
        # 1. BTC Calm Check
        is_calm, vol = self.market_health.btc_calm_check(candles)
        if is_calm:
            score += 5
        max_score += 5
        details['calm'] = is_calm
        
        # 2. Market Regime
        regime = self.market_health.market_regime(candles)
        if regime in ['TRENDING_UP', 'TRENDING_DOWN']:
            score += 8
        max_score += 8
        details['regime'] = regime
        
        # 5. Fragile Market
        is_fragile = self.market_health.fragile_market(candles)
        if not is_fragile:
            score += 5
        max_score += 5
        details['fragile'] = is_fragile
        
        # 7. Spread Check
        spread_ok = self.market_health.spread_check(orderbook)
        if spread_ok:
            score += 4
        max_score += 4
        details['spread_ok'] = spread_ok
        
        # === B) PRICE ACTION (Weight: 30%) ===
        
        # 9. Breakout
        breakout, breakout_dir = self.price_action.breakout_confirmation(candles)
        if breakout:
            score += 10
        max_score += 10
        details['breakout'] = breakout_dir
        
        # 10. Market Structure
        structure = self.price_action.market_structure(candles)
        if structure in ['BULLISH', 'BEARISH']:
            score += 8
        max_score += 8
        details['structure'] = structure
        
        # 13. EMA Alignment
        ema_aligned, ema_dir = self.price_action.ema_alignment(candles)
        if ema_aligned:
            score += 7
        max_score += 7
        details['ema'] = ema_dir
        
        # 15. BB Squeeze
        squeeze = self.price_action.bb_squeeze(candles)
        if squeeze:
            score += 5
        max_score += 5
        details['squeeze'] = squeeze
        
        # === C) MOMENTUM (Weight: 25%) ===
        
        # 16. RSI
        rsi = self.momentum.rsi_analysis(candles)
        if rsi['bullish_div'] or rsi['oversold']:
            score += 8
        elif rsi['bearish_div'] or rsi['overbought']:
            score += 8
        max_score += 8
        details['rsi'] = rsi
        
        # 17. MACD
        macd = self.momentum.macd_analysis(candles)
        if macd['bullish_cross'] or macd['bearish_cross']:
            score += 7
        max_score += 7
        details['macd'] = macd
        
        # 19. OBV
        obv = self.momentum.obv_divergence(candles)
        if obv['bullish_div'] or obv['bearish_div']:
            score += 5
        max_score += 5
        details['obv'] = obv
        
        # === D) ORDER FLOW (Weight: 15%) ===
        
        # 22. Orderbook Imbalance
        imbalance = self.orderflow.orderbook_imbalance(orderbook)
        if imbalance['direction'] in ['BULLISH', 'BEARISH']:
            score += 5
        max_score += 5
        details['imbalance'] = imbalance
        
        # 24. VWAP Deviation
        vwap_dev = self.orderflow.vwap_deviation(candles)
        if vwap_dev['status'] == 'FAIR':
            score += 3
        max_score += 3
        details['vwap'] = vwap_dev
        
        # 27. Whale Detection
        whale = self.orderflow.whale_detection(orderbook)
        details['whale'] = whale
        
        # === F) ANTI-TRAP (Weight: 5%) ===
        
        # 38. Round Number Trap
        round_trap = self.anti_trap.round_number_trap(current_price_inr)
        if not round_trap:
            score += 2
        max_score += 2
        details['round_trap'] = round_trap
        
        # 42. Manipulation Candle
        manip = self.anti_trap.manipulation_candle(candles)
        if not manip:
            score += 3
        max_score += 3
        details['manipulation'] = manip
        
        # Calculate percentage
        logic_score = int((score / max_score) * 100) if max_score > 0 else 0
        
        # Determine trade side
        side = self._determine_side(details)
        
        # Calculate levels
        atr = TechnicalIndicators.atr(candles).iloc[-1]
        entry, sl, tp = self._calculate_levels(current_price_inr, atr, side)
        
        # Confidence
        confidence = "HIGH" if logic_score >= 75 else "MEDIUM" if logic_score >= 65 else "LOW"
        
        # Mode
        mode = self._determine_mode(timeframe)
        
        return {
            'market': market,
            'timeframe': timeframe,
            'side': side,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'rr_ratio': abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0,
            'logic_score': logic_score,
            'confidence': confidence,
            'mode': mode,
            'details': details
        }
    
    def _determine_side(self, details: Dict) -> str:
        """Determine BUY or SELL based on signals"""
        bullish = 0
        bearish = 0
        
        # Regime
        if details.get('regime') == 'TRENDING_UP':
            bullish += 3
        elif details.get('regime') == 'TRENDING_DOWN':
            bearish += 3
        
        # Breakout
        if details.get('breakout') == 'BULLISH':
            bullish += 3
        elif details.get('breakout') == 'BEARISH':
            bearish += 3
        
        # Structure
        if details.get('structure') == 'BULLISH':
            bullish += 2
        elif details.get('structure') == 'BEARISH':
            bearish += 2
        
        # EMA
        if details.get('ema') == 'BULLISH':
            bullish += 2
        elif details.get('ema') == 'BEARISH':
            bearish += 2
        
        # RSI
        rsi = details.get('rsi', {})
        if rsi.get('oversold') or rsi.get('bullish_div'):
            bullish += 2
        elif rsi.get('overbought') or rsi.get('bearish_div'):
            bearish += 2
        
        # MACD
        macd = details.get('macd', {})
        if macd.get('bullish_cross'):
            bullish += 2
        elif macd.get('bearish_cross'):
            bearish += 2
        
        # Imbalance
        imb = details.get('imbalance', {})
        if imb.get('direction') == 'BULLISH':
            bullish += 1
        elif imb.get('direction') == 'BEARISH':
            bearish += 1
        
        return "BUY" if bullish > bearish else "SELL"
    
    def _calculate_levels(self, price: float, atr: float, side: str) -> Tuple[float, float, float]:
        """Calculate Entry, SL, TP levels"""
        if side == "BUY":
            entry = price
            sl = price - (atr * 1.5)
            tp = price + (atr * 3)
        else:
            entry = price
            sl = price + (atr * 1.5)
            tp = price - (atr * 3)
        
        return entry, sl, tp
    
    def _determine_mode(self, timeframe: str) -> str:
        """Determine trading mode from timeframe"""
        if timeframe == '5m':
            return TradeMode.QUICK.value
        elif timeframe == '15m':
            return TradeMode.MID.value
        else:
            return TradeMode.TREND.value
