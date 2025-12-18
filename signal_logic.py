from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic
import numpy as np

class SignalGenerator:
    
    def __init__(self):
        self.config = Config()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()
    
    def analyze_market(self, candles_data):
        if not candles_data or len(candles_data) < 100:
            return None
        
        candles = candles_data[-100:]
        
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        ema_fast = self.indicators.calculate_ema(closes, self.config.EMA_FAST)
        ema_slow = self.indicators.calculate_ema(closes, self.config.EMA_SLOW)
        
        rsi = self.indicators.calculate_rsi(closes, self.config.RSI_PERIOD)
        macd, signal, histogram = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(highs, lows, closes, self.config.ATR_PERIOD)
        adx = self.indicators.calculate_adx(highs, lows, closes, self.config.ADX_PERIOD)
        
        bullish_engulfing = self.patterns.is_bullish_engulfing(candles)
        bearish_engulfing = self.patterns.is_bearish_engulfing(candles)
        hammer = self.patterns.is_hammer(candles[-1])
        shooting_star = self.patterns.is_shooting_star(candles[-1])
        morning_star = self.patterns.is_morning_star(candles)
        evening_star = self.patterns.is_evening_star(candles)
        
        market_regime = self.smart.detect_market_regime(closes, atr, adx)
        liquidity_grab = self.smart.detect_liquidity_grab(candles)
        order_flow = self.smart.calculate_order_flow(candles)
        
        return {
            'price': closes[-1],
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'atr': atr,
            'adx': adx,
            'patterns': {
                'bullish_engulfing': bullish_engulfing,
                'bearish_engulfing': bearish_engulfing,
                'hammer': hammer,
                'shooting_star': shooting_star,
                'morning_star': morning_star,
                'evening_star': evening_star
            },
            'smart': {
                'market_regime': market_regime,
                'liquidity_grab': liquidity_grab,
                'order_flow': order_flow
            }
        }
    
    def calculate_signal_score(self, analysis, direction, market_name):
        score = 0
        reasons = []
        blocks = []
        
        if not analysis:
            return 0, [], ['No analysis data']
        
        regime = analysis['smart']['market_regime']
        
        if self.config.BLOCK_RANGING_MARKETS and regime == 'ranging':
            adx_val = analysis['adx'] if analysis['adx'] else 0
            blocks.append(f'BLOCKED: Ranging (ADX: {adx_val:.1f})')
            return 0, [], blocks
        
        if self.config.BLOCK_HIGH_VOLATILITY and regime == 'volatile':
            blocks.append(f'BLOCKED: High volatility')
            return 0, [], blocks
        
        # EMA TREND (20 points)
        if direction == 'LONG':
            if analysis['ema_fast'] and analysis['ema_slow']:
                if analysis['ema_fast'] > analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bullish')
                elif analysis['ema_fast'] > analysis['ema_slow'] * 0.998:
                    score += 10
                    reasons.append('EMA near bullish')
        else:
            if analysis['ema_fast'] and analysis['ema_slow']:
                if analysis['ema_fast'] < analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bearish')
                elif analysis['ema_fast'] < analysis['ema_slow'] * 1.002:
                    score += 10
                    reasons.append('EMA near bearish')
        
        # RSI (15 points) - RELAXED ZONES
        if analysis['rsi']:
            if direction == 'LONG' and 25 < analysis['rsi'] < 65:
                score += 15
                reasons.append(f'RSI favorable ({analysis["rsi"]:.1f})')
            elif direction == 'SHORT' and 35 < analysis['rsi'] < 75:
                score += 15
                reasons.append(f'RSI favorable ({analysis["rsi"]:.1f})')
            elif 30 < analysis['rsi'] < 70:
                score += 8
                reasons.append(f'RSI neutral ({analysis["rsi"]:.1f})')
        
        # MACD (15 points)
        if analysis['macd'] and analysis['macd_signal']:
            if direction == 'LONG' and analysis['macd'] > analysis['macd_signal']:
                score += 15
                reasons.append('MACD bullish')
            elif direction == 'SHORT' and analysis['macd'] < analysis['macd_signal']:
                score += 15
                reasons.append('MACD bearish')
            elif abs(analysis['macd'] - analysis['macd_signal']) < abs(analysis['macd']) * 0.1:
                score += 8
                reasons.append('MACD near crossover')
        
        # PATTERNS (10 points) - BONUS IF PRESENT
        patterns = analysis['patterns']
        if direction == 'LONG':
            if patterns['bullish_engulfing'] or patterns['hammer'] or patterns['morning_star']:
                score += 10
                reasons.append('Bullish pattern')
        else:
            if patterns['bearish_engulfing'] or patterns['shooting_star'] or patterns['evening_star']:
                score += 10
                reasons.append('Bearish pattern')
        
        # ADX (10 points) - RELAXED
        if analysis['adx']:
            if analysis['adx'] > 25:
                score += 10
                reasons.append(f'Strong trend (ADX {analysis["adx"]:.1f})')
            elif analysis['adx'] > 20:
                score += 8
                reasons.append(f'Moderate trend (ADX {analysis["adx"]:.1f})')
            elif analysis['adx'] > 15:
                score += 5
                reasons.append(f'Weak trend (ADX {analysis["adx"]:.1f})')
        
        # SMART MONEY (10 points)
        smart = analysis['smart']
        
        if smart['liquidity_grab']:
            if (direction == 'LONG' and smart['liquidity_grab'] == 'bullish_sweep') or \
               (direction == 'SHORT' and smart['liquidity_grab'] == 'bearish_sweep'):
                score += 10
                reasons.append('Liquidity grab')
        
        # ORDER FLOW (10 points) - RELAXED
        if smart['order_flow']:
            if (direction == 'LONG' and smart['order_flow'] > 0.2) or \
               (direction == 'SHORT' and smart['order_flow'] < -0.2):
                score += 10
                reasons.append(f'Order flow ({smart["order_flow"]:.2f})')
            elif (direction == 'LONG' and smart['order_flow'] > 0) or \
                 (direction == 'SHORT' and smart['order_flow'] < 0):
                score += 5
                reasons.append(f'Slight order flow ({smart["order_flow"]:.2f})')
        
        # REGIME BONUS (10 points)
        if regime == 'trending':
            score += 10
            reasons.append('Trending regime')
        elif regime == 'mixed':
            score += 5
            reasons.append('Mixed regime')
        
        return score, reasons, blocks
    
    def generate_signal(self, market, candles_data):
        analysis = self.analyze_market(candles_data)
        
        if not analysis or not analysis['atr']:
            print(f"      ⚠️ {market}: Insufficient data")
            return None
        
        long_score, long_reasons, long_blocks = self.calculate_signal_score(analysis, 'LONG', market)
        short_score, short_reasons, short_blocks = self.calculate_signal_score(analysis, 'SHORT', market)
        
        best_direction = 'LONG' if long_score >= short_score else 'SHORT'
        best_score = max(long_score, short_score)
        best_reasons = long_reasons if long_score >= short_score else short_reasons
        best_blocks = long_blocks if long_score >= short_score else short_blocks
        
        print(f"      📊 {market}: {best_direction} = {best_score}/100 (need {self.config.MIN_SIGNAL_SCORE}+)")
        
        if best_score < self.config.MIN_SIGNAL_SCORE:
            if best_blocks:
                print(f"         Blocked: {best_blocks[0]}")
            return None
        
        if long_score >= short_score and long_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'LONG'
            score = long_score
            reasons = long_reasons
        elif short_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'SHORT'
            score = short_score
            reasons = short_reasons
        else:
            return None
        
        entry = analysis['price']
        atr = analysis['atr']
        
        if direction == 'LONG':
            sl = entry - (atr * self.config.ATR_SL_MULTIPLIER)
            tp1 = entry + (atr * self.config.ATR_TP1_MULTIPLIER)
            tp2 = entry + (atr * self.config.ATR_TP2_MULTIPLIER)
        else:
            sl = entry + (atr * self.config.ATR_SL_MULTIPLIER)
            tp1 = entry - (atr * self.config.ATR_TP1_MULTIPLIER)
            tp2 = entry - (atr * self.config.ATR_TP2_MULTIPLIER)
        
        risk = abs(entry - sl)
        reward = abs(tp2 - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.config.MIN_RR_RATIO:
            print(f"      ❌ {market}: R:R too low ({rr_ratio:.2f})")
            return None
        
        signal = {
            'market': market,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'score': score,
            'rr_ratio': rr_ratio,
            'reasons': reasons,
            'analysis': {
                'rsi': analysis['rsi'],
                'adx': analysis['adx'],
                'market_regime': analysis['smart']['market_regime']
            }
        }
        
        return signal