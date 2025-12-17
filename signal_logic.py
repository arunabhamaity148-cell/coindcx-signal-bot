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
        """Complete market analysis"""
        if not candles_data or len(candles_data) < 100:
            return None
        
        candles = candles_data[-100:]
        
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        # Calculate all indicators
        ema_9 = self.indicators.calculate_ema(closes, self.config.EMA_SHORT)
        ema_21 = self.indicators.calculate_ema(closes, self.config.EMA_MEDIUM)
        ema_50 = self.indicators.calculate_ema(closes, self.config.EMA_LONG)
        
        rsi = self.indicators.calculate_rsi(closes, self.config.RSI_PERIOD)
        macd, signal, histogram = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(highs, lows, closes, self.config.ATR_PERIOD)
        adx = self.indicators.calculate_adx(highs, lows, closes, self.config.ADX_PERIOD)
        
        # Pattern detection
        bullish_engulfing = self.patterns.is_bullish_engulfing(candles)
        bearish_engulfing = self.patterns.is_bearish_engulfing(candles)
        hammer = self.patterns.is_hammer(candles[-1])
        shooting_star = self.patterns.is_shooting_star(candles[-1])
        morning_star = self.patterns.is_morning_star(candles)
        evening_star = self.patterns.is_evening_star(candles)
        
        # Smart money analysis
        market_regime = self.smart.detect_market_regime(closes, atr, adx)
        fvg = self.smart.detect_fair_value_gap(candles)
        liquidity_grab = self.smart.detect_liquidity_grab(candles)
        order_flow = self.smart.calculate_order_flow_imbalance(candles)
        poc = self.smart.calculate_volume_profile_poc(candles)
        fib_levels = self.smart.calculate_fibonacci_levels(candles)
        
        # Divergence detection
        rsi_values = [self.indicators.calculate_rsi(closes[:i+1]) for i in range(len(closes)-15, len(closes))]
        rsi_values = [r for r in rsi_values if r is not None]
        divergence = self.indicators.detect_divergence(closes[-15:], rsi_values) if len(rsi_values) > 10 else None
        
        return {
            'price': closes[-1],
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50,
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
                'fvg': fvg,
                'liquidity_grab': liquidity_grab,
                'order_flow': order_flow,
                'poc': poc,
                'fib_levels': fib_levels,
                'divergence': divergence
            }
        }
    
    def calculate_signal_score(self, analysis, direction):
        """Calculate signal score 0-100"""
        score = 0
        reasons = []
        
        if not analysis:
            return 0, []
        
        # Trend alignment (20 points)
        if direction == 'LONG':
            if analysis['ema_9'] and analysis['ema_21'] and analysis['ema_50']:
                if analysis['ema_9'] > analysis['ema_21'] > analysis['ema_50']:
                    score += 20
                    reasons.append('Strong uptrend (EMA aligned)')
                elif analysis['ema_9'] > analysis['ema_21']:
                    score += 10
                    reasons.append('Short-term uptrend')
        else:
            if analysis['ema_9'] and analysis['ema_21'] and analysis['ema_50']:
                if analysis['ema_9'] < analysis['ema_21'] < analysis['ema_50']:
                    score += 20
                    reasons.append('Strong downtrend (EMA aligned)')
                elif analysis['ema_9'] < analysis['ema_21']:
                    score += 10
                    reasons.append('Short-term downtrend')
        
        # RSI confirmation (15 points)
        if analysis['rsi']:
            if direction == 'LONG' and 30 < analysis['rsi'] < 50:
                score += 15
                reasons.append(f'RSI oversold recovery ({analysis["rsi"]:.1f})')
            elif direction == 'SHORT' and 50 < analysis['rsi'] < 70:
                score += 15
                reasons.append(f'RSI overbought rejection ({analysis["rsi"]:.1f})')
            elif direction == 'LONG' and 50 < analysis['rsi'] < 60:
                score += 8
            elif direction == 'SHORT' and 40 < analysis['rsi'] < 50:
                score += 8
        
        # MACD momentum (15 points)
        if analysis['macd'] and analysis['macd_signal']:
            if direction == 'LONG' and analysis['macd'] > analysis['macd_signal'] and analysis['macd_histogram'] > 0:
                score += 15
                reasons.append('MACD bullish crossover')
            elif direction == 'SHORT' and analysis['macd'] < analysis['macd_signal'] and analysis['macd_histogram'] < 0:
                score += 15
                reasons.append('MACD bearish crossover')
        
        # Candlestick patterns (10 points)
        patterns = analysis['patterns']
        if direction == 'LONG':
            if patterns['bullish_engulfing'] or patterns['hammer'] or patterns['morning_star']:
                score += 10
                reasons.append('Bullish reversal pattern detected')
        else:
            if patterns['bearish_engulfing'] or patterns['shooting_star'] or patterns['evening_star']:
                score += 10
                reasons.append('Bearish reversal pattern detected')
        
        # Volume/ADX confirmation (10 points)
        if analysis['adx'] and analysis['adx'] > 25:
            score += 10
            reasons.append(f'Strong trend (ADX {analysis["adx"]:.1f})')
        elif analysis['adx'] and analysis['adx'] > 20:
            score += 5
        
        # Smart money concepts (10 points)
        smart = analysis['smart']
        
        if smart['liquidity_grab']:
            if (direction == 'LONG' and smart['liquidity_grab'] == 'bullish_sweep') or \
               (direction == 'SHORT' and smart['liquidity_grab'] == 'bearish_sweep'):
                score += 5
                reasons.append(f'Liquidity grab detected ({smart["liquidity_grab"]})')
        
        if smart['fvg']:
            if (direction == 'LONG' and smart['fvg']['type'] == 'bullish') or \
               (direction == 'SHORT' and smart['fvg']['type'] == 'bearish'):
                score += 5
                reasons.append(f'Fair Value Gap ({smart["fvg"]["type"]})')
        
        # Order flow (10 points)
        if smart['order_flow']:
            if (direction == 'LONG' and smart['order_flow'] > 0.3) or \
               (direction == 'SHORT' and smart['order_flow'] < -0.3):
                score += 10
                reasons.append(f'Strong order flow imbalance ({smart["order_flow"]:.2f})')
            elif (direction == 'LONG' and smart['order_flow'] > 0.1) or \
                 (direction == 'SHORT' and smart['order_flow'] < -0.1):
                score += 5
        
        # Divergence (10 points)
        if smart['divergence']:
            if (direction == 'LONG' and smart['divergence'] == 'bullish') or \
               (direction == 'SHORT' and smart['divergence'] == 'bearish'):
                score += 10
                reasons.append(f'{smart["divergence"].capitalize()} divergence')
        
        return score, reasons
    
    def generate_signal(self, market, candles_data):
        """Generate trading signal"""
        analysis = self.analyze_market(candles_data)
        
        if not analysis or not analysis['atr']:
            return None
        
        # Check both directions
        long_score, long_reasons = self.calculate_signal_score(analysis, 'LONG')
        short_score, short_reasons = self.calculate_signal_score(analysis, 'SHORT')
        
        # Select highest scoring direction
        if long_score < self.config.MIN_SIGNAL_SCORE and short_score < self.config.MIN_SIGNAL_SCORE:
            return None
        
        if long_score > short_score:
            direction = 'LONG'
            score = long_score
            reasons = long_reasons
        else:
            direction = 'SHORT'
            score = short_score
            reasons = short_reasons
        
        # Calculate entry, SL, TP
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
        
        # Calculate R:R ratio
        risk = abs(entry - sl)
        reward = abs(tp2 - entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.config.MIN_RR_RATIO:
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