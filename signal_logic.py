from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic

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
        
        if not analysis:
            return 0, [], 'No data'
        
        # EMA TREND (20 points)
        if direction == 'LONG':
            if analysis['ema_fast'] and analysis['ema_slow'] and analysis['ema_fast'] > analysis['ema_slow']:
                score += 20
                reasons.append('EMA bullish')
        else:
            if analysis['ema_fast'] and analysis['ema_slow'] and analysis['ema_fast'] < analysis['ema_slow']:
                score += 20
                reasons.append('EMA bearish')
        
        # RSI (15 points)
        if analysis['rsi']:
            if direction == 'LONG' and 30 < analysis['rsi'] < 55:
                score += 15
                reasons.append(f'RSI optimal ({analysis["rsi"]:.1f})')
            elif direction == 'SHORT' and 45 < analysis['rsi'] < 70:
                score += 15
                reasons.append(f'RSI optimal ({analysis["rsi"]:.1f})')
        
        # MACD (15 points)
        if analysis['macd'] and analysis['macd_signal']:
            if direction == 'LONG' and analysis['macd'] > analysis['macd_signal']:
                score += 15
                reasons.append('MACD bullish')
            elif direction == 'SHORT' and analysis['macd'] < analysis['macd_signal']:
                score += 15
                reasons.append('MACD bearish')
        
        # PATTERNS (10 points)
        patterns = analysis['patterns']
        if direction == 'LONG' and (patterns['bullish_engulfing'] or patterns['hammer'] or patterns['morning_star']):
            score += 10
            reasons.append('Bullish pattern')
        elif direction == 'SHORT' and (patterns['bearish_engulfing'] or patterns['shooting_star'] or patterns['evening_star']):
            score += 10
            reasons.append('Bearish pattern')
        
        # ADX TREND STRENGTH (10 points)
        if analysis['adx']:
            if analysis['adx'] > 25:
                score += 10
                reasons.append(f'Strong trend (ADX {analysis["adx"]:.1f})')
            elif analysis['adx'] > 20:
                score += 7
                reasons.append(f'Moderate trend (ADX {analysis["adx"]:.1f})')
        
        # SMART MONEY (10 points)
        smart = analysis['smart']
        if smart['liquidity_grab']:
            if (direction == 'LONG' and smart['liquidity_grab'] == 'bullish_sweep') or \
               (direction == 'SHORT' and smart['liquidity_grab'] == 'bearish_sweep'):
                score += 10
                reasons.append('Liquidity grab')
        
        # ORDER FLOW (10 points)
        if smart['order_flow']:
            if (direction == 'LONG' and smart['order_flow'] > 0.25) or \
               (direction == 'SHORT' and smart['order_flow'] < -0.25):
                score += 10
                reasons.append(f'Order flow ({smart["order_flow"]:.2f})')
        
        # REGIME BONUS (10 points)
        if smart['market_regime'] == 'trending':
            score += 10
            reasons.append('Trending regime')
        
        return score, reasons, analysis['smart']['market_regime']
    
    def apply_smart_filters(self, score, regime, adx, market):
        # HARD BLOCK: Ranging markets with low score
        if regime == 'ranging' and score < self.config.BLOCK_RANGING_IF_SCORE_BELOW:
            print(f"      ❌ BLOCKED: Ranging market (ADX: {adx:.1f}, Score: {score})")
            return False, 'RANGING_BLOCKED'
        
        # HARD BLOCK: Volatile markets with low score
        if regime == 'volatile' and score < self.config.BLOCK_VOLATILE_IF_SCORE_BELOW:
            print(f"      ❌ BLOCKED: Volatile market (Score: {score})")
            return False, 'VOLATILE_BLOCKED'
        
        # HARD BLOCK: Very weak ADX even with decent score
        if adx and adx < self.config.MIN_ADX_FOR_RANGING_OVERRIDE and score < 72:
            print(f"      ❌ BLOCKED: Weak ADX ({adx:.1f}) + Score {score}")
            return False, 'WEAK_TREND_BLOCKED'
        
        return True, None
    
    def get_quality_tier(self, score):
        if score >= self.config.HIGH_QUALITY_THRESHOLD:
            return 'HIGH', '🟢'
        elif score >= self.config.MEDIUM_QUALITY_THRESHOLD:
            return 'MEDIUM', '🟡'
        else:
            return 'LOWER', '🟠'
    
    def generate_signal(self, market, candles_data):
        analysis = self.analyze_market(candles_data)
        
        if not analysis or not analysis['atr']:
            print(f"      ⚠️ {market}: Insufficient data")
            return None
        
        long_score, long_reasons, long_regime = self.calculate_signal_score(analysis, 'LONG', market)
        short_score, short_reasons, short_regime = self.calculate_signal_score(analysis, 'SHORT', market)
        
        if long_score >= short_score and long_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'LONG'
            score = long_score
            reasons = long_reasons
            regime = long_regime
        elif short_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'SHORT'
            score = short_score
            reasons = short_reasons
            regime = short_regime
        else:
            print(f"      📊 {market}: Best={max(long_score, short_score)} (need {self.config.MIN_SIGNAL_SCORE}+)")
            return None
        
        # APPLY SMART FILTERS
        passed, block_reason = self.apply_smart_filters(score, regime, analysis['adx'], market)
        if not passed:
            return None
        
        # Calculate levels
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
        
        quality_tier, emoji = self.get_quality_tier(score)
        
        print(f"      🎯 {emoji} {market} {direction}: Score {score} ({quality_tier})")
        
        signal = {
            'market': market,
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'score': score,
            'quality_tier': quality_tier,
            'quality_emoji': emoji,
            'rr_ratio': rr_ratio,
            'reasons': reasons,
            'analysis': {
                'rsi': analysis['rsi'],
                'adx': analysis['adx'],
                'market_regime': regime
            }
        }
        
        return signal