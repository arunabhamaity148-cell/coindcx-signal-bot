class ScoringEngine:
    """
    RELAXED scoring for real INR futures trading
    Total = 100 points, more achievable thresholds
    """
    
    def __init__(self, config):
        self.config = config
    
    def calculate_score(self, analysis, direction, mtf_score):
        score = 0
        reasons = []
        
        if not analysis:
            return 0, [], 'No data'
        
        # 1. EMA TREND (20 points)
        if direction == 'LONG':
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] > analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bullish')
                elif analysis['ema_fast'] > analysis['ema_slow'] * 0.998:
                    score += 12
                    reasons.append('EMA slightly bullish')
        else:
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] < analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bearish')
                elif analysis['ema_fast'] < analysis['ema_slow'] * 1.002:
                    score += 12
                    reasons.append('EMA slightly bearish')
        
        # 2. RSI ZONE (15 points) - WIDENED ranges
        rsi = analysis.get('rsi')
        if rsi:
            if direction == 'LONG':
                if 25 < rsi < 60:
                    score += 15
                    reasons.append(f'RSI good ({rsi:.1f})')
                elif 20 < rsi <= 70:
                    score += 10
                    reasons.append(f'RSI ok ({rsi:.1f})')
            else:
                if 40 < rsi < 75:
                    score += 15
                    reasons.append(f'RSI good ({rsi:.1f})')
                elif 30 <= rsi <= 80:
                    score += 10
                    reasons.append(f'RSI ok ({rsi:.1f})')
        
        # 3. MACD MOMENTUM (15 points)
        macd = analysis.get('macd')
        macd_signal = analysis.get('macd_signal')
        if macd and macd_signal:
            if direction == 'LONG' and macd > macd_signal:
                score += 15
                reasons.append('MACD bullish')
            elif direction == 'SHORT' and macd < macd_signal:
                score += 15
                reasons.append('MACD bearish')
            else:
                score += 5  # Partial credit
        
        # 4. ADX TREND STRENGTH (10 points) - RELAXED
        adx = analysis.get('adx')
        if adx:
            if adx > 20:
                score += 10
                reasons.append(f'Strong trend (ADX {adx:.1f})')
            elif adx > 15:
                score += 8
                reasons.append(f'Good trend (ADX {adx:.1f})')
            elif adx > 12:
                score += 5
                reasons.append(f'Moderate trend (ADX {adx:.1f})')
        
        # 5. CANDLESTICK PATTERN (10 points)
        patterns = analysis.get('patterns', {})
        if direction == 'LONG':
            if patterns.get('bullish_engulfing') or patterns.get('morning_star'):
                score += 10
                reasons.append('Strong bullish pattern')
            elif patterns.get('hammer'):
                score += 7
                reasons.append('Hammer pattern')
        else:
            if patterns.get('bearish_engulfing') or patterns.get('evening_star'):
                score += 10
                reasons.append('Strong bearish pattern')
            elif patterns.get('shooting_star'):
                score += 7
                reasons.append('Shooting star')
        
        # 6. LIQUIDITY GRAB (10 points)
        smart = analysis.get('smart', {})
        liq_grab = smart.get('liquidity_grab')
        if liq_grab:
            if (direction == 'LONG' and liq_grab == 'bullish_sweep') or \
               (direction == 'SHORT' and liq_grab == 'bearish_sweep'):
                score += 10
                reasons.append('Liquidity sweep')
        
        # 7. ORDER FLOW (10 points) - RELAXED thresholds
        order_flow = smart.get('order_flow', 0)
        if direction == 'LONG':
            if order_flow > 0.25:
                score += 10
                reasons.append(f'Strong buy flow ({order_flow:.2f})')
            elif order_flow > 0.1:
                score += 7
                reasons.append(f'Buy flow ({order_flow:.2f})')
            elif order_flow > 0:
                score += 3
        else:
            if order_flow < -0.25:
                score += 10
                reasons.append(f'Strong sell flow ({order_flow:.2f})')
            elif order_flow < -0.1:
                score += 7
                reasons.append(f'Sell flow ({order_flow:.2f})')
            elif order_flow < 0:
                score += 3
        
        # 8. MTF ALIGNMENT (10 points)
        score += mtf_score
        if mtf_score >= 8:
            reasons.append('MTF fully aligned')
        elif mtf_score >= 5:
            reasons.append('MTF aligned')
        elif mtf_score >= 3:
            reasons.append('MTF acceptable')
        
        return score, reasons, smart.get('market_regime', 'unknown')
    
    def get_quality_tier(self, score):
        if score >= self.config.HIGH_QUALITY_THRESHOLD:
            return 'HIGH', '🟢'
        elif score >= self.config.MEDIUM_QUALITY_THRESHOLD:
            return 'MEDIUM', '🟡'
        else:
            return 'LOWER', '🟠'