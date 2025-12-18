class ScoringEngine:
    """
    Clean 0-100 scoring system
    Total = 100 points
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
        else:
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] < analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bearish')
        
        # 2. RSI ZONE (15 points)
        rsi = analysis.get('rsi')
        if rsi:
            if direction == 'LONG' and 30 < rsi < 55:
                score += 15
                reasons.append(f'RSI optimal ({rsi:.1f})')
            elif direction == 'SHORT' and 45 < rsi < 70:
                score += 15
                reasons.append(f'RSI optimal ({rsi:.1f})')
            elif direction == 'LONG' and 55 <= rsi < 65:
                score += 10
                reasons.append(f'RSI acceptable ({rsi:.1f})')
            elif direction == 'SHORT' and 35 < rsi <= 45:
                score += 10
                reasons.append(f'RSI acceptable ({rsi:.1f})')
        
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
        
        # 4. ADX TREND STRENGTH (10 points)
        adx = analysis.get('adx')
        if adx:
            if adx > 25:
                score += 10
                reasons.append(f'Strong trend (ADX {adx:.1f})')
            elif adx > 20:
                score += 7
                reasons.append(f'Moderate trend (ADX {adx:.1f})')
            elif adx > 15:
                score += 5
        
        # 5. CANDLESTICK PATTERN (10 points)
        patterns = analysis.get('patterns', {})
        if direction == 'LONG':
            if patterns.get('bullish_engulfing') or patterns.get('morning_star'):
                score += 10
                reasons.append('Bullish pattern')
            elif patterns.get('hammer'):
                score += 7
                reasons.append('Hammer')
        else:
            if patterns.get('bearish_engulfing') or patterns.get('evening_star'):
                score += 10
                reasons.append('Bearish pattern')
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
        
        # 7. ORDER FLOW (10 points)
        order_flow = smart.get('order_flow', 0)
        if direction == 'LONG' and order_flow > 0.3:
            score += 10
            reasons.append(f'Strong buy flow ({order_flow:.2f})')
        elif direction == 'LONG' and order_flow > 0.15:
            score += 7
            reasons.append(f'Buy flow ({order_flow:.2f})')
        elif direction == 'SHORT' and order_flow < -0.3:
            score += 10
            reasons.append(f'Strong sell flow ({order_flow:.2f})')
        elif direction == 'SHORT' and order_flow < -0.15:
            score += 7
            reasons.append(f'Sell flow ({order_flow:.2f})')
        
        # 8. MTF ALIGNMENT (10 points)
        score += mtf_score
        if mtf_score >= 8:
            reasons.append('MTF fully aligned')
        elif mtf_score >= 5:
            reasons.append('MTF aligned')
        
        return score, reasons, smart.get('market_regime', 'unknown')
    
    def get_quality_tier(self, score):
        if score >= self.config.HIGH_QUALITY_THRESHOLD:
            return 'HIGH', '🟢'
        elif score >= self.config.MEDIUM_QUALITY_THRESHOLD:
            return 'MEDIUM', '🟡'
        else:
            return 'LOW', '🔴'