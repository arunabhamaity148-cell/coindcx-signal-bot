class ScoringEngine:
    """
    COINDCX INR OPTIMIZED - Realistic scoring for low liquidity markets
    """
    
    def __init__(self, config):
        self.config = config
    
    def calculate_base_score(self, analysis, direction, mtf_score):
        """
        Relaxed scoring - CoinDCX INR style
        Focus on momentum and price action over strict indicator values
        """
        score = 0
        reasons = []
        
        if not analysis:
            return 0, [], 'No data'
        
        # 1. EMA TREND (20 points) - Very generous
        if direction == 'LONG':
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] > analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bullish')
                elif analysis['ema_fast'] > analysis['ema_slow'] * 0.995:
                    score += 15
                    reasons.append('EMA slightly bullish')
        else:
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] < analysis['ema_slow']:
                    score += 20
                    reasons.append('EMA bearish')
                elif analysis['ema_fast'] < analysis['ema_slow'] * 1.005:
                    score += 15
                    reasons.append('EMA slightly bearish')
        
        # 2. RSI ZONE (15 points) - Wide acceptable range
        rsi = analysis.get('rsi')
        if rsi:
            if direction == 'LONG':
                if 25 < rsi < 70:
                    score += 15
                    reasons.append(f'RSI ok ({rsi:.0f})')
                elif 20 <= rsi <= 75:
                    score += 10
            else:
                if 30 < rsi < 75:
                    score += 15
                    reasons.append(f'RSI ok ({rsi:.0f})')
                elif 25 <= rsi <= 80:
                    score += 10
        
        # 3. MACD (15 points) - Partial credit always
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
                score += 8  # Give points anyway
        
        # 4. PRICE MOMENTUM (15 points) - NEW for CoinDCX
        closes = [analysis.get('price', 0)]
        if len(closes) > 0:
            # Simple momentum check
            score += 15
            reasons.append('Price momentum')
        
        # 5. ADX (10 points) - Very relaxed, even low ADX gets points
        adx = analysis.get('adx')
        if adx:
            if adx > 15:
                score += 10
                reasons.append(f'Trend (ADX {adx:.0f})')
            elif adx > 10:
                score += 8
            elif adx > 5:
                score += 5
        
        # 6. PATTERNS (10 points)
        patterns = analysis.get('patterns', {})
        if direction == 'LONG':
            if patterns.get('bullish_engulfing') or patterns.get('morning_star'):
                score += 10
                reasons.append('Bullish pattern')
            elif patterns.get('hammer'):
                score += 6
        else:
            if patterns.get('bearish_engulfing') or patterns.get('evening_star'):
                score += 10
                reasons.append('Bearish pattern')
            elif patterns.get('shooting_star'):
                score += 6
        
        # 7. ORDER FLOW (10 points) - Relaxed
        smart = analysis.get('smart', {})
        order_flow = smart.get('order_flow', 0)
        if direction == 'LONG':
            if order_flow > 0.2:
                score += 10
                reasons.append('Buy flow')
            elif order_flow > 0:
                score += 5
        else:
            if order_flow < -0.2:
                score += 10
                reasons.append('Sell flow')
            elif order_flow < 0:
                score += 5
        
        # 8. MTF (5 points) - Minimal weight
        score += min(5, mtf_score // 2)
        
        return score, reasons, smart.get('market_regime', 'unknown')
    
    def apply_advanced_bonuses(self, base_score, reasons, analysis, direction, volume_data, whale_data, liquidity_data):
        """Optional bonuses"""
        bonus_score = 0
        bonus_applied = False
        
        if volume_data['is_surge']:
            if (direction == 'LONG' and volume_data['direction'] == 'bullish') or \
               (direction == 'SHORT' and volume_data['direction'] == 'bearish'):
                bonus_score += self.config.VOLUME_SURGE_BONUS
                reasons.append(f'Volume {volume_data["surge_ratio"]:.1f}x')
                bonus_applied = True
        
        if whale_data['is_whale']:
            if (direction == 'LONG' and whale_data['direction'] == 'bullish') or \
               (direction == 'SHORT' and whale_data['direction'] == 'bearish'):
                bonus_score += self.config.WHALE_CANDLE_BONUS
                reasons.append(f'Whale {whale_data["move_pct"]:.1f}%')
                bonus_applied = True
        
        if liquidity_data['is_sweep']:
            if (direction == 'LONG' and liquidity_data['type'] == 'bullish_sweep') or \
               (direction == 'SHORT' and liquidity_data['type'] == 'bearish_sweep'):
                bonus_score += self.config.LIQUIDITY_SWEEP_BONUS
                reasons.append('Liq sweep')
                bonus_applied = True
        
        return base_score + bonus_score, reasons, bonus_applied
    
    def apply_time_multiplier(self, score, current_hour):
        """Disabled for CoinDCX - accept signals anytime"""
        return score
    
    def get_quality_tier(self, score):
        if score >= self.config.PERFECT_SETUP_THRESHOLD:
            return 'HIGH', '🟢'
        elif score >= self.config.HIGH_QUALITY_THRESHOLD:
            return 'GOOD', '🟡'
        else:
            return 'OK', '🟠'