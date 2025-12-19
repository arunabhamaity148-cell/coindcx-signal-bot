class ScoringEngine:
    """
    ADVANCED SMART scoring with dynamic adjustments
    """
    
    def __init__(self, config):
        self.config = config
    
    def calculate_base_score(self, analysis, direction, mtf_score):
        """Calculate base technical score (0-70 points)"""
        score = 0
        reasons = []
        
        if not analysis:
            return 0, [], 'No data'
        
        # 1. EMA TREND (15 points)
        if direction == 'LONG':
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] > analysis['ema_slow']:
                    score += 15
                    reasons.append('EMA bullish')
                elif analysis['ema_fast'] > analysis['ema_slow'] * 0.998:
                    score += 10
        else:
            if analysis.get('ema_fast') and analysis.get('ema_slow'):
                if analysis['ema_fast'] < analysis['ema_slow']:
                    score += 15
                    reasons.append('EMA bearish')
                elif analysis['ema_fast'] < analysis['ema_slow'] * 1.002:
                    score += 10
        
        # 2. RSI ZONE (12 points)
        rsi = analysis.get('rsi')
        if rsi:
            if direction == 'LONG':
                if 30 < rsi < 55:
                    score += 12
                    reasons.append(f'RSI optimal ({rsi:.0f})')
                elif 25 < rsi <= 65:
                    score += 8
            else:
                if 45 < rsi < 70:
                    score += 12
                    reasons.append(f'RSI optimal ({rsi:.0f})')
                elif 35 <= rsi <= 75:
                    score += 8
        
        # 3. MACD (12 points)
        macd = analysis.get('macd')
        macd_signal = analysis.get('macd_signal')
        if macd and macd_signal:
            if direction == 'LONG' and macd > macd_signal:
                score += 12
                reasons.append('MACD bullish')
            elif direction == 'SHORT' and macd < macd_signal:
                score += 12
                reasons.append('MACD bearish')
        
        # 4. ADX (8 points)
        adx = analysis.get('adx')
        if adx:
            if adx > 20:
                score += 8
                reasons.append(f'Strong trend (ADX {adx:.0f})')
            elif adx > 15:
                score += 6
            elif adx > 12:
                score += 4
        
        # 5. PATTERNS (8 points)
        patterns = analysis.get('patterns', {})
        if direction == 'LONG':
            if patterns.get('bullish_engulfing') or patterns.get('morning_star'):
                score += 8
                reasons.append('Bullish pattern')
            elif patterns.get('hammer'):
                score += 5
        else:
            if patterns.get('bearish_engulfing') or patterns.get('evening_star'):
                score += 8
                reasons.append('Bearish pattern')
            elif patterns.get('shooting_star'):
                score += 5
        
        # 6. ORDER FLOW (7 points)
        smart = analysis.get('smart', {})
        order_flow = smart.get('order_flow', 0)
        if direction == 'LONG':
            if order_flow > 0.4:
                score += 7
                reasons.append('Strong buy flow')
            elif order_flow > 0.2:
                score += 5
        else:
            if order_flow < -0.4:
                score += 7
                reasons.append('Strong sell flow')
            elif order_flow < -0.2:
                score += 5
        
        # 7. MTF (8 points)
        score += mtf_score
        if mtf_score >= 8:
            reasons.append('MTF aligned')
        
        return score, reasons, smart.get('market_regime', 'unknown')
    
    def apply_advanced_bonuses(self, base_score, reasons, analysis, direction, volume_data, whale_data, liquidity_data):
        """
        Apply SMART bonuses based on volume/whale/liquidity
        Returns: (final_score, updated_reasons, bonus_applied)
        """
        bonus_score = 0
        bonus_applied = False
        
        # VOLUME SURGE BONUS (up to 15 points)
        if volume_data['is_surge']:
            if (direction == 'LONG' and volume_data['direction'] == 'bullish') or \
               (direction == 'SHORT' and volume_data['direction'] == 'bearish'):
                surge_bonus = min(15, int(volume_data['surge_ratio'] * 3))
                bonus_score += surge_bonus
                reasons.append(f'Volume surge {volume_data["surge_ratio"]:.1f}x')
                bonus_applied = True
        
        # WHALE CANDLE BONUS (up to 10 points)
        if whale_data['is_whale']:
            if (direction == 'LONG' and whale_data['direction'] == 'bullish') or \
               (direction == 'SHORT' and whale_data['direction'] == 'bearish'):
                whale_bonus = min(10, int(whale_data['move_pct'] * 3))
                bonus_score += whale_bonus
                reasons.append(f'Whale move {whale_data["move_pct"]:.1f}%')
                bonus_applied = True
        
        # LIQUIDITY SWEEP BONUS (up to 12 points)
        if liquidity_data['is_sweep']:
            if (direction == 'LONG' and liquidity_data['type'] == 'bullish_sweep') or \
               (direction == 'SHORT' and liquidity_data['type'] == 'bearish_sweep'):
                sweep_bonus = liquidity_data['strength'] * 4
                bonus_score += sweep_bonus
                reasons.append(f'Liquidity sweep (S:{liquidity_data["strength"]})')
                bonus_applied = True
        
        final_score = base_score + bonus_score
        
        return final_score, reasons, bonus_applied
    
    def apply_time_multiplier(self, score, current_hour):
        """Apply time-of-day multiplier"""
        if not self.config.TIME_OF_DAY_MULTIPLIER:
            return score
        
        # Peak hours bonus
        if 4 <= current_hour <= 9:  # 10 AM - 3 PM IST
            return int(score * 1.15)
        elif 10 <= current_hour <= 14:
            return int(score * 1.05)
        
        return score
    
    def get_quality_tier(self, score):
        if score >= self.config.PERFECT_SETUP_THRESHOLD:
            return 'PERFECT', '🟢'
        elif score >= self.config.HIGH_QUALITY_THRESHOLD:
            return 'HIGH', '🟢'
        elif score >= self.config.BASE_MIN_SCORE:
            return 'GOOD', '🟡'
        else:
            return 'LOW', '🔴'