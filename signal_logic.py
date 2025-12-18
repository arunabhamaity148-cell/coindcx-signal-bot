from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic
from mtf_logic import MTFLogic
from scoring import ScoringEngine
import requests

class SignalGenerator:

    def __init__(self):
        self.config = Config()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()
        self.mtf = MTFLogic()
        self.scorer = ScoringEngine(self.config)
        self.last_candle_time = {}

    def fetch_candles(self, market, interval='5m', limit=100):
        """Fetch candles from CoinDCX"""
        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {'pair': market, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return None
            
            candles = []
            for candle in data:
                candles.append({
                    'time': candle['time'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            return candles
        except Exception as e:
            print(f"   ⚠️ Fetch error {market} {interval}: {str(e)}")
            return None

    def check_btc_stability(self):
        """
        OPTIONAL BTC check - only if enabled
        Uses SPOT market (B-BTC_USDT) which has reliable data
        """
        if not self.config.ENABLE_BTC_CHECK:
            return True, 'BTC check disabled', 'neutral'
        
        try:
            btc_5m = self.fetch_candles(self.config.BTC_PAIR, '5m', 20)
            
            if not btc_5m or len(btc_5m) < 10:
                # No data = neutral (don't block)
                return True, 'BTC data unavailable (neutral)', 'neutral'
            
            closes = [c['close'] for c in btc_5m]
            highs = [c['high'] for c in btc_5m]
            lows = [c['low'] for c in btc_5m]
            
            # Only check for extreme dump (>5% red candle)
            for candle in btc_5m[-3:]:
                body = candle['close'] - candle['open']
                body_pct = abs(body) / candle['open'] * 100 if candle['open'] > 0 else 0
                
                if body < 0 and body_pct > 5.0:
                    return False, f'BTC extreme dump (-{body_pct:.1f}%)', 'dump'
            
            # Check volatility (only extreme)
            atr = self.indicators.calculate_atr(highs, lows, closes, 14)
            volatility = self.indicators.calculate_volatility(closes, atr)
            
            if volatility and volatility > 8.0:  # Very high threshold
                return False, f'BTC extreme volatility ({volatility:.1f}%)', 'volatile'
            
            return True, 'BTC stable', 'stable'
            
        except Exception as e:
            # On error, don't block
            return True, f'BTC check error (neutral)', 'neutral'

    def check_same_candle(self, market, candle_time):
        """Prevent duplicate signals on exact same candle"""
        if market in self.last_candle_time:
            if self.last_candle_time[market] == candle_time:
                return False
        self.last_candle_time[market] = candle_time
        return True

    def analyze_market(self, candles_data):
        """
        Analyze with RELAXED data requirements
        Works with 50+ candles instead of 100
        """
        if not candles_data:
            return None
        
        # RELAXED: Accept 50+ candles
        if len(candles_data) < self.config.MIN_CANDLES_REQUIRED:
            return None

        candles = candles_data[-100:] if len(candles_data) >= 100 else candles_data
        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]

        ema_fast = self.indicators.calculate_ema(closes, self.config.EMA_FAST)
        ema_slow = self.indicators.calculate_ema(closes, self.config.EMA_SLOW)
        rsi = self.indicators.calculate_rsi(closes, self.config.RSI_PERIOD)
        macd, signal, histogram = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(highs, lows, closes, self.config.ATR_PERIOD)
        adx = self.indicators.calculate_adx(highs, lows, closes, self.config.ADX_PERIOD)

        patterns_result = {
            'bullish_engulfing': self.patterns.is_bullish_engulfing(candles),
            'bearish_engulfing': self.patterns.is_bearish_engulfing(candles),
            'hammer': self.patterns.is_hammer(candles[-1]),
            'shooting_star': self.patterns.is_shooting_star(candles[-1]),
            'morning_star': self.patterns.is_morning_star(candles),
            'evening_star': self.patterns.is_evening_star(candles)
        }

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
            'patterns': patterns_result,
            'smart': {
                'market_regime': market_regime,
                'liquidity_grab': liquidity_grab,
                'order_flow': order_flow
            }
        }

    def apply_hard_filters(self, score, regime, adx, atr):
        """
        RELAXED hard filters for INR futures
        Only block extreme cases
        """
        # Filter 1: Very weak trend (relaxed)
        if adx and adx < self.config.MIN_ADX_THRESHOLD:
            # Only block if score is also very low
            if score < 50:
                return False, f'Weak ADX ({adx:.1f}) + Low score'
        
        # Filter 2: Ranging market (relaxed)
        if regime == 'ranging' and score < self.config.BLOCK_RANGING_SCORE:
            return False, f'Ranging (score {score})'
        
        # Filter 3: Volatile market (relaxed)
        if regime == 'volatile' and score < self.config.BLOCK_VOLATILE_SCORE:
            return False, f'Volatile (score {score})'
        
        # Filter 4: ATR too low (VERY relaxed for INR)
        if atr and atr < self.config.MIN_ATR_THRESHOLD:
            return False, 'ATR too low'
        
        return True, None

    def generate_signal(self, market, candles_5m, candles_15m, candles_1h):
        """Generate signal with RELAXED MTF requirements"""
        
        # Same candle check
        if candles_5m and not self.check_same_candle(market, candles_5m[-1]['time']):
            return None
        
        # Analyze 5m timeframe
        analysis_5m = self.analyze_market(candles_5m)
        if not analysis_5m:
            return None
        
        # Get MTF trends (with fallback for missing data)
        trend_15m = self.mtf.get_trend_direction(candles_15m) if candles_15m else 'neutral'
        bias_1h = self.mtf.get_trend_direction(candles_1h) if candles_1h else 'neutral'
        
        # Check both directions with RELAXED MTF
        long_aligned, _ = self.mtf.check_mtf_alignment(
            trend_15m, bias_1h, 'LONG', self.config.MTF_STRICT_MODE
        )
        short_aligned, _ = self.mtf.check_mtf_alignment(
            trend_15m, bias_1h, 'SHORT', self.config.MTF_STRICT_MODE
        )
        
        # Calculate scores
        long_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'LONG')
        short_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'SHORT')
        
        long_score, long_reasons, long_regime = self.scorer.calculate_score(
            analysis_5m, 'LONG', long_mtf_score
        )
        short_score, short_reasons, short_regime = self.scorer.calculate_score(
            analysis_5m, 'SHORT', short_mtf_score
        )
        
        # Pick best direction (allow even without perfect MTF)
        if long_score >= short_score and long_score >= self.config.MIN_SIGNAL_SCORE and long_aligned:
            direction = 'LONG'
            score = long_score
            reasons = long_reasons
            regime = long_regime
        elif short_score >= self.config.MIN_SIGNAL_SCORE and short_aligned:
            direction = 'SHORT'
            score = short_score
            reasons = short_reasons
            regime = short_regime
        else:
            return None
        
        # Apply RELAXED hard filters
        passed, block_reason = self.apply_hard_filters(
            score, regime, analysis_5m['adx'], analysis_5m['atr']
        )
        if not passed:
            return None
        
        # Calculate levels
        entry = analysis_5m['price']
        atr = analysis_5m['atr']
        
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
            return None
        
        quality_tier, emoji = self.scorer.get_quality_tier(score)
        
        return {
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
            'mtf': {
                'trend_15m': trend_15m,
                'bias_1h': bias_1h
            },
            'analysis': {
                'rsi': analysis_5m['rsi'],
                'adx': analysis_5m['adx'],
                'market_regime': regime
            }
        }