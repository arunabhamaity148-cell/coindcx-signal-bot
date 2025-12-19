from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic
from mtf_logic import MTFLogic
from scoring import ScoringEngine
from volume_whale import VolumeWhaleDetector
from signal_tracker import SignalTracker
import requests
import hmac
import hashlib
import json
import time
from datetime import datetime

class SignalGenerator:

    def __init__(self):
        self.config = Config()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()
        self.mtf = MTFLogic()
        self.scorer = ScoringEngine(self.config)
        self.volume_whale = VolumeWhaleDetector()
        self.tracker = SignalTracker()
        self.futures_prices = {}
        self.price_validation_tolerance = 0.30

    def get_authenticated_headers(self, secret_key, api_key, payload):
        try:
            json_payload = json.dumps(payload, separators=(',', ':'))
            signature = hmac.new(
                secret_key.encode('utf-8'),
                json_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': api_key,
                'X-AUTH-SIGNATURE': signature
            }
        except:
            return None

    def fetch_candles_authenticated(self, market, interval='5m', limit=100):
        try:
            url = f"{self.config.COINDCX_BASE_URL}/exchange/v1/candles"
            interval_map = {'5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
            
            payload = {'pair': market, 'interval': interval_map.get(interval, interval), 'limit': limit}
            headers = self.get_authenticated_headers(self.config.COINDCX_API_SECRET, self.config.COINDCX_API_KEY, payload)
            
            if not headers:
                return None
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'candles' not in data:
                return None
            
            candles = []
            for candle in data['candles']:
                candles.append({
                    'time': candle['time'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            return candles
        except:
            return None

    def fetch_candles_public(self, market, interval='5m', limit=100):
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
        except:
            return None

    def get_live_futures_price(self, futures_market):
        try:
            url = f"{self.config.COINDCX_BASE_URL}/exchange/ticker"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            for ticker in data:
                if ticker.get('market') == futures_market:
                    last_price = float(ticker.get('last_price', 0))
                    bid = float(ticker.get('bid', 0))
                    ask = float(ticker.get('ask', 0))
                    
                    if last_price > 0 and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        self.futures_prices[futures_market] = (time.time(), mid_price, bid, ask)
                        return mid_price, bid, ask
            
            return None, None, None
        except:
            return None, None, None

    def validate_price_sanity(self, spot_price, futures_price, market_name):
        if not spot_price or not futures_price:
            return False, "Missing price"
        
        diff_pct = abs(futures_price - spot_price) / spot_price * 100
        
        if diff_pct > 30:
            return False, f"Price mismatch {diff_pct:.1f}%"
        
        ratio = futures_price / spot_price if spot_price > 0 else 0
        
        if ratio > 5 or ratio < 0.2:
            return False, f"Decimal error ratio={ratio:.2f}"
        
        return True, "OK"

    def fetch_candles(self, market, interval='5m', limit=100):
        if self.config.USE_AUTHENTICATED_API and self.config.COINDCX_API_KEY:
            if market.startswith('F-'):
                candles = self.fetch_candles_authenticated(market, interval, limit)
                if candles:
                    return candles
                spot_market = None
                for spot, futures in self.config.SPOT_TO_FUTURES_MAP.items():
                    if futures == market:
                        spot_market = spot
                        break
                if spot_market:
                    return self.fetch_candles_public(spot_market, interval, limit)
        
        return self.fetch_candles_public(market, interval, limit)

    def check_btc_stability(self):
        if not self.config.ENABLE_BTC_CHECK:
            return True, 'BTC check disabled', 'neutral'
        
        try:
            btc_5m = self.fetch_candles(self.config.BTC_PAIR, '5m', 20)
            
            if not btc_5m or len(btc_5m) < 10:
                return True, 'BTC data unavailable', 'neutral'
            
            closes = [c['close'] for c in btc_5m]
            highs = [c['high'] for c in btc_5m]
            lows = [c['low'] for c in btc_5m]
            
            for candle in btc_5m[-3:]:
                body = candle['close'] - candle['open']
                body_pct = abs(body) / candle['open'] * 100 if candle['open'] > 0 else 0
                
                if body < 0 and body_pct > self.config.BTC_DUMP_THRESHOLD:
                    return False, f'BTC dump -{body_pct:.1f}%', 'dump'
            
            atr = self.indicators.calculate_atr(highs, lows, closes, 14)
            volatility = self.indicators.calculate_volatility(closes, atr)
            
            if volatility and volatility > self.config.BTC_VOLATILITY_THRESHOLD:
                return False, f'BTC volatile {volatility:.1f}%', 'volatile'
            
            return True, 'BTC stable', 'stable'
        except:
            return True, 'BTC check error', 'neutral'

    def analyze_market(self, candles_data):
        if not candles_data or len(candles_data) < self.config.MIN_CANDLES_REQUIRED:
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
        if adx and adx < self.config.MIN_ADX_THRESHOLD:
            return False, f'Weak ADX ({adx:.1f})'
        
        if regime == 'ranging' and score < self.config.BLOCK_RANGING_SCORE:
            return False, f'Ranging (need {self.config.BLOCK_RANGING_SCORE}+)'
        
        if regime == 'volatile' and score < self.config.BLOCK_VOLATILE_SCORE:
            return False, f'Volatile (need {self.config.BLOCK_VOLATILE_SCORE}+)'
        
        if atr and atr < self.config.MIN_ATR_THRESHOLD:
            return False, 'ATR too low'
        
        return True, None

    def generate_signal(self, market, candles_5m, candles_15m, candles_1h):
        """
        COINDCX INR OPTIMIZED - Price action priority, detailed logging
        """
        print(f"      📈 Analyzing...")
        
        analysis_5m = self.analyze_market(candles_5m)
        if not analysis_5m:
            print(f"      ❌ REJECT: Need {self.config.MIN_CANDLES_REQUIRED}+ candles, got {len(candles_5m) if candles_5m else 0}")
            return None
        
        print(f"         RSI: {analysis_5m['rsi']:.1f}, ADX: {analysis_5m['adx']:.1f}, Regime: {analysis_5m['smart']['market_regime']}")
        
        # Volume/Whale/Liquidity detection
        volume_surge, surge_ratio, vol_direction = self.volume_whale.detect_volume_surge(candles_5m)
        whale_detected, whale_pct, whale_direction = self.volume_whale.detect_whale_candle(candles_5m[-1])
        liq_sweep, sweep_type, sweep_strength = self.volume_whale.detect_liquidity_sweep_advanced(candles_5m)
        
        if volume_surge or whale_detected or liq_sweep:
            print(f"         🎯 Special: Vol={volume_surge}, Whale={whale_detected}, Sweep={liq_sweep}")
        
        # MTF (optional for CoinDCX)
        trend_15m = self.mtf.get_trend_direction(candles_15m) if candles_15m else 'neutral'
        bias_1h = self.mtf.get_trend_direction(candles_1h) if candles_1h else 'neutral'
        
        print(f"         MTF: 15m={trend_15m}, 1h={bias_1h}")
        
        long_aligned, _ = self.mtf.check_mtf_alignment(trend_15m, bias_1h, 'LONG', self.config.MTF_STRICT_MODE)
        short_aligned, _ = self.mtf.check_mtf_alignment(trend_15m, bias_1h, 'SHORT', self.config.MTF_STRICT_MODE)
        
        # MTF check only if required (disabled by default)
        if self.config.REQUIRE_MTF_ALIGNMENT:
            if not long_aligned and not short_aligned:
                print(f"      ❌ REJECT: MTF - LONG blocked, SHORT blocked")
                return None
        
        # Scores
        long_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'LONG')
        short_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'SHORT')
        
        long_base, long_reasons, long_regime = self.scorer.calculate_base_score(analysis_5m, 'LONG', long_mtf_score)
        short_base, short_reasons, short_regime = self.scorer.calculate_base_score(analysis_5m, 'SHORT', short_mtf_score)
        
        print(f"         Scores: LONG={long_base}, SHORT={short_base}")
        
        # Pick direction - both allowed if not strict MTF
        if long_base >= short_base and (long_aligned or not self.config.REQUIRE_MTF_ALIGNMENT):
            direction = 'LONG'
            base_score = long_base
            reasons = long_reasons
            regime = long_regime
        elif short_aligned or not self.config.REQUIRE_MTF_ALIGNMENT:
            direction = 'SHORT'
            base_score = short_base
            reasons = short_reasons
            regime = short_regime
        else:
            print(f"      ❌ REJECT: No valid direction (L:{long_base}/{long_aligned}, S:{short_base}/{short_aligned})")
            return None
        
        # Bonuses
        volume_data = {'is_surge': volume_surge, 'surge_ratio': surge_ratio, 'direction': vol_direction}
        whale_data = {'is_whale': whale_detected, 'move_pct': whale_pct, 'direction': whale_direction}
        liquidity_data = {'is_sweep': liq_sweep, 'type': sweep_type, 'strength': sweep_strength}
        
        final_score, final_reasons, bonus_applied = self.scorer.apply_advanced_bonuses(
            base_score, reasons, analysis_5m, direction, volume_data, whale_data, liquidity_data
        )
        
        if bonus_applied:
            print(f"         💎 Bonus: {base_score} → {final_score}")
        
        # Volume/Whale requirement check (disabled by default for CoinDCX)
        if self.config.REQUIRE_VOLUME_OR_WHALE:
            if not bonus_applied and final_score < self.config.PERFECT_SETUP_THRESHOLD:
                print(f"      ❌ REJECT: Need volume/whale confirmation (score={final_score})")
                return None
        
        # Time multiplier (disabled for CoinDCX)
        final_score = self.scorer.apply_time_multiplier(final_score, datetime.utcnow().hour)
        
        # MIN SCORE CHECK
        if final_score < self.config.MIN_SIGNAL_SCORE:
            print(f"      ❌ REJECT: Score {final_score} < {self.config.MIN_SIGNAL_SCORE} (Base:{base_score}, Bonus:{final_score-base_score})")
            return None
        
        # Hard filters (very relaxed)
        passed, block_reason = self.apply_hard_filters(final_score, regime, analysis_5m['adx'], analysis_5m['atr'])
        if not passed:
            print(f"      ❌ REJECT: Filter - {block_reason}")
            return None
        
        # Price validation
        spot_price = analysis_5m['price']
        entry_price = spot_price
        futures_market = None
        
        if market.startswith('B-') and market in self.config.SPOT_TO_FUTURES_MAP:
            futures_market = self.config.SPOT_TO_FUTURES_MAP[market]
            futures_price, bid, ask = self.get_live_futures_price(futures_market)
            
            if not futures_price:
                print(f"      ❌ REJECT: No futures price for {futures_market}")
                return None
            
            is_valid, reason = self.validate_price_sanity(spot_price, futures_price, futures_market)
            
            if not is_valid:
                print(f"      ❌ REJECT: Price invalid - {reason}")
                return None
            
            entry_price = futures_price
            print(f"         ✓ Price: ₹{futures_price:.2f}")
        
        # Levels
        atr = analysis_5m['atr']
        
        if futures_market and abs(entry_price - spot_price) / spot_price > 0.05:
            atr_scale = entry_price / spot_price
            atr = atr * atr_scale
        
        if direction == 'LONG':
            sl = entry_price - (atr * self.config.ATR_SL_MULTIPLIER)
            tp1 = entry_price + (atr * self.config.ATR_TP1_MULTIPLIER)
            tp2 = entry_price + (atr * self.config.ATR_TP2_MULTIPLIER)
        else:
            sl = entry_price + (atr * self.config.ATR_SL_MULTIPLIER)
            tp1 = entry_price - (atr * self.config.ATR_TP1_MULTIPLIER)
            tp2 = entry_price - (atr * self.config.ATR_TP2_MULTIPLIER)
        
        risk = abs(entry_price - sl)
        reward = abs(tp2 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < self.config.MIN_RR_RATIO:
            print(f"      ❌ REJECT: R:R {rr_ratio:.2f} < {self.config.MIN_RR_RATIO}")
            return None
        
        quality_tier, emoji = self.scorer.get_quality_tier(final_score)
        
        display_market = futures_market if futures_market else market
        
        print(f"      ✅ SIGNAL APPROVED: {direction} {final_score} {emoji} R:R={rr_ratio:.2f}")
        
        return {
            'market': display_market,
            'direction': direction,
            'entry': entry_price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'score': final_score,
            'quality_tier': quality_tier,
            'quality_emoji': emoji,
            'rr_ratio': rr_ratio,
            'reasons': final_reasons,
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
        if not analysis_5m:
            print(f"      ⚠️ SKIP: Analysis failed (need {self.config.MIN_CANDLES_REQUIRED}+ candles)")
            return None
        
        # ADVANCED: Volume surge detection
        volume_surge, surge_ratio, vol_direction = self.volume_whale.detect_volume_surge(candles_5m)
        
        # ADVANCED: Whale candle detection
        whale_detected, whale_pct, whale_direction = self.volume_whale.detect_whale_candle(candles_5m[-1])
        
        # ADVANCED: Liquidity sweep detection
        liq_sweep, sweep_type, sweep_strength = self.volume_whale.detect_liquidity_sweep_advanced(candles_5m)
        
        # MTF trends
        trend_15m = self.mtf.get_trend_direction(candles_15m) if candles_15m else 'neutral'
        bias_1h = self.mtf.get_trend_direction(candles_1h) if candles_1h else 'neutral'
        
        # MTF alignment
        long_aligned, long_mtf_reason = self.mtf.check_mtf_alignment(trend_15m, bias_1h, 'LONG', self.config.MTF_STRICT_MODE)
        short_aligned, short_mtf_reason = self.mtf.check_mtf_alignment(trend_15m, bias_1h, 'SHORT', self.config.MTF_STRICT_MODE)
        
        if self.config.REQUIRE_MTF_ALIGNMENT:
            if not long_aligned and not short_aligned:
                print(f"      ⛔ SKIP: MTF blocked")
                print(f"         LONG: {long_mtf_reason} (15m={trend_15m}, 1h={bias_1h})")
                print(f"         SHORT: {short_mtf_reason}")
                return None
        
        # Calculate MTF scores
        long_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'LONG')
        short_mtf_score = self.mtf.get_mtf_score(trend_15m, bias_1h, 'SHORT')
        
        # BASE SCORING
        long_base, long_reasons, long_regime = self.scorer.calculate_base_score(analysis_5m, 'LONG', long_mtf_score)
        short_base, short_reasons, short_regime = self.scorer.calculate_base_score(analysis_5m, 'SHORT', short_mtf_score)
        
        # Pick direction
        if long_base >= short_base and long_aligned:
            direction = 'LONG'
            base_score = long_base
            reasons = long_reasons
            regime = long_regime
        elif short_aligned:
            direction = 'SHORT'
            base_score = short_base
            reasons = short_reasons
            regime = short_regime
        else:
            print(f"      ⛔ SKIP: No aligned direction")
        