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

    # ======================================================================
    # 🔥 DIRECT SPOT CANDLE FETCHER (NO MAPPING, NO BUGS)
    # ======================================================================
    def fetch_candles_direct(self, pair, interval='5m', limit=100):
        """Fetch SPOT candles directly from CoinDCX."""
        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {'pair': pair, 'interval': interval, 'limit': limit}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                return None

            candles = []
            for c in data:
                candles.append({
                    'time': c['time'],
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': float(c['volume'])
                })
            return candles

        except Exception as e:
            print(f"⚠️ Fetch error {pair}: {e}")
            return None

    # ======================================================================
    # BTC RELAXED CHECK
    # ======================================================================
    def check_btc_stability(self):
        if not self.config.ENABLE_BTC_CHECK:
            return True, 'BTC check disabled', 'neutral'
        try:
            btc_5m = self.fetch_candles_direct("B-BTC_USDT", '5m', 20)
            if not btc_5m or len(btc_5m) < 10:
                return True, 'BTC data missing → neutral', 'neutral'

            closes = [c['close'] for c in btc_5m]
            highs = [c['high'] for c in btc_5m]
            lows = [c['low'] for c in btc_5m]

            # Extreme Dump (>5%)
            for candle in btc_5m[-3:]:
                body = candle['close'] - candle['open']
                pct = abs(body) / candle['open'] * 100
                if body < 0 and pct > 5:
                    return False, f'BTC dump {pct:.1f}%', 'dump'

            # Volatility Check
            atr = self.indicators.calculate_atr(highs, lows, closes)
            vol = self.indicators.calculate_volatility(closes, atr)
            if vol and vol > 8:
                return False, f'BTC volatile {vol:.1f}%', 'volatile'

            return True, 'BTC stable', 'stable'

        except:
            return True, 'BTC check error → neutral', 'neutral'

    # ======================================================================
    def check_same_candle(self, market, candle_time):
        if market in self.last_candle_time:
            if self.last_candle_time[market] == candle_time:
                return False
        self.last_candle_time[market] = candle_time
        return True

    # ======================================================================
    def analyze_market(self, candles):
        if not candles:
            return None

        if len(candles) < self.config.MIN_CANDLES_REQUIRED:
            return None

        candles = candles[-100:] if len(candles) > 100 else candles

        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]

        ema_fast = self.indicators.calculate_ema(closes, self.config.EMA_FAST)
        ema_slow = self.indicators.calculate_ema(closes, self.config.EMA_SLOW)
        rsi = self.indicators.calculate_rsi(closes)
        macd, sig, hist = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(highs, lows, closes)
        adx = self.indicators.calculate_adx(highs, lows, closes)

        patterns = {
            'bullish_engulfing': self.patterns.is_bullish_engulfing(candles),
            'bearish_engulfing': self.patterns.is_bearish_engulfing(candles),
            'hammer': self.patterns.is_hammer(candles[-1]),
            'shooting_star': self.patterns.is_shooting_star(candles[-1]),
            'morning_star': self.patterns.is_morning_star(candles),
            'evening_star': self.patterns.is_evening_star(candles)
        }

        regime = self.smart.detect_market_regime(closes, atr, adx)
        liquidity = self.smart.detect_liquidity_grab(candles)
        flow = self.smart.calculate_order_flow(candles)

        return {
            'price': closes[-1],
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': sig,
            'atr': atr,
            'adx': adx,
            'patterns': patterns,
            'smart': {
                'market_regime': regime,
                'liquidity_grab': liquidity,
                'order_flow': flow
            }
        }

    # ======================================================================
    def apply_hard_filters(self, score, regime, adx, atr):
        if adx and adx < self.config.MIN_ADX_THRESHOLD and score < 50:
            return False, "Weak ADX + Low Score"

        if regime == 'ranging' and score < self.config.BLOCK_RANGING_SCORE:
            return False, f"Ranging {score}"

        if regime == 'volatile' and score < self.config.BLOCK_VOLATILE_SCORE:
            return False, f"Volatile {score}"

        if atr and atr < self.config.MIN_ATR_THRESHOLD:
            return False, "ATR too low"

        return True, None

    # ======================================================================
    # 🔥 FINAL SIGNAL GENERATOR (SPOT-ONLY)
    # ======================================================================
    def generate_signal(self, market):
        """Pure SPOT analysis & signal generation."""

        # Fetch SPOT candles directly
        c5 = self.fetch_candles_direct(market, '5m', 100)
        c15 = self.fetch_candles_direct(market, '15m', 100)
        c1h = self.fetch_candles_direct(market, '1h', 100)

        if not c5:
            return None

        # Same candle protect
        if not self.check_same_candle(market, c5[-1]['time']):
            return None

        # Core analysis
        a5 = self.analyze_market(c5)
        if not a5:
            return None

        trend = self.mtf.get_trend_direction(c15) if c15 else 'neutral'
        bias = self.mtf.get_trend_direction(c1h) if c1h else 'neutral'

        long_score, long_reasons, regime_long = self.scorer.calculate_score(
            a5, 'LONG', self.mtf.get_mtf_score(trend, bias, 'LONG')
        )
        short_score, short_reasons, regime_short = self.scorer.calculate_score(
            a5, 'SHORT', self.mtf.get_mtf_score(trend, bias, 'SHORT')
        )

        # Pick best direction
        if long_score >= short_score and long_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'LONG'
            score = long_score
            reasons = long_reasons
            regime = regime_long

        elif short_score >= self.config.MIN_SIGNAL_SCORE:
            direction = 'SHORT'
            score = short_score
            reasons = short_reasons
            regime = regime_short

        else:
            return None

        # Hard filters
        ok, reason = self.apply_hard_filters(score, regime, a5['adx'], a5['atr'])
        if not ok:
            return None

        entry = a5['price']
        atr = a5['atr']

        # Levels
        if direction == 'LONG':
            sl = entry - atr * self.config.ATR_SL_MULTIPLIER
            tp1 = entry + atr * self.config.ATR_TP1_MULTIPLIER
            tp2 = entry + atr * self.config.ATR_TP2_MULTIPLIER
        else:
            sl = entry + atr * self.config.ATR_SL_MULTIPLIER
            tp1 = entry - atr * self.config.ATR_TP1_MULTIPLIER
            tp2 = entry - atr * self.config.ATR_TP2_MULTIPLIER

        return {
            'market': market,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'direction': direction,
            'score': score,
            'reasons': reasons,
            'analysis': a5,
            'regime': regime
        }