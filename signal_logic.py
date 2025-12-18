from config import Config
from indicators import TechnicalIndicators
from patterns import CandlestickPatterns
from smart_logic import SmartMoneyLogic
from mtf_logic import MTFLogic
from scoring import ScoringEngine
import requests

# 🔥 SPOT → FUTURES Mapping
SPOT_TO_FUTURES = {
    'BTC_INR': 'F-BTC_INR',
    'ETH_INR': 'F-ETH_INR',
    'SOL_INR': 'F-SOL_INR',
    'MATIC_INR': 'F-MATIC_INR',
    'XRP_INR': 'F-XRP_INR',
    'ADA_INR': 'F-ADA_INR',
    'DOGE_INR': 'F-DOGE_INR',
    'DOT_INR': 'F-DOT_INR',
    'LTC_INR': 'F-LTC_INR',
    'LINK_INR': 'F-LINK_INR',
    'UNI_INR': 'F-UNI_INR',
    'AVAX_INR': 'F-AVAX_INR',
    'ATOM_INR': 'F-ATOM_INR',
    'TRX_INR': 'F-TRX_INR',
    'SHIB_INR': 'F-SHIB_INR',
    'ARB_INR': 'F-ARB_INR',
    'OP_INR': 'F-OP_INR',
    'APT_INR': 'F-APT_INR',
    'SUI_INR': 'F-SUI_INR',
    'INJ_INR': 'F-INJ_INR'
}

# Reverse lookup (Futures → Spot)
FUTURES_TO_SPOT = {v: k for k, v in SPOT_TO_FUTURES.items()}


class SignalGenerator:

    def __init__(self):
        self.config = Config()
        self.indicators = TechnicalIndicators()
        self.patterns = CandlestickPatterns()
        self.smart = SmartMoneyLogic()
        self.mtf = MTFLogic()
        self.scorer = ScoringEngine(self.config)
        self.last_candle_time = {}

    # -------------------------------------------------------------------------
    # 🔥 USE SPOT CANDLES ONLY
    # -------------------------------------------------------------------------
    def fetch_spot_candles(self, futures_pair, interval='5m', limit=100):
        """Convert FUTURES → SPOT and fetch clean SPOT candles."""

        if futures_pair not in FUTURES_TO_SPOT:
            print(f"❌ No SPOT mapping for {futures_pair}")
            return None

        spot_pair = FUTURES_TO_SPOT[futures_pair]

        try:
            url = f"{self.config.COINDCX_BASE_URL}/market_data/candles"
            params = {'pair': spot_pair, 'interval': interval, 'limit': limit}

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
                    'close': float(c['close']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'volume': float(c['volume'])
                })
            return candles

        except Exception as e:
            print(f"⚠️ SPOT Fetch Error {spot_pair}: {e}")
            return None

    # -------------------------------------------------------------------------
    # BTC RELAXED CHECK
    # -------------------------------------------------------------------------
    def check_btc_stability(self):
        if not self.config.ENABLE_BTC_CHECK:
            return True, 'BTC check disabled', 'neutral'

        try:
            btc_5m = self.fetch_spot_candles(self.config.BTC_PAIR, '5m', 20)

            if not btc_5m or len(btc_5m) < 10:
                return True, 'BTC data missing → neutral', 'neutral'

            closes = [c['close'] for c in btc_5m]
            highs = [c['high'] for c in btc_5m]
            lows = [c['low'] for c in btc_5m]

            # Extreme dump check
            for candle in btc_5m[-3:]:
                body = candle['close'] - candle['open']
                pct = abs(body) / candle['open'] * 100
                if body < 0 and pct > 5:
                    return False, f'BTC dump {pct:.1f}%', 'dump'

            # Volatility threshold
            atr = self.indicators.calculate_atr(highs, lows, closes)
            vol = self.indicators.calculate_volatility(closes, atr)
            if vol and vol > 8:
                return False, f'BTC volatile {vol:.1f}%', 'volatile'

            return True, 'BTC stable', 'stable'

        except:
            return True, 'BTC check error → neutral', 'neutral'

    # -------------------------------------------------------------------------
    def check_same_candle(self, market, candle_time):
        if market in self.last_candle_time:
            if self.last_candle_time[market] == candle_time:
                return False
        self.last_candle_time[market] = candle_time
        return True

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def generate_signal(self, market):
        """FULL SPOT→FUTURES SYSTEM"""

        # Fetch SPOT candles
        c5 = self.fetch_spot_candles(market, '5m', 100)
        c15 = self.fetch_spot_candles(market, '15m', 100)
        c1h = self.fetch_spot_candles(market, '1h', 100)

        if not c5:
            return None

        # Duplicate candle check
        if not self.check_same_candle(market, c5[-1]['time']):
            return None

        # Analysis
        a5 = self.analyze_market(c5)
        if not a5:
            return None

        trend = self.mtf.get_trend_direction(c15) if c15 else 'neutral'
        bias = self.mtf.get_trend_direction(c1h) if c1h else 'neutral'

        # Scores
        long_score, long_reasons, regime_long = self.scorer.calculate_score(
            a5, 'LONG', self.mtf.get_mtf_score(trend, bias, 'LONG')
        )
        short_score, short_reasons, regime_short = self.scorer.calculate_score(
            a5, 'SHORT', self.mtf.get_mtf_score(trend, bias, 'SHORT')
        )

        # Choose direction
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

        # Filters
        ok, reason = self.apply_hard_filters(score, regime, a5['adx'], a5['atr'])
        if not ok:
            return None

        entry = a5['price']
        atr = a5['atr']

        if direction == 'LONG':
            sl = entry - atr * self.config.ATR_SL_MULTIPLIER
            tp1 = entry + atr * self.config.ATR_TP1_MULTIPLIER
            tp2 = entry + atr * self.config.ATR_TP2_MULTIPLIER
        else:
            sl = entry + atr * self.config.ATR_SL_MULTIPLIER
            tp1 = entry - atr * self.config.ATR_TP1_MULTIPLIER
            tp2 = entry - atr * self.config.ATR_TP2_MULTIPLIER

        return {
            'market': market,      # FUTURES output
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