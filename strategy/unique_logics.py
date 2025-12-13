"""
UNIQUE TRADING LOGICS – 45 Advanced Strategies
Includes: Liquidation, Gamma, Funding, OB, CVD, VWAP + More
"""
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class UniqueLogics:
    def __init__(self, config: Dict):
        self.config = config

    # ==================== A) MARKET HEALTH FILTERS (8) ====================

    def check_btc_calm(self, df: pd.DataFrame) -> bool:
        """1. BTC calm check - no sudden spikes"""
        if len(df) < 20:
            return True

        recent = df.tail(20)
        volatility = (recent['high'] - recent['low']) / recent['close']
        avg_volatility = volatility.mean()

        is_calm = avg_volatility < self.config.get('btc_calm_threshold', 0.015)
        logger.info(f"BTC Calm: {is_calm} (volatility: {avg_volatility:.4f})")
        return is_calm

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """2. Market regime detection (trend/range/volatile)"""
        if len(df) < 50:
            return 'unknown'

        # ADX calculation for trend strength
        adx = self._calculate_adx(df, period=14)
        atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        atr_percent = (atr / df['close']).iloc[-1]

        if adx > 25:
            regime = 'trending'
        elif atr_percent > 0.03:
            regime = 'volatile'
        else:
            regime = 'ranging'

        logger.info(f"Market regime: {regime} (ADX: {adx:.1f})")
        return regime

    def check_funding_rate_normal(self, funding_rate: float) -> bool:
        """3. Funding rate normal check"""
        threshold = self.config.get('funding_rate_extreme', 0.0015)
        is_normal = abs(funding_rate) < threshold
        logger.info(f"Funding normal: {is_normal} (rate: {funding_rate:.4f})")
        return is_normal

    def check_fear_greed_index(self, index_value: int) -> bool:
        """4. Fear & Greed Index filter"""
        low = self.config.get('fear_greed_extreme_low', 20)
        high = self.config.get('fear_greed_extreme_high', 80)

        is_safe = low < index_value < high
        logger.info(f"Fear/Greed safe: {is_safe} (value: {index_value})")
        return is_safe

    def fragile_btc_mode(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """5. Fragile BTC market detection - auto conservative mode"""
        if len(df) < 30:
            return False, 'normal'

        recent = df.tail(30)

        # Check for rapid price swings
        price_changes = recent['close'].pct_change()
        large_moves = (abs(price_changes) > 0.02).sum()  # >2% moves

        # Check volume spikes
        vol_avg = recent['volume'].mean()
        vol_recent = recent['volume'].tail(5).mean()
        vol_spike = vol_recent > vol_avg * 2

        is_fragile = large_moves > 5 or vol_spike
        mode = 'conservative' if is_fragile else 'normal'

        logger.warning(f"🚨 Fragile mode: {is_fragile} -> {mode}")
        return is_fragile, mode

    def check_news_filter(self, current_time: datetime, news_times: list) -> bool:
        """6. High-impact news filter"""
        skip_minutes = self.config.get('news_skip_minutes', 30)

        for news_time in news_times:
            time_diff = abs((current_time - news_time).total_seconds() / 60)
            if time_diff < skip_minutes:
                logger.warning(f"⚠️ News in {time_diff:.0f} min - SKIP TRADE")
                return False
        return True

    def check_spread_slippage(self, orderbook: Dict) -> bool:
        """7. Spread & slippage safety filter"""
        spread = orderbook.get('spread', 0)
        mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
        spread_percent = spread / mid_price

        max_spread = self.config.get('spread_max_percent', 0.001)
        is_safe = spread_percent < max_spread

        logger.info(f"Spread safe: {is_safe} ({spread_percent:.4f}%)")
        return is_safe

    def check_liquidity_window(self, current_time: datetime) -> bool:
        """8. Time-of-day liquidity window"""
        hour = current_time.hour
        low_liquidity_hours = self.config.get('low_liquidity_hours', [2, 3, 4, 5])

        is_good = hour not in low_liquidity_hours
        if not is_good:
            logger.warning(f"⚠️ Low liquidity hour: {hour}:00 UTC")
        return is_good

    # ==================== B) PRICE ACTION & STRUCTURE (7) ====================

    def check_breakout_confirmation(self, candle: pd.Series) -> bool:
        """9. Breakout confirmation (body > wick)"""
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return False

        body_ratio = body / total_range
        is_valid = body_ratio > 0.6  # Body is 60%+ of candle

        logger.info(f"Breakout valid: {is_valid} (body ratio: {body_ratio:.2f})")
        return is_valid

    def detect_market_structure_shift(self, df: pd.DataFrame) -> str:
        """10. Market structure shift (HH/HL or LH/LL)"""
        if len(df) < 20:
            return 'neutral'

        highs = df['high'].tail(20)
        lows = df['low'].tail(20)

        # Simple structure detection
        recent_high = highs.iloc[-5:].max()
        previous_high = highs.iloc[-15:-5].max()
        recent_low = lows.iloc[-5:].min()
        previous_low = lows.iloc[-15:-5].min()

        if recent_high > previous_high and recent_low > previous_low:
            return 'bullish'  # Higher highs, higher lows
        elif recent_high < previous_high and recent_low < previous_low:
            return 'bearish'  # Lower highs, lower lows
        else:
            return 'neutral'

    def check_orderblock_retest(self, df: pd.DataFrame, current_price: float) -> Dict:
        """11. Orderblock retest confirmation"""
        # Simplified orderblock logic
        if len(df) < 50:
            return {'found': False}

        # Find significant price zones (high volume candles)
        df_copy = df.copy()
        df_copy['vol_rank'] = df_copy['volume'].rank(pct=True)
        high_vol_zones = df_copy[df_copy['vol_rank'] > 0.9].tail(10)

        for idx, zone in high_vol_zones.iterrows():
            distance = abs(current_price - zone['close']) / current_price
            if distance < 0.005:  # Within 0.5%
                return {
                    'found': True,
                    'price': zone['close'],
                    'type': 'bullish' if zone['close'] > zone['open'] else 'bearish'
                }

        return {'found': False}

    def detect_fair_value_gap(self, df: pd.DataFrame) -> Dict:
        """12. Fair value gap detection"""
        if len(df) < 3:
            return {'found': False}

        recent = df.tail(3)
        candles = recent.to_dict('records')

        # Check for gap between candle 1 and candle 3
        gap_up = candles[0]['high'] < candles[2]['low']
        gap_down = candles[0]['low'] > candles[2]['high']

        if gap_up:
            return {
                'found': True,
                'type': 'bullish',
                'gap_low': candles[0]['high'],
                'gap_high': candles[2]['low']
            }
        elif gap_down:
            return {
                'found': True,
                'type': 'bearish',
                'gap_low': candles[2]['high'],
                'gap_high': candles[0]['low']
            }

        return {'found': False}

    def check_ema_alignment(self, df: pd.DataFrame) -> str:
        """13. EMA/SMA trend alignment"""
        if len(df) < 200:
            return 'neutral'

        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema_200 = df['close'].ewm(span=200).mean().iloc[-1]

        if ema_20 > ema_50 > ema_200:
            return 'bullish'
        elif ema_20 < ema_50 < ema_200:
            return 'bearish'
        else:
            return 'neutral'

    def calculate_atr_filter(self, df: pd.DataFrame) -> float:
        """14. ATR volatility filter"""
        atr = self._calculate_atr(df, period=14)
        return atr

    def detect_bollinger_squeeze(self, df: pd.DataFrame) -> bool:
        """15. Bollinger band squeeze/expansion"""
        if len(df) < 20:
            return False

        bb_upper, bb_lower = self._calculate_bollinger_bands(df)
        bb_width = (bb_upper - bb_lower) / df['close']

        # Squeeze = narrow bands
        is_squeeze = bb_width.iloc[-1] < bb_width.rolling(50).mean().iloc[-1] * 0.8
        logger.info(f"BB Squeeze: {is_squeeze}")
        return is_squeeze

    # ==================== C) MOMENTUM (6) ====================

    def check_rsi_conditions(self, df: pd.DataFrame) -> Dict:
        """16. RSI oversold/overbought + divergence"""
        rsi = self._calculate_rsi(df)

        if len(rsi) < 14:
            return {'signal': 'neutral'}

        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            signal = 'oversold'
        elif current_rsi > 70:
            signal = 'overbought'
        else:
            signal = 'neutral'

        # Divergence detection (simplified)
        divergence = self._detect_rsi_divergence(df, rsi)

        return {
            'signal': signal,
            'value': current_rsi,
            'divergence': divergence
        }

    def check_macd_cross(self, df: pd.DataFrame) -> Dict:
        """17. MACD cross + momentum slope"""
        macd, signal, hist = self._calculate_macd(df)

        if len(hist) < 2:
            return {'signal': 'neutral'}

        cross_up = hist.iloc[-2] < 0 and hist.iloc[-1] > 0
        cross_down = hist.iloc[-2] > 0 and hist.iloc[-1] < 0

        if cross_up:
            return {'signal': 'bullish', 'macd': macd.iloc[-1], 'hist': hist.iloc[-1]}
        elif cross_down:
            return {'signal': 'bearish', 'macd': macd.iloc[-1], 'hist': hist.iloc[-1]}
        else:
            return {'signal': 'neutral'}

    def check_stochastic(self, df: pd.DataFrame) -> Dict:
        """18. Stochastic reversal confirmation"""
        k, d = self._calculate_stochastic(df)

        if len(k) < 2:
            return {'signal': 'neutral'}

        oversold = k.iloc[-1] < 20
        overbought = k.iloc[-1] > 80
        cross_up = k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
        cross_down = k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]

        if oversold and cross_up:
            return {'signal': 'bullish', 'k': k.iloc[-1]}
        elif overbought and cross_down:
            return {'signal': 'bearish', 'k': k.iloc[-1]}
        else:
            return {'signal': 'neutral'}

    def check_obv_divergence(self, df: pd.DataFrame) -> str:
        """19. OBV divergence (volume-based)"""
        obv = self._calculate_obv(df)

        if len(obv) < 20:
            return 'neutral'

        # Simple divergence check
        price_trend = df['close'].iloc[-10:].is_monotonic_increasing
        obv_trend = obv.iloc[-10:].is_monotonic_increasing

        if price_trend and not obv_trend:
            return 'bearish_divergence'
        elif not price_trend and obv_trend:
            return 'bullish_divergence'
        else:
            return 'neutral'

    def check_mfi(self, df: pd.DataFrame) -> Dict:
        """20. MFI (Money Flow) direction"""
        mfi = self._calculate_mfi(df)

        if len(mfi) < 14:
            return {'signal': 'neutral'}

        current_mfi = mfi.iloc[-1]

        if current_mfi > 80:
            return {'signal': 'overbought', 'value': current_mfi}
        elif current_mfi < 20:
            return {'signal': 'oversold', 'value': current_mfi}
        else:
            return {'signal': 'neutral', 'value': current_mfi}

    def check_roc(self, df: pd.DataFrame) -> float:
        """21. ROC (Rate of Change) momentum"""
        if len(df) < 20:
            return 0

        roc = ((df['close'].iloc[-1] - df['close'].iloc[-20]) /
               df['close'].iloc[-20]) * 100
        return roc

    # ==================== D) ORDER FLOW & DEPTH (10) ====================

    def check_orderbook_imbalance(self, orderbook: Dict) -> Dict:
        """22. Orderbook imbalance check"""
        imbalance = orderbook.get('imbalance', 1.0)
        threshold_high = self.config.get('orderbook_imbalance_threshold', 1.2)
        threshold_low = 1 / threshold_high

        if imbalance > threshold_high:
            signal = 'bullish'
        elif imbalance < threshold_low:
            signal = 'bearish'
        else:
            signal = 'neutral'

        logger.info(f"OB Imbalance: {imbalance:.2f} -> {signal}")
        return {'signal': signal, 'ratio': imbalance}

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """23. VWAP calculation"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

    def check_vwap_deviation(self, df: pd.DataFrame) -> Dict:
        """24. VWAP fair-value deviation check"""
        vwap = self.calculate_vwap(df)
        current_price = df['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]

        deviation = (current_price - current_vwap) / current_vwap
        threshold = self.config.get('vwap_deviation_percent', 0.005)

        if abs(deviation) > threshold:
            signal = 'far' if deviation > 0 else 'near'
        else:
            signal = 'normal'

        return {
            'signal': signal,
            'deviation': deviation,
            'vwap': current_vwap,
            'price': current_price
        }

    def check_vwap_logic(self, df: pd.DataFrame) -> str:
        """25. VWAP bounce/reclaim/rejection logic"""
        vwap = self.calculate_vwap(df)
        recent = df.tail(5)

        # Check if price bouncing off VWAP
        crosses = 0
        for i in range(1, len(recent)):
            prev_above = recent['close'].iloc[i-1] > vwap.iloc[-len(recent)+i-1]
            curr_above = recent['close'].iloc[i] > vwap.iloc[-len(recent)+i]
            if prev_above != curr_above:
                crosses += 1

        if crosses >= 2:
            return 'bounce'
        elif recent['close'].iloc[-1] > vwap.iloc[-1]:
            return 'reclaim'
        else:
            return 'rejection'

    def calculate_cvd(self, trades: list) -> float:
        """26. CVD (Cumulative Volume Delta) direction"""
        buy_volume = sum([t['amount'] for t in trades if t['side'] == 'buy'])
        sell_volume = sum([t['amount'] for t in trades if t['side'] == 'sell'])
        cvd = buy_volume - sell_volume
        return cvd

    def detect_large_order(self, orderbook: Dict) -> list:
        """27. Large order detection"""
        threshold = self.config.get('large_order_threshold', 100000)
        large_orders = []

        # Check bids
        for bid in orderbook.get('bids', []):
            value = bid[0] * bid[1]
            if value >= threshold:
                large_orders.append({'side': 'bid', 'price': bid[0], 'value': value})

        # Check asks
        for ask in orderbook.get('asks', []):
            value = ask[0] * ask[1]
            if value >= threshold:
                large_orders.append({'side': 'ask', 'price': ask[0], 'value': value})

        if large_orders:
            logger.info(f"🐋 {len(large_orders)} large orders detected")
        return large_orders

    def detect_spoofing_wall(self, orderbook: Dict) -> Dict:
        """28. Spoofing wall detection (fake liquidity)"""
        threshold = self.config.get('spoofing_wall_threshold', 500000)

        # Check for abnormally large orders
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        bid_walls = [b for b in bids if b[0] * b[1] > threshold]
        ask_walls = [a for a in asks if a[0] * a[1] > threshold]

        return {
            'bid_walls': len(bid_walls),
            'ask_walls': len(ask_walls),
            'suspicious': len(bid_walls) + len(ask_walls) > 0
        }

    def calculate_true_liquidity(self, orderbook: Dict) -> float:
        """29. True liquidity depth (top 20 levels)"""
        total_liquidity = 0

        for bid in orderbook.get('bids', [])[:20]:
            total_liquidity += bid[0] * bid[1]

        for ask in orderbook.get('asks', [])[:20]:
            total_liquidity += ask[0] * ask[1]

        return total_liquidity

    def calculate_aggression_ratio(self, trades: list) -> float:
        """30. Aggression ratio (market buy vs sell pressure)"""
        if not trades:
            return 0

        market_buys = sum([1 for t in trades if t['side'] == 'buy'])
        market_sells = sum([1 for t in trades if t['side'] == 'sell'])

        if market_sells == 0:
            return 999  # All buys

        ratio = market_buys / market_sells
        return ratio

    def check_spread_velocity(self, orderbook_history: list) -> float:
        """31. Spread velocity check (microstructure shift)"""
        if len(orderbook_history) < 2:
            return 0

        spread_now = orderbook_history[-1].get('spread', 0)
        spread_before = orderbook_history[-2].get('spread', 0)

        velocity = spread_now - spread_before
        return velocity

    # ==================== E) DERIVATIVES & FUTURES LOGICS (6) ====================

    def check_oi_trend(self, oi_history: list) -> str:
        """32. Open interest trend"""
        if len(oi_history) < 10:
            return 'neutral'

        recent_oi = oi_history[-5:]
        older_oi = oi_history[-10:-5]

        avg_recent = np.mean(recent_oi)
        avg_older = np.mean(older_oi)

        if avg_recent > avg_older * 1.1:
            return 'increasing'
        elif avg_recent < avg_older * 0.9:
            return 'decreasing'
        else:
            return 'stable'

    def check_oi_divergence(self, df: pd.DataFrame, oi_history: list) -> str:
        """33. OI divergence (price vs OI conflict)"""
        if len(oi_history) < 10 or len(df) < 10:
            return 'neutral'

        # basic trending boolean arrays
        price_trend = df['close'].iloc[-10:].pct_change().mean() > 0
        oi_recent = np.array(oi_history[-10:])
        oi_older = np.array(oi_history[-20:-10]) if len(oi_history) >= 20 else oi_recent * 0.99

        oi_trend = oi_recent.mean() > oi_older.mean()

        if price_trend and not oi_trend:
            return 'bearish_divergence'
        elif (not price_trend) and oi_trend:
            return 'bullish_divergence'
        else:
            return 'neutral'

    def check_liquidation_proximity(self, current_price: float,
                                     liquidation_clusters: list) -> Dict:
        """34. Liquidation cluster proximity check"""
        proximity_threshold = self.config.get('liquidation_proximity_percent', 0.02)

        nearby_liq = []
        for cluster in liquidation_clusters:
            distance = abs(current_price - cluster.get('price', current_price)) / current_price
            if distance < proximity_threshold:
                nearby_liq.append(cluster)

        if nearby_liq:
            logger.warning(f"⚠️ {len(nearby_liq)} liquidation clusters nearby")
            return {'nearby': True, 'clusters': nearby_liq}

        return {'nearby': False}

    def check_funding_arbitrage(self, funding_rates: Dict) -> Dict:
        """35. Funding arbitrage opportunity"""
        # funding_rates: {'bybit':0.0001, 'binance': -0.00005, ...}
        rates = list(funding_rates.values())
        if len(rates) < 2:
            return {'opportunity': False, 'spread': 0.0, 'rates': funding_rates}

        max_rate = max(rates)
        min_rate = min(rates)
        spread = max_rate - min_rate
        opportunity = abs(spread) > 0.0005  # 0.05%

        return {
            'opportunity': opportunity,
            'spread': spread,
            'rates': funding_rates
        }

    def check_gamma_exposure(self, options_data: Dict) -> str:
        """36. Gamma exposure direction"""
        gamma = options_data.get('gamma', 0)
        threshold = self.config.get('gamma_exposure_threshold', 0.5)

        if gamma > threshold:
            return 'positive'
        elif gamma < -threshold:
            return 'negative'
        else:
            return 'neutral'

    def gamma_adjusted_sizing(self, position_size: float, gamma: str) -> float:
        """37. Gamma-adjusted risk sizing"""
        if gamma == 'negative':
            return position_size * 0.5
        elif gamma == 'positive':
            return position_size * 1.0
        else:
            return position_size * 0.75

    # ---- Remaining anti-trap & helper methods (continuation / completeness) ----

    def avoid_round_numbers(self, price: float) -> bool:
        """38. Avoid round-number trap zones"""
        avoid_distance = self.config.get('round_number_avoid_distance', 0.001)
        rounded = round(price, -3)
        distance = abs(price - rounded) / price
        is_safe = distance > avoid_distance
        if not is_safe:
            logger.warning(f"⚠️ Price near round number: {rounded}")
        return is_safe

    def avoid_obvious_sr(self, price: float, support_resistance: list) -> bool:
        """39. Avoid obvious support/resistance entries"""
        for sr in support_resistance:
            distance = abs(price - sr) / price
            if distance < 0.005:
                logger.warning(f"⚠️ Price near S/R: {sr}")
                return False
        return True

    def detect_sl_hunting_zone(self, df: pd.DataFrame) -> bool:
        """40. SL-hunting zone detection"""
        if len(df) < 20:
            return False
        recent = df.tail(20)
        large_wicks = 0
        for _, candle in recent.iterrows():
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            if body == 0:
                continue
            if upper_wick > body * 2 or lower_wick > body * 2:
                large_wicks += 1
        is_hunting_zone = large_wicks > 3
        if is_hunting_zone:
            logger.warning("⚠️ SL hunting zone detected")
        return is_hunting_zone

    def check_odd_time_entry(self, current_time: datetime) -> bool:
        """41. Odd-time entries (avoid round hours)"""
        avoid_hours = self.config.get('avoid_round_hours', [9, 10, 17])
        minute = current_time.minute
        hour = current_time.hour
        is_odd_time = (minute not in [0, 15, 30, 45]) or (hour not in avoid_hours)
        return is_odd_time

    def filter_sudden_wick(self, df: pd.DataFrame) -> bool:
        """42. Avoid sudden 1-min wick shocks"""
        if len(df) < 2:
            return True
        last_candle = df.iloc[-1]
        range_pct = (last_candle['high'] - last_candle['low']) / last_candle['close']
        is_normal = range_pct < 0.01
        if not is_normal:
            logger.warning(f"⚠️ Sudden wick detected: {range_pct:.2%}")
        return is_normal

    def avoid_bot_rush_time(self, current_time: datetime) -> bool:
        """43. Avoid bot-rush time"""
        rush_hours = self.config.get('avoid_round_hours', [9, 10, 17])
        hour = current_time.hour
        is_safe = hour not in rush_hours
        return is_safe

    def filter_manipulation_candle(self, candle: pd.Series) -> bool:
        """44. Manipulation candle filter"""
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            return False
        body_ratio = body / total_range
        is_normal = body_ratio > 0.2
        if not is_normal:
            logger.warning("⚠️ Manipulation candle detected")
        return is_normal

    def check_consecutive_losses(self, recent_trades: list) -> Tuple[bool, int]:
        """45. No-trade after 2 consecutive losses (cooldown)"""
        if len(recent_trades) < 2:
            return True, 0
        consecutive_losses = 0
        for trade in reversed(recent_trades):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        should_trade = consecutive_losses < 2
        if not should_trade:
            logger.warning(f"🛑 Cooldown: {consecutive_losses} consecutive losses")
        return should_trade, consecutive_losses

    # ----------------- Aggregator: evaluate_all_logics -----------------
    def evaluate_all_logics(self, df: pd.DataFrame, orderbook: Dict = None,
                            funding_rate: float = 0.0, oi_history: list = None,
                            recent_trades: list = None, fear_greed_index: int = 50,
                            news_times: list = None, liquidation_clusters: list = None,
                            trades_snapshot: list = None) -> Dict:
        """
        Run all 45 logics, compute a composite score and a trade_allowed flag.
        Returns a dict with detailed sections so main bot can use them.
        """
        orderbook = orderbook or {'bids': [], 'asks': [], 'spread': 0, 'imbalance': 1.0}
        oi_history = oi_history or []
        recent_trades = recent_trades or []
        news_times = news_times or []
        liquidation_clusters = liquidation_clusters or []
        trades_snapshot = trades_snapshot or []

        results = {}
        score = 0.0
        weight_total = 0.0

        # A) Market health (weight 30)
        mh_checks = {}
        try:
            mh_checks['btc_calm'] = self.check_btc_calm(df)
            mh_checks['regime'] = self.detect_market_regime(df)
            mh_checks['funding_normal'] = self.check_funding_rate_normal(funding_rate)
            mh_checks['fear_greed'] = self.check_fear_greed_index(fear_greed_index)
            mh_checks['fragile'], mh_checks['mode'] = self.fragile_btc_mode(df)
            mh_checks['news_ok'] = self.check_news_filter(datetime.now(), news_times)
            mh_checks['spread_ok'] = self.check_spread_slippage(orderbook)
            mh_checks['liquidity_ok'] = self.check_liquidity_window(datetime.utcnow())
        except Exception as e:
            logger.warning("Market health checks error: %s", e)
        # score market health
        mh_score = sum([1.0 if mh_checks.get(k) in [True, 'trending', 'normal', 'reclaim'] else 0.0
                        for k in ['btc_calm', 'funding_normal', 'fear_greed', 'news_ok', 'spread_ok', 'liquidity_ok']])
        # fragile reduces score
        if mh_checks.get('fragile'):
            mh_score -= 1.0
        mh_weight = 30
        score += max(mh_score, 0) * (mh_weight / 6.0)
        weight_total += mh_weight

        # B) Price action & structure (weight 20)
        pa = {}
        try:
            pa['structure'] = self.detect_market_structure_shift(df)
            pa['orderblock'] = self.check_orderblock_retest(df, df['close'].iloc[-1])
            pa['fvg'] = self.detect_fair_value_gap(df)
            pa['ema_align'] = self.check_ema_alignment(df)
            pa['atr'] = float(self.calculate_atr_filter(df).iloc[-1]) if len(df) >= 14 else 0.0
            pa['bb_squeeze'] = self.detect_bollinger_squeeze(df)
        except Exception as e:
            logger.warning("Price action checks error: %s", e)
        pa_score = 0
        pa_score += 1 if pa.get('structure') in ['bullish', 'bearish'] else 0
        pa_score += 1 if pa.get('orderblock', {}).get('found') else 0
        pa_score += 1 if pa.get('fvg', {}).get('found') else 0
        pa_score += 1 if pa.get('ema_align') in ['bullish', 'bearish'] else 0
        pa_score += 1 if pa.get('bb_squeeze') else 0
        pa_weight = 20
        score += pa_score * (pa_weight / 5.0)
        weight_total += pa_weight

        # C) Momentum (weight 10)
        mom = {}
        try:
            mom['rsi'] = self.check_rsi_conditions(df)
            mom['macd'] = self.check_macd_cross(df)
            mom['stoch'] = self.check_stochastic(df)
            mom['obv'] = self.check_obv_divergence(df)
            mom['mfi'] = self.check_mfi(df)
            mom['roc'] = self.check_roc(df)
        except Exception as e:
            logger.warning("Momentum checks error: %s", e)
        mom_score = 0
        mom_score += 1 if mom['rsi'].get('signal') in ['oversold', 'overbought'] else 0
        mom_score += 1 if mom['macd'].get('signal') in ['bullish', 'bearish'] else 0
        mom_score += 1 if mom['stoch'].get('signal') in ['bullish', 'bearish'] else 0
        mom_score += 1 if mom['obv'] != 'neutral' else 0
        mom_score += 1 if mom['mfi'].get('signal') in ['oversold', 'overbought'] else 0
        mom_weight = 10
        score += mom_score * (mom_weight / 5.0)
        weight_total += mom_weight

        # D) Order flow & depth (weight 15)
        of = {}
        try:
            of['ob_imbalance'] = self.check_orderbook_imbalance(orderbook)
            of['vwap_dev'] = self.check_vwap_deviation(df)
            of['vwap_logic'] = self.check_vwap_logic(df)
            of['cvd'] = self.calculate_cvd(trades_snapshot or [])
            of['large_orders'] = self.detect_large_order(orderbook)
            of['spoof'] = self.detect_spoofing_wall(orderbook)
        except Exception as e:
            logger.warning("Orderflow checks error: %s", e)
        of_score = 0
        of_score += 1 if of['ob_imbalance'].get('signal') in ['bullish', 'bearish'] else 0
        of_score += 1 if abs(of['vwap_dev'].get('deviation', 0)) > self.config.get('vwap_deviation_percent', 0.005) else 0
        of_score += 1 if len(of['large_orders']) > 0 else 0
        of_weight = 15
        score += of_score * (of_weight / 3.0)
        weight_total += of_weight

        # E) Derivatives & futures (weight 10)
        dfut = {}
        try:
            dfut['oi_trend'] = self.check_oi_trend(oi_history or [])
            dfut['oi_div'] = self.check_oi_divergence(df, oi_history or [])
            dfut['liq_prox'] = self.check_liquidation_proximity(df['close'].iloc[-1], liquidation_clusters or [])
            dfut['fund_arb'] = self.check_funding_arbitrage({})  # placeholder
            dfut['gamma'] = self.check_gamma_exposure({})
        except Exception as e:
            logger.warning("Derivatives checks error: %s", e)
        dfut_score = 0
        dfut_score += 1 if dfut['oi_trend'] in ['increasing', 'decreasing'] else 0
        dfut_score += 1 if dfut['oi_div'] != 'neutral' else 0
        dfut_weight = 10
        score += dfut_score * (dfut_weight / 2.0)
        weight_total += dfut_weight

        # F) Anti-trap (weight 15)
        at = {}
        try:
            at['avoid_round'] = self.avoid_round_numbers(df['close'].iloc[-1])
            at['avoid_sr'] = self.avoid_obvious_sr(df['close'].iloc[-1], [])
            at['sl_hunt'] = not self.detect_sl_hunting_zone(df)
            at['odd_time'] = self.check_odd_time_entry(datetime.utcnow())
            at['sudden_wick'] = self.filter_sudden_wick(df)
            at['manipulation'] = self.filter_manipulation_candle(df.iloc[-1])
            at['consecutive_ok'], at['consec_cnt'] = self.check_consecutive_losses(recent_trades)
        except Exception as e:
            logger.warning("Anti-trap checks error: %s", e)
        at_score = sum([1 for v in [at['avoid_round'], at['avoid_sr'], at['sl_hunt'],
                                   at['odd_time'], at['sudden_wick'], at['manipulation'], at['consecutive_ok']] if v])
        at_weight = 15
        score += at_score * (at_weight / 7.0)
        weight_total += at_weight

        # Normalize final score to 0-100
        normalized = (score / weight_total) * 100 if weight_total else 0.0
        trade_allowed = normalized >= 50.0 and mh_checks.get('news_ok', True) and at.get('consecutive_ok', True)
        # If fragile mode, be stricter
        if mh_checks.get('fragile'):
            trade_allowed = trade_allowed and normalized >= 60.0

        result = {
            'final_score': normalized,
            'trade_allowed': bool(trade_allowed),
            'market_health': mh_checks,
            'price_action': pa,
            'momentum': mom,
            'order_flow': of,
            'derivatives': dfut,
            'anti_trap': at,
            'detailed': {
                'raw_score': score,
                'weight_total': weight_total
            }
        }
        return result

    # ----------------- Helper indicator functions (repeated for completeness) -----------------
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            return super()._calculate_adx(df, period)
        except Exception:
            # fallback: simple approximation
            return 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            return super()._calculate_atr(df, period)
        except Exception:
            high = df['high']; low = df['low']; close = df['close']
            tr = (high - low).combine((high - close.shift()).abs(), max).combine((low - close.shift()).abs(), max)
            return tr.rolling(window=period).mean()

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2):
        try:
            return super()._calculate_bollinger_bands(df, period, std_dev)
        except Exception:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            return sma + std * std_dev, sma - std * std_dev

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            return super()._calculate_rsi(df, period)
        except Exception:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        try:
            return super()._calculate_macd(df, fast, slow, signal)
        except Exception:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            sig = macd.ewm(span=signal).mean()
            hist = macd - sig
            return macd, sig, hist

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        try:
            return super()._calculate_stochastic(df, k_period, d_period)
        except Exception:
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d = k.rolling(window=d_period).mean()
            return k, d

    def _calculate_obv(self, df: pd.DataFrame):
        try:
            return super()._calculate_obv(df)
        except Exception:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            return obv

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14):
        try:
            return super()._calculate_mfi(df, period)
        except Exception:
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            money_flow = typical_price * df['volume']
            positive = pd.Series(0, index=df.index, dtype=float)
            negative = pd.Series(0, index=df.index, dtype=float)
            for i in range(1, len(df)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive.iloc[i] = money_flow.iloc[i]
                else:
                    negative.iloc[i] = money_flow.iloc[i]
            pos_sum = positive.rolling(window=period).sum()
            neg_sum = negative.rolling(window=period).sum().replace(0, np.nan)
            mfi = 100 - (100 / (1 + (pos_sum / neg_sum)))
            return mfi.fillna(50)

    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> str:
        try:
            return super()._detect_rsi_divergence(df, rsi)
        except Exception:
            # very simple
            price_trend = df['close'].iloc[-10:].pct_change().mean() > 0
            rsi_trend = rsi.iloc[-10:].pct_change().mean() > 0
            if price_trend and not rsi_trend:
                return 'bearish_divergence'
            elif (not price_trend) and rsi_trend:
                return 'bullish_divergence'
            return 'no_divergence'


# compatibility alias: older code expects LogicEvaluator
class LogicEvaluator(UniqueLogics):
    """Compatibility alias so `from strategy.unique_logics import LogicEvaluator` works."""
    pass
