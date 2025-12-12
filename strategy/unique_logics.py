"""
UNIQUE TRADING LOGICS - 45 Advanced Strategies
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

        price_trend = df['close'].iloc[-10:].is_monotonic_increasing
        oi_trend = oi_history[-10:] > oi_history[-20:-10]

        if price_trend and not oi_trend.mean() > 0.5:
            return 'bearish_divergence'
        elif not price_trend and oi_trend.mean() > 0.5:
            return 'bullish_divergence'
        else:
            return 'neutral'

    def check_liquidation_proximity(self, current_price: float, 
                                     liquidation_clusters: list) -> Dict:
        """34. Liquidation cluster proximity check"""
        proximity_threshold = self.config.get('liquidation_proximity_percent', 0.02)

        nearby_liq = []
        for cluster in liquidation_clusters:
            distance = abs(current_price - cluster['price']) / current_price
            if distance < proximity_threshold:
                nearby_liq.append(cluster)

        if nearby_liq:
            logger.warning(f"⚠️ {len(nearby_liq)} liquidation clusters nearby")
            return {'nearby': True, 'clusters': nearby_liq}

        return {'nearby': False}

    def check_funding_arbitrage(self, funding_rates: Dict) -> Dict:
        """35. Funding arbitrage opportunity"""
        # Check if funding rates differ significantly across exchanges
        rates = list(funding_rates.values())

        if len(rates) < 2:
            return {'opportunity': False}

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
        # Simplified gamma logic (requires options data)
        gamma = options_data.get('gamma', 0)
        threshold = self.config.get('gamma_exposure_threshold', 0.5)

        if gamma > threshold:
            return 'positive'  # Bullish gamma
        elif gamma < -threshold:
            return 'negative'  # Bearish gamma
        else:
            return 'neutral'

    def gamma_adjusted_sizing(self, position_size: float, gamma: str) -> float:
        """37. Gamma-adjusted risk sizing"""
        if gamma == 'negative':
            # Reduce risk in negative gamma environment
            return position_size * 0.5
        elif gamma == 'positive':
            return position_size * 1.0
        else:
            return position_size * 0.75
# ==================== F) ANTI-TRAP MECHANISMS (8) ====================

    def avoid_round_numbers(self, price: float) -> bool:
        """38. Avoid round-number trap zones"""
        avoid_distance = self.config.get('round_number_avoid_distance', 0.001)

        # Check if price near round numbers (50000, 51000, etc.)
        rounded = round(price, -3)  # Round to nearest 1000
        distance = abs(price - rounded) / price

        is_safe = distance > avoid_distance
        if not is_safe:
            logger.warning(f"⚠️ Price near round number: {rounded}")
        return is_safe

    def avoid_obvious_sr(self, price: float, support_resistance: list) -> bool:
        """39. Avoid obvious support/resistance entries"""
        for sr in support_resistance:
            distance = abs(price - sr) / price
            if distance < 0.005:  # Within 0.5%
                logger.warning(f"⚠️ Price near S/R: {sr}")
                return False
        return True

    def detect_sl_hunting_zone(self, df: pd.DataFrame) -> bool:
        """40. SL-hunting zone detection"""
        if len(df) < 20:
            return False

        # Check for wicks that hunt stops
        recent = df.tail(20)
        large_wicks = 0

        for _, candle in recent.iterrows():
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']

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

        # Avoid entries at exact hours
        is_odd_time = minute not in [0, 15, 30, 45] or hour not in avoid_hours
        return is_odd_time

    def filter_sudden_wick(self, df: pd.DataFrame) -> bool:
        """42. Avoid sudden 1-min wick shocks"""
        if len(df) < 2:
            return True

        last_candle = df.iloc[-1]
        range_pct = (last_candle['high'] - last_candle['low']) / last_candle['close']

        # If range > 1% in 1 minute, suspicious
        is_normal = range_pct < 0.01
        if not is_normal:
            logger.warning(f"⚠️ Sudden wick detected: {range_pct:.2%}")
        return is_normal

    def avoid_bot_rush_time(self, current_time: datetime) -> bool:
        """43. Avoid bot-rush time"""
        rush_hours = [9, 10, 17]  # UTC
        hour = current_time.hour

        is_safe = hour not in rush_hours
        return is_safe

    def filter_manipulation_candle(self, candle: pd.Series) -> bool:
        """44. Manipulation candle filter"""
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return False

        # Very small body with huge wicks = manipulation
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
            if trade['pnl'] < 0:
                consecutive_losses += 1
            else:
                break

        should_trade = consecutive_losses < 2
        if not should_trade:
            logger.warning(f"🛑 Cooldown: {consecutive_losses} consecutive losses")

        return should_trade, consecutive_losses

    # ==================== HELPER FUNCTIONS ====================
def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX indicator"""
        # ensure numeric series
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)

        # True Range components
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional movements
        up_move = high.diff()
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Wilder smoothing
        tr_smt = tr.copy()
        if len(tr) >= period:
            tr_smt.iloc[period-1] = tr.iloc[:period].sum()
            for i in range(period, len(tr)):
                tr_smt.iloc[i] = tr_smt.iloc[i-1] - (tr_smt.iloc[i-1] / period) + tr.iloc[i]
        else:
            tr_smt = tr_smt.rolling(window=period).sum()

        plus_dm_smt = plus_dm.copy()
        minus_dm_smt = minus_dm.copy()
        if len(plus_dm) >= period:
            plus_dm_smt.iloc[period-1] = plus_dm.iloc[:period].sum()
            minus_dm_smt.iloc[period-1] = minus_dm.iloc[:period].sum()
            for i in range(period, len(plus_dm)):
                plus_dm_smt.iloc[i] = plus_dm_smt.iloc[i-1] - (plus_dm_smt.iloc[i-1] / period) + plus_dm.iloc[i]
                minus_dm_smt.iloc[i] = minus_dm_smt.iloc[i-1] - (minus_dm_smt.iloc[i-1] / period) + minus_dm.iloc[i]
        else:
            plus_dm_smt = plus_dm.rolling(window=period).sum()
            minus_dm_smt = minus_dm.rolling(window=period).sum()

        # Avoid division by zero
        tr_smt = tr_smt.replace(0, np.nan)

        plus_di = 100 * (plus_dm_smt / tr_smt)
        minus_di = 100 * (minus_dm_smt / tr_smt)

        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        if len(dx) == 0:
            return 0.0

        # ADX smoothing
        adx = dx.copy()
        if len(dx) >= period:
            adx.iloc[period-1] = dx.iloc[:period].mean()
            for i in range(period, len(dx)):
                adx.iloc[i] = ((adx.iloc[i-1] * (period - 1)) + dx.iloc[i]) / period
        else:
            adx = dx.rolling(window=period).mean()

        # return last value
        last = adx.iloc[-1] if len(adx) > 0 else 0.0
        return float(last)

# compatibility alias: older code expects LogicEvaluator
class LogicEvaluator(UniqueLogics):
    """Compatibility alias so `from strategy.unique_logics import LogicEvaluator` works."""
    pass