import pandas as pd
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from indicators import Indicators
from trap_detector import TrapDetector
from coindcx_api import CoinDCXAPI
from news_guard import news_guard
from chatgpt_advisor import ChatGPTAdvisor
from signal_explainer import SignalExplainer
from telegram_notifier import TelegramNotifier


class SignalGenerator:
    """Smart Rule-Based Signal Generator - Penalty-Based Scoring System
    
    REFACTORED LOGIC:
    - Hard blocks ONLY for fatal risks (news, BTC crash, liquidation)
    - Everything else uses penalty-based scoring
    - ChatGPTAdvisor is final decision layer
    - Mode-aware tracking (active_trends, last_signal_price)
    """

    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        # BUGFIX: Mode-aware last signal price tracking
        self.last_signal_price: Dict[str, float] = {}  # Key: {pair}_{mode}
        self.last_reset_date = datetime.now().date()
        self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
        self.coin_signal_count: Dict[str, int] = {}
        self.coin_mode_signal_count: Dict[str, int] = {}
        # BUGFIX: Mode-aware active trends tracking
        self.active_trends: Dict[str, Dict] = {}  # Key: {pair}_{mode}_{trend}
        self.chatgpt_advisor = ChatGPTAdvisor()
        self.chatgpt_approved = 0
        self.chatgpt_rejected = 0
        self.active_positions: Dict[str, Dict] = {}

    def get_stats(self) -> dict:
        """Return a dict with counters that main.py prints/shows."""
        return {
            "signals_today": self.signal_count,
            "mode_breakdown": dict(self.mode_signal_count),
            "coin_breakdown": dict(self.coin_signal_count),
            "chatgpt_approved": self.chatgpt_approved,
            "chatgpt_rejected": self.chatgpt_rejected,
            "signals_today_list": self.signals_today.copy()
        }

    def _reset_daily_counters(self):
        """Reset signal counters at midnight"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.mode_signal_count = {mode: 0 for mode in config.ACTIVE_MODES}
            self.coin_signal_count = {}
            self.coin_mode_signal_count = {}
            self.last_signal_price = {}
            self.active_trends = {}
            self.chatgpt_approved = 0
            self.chatgpt_rejected = 0
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")

    def _check_cooldown(self, pair: str, mode: str) -> bool:
        """HARD BLOCK: Check if pair is in cooldown period"""
        key = f"{pair}_{mode}"
        if key in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[key]
            if mode == 'TREND':
                cooldown_duration = timedelta(hours=2)
            elif mode == 'MID':
                cooldown_duration = timedelta(minutes=45)
            else:
                cooldown_duration = timedelta(minutes=config.SAME_PAIR_COOLDOWN_MINUTES)

            if time_since_last < cooldown_duration:
                remaining = cooldown_duration - time_since_last
                remaining_mins = int(remaining.total_seconds() / 60)
                print(f"⏸️  {pair} ({mode}) in cooldown - {remaining_mins}m remaining")
                return False
        return True

    def _check_coin_daily_limit(self, pair: str, mode: str) -> bool:
        """HARD BLOCK: Check per-coin per-mode daily signal limit"""
        LIMITS = {'TREND': 3, 'MID': 4, 'QUICK': 5, 'SCALP': 5}
        max_signals = LIMITS.get(mode, 4)

        coin = pair.split('USDT')[0]
        coin_mode_key = f"{coin}_{mode}"

        if coin_mode_key not in self.coin_mode_signal_count:
            self.coin_mode_signal_count[coin_mode_key] = 0

        if self.coin_mode_signal_count[coin_mode_key] >= max_signals:
            print(f"❌ BLOCKED: {pair} | {mode} coin limit ({max_signals})")
            return False
        return True

    def _check_active_trend(self, pair: str, trend: str, mode: str, current_price: float, ema_fast: float, ema_slow: float, macd_hist: float) -> bool:
        """HARD BLOCK: Smart trend invalidation with mode-aware tracking
        
        BUGFIX: Key now includes mode to prevent cross-mode conflicts
        """
        # Mode-aware key
        trend_key = f"{pair}_{mode}_{trend}"
        
        if trend_key in self.active_trends:
            active = self.active_trends[trend_key]
            time_elapsed = (datetime.now() - active['started']).total_seconds() / 3600

            if time_elapsed > 8:
                del self.active_trends[trend_key]
                print(f"🔓 TREND expired: {pair} {mode} {trend} (8h TTL)")
                return True

            price_change = (current_price - active['entry_price']) / active['entry_price'] * 100

            structural_break = False
            if trend == 'LONG':
                if price_change < -2.5:
                    structural_break = True
                if ema_fast < ema_slow:
                    structural_break = True
            else:
                if price_change > 2.5:
                    structural_break = True
                if ema_fast > ema_slow:
                    structural_break = True

            if structural_break:
                del self.active_trends[trend_key]
                print(f"🔓 TREND invalidated: {pair} {mode} {trend} (structure broken)")
                return True
            else:
                print(f"❌ BLOCKED: {pair} | {mode} TREND active since {active['started'].strftime('%H:%M')} ({time_elapsed:.1f}h)")
                return False
        return True

    def _check_btc_context(self) -> tuple[bool, str]:
        """HARD BLOCK: BTC extreme volatility check"""
        try:
            btc_candles = CoinDCXAPI.get_candles('BTCUSDT', '15m', 20)
            if btc_candles is None or btc_candles.empty or len(btc_candles) < 10:
                return True, "BTC check skipped"

            btc_close = btc_candles['close']
            btc_high = btc_candles['high']
            btc_low = btc_candles['low']

            btc_atr = Indicators.atr(btc_high, btc_low, btc_close)
            if btc_atr is None or len(btc_atr) < 2:
                return True, "BTC check skipped"

            current_btc_atr = float(btc_atr.iloc[-1])
            avg_btc_atr = float(btc_atr.tail(10).mean())

            # HARD BLOCK: Only extreme volatility (2.5x instead of 2.0x)
            if current_btc_atr > avg_btc_atr * 2.5:
                return False, "BTC extreme volatility"

            return True, "BTC stable"
        except Exception as e:
            print(f"⚠️ BTC check error: {e}")
            return True, "BTC check skipped"

    def _check_trading_hours(self) -> tuple[bool, str]:
        """Trading hours check - 24/7 enabled"""
        return True, "24/7 Trading"

    def _calculate_entry_sl_tp(self, direction: str, current_price: float, atr: float, mode_config: Dict) -> Optional[Dict]:
        """Calculate entry, SL, TP with minimum distance enforcement"""
        sl_multiplier = mode_config['atr_sl_multiplier']
        tp1_multiplier = mode_config['atr_tp1_multiplier']
        tp2_multiplier = mode_config['atr_tp2_multiplier']
        decimal_places = config.get_decimal_places(current_price)

        if direction == "LONG":
            entry = current_price
            sl_raw = entry - (atr * sl_multiplier)
            tp1_raw = entry + (atr * tp1_multiplier)
            tp2_raw = entry + (atr * tp2_multiplier)
        else:
            entry = current_price
            sl_raw = entry + (atr * sl_multiplier)
            tp1_raw = entry - (atr * tp1_multiplier)
            tp2_raw = entry - (atr * tp2_multiplier)

        entry = round(entry, decimal_places)
        sl = round(sl_raw, decimal_places)
        tp1 = round(tp1_raw, decimal_places)
        tp2 = round(tp2_raw, decimal_places)

        min_sl_distance = entry * 0.008
        min_tp_distance = entry * (config.MIN_TP_DISTANCE_PERCENT / 100)

        if direction == "LONG":
            if entry - sl < min_sl_distance:
                sl = round(entry - min_sl_distance, decimal_places)
            if tp1 - entry < min_tp_distance:
                tp1 = round(entry + min_tp_distance, decimal_places)
            if tp2 - tp1 < min_tp_distance * 0.5:
                tp2 = round(tp1 + min_tp_distance * 0.5, decimal_places)
        else:
if sl - entry < min_sl_distance:
                sl = round(entry + min_sl_distance, decimal_places)
            if entry - tp1 < min_tp_distance:
                tp1 = round(entry - min_tp_distance, decimal_places)
            if tp1 - tp2 < min_tp_distance * 0.5:
                tp2 = round(tp1 - min_tp_distance * 0.5, decimal_places)

        if direction == "LONG":
            if tp1 <= entry or tp2 <= entry or tp1 >= tp2 or sl >= entry:
                return None
        else:
            if tp1 >= entry or tp2 >= entry or tp1 <= tp2 or sl <= entry:
                return None

        return {'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2}

    def _check_liquidation_safety(self, entry: float, sl: float) -> tuple[bool, float]:
        """HARD BLOCK: Ensure SL is far enough from liquidation price"""
        distance_pct = abs(entry - sl) / entry * 100
        MIN_DISTANCE = config.LIQUIDATION_BUFFER * 100
        is_safe = distance_pct >= MIN_DISTANCE
        return is_safe, distance_pct

    def _check_minimum_rr(self, entry: float, sl: float, tp1: float, mode: str) -> tuple[bool, float]:
        """HARD BLOCK: Check minimum Risk-Reward ratio"""
        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr = tp1_distance / sl_distance if sl_distance > 0 else 0

        # Slightly relaxed RR requirements
        MIN_RR = {'TREND': 1.8, 'MID': 1.6, 'QUICK': 1.4, 'SCALP': 1.4}
        required_rr = MIN_RR.get(mode, 1.4)

        if rr < required_rr:
            return False, rr
        return True, rr

    def _log_signal_performance(self, signal: Dict):
        """Log signal to CSV"""
        if not hasattr(config, 'TRACK_PERFORMANCE') or not config.TRACK_PERFORMANCE:
            return
        log_file = getattr(config, 'PERFORMANCE_LOG_FILE', 'signal_performance.csv')
        log_data = {
            'timestamp': signal.get('timestamp', datetime.now().isoformat()),
            'pair': signal['pair'],
            'direction': signal['direction'],
            'mode': signal.get('mode', 'UNKNOWN'),
            'timeframe': signal.get('timeframe', 'UNKNOWN'),
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp1': signal['tp1'],
            'tp2': signal['tp2'],
            'score': signal['score'],
            'rsi': signal['rsi'],
            'adx': signal['adx'],
            'leverage': signal['leverage']
        }
        try:
            file_exists = False
            try:
                with open(log_file, 'r') as f:
                    file_exists = True
            except FileNotFoundError:
                file_exists = False
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)
        except Exception as e:
            print(f"⚠️ Performance logging failed: {e}")

    def _calculate_base_score(self, rsi: float, adx: float, macd_hist: float, prev_macd_hist: float) -> int:
        """Calculate base score from core indicators
        
        REFACTORED: Separated from penalty logic for clarity
        """
        score = 0
        
        # RSI scoring (ideal range: 35-65)
        if 35 < rsi < 65:
            score += 18
        elif 30 < rsi < 70:
            score += 14
        else:
            score += 8
        
        # ADX scoring (strength)
        if adx > 40:
            score += 22
        elif adx > 30:
            score += 18
        elif adx > 25:
            score += 14
        else:
            score += 8
        
        # MACD momentum
        if abs(macd_hist) > abs(prev_macd_hist):
            score += 16
        else:
            score += 8
        
        return score

    def _apply_volume_penalty(self, score: int, volume_surge: float, mode: str) -> tuple[int, str]:
        """PENALTY-BASED: Volume check reduces score instead of blocking
        
        REFACTORED: Converted from hard block to penalty system
        """
        penalty = 0
        reason = ""
        
        if mode == 'TREND':
            if volume_surge >= 1.0:
                score += 14
            elif volume_surge >= 0.7:
                score += 10
            elif volume_surge >= 0.5:
                score += 6
                penalty = 8
                reason = f"Low volume ({volume_surge:.2f}x)"
            else:
                score += 2
                penalty = 15
                reason = f"Very low volume ({volume_surge:.2f}x)"
        elif mode == 'MID':
            if volume_surge >= 1.0:
                score += 14
            elif volume_surge >= 0.7:
                score += 10
            elif volume_surge >= 0.5:
                score += 6
                penalty = 6
                reason = f"Low volume ({volume_surge:.2f}x)"
            else:
                score += 2
                penalty = 12
                reason = f"Very low volume ({volume_surge:.2f}x)"
        elif mode == 'QUICK':
            if volume_surge >= 1.2:
                score += 14
            elif volume_surge >= 0.9:
                score += 10
            elif volume_surge >= 0.7:
                score += 6
                penalty = 5
                reason = f"Low volume ({volume_surge:.2f}x)"
            else:
                score += 2
                penalty = 10
                reason = f"Very low volume ({volume_surge:.2f}x)"
        else:  # SCALP
            if volume_surge > 1.5:
                score += 14
            elif volume_surge > 1.0:
                score += 10
            else:
                score += 5
        
        return score - penalty, reason

    def _apply_mtf_penalty(self, score: int, mtf_trend: str, mode: str) -> tuple[int, str]:
        """PENALTY-BASED: MTF trend weakness reduces score
        
        REFACTORED: Converted from hard block to penalty system
        """
        penalty = 0
        reason = ""
        
        if mtf_trend in ['STRONG_UP', 'STRONG_DOWN']:
            score += 16
        elif mtf_trend in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 12
            if mode == 'TREND':
                penalty = 8
                reason = f"TREND prefers STRONG MTF (got {mtf_trend})"
        elif mtf_trend in ['WEAK_UP', 'WEAK_DOWN']:
            score += 4
            penalty = 12
            reason = f"Weak MTF trend ({mtf_trend})"
        elif mtf_trend == 'CHOPPY':
            score += 0
            penalty = 15
            reason = "Choppy market"
        else:  # UNKNOWN
            score += 4
            penalty = 5
            reason = "MTF trend unknown"
        
        return score - penalty, reason

    def _apply_rsi_adx_penalty(self, score: int, rsi: float, adx: float, trend: str, mode: str) -> tuple[int, str]:
        """PENALTY-BASED: RSI/ADX outside ideal range reduces score
        
        REFACTORED: Converted from hard blocks to penalties
        Only HARD BLOCK extreme cases that are clearly wrong
        """
        penalty = 0
        reasons = []
        
        # Mode-specific RSI penalties
        if mode == 'TREND':
            if trend == "LONG":
                if rsi > 70:
                    return 0, f"TREND LONG RSI too high ({rsi:.1f})"  # HARD BLOCK
                elif rsi > 65:
                    penalty += 8
                    reasons.append(f"RSI high ({rsi:.1f})")
                elif rsi < 35:
                    penalty += 10
                    reasons.append(f"RSI low ({rsi:.1f})")
                elif rsi < 30:
                    return 0, f"TREND LONG RSI too low ({rsi:.1f})"  # HARD BLOCK
            else:  # SHORT
                if rsi < 30:
                    return 0, f"TREND SHORT RSI too low ({rsi:.1f})"  # HARD BLOCK
                elif rsi < 35:
                    penalty += 8
                    reasons.append(f"RSI low ({rsi:.1f})")
                elif rsi > 65:
                    penalty += 10
                    reasons.append(f"RSI high ({rsi:.1f})")
                elif rsi > 70:
                    return 0, f"TREND SHORT RSI too high ({rsi:.1f})"  # HARD BLOCK
        
        elif mode == 'MID':
            if trend == "LONG":
                if rsi > config.RSI_OVERBOUGHT:
                    return 0, f"MID LONG RSI overbought ({rsi:.1f})"  # HARD BLOCK
                elif rsi < 30:
                    penalty += 8
                    reasons.append(f"RSI low ({rsi:.1f})")
                elif rsi < 25:
                    return 0, f"MID LONG RSI too low ({rsi:.1f})"  # HARD BLOCK
            else:  # SHORT
                # Exhaustion check
                if adx > 50 and rsi < 30:
                    return 0, f"MID SHORT exhaustion (ADX {adx:.1f}, RSI {rsi:.1f})"  # HARD BLOCK
                if rsi < 28:
                    return 0, f"MID SHORT RSI too low ({rsi:.1f})"  # HARD BLOCK
                elif rsi < 32:
                    penalty += 6
                    reasons.append(f"RSI low ({rsi:.1f})")
                elif rsi > 70:
                    return 0, f"MID SHORT RSI too high ({rsi:.1f})"  # HARD BLOCK
                elif rsi > 65:
                    penalty += 8
                    reasons.append(f"RSI high ({rsi:.1f})")
        
        elif mode == 'QUICK':
            if trend == "LONG":
                if rsi > 68:
                    return 0, f"QUICK LONG RSI too high ({rsi:.1f})"  # HARD BLOCK
                elif rsi > 62:
                    penalty += 6
                    reasons.append(f"RSI high ({rsi:.1f})")
            else:  # SHORT
                if rsi < 32:
                    return 0, f"QUICK SHORT RSI too low ({rsi:.1f})"  # HARD BLOCK
                elif rsi < 38:
                    penalty += 6
                    reasons.append(f"RSI low ({rsi:.1f})")
            
            # QUICK mode ADX ceiling
            if adx > 45:
                return 0, f"QUICK ADX too strong ({adx:.1f})"  # HARD BLOCK
            elif adx > 38:
                penalty += 8
                reasons.append(f"ADX strong ({adx:.1f})")
        
        else:  # SCALP
            if trend == "LONG" and rsi > config.RSI_OVERBOUGHT:
                return 0, "SCALP LONG RSI overbought"  # HARD BLOCK
            if trend == "SHORT" and rsi < config.RSI_OVERSOLD:
                return 0, "SCALP SHORT RSI oversold"  # HARD BLOCK
        
        # ADX minimum (applies to all modes except check above)
        min_adx = {'TREND': 25, 'MID': 23, 'QUICK': 20, 'SCALP': 20}.get(mode, 20)
        if adx < min_adx:
            if adx < min_adx - 5:
                return 0, f"{mode} ADX too weak ({adx:.1f})"  # HARD BLOCK
            else:
                penalty += 8
                reasons.append(f"ADX weak ({adx:.1f})")
        
        reason = ", ".join(reasons) if reasons else ""
        return score - penalty, reason

    def _apply_htf_penalty(self, score: int, trend: str, mode: str, candles_1h: pd.DataFrame) -> tuple[int, str]:
        """PENALTY-BASED: HTF conflict reduces score instead of blocking
        
        REFACTORED: Simplified - only checks major conflicts
        """
        penalty = 0
        reason = ""
        
        try:
            if candles_1h.empty or len(candles_1h) < 20:
                return score, "HTF check skipped"
            
            htf_ema_fast = Indicators.ema(candles_1h['close'], 20)
            htf_ema_slow = Indicators.ema(candles_1h['close'], 50)
            
            if len(htf_ema_fast) > 0 and len(htf_ema_slow) > 0:
                htf_bullish = htf_ema_fast.iloc[-1] > htf_ema_slow.iloc[-1]
                htf_bearish = htf_ema_fast.iloc[-1] < htf_ema_slow.iloc[-1]
                
                # Check for major conflicts
                if trend == "LONG" and htf_bearish:
                    if mode == 'TREND':
                        penalty = 15  # Heavy penalty for TREND mode
                        reason = "1H bearish conflict (TREND)"
                    else:
                        penalty = 8  # Light penalty for other modes
                        reason = "1H bearish conflict"
                
                elif trend == "SHORT" and htf_bullish:
                    if mode == 'TREND':
                        penalty = 15  # Heavy penalty for TREND mode
                        reason = "1H bullish conflict (TREND)"
                    else:
                        penalty = 8  # Light penalty for other modes
                        reason = "1H bullish conflict"
        
        except Exception as e:
            print(f"⚠️ HTF check error: {e}")
            return score, "HTF check failed"
        
        return score - penalty, reason

    def _apply_counter_trend_penalty(self, score: int, mtf_trend: str, trend: str, rsi: float, mode: str) -> tuple[int, str]:
        """PENALTY-BASED: Counter-trend setup penalties
        
        REFACTORED: Simplified - removed complex price structure checks
        """
        if mode != 'TREND':
            return score, ""
        
        penalty = 0
        reason = ""
        
        # Counter-trend LONG (against STRONG_DOWN)
        if mtf_trend == 'STRONG_DOWN' and trend == "LONG":
            if rsi <= 45:
                penalty = 20  # Heavy penalty
                reason = f"Counter-trend LONG with weak RSI ({rsi:.1f})"
            elif rsi <= 50:
                penalty = 10  # Moderate penalty
                reason = f"Counter-trend LONG (RSI {rsi:.1f})"
        
        # Counter-trend SHORT (against STRONG_UP)
        if mtf_trend == 'STRONG_UP' and trend == "SHORT":
            if rsi >= 55:
                penalty = 20  # Heavy penalty
                reason = f"Counter-trend SHORT with weak RSI ({rsi:.1f})"
            elif rsi >= 50:
                penalty = 10  # Moderate penalty
                reason = f"Counter-trend SHORT (RSI {rsi:.1f})"
        
        return score - penalty, reason

    def _apply_smart_signals_bonus(self, score: int, sweep: bool, ob: bool, fvg: bool, key_level: bool) -> int:
        """Bonus points for smart signal confirmations"""
        if sweep:
            score += 8
        if ob:
            score += 8
        if fvg:
            score += 6
        if key_level:
            score += 6
        return score

    def analyze(self, pair: str, candles: pd.DataFrame, mode: str = None) -> Optional[Dict]:
        """Smart Rule-Based Analysis with Penalty-Based Scoring System
        
        REFACTORED FLOW:
        1. HARD BLOCKS: Fatal risks only (news, BTC crash, limits, cooldowns)
        2. Calculate all indicators
        3. Detect trend
        4. Calculate base score
        5. Apply penalties (volume, MTF, RSI/ADX, HTF, counter-trend)
        6. Apply bonuses (sweeps, OB, FVG, key levels)
        7. Check minimum score threshold
        8. ChatGPT final decision
        """
        if mode is None:
            mode = config.MODE
        mode_config = config.MODE_CONFIG[mode]
        min_score = mode_config.get('min_score', config.MIN_SIGNAL_SCORE)
        
        self._reset_daily_counters()
        
        # ============================================================
        # PHASE 1: HARD BLOCKS (Fatal Risks Only)
        # ============================================================
        
        trading_allowed, time_reason = self._check_trading_hours()
        if not trading_allowed:
            print(f"❌ BLOCKED: {pair} | {mode} | {time_reason}")
            return None

        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"❌ BLOCKED: {pair} | {mode} | News: {reason}")
            return None

        btc_stable, btc_reason = self._check_btc_context()
        if not btc_stable:
            print(f"❌ BLOCKED: {pair} | {mode} | {btc_reason}")
            return None

        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            print(f"❌ BLOCKED: {pair} | {mode} | Daily limit")
            return None

        max_per_mode = config.MAX_SIGNALS_PER_DAY // len(config.ACTIVE_MODES)
        if self.mode_signal_count[mode] >= max_per_mode:
            print(f"❌ BLOCKED: {pair} | {mode} | Mode limit")
            return None

        if not self._check_cooldown(pair, mode):
            return None

        if not self._check_coin_daily_limit(pair, mode):
            return None

        if len(candles) < 50:
            print(f"❌ BLOCKED: {pair} | {mode} | Insufficient data")
            return None

        # ============================================================
        # PHASE 2: CALCULATE ALL INDICATORS
        # ============================================================
        
        try:
            candles = candles.dropna()
            if len(candles) < 50:
                return None

            close = candles['close']
            high = candles['high']
            low = candles['low']
            volume = candles['volume']

            ema_fast = Indicators.ema(close, mode_config['ema_fast'])
            ema_slow = Indicators.ema(close, mode_config['ema_slow'])
            macd_line, signal_line, histogram = Indicators.macd(close)
            rsi = Indicators.rsi(close)
            adx, plus_di, minus_di = Indicators.adx(high, low, close)
            atr = Indicators.atr(high, low, close)
            volume_surge = Indicators.volume_surge(volume)

            ema_fast = ema_fast.dropna()
            ema_slow = ema_slow.dropna()
            macd_line = macd_line.dropna()
            signal_line = signal_line.dropna()
            histogram = histogram.dropna()
            rsi = rsi.dropna()
            adx = adx.dropna()
            atr = atr.dropna()
            volume_surge = volume_surge.dropna()

            if len(rsi) < 2 or len(adx) < 2 or len(atr) < 2:
                return None

            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            current_adx = float(adx.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_macd_hist = float(histogram.iloc[-1])
            prev_macd_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
            current_volume_surge = float(volume_surge.iloc[-1]) if len(volume_surge) > 0 else 1.0
            current_ema_fast = float(ema_fast.iloc[-1])
            current_ema_slow = float(ema_slow.iloc[-1])

            if any(pd.isna([current_price, current_rsi, current_adx, current_atr])):
                return None

            # ============================================================
            # PHASE 3: TRAP DETECTION (HARD BLOCK if >= 4 traps)
            # ============================================================
            
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price
            traps = TrapDetector.check_all_traps(candles, bid, ask, current_rsi, current_adx, current_macd_hist)
            trapped_count = sum(traps.values())
            if trapped_count >= 4:
                print(f"❌ BLOCKED: {pair} | {mode} | Traps: {trapped_count}")
                return None

            # ============================================================
            # PHASE 4: DETECT TREND
            # ============================================================
            
            trend = None
            if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and macd_line.iloc[-1] > signal_line.iloc[-1] and plus_di.iloc[-1] > minus_di.iloc[-1]):
                trend = "LONG"
            elif (ema_fast.iloc[-1] < ema_slow.iloc[-1] and macd_line.iloc[-1] < signal_line.iloc[-1] and minus_di.iloc[-1] > plus_di.iloc[-1]):
                trend = "SHORT"
            else:
                print(f"❌ BLOCKED: {pair} | {mode} | No clear trend")
                return None

            # ==============================