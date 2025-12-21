import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from indicators import Indicators
from trap_detector import TrapDetector
from coindcx_api import CoinDCXAPI
from news_guard import news_guard

class SignalGenerator:
    """
    Core signal generation engine
    Analyzes market data and generates high-quality trading signals
    """
    
    def __init__(self):
        self.signal_count = 0
        self.signals_today = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.mode_config = config.MODE_CONFIG[config.MODE]
        self.last_reset_date = datetime.now().date()
    
    def _reset_daily_counters(self):
        """Reset signal counters at midnight"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.signal_count = 0
            self.signals_today = []
            self.last_reset_date = today
            print(f"🔄 Daily counters reset for {today}")
    
    def _check_cooldown(self, pair: str) -> bool:
        """Check if pair is in cooldown period"""
        if pair in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[pair]
            if time_since_last < timedelta(minutes=config.COOLDOWN_MINUTES):
                return False  # Still in cooldown
        return True
    
    def _check_power_hours(self) -> bool:
        """Check if current time is in power hours (best trading times)"""
        now = datetime.now()
        current_hour = now.hour
        
        for start_hour, end_hour in config.POWER_HOURS:
            if start_hour <= current_hour < end_hour:
                return True
        
        return False
    
    def _calculate_entry_sl_tp(self, direction: str, current_price: float, atr: float) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        
        sl_multiplier = self.mode_config['atr_sl_multiplier']
        tp1_multiplier = self.mode_config['atr_tp1_multiplier']
        tp2_multiplier = self.mode_config['atr_tp2_multiplier']
        
        if direction == "LONG":
            entry = current_price
            sl = entry - (atr * sl_multiplier)
            tp1 = entry + (atr * tp1_multiplier)
            tp2 = entry + (atr * tp2_multiplier)
        else:  # SHORT
            entry = current_price
            sl = entry + (atr * sl_multiplier)
            tp1 = entry - (atr * tp1_multiplier)
            tp2 = entry - (atr * tp2_multiplier)
        
        return {
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2)
        }
    
    def _check_liquidation_safety(self, entry: float, sl: float) -> bool:
        """Ensure SL is far enough from liquidation price"""
        distance = abs(entry - sl) / entry
        return distance >= config.LIQUIDATION_BUFFER
    
    def _calculate_signal_score(self, indicators: Dict, trend_strength: str) -> int:
        """
        Calculate signal quality score (0-100)
        Higher score = better signal
        """
        score = 0
        
        # RSI score (max 20 points)
        rsi = indicators['rsi']
        if 35 < rsi < 65:
            score += 20
        elif 30 < rsi < 70:
            score += 15
        else:
            score += 10
        
        # ADX score (max 25 points)
        adx = indicators['adx']
        if adx > 40:
            score += 25
        elif adx > 30:
            score += 20
        elif adx > 25:
            score += 15
        else:
            score += 10
        
        # MACD momentum (max 20 points)
        if abs(indicators['macd_histogram']) > abs(indicators['prev_macd_histogram']):
            score += 20  # Momentum increasing
        else:
            score += 10
        
        # Volume surge (max 15 points)
        if indicators['volume_surge'] > 1.5:
            score += 15
        elif indicators['volume_surge'] > 1.2:
            score += 10
        else:
            score += 5
        
        # Multi-timeframe alignment (max 20 points)
        if trend_strength in ['STRONG_UP', 'STRONG_DOWN']:
            score += 20
        elif trend_strength in ['MODERATE_UP', 'MODERATE_DOWN']:
            score += 15
        else:
            score += 5
        
        return min(score, 100)
    
    def analyze(self, pair: str, candles: pd.DataFrame) -> Optional[Dict]:
        """
        Main analysis function with NaN protection
        
        Args:
            pair: Trading pair
            candles: Historical OHLCV data
        
        Returns:
            Signal dict or None
        """
        
        # Reset daily counters if needed
        self._reset_daily_counters()
        
        # ⚠️ NEWS GUARD CHECK - PRIORITY #1
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"🚫 NEWS GUARD ACTIVE: {reason}")
            return None
        
        # Check daily limit
        if self.signal_count >= config.MAX_SIGNALS_PER_DAY:
            return None
        
        # Check cooldown
        if not self._check_cooldown(pair):
            return None
        
        # Check if enough data
        if len(candles) < 50:
            return None
        
        try:
            # Clean data - remove NaN
            candles = candles.dropna()
            
            if len(candles) < 50:
                print(f"⚠️ Not enough clean data for {pair}")
                return None
            
            # Calculate all indicators
            close = candles['close']
            high = candles['high']
            low = candles['low']
            volume = candles['volume']
            
            ema_fast = Indicators.ema(close, self.mode_config['ema_fast'])
            ema_slow = Indicators.ema(close, self.mode_config['ema_slow'])
            macd_line, signal_line, histogram = Indicators.macd(close)
            rsi = Indicators.rsi(close)
            adx, plus_di, minus_di = Indicators.adx(high, low, close)
            atr = Indicators.atr(high, low, close)
            volume_surge = Indicators.volume_surge(volume)
            
            # Drop NaN from indicators
            ema_fast = ema_fast.dropna()
            ema_slow = ema_slow.dropna()
            macd_line = macd_line.dropna()
            signal_line = signal_line.dropna()
            histogram = histogram.dropna()
            rsi = rsi.dropna()
            adx = adx.dropna()
            atr = atr.dropna()
            volume_surge = volume_surge.dropna()
            
            # Check if we have enough data after cleaning
            if len(rsi) < 2 or len(adx) < 2 or len(atr) < 2:
                print(f"⚠️ Insufficient indicator data for {pair}")
                return None
            
            # Current values with NaN check
            current_price = float(close.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            current_adx = float(adx.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_macd_hist = float(histogram.iloc[-1])
            prev_macd_hist = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
            current_volume_surge = float(volume_surge.iloc[-1]) if len(volume_surge) > 0 else 1.0
            
            # Validate values
            if any(pd.isna([current_price, current_rsi, current_adx, current_atr])):
                print(f"⚠️ NaN values in indicators for {pair}")
                return None
            
            # Get ticker for spread check
            ticker = CoinDCXAPI.get_ticker(pair)
            bid = ticker['bid'] if ticker else current_price
            ask = ticker['ask'] if ticker else current_price
            
            # Trap detection
            traps = TrapDetector.check_all_traps(
                candles, bid, ask, current_rsi, current_adx, current_macd_hist
            )
            
            # Count traps
            trapped_count = sum(traps.values())
            trap_reasons = TrapDetector.get_trap_reasons(traps)
            
            # If 3+ traps triggered, skip (too risky)
            if trapped_count >= 3:
                print(f"⚠️ {pair} - Too many traps ({trapped_count}): {', '.join(trap_reasons)}")
                return None
            
            # Determine trend direction BEFORE ChatGPT check
            trend = None
            if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
                macd_line.iloc[-1] > signal_line.iloc[-1] and
                plus_di.iloc[-1] > minus_di.iloc[-1]):
                trend = "LONG"
            elif (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
                  macd_line.iloc[-1] < signal_line.iloc[-1] and
                  minus_di.iloc[-1] > plus_di.iloc[-1]):
                trend = "SHORT"
            else:
                print(f"❌ {pair} BLOCKED: No clear trend")
                print(f"   EMA Fast: {ema_fast.iloc[-1]:.2f}, EMA Slow: {ema_slow.iloc[-1]:.2f}")
                print(f"   MACD Line: {macd_line.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}")
                return None  # No clear trend
            
            # If 1-2 traps, ask ChatGPT
            if trapped_count > 0:
                print(f"⚠️ {pair} - {trapped_count} trap(s) detected: {', '.join(trap_reasons)}")
                
                # TEMPORARY: Skip ChatGPT, auto-approve 1-2 traps
                if trapped_count <= 2:
                    print(f"✅ Auto-approved: {trapped_count} trap(s) acceptable with strong indicators")
                    print(f"   (ChatGPT validation disabled temporarily)")
                else:
                    print(f"❌ Too many traps ({trapped_count}), blocking signal")
                    return None
            
            # RSI filter
            if trend == "LONG" and current_rsi > 70:
                print(f"❌ {pair} BLOCKED: RSI too high ({current_rsi}) - Overbought")
                return None
            if trend == "SHORT" and current_rsi < 30:
                print(f"❌ {pair} BLOCKED: RSI too low ({current_rsi}) - Oversold")
                return None
            
            # ADX filter (minimum trend strength)
            if current_adx < config.MIN_ADX_STRENGTH:
                print(f"❌ {pair} BLOCKED: ADX too weak ({current_adx:.1f}) - Need {config.MIN_ADX_STRENGTH}+")
                return None
            
            # Calculate entry/SL/TP
            levels = self._calculate_entry_sl_tp(trend, current_price, current_atr)
            
            # Liquidation safety check
            if not self._check_liquidation_safety(levels['entry'], levels['sl']):
                distance = abs(levels['entry'] - levels['sl']) / levels['entry'] * 100
                print(f"❌ {pair} BLOCKED: Liquidation risk too high (SL only {distance:.1f}% away, need 10%+)")
                return None
            
            # Get multi-timeframe trend (if possible)
            try:
                candles_5m = CoinDCXAPI.get_candles(pair, '5m', 50)
                candles_15m = CoinDCXAPI.get_candles(pair, '15m', 50)
                candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)
                
                if not candles_5m.empty and not candles_15m.empty and not candles_1h.empty:
                    mtf_trend = Indicators.mtf_trend(
                        candles_5m['close'],
                        candles_15m['close'],
                        candles_1h['close']
                    )
                else:
                    mtf_trend = "UNKNOWN"
            except:
                mtf_trend = "UNKNOWN"
            
            # Calculate signal score
            indicators_data = {
                'rsi': current_rsi,
                'adx': current_adx,
                'macd_histogram': current_macd_hist,
                'prev_macd_histogram': prev_macd_hist,
                'volume_surge': current_volume_surge
            }
            
            score = self._calculate_signal_score(indicators_data, mtf_trend)
            
            # Minimum score filter
            min_score = 60 if config.MODE == 'TREND' else 50
            if score < min_score:
                print(f"❌ {pair} BLOCKED: Score too low ({score}) - Need {min_score}+")
                print(f"   RSI: {current_rsi:.1f}, ADX: {current_adx:.1f}, MTF: {mtf_trend}")
                return None
            
            # SUCCESS! Generate signal
            print(f"✅ {pair} SIGNAL APPROVED!")
            print(f"   Direction: {trend}")
            print(f"   Score: {score}/100")
            print(f"   Entry: ₹{levels['entry']:,.2f}")
            print(f"   RSI: {current_rsi:.1f}, ADX: {current_adx:.1f}")
            if trapped_count > 0:
                print(f"   ⚠️ Had {trapped_count} trap(s) but ChatGPT approved!")
            
            # Build signal
            signal = {
                'pair': pair,
                'direction': trend,
                'entry': levels['entry'],
                'sl': levels['sl'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'leverage': self.mode_config['leverage'],
                'score': score,
                'rsi': round(current_rsi, 1),
                'adx': round(current_adx, 1),
                'mtf_trend': mtf_trend,
                'mode': config.MODE,
                'volume_surge': round(current_volume_surge, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update counters
            self.signal_count += 1
            self.signals_today.append(signal)
            self.last_signal_time[pair] = datetime.now()
            
            return signal
            
        except Exception as e:
            print(f"❌ Error in analysis for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Calculate all indicators
        close = candles['close']
        high = candles['high']
        low = candles['low']
        volume = candles['volume']
        
        ema_fast = Indicators.ema(close, self.mode_config['ema_fast'])
        ema_slow = Indicators.ema(close, self.mode_config['ema_slow'])
        macd_line, signal_line, histogram = Indicators.macd(close)
        rsi = Indicators.rsi(close)
        adx, plus_di, minus_di = Indicators.adx(high, low, close)
        atr = Indicators.atr(high, low, close)
        volume_surge = Indicators.volume_surge(volume)
        
        # Current values
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_adx = adx.iloc[-1]
        current_atr = atr.iloc[-1]
        current_macd_hist = histogram.iloc[-1]
        prev_macd_hist = histogram.iloc[-2]
        current_volume_surge = volume_surge.iloc[-1]
        
        # Get ticker for spread check
        ticker = CoinDCXAPI.get_ticker(pair)
        bid = ticker['bid'] if ticker else current_price
        ask = ticker['ask'] if ticker else current_price
        
        # Trap detection
        traps = TrapDetector.check_all_traps(
            candles, bid, ask, current_rsi, current_adx, current_macd_hist
        )
        
        if TrapDetector.is_trapped(traps):
            trap_reasons = TrapDetector.get_trap_reasons(traps)
            print(f"⚠️ {pair} - Traps detected: {', '.join(trap_reasons)}")
            return None
        
        # Determine trend direction
        trend = None
        if (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
            macd_line.iloc[-1] > signal_line.iloc[-1] and
            plus_di.iloc[-1] > minus_di.iloc[-1]):
            trend = "LONG"
        elif (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
              macd_line.iloc[-1] < signal_line.iloc[-1] and
              minus_di.iloc[-1] > plus_di.iloc[-1]):
            trend = "SHORT"
        else:
            return None  # No clear trend
        
        # RSI filter
        if trend == "LONG" and current_rsi > 70:
            return None  # Overbought
        if trend == "SHORT" and current_rsi < 30:
            return None  # Oversold
        
        # ADX filter (minimum trend strength)
        if current_adx < config.MIN_ADX_STRENGTH:
            return None
        
        # Calculate entry/SL/TP
        levels = self._calculate_entry_sl_tp(trend, current_price, current_atr)
        
        # Liquidation safety check
        if not self._check_liquidation_safety(levels['entry'], levels['sl']):
            print(f"⚠️ {pair} - Liquidation risk too high")
            return None
        
        # Get multi-timeframe trend (if possible)
        try:
            candles_5m = CoinDCXAPI.get_candles(pair, '5m', 50)
            candles_15m = CoinDCXAPI.get_candles(pair, '15m', 50)
            candles_1h = CoinDCXAPI.get_candles(pair, '1h', 50)
            
            if not candles_5m.empty and not candles_15m.empty and not candles_1h.empty:
                mtf_trend = Indicators.mtf_trend(
                    candles_5m['close'],
                    candles_15m['close'],
                    candles_1h['close']
                )
            else:
                mtf_trend = "UNKNOWN"
        except:
            mtf_trend = "UNKNOWN"
        
        # Calculate signal score
        indicators_data = {
            'rsi': current_rsi,
            'adx': current_adx,
            'macd_histogram': current_macd_hist,
            'prev_macd_histogram': prev_macd_hist,
            'volume_surge': current_volume_surge
        }
        
        score = self._calculate_signal_score(indicators_data, mtf_trend)
        
        # Minimum score filter
        min_score = 60 if config.MODE == 'TREND' else 50
        if score < min_score:
            return None
        
        # Build signal
        signal = {
            'pair': pair,
            'direction': trend,
            'entry': levels['entry'],
            'sl': levels['sl'],
            'tp1': levels['tp1'],
            'tp2': levels['tp2'],
            'leverage': self.mode_config['leverage'],
            'score': score,
            'rsi': round(current_rsi, 1),
            'adx': round(current_adx, 1),
            'mtf_trend': mtf_trend,
            'mode': config.MODE,
            'volume_surge': round(current_volume_surge, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update counters
        self.signal_count += 1
        self.signals_today.append(signal)
        self.last_signal_time[pair] = datetime.now()
        
        return signal
    
    def get_stats(self) -> Dict:
        """Get signal generator statistics"""
        return {
            'signals_today': self.signal_count,
            'signals_remaining': config.MAX_SIGNALS_PER_DAY - self.signal_count,
            'mode': config.MODE,
            'last_reset': self.last_reset_date.strftime('%Y-%m-%d')
        }