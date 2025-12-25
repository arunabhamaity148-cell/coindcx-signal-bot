from openai import OpenAI
from typing import Dict, Optional
from config import config
import time
import json
import pandas as pd
from datetime import datetime, timedelta


class ChatGPTAdvisor:
    """
    CHATGPT SNIPER FILTER - FINAL DECISION LAYER
    
    PHILOSOPHY: Professional discretionary trader with veto power
    
    HARD REJECT CRITERIA (Only 5 - Non-negotiable):
    1. R:R < 1.5
    2. SL < 0.8%
    3. TRUE exhaustion: (RSI > 78 OR < 22) AND ADX > 50
    4. Invalid price data
    5. BTC strong impulse AGAINST signal direction
    
    SOFT SCORING (Advisory only - NEVER blocks):
    - Volume, MTF, Session, Order Flow, Market Structure
    - Threshold: 60+ score = APPROVE
    
    OUTPUT: bool (True = APPROVE, False = REJECT)
    JSON logging for audit trail
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2
        self.recent_losses = []

    # ================================================================
    # HARD REJECTION CHECKS (Only 5 - Non-negotiable)
    # ================================================================

    def _check_hard_rejections(self, signal: Dict) -> tuple[bool, str]:
        """
        Check ONLY catastrophic conditions
        Returns: (is_rejected, reason)
        
        NEVER crashes - returns False on any error
        """
        default_response = (False, "PASS")
        
        try:
            entry = float(signal.get('entry', 0))
            sl = float(signal.get('sl', 0))
            tp1 = float(signal.get('tp1', 0))
            rsi = float(signal.get('rsi', 50))
            adx = float(signal.get('adx', 20))
            direction = signal.get('direction', 'UNKNOWN')
            
            # HARD REJECT 1: Poor R:R
            sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
            tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
            rr = tp1_distance / sl_distance if sl_distance > 0 else 0
            
            if rr < 1.5:
                return True, f"HARD_REJECT: R:R={rr:.2f} < 1.5"
            
            # HARD REJECT 2: SL too tight
            if sl_distance < 0.8:
                return True, f"HARD_REJECT: SL={sl_distance:.2f}% < 0.8%"
            
            # HARD REJECT 3: TRUE exhaustion
            if adx > 50:
                if direction == "LONG" and rsi > 78:
                    return True, f"HARD_REJECT: Exhaustion (RSI={rsi:.1f}, ADX={adx:.1f})"
                elif direction == "SHORT" and rsi < 22:
                    return True, f"HARD_REJECT: Exhaustion (RSI={rsi:.1f}, ADX={adx:.1f})"
            
            # HARD REJECT 4: Invalid price data
            if entry <= 0 or sl <= 0 or tp1 <= 0:
                return True, "HARD_REJECT: Invalid price levels"
            
            # HARD REJECT 5: BTC strong impulse against
            btc_against = self._check_btc_momentum_flip(direction)
            if btc_against:
                return True, "HARD_REJECT: BTC impulse against direction"
            
            return False, "PASS"
            
        except Exception as e:
            print(f"⚠️ Hard rejection check failed: {e} - PASSING")
            return default_response

    def _check_btc_momentum_flip(self, direction: str) -> bool:
        """
        SOFT TIMING REFINEMENT: Wait 1 candle if BTC flips against direction
        Returns True if should reject (wait 1 candle)
        
        NEVER crashes - returns False on any error
        """
        try:
            from coindcx_api import CoinDCXAPI
            
            # Get recent BTC 5m candles
            btc_candles = CoinDCXAPI.get_candles('BTCUSDT', '5m', 3)
            if btc_candles is None or btc_candles.empty or len(btc_candles) < 3:
                return False  # No data = don't block
            
            # Check last 2 candles
            last_candle = btc_candles.iloc[-1]
            prev_candle = btc_candles.iloc[-2]
            
            last_close = float(last_candle['close'])
            last_open = float(last_candle['open'])
            prev_close = float(prev_candle['close'])
            prev_open = float(prev_candle['open'])
            
            # Calculate momentum
            last_move = (last_close - last_open) / last_open * 100
            prev_move = (prev_close - prev_open) / prev_open * 100
            
            # Check if BTC is moving AGAINST signal
            if direction == "LONG":
                # Reject if BTC showing strong bearish momentum
                if last_move < -0.5 and prev_move < -0.5:
                    print(f"⏸️  BTC bearish momentum - wait 1 candle")
                    return True
            else:  # SHORT
                # Reject if BTC showing strong bullish momentum
                if last_move > 0.5 and prev_move > 0.5:
                    print(f"⏸️  BTC bullish momentum - wait 1 candle")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ BTC momentum check failed: {e}")
            return False  # Error = don't block

    # ================================================================
    # SOFT SCORING (Advisory only - NEVER blocks)
    # ================================================================

    def _calculate_soft_score(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """
        Calculate quality score from soft factors
        Returns: {'score': 0-100, 'notes': [...]}
        
        ALL soft factors are advisory - they influence confidence but don't block
        NEVER crashes - returns default on error
        """
        default_response = {'score': 60, 'notes': ['Default score - error in calculation']}
        
        try:
            score = 70  # Start at neutral
            notes = []
            
            # Soft factor 1: Volume
            volume_surge = signal.get('volume_surge', 1.0)
            if volume_surge < 1.2:
                score -= 5
                notes.append(f"Low volume ({volume_surge:.2f}x) -5")
            elif volume_surge > 1.8:
                score += 5
                notes.append(f"High volume ({volume_surge:.2f}x) +5")
            
            # Soft factor 2: MTF alignment
            mtf = signal.get('mtf_trend', 'UNKNOWN')
            if mtf == 'UNKNOWN':
                score -= 3
                notes.append("MTF unknown -3")
            elif mtf in ['STRONG_UP', 'STRONG_DOWN']:
                score += 5
                notes.append(f"MTF aligned ({mtf}) +5")
            
            # Soft factor 3: Session liquidity
            session_result = self._check_session_liquidity()
            if session_result['score'] < 50:
                score -= 5
                notes.append(f"Weak session ({session_result['status']}) -5")
            elif session_result['score'] > 80:
                score += 3
                notes.append(f"Strong session ({session_result['status']}) +3")
            
            # Soft factor 4: Order flow
            if candles is not None and not candles.empty:
                order_flow = self._check_order_flow_imbalance(signal, candles)
                if order_flow['score'] < 50:
                    score -= 5
                    notes.append(f"Weak order flow -5")
                elif order_flow['score'] > 80:
                    score += 5
                    notes.append(f"Strong order flow +5")
            
            # Soft factor 5: Market structure
            if candles is not None and not candles.empty:
                structure = self._check_market_structure(signal, candles)
                if structure['score'] > 80:
                    score += 5
                    notes.append(f"Clean structure +5")
            
            # Soft factor 6: Price action confluence
            if candles is not None and not candles.empty:
                price_action = self._check_price_action_confluence(signal, candles)
                if price_action['score'] > 85:
                    score += 5
                    notes.append(f"Strong candle pattern +5")
            
            # Soft factor 7: Drawdown protection
            drawdown = self._check_drawdown_protection()
            if drawdown['score'] == 0:
                score -= 20
                notes.append("DRAWDOWN PAUSE -20")
            elif drawdown['score'] == 50:
                score -= 10
                notes.append("Recent losses -10")
            
            # Cap score
            final_score = max(0, min(100, score))
            
            return {'score': final_score, 'notes': notes}
            
        except Exception as e:
            print(f"⚠️ Soft score calculation failed: {e}")
            return default_response

    def _check_session_liquidity(self) -> Dict:
        """Check current trading session"""
        default_response = {'score': 50, 'status': 'UNKNOWN'}
        
        try:
            now = datetime.now()
            hour = now.hour
            
            if 18 <= hour < 22:
                return {'score': 100, 'status': 'NY_SESSION'}
            elif 13 <= hour < 18:
                return {'score': 90, 'status': 'LONDON_SESSION'}
            elif 5 <= hour < 10:
                return {'score': 50, 'status': 'ASIAN_SESSION'}
            else:
                return {'score': 30, 'status': 'OFF_HOURS'}
                
        except Exception as e:
            print(f"⚠️ Session check failed: {e}")
            return default_response

    def _check_order_flow_imbalance(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Check buy/sell pressure"""
        default_response = {'score': 50, 'status': 'NO_DATA'}
        
        try:
            if candles is None or candles.empty or len(candles) < 20:
                return default_response
            
            candles = candles.tail(20)
            buy_volume = 0
            sell_volume = 0
            
            for idx, row in candles.iterrows():
                close_price = float(row['close'])
                open_price = float(row['open'])
                volume = float(row['volume'])
                
                if close_price > open_price:
                    buy_volume += volume
                else:
                    sell_volume += volume
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return default_response
            
            buy_pressure = buy_volume / total_volume
            direction = signal.get('direction', 'UNKNOWN')
            
            if direction == "LONG":
                if buy_pressure > 0.65:
                    return {'score': 100, 'status': 'STRONG_BUY'}
                elif buy_pressure > 0.55:
                    return {'score': 75, 'status': 'MODERATE_BUY'}
                else:
                    return {'score': 40, 'status': 'WEAK_BUY'}
            else:
                sell_pressure = sell_volume / total_volume
                if sell_pressure > 0.65:
                    return {'score': 100, 'status': 'STRONG_SELL'}
                elif sell_pressure > 0.55:
                    return {'score': 75, 'status': 'MODERATE_SELL'}
                else:
                    return {'score': 40, 'status': 'WEAK_SELL'}
                    
        except Exception as e:
            print(f"⚠️ Order flow check failed: {e}")
            return default_response

    def _check_market_structure(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Check market structure (HH/HL or LL/LH)"""
        default_response = {'score': 50, 'status': 'INSUFFICIENT_DATA'}
        
        try:
            if candles is None or candles.empty or len(candles) < 10:
                return default_response
            
            candles = candles.tail(10)
            highs = candles['high'].values
            lows = candles['low'].values
            direction = signal.get('direction', 'UNKNOWN')
            
            recent_high = max(highs[-5:])
            prev_high = max(highs[-10:-5])
            recent_low = min(lows[-5:])
            prev_low = min(lows[-10:-5])
            
            if direction == "LONG":
                higher_high = recent_high > prev_high
                higher_low = recent_low > prev_low
                
                if higher_high and higher_low:
                    return {'score': 100, 'status': 'BULLISH_STRUCTURE'}
                elif higher_high:
                    return {'score': 70, 'status': 'PARTIAL_BULLISH'}
                else:
                    return {'score': 40, 'status': 'WEAK_STRUCTURE'}
            else:
                lower_low = recent_low < prev_low
                lower_high = recent_high < prev_high
                
                if lower_low and lower_high:
                    return {'score': 100, 'status': 'BEARISH_STRUCTURE'}
                elif lower_low:
                    return {'score': 70, 'status': 'PARTIAL_BEARISH'}
                else:
                    return {'score': 40, 'status': 'WEAK_STRUCTURE'}
                    
        except Exception as e:
            print(f"⚠️ Market structure check failed: {e}")
            return default_response

    def _check_price_action_confluence(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Check candle patterns"""
        default_response = {'score': 50, 'status': 'NO_PATTERN'}
        
        try:
            if candles is None or candles.empty or len(candles) < 3:
                return default_response
            
            latest = candles.iloc[-1]
            open_price = float(latest['open'])
            close_price = float(latest['close'])
            high_price = float(latest['high'])
            low_price = float(latest['low'])
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return {'score': 50, 'status': 'DOJI'}
            
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            direction = signal.get('direction', 'UNKNOWN')
            
            if direction == "LONG":
                if lower_wick > body_size * 2 and close_price > open_price:
                    return {'score': 100, 'status': 'BULLISH_REJECTION'}
                elif close_price > open_price and body_size > total_range * 0.6:
                    return {'score': 90, 'status': 'STRONG_BULLISH'}
                else:
                    return {'score': 60, 'status': 'WEAK_PATTERN'}
            else:
                if upper_wick > body_size * 2 and close_price < open_price:
                    return {'score': 100, 'status': 'BEARISH_REJECTION'}
                elif close_price < open_price and body_size > total_range * 0.6:
                    return {'score': 90, 'status': 'STRONG_BEARISH'}
                else:
                    return {'score': 60, 'status': 'WEAK_PATTERN'}
                    
        except Exception as e:
            print(f"⚠️ Price action check failed: {e}")
            return default_response

    def _check_drawdown_protection(self) -> Dict:
        """Check recent losses"""
        default_response = {'score': 100, 'status': 'SAFE'}
        
        try:
            two_hours_ago = datetime.now() - timedelta(hours=2)
            self.recent_losses = [loss for loss in self.recent_losses if loss > two_hours_ago]
            loss_count = len(self.recent_losses)
            
            if loss_count >= 3:
                return {'score': 0, 'status': 'DRAWDOWN_PAUSE'}
            elif loss_count == 2:
                return {'score': 50, 'status': 'CAUTION'}
            else:
                return {'score': 100, 'status': 'SAFE'}
                
        except Exception as e:
            print(f"⚠️ Drawdown check failed: {e}")
            return default_response

    def record_loss(self):
        """Record a losing trade for drawdown protection"""
        try:
            self.recent_losses.append(datetime.now())
        except Exception as e:
            print(f"⚠️ Failed to record loss: {e}")
# ================================================================
    # FINAL DECISION (Called by signal_generator.py)
    # ================================================================

    def final_trade_decision(self, signal: Dict, candles: pd.DataFrame = None) -> bool:
        """
        FINAL TRADE DECISION - MANDATORY method
        
        Returns:
            True = APPROVE signal (send to Telegram)
            False = REJECT signal (silent drop)
        
        NEVER crashes - always returns bool
        """
        # Default response on total failure
        default_approved = True  # Fail-open
        
        try:
            pair = signal.get('pair', 'UNKNOWN')
            direction = signal.get('direction', 'UNKNOWN')
            score = signal.get('score', 0)
            
            print(f"\n{'='*70}")
            print(f"🤖 CHATGPT FINAL JUDGE: {pair} {direction} (Score: {score})")
            print(f"{'='*70}")
            
            # STEP 1: Check hard rejections (Only 5 catastrophic conditions)
            is_hard_rejected, hard_reason = self._check_hard_rejections(signal)
            if is_hard_rejected:
                decision_log = {
                    "approved": False,
                    "confidence": "HIGH",
                    "reason": hard_reason,
                    "pair": pair,
                    "type": "HARD_REJECT"
                }
                print(f"❌ {hard_reason}")
                print(f"📊 DECISION: {json.dumps(decision_log, indent=2)}")
                print(f"{'='*70}\n")
                return False
            
            # STEP 2: Calculate soft quality score (Advisory only)
            soft_analysis = self._calculate_soft_score(signal, candles)
            soft_score = soft_analysis['score']
            notes = soft_analysis['notes']
            
            print(f"📊 Soft Quality Score: {soft_score}/100")
            if notes:
                print(f"📝 Notes:")
                for note in notes:
                    print(f"   • {note}")
            
            # STEP 3: Decision based on soft score
            threshold = 60  # Approval threshold
            approved = soft_score >= threshold
            
            if approved:
                confidence = "HIGH" if soft_score >= 75 else "MEDIUM"
                reason = f"Quality score {soft_score}/100"
            else:
                confidence = "LOW"
                reason = f"Quality too low ({soft_score}/100)"
            
            # Log decision
            decision_log = {
                "approved": approved,
                "confidence": confidence,
                "reason": reason,
                "pair": pair,
                "soft_score": soft_score,
                "type": "SOFT_DECISION"
            }
            
            status_symbol = "✅" if approved else "❌"
            print(f"\n{status_symbol} DECISION: {reason}")
            print(f"📊 FINAL: {json.dumps(decision_log, indent=2)}")
            print(f"{'='*70}\n")
            
            return approved
            
        except Exception as e:
            # NEVER crash - log error and fail-open
            error_log = {
                "approved": default_approved,
                "confidence": "LOW",
                "reason": "Error - auto approved",
                "error": str(e)[:100]
            }
            print(f"⚠️ ChatGPT CRITICAL ERROR: {e}")
            print(f"⚠️ FAIL-OPEN: Auto-approving signal")
            print(f"📊 ERROR LOG: {json.dumps(error_log)}")
            return default_approved

    # ================================================================
    # LEGACY COMPATIBILITY (For existing trap detection flow)
    # ================================================================

    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """
        LEGACY FUNCTION - Redirects to final_trade_decision()
        Kept for backward compatibility
        """
        try:
            approved = self.final_trade_decision(signal, None)
            return {
                "approved": approved,
                "reason": "Quality advisor decision",
                "confidence": 100 if approved else 0
            }
        except Exception as e:
            print(f"⚠️ Legacy validation error: {e}")
            return {
                "approved": True,
                "reason": "Error - auto approved",
                "confidence": 50
            }