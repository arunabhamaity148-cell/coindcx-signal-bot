from openai import OpenAI
from typing import Dict, Optional
from config import config
import time
import json
import pandas as pd
from datetime import datetime, timedelta


class ChatGPTAdvisor:
    """
    ULTRA SMART FINAL VALIDATOR
    
    Philosophy: Context-aware decisions without hard blocks
    - Validates price action context
    - Detects timing issues (late/early/chasing)
    - Confirms candle patterns support direction
    - NO arbitrary hard rules from GPT
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2
        self.recent_losses = []

    def _build_smart_prompt(self, signal: Dict, candles: pd.DataFrame) -> str:
        """Build context-aware prompt for ChatGPT"""
        
        # Recent price action
        recent_10 = candles.tail(10)
        current_price = float(signal['entry'])
        price_10_candles_ago = float(recent_10.iloc[0]['close'])
        recent_move_pct = ((current_price - price_10_candles_ago) / price_10_candles_ago) * 100
        
        # Last 3 candles analysis
        last_3 = candles.tail(3)
        candle_1 = last_3.iloc[-3]
        candle_2 = last_3.iloc[-2]
        candle_3 = last_3.iloc[-1]  # Current candle
        
        # Candle momentum
        c3_body = abs(float(candle_3['close']) - float(candle_3['open']))
        c3_range = float(candle_3['high']) - float(candle_3['low'])
        c3_body_ratio = (c3_body / c3_range * 100) if c3_range > 0 else 0
        
        # Wick analysis
        if float(candle_3['close']) > float(candle_3['open']):  # Bullish
            upper_wick = float(candle_3['high']) - float(candle_3['close'])
            lower_wick = float(candle_3['open']) - float(candle_3['low'])
        else:  # Bearish
            upper_wick = float(candle_3['high']) - float(candle_3['open'])
            lower_wick = float(candle_3['close']) - float(candle_3['low'])
        
        wick_ratio = (max(upper_wick, lower_wick) / c3_range * 100) if c3_range > 0 else 0
        
        # Volume context
        volume_last_3 = [float(c['volume']) for c in [candle_1, candle_2, candle_3]]
        volume_trend = "INCREASING" if volume_last_3[-1] > volume_last_3[0] else "DECREASING"
        
        # Distance from recent high/low
        recent_20 = candles.tail(20)
        recent_high = float(recent_20['high'].max())
        recent_low = float(recent_20['low'].min())
        
        distance_from_high = ((recent_high - current_price) / current_price) * 100
        distance_from_low = ((current_price - recent_low) / current_price) * 100
        
        prompt = f"""You are analyzing a {signal['direction']} trade signal for {signal['pair']}.

SIGNAL DATA:
- Direction: {signal['direction']}
- Entry: ₹{signal['entry']:,.2f}
- Stop Loss: ₹{signal['sl']:,.2f}
- Take Profit 1: ₹{signal['tp1']:,.2f}
- Risk/Reward: {abs(signal['tp1'] - signal['entry']) / abs(signal['entry'] - signal['sl']):.2f}:1
- Mode: {signal['mode']} ({signal['timeframe']})
- Score: {signal['score']}/100
- RSI: {signal['rsi']} | ADX: {signal['adx']}

PRICE CONTEXT:
- Recent 10-candle move: {recent_move_pct:+.2f}%
- Current vs Recent High: {distance_from_high:.2f}% below
- Current vs Recent Low: {distance_from_low:.2f}% above
- Volume trend (last 3): {volume_trend}

CURRENT CANDLE:
- Body size: {c3_body_ratio:.1f}% of range
- Largest wick: {wick_ratio:.1f}% of range
- Pattern: {"Strong momentum" if c3_body_ratio > 60 else "Weak momentum" if c3_body_ratio < 30 else "Normal"}

VALIDATION CHECKLIST (Answer YES/NO for each):

1. TIMING: Is this entry at a good point (not chasing, not too late)?
2. MOMENTUM: Does current candle momentum support the {signal['direction']} direction?
3. RISK: Is the setup worth taking given current market position?
4. CONVICTION: Would you personally take this trade as a manual trader?

Respond ONLY in this JSON format:
{{
    "approved": true/false,
    "confidence": 0-100,
    "reason": "1-2 sentence explanation",
    "timing": "GOOD/LATE/EARLY/CHASING",
    "red_flags": ["list any concerns"]
}}"""

        return prompt

    def _call_chatgpt_smart(self, prompt: str) -> Optional[Dict]:
        """Call ChatGPT with smart analysis"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert crypto trader who validates trade setups. You are conservative but not overly strict. You focus on timing, momentum, and risk/reward. Respond ONLY in valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    timeout=self.timeout
                )
                
                result = response.choices[0].message.content.strip()
                result = result.replace("```json", "").replace("```", "").strip()
                
                return json.loads(result)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ ChatGPT JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"⚠️ ChatGPT call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return None

    def _check_basic_safety(self, signal: Dict) -> tuple[bool, str]:
        """Quick safety checks (no API call)"""
        try:
            entry = float(signal.get('entry', 0))
            sl = float(signal.get('sl', 0))
            tp1 = float(signal.get('tp1', 0))

            if entry <= 0 or sl <= 0 or tp1 <= 0:
                return False, "INVALID_PRICES"

            if abs(sl - entry) < 0.000001:
                return False, "SL_EQUALS_ENTRY"

            sl_distance = abs(entry - sl) / entry * 100
            tp1_distance = abs(tp1 - entry) / entry * 100
            rr = tp1_distance / sl_distance if sl_distance > 0 else 0

            if rr < 1.5:
                return False, f"LOW_RR: {rr:.2f}"

            if sl_distance < 0.8:
                return False, f"SL_TOO_TIGHT: {sl_distance:.2f}%"

            direction = signal.get('direction', 'UNKNOWN')
            if direction == "LONG":
                if sl >= entry or tp1 <= entry:
                    return False, "LONG_LOGIC_ERROR"
            elif direction == "SHORT":
                if sl <= entry or tp1 >= entry:
                    return False, "SHORT_LOGIC_ERROR"

            return True, "BASIC_SAFETY_OK"

        except Exception as e:
            return False, f"SAFETY_ERROR: {str(e)[:50]}"

    def _check_drawdown_protection(self) -> tuple[bool, str]:
        """Check recent losses"""
        try:
            two_hours_ago = datetime.now() - timedelta(hours=2)
            self.recent_losses = [loss for loss in self.recent_losses if loss > two_hours_ago]
            loss_count = len(self.recent_losses)

            if loss_count >= 3:
                return False, f"DRAWDOWN_PAUSE: {loss_count} losses in 2h"

            return True, "DRAWDOWN_OK"

        except:
            return True, "DRAWDOWN_CHECK_SKIPPED"

    def record_loss(self):
        """Record a losing trade"""
        try:
            self.recent_losses.append(datetime.now())
        except:
            pass

    def final_trade_decision(self, signal: Dict, candles: pd.DataFrame = None) -> bool:
        """
        ULTRA SMART FINAL VALIDATOR
        
        Returns:
            True = Signal approved (high quality)
            False = Signal rejected (timing/context issues)
        """
        try:
            pair = signal.get('pair', 'UNKNOWN')
            direction = signal.get('direction', 'UNKNOWN')
            mode = signal.get('mode', 'UNKNOWN')

            print(f"\n{'='*70}")
            print(f"🤖 ULTRA SMART VALIDATOR: {pair} {direction} | {mode}")
            print(f"{'='*70}")

            # STEP 1: Basic safety (no API call)
            safety_ok, safety_reason = self._check_basic_safety(signal)
            if not safety_ok:
                print(f"❌ BASIC SAFETY FAIL: {safety_reason}")
                print(f"{'='*70}\n")
                return False

            # STEP 2: Drawdown protection (no API call)
            drawdown_ok, drawdown_reason = self._check_drawdown_protection()
            if not drawdown_ok:
                print(f"❌ {drawdown_reason}")
                print(f"{'='*70}\n")
                return False

            # STEP 3: Smart ChatGPT analysis (1 API call)
            if candles is None or candles.empty:
                print(f"⚠️ No candles data - approving based on basic checks")
                print(f"{'='*70}\n")
                return True

            prompt = self._build_smart_prompt(signal, candles)
            gpt_response = self._call_chatgpt_smart(prompt)

            if gpt_response is None:
                print(f"⚠️ ChatGPT failed - BLOCKING for safety")
                print(f"{'='*70}\n")
                return False

            # Parse ChatGPT decision
            approved = gpt_response.get('approved', False)
            confidence = gpt_response.get('confidence', 0)
            reason = gpt_response.get('reason', 'No reason provided')
            timing = gpt_response.get('timing', 'UNKNOWN')
            red_flags = gpt_response.get('red_flags', [])

            # Display analysis
            print(f"📊 ChatGPT Analysis:")
            print(f"   Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}")
            print(f"   Confidence: {confidence}%")
            print(f"   Timing: {timing}")
            print(f"   Reason: {reason}")
            if red_flags:
                print(f"   🚩 Red Flags: {', '.join(red_flags)}")

            print(f"{'='*70}\n")

            return approved

        except Exception as e:
            print(f"⚠️ CRITICAL ERROR: {e}")
            print(f"⚠️ FAIL-SAFE: Signal BLOCKED")
            print(f"{'='*70}\n")
            import traceback
            traceback.print_exc()
            return False

    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """Legacy compatibility"""
        try:
            approved = self.final_trade_decision(signal, None)
            return {
                "approved": approved,
                "reason": "Smart validation",
                "confidence": 100 if approved else 0
            }
        except Exception as e:
            return {
                "approved": False,
                "reason": f"Error: {str(e)[:50]}",
                "confidence": 0
            }