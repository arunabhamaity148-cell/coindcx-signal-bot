from openai import OpenAI
from typing import Dict, List, Optional
from config import config
import time


class ChatGPTAdvisor:
    """
    CHATGPT AS FINAL TRADE JUDGE
    -----------------------------
    Every signal MUST pass ChatGPT validation before sending.
    ChatGPT acts as an experienced discretionary trader.
    
    REJECTION CRITERIA:
    - Late entries (momentum exhausted)
    - Low volume trend continuation
    - Exhausted RSI + high ADX
    - Poor risk/reward setup
    - Prefer pullback/clean breakdown confirmation
    
    OUTPUT: Strict JSON only
    {"approved": true/false}
    
    ✅ NEW: Rejection reason logging (inferred from signal data)
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8  # 8 second timeout
        self.max_retries = 2


    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        """
        Call ChatGPT with timeout protection.
        Returns None on failure (timeout/error).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=100,  # Minimal - only need JSON
                    temperature=0.2,  # Low temp for consistency
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                continue
        
        return None  # All attempts failed


    def _parse_decision(self, response: str) -> bool:
        """
        Parse ChatGPT response - STRICT JSON only.
        Returns False on any parsing error (safety first).
        """
        if not response:
            return False
        
        try:
            # Try to extract JSON
            import json
            
            # Remove markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate structure
            if "approved" not in data:
                print("⚠️ ChatGPT response missing 'approved' field")
                return False
            
            # Return boolean value
            return bool(data["approved"])
        
        except json.JSONDecodeError as e:
            print(f"⚠️ ChatGPT JSON parse error: {e}")
            print(f"   Response: {response[:200]}")
            return False
        
        except Exception as e:
            print(f"⚠️ ChatGPT response validation error: {e}")
            return False


    def _infer_rejection_reasons(self, signal: Dict) -> List[str]:
        """
        Infer likely rejection reasons based on signal data.
        This is LOCAL analysis for logging only.
        Does NOT affect ChatGPT decision.
        
        Returns:
            List of rejection reason strings
        """
        reasons = []
        
        # Extract signal data
        direction = signal.get("direction", "UNKNOWN")
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        
        # Calculate metrics
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # ================================
        # REJECTION REASON INFERENCE
        # ================================
        
        # 1. LOW VOLUME
        if volume_surge < 1.2:
            reasons.append("LOW_VOLUME")
        
        # 2. LATE ENTRY (exhausted momentum)
        if direction == "LONG" and rsi > 65 and adx > 40:
            reasons.append("LATE_ENTRY")
        elif direction == "SHORT" and rsi < 35 and adx > 40:
            reasons.append("LATE_ENTRY")
        
        # 3. EXHAUSTED RSI
        if rsi > 70 or rsi < 30:
            reasons.append("EXHAUSTED_RSI")
        
        # 4. POOR RISK/REWARD
        if rr_ratio < 1.5:
            reasons.append("POOR_RR")
        
        # 5. ADX TOO HIGH WITH EXTREME RSI (overbought/oversold in strong trend)
        if adx > 45 and (rsi > 70 or rsi < 30):
            reasons.append("ADX_TOO_HIGH_WITH_EXTREME_RSI")
        
        # 6. SL TOO CLOSE
        if sl_distance < 1.0:
            reasons.append("SL_TOO_CLOSE")
        
        # 7. CHASING PRICE (high RSI with low volume)
        if direction == "LONG" and rsi > 60 and volume_surge < 1.3:
            reasons.append("CHASING_PRICE")
        elif direction == "SHORT" and rsi < 40 and volume_surge < 1.3:
            reasons.append("CHASING_PRICE")
        
        # 8. WEAK TREND (low ADX with extreme RSI)
        if adx < 25 and (rsi > 65 or rsi < 35):
            reasons.append("WEAK_TREND")
        
        # 9. NO PULLBACK CONFIRMATION (momentum continuation without rest)
        if direction == "LONG" and rsi > 62 and adx > 35:
            reasons.append("NO_PULLBACK")
        elif direction == "SHORT" and rsi < 38 and adx > 35:
            reasons.append("NO_PULLBACK")
        
        return reasons


    def _log_rejection(self, signal: Dict, reasons: List[str]):
        """
        Print detailed rejection log.
        This is for debugging/analysis only.
        """
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        rsi = signal.get("rsi", 0)
        adx = signal.get("adx", 0)
        volume_surge = signal.get("volume_surge", 0)
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp1 = signal.get("tp1", 0)
        
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"🚫 SIGNAL REJECTED BY CHATGPT")
        print(f"{'='*70}")
        print(f"Pair: {pair}")
        print(f"Direction: {direction}")
        print(f"Entry: {entry}")
        print(f"SL: {sl} (Distance: {sl_distance:.2f}%)")
        print(f"TP1: {tp1} (R/R: {rr_ratio:.2f})")
        print(f"\nIndicators:")
        print(f"  RSI: {rsi}")
        print(f"  ADX: {adx}")
        print(f"  Volume Surge: {volume_surge}x")
        print(f"\n🔍 Inferred Rejection Reasons:")
        
        if reasons:
            for reason in reasons:
                print(f"  ❌ {reason}")
        else:
            print(f"  ⚠️  No specific reasons detected (general quality issue)")
        
        print(f"{'='*70}\n")


    def final_trade_decision(self, signal: Dict, candles_data: Dict = None) -> bool:
        """
        🎯 FINAL TRADE DECISION - ChatGPT as Experienced Trader
        
        Args:
            signal: Signal dictionary from rule-based system
            candles_data: Optional recent price action data
        
        Returns:
            True = APPROVED (send signal)
            False = REJECTED (silently drop, no Telegram message)
        """
        
        # Extract signal data
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        tp2 = float(signal.get("tp2", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        score = int(signal.get("score", 0))
        mode = signal.get("mode", "UNKNOWN")
        
        # Calculate metrics
        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        # Build professional trader prompt
        prompt = f"""You are an experienced cryptocurrency futures trader with 10+ years of experience.

SIGNAL TO EVALUATE:
Pair: {pair}
Direction: {direction}
Entry: {entry}
Stop Loss: {sl} ({sl_distance:.2f}% away)
Take Profit 1: {tp1} (R/R: {rr_ratio:.2f})
Mode: {mode}

INDICATORS:
RSI: {rsi}
ADX: {adx}
Volume Surge: {volume_surge}x
Rule-based Score: {score}/100

YOUR EXPERTISE - REJECT IF:
1. Late entry (RSI > 65 for LONG, RSI < 35 for SHORT + ADX > 40)
2. Exhausted momentum (RSI > 70 or < 30 with ADX > 45)
3. Low volume trend continuation (volume_surge < 1.2 + RSI > 60)
4. Poor risk/reward (R/R < 1.5 or SL distance < 1%)
5. Chasing price (no pullback confirmation)

PREFER:
- Fresh breakouts with volume confirmation
- Pullback entries in strong trends
- Clean support/resistance breaks
- RSI 40-60 range with rising ADX
- Volume surge > 1.5x on entry

RESPOND WITH STRICT JSON ONLY (no explanation):
{{"approved": true}}  OR  {{"approved": false}}"""

        # Build messages
        messages = [
            {
                "role": "system",
                "content": "You are a strict, experienced trader. You ONLY respond with JSON format: {\"approved\": true/false}. No explanations. Reject late entries, exhausted momentum, and low-quality setups."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Call ChatGPT with timeout protection
        print(f"🤖 Consulting ChatGPT for {pair} {direction}...")
        response = self._call_chatgpt_with_timeout(messages)
        
        # If ChatGPT failed/timeout → REJECT (safety first)
        if response is None:
            print(f"❌ {pair} REJECTED: ChatGPT timeout/error (safety protocol)")
            # Infer and log reasons even for timeout
            reasons = ["CHATGPT_TIMEOUT_ERROR"]
            self._log_rejection(signal, reasons)
            return False
        
        # Parse decision
        approved = self._parse_decision(response)
        
        if approved:
            print(f"✅ {pair} APPROVED by ChatGPT - Signal will be sent")
        else:
            print(f"❌ {pair} REJECTED by ChatGPT - Signal silently dropped")
            
            # ================================
            # 🔍 INFER AND LOG REJECTION REASONS
            # ================================
            reasons = self._infer_rejection_reasons(signal)
            self._log_rejection(signal, reasons)
        
        return approved


    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        """
        LEGACY FUNCTION - Now redirects to final_trade_decision()
        Kept for backward compatibility with existing trap detection flow.
        """
        approved = self.final_trade_decision(signal)
        
        return {
            "approved": approved,
            "reason": "ChatGPT final decision",
            "confidence": 100 if approved else 0
        }