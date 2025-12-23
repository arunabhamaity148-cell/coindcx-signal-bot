from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time


class ChatGPTAdvisor:
    """
    CHATGPT AS FINAL TRADE JUDGE
    -----------------------------
    Every signal MUST pass ChatGPT validation before sending.
    ChatGPT acts as an experienced discretionary trader.
    
    OUTPUT: Strict JSON only
    {"approved": true/false}
    
    ✅ EXACT REJECTION REASON LOGGING (Deterministic)
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
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
                    max_tokens=100,
                    temperature=0.2,
                    timeout=self.timeout
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                continue
        
        return None


    def _parse_decision(self, response: str) -> bool:
        """
        Parse ChatGPT response - STRICT JSON only.
        Returns False on any parsing error (safety first).
        """
        if not response:
            return False
        
        try:
            import json
            
            # Remove markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate structure
            if "approved" not in data:
                print("⚠️ ChatGPT response missing 'approved' field")
                return False
            
            return bool(data["approved"])
        
        except json.JSONDecodeError as e:
            print(f"⚠️ ChatGPT JSON parse error: {e}")
            print(f"   Response: {response[:200]}")
            return False
        
        except Exception as e:
            print(f"⚠️ ChatGPT response validation error: {e}")
            return False


    def _infer_exact_rejection_reasons(self, signal: Dict) -> List[Tuple[str, str]]:
        """
        Infer EXACT rejection reasons with deterministic rules.
        
        Returns:
            List of (reason_code, detail_message) tuples
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
        # DETERMINISTIC REJECTION RULES
        # ================================
        
        # RULE 1: RSI EXTREME ZONES
        if rsi >= 75:
            reasons.append((
                "RSI_OVERBOUGHT",
                f"RSI={rsi:.1f} >= 75 (extreme overbought)"
            ))
        elif rsi <= 25:
            reasons.append((
                "RSI_OVERSOLD",
                f"RSI={rsi:.1f} <= 25 (extreme oversold)"
            ))
        
        # RULE 2: RSI OVERBOUGHT/OVERSOLD (moderate)
        elif rsi >= 70:
            reasons.append((
                "RSI_OVERBOUGHT",
                f"RSI={rsi:.1f} >= 70 (overbought zone)"
            ))
        elif rsi <= 30:
            reasons.append((
                "RSI_OVERSOLD",
                f"RSI={rsi:.1f} <= 30 (oversold zone)"
            ))
        
        # RULE 3: EXHAUSTED TREND (high ADX + extreme RSI)
        if adx >= 45:
            if direction == "LONG" and rsi >= 65:
                reasons.append((
                    "EXHAUSTED_TREND",
                    f"LONG with ADX={adx:.1f} + RSI={rsi:.1f} (uptrend exhaustion)"
                ))
            elif direction == "SHORT" and rsi <= 35:
                reasons.append((
                    "EXHAUSTED_TREND",
                    f"SHORT with ADX={adx:.1f} + RSI={rsi:.1f} (downtrend exhaustion)"
                ))
        
        # RULE 4: LATE ENTRY (moderate ADX + unfavorable RSI)
        if adx >= 40:
            if direction == "LONG" and rsi >= 68:
                reasons.append((
                    "LATE_ENTRY",
                    f"LONG entry at RSI={rsi:.1f} with ADX={adx:.1f} (momentum exhausted)"
                ))
            elif direction == "SHORT" and rsi <= 32:
                reasons.append((
                    "LATE_ENTRY",
                    f"SHORT entry at RSI={rsi:.1f} with ADX={adx:.1f} (momentum exhausted)"
                ))
        
        # RULE 5: LOW VOLUME
        if volume_surge < 1.2:
            reasons.append((
                "LOW_VOLUME",
                f"Volume surge={volume_surge:.2f}x < 1.2x (weak confirmation)"
            ))
        elif volume_surge < 1.5:
            # Moderate volume with unfavorable conditions
            if (direction == "LONG" and rsi >= 62) or (direction == "SHORT" and rsi <= 38):
                reasons.append((
                    "LOW_VOLUME",
                    f"Volume surge={volume_surge:.2f}x < 1.5x with RSI={rsi:.1f} (insufficient momentum)"
                ))
        
        # RULE 6: POOR RISK/REWARD
        if rr_ratio < 1.5:
            reasons.append((
                "POOR_RR",
                f"Risk/Reward={rr_ratio:.2f} < 1.5 (unfavorable risk profile)"
            ))
        elif rr_ratio < 2.0:
            # Marginal R/R with weak setup
            if adx < 30 or volume_surge < 1.3:
                reasons.append((
                    "POOR_RR",
                    f"Risk/Reward={rr_ratio:.2f} < 2.0 with weak setup (ADX={adx:.1f}, Volume={volume_surge:.2f}x)"
                ))
        
        # RULE 7: SL TOO TIGHT
        if sl_distance < 0.8:
            reasons.append((
                "SL_TOO_TIGHT",
                f"SL distance={sl_distance:.2f}% < 0.8% (high whipsaw risk)"
            ))
        elif sl_distance < 1.0:
            reasons.append((
                "SL_TOO_TIGHT",
                f"SL distance={sl_distance:.2f}% < 1.0% (tight stop)"
            ))
        
        # RULE 8: CHASING PRICE (no pullback)
        if direction == "LONG" and rsi >= 62 and adx >= 35 and volume_surge < 1.5:
            reasons.append((
                "CHASING_PRICE",
                f"LONG at RSI={rsi:.1f} without pullback (ADX={adx:.1f}, Volume={volume_surge:.2f}x)"
            ))
        elif direction == "SHORT" and rsi <= 38 and adx >= 35 and volume_surge < 1.5:
            reasons.append((
                "CHASING_PRICE",
                f"SHORT at RSI={rsi:.1f} without pullback (ADX={adx:.1f}, Volume={volume_surge:.2f}x)"
            ))
        
        # RULE 9: WEAK TREND STRUCTURE (low ADX + unfavorable RSI)
        if adx < 25:
            if rsi >= 65 or rsi <= 35:
                reasons.append((
                    "WEAK_TREND",
                    f"ADX={adx:.1f} < 25 with RSI={rsi:.1f} (no clear trend)"
                ))
        
        # RULE 10: OVEREXTENDED MOMENTUM (ADX very high)
        if adx >= 55:
            reasons.append((
                "OVEREXTENDED_MOMENTUM",
                f"ADX={adx:.1f} >= 55 (trend likely near exhaustion)"
            ))
        
        # RULE 11: RSI DIVERGENCE FROM TREND
        if direction == "LONG" and rsi < 45 and adx >= 30:
            reasons.append((
                "RSI_DIVERGENCE",
                f"LONG signal but RSI={rsi:.1f} < 45 (bearish divergence risk)"
            ))
        elif direction == "SHORT" and rsi > 55 and adx >= 30:
            reasons.append((
                "RSI_DIVERGENCE",
                f"SHORT signal but RSI={rsi:.1f} > 55 (bullish divergence risk)"
            ))
        
        # If no specific reasons found, it's a discretionary rejection
        if not reasons:
            reasons.append((
                "DISCRETIONARY_REJECT",
                f"No specific technical violation (ChatGPT quality filter)"
            ))
        
        return reasons


    def _log_rejection(self, signal: Dict, reasons: List[Tuple[str, str]]):
        """
        Print exact rejection log with deterministic reasons.
        """
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        rsi = signal.get("rsi", 0)
        adx = signal.get("adx", 0)
        volume_surge = signal.get("volume_surge", 0)
        entry = signal.get("entry", 0)
        sl = signal.get("sl", 0)
        tp1 = signal.get("tp1", 0)
        mode = signal.get("mode", "UNKNOWN")
        
        sl_distance = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp1_distance = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0
        
        print(f"\n{'='*75}")
        print(f"🚫 SIGNAL REJECTED BY CHATGPT")
        print(f"{'='*75}")
        print(f"Pair:      {pair}")
        print(f"Direction: {direction}")
        print(f"Mode:      {mode}")
        print(f"Entry:     {entry:,.6f}")
        print(f"SL:        {sl:,.6f} ({sl_distance:.2f}% distance)")
        print(f"TP1:       {tp1:,.6f} (R/R: {rr_ratio:.2f})")
        print(f"\n📊 Indicators:")
        print(f"   RSI:          {rsi:.1f}")
        print(f"   ADX:          {adx:.1f}")
        print(f"   Volume Surge: {volume_surge:.2f}x")
        print(f"\n🔍 EXACT REJECTION REASONS:")
        
        for i, (reason_code, detail_message) in enumerate(reasons, 1):
            print(f"   {i}. [{reason_code}]")
            print(f"      {detail_message}")
        
        print(f"{'='*75}\n")


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
            reasons = [("CHATGPT_TIMEOUT", "API call failed or timed out")]
            self._log_rejection(signal, reasons)
            return False
        
        # Parse decision
        approved = self._parse_decision(response)
        
        if approved:
            print(f"✅ {pair} APPROVED by ChatGPT - Signal will be sent")
        else:
            print(f"❌ {pair} REJECTED by ChatGPT - Signal silently dropped")
            
            # ================================
            # 🔍 INFER EXACT REJECTION REASONS
            # ================================
            reasons = self._infer_exact_rejection_reasons(signal)
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