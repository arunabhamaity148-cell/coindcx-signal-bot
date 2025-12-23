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
            return False
        
        # Parse decision
        approved = self._parse_decision(response)
        
        if approved:
            print(f"✅ {pair} APPROVED by ChatGPT - Signal will be sent")
        else:
            print(f"❌ {pair} REJECTED by ChatGPT - Signal silently dropped")
        
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