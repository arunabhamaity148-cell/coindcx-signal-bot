from openai import OpenAI
from typing import Dict, List, Optional
from config import config
import time
import json


class ChatGPTAdvisor:
    """
    CHATGPT AS FINAL TRADE JUDGE
    -----------------------------
    Every signal MUST pass ChatGPT validation before sending.
    ChatGPT acts as an experienced discretionary trader.

    OUTPUT: Strict JSON only
    {"approved": true/false}
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2


    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
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
        if not response:
            return False

        try:
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)

            if "approved" not in data:
                return False

            return bool(data["approved"])

        except Exception:
            return False


    def _infer_reject_reason(
        self,
        rsi: float,
        adx: float,
        volume_surge: float,
        rr_ratio: float,
        sl_distance: float
    ) -> str:
        """
        Infer rejection reason locally (LOG ONLY)
        """
        reasons = []

        if volume_surge < 1.2:
            reasons.append("LOW_VOLUME")

        if adx > 40 and (rsi > 65 or rsi < 35):
            reasons.append("LATE_ENTRY")

        if adx > 45 and (rsi > 70 or rsi < 30):
            reasons.append("EXHAUSTED_MOMENTUM")

        if rr_ratio < 1.5:
            reasons.append("POOR_RR")

        if sl_distance < 1.0:
            reasons.append("TIGHT_SL")

        if not reasons:
            return "DISCRETIONARY_REJECT"

        return ", ".join(reasons)


    def final_trade_decision(self, signal: Dict, candles_data: Dict = None) -> bool:
        pair = signal.get("pair", "UNKNOWN")
        direction = signal.get("direction", "UNKNOWN")
        entry = float(signal.get("entry", 0))
        sl = float(signal.get("sl", 0))
        tp1 = float(signal.get("tp1", 0))
        rsi = float(signal.get("rsi", 50))
        adx = float(signal.get("adx", 20))
        volume_surge = float(signal.get("volume_surge", 1.0))
        score = int(signal.get("score", 0))
        mode = signal.get("mode", "UNKNOWN")

        sl_distance = abs(entry - sl) / entry * 100
        tp1_distance = abs(tp1 - entry) / entry * 100
        rr_ratio = tp1_distance / sl_distance if sl_distance > 0 else 0

        prompt = f"""
You are an experienced cryptocurrency futures trader.

SIGNAL:
Pair: {pair}
Direction: {direction}
Entry: {entry}
SL: {sl}
TP1: {tp1}
RSI: {rsi}
ADX: {adx}
Volume: {volume_surge}
RR: {rr_ratio:.2f}
Mode: {mode}

RESPOND STRICT JSON ONLY:
{{"approved": true}} OR {{"approved": false}}
"""

        messages = [
            {
                "role": "system",
                "content": "Respond ONLY with JSON {\"approved\": true/false}. No explanation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        print(f"🤖 Consulting ChatGPT for {pair} {direction}...")
        response = self._call_chatgpt_with_timeout(messages)

        if response is None:
            print(f"❌ {pair} REJECTED | REASON: CHATGPT_TIMEOUT")
            return False

        approved = self._parse_decision(response)

        if approved:
            print(f"✅ {pair} APPROVED by ChatGPT")
            return True

        # 🔴 LOG REJECT REASON (ONLY CONSOLE)
        reason = self._infer_reject_reason(
            rsi=rsi,
            adx=adx,
            volume_surge=volume_surge,
            rr_ratio=rr_ratio,
            sl_distance=sl_distance
        )

        print(
            f"❌ {pair} REJECTED by ChatGPT | "
            f"REASON: {reason} | "
            f"RSI={rsi:.1f}, ADX={adx:.1f}, VOL={volume_surge}x, RR={rr_ratio:.2f}"
        )

        return False


    def validate_signal_with_traps(self, signal: Dict) -> Dict:
        approved = self.final_trade_decision(signal)

        return {
            "approved": approved,
            "reason": "ChatGPT final decision",
            "confidence": 100 if approved else 0
        }