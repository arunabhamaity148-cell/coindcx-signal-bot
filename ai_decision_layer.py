# ==============================================
# ai_decision_layer.py
# ARUN HYBRID TRADER GPT V2 — Decision Layer
# ==============================================

import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
logger = logging.getLogger("decision_layer_v2")


async def decision_layer_v2(signal_payload: dict):
    """
    The bot sends raw signal data here.
    ChatGPT V2 analyzes the data and returns a JSON decision.
    """

    prompt = f"""
You are ARUN HYBRID TRADER GPT V2, a safe self-improving trading AI.

Rules (apply strictly):
- Reject if btc_calm is false.
- Reject if spread > 0.10 (percentage).
- Reject if depth < 3000 (USD).
- Reject if sl_distance_pct < 5 (percent).
- Reject if score < 5.0.
- Determine side from triggers (bullish vs bearish words).
- Leverage rules:
    * score >= 9.0 and market stable -> 50
    * score >= 9.0 and market volatile -> max 30
    * score 7-8 -> 20-25
    * score 5-6 -> 15-20
- Adjust TP/SL if market regime suggests (return modifiers).
- Suggest small weight adjustments for learning.

Return ONLY valid JSON with EXACT keys:
{
  "decision": "APPROVE" | "MODIFY" | "REJECT" | "DELAY",
  "side": "LONG" | "SHORT",
  "confidence": 0-100,
  "recommended_leverage": 15-50,
  "tp_modifier": 1.0,
  "sl_modifier": 1.0,
  "adjusted_weights": { "name": 0.1, ... },
  "reason": "short explanation"
}

Here is the raw payload:
{json.dumps(signal_payload, indent=2)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        content = resp.choices[0].message.content.strip()
        # Parse JSON safely
        decision = json.loads(content)
        return decision

    except Exception as e:
        logger.error(f"Decision Layer Error: {e}")
        # On error, be safe and reject
        return {
            "decision": "REJECT",
            "side": None,
            "confidence": 0,
            "recommended_leverage": 15,
            "tp_modifier": 1.0,
            "sl_modifier": 1.0,
            "adjusted_weights": {},
            "reason": f"Decision error: {e}"
        }