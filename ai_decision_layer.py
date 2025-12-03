import os
import aiohttp
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

async def ai_review_signal(symbol, mode, direction, score, triggers, price):
    # Fallback heuristic if API key missing
    if not OPENAI_API_KEY:
        if score >= 8.5:
            return {"allow": True, "confidence": 92, "reason": "Strong internal score"}
        if score >= 6:
            return {"allow": True, "confidence": 80, "reason": "Medium score, allowed"}
        return {"allow": False, "confidence": 50, "reason": "Weak score"}

    prompt = (
        f"Signal review:\n"
        f"Symbol: {symbol}\nMode: {mode}\nDirection: {direction}\n"
        f"Score: {score}\nPrice: {price}\nTriggers: {triggers}\n"
        f"Reply JSON with: allow, confidence, reason."
    )

    try:
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.0
                }
            )
            data = await r.json()
            text = data["choices"][0]["message"]["content"]

            # extract JSON
            idx = text.find("{")
            if idx >= 0:
                raw = text[idx:]
                return json.loads(raw)

    except Exception:
        pass

    # fallback
    return {"allow": True, "confidence": 80, "reason": "Fallback allow"}