# helpers_openai.py
# OpenAI verifier for Hybrid AI Signals (async, fail-open)

import os
import aiohttp
import json

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "10"))

async def call_openai_decision(symbol, mode, price, score, live_data_summary):
    """
    Sends a concise prompt to OpenAI and expects a JSON response with:
    { "approve": true|false, "tp": <float|null>, "sl": <float|null>, "note": "..." }
    Fail-open: if no key / error / parse fails -> return approve=True (local decision allowed).
    """
    # No key -> fail-open immediately
    if not OPENAI_KEY:
        return {"approve": True, "tp": None, "sl": None, "note": "no_openai_key_fail_open"}

    prompt = f"""
You are a conservative crypto signal verifier. Respond ONLY with a single JSON object, nothing else.
Input:
symbol: {symbol}
mode: {mode}
price: {price}
score: {score}
live_summary: {json.dumps(live_data_summary)}

Return JSON with keys:
- approve (boolean)
- tp (number|null)
- sl (number|null)
- note (string, short)
Example:
{{"approve": true, "tp": null, "sl": null, "note": "ok"}}
"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a signal risk manager. Be concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body, timeout=OPENAI_TIMEOUT) as resp:
                if resp.status != 200:
                    return {"approve": True, "tp": None, "sl": None, "note": f"openai_status_{resp.status}"}
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                # try to load JSON from response
                try:
                    j = json.loads(content)
                    return {
                        "approve": bool(j.get("approve", False)),
                        "tp": float(j["tp"]) if j.get("tp") not in (None, "", False) else None,
                        "sl": float(j["sl"]) if j.get("sl") not in (None, "", False) else None,
                        "note": str(j.get("note", ""))[:200]
                    }
                except Exception:
                    # parsing failed -> fail-open
                    return {"approve": True, "tp": None, "sl": None, "note": "openai_parse_fail"}
    except Exception:
        # network / timeout -> fail-open
        return {"approve": True, "tp": None, "sl": None, "note": "openai_error"}