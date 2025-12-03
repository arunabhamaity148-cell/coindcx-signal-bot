# ==========================================
# ai_layer.py — Full AI Decision Engine (30 Features)
# ==========================================

import os
import json
import time
import logging
from datetime import datetime

import aiohttp

logger = logging.getLogger("ai_layer")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# ======================================================
# MINIMAL TOKEN PROMPT (LOW COST)
# ======================================================
def build_prompt(symbol, mode, direction, score, triggers, spread, depth_ok, btc_ok):
    return (
        "You are a crypto trading AI.\n"
        "Return only JSON. No explanation.\n"
        "Keys: allow(boolean), confidence(0-100), reason(short), leverage(int).\n\n"

        f"Symbol: {symbol}\n"
        f"Mode: {mode}\n"
        f"Direction: {direction}\n"
        f"Score: {score}\n"
        f"Spread_OK: {spread}\n"
        f"Depth_OK: {depth_ok}\n"
        f"BTC_Calm: {btc_ok}\n"
        f"Triggers:\n{triggers}\n\n"

        "Rules:\n"
        "- Reject if BTC_Calm is false\n"
        "- Reject if Spread_OK is false\n"
        "- Reject if Depth_OK is false\n"
        "- Approve only if Score >= 6\n"
        "- If direction is none → reject\n"
        "- Leverage rules: score>=9 →50x, trend=20–30x, mid=20–25x, quick=15–35x\n"
        "Only JSON object as output."
    )

# ======================================================
# FALLBACK ENGINE (NO OPENAI)
# ======================================================
def fallback_decision(mode, direction, score, btc_ok, spread_ok, depth_ok):
    if not btc_ok or not spread_ok or not depth_ok:
        return {
            "allow": False,
            "confidence": 40,
            "reason": "Risk filters failed",
            "leverage": 15
        }

    if direction == "none":
        return {
            "allow": False,
            "confidence": 45,
            "reason": "No clear direction",
            "leverage": 15
        }

    lev = 15
    if score >= 9:
        lev = 50
    elif mode == "TREND":
        lev = 25
    elif mode == "MID":
        lev = 20
    else:
        lev = 18

    return {
        "allow": True,
        "confidence": min(int(score * 10), 95),
        "reason": "Local heuristic",
        "leverage": lev
    }

# ======================================================
# CALL OPENAI API (CHEAP)
# ======================================================
async def call_openai(prompt):
    if not OPENAI_API_KEY:
        return None

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 120
    }

    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, headers=headers, json=payload, timeout=12) as r:
                if r.status != 200:
                    logger.warning(f"OpenAI bad status {r.status}")
                    return None
                data = await r.json()
                txt = data["choices"][0]["message"]["content"]
                return txt
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return None

# ======================================================
# PARSE AI JSON SAFELY
# ======================================================
def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        try:
            cleaned = text[text.find("{"): text.rfind("}")+1]
            return json.loads(cleaned)
        except:
            return None

# ======================================================
# MAIN PUBLIC FUNCTION
# ======================================================
async def ai_review_signal(symbol, mode, direction, score, triggers, spread_ok, depth_ok, btc_calm):
    """
    MASTER FUNCTION — called by main.py for every signal
    Returns:
    {
        "allow": bool,
        "confidence": int,
        "reason": str,
        "leverage": int
    }
    """

    # Quick reject before API
    if direction == "none":
        return fallback_decision(mode, direction, score, btc_calm, spread_ok, depth_ok)

    if score < 5:
        return {
            "allow": False,
            "confidence": 40,
            "reason": "Score too low",
            "leverage": 15
        }

    # AI Prompt
    prompt = build_prompt(
        symbol, mode, direction, score,
        triggers, spread_ok, depth_ok, btc_calm
    )

    ai_text = await call_openai(prompt)

    if not ai_text:
        return fallback_decision(mode, direction, score, btc_calm, spread_ok, depth_ok)

    parsed = safe_parse_json(ai_text)

    if not parsed or "allow" not in parsed:
        return fallback_decision(mode, direction, score, btc_calm, spread_ok, depth_ok)

    # Extra protection rules
    if parsed["allow"] and parsed["confidence"] < 60:
        parsed["allow"] = False
        parsed["reason"] = "Low AI confidence"

    # Apply leverage override rules
    if parsed["allow"]:
        if score >= 9:
            parsed["leverage"] = 50
        elif mode == "TREND":
            parsed["leverage"] = 25
        elif mode == "MID":
            parsed["leverage"] = 20
        else:
            parsed["leverage"] = 18

    return parsed

# ======================================================
# END
# ======================================================