from openai import OpenAI
from typing import Dict, List, Optional
from config import config
import time
import json
import pandas as pd


class ChatGPTAdvisor:
    """
    CHATGPT SNIPER FILTER + ADVANCED QUALITY LAYER
    FINAL DISCRETIONARY DECISION ENGINE
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2
        self.recent_losses = []

    # ------------------------------------------------------------------
    # SAFE CHATGPT CALL (non-blocking)
    # ------------------------------------------------------------------
    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.2,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ ChatGPT call failed ({attempt+1}): {e}")
                time.sleep(1)
        return None

    # ------------------------------------------------------------------
    # MODULE 1: SESSION LIQUIDITY  ✅ FIXED
    # ------------------------------------------------------------------
    def _check_session_liquidity(self) -> Dict:
        try:
            from datetime import datetime
            hour = datetime.now().hour

            if 18 <= hour < 22:
                return {"score": 100, "status": "NY_SESSION"}
            elif 13 <= hour < 18:
                return {"score": 90, "status": "LONDON_SESSION"}
            elif 5 <= hour < 10:
                return {"score": 50, "status": "ASIAN_SESSION"}
            else:
                return {"score": 30, "status": "OFF_HOURS"}

        except Exception as e:
            print(f"⚠️ Session check error: {e}")
            return {"score": 50, "status": "ERROR"}

    # ------------------------------------------------------------------
    # MODULE 2: VOLATILITY REGIME  ✅ FIXED INDENT
    # ------------------------------------------------------------------
    def _check_volatility_regime(self, candles: pd.DataFrame) -> Dict:
        try:
            if candles is None or candles.empty or len(candles) < 20:
                return {"score": 50, "status": "NO_DATA"}

            from indicators import Indicators

            atr = Indicators.atr(candles['high'], candles['low'], candles['close'])
            if atr.empty:
                return {"score": 50, "status": "NO_ATR"}

            current_atr = float(atr.iloc[-1])
            avg_atr = float(atr.mean())

            if avg_atr == 0:
                return {"score": 50, "status": "ZERO_ATR"}

            ratio = current_atr / avg_atr

            if 0.9 <= ratio <= 1.3:
                return {"score": 100, "status": "NORMAL_VOL"}
            elif 0.7 <= ratio <= 1.6:
                return {"score": 70, "status": "ACCEPTABLE_VOL"}
            elif ratio > 2.0:
                return {"score": 30, "status": "EXTREME_VOL"}
            else:
                return {"score": 40, "status": "LOW_VOL"}

        except Exception as e:
            print(f"⚠️ Volatility check error: {e}")
            return {"score": 50, "status": "ERROR"}

    # ------------------------------------------------------------------
    # MODULE 3: TIME DECAY
    # ------------------------------------------------------------------
    def _check_time_decay(self, signal: Dict) -> Dict:
        try:
            from datetime import datetime
            ts = signal.get("timestamp")
            if not ts:
                return {"score": 50}

            signal_time = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            age = (datetime.now() - signal_time).total_seconds() / 60

            if age < 2:
                return {"score": 100}
            elif age < 5:
                return {"score": 80}
            elif age < 10:
                return {"score": 50}
            else:
                return {"score": 20}

        except Exception:
            return {"score": 70}

    # ------------------------------------------------------------------
    # MODULE 4: DRAWDOWN PROTECTOR
    # ------------------------------------------------------------------
    def _check_drawdown(self) -> Dict:
        try:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(hours=2)
            self.recent_losses = [t for t in self.recent_losses if t > cutoff]

            if len(self.recent_losses) >= 3:
                return {"score": 0}
            elif len(self.recent_losses) == 2:
                return {"score": 50}
            else:
                return {"score": 100}

        except Exception:
            return {"score": 70}

    # ------------------------------------------------------------------
    # ADVANCED QUALITY SCORE (SOFT ONLY)
    # ------------------------------------------------------------------
    def calculate_quality_score(self, signal: Dict, candles: pd.DataFrame) -> float:
        scores = []

        scores.append(self._check_session_liquidity()['score'])
        scores.append(self._check_volatility_regime(candles)['score'])
        scores.append(self._check_time_decay(signal)['score'])
        scores.append(self._check_drawdown()['score'])

        return round(sum(scores) / len(scores), 1)

    # ------------------------------------------------------------------
    # FINAL DECISION (CALLED BY SignalGenerator)
    # ------------------------------------------------------------------
    def final_trade_decision(self, signal: Dict, candles: pd.DataFrame) -> bool:
        try:
            print(f"\n🤖 CHATGPT SNIPER: {signal['pair']} {signal['direction']}")

            quality = self.calculate_quality_score(signal, candles)
            MIN_SCORE = 60  # lenient

            if quality >= MIN_SCORE:
                print(f"✅ APPROVED | Quality={quality}")
                return True
            else:
                print(f"❌ REJECTED | Quality={quality}")
                return False

        except Exception as e:
            print(f"⚠️ Advisor error: {e} → AUTO APPROVE")
            return True

    # ------------------------------------------------------------------
    # LOSS RECORD
    # ------------------------------------------------------------------
    def record_loss(self):
        from datetime import datetime
        self.recent_losses.append(datetime.now())