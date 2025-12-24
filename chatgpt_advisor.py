from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from config import config
import time
import json
import pandas as pd


class ChatGPTAdvisor:
    """
    CHATGPT SNIPER FILTER + 10 ADVANCED MODULES - FINAL DECISION LAYER
    -------------------------------------------------------------------
    PHILOSOPHY: Discretionary Human Trader + Data-Driven Precision
    
    HARD REJECT CRITERIA (Non-negotiable):
    1. RR < 1.5
    2. SL < 0.8%
    3. TRUE exhaustion: (RSI > 78 OR < 22) AND ADX > 50
    4. Invalid price data
    5. BTC strong impulse AGAINST signal direction
    
    NEW ADVANCED FILTERS (Score-based, not blockers):
    6. Order Flow Imbalance
    7. Market Structure Break
    8. Session Liquidity
    9. Correlation Filter
    10. Volume Profile
    11. Volatility Regime
    12. Macro Sentiment
    13. Price Action Confluence
    14. Time Decay Filter
    15. Drawdown Protector
    
    OUTPUT: {"approved": true/false, "quality_score": 0-100}
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.CHATGPT_API_KEY)
        self.model = config.CHATGPT_MODEL
        self.timeout = 8
        self.max_retries = 2
        self.recent_losses = []

    def _call_chatgpt_with_timeout(self, messages: List[Dict]) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                return (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1200,
                        temperature=0.2,
                        timeout=self.timeout,
                    )
                    .choices[0]
                    .message.content.strip()
                )
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return None

    def _parse_decision(self, response: str) -> bool:
        if not response:
            return False
        try:
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return bool(data["approved"])
        except Exception as e:
            print(f"⚠️ ChatGPT parse error: {e}")
            return False

    def _check_order_flow_imbalance(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Module 1: Order Flow Imbalance"""
        try:
            if candles is None or candles.empty:
                return {"score": 50, "status": "NO_DATA"}

            candles = candles.tail(20)
            buy_volume = 0
            sell_volume = 0

            for idx, row in candles.iterrows():
                close_price = float(row['close'])
                open_price = float(row['open'])
                volume = float(row['volume'])

                if close_price > open_price:
                    buy_volume += volume
                else:
                    sell_volume += volume

            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return {"score": 50, "status": "LOW_VOLUME"}

            buy_pressure = buy_volume / total_volume
            sell_pressure = sell_volume / total_volume
            direction = signal.get('direction', 'UNKNOWN')

            if direction == "LONG":
                if buy_pressure > 0.65:
                    return {"score": 100, "status": "STRONG_BUY_PRESSURE", "buy_ratio": round(buy_pressure, 2)}
                elif buy_pressure > 0.55:
                    return {"score": 75, "status": "MODERATE_BUY", "buy_ratio": round(buy_pressure, 2)}
                else:
                    return {"score": 40, "status": "WEAK_BUY", "buy_ratio": round(buy_pressure, 2)}
            else:
                if sell_pressure > 0.65:
                    return {"score": 100, "status": "STRONG_SELL_PRESSURE", "sell_ratio": round(sell_pressure, 2)}
                elif sell_pressure > 0.55:
                    return {"score": 75, "status": "MODERATE_SELL", "sell_ratio": round(sell_pressure, 2)}
                else:
                    return {"score": 40, "status": "WEAK_SELL", "sell_ratio": round(sell_pressure, 2)}

        except Exception as e:
            print(f"⚠️ Order flow check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_market_structure(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Module 2: Market Structure Break"""
        try:
            if candles is None or candles.empty or len(candles) < 10:
                return {"score": 50, "status": "INSUFFICIENT_DATA"}

            candles = candles.tail(10)
            highs = candles['high'].values
            lows = candles['low'].values
            direction = signal.get('direction', 'UNKNOWN')

            recent_high = max(highs[-5:])
            prev_high = max(highs[-10:-5])
            recent_low = min(lows[-5:])
            prev_low = min(lows[-10:-5])

            if direction == "LONG":
                higher_high = recent_high > prev_high
                higher_low = recent_low > prev_low

                if higher_high and higher_low:
                    return {"score": 100, "status": "BULLISH_STRUCTURE_INTACT", "pattern": "HH_HL"}
                elif higher_high:
                    return {"score": 70, "status": "PARTIAL_BULLISH", "pattern": "HH_ONLY"}
                else:
                    return {"score": 40, "status": "WEAK_STRUCTURE", "pattern": "NO_HH"}
            else:
                lower_low = recent_low < prev_low
                lower_high = recent_high < prev_high

                if lower_low and lower_high:
                    return {"score": 100, "status": "BEARISH_STRUCTURE_INTACT", "pattern": "LL_LH"}
                elif lower_low:
                    return {"score": 70, "status": "PARTIAL_BEARISH", "pattern": "LL_ONLY"}
                else:
                    return {"score": 40, "status": "WEAK_STRUCTURE", "pattern": "NO_LL"}

        except Exception as e:
            print(f"⚠️ Market structure check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_session_liquidity(self) -> Dict:
        """Module 3: Session Liquidity"""
        try:
            from datetime import datetime
            now = datetime.now()
            hour = now.hour

            if 18 <= hour < 22:
                return {"score": 100, "status": "NY_SESSION", "liquidity": "VERY_HIGH"}
            elif 13 <= hour < 18:
                return {"score": 90, "status": "LONDON_SESSION", "liquidity": "HIGH"}
            elif 5 <= hour < 10:
                return {"score": 50, "status": "ASIAN_SESSION", "liquidity": "LOW"}
            else:
                return {"score": 30, "status": "OFF_HOURS", "liquidity": "VERY_LOW"}

        except Exception as e:
            print(f"⚠️ Session check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_correlation_filter(self, signal: Dict) -> Dict:
        """Module 4: Correlation Filter"""
        try:
            from coindcx_api import CoinDCXAPI

            direction = signal.get('direction', 'UNKNOWN')
            pair = signal.get('pair', 'UNKNOWN')
            check_pairs = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT']

            if pair in check_pairs:
                check_pairs.remove(pair)

            agreement_count = 0
            total_checked = 0

            for check_pair in check_pairs[:3]:
                try:
                    candles = CoinDCXAPI.get_candles(check_pair, '5m', 10)
                    if candles.empty:
                        continue

                    latest = candles.iloc[-1]
                    close_price = float(latest['close'])
                    open_price = float(latest['open'])
                    is_bullish = close_price > open_price

                    if direction == "LONG" and is_bullish:
                        agreement_count += 1
                    elif direction == "SHORT" and not is_bullish:
                        agreement_count += 1

                    total_checked += 1
                except:
                    continue

            if total_checked == 0:
                return {"score": 50, "status": "NO_DATA"}

            agreement_ratio = agreement_count / total_checked

            if agreement_ratio >= 0.75:
                return {"score": 100, "status": "STRONG_CORRELATION", "ratio": round(agreement_ratio, 2)}
            elif agreement_ratio >= 0.5:
                return {"score": 70, "status": "MODERATE_CORRELATION", "ratio": round(agreement_ratio, 2)}
            else:
                return {"score": 40, "status": "WEAK_CORRELATION", "ratio": round(agreement_ratio, 2)}

        except Exception as e:
            print(f"⚠️ Correlation check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_volume_profile(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Module 5: Volume Profile"""
        try:
            if candles is None or candles.empty or len(candles) < 20:
                return {"score": 50, "status": "INSUFFICIENT_DATA"}

            candles = candles.tail(50)
            entry = signal.get('entry', 0)
            price_volume = {}

            for idx, row in candles.iterrows():
                price = round(float(row['close']), 4)
                volume = float(row['volume'])
                price_volume[price] = price_volume.get(price, 0) + volume

            if not price_volume:
                return {"score": 50, "status": "NO_VOLUME_DATA"}

            poc_price = max(price_volume, key=price_volume.get)
            distance_pct = abs(entry - poc_price) / entry * 100

            if distance_pct < 0.5:
                return {"score": 100, "status": "AT_POC", "distance": f"{distance_pct:.2f}%"}
            elif distance_pct < 1.0:
                return {"score": 80, "status": "NEAR_POC", "distance": f"{distance_pct:.2f}%"}
            elif distance_pct < 2.0:
                return {"score": 60, "status": "MODERATE_DISTANCE", "distance": f"{distance_pct:.2f}%"}
            else:
                return {"score": 40, "status": "FAR_FROM_POC", "distance": f"{distance_pct:.2f}%"}

        except Exception as e:
            print(f"⚠️ Volume profile check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_volatility_regime(self, candles: pd.DataFrame) -> Dict:
        """Module 6: Volatility Regime"""
        try:
            if candles is None or candles.empty or len(candles) < 20:
                return {"score": 50, "status": "INSUFFICIENT_DATA"}

            from indicators import Indicators

            candles = candles.tail(20)
            atr = Indicators.atr(candles['high'], candles['low'], candles['close'])

            if atr.empty:
                return {"score": 50, "status": "NO_ATR_DATA"}

            current_atr = float(atr.iloc[-1])
            avg_atr = float(atr.mean())

            if avg_atr == 0:
                return {"score": 50, "status": "ZERO_ATR"}

            atr_ratio = current_atr / avg_atr

            if 0.9 <= atr_ratio <= 1.3:
                return {"score": 100, "status": "NORMAL_VOLATILITY", "atr_ratio": round(atr_ratio, 2)}
            elif 0.7 <= atr_ratio <= 1.6:
                return {"score": 70, "status": "ACCEPTABLE_VOLATILITY", "atr_ratio": round(atr_ratio, 2)}
            elif atr_ratio > 2.0:
                return {"score": 30, "status": "EXTREME_VOLATILITY", "atr_ratio": round(atr_ratio, 2)}
            else:
                return {"score": 40, "status": "LOW_VOLATILITY", "atr_ratio": round(atr_ratio, 2)}

        except Exception as e:
            print(f"⚠️ Volatility check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_macro_sentiment(self, signal: Dict) -> Dict:
        """Module 7: Macro Sentiment"""
        try:
            return {"score": 70, "status": "NEUTRAL_SENTIMENT", "note": "Real-time data not available"}
        except Exception as e:
            print(f"⚠️ Macro sentiment check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_price_action_confluence(self, signal: Dict, candles: pd.DataFrame) -> Dict:
        """Module 8: Price Action Confluence"""
        try:
            if candles is None or candles.empty or len(candles) < 3:
                return {"score": 50, "status": "INSUFFICIENT_DATA"}

            latest = candles.iloc[-1]
            open_price = float(latest['open'])
            close_price = float(latest['close'])
            high_price = float(latest['high'])
            low_price = float(latest['low'])

            body_size = abs(close_price - open_price)
            total_range = high_price - low_price

            if total_range == 0:
                return {"score": 50, "status": "DOJI"}

            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            direction = signal.get('direction', 'UNKNOWN')

            if direction == "LONG":
                if lower_wick > body_size * 2 and close_price > open_price:
                    return {"score": 100, "status": "BULLISH_REJECTION", "pattern": "HAMMER"}
                elif close_price > open_price and body_size > total_range * 0.6:
                    return {"score": 90, "status": "STRONG_BULLISH", "pattern": "MARUBOZU"}
                else:
                    return {"score": 60, "status": "WEAK_PATTERN"}
            else:
                if upper_wick > body_size * 2 and close_price < open_price:
                    return {"score": 100, "status": "BEARISH_REJECTION", "pattern": "SHOOTING_STAR"}
                elif close_price < open_price and body_size > total_range * 0.6:
                    return {"score": 90, "status": "STRONG_BEARISH", "pattern": "MARUBOZU"}
                else:
                    return {"score": 60, "status": "WEAK_PATTERN"}

        except Exception as e:
            print(f"⚠️ Price action check failed: {e}")
            return {"score": 50, "status": "ERROR"}

    def _check_time_decay(self, signal: Dict) -> Dict:
        """Module 9: Time Decay Filter"""
        try:
            from datetime import datetime

            signal_time_str = signal.get('timestamp', '')
            if not signal_time_str:
                return {"score": 50, "status": "NO_TIMESTAMP"}

            signal_time = datetime.strptime(signal_time_str, '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            age_minutes = (now - signal_time).total_seconds() / 60

            if age_minutes < 2:
                return {"score": 100, "status": "FRESH", "age": f"{age_minutes:.1f}m"}
            elif age_minutes < 5:
                return {"score": 80, "status": "RECENT", "age": f"{age_minutes:.1f}m"}
            elif age_minutes < 10:
                return {"score": 50, "status": "AGING", "age": f"{age_minutes:.1f}m"}
            else:
                return {"score": 20, "status": "STALE", "age": f"{age_minutes:.1f}m"}

        except Exception as e:
            print(f"⚠️ Time decay check failed: {e}")
            return {"score": 70, "status": "ERROR"}

    def _check_drawdown_protection(self) -> Dict:
        """Module 10: Drawdown Protector"""
        try:
            from datetime import datetime, timedelta

            two_hours_ago = datetime.now() - timedelta(hours=2)
            self.recent_losses = [loss for loss in self.recent_losses if loss > two_hours_ago]
            loss_count = len(self.recent_losses)

            if loss_count >= 3:
                return {"score": 0, "status": "DRAWDOWN_PAUSE", "recent_losses": loss_count}
            elif loss_count == 2:
                return {"score": 50, "status": "CAUTION", "recent_losses": loss_count}
            else:
                return {"score": 100, "status": "SAFE", "recent_losses": loss_count}

        except Exception as e:
            print(f"⚠️ Drawdown check failed: {e}")
            return {"score": 70, "status": "ERROR"}

    def record_loss(self):
        """Record a losing trade"""
        from datetime import datetime
        self.recent_losses.append(datetime.now())

    def calculate_advanced_quality_score(self, signal: Dict, candles: pd.DataFrame = None) -> Dict:
        """Calculate comprehensive quality score using all 10 advanced modules"""
        print(f"\n{'='*70}")
        print(f"🔬 ADVANCED QUALITY ANALYSIS: {signal.get('pair')} {signal.get('direction')}")
        print(f"{'='*70}")

        results = {}

        results['order_flow'] = self._check_order_flow_imbalance(signal, candles)
        print(f"1️⃣  Order Flow: {results['order_flow']['score']}/100 - {results['order_flow']['status']}")

        results['market_structure'] = self._check_market_structure(signal, candles)
        print(f"2️⃣  Market Structure: {results['market_structure']['score']}/100 - {results['market_structure']['status']}")

        results['session'] = self._check_session_liquidity()
        print(f"3️⃣  Session: {results['session']['score']}/100 - {results['session']['status']}")

        results['correlation'] = self._check_correlation_filter(signal)
        print(f"4️⃣  Correlation: {results['correlation']['score']}/100 - {results['correlation']['status']}")

        results['volume_profile'] = self._check_volume_profile(signal, candles)
        print(f"5️⃣  Volume Profile: {results['volume_profile']['score']}/100 - {results['volume_profile']['status']}")

        results['volatility'] = self._check_volatility_regime(candles)
        print(f"6️⃣  Volatility: {results['volatility']['score']}/100 - {results['volatility']['status']}")

        results['sentiment'] = self._check_macro_sentiment(signal)
        print(f"7️⃣  Sentiment: {results['sentiment']['score']}/100 - {results['sentiment']['status']}")

        results['price_action'] = self._check_price_action_confluence(signal, candles)
        print(f"8️⃣  Price Action: {results['price_action']['score']}/100 - {results['price_action']['status']}")

        results['time_decay'] = self._check_time_decay(signal)
        print(f"9️⃣  Time Decay: {results['time_decay']['score']}/100 - {results['time_decay']['status']}")

        results['drawdown'] = self._check_drawdown_protection()
        print(f"🔟 Drawdown: {results['drawdown']['score']}/100 - {results['drawdown']['status']}")

        weights = {
            'order_flow': 0.12,
            'market_structure': 0.15,
            'session': 0.08,
            'correlation': 0.10,
            'volume_profile': 0.10,
            'volatility': 0.10,
            'sentiment': 0.08,
            'price_action': 0.12,
            'time_decay': 0.05,
            'drawdown': 0.10
        }

        total_score = sum(results[key]['score'] * weights[key] for key in weights)

        print(f"{'='*70}")
        print(f"📊 TOTAL ADVANCED SCORE: {total_score:.1f}/100")
        print(f"{'='*70}\n")

        return {
            'total_score': round(total_score, 1),
            'results': results
        }

    def final_trade_decision(self, signal: Dict, candles: pd.DataFrame) -> bool:
        """
        FINAL TRADE DECISION - Called by SignalGenerator
        
        This is the MANDATORY method that signal_generator.py expects.
        Returns True to approve signal, False to reject.
        """
        try:
            print(f"\n{'='*70}")
            print(f"🤖 CHATGPT FINAL JUDGE: {signal.get('pair')} {signal.get('direction')}")
            print(f"{'='*70}")

            # Run advanced quality analysis
            quality_analysis = self.calculate_advanced_quality_score(signal, candles)
            
            # Safe fallback if analysis fails
            if not quality_analysis or 'total_score' not in quality_analysis:
                print("⚠ ChatGPT error")