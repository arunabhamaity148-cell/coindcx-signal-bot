"""
UNIQUE & SMART CoinDCX Signal Bot
- 10 Proprietary Indicators
- 15-25 Quality Signals Daily
- Ticker-based (Real CoinDCX Data)
- Not too strict, but intelligent
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
from collections import deque
import numpy as np

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    # API Keys
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Coins (Popular + Liquid)
    COINS_TO_MONITOR = [
        'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'DOGE', 'MATIC',
        'DOT', 'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'FTM',
        'AAVE', 'GRT', 'ALGO', 'TRX'
    ]
    
    # BALANCED SETTINGS (15-25 signals/day)
    MIN_SCORE = 55  # Not too strict (was 70)
    SCAN_INTERVAL = 40  # 40 seconds
    PRICE_HISTORY_SIZE = 50
    MIN_DATA_POINTS = 25  # Reasonable (not 40)
    
config = Config()

class SmartPriceTracker:
    """Track price history with validation"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = {}
        self.last_update = {}
    
    def add_price(self, market: str, price: float, volume: float):
        # Validate price change
        if market in self.data and len(self.data[market]) > 0:
            last_price = self.data[market][-1]['price']
            change = abs(price - last_price) / last_price
            
            if change > 0.15:  # Skip if >15% jump (bad data)
                return
        
        if market not in self.data:
            self.data[market] = deque(maxlen=self.max_size)
        
        self.data[market].append({
            'timestamp': datetime.now(),
            'price': price,
            'volume': volume
        })
        
        self.last_update[market] = datetime.now()
    
    def get_history(self, market: str) -> List[Dict]:
        return list(self.data.get(market, []))
    
    def has_enough_data(self, market: str, min_points: int) -> bool:
        if market not in self.data:
            return False
        
        if len(self.data[market]) < min_points:
            return False
        
        # Check freshness (3 min)
        if market in self.last_update:
            age = (datetime.now() - self.last_update[market]).seconds
            if age > 180:
                return False
        
        return True

class UniqueAnalyzer:
    """10 Unique Proprietary Indicators"""
    
    @staticmethod
    def momentum_wave(history: List[Dict]) -> Tuple[str, int]:
        """1. Momentum Wave Detection - Velocity + Acceleration"""
        if len(history) < 10:
            return "DORMANT", 0
        
        prices = [h['price'] for h in history]
        # Velocity (rate of change)
        velocity = (prices[-1] - prices[-5]) / prices[-5]
        
        # Acceleration (change in velocity)
        prev_velocity = (prices[-5] - prices[-10]) / prices[-10]
        acceleration = velocity - prev_velocity
        
        if velocity > 0.01 and acceleration > 0:
            return "ACCELERATING_UP", 12
        elif velocity < -0.01 and acceleration < 0:
            return "ACCELERATING_DOWN", 12
        elif abs(velocity) > 0.005:
            return "MOVING", 7
        else:
            return "DORMANT", 0
    
    @staticmethod
    def smart_money_index(history: List[Dict]) -> Tuple[str, int]:
        """2. Smart Money Tracker - Volume + Price Action"""
        if len(history) < 15:
            return "NEUTRAL", 0
        
        volumes = [h['volume'] for h in history]
        prices = [h['price'] for h in history]
        
        # Recent volume surge
        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes[-15:])
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Price movement
        price_change = (prices[-1] - prices[-5]) / prices[-5]
        
        if vol_surge > 1.5 and price_change > 0.008:
            return "SMART_BUY", 10
        elif vol_surge > 1.5 and price_change < -0.008:
            return "SMART_SELL", 10
        elif vol_surge > 1.3:
            return "ACTIVITY", 5
        else:
            return "NEUTRAL", 0
    
    @staticmethod
    def market_sync_index(history: List[Dict]) -> Tuple[str, int]:
        """3. Market Synchrony - Price + Volume alignment"""
        if len(history) < 15:
            return "DESYNC", 0
        
        prices = [h['price'] for h in history[-10:]]
        volumes = [h['volume'] for h in history]
        
        price_up = prices[-1] > prices[0]
        
        recent_vol = np.mean(volumes[-5:])
        past_vol = np.mean(volumes[-15:-5])
        vol_up = recent_vol > past_vol
        
        if price_up and vol_up:
            return "BULLISH_SYNC", 10
        elif not price_up and vol_up:
            return "BEARISH_SYNC", 10
        else:
            return "DESYNC", 0
    
    @staticmethod
    def adaptive_volatility(history: List[Dict]) -> Tuple[float, int]:
        """4. Adaptive Volatility - Optimal trading range"""
        if len(history) < 20:
            return 0, 0
        
        prices = [h['price'] for h in history[-20:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(returns)
        
        # Optimal range: 0.5% - 3%
        if 0.005 <= volatility <= 0.03:
            return volatility, 10
        elif 0.003 <= volatility <= 0.04:
            return volatility, 5
        else:
            return volatility, 0
    
    @staticmethod
    def trend_persistence(history: List[Dict]) -> Tuple[str, int]:
        """5. Trend Persistence - How long trend continues"""
        if len(history) < 15:
            return "NONE", 0
        
        prices = [h['price'] for h in history[-15:]]
        
        # Count consecutive ups/downs
        ups = 0
        downs = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                ups += 1
                downs = 0
            else:
                downs += 1
                ups = 0
        
        if ups >= 5:
            return "STRONG_UP", 8
        elif downs >= 5:
            return "STRONG_DOWN", 8
        elif ups >= 3:
            return "UP", 5
        elif downs >= 3:
            return "DOWN", 5
        else:
            return "NONE", 0
    
    @staticmethod
    def momentum_divergence(history: List[Dict]) -> Tuple[bool, str, int]:
        """6. Momentum Divergence - Price vs Momentum mismatch"""
        if len(history) < 20:
            return False, "NONE", 0
        
        prices = [h['price'] for h in history]
        
        # Price trend
        price_trend = prices[-1] - prices[-10]
        
        # Momentum (recent vs older)
        recent_mom = (prices[-1] - prices[-5]) / prices[-5]
        older_mom = (prices[-10] - prices[-15]) / prices[-15]
        
        # Divergence
        if price_trend > 0 and recent_mom < older_mom:
            return True, "BEARISH_DIV", 8
        elif price_trend < 0 and recent_mom > older_mom:
            return True, "BULLISH_DIV", 8
        else:
            return False, "NONE", 0
    
    @staticmethod
    def volume_profile(history: List[Dict]) -> Tuple[str, int]:
        """7. Volume Profile - Distribution analysis"""
        if len(history) < 20:
            return "UNKNOWN", 0
        
        volumes = [h['volume'] for h in history[-20:]]
        
        # Check if volume increasing
        first_half = np.mean(volumes[:10])
        second_half = np.mean(volumes[10:])
        
        if second_half > first_half * 1.4:
            return "INCREASING", 8
        elif second_half < first_half * 0.7:
            return "DECREASING", 3
        else:
            return "STABLE", 5
    
    @staticmethod
    def price_elasticity(history: List[Dict]) -> Tuple[float, int]:
        """8. Price Elasticity - How responsive to changes"""
        if len(history) < 15:
            return 0, 0
        
        prices = [h['price'] for h in history[-15:]]
        
        # Calculate price changes
        changes = [abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
        avg_change = np.mean(changes)
        
        # Elastic = responsive
        if 0.005 <= avg_change <= 0.02:
            return avg_change, 7
        else:
            return avg_change, 0
    
    @staticmethod
    def sentiment_momentum(history: List[Dict]) -> Tuple[str, int]:
        """9. Sentiment Momentum - Candle pattern analysis"""
        if len(history) < 10:
            return "NEUTRAL", 0
        
        prices = [h['price'] for h in history[-10:]]
        
        # Count bullish vs bearish moves
        bullish = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        bearish = len(prices) - 1 - bullish
        
        if bullish >= 7:
            return "STRONG_BULLISH", 8
        elif bearish >= 7:
            return "STRONG_BEARISH", 8
        elif bullish >= 6:
            return "BULLISH", 5
        elif bearish >= 6:
            return "BEARISH", 5
        else:
            return "NEUTRAL", 0
    
    @staticmethod
    def liquidity_flow(history: List[Dict]) -> Tuple[str, int]:
        """10. Liquidity Flow - Money flow analysis"""
        if len(history) < 15:
            return "UNKNOWN", 0
        
        volumes = [h['volume'] for h in history]
        prices = [h['price'] for h in history]
        
        # Calculate money flow
        money_flow = []
        for i in range(1, len(history)):
            typical_price = prices[i]
            if prices[i] > prices[i-1]:
                money_flow.append(typical_price * volumes[i])
            else:
                money_flow.append(-typical_price * volumes[i])
        
        total_flow = sum(money_flow[-10:])
        
        if total_flow > 0:
            return "INFLOW", 7
        else:
            return "OUTFLOW", 7

class SmartSignalBot:
    """Unique Smart Signal Generation"""
    
    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager('signals.db')
        self.tracker = SmartPriceTracker(config.PRICE_HISTORY_SIZE)
        self.analyzer = UniqueAnalyzer()
        
        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()
        
        logger.info("✅ Smart Unique Bot initialized")
    
    async def update_prices(self, markets: List[str]):
        """Update price data"""
        try:
            session = await self.dcx._get_session()
            async with session.get(f"{self.dcx.PUBLIC_URL}/market_data/ticker") as response:
                tickers = await response.json()
            
            if not isinstance(tickers, list):
                return
            
            for ticker in tickers:
                market = ticker.get('market', '')
                
                if market in markets:
                    price = float(ticker.get('last_price', 0) or 0)
                    volume = float(ticker.get('volume', 0) or 0)
                    
                    if price > 0:
                        self.tracker.add_price(market, price, volume)
        
        except Exception as e:
            logger.error(f"Price update error: {e}")
    
    def analyze_market(self, market: str) -> Optional[Dict]:
        """Analyze using 10 unique indicators"""
        
        if not self.tracker.has_enough_data(market, config.MIN_DATA_POINTS):
            return None
        
        history = self.tracker.get_history(market)
        
        score = 0
        details = {}
        
        # 1. Momentum Wave (12 pts)
        wave_state, wave_score = self.analyzer.momentum_wave(history)
        score += wave_score
        details['momentum_wave'] = wave_state
        
        # 2. Smart Money (10 pts)
        smart_state, smart_score = self.analyzer.smart_money_index(history)
        score += smart_score
        details['smart_money'] = smart_state
        
        # 3. Market Sync (10 pts)
        sync_state, sync_score = self.analyzer.market_sync_index(history)
        score += sync_score
        details['market_sync'] = sync_state
        
        # 4. Adaptive Volatility (10 pts)
        vol_val, vol_score = self.analyzer.adaptive_volatility(history)
        score += vol_score
        details['volatility'] = round(vol_val * 100, 2)
        
        # 5. Trend Persistence (8 pts)
        trend_state, trend_score = self.analyzer.trend_persistence(history)
        score += trend_score
        details['trend'] = trend_state
        
        # 6. Momentum Divergence (8 pts)
        has_div, div_type, div_score = self.analyzer.momentum_divergence(history)
        score += div_score
        details['divergence'] = div_type
        
        # 7. Volume Profile (8 pts)
        vol_prof, vol_prof_score = self.analyzer.volume_profile(history)
        score += vol_prof_score
        details['volume_profile'] = vol_prof
        
        # 8. Price Elasticity (7 pts)
        elast_val, elast_score = self.analyzer.price_elasticity(history)
        score += elast_score
        details['elasticity'] = round(elast_val * 100, 2)
        
        # 9. Sentiment Momentum (8 pts)
        sent_state, sent_score = self.analyzer.sentiment_momentum(history)
        score += sent_score
        details['sentiment'] = sent_state
        
        # 10. Liquidity Flow (7 pts)
        liq_state, liq_score = self.analyzer.liquidity_flow(history)
        score += liq_score
        details['liquidity'] = liq_state
        
        # Calculate percentage
        max_possible = 88  # Sum of all max scores
        final_score = int((score / max_possible) * 100)
        
        logger.debug(f"{market}: Score={final_score}% | Wave={wave_score} Smart={smart_score} Sync={sync_score}")
        
        if final_score < config.MIN_SCORE:
            return None
        
        # Determine direction
        bullish_signals = 0
        bearish_signals = 0
        
        if "UP" in wave_state:
            bullish_signals += 3
        elif "DOWN" in wave_state:
            bearish_signals += 3
        
        if "BUY" in smart_state:
            bullish_signals += 2
        elif "SELL" in smart_state:
            bearish_signals += 2
        
        if "BULLISH" in sync_state:
            bullish_signals += 2
        elif "BEARISH" in sync_state:
            bearish_signals += 2
        
        if "UP" in trend_state:
            bullish_signals += 2
        elif "DOWN" in trend_state:
            bearish_signals += 2
        
        if "BULLISH" in div_type:
            bullish_signals += 2
        elif "BEARISH" in div_type:
            bearish_signals += 2
        
        if "BULLISH" in sent_state:
            bullish_signals += 1
        elif "BEARISH" in sent_state:
            bearish_signals += 1
        
        if bullish_signals <= bearish_signals:
            if bearish_signals - bullish_signals < 2:
                return None
            side = "SELL"
        else:
            if bullish_signals - bearish_signals < 2:
                return None
            side = "BUY"
        
        # Calculate levels
        current_price = history[-1]['price']
        price_changes = [abs(history[i]['price'] - history[i-1]['price']) 
                        for i in range(1, len(history))]
        atr = np.mean(price_changes[-15:])
        
        if side == "BUY":
            entry = current_price
            sl = entry - (atr * 2)
            tp = entry + (atr * 3.5)
        else:
            entry = current_price
            sl = entry + (atr * 2)
            tp = entry - (atr * 3.5)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        confidence = "HIGH" if final_score >= 70 else "MEDIUM" if final_score >= 60 else "GOOD"
        
        return {
            'market': market,
            'timeframe': 'LIVE',
            'side': side,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'rr_ratio': round(rr_ratio, 1),
            'logic_score': final_score,
            'confidence': confidence,
            'mode': 'SMART',
            'details': details
        }
    
    async def send_signal(self, signal: Dict):
        """Send unique signal"""
        
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "GOOD": "✨"}[signal['confidence']]
        
        details = signal['details']
        
        insights = []
        if "ACCELERATING" in details.get('momentum_wave', ''):
            insights.append(f"⚡ {details['momentum_wave']}")
        
        if "SMART" in details.get('smart_money', ''):
            insights.append(f"🐋 {details['smart_money']}")
        
        if "SYNC" in details.get('market_sync', ''):
            insights.append(f"🎯 {details['market_sync']}")
        
        if "DIV" in details.get('divergence', ''):
            insights.append(f"🔄 {details['divergence']}")
        
        insight_text = "\n".join([f"  • {i}" for i in insights[:3]]) if insights else "  • Smart setup detected"
        
        message = f"""🎨 *UNIQUE SIGNAL* 🎨

📌 *Pair:* {signal['market']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎨 *Unique Insights:*
{insight_text}

⚠️ *CoinDCX Manual Trade*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Signal: {signal['market']} {signal['side']} ({signal['logic_score']}%)")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def scan(self):
        """Main scan"""
        
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New day: {today}")
        
        markets = await self.dcx.get_markets()
        if not markets:
            return
        
        inr_markets = []
        for m in markets:
            symbol = m.get('symbol', '') or m.get('pair', '')
            for coin in config.COINS_TO_MONITOR:
                if coin in symbol and ('INR' in symbol or 'INRT' in symbol):
                    if symbol not in inr_markets:
                        inr_markets.append(symbol)
                    break
        
        if not inr_markets:
            return
        
        await self.update_prices(inr_markets)
        
        found = 0
        
        for market in inr_markets:
            try:
                signal = self.analyze_market(market)
                
                if signal:
                    key = f"{market}_{signal['side']}_{datetime.now().strftime('%Y%m%d%H')}"
                    
                    if key not in self.processed:
                        await self.send_signal(signal)
                        self.db.save_signal(signal)
                        self.processed.add(key)
                        self.daily_signals += 1
                        found += 1
                        
                        if len(self.processed) > 200:
                            old = list(self.processed)[:50]
                            for o in old:
                                self.processed.remove(o)
            
            except Exception as e:
                logger.error(f"Error {market}: {e}")
                continue
        
        logger.info(f"✅ Scan done. Signals: {found} | Today: {self.daily_signals}")
    
    async def run(self):
        """Main loop"""
logger.info("🎨 UNIQUE SMART Signal Bot Started!")
        logger.info("=" * 50)
        logger.info("📊 10 Proprietary Indicators")
        logger.info(f"🎯 Min Score: {config.MIN_SCORE}%")
        logger.info(f"⏱️ Scan: {config.SCAN_INTERVAL}s")
        logger.info("=" * 50)
        
        try:
            await self.telegram.send_message(
                "🎨 *UNIQUE Bot Started*\n\n"
                "📊 10 Proprietary Indicators\n"
                f"🎯 Min Score: {config.MIN_SCORE}%\n"
                f"⏱️ Scan: {config.SCAN_INTERVAL}s\n\n"
                "Target: 15-25 quality signals/day"
            )
        except:
            pass
        
        # Build initial data (2 min)
        logger.info("Building data... (2 minutes)")
        for _ in range(3):
            markets = await self.dcx.get_markets()
            if markets:
                inr = [m.get('symbol', '') or m.get('pair', '') 
                      for m in markets 
                      if any(c in (m.get('symbol', '') or m.get('pair', '')) 
                            for c in config.COINS_TO_MONITOR)]
                await self.update_prices(inr)
            await asyncio.sleep(40
  logger.info("✅ Ready for signals!")
        
        while True:
            try:
                await self.scan()
                await asyncio.sleep(config.SCAN_INTERVAL)
            
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                break
            
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(10)
        
        await self.dcx.close()

async def main():
    bot = SmartSignalBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())