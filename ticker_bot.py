"""
WORKING SOLUTION
Uses real-time ticker data from CoinDCX
No candles needed - tracks price movements in memory
Generates signals based on price velocity and momentum
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import os
from dotenv import load_dotenv
from collections import deque
import numpy as np

from helpers import CoinDCXAPI, TelegramNotifier, DatabaseManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    COINS_TO_MONITOR = [
        'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 'DOT',
        'AVAX', 'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'AAVE', 'GRT'
    ]
    
    MIN_SCORE = 55  # Higher threshold for ticker-based
    SCAN_INTERVAL = 30  # Faster scans (30 seconds)
    PRICE_HISTORY_SIZE = 50  # Track last 50 price points
    
config = Config()

class PriceTracker:
    """Track price history for each coin"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = {}  # {market: deque of (timestamp, price, volume)}
    
    def add_price(self, market: str, price: float, volume: float):
        """Add new price point"""
        if market not in self.data:
            self.data[market] = deque(maxlen=self.max_size)
        
        self.data[market].append({
            'timestamp': datetime.now(),
            'price': price,
            'volume': volume
        })
    
    def get_history(self, market: str) -> List[Dict]:
        """Get price history"""
        return list(self.data.get(market, []))
    
    def has_enough_data(self, market: str, min_points: int = 20) -> bool:
        """Check if enough data to analyze"""
        return market in self.data and len(self.data[market]) >= min_points

class TickerAnalyzer:
    """Analyze ticker data for signals"""
    
    @staticmethod
    def calculate_momentum(history: List[Dict]) -> float:
        """Calculate price momentum"""
        if len(history) < 10:
            return 0
        
        prices = [h['price'] for h in history]
        
        # Recent momentum (last 10 points)
        recent_change = (prices[-1] - prices[-10]) / prices[-10]
        
        return recent_change
    
    @staticmethod
    def calculate_velocity(history: List[Dict]) -> float:
        """Calculate price velocity (rate of change)"""
        if len(history) < 5:
            return 0
        
        prices = [h['price'] for h in history[-5:]]
        velocities = []
        
        for i in range(1, len(prices)):
            vel = (prices[i] - prices[i-1]) / prices[i-1]
            velocities.append(vel)
        
        return np.mean(velocities)
    
    @staticmethod
    def calculate_acceleration(history: List[Dict]) -> float:
        """Calculate price acceleration"""
        if len(history) < 10:
            return 0
        
        prices = [h['price'] for h in history[-10:]]
        
        # First half vs second half velocity
        mid = len(prices) // 2
        first_vel = (prices[mid] - prices[0]) / prices[0]
        second_vel = (prices[-1] - prices[mid]) / prices[mid]
        
        acceleration = second_vel - first_vel
        return acceleration
    
    @staticmethod
    def detect_volume_surge(history: List[Dict]) -> bool:
        """Detect volume spike"""
        if len(history) < 20:
            return False
        
        volumes = [h['volume'] for h in history]
        
        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes[:-5])
        
        return recent_vol > avg_vol * 1.5
    
    @staticmethod
    def calculate_volatility(history: List[Dict]) -> float:
        """Calculate price volatility"""
        if len(history) < 20:
            return 0
        
        prices = [h['price'] for h in history[-20:]]
        returns = []
        
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        return np.std(returns)

class TickerSignalBot:
    """Signal bot using ticker data"""
    
    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager('ticker_signals.db')
        self.tracker = PriceTracker(config.PRICE_HISTORY_SIZE)
        self.analyzer = TickerAnalyzer()
        
        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()
        
        logger.info("✅ Ticker Signal Bot initialized")
    
    async def update_prices(self, markets: List[str]):
        """Update price history for all markets"""
        try:
            # Get all tickers at once
            session = await self.dcx._get_session()
            async with session.get(f"{self.dcx.PUBLIC_URL}/market_data/ticker") as response:
                tickers = await response.json()
            
            if not isinstance(tickers, list):
                return
            
            # Update tracker
            for ticker in tickers:
                market = ticker.get('market', '')
                
                if market in markets:
                    price = float(ticker.get('last_price', 0) or ticker.get('bid', 0) or 0)
                    volume = float(ticker.get('volume', 0) or 0)
                    
                    if price > 0:
                        self.tracker.add_price(market, price, volume)
        
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    def analyze_market(self, market: str) -> Dict:
        """Analyze market and generate signal"""
        
        if not self.tracker.has_enough_data(market, 20):
            return None
        
        history = self.tracker.get_history(market)
        
        # Calculate indicators
        momentum = self.analyzer.calculate_momentum(history)
        velocity = self.analyzer.calculate_velocity(history)
        acceleration = self.analyzer.calculate_acceleration(history)
        volume_surge = self.analyzer.detect_volume_surge(history)
        volatility = self.analyzer.calculate_volatility(history)
        
        # Scoring
        score = 0
        max_score = 100
        
        # Momentum score (30 points)
        if abs(momentum) > 0.02:
            score += 20
        elif abs(momentum) > 0.01:
            score += 15
        elif abs(momentum) > 0.005:
            score += 10
        
        # Velocity score (20 points)
        if abs(velocity) > 0.005:
            score += 15
        elif abs(velocity) > 0.002:
            score += 10
        
        # Acceleration score (20 points)
        if momentum > 0 and acceleration > 0:
            score += 15  # Accelerating upward
        elif momentum < 0 and acceleration < 0:
            score += 15  # Accelerating downward
        elif abs(acceleration) > 0.01:
            score += 10
        
        # Volume score (15 points)
        if volume_surge:
            score += 15
        
        # Volatility score (15 points)
        if 0.01 < volatility < 0.04:
            score += 15  # Optimal volatility
        elif 0.005 < volatility < 0.05:
            score += 10
        
        final_score = int((score / max_score) * 100)
        
        if final_score < config.MIN_SCORE:
            return None
        
        # Determine direction
        if momentum > 0 and velocity > 0:
            side = "BUY"
        elif momentum < 0 and velocity < 0:
            side = "SELL"
        else:
            return None
        
        # Get current price
        current_price = history[-1]['price']
        
        # Calculate levels using ATR approximation
        price_changes = [abs(history[i]['price'] - history[i-1]['price']) 
                        for i in range(1, len(history))]
        atr = np.mean(price_changes[-14:]) if len(price_changes) >= 14 else np.mean(price_changes)
        
        if side == "BUY":
            entry = current_price
            sl = entry - (atr * 2)
            tp = entry + (atr * 3)
        else:
            entry = current_price
            sl = entry + (atr * 2)
            tp = entry - (atr * 3)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        confidence = "HIGH" if final_score >= 75 else "MEDIUM" if final_score >= 60 else "LOW"
        
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
            'mode': 'LIVE',
            'details': {
                'momentum': round(momentum * 100, 2),
                'velocity': round(velocity * 100, 2),
                'acceleration': round(acceleration * 100, 2),
                'volume_surge': volume_surge,
                'volatility': round(volatility * 100, 2)
            }
        }
    
    async def send_signal(self, signal: Dict):
        """Send signal to Telegram"""
        
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}[signal['confidence']]
        
        details = signal['details']
        
        insights = []
        if abs(details['momentum']) > 1.5:
            insights.append(f"⚡ Strong Momentum: {details['momentum']:+.2f}%")
        if details['volume_surge']:
            insights.append("📊 Volume Spike Detected")
        if abs(details['acceleration']) > 1:
            insights.append(f"🚀 Accelerating: {details['acceleration']:+.2f}%")
        
        insight_text = "\n".join([f"  • {i}" for i in insights]) if insights else "  • Standard setup"
        
        message = f"""🚨 *LIVE SIGNAL* 🚨

📌 *Pair:* {signal['market']}
📊 *Mode:* LIVE TICKER
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎨 *Live Analysis:*
{insight_text}

⚠️ *Trade on CoinDCX*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Signal: {signal['market']} {signal['side']}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def scan(self):
        """Main scan cycle"""
        
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New day: {today}")
        
        # Get markets
        markets = await self.dcx.get_markets()
        
        if not markets:
            logger.warning("No markets")
            return
        
        # Filter INR markets
        inr_markets = []
        for m in markets:
            symbol = m.get('symbol', '') or m.get('pair', '')
            for coin in config.COINS_TO_MONITOR:
                if coin in symbol and ('INR' in symbol or 'INRT' in symbol):
                    if symbol not in inr_markets:
                        inr_markets.append(symbol)
                    break
        
        if not inr_markets:
            logger.warning("No INR markets matched")
            return
        
        # Update all prices
        await self.update_prices(inr_markets)
        
        # Analyze each market
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
                logger.error(f"Error analyzing {market}: {e}")
                continue
        
        logger.info(f"✅ Scan done. Signals: {found} | Today: {self.daily_signals}")
    
    async def run(self):
        """Main loop"""
        
        logger.info("🚀 Ticker Signal Bot Started!")
        logger.info("📊 Using live ticker data")
        logger.info(f"🎯 Min Score: {config.MIN_SCORE}%")
        logger.info(f"⏱️ Scan: {config.SCAN_INTERVAL}s")
        
        try:
            await self.telegram.send_message(
                "🚀 *Ticker Bot Started*\n\n"
                "📊 Live ticker analysis\n"
                f"🎯 Min Score: {config.MIN_SCORE}%\n"
                f"⏱️ Scan: {config.SCAN_INTERVAL}s\n\n"
                "Tracking price momentum in real-time!"
            )
        except:
            pass
        
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
    bot = TickerSignalBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())