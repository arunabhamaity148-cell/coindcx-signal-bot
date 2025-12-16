"""
HIGH QUALITY SIGNAL BOT
Only genuine, high-probability trades
Smart multi-factor analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
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
    
    # HIGH QUALITY coins only (Best liquidity + volume)
    COINS_TO_MONITOR = [
        'BTC', 'ETH', 'SOL', 'XRP', 'BNB',  # Top tier
        'ADA', 'DOGE', 'MATIC', 'DOT', 'AVAX',  # High volume
        'LINK', 'UNI', 'ATOM', 'LTC'  # Established
    ]
    
    # QUALITY SETTINGS
    MIN_SCORE = 70  # Only HIGH quality (70%+)
    SCAN_INTERVAL = 45  # 45s - balanced
    PRICE_HISTORY_SIZE = 60  # More data = better analysis
    MIN_DATA_POINTS = 40  # Need substantial history
    
    # RISK MANAGEMENT
    MAX_SIGNALS_PER_HOUR = 5  # Don't spam
    MIN_SIGNAL_GAP_MINUTES = 10  # Wait between signals
    
config = Config()

class SmartPriceTracker:
    """Advanced price tracking with quality checks"""
    
    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        self.data = {}
        self.last_update = {}
    
    def add_price(self, market: str, price: float, volume: float):
        """Add price with validation"""
        
        # Validate price change (detect bad data)
        if market in self.data and len(self.data[market]) > 0:
            last_price = self.data[market][-1]['price']
            change = abs(price - last_price) / last_price
            
            # Skip if change > 10% (likely bad data)
            if change > 0.10:
                logger.warning(f"Suspicious price change for {market}: {change*100:.1f}%")
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
    
    def has_quality_data(self, market: str, min_points: int) -> bool:
        """Check if data is sufficient AND fresh"""
        if market not in self.data:
            return False
        
        if len(self.data[market]) < min_points:
            return False
        
        # Check data freshness (last update within 2 minutes)
        if market in self.last_update:
            age = (datetime.now() - self.last_update[market]).seconds
            if age > 120:
                return False
        
        return True

class AdvancedAnalyzer:
    """High-quality signal analysis"""
    
    @staticmethod
    def calculate_trend_strength(history: List[Dict]) -> float:
        """Calculate trend strength (0-1)"""
        if len(history) < 20:
            return 0
        
        prices = [h['price'] for h in history]
        
        # Linear regression slope
        x = np.arange(len(prices))
        y = np.array(prices)
        
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope
        avg_price = np.mean(prices)
        trend_strength = abs(slope / avg_price) * 100
        
        return min(trend_strength, 1.0)
    
    @staticmethod
    def calculate_momentum_quality(history: List[Dict]) -> Dict:
        """Multi-timeframe momentum"""
        if len(history) < 30:
            return {'quality': 0, 'direction': 'NEUTRAL'}
        
        prices = [h['price'] for h in history]
        
        # Short-term (last 10)
        short_mom = (prices[-1] - prices[-10]) / prices[-10]
        
        # Medium-term (last 20)
        med_mom = (prices[-1] - prices[-20]) / prices[-20]
        
        # Long-term (last 30)
        long_mom = (prices[-1] - prices[-30]) / prices[-30]
        
        # All should align for quality
        bullish_aligned = all([short_mom > 0, med_mom > 0, long_mom > 0])
        bearish_aligned = all([short_mom < 0, med_mom < 0, long_mom < 0])
        
        if bullish_aligned:
            quality = min(abs(short_mom + med_mom + long_mom), 0.1) * 10
            return {'quality': quality, 'direction': 'BULLISH'}
        elif bearish_aligned:
            quality = min(abs(short_mom + med_mom + long_mom), 0.1) * 10
            return {'quality': quality, 'direction': 'BEARISH'}
        else:
            return {'quality': 0, 'direction': 'NEUTRAL'}
    
    @staticmethod
    def detect_volume_confirmation(history: List[Dict]) -> bool:
        """Volume must confirm price move"""
        if len(history) < 20:
            return False
        
        prices = [h['price'] for h in history[-10:]]
        volumes = [h['volume'] for h in history]
        
        # Price direction
        price_up = prices[-1] > prices[0]
        
        # Volume trend
        recent_vol = np.mean(volumes[-10:])
        past_vol = np.mean(volumes[-30:-10])
        
        vol_increasing = recent_vol > past_vol * 1.3
        
        return vol_increasing
    
    @staticmethod
    def calculate_volatility_quality(history: List[Dict]) -> float:
        """Good volatility = tradeable, not chaotic"""
        if len(history) < 30:
            return 0
        
        prices = [h['price'] for h in history[-30:]]
        returns = []
        
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        volatility = np.std(returns)
        
        # Optimal range: 0.5% - 2.5%
        if 0.005 <= volatility <= 0.025:
            return 1.0
        elif 0.003 <= volatility <= 0.035:
            return 0.7
        elif volatility < 0.002:
            return 0.3  # Too flat
        else:
            return 0.2  # Too volatile
    
    @staticmethod
    def detect_support_resistance(history: List[Dict]) -> Dict:
        """Find if near key levels"""
        if len(history) < 40:
            return {'near_level': False, 'level_type': None}
        
        prices = [h['price'] for h in history]
        current = prices[-1]
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(prices)-2):
            # Swing high
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    highs.append(prices[i])
            
            # Swing low
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    lows.append(prices[i])
        
        # Check if current price near any level
        for high in highs[-5:]:
            if abs(current - high) / current < 0.01:  # Within 1%
                return {'near_level': True, 'level_type': 'RESISTANCE'}
        
        for low in lows[-5:]:
            if abs(current - low) / current < 0.01:
                return {'near_level': True, 'level_type': 'SUPPORT'}
        
        return {'near_level': False, 'level_type': None}

class HighQualitySignalBot:
    """Premium quality signals only"""
    
    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager('quality_signals.db')
        self.tracker = SmartPriceTracker(config.PRICE_HISTORY_SIZE)
        self.analyzer = AdvancedAnalyzer()
        
        self.processed = set()
        self.daily_signals = 0
        self.hourly_signals = 0
        self.last_signal_time = {}
        self.last_hour_reset = datetime.now()
        self.last_date = datetime.now().date()
        
        logger.info("✅ HIGH QUALITY Signal Bot initialized")
    
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
                    price = float(ticker.get('last_price', 0) or ticker.get('bid', 0) or 0)
                    volume = float(ticker.get('volume', 0) or 0)
                    
                    if price > 0 and volume > 0:
                        self.tracker.add_price(market, price, volume)
        
        except Exception as e:
            logger.error(f"Price update error: {e}")
    
    def analyze_market(self, market: str) -> Optional[Dict]:
        """Deep quality analysis"""
        
        # Check data quality
        if not self.tracker.has_quality_data(market, config.MIN_DATA_POINTS):
            return None
        
        history = self.tracker.get_history(market)
        
        # === QUALITY CHECKS ===
        
        # 1. Trend Strength (20 points)
        trend_strength = self.analyzer.calculate_trend_strength(history)
        trend_score = int(trend_strength * 20)
        
        # 2. Momentum Quality (25 points)
        momentum = self.analyzer.calculate_momentum_quality(history)
        momentum_score = int(momentum['quality'] * 25)
        
        if momentum['direction'] == 'NEUTRAL':
            return None  # No clear direction
        
        # 3. Volume Confirmation (20 points)
        volume_confirmed = self.analyzer.detect_volume_confirmation(history)
        volume_score = 20 if volume_confirmed else 5
        
        # 4. Volatility Quality (15 points)
        vol_quality = self.analyzer.calculate_volatility_quality(history)
        vol_score = int(vol_quality * 15)
        
        # 5. Support/Resistance (20 points)
        sr_data = self.analyzer.detect_support_resistance(history)
        
        sr_score = 0
        if sr_data['near_level']:
            if (momentum['direction'] == 'BULLISH' and sr_data['level_type'] == 'SUPPORT') or \
               (momentum['direction'] == 'BEARISH' and sr_data['level_type'] == 'RESISTANCE'):
                sr_score = 20  # Good: bounce from support or rejection from resistance
        else:
            sr_score = 10  # Neutral: not near any level
        
        # === CALCULATE FINAL SCORE ===
        total_score = trend_score + momentum_score + volume_score + vol_score + sr_score
        
        logger.info(f"{market}: Trend={trend_score} Mom={momentum_score} Vol={volume_score} Volat={vol_score} SR={sr_score} | Total={total_score}")
        
        if total_score < config.MIN_SCORE:
            return None
        
        # === SIGNAL GENERATION ===
        
        side = "BUY" if momentum['direction'] == 'BULLISH' else "SELL"
        current_price = history[-1]['price']
        
        # Calculate ATR for SL/TP
        price_changes = [abs(history[i]['price'] - history[i-1]['price']) 
                        for i in range(1, len(history))]
        atr = np.mean(price_changes[-20:])
        
        if side == "BUY":
            entry = current_price
            sl = entry - (atr * 2.5)  # Wider stops for quality
            tp = entry + (atr * 4)     # Better R:R
        else:
            entry = current_price
            sl = entry + (atr * 2.5)
            tp = entry - (atr * 4)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        # Confidence
        if total_score >= 85:
            confidence = "VERY_HIGH"
        elif total_score >= 75:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"
        
        return {
            'market': market,
            'timeframe': 'QUALITY',
            'side': side,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'rr_ratio': round(rr_ratio, 1),
            'logic_score': total_score,
            'confidence': confidence,
            'mode': 'QUALITY',
            'details': {
                'trend_strength': round(trend_strength, 3),
                'momentum_direction': momentum['direction'],
                'volume_confirmed': volume_confirmed,
                'volatility_quality': round(vol_quality, 2),
                'near_sr': sr_data['near_level'],
                'sr_type': sr_data['level_type']
            }
        }
    
    def check_signal_limits(self, market: str) -> bool:
        """Rate limiting for quality"""
        
        # Hourly reset
        if (datetime.now() - self.last_hour_reset).seconds > 3600:
            self.hourly_signals = 0
            self.last_hour_reset = datetime.now()
        
        # Max signals per hour
        if self.hourly_signals >= config.MAX_SIGNALS_PER_HOUR:
            return False
        
        # Min gap between signals for same market
        if market in self.last_signal_time:
            gap = (datetime.now() - self.last_signal_time[market]).seconds
            if gap < config.MIN_SIGNAL_GAP_MINUTES * 60:
                return False
        
        return True
    
    async def send_signal(self, signal: Dict):
        """Send premium signal"""
        
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_map = {
            "VERY_HIGH": "🔥🔥🔥",
            "HIGH": "🔥🔥",
            "MEDIUM": "🔥"
        }
        conf_emoji = conf_map[signal['confidence']]
        
        details = signal['details']
        
        insights = []
        if details['trend_strength'] > 0.7:
            insights.append(f"📊 Strong Trend: {details['trend_strength']:.1%}")
        
        insights.append(f"⚡ Momentum: {details['momentum_direction']}")
        
        if details['volume_confirmed']:
            insights.append("✅ Volume Confirmed")
        
        if details['near_sr']:
            insights.append(f"🎯 Near {details['sr_type']}")
        
        insight_text = "\n".join([f"  • {i}" for i in insights])
        
        message = f"""🏆 *QUALITY SIGNAL* 🏆

📌 *Pair:* {signal['market']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎨 *Quality Factors:*
{insight_text}

✅ *CoinDCX Manual Trade*
⚠️ *High Probability Setup*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ QUALITY Signal: {signal['market']} {signal['side']} ({signal['logic_score']}%)")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def scan(self):
        """Quality-focused scan"""
        
        # Daily reset
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New day: {today}")
        
        # Get markets
        markets = await self.dcx.get_markets()
        if not markets:
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
            return
        
        # Update prices
        await self.update_prices(inr_markets)
        
        # Analyze
        found = 0
        
        for market in inr_markets:
            try:
                # Check limits
                if not self.check_signal_limits(market):
                    continue
                
                signal = self.analyze_market(market)
                
                if signal:
                    key = f"{market}_{signal['side']}_{datetime.now().strftime('%Y%m%d%H')}"
                    
                    if key not in self.processed:
                        await self.send_signal(signal)
                        self.db.save_signal(signal)
                        
                        self.processed.add(key)
                        self.daily_signals += 1
                        self.hourly_signals += 1
                        self.last_signal_time[market] = datetime.now()
                        found += 1
                        
                        if len(self.processed) > 200:
                            old = list(self.processed)[:50]
                            for o in old:
                                self.processed.remove(o)
            
            except Exception as e:
                logger.error(f"Analysis error {market}: {e}")
                continue
        
        logger.info(f"✅ Quality scan done. Signals: {found} | Today: {self.daily_signals} | Hour: {self.hourly_signals}")
    
    async def run(self):
        """Main loop"""
        
        logger.info("🏆 HIGH QUALITY Signal Bot Started!")
        logger.info("=" * 50)
        logger.info("📊 Premium signals only")
        logger.info(f"🎯 Min Score: {config.MIN_SCORE}%")
        logger.info(f"⏱️ Scan: {config.SCAN_INTERVAL}s")
        logger.info(f"🎯 Max/hour: {config.MAX_SIGNALS_PER_HOUR}")
        logger.info("=" * 50)
        
        try:
            await self.telegram.send_message(
                "🏆 *QUALITY Bot Started*\n\n"
                "📊 Premium signals only\n"
                f"🎯 Min Score: {config.MIN_SCORE}%\n"
                f"🔒 Max {config.MAX_SIGNALS_PER_HOUR} signals/hour\n\n"
                "High-probability trades only!"
            )
        except:
            pass
        
        # Build initial data
        logger.info("Building price history... (2 minutes)")
        for _ in range(4):
            markets = await self.dcx.get_markets()
            if markets:
                inr_markets = [m.get('symbol', '') or m.get('pair', '') 
                             for m in markets 
                             if any(c in (m.get('symbol', '') or m.get('pair', '')) 
                                   for c in config.COINS_TO_MONITOR)]
                await self.update_prices(inr_markets)
            await asyncio.sleep(30)
        
        logger.info("✅ Ready to generate signals!")
        
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
    bot = HighQualitySignalBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())