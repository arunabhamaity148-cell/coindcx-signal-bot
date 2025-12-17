"""
WINNING CoinDCX Signal Bot
Professional-grade strategy for HIGH WIN RATE
Only CoinDCX data, Manual trading
Target: 75%+ win rate, 10-20 signals/day
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('winning_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # HIGH WIN RATE coins (Best liquidity on CoinDCX)
    COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'MATIC', 'DOGE', 'ADA', 'DOT', 'LINK', 'UNI']
    
    # WINNING SETTINGS
    MIN_SCORE = 65  # High quality threshold
    SCAN_INTERVAL = 30  # Fast scanning
    PRICE_HISTORY = 40  # Good data depth
    MIN_DATA_POINTS = 20  # Quick signals
    
    # RISK MANAGEMENT
    MIN_RR_RATIO = 1.8  # Minimum 1:1.8 R:R
    MAX_DAILY_SIGNALS = 20  # Quality control
    
config = Config()

class PriceTracker:
    """Track price + orderbook data"""
    
    def __init__(self, size: int = 40):
        self.prices = {}  # {market: deque of prices}
        self.volumes = {}  # {market: deque of volumes}
        self.orderbooks = {}  # {market: latest orderbook}
        self.last_update = {}
        self.size = size
    
    def add_data(self, market: str, price: float, volume: float, orderbook: dict):
        """Add new data point"""
        
        # Validate
        if market in self.prices and len(self.prices[market]) > 0:
            last = self.prices[market][-1]
            change = abs(price - last) / last
            if change > 0.20:  # Skip >20% jump
                return
        
        if market not in self.prices:
            self.prices[market] = deque(maxlen=self.size)
            self.volumes[market] = deque(maxlen=self.size)
        
        self.prices[market].append(price)
        self.volumes[market].append(volume)
        self.orderbooks[market] = orderbook
        self.last_update[market] = datetime.now()
    
    def has_data(self, market: str, min_points: int) -> bool:
        """Check if enough data"""
        if market not in self.prices:
            return False
        
        if len(self.prices[market]) < min_points:
            return False
        
        # Freshness check
        if market in self.last_update:
            age = (datetime.now() - self.last_update[market]).seconds
            if age > 120:  # 2 minutes max
                return False
        
        return True
    
    def get_prices(self, market: str) -> List[float]:
        return list(self.prices.get(market, []))
    
    def get_volumes(self, market: str) -> List[float]:
        return list(self.volumes.get(market, []))
    
    def get_orderbook(self, market: str) -> dict:
        return self.orderbooks.get(market, {})

class WinningAnalyzer:
    """Professional analysis for HIGH WIN RATE"""
    
    @staticmethod
    def orderbook_imbalance(orderbook: dict) -> tuple:
        """Factor 1: Smart money detection"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0, "NEUTRAL", 0
        
        # Top 5 levels
        bid_vol = sum([b[1] for b in orderbook['bids'][:5]])
        ask_vol = sum([a[1] for a in orderbook['asks'][:5]])
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0, "NEUTRAL", 0
        
        imbalance = (bid_vol - ask_vol) / total
        # Scoring
        if imbalance > 0.25:
            return imbalance, "STRONG_BUY", 15
        elif imbalance > 0.15:
            return imbalance, "BUY", 10
        elif imbalance < -0.25:
            return imbalance, "STRONG_SELL", 15
        elif imbalance < -0.15:
            return imbalance, "SELL", 10
        else:
            return imbalance, "NEUTRAL", 0
    
    @staticmethod
    def volume_surge(volumes: List[float]) -> tuple:
        """Factor 2: Institutional activity"""
        if len(volumes) < 15:
            return False, 0
        
        recent = np.mean(volumes[-5:])
        avg = np.mean(volumes[-15:])
        
        if avg == 0:
            return False, 0
        
        ratio = recent / avg
        
        if ratio > 2.5:
            return True, 15  # Very strong
        elif ratio > 2.0:
            return True, 12
        elif ratio > 1.5:
            return True, 8
        else:
            return False, 0
    
    @staticmethod
    def momentum_strength(prices: List[float]) -> tuple:
        """Factor 3: Strong trend detection"""
        if len(prices) < 20:
            return "NONE", 0
        
        # Multi-timeframe momentum
        short = (prices[-1] - prices[-5]) / prices[-5]  # 5-period
        medium = (prices[-1] - prices[-10]) / prices[-10]  # 10-period
        long = (prices[-1] - prices[-20]) / prices[-20]  # 20-period
        
        # All aligned = strong trend
        if all([short > 0.01, medium > 0.015, long > 0.02]):
            return "STRONG_UP", 15
        elif all([short < -0.01, medium < -0.015, long < -0.02]):
            return "STRONG_DOWN", 15
        elif short > 0.008 and medium > 0.01:
            return "UP", 10
        elif short < -0.008 and medium < -0.01:
            return "DOWN", 10
        else:
            return "NONE", 0
    
    @staticmethod
    def spread_quality(orderbook: dict) -> tuple:
        """Factor 4: Execution advantage"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0, 0
        
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        
        if best_bid == 0:
            return 0, 0
        
        spread_pct = ((best_ask - best_bid) / best_bid) * 100
        
        # Tighter spread = better execution
        if spread_pct < 0.05:
            return spread_pct, 10
        elif spread_pct < 0.10:
            return spread_pct, 7
        elif spread_pct < 0.15:
            return spread_pct, 4
        else:
            return spread_pct, 0
    
    @staticmethod
    def price_consolidation(prices: List[float]) -> tuple:
        """Factor 5: Breakout setup detection"""
        if len(prices) < 20:
            return False, 0
        
        # Check if price consolidating (low volatility)
        recent = prices[-10:]
        volatility = np.std(recent) / np.mean(recent)
        
        # Low volatility before breakout
        if volatility < 0.015:  # Very tight
            # Check if approaching breakout
            if prices[-1] == max(recent) or prices[-1] == min(recent):
                return True, 12  # At range extreme
            else:
                return True, 8  # Consolidating
        else:
            return False, 0
    
    @staticmethod
    def volume_price_divergence(prices: List[float], volumes: List[float]) -> tuple:
        """Factor 6: Hidden strength/weakness"""
        if len(prices) < 15 or len(volumes) < 15:
            return "NONE", 0
        
        price_trend = prices[-1] - prices[-10]
        
        vol_recent = np.mean(volumes[-5:])
        vol_past = np.mean(volumes[-15:-5])
        
        # Bullish: Price down but volume increasing (accumulation)
        if price_trend < 0 and vol_recent > vol_past * 1.5:
            return "BULLISH_DIV", 10
        
        # Bearish: Price up but volume decreasing (distribution)
        elif price_trend > 0 and vol_recent < vol_past * 0.7:
            return "BEARISH_DIV", 10
        
        else:
            return "NONE", 0

class WinningSignalBot:
    """Professional signal generation - HIGH WIN RATE"""
    
    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager('winning_signals.db')
        self.tracker = PriceTracker(config.PRICE_HISTORY)
        self.analyzer = WinningAnalyzer()
        
        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()
        
        logger.info("✅ WINNING Signal Bot initialized")
    
    async def update_all_data(self, markets: List[str]):
        """Update price + orderbook data"""
        try:
            # Get tickers
            session = await self.dcx._get_session()
            async with session.get(f"{self.dcx.PUBLIC_URL}/market_data/ticker") as response:
                tickers = await response.json()
            
            if not isinstance(tickers, list):
                return
            
            # Process each market
            for market in markets:
                # Find ticker
                ticker = None
                for t in tickers:
                    if t.get('market', '') == market:
                        ticker = t
                        break
                
                if not ticker:
                    continue
                
                price = float(ticker.get('last_price', 0) or 0)
                volume = float(ticker.get('volume', 0) or 0)
                
                if price == 0:
                    continue
                
                # Get orderbook
                orderbook = await self.dcx.get_orderbook(market)
                
                # Add to tracker
                self.tracker.add_data(market, price, volume, orderbook)
        
        except Exception as e:
            logger.error(f"Data update error: {e}")
    
    def analyze_market(self, market: str) -> Optional[Dict]:
        """Deep professional analysis"""
        
        if not self.tracker.has_data(market, config.MIN_DATA_POINTS):
            return None
        
        prices = self.tracker.get_prices(market)
        volumes = self.tracker.get_volumes(market)
        orderbook = self.tracker.get_orderbook(market)
        
        # === 6 PROFESSIONAL FACTORS ===
        
        score = 0
        factors = {}
        
        # 1. Orderbook Imbalance (15 pts)
        imb_val, imb_dir, imb_score = self.analyzer.orderbook_imbalance(orderbook)
        score += imb_score
        factors['imbalance'] = {'value': round(imb_val, 3), 'direction': imb_dir}
        
        # 2. Volume Surge (15 pts)
        vol_surge, vol_score = self.analyzer.volume_surge(volumes)
        score += vol_score
        factors['volume_surge'] = vol_surge
        
        # 3. Momentum Strength (15 pts)
        mom_dir, mom_score = self.analyzer.momentum_strength(prices)
        score += mom_score
        factors['momentum'] = mom_dir
        
        # 4. Spread Quality (10 pts)
        spread_val, spread_score = self.analyzer.spread_quality(orderbook)
        score += spread_score
        factors['spread'] = round(spread_val, 4)
        
        # 5. Price Consolidation (12 pts)
        consol, consol_score = self.analyzer.price_consolidation(prices)
        score += consol_score
        factors['consolidation'] = consol
        
        # 6. Volume-Price Divergence (10 pts)
        div_type, div_score = self.analyzer.volume_price_divergence(prices, volumes)
        score += div_score
        factors['divergence'] = div_type
        
        # Calculate final score
        max_possible = 77
        final_score = int((score / max_possible) * 100)
        
        logger.info(f"{market}: Score={final_score}% | Imb={imb_score} Vol={vol_score} Mom={mom_score} Spread={spread_score}")
        
        if final_score < config.MIN_SCORE:
            return None
        
        # === DETERMINE DIRECTION ===
        
        bullish = 0
        bearish = 0
        
        # Orderbook
        if "BUY" in imb_dir:
            bullish += 3
        elif "SELL" in imb_dir:
            bearish += 3
        
        # Momentum
        if "UP" in mom_dir:
            bullish += 3
        elif "DOWN" in mom_dir:
            bearish += 3
        
        # Divergence
        if "BULLISH" in div_type:
            bullish += 2
        elif "BEARISH" in div_type:
            bearish += 2
        
        # Volume surge adds weight to direction
        if vol_surge:
            if bullish > bearish:
                bullish += 2
            elif bearish > bullish:
                bearish += 2
        
        # Decision
        if bullish <= bearish:
            if bearish - bullish < 2:
                return None
            side = "SELL"
        else:
            if bullish - bearish < 2:
                return None
            side = "BUY"
        
        # === CALCULATE LEVELS ===
        
        current_price = prices[-1]
        
        # ATR for stop/target
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = np.mean(price_changes[-14:])
        
        if side == "BUY":
            entry = current_price
            sl = entry - (atr * 2.5)
            tp = entry + (atr * 5)  # 1:2 R:R
        else:
            entry = current_price
            sl = entry + (atr * 2.5)
            tp = entry - (atr * 5)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        # Check minimum R:R
        if rr_ratio < config.MIN_RR_RATIO:
            return None
        
        # Confidence
        if final_score >= 80:
            confidence = "VERY_HIGH"
        elif final_score >= 72:
            confidence = "HIGH"
        else:
            confidence = "GOOD"
        
        return {
            'market': market,
            'timeframe': 'PROFESSIONAL',
            'side': side,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'rr_ratio': round(rr_ratio, 1),
            'logic_score': final_score,
            'confidence': confidence,
            'mode': 'WIN',
            'factors': factors
        }
    
    async def send_signal(self, signal: Dict):
        """Send winning signal"""
        
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_map = {
            "VERY_HIGH": "🔥🔥🔥",
            "HIGH": "🔥🔥",
            "GOOD": "🔥"
        }
        conf_emoji = conf_map[signal['confidence']]
        
        factors = signal['factors']
        
        insights = []
        
        imb = factors['imbalance']
        if "STRONG" in imb['direction']:
            insights.append(f"🐋 {imb['direction']} ({imb['value']:+.2%})")
        
        if factors['volume_surge']:
            insights.append("📊 Volume Surge Detected")
        
        if "STRONG" in factors['momentum']:
            insights.append(f"⚡ {factors['momentum']}")
        
        if factors['consolidation']:
            insights.append("🎯 Breakout Setup")
        
        if "DIV" in factors['divergence']:
            insights.append(f"🔄 {factors['divergence']}")
        
        insight_text = "\n".join([f"  • {i}" for i in insights[:4]]) if insights else "  • Strong setup"
        
        message = f"""🏆 *WINNING SIGNAL* 🏆

📌 *Pair:* {signal['market']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎨 *Professional Factors:*
{insight_text}

💼 *CoinDCX Manual Trade*
✅ *High Win Probability*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ WINNING Signal: {signal['market']} {signal['side']} ({signal['logic_score']}%)")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def scan(self):
        """Professional market scan"""
        
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New trading day: {today}")
        
        # Check daily limit
        if self.daily_signals >= config.MAX_DAILY_SIGNALS:
            logger.info(f"Daily limit reached: {self.daily_signals}/{config.MAX_DAILY_SIGNALS}")
            return
        
        # Get markets
        markets_data = await self.dcx.get_markets()
        if not markets_data:
            return
        
        # Filter INR markets
        inr_markets = []
        for m in markets_data:
            symbol = m.get('symbol', '') or m.get('pair', '')
            for coin in config.COINS:
                if coin in symbol and ('INR' in symbol or 'INRT' in symbol):
                    if symbol not in inr_markets:
                        inr_markets.append(symbol)
                    break
        
        if not inr_markets:
            return
        
        # Update data
        await self.update_all_data(inr_markets)
        
        # Analyze
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
                logger.error(f"Analysis error {market}: {e}")
                continue
        
        logger.info(f"✅ Scan complete. Signals: {found} | Today: {self.daily_signals}/{config.MAX_DAILY_SIGNALS}")
    
    async def run(self):
        """Main loop"""
        
        logger.info("🏆 WINNING Signal Bot Started!")
        logger.info("=" * 50)
        logger.info("📊 Professional 6-Factor Analysis")
        logger.info(f"🎯 Min Score: {config.MIN_SCORE}%")
        logger.info(f"📐 Min R:R: 1:{config.MIN_RR_RATIO}")
        logger.info(f"⏱️ Scan: {config.SCAN_INTERVAL}s")
        logger.info(f"🎯 Max signals/day: {config.MAX_DAILY_SIGNALS}")
        logger.info("=" * 50)
        
        try:
            await self.telegram.send_message(
                "🏆 *WINNING Bot Started*\n\n"
                "📊 Professional Strategy\n"
                f"🎯 Min Score: {config.MIN_SCORE}%\n"
                f"📐 Min R:R: 1:{config.MIN_RR_RATIO}\n"
                f"🎯 Max: {config.MAX_DAILY_SIGNALS} signals/day\n\n"
                "Target: 75%+ win rate!"
            )
        except:
            pass
        
        # Build data (90 seconds)
        logger.info("Building professional data... (90 seconds)")
        for _ in range(3):
            markets_data = await self.dcx.get_markets()
            if markets_data:
                inr = [m.get('symbol', '') or m.get('pair', '') 
                      for m in markets_data 
                      if any(c in (m.get('symbol', '') or m.get('pair', '')) 
                            for c in config.COINS)]
                await self.update_all_data(inr)
            await asyncio.sleep(30)
        
        logger.info("✅ Ready for WINNING signals!")
        
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
    bot = WinningSignalBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())