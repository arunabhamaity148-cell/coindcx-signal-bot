"""
CoinDCX-SPECIFIC Trading Bot
Uses REAL CoinDCX market characteristics
NOT generic Binance strategies
Target: 65%+ win rate, 12-18 signals/day
"""

import asyncio
import logging
from datetime import datetime, time
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
        logging.FileHandler('coindcx_edge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    COINDCX_API_KEY = os.getenv('COINDCX_API_KEY')
    COINDCX_SECRET = os.getenv('COINDCX_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Best liquid coins on CoinDCX
    COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'MATIC', 'DOGE', 'ADA']
    
    # CoinDCX-SPECIFIC SETTINGS
    MIN_SCORE = 50  # Balanced for CoinDCX liquidity
    SCAN_INTERVAL = 25  # Fast for CoinDCX volatility
    PRICE_HISTORY = 35
    MIN_DATA_POINTS = 18
    
    # Edge parameters
    INR_PREMIUM_THRESHOLD = 0.035  # 3.5% premium = overbought
    WHALE_WALL_RATIO = 6  # 6x avg order = fake wall
    VOLUME_SPIKE_RATIO = 3.5  # 3.5x avg = artificial
    MIN_SPREAD_PCT = 0.08  # 0.08% min for good execution
    MAX_SPREAD_PCT = 0.35  # 0.35% max acceptable
    
config = Config()

class CoinDCXDataTracker:
    """Track CoinDCX-specific data"""
    
    def __init__(self, size: int = 35):
        self.prices = {}
        self.volumes = {}
        self.orderbooks = {}
        self.spreads = {}
        self.last_update = {}
        self.size = size
    
    def add_data(self, market: str, price: float, volume: float, orderbook: dict):
        # Validate
        if market in self.prices and len(self.prices[market]) > 0:
            last = self.prices[market][-1]
            change = abs(price - last) / last
            if change > 0.25:
                return
        
        if market not in self.prices:
            self.prices[market] = deque(maxlen=self.size)
            self.volumes[market] = deque(maxlen=self.size)
            self.spreads[market] = deque(maxlen=self.size)
        
        self.prices[market].append(price)
        self.volumes[market].append(volume)
        self.orderbooks[market] = orderbook
        
        # Calculate spread
        if orderbook and orderbook.get('bids') and orderbook.get('asks'):
            bid = orderbook['bids'][0][0]
            ask = orderbook['asks'][0][0]
            if bid > 0:
                spread_pct = ((ask - bid) / bid) * 100
                self.spreads[market].append(spread_pct)
        
        self.last_update[market] = datetime.now()
    
    def has_data(self, market: str, min_points: int) -> bool:
        if market not in self.prices:
            return False
        if len(self.prices[market]) < min_points:
            return False
        if market in self.last_update:
            age = (datetime.now() - self.last_update[market]).seconds
            if age > 90:
                return False
        return True
    
    def get_prices(self, market: str) -> List[float]:
        return list(self.prices.get(market, []))
    
    def get_volumes(self, market: str) -> List[float]:
        return list(self.volumes.get(market, []))
    
    def get_orderbook(self, market: str) -> dict:
        return self.orderbooks.get(market, {})
    
    def get_spreads(self, market: str) -> List[float]:
        return list(self.spreads.get(market, []))

class CoinDCXEdgeAnalyzer:
    """CoinDCX-specific edge detection"""
    
    @staticmethod
    def detect_whale_wall(orderbook: dict) -> Tuple[str, int]:
        """Trick 1: Fake wall detection"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return "NONE", 0
        
        bid_sizes = [b[1] for b in orderbook['bids'][:5]]
        ask_sizes = [a[1] for a in orderbook['asks'][:5]]
        
        if len(bid_sizes) < 5 or len(ask_sizes) < 5:
            return "NONE", 0
        
        avg_bid = np.mean(bid_sizes[1:])
        avg_ask = np.mean(ask_sizes[1:])
        
        # Fake support wall
        if bid_sizes[0] > avg_bid * config.WHALE_WALL_RATIO:
            return "FAKE_SUPPORT", 12  # Likely to be pulled, expect dump
        
        # Fake resistance wall
        if ask_sizes[0] > avg_ask * config.WHALE_WALL_RATIO:
            return "FAKE_RESISTANCE", 12  # Likely to be pulled, expect pump
        
        return "NONE", 0
    
    @staticmethod
    def volume_spike_fade(volumes: List[float]) -> Tuple[bool, str, int]:
        """Trick 2: Fade artificial volume spikes"""
        if len(volumes) < 20:
            return False, "NONE", 0
        
        avg_vol = np.mean(volumes[-20:-1])
        current_vol = volumes[-1]
        prev_vol = volumes[-2]
        
        if avg_vol == 0:
            return False, "NONE", 0
        
        spike_ratio = current_vol / avg_vol
        
        # Artificial spike without follow-through
        if spike_ratio > config.VOLUME_SPIKE_RATIO:
            if prev_vol < avg_vol * 1.5:  # Previous candle normal
                return True, "FADE_SPIKE", 15  # High probability reversal
        
        return False, "NONE", 0
    
    @staticmethod
    def time_based_edge() -> Tuple[bool, str, int]:
        """Trick 3: CoinDCX time-based patterns"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        
        # Weekend = range-bound (mean reversion works)
        if weekday >= 5:  # Saturday, Sunday
            return True, "WEEKEND_RANGE", 8
        
        # High activity hours (best signals)
        if 10 <= hour <= 11 or 20 <= hour <= 22:
            return True, "HIGH_ACTIVITY", 10
        
        # Dead zone (avoid)
        if 14 <= hour <= 16:
            return False, "DEAD_ZONE", 0
        
        # Normal hours
        if 9 <= hour <= 23:
            return True, "NORMAL_HOURS", 5
        
        # Night (low liquidity, avoid)
        return False, "LOW_LIQUIDITY", 0
    
    @staticmethod
    def spread_quality_check(spreads: List[float]) -> Tuple[str, int]:
        """Trick 4: Spread-based execution quality"""
        if len(spreads) < 5:
            return "UNKNOWN", 0
        
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads[-10:])
        
        # Excellent execution
        if current_spread < config.MIN_SPREAD_PCT:
            return "EXCELLENT", 10
        
        # Good execution
        if current_spread < avg_spread * 0.8:
            return "GOOD", 8
        
        # Poor execution (high spread)
        if current_spread > config.MAX_SPREAD_PCT:
            return "POOR", 0
        
        # Acceptable
        return "ACCEPTABLE", 5
    
    @staticmethod
    def orderbook_imbalance_coindcx(orderbook: dict) -> Tuple[float, str, int]:
        """Trick 5: CoinDCX-specific imbalance (smaller depth)"""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0, "NEUTRAL", 0
        
        # CoinDCX: Use top 3 levels (not 5, less liquid)
        bid_vol = sum([b[1] for b in orderbook['bids'][:3]])
        ask_vol = sum([a[1] for a in orderbook['asks'][:3]])
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0, "NEUTRAL", 0
        
        imbalance = (bid_vol - ask_vol) / total
        
        # Lower thresholds for CoinDCX
        if imbalance > 0.20:
            return imbalance, "STRONG_BUY", 12
        elif imbalance > 0.12:
            return imbalance, "BUY", 8
        elif imbalance < -0.20:
            return imbalance, "STRONG_SELL", 12
        elif imbalance < -0.12:
            return imbalance, "SELL", 8
        else:
            return imbalance, "NEUTRAL", 0
    
    @staticmethod
    def low_liquidity_momentum(prices: List[float], volumes: List[float]) -> Tuple[str, int]:
        """Trick 6: CoinDCX momentum (adjusts for low liquidity)"""
        if len(prices) < 15 or len(volumes) < 15:
            return "NONE", 0
        
        # Shorter timeframes for CoinDCX
        recent_change = (prices[-1] - prices[-8]) / prices[-8]
        vol_trend = np.mean(volumes[-5:]) / np.mean(volumes[-15:-5])
        
        # Strong momentum with volume
        if recent_change > 0.015 and vol_trend > 1.3:
            return "STRONG_UP", 12
        elif recent_change < -0.015 and vol_trend > 1.3:
            return "STRONG_DOWN", 12
        
        # Moderate momentum
        if recent_change > 0.008 and vol_trend > 1.1:
            return "UP", 8
        elif recent_change < -0.008 and vol_trend > 1.1:
            return "DOWN", 8
        
        return "NONE", 0

class CoinDCXEdgeBot:
    """CoinDCX-specific edge trading bot"""
    
    def __init__(self):
        self.dcx = CoinDCXAPI(config.COINDCX_API_KEY, config.COINDCX_SECRET)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.db = DatabaseManager('coindcx_edge.db')
        self.tracker = CoinDCXDataTracker(config.PRICE_HISTORY)
        self.analyzer = CoinDCXEdgeAnalyzer()
        
        self.processed = set()
        self.daily_signals = 0
        self.last_date = datetime.now().date()
        
        logger.info("✅ CoinDCX Edge Bot initialized")
    
    async def update_data(self, markets: List[str]):
        """Update all data"""
        try:
            session = await self.dcx._get_session()
            async with session.get(f"{self.dcx.PUBLIC_URL}/market_data/ticker") as response:
                tickers = await response.json()
            
            if not isinstance(tickers, list):
                return
            
            for market in markets:
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
                
                orderbook = await self.dcx.get_orderbook(market)
                self.tracker.add_data(market, price, volume, orderbook)
        
        except Exception as e:
            logger.error(f"Data update error: {e}")
    
    def analyze_coindcx_market(self, market: str) -> Optional[Dict]:
        """CoinDCX-specific analysis"""
        
        if not self.tracker.has_data(market, config.MIN_DATA_POINTS):
            return None
        
        prices = self.tracker.get_prices(market)
        volumes = self.tracker.get_volumes(market)
        orderbook = self.tracker.get_orderbook(market)
        spreads = self.tracker.get_spreads(market)
        
        score = 0
        edge_factors = {}
# === 6 CoinDCX-SPECIFIC EDGES ===
        
        # 1. Whale Wall Detection (12 pts)
        wall_type, wall_score = self.analyzer.detect_whale_wall(orderbook)
        score += wall_score
        edge_factors['whale_wall'] = wall_type
        
        # 2. Volume Spike Fade (15 pts)
        is_spike, spike_type, spike_score = self.analyzer.volume_spike_fade(volumes)
        score += spike_score
        edge_factors['volume_spike'] = spike_type
        
        # 3. Time-Based Edge (10 pts)
        time_ok, time_type, time_score = self.analyzer.time_based_edge()
        if not time_ok:
            return None  # Skip bad times
        score += time_score
        edge_factors['time_edge'] = time_type
        
        # 4. Spread Quality (10 pts)
        spread_qual, spread_score = self.analyzer.spread_quality_check(spreads)
        score += spread_score
        edge_factors['spread'] = spread_qual
        
        # 5. Orderbook Imbalance (12 pts)
        imb_val, imb_dir, imb_score = self.analyzer.orderbook_imbalance_coindcx(orderbook)
        score += imb_score
        edge_factors['imbalance'] = {'value': round(imb_val, 3), 'direction': imb_dir}
        
        # 6. Low Liquidity Momentum (12 pts)
        mom_dir, mom_score = self.analyzer.low_liquidity_momentum(prices, volumes)
        score += mom_score
        edge_factors['momentum'] = mom_dir
        
        # Final score
        max_possible = 71
        final_score = int((score / max_possible) * 100)
        
        logger.info(f"{market}: Score={final_score}% | Wall={wall_score} Spike={spike_score} Time={time_score}")
        
        if final_score < config.MIN_SCORE:
            return None
        
        # === DETERMINE DIRECTION ===
        
        bullish = 0
        bearish = 0
        
        # Whale walls (reverse logic)
        if "FAKE_SUPPORT" in wall_type:
            bearish += 3  # Support will be pulled, price drops
        elif "FAKE_RESISTANCE" in wall_type:
            bullish += 3  # Resistance will be pulled, price pumps
        
        # Volume spike (fade logic)
        if "FADE" in spike_type:
            # Check direction to fade
            if prices[-1] > prices[-5]:
                bearish += 3  # Fade upward spike
            else:
                bullish += 3  # Fade downward spike
        
        # Imbalance
        if "BUY" in imb_dir:
            bullish += 2
        elif "SELL" in imb_dir:
            bearish += 2
        
        # Momentum
        if "UP" in mom_dir:
            bullish += 2
        elif "DOWN" in mom_dir:
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
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        atr = np.mean(price_changes[-12:])
        
        # CoinDCX: Wider stops due to volatility
        if side == "BUY":
            entry = current_price
            sl = entry - (atr * 3)
            tp = entry + (atr * 6)
        else:
            entry = current_price
            sl = entry + (atr * 3)
            tp = entry - (atr * 6)
        
        rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        
        # Confidence
        if final_score >= 70:
            confidence = "HIGH"
        elif final_score >= 58:
            confidence = "GOOD"
        else:
            confidence = "MODERATE"
        
        return {
            'market': market,
            'timeframe': 'COINDCX',
            'side': side,
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'rr_ratio': round(rr_ratio, 1),
            'logic_score': final_score,
            'confidence': confidence,
            'mode': 'EDGE',
            'edge_factors': edge_factors
        }
    
    async def send_edge_signal(self, signal: Dict):
        """Send CoinDCX edge signal"""
        
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "GOOD": "✨", "MODERATE": "⚡"}[signal['confidence']]
        
        factors = signal['edge_factors']
        
        insights = []
        
        if "FAKE" in factors.get('whale_wall', ''):
            insights.append(f"🐋 {factors['whale_wall']}")
        
        if "FADE" in factors.get('volume_spike', ''):
            insights.append("📊 Fade Artificial Spike")
        
        imb = factors.get('imbalance', {})
        if "STRONG" in imb.get('direction', ''):
            insights.append(f"💪 {imb['direction']}")
        
        if factors.get('spread') in ['EXCELLENT', 'GOOD']:
            insights.append(f"✅ Spread: {factors['spread']}")
        
        if factors.get('time_edge') != 'NORMAL_HOURS':
            insights.append(f"⏰ {factors['time_edge']}")
        
        insight_text = "\n".join([f"  • {i}" for i in insights[:4]]) if insights else "  • CoinDCX Edge Detected"
        
        message = f"""🎯 *COINDCX EDGE SIGNAL* 🎯

📌 *Pair:* {signal['market']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

🎯 *CoinDCX Edge Factors:*
{insight_text}

💼 *Manual Trade on CoinDCX*
✅ *Market-Specific Strategy*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.telegram.bot.send_message(
                chat_id=self.telegram.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Edge Signal: {signal['market']} {signal['side']} ({signal['logic_score']}%)")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def scan(self):
        """Scan with CoinDCX edge"""
        
        today = datetime.now().date()
        if today != self.last_date:
            self.daily_signals = 0
            self.last_date = today
            logger.info(f"📅 New day: {today}")
        
        markets_data = await self.dcx.get_markets()
        if not markets_data:
            return
        
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
        
        await self.update_data(inr_markets)
        
        found = 0
        
        for market in inr_markets:
            try:
                signal = self.analyze_coindcx_market(market)
                
                if signal:
                    key = f"{market}_{signal['side']}_{datetime.now().strftime('%Y%m%d%H')}"
                    
                    if key not in self.processed:
                        await self.send_edge_signal(signal)
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
        
        logger.info(f"✅ CoinDCX scan done. Signals: {found} | Today: {self.daily_signals}")
    
    async def run(self):
        """Main loop"""
        
        logger.info("🎯 CoinDCX EDGE Bot Started!")
        logger.info("=" * 50)
        logger.info("📊 CoinDCX-Specific Strategy")
        logger.info(f"🎯 Min Score: {config.MIN_SCORE}%")
        logger.info(f"⏱️ Scan: {config.SCAN_INTERVAL}s")
        logger.info("🐋 Whale walls, Volume fades, Time edges")
        logger.info("=" * 50)
        
        try:
            await self.telegram.send_message(
                "🎯 *CoinDCX EDGE Bot Started*\n\n"
                "📊 Market-Specific Strategy:\n"
                "  • Whale wall detection\n"
                "  • Volume spike fading\n"
                "  • Time-based edges\n"
                "  • Spread optimization\n\n"
                f"🎯 Min Score: {config.MIN_SCORE}%\n"
                "Target: 12-18 signals/day"
            )
        except:
            pass
        
        # Build data
        logger.info("Building CoinDCX data... (60 seconds)")
        for _ in range(3):
            markets_data = await self.dcx.get_markets()
            if markets_data:
                inr = [m.get('symbol', '') or m.get('pair', '') 
                      for m in markets_data 
                      if any(c in (m.get('symbol', '') or m.get('pair', '')) 
                            for c in config.COINS)]
                await self.update_data(inr)
            await asyncio.sleep(20)
        
        logger.info("✅ Ready for CoinDCX edge signals!")
        
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
    bot = CoinDCXEdgeBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())