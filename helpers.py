"""
Helper functions for CoinDCX API, Telegram, and Database operations
"""

import hmac
import hashlib
import json
import time
import aiohttp
import telegram
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ==================== COINDCX API ====================
class CoinDCXAPI:
    """CoinDCX Exchange API Wrapper"""
    
    BASE_URL = "https://api.coindcx.com"
    PUBLIC_URL = "https://public.coindcx.com"
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _generate_signature(self, payload: dict) -> str:
        """Generate HMAC signature for authenticated requests"""
        json_payload = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(
            self.secret.encode(),
            json_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _request(self, method: str, endpoint: str, payload: dict = None, authenticated: bool = False):
        """Make HTTP request to CoinDCX API"""
        session = await self._get_session()
        
        if authenticated:
            url = f"{self.BASE_URL}{endpoint}"
            timestamp = int(time.time() * 1000)
            
            if payload is None:
                payload = {}
            
            payload['timestamp'] = timestamp
            
            headers = {
                'Content-Type': 'application/json',
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': self._generate_signature(payload)
            }
            
            async with session.post(url, json=payload, headers=headers) as response:
                return await response.json()
        else:
            url = f"{self.PUBLIC_URL}{endpoint}"
            
            if method == 'GET':
                async with session.get(url, params=payload) as response:
                    return await response.json()
            else:
                async with session.post(url, json=payload) as response:
                    return await response.json()
    
    async def get_markets(self) -> List[Dict]:
        """Get all available markets"""
        try:
            data = await self._request('GET', '/market_data/markets')
            return data
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    async def get_ticker(self, market: str) -> Dict:
        """Get current ticker for a market"""
        try:
            data = await self._request('GET', '/market_data/ticker')
            
            # Find the specific market
            for ticker in data:
                if ticker.get('market') == market:
                    return ticker
            
            return {}
        except Exception as e:
            logger.error(f"Error fetching ticker for {market}: {e}")
            return {}
    
    async def get_candles(self, market: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Get candlestick data
        interval: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '1d', '1w', '1M'
        """
        try:
            # CoinDCX interval mapping
            interval_map = {
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '1d': '1d'
            }
            
            dcx_interval = interval_map.get(interval, '15m')
            
            payload = {
                'pair': market,
                'interval': dcx_interval,
                'limit': limit
            }
            
            data = await self._request('GET', '/market_data/candles', payload)
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to standard OHLCV format
            if 'time' in df.columns:
                df = df.rename(columns={
                    'time': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert string prices to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.sort_values('timestamp').reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Error fetching candles for {market}: {e}")
            return pd.DataFrame()
    
    async def get_orderbook(self, market: str, depth: int = 20) -> Dict:
        """Get order book data"""
        try:
            payload = {
                'pair': market
            }
            
            data = await self._request('GET', '/market_data/orderbook', payload)
            
            if not data:
                return {}
            
            # Parse orderbook
            bids = []
            asks = []
            
            if 'bids' in data:
                for bid in data['bids'][:depth]:
                    bids.append([float(bid['price']), float(bid['quantity'])])
            
            if 'asks' in data:
                for ask in data['asks'][:depth]:
                    asks.append([float(ask['price']), float(ask['quantity'])])
            
            return {
                'bids': bids,
                'asks': asks
            }
        
        except Exception as e:
            logger.error(f"Error fetching orderbook for {market}: {e}")
            return {}
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    """Send trading signals to Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
    
    async def send_message(self, message: str):
        """Send plain text message"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    async def send_signal(self, signal: Dict):
        """Send formatted trading signal"""
        
        # Determine emoji based on side and confidence
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        confidence_emoji = "🔥" if signal['confidence'] == "HIGH" else "⚡" if signal['confidence'] == "MEDIUM" else "⚠️"
        
        message = f"""🚨 *{signal['mode']} MODE SIGNAL* 🚨

📌 *Pair:* {signal['market']}
📊 *TF:* {signal['timeframe']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Logic Score:* {signal['logic_score']}%
{confidence_emoji} *Confidence:* {signal['confidence']}

⏱️ *Mode:* {signal['mode']}
⚠️ *Trade manually on CoinDCX*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Signal sent: {signal['market']} {signal['side']}")
        except Exception as e:
            logger.error(f"❌ Error sending signal: {e}")

# ==================== DATABASE MANAGER ====================
class DatabaseManager:
    """SQLite database for storing signals and trade history"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                market TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                side TEXT NOT NULL,
                entry REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                rr_ratio REAL,
                logic_score INTEGER,
                confidence TEXT,
                mode TEXT,
                details TEXT
            )
        ''')
        
        # Trade history table (for manual tracking)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                entry_time DATETIME,
                exit_time DATETIME,
                market TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                sl_price REAL,
                tp_price REAL,
                pnl_inr REAL,
                pnl_percent REAL,
                status TEXT,
                notes TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"📁 Database initialized: {self.db_path}")
    
    def save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (
                    market, timeframe, side, entry, sl, tp, 
                    rr_ratio, logic_score, confidence, mode, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['market'],
                signal['timeframe'],
                signal['side'],
                signal['entry'],
                signal['sl'],
                signal['tp'],
                signal['rr_ratio'],
                signal['logic_score'],
                signal['confidence'],
                signal['mode'],
                json.dumps(signal.get('details', {}))
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"💾 Signal saved to database: {signal['market']}")
        
        except Exception as e:
            logger.error(f"❌ Error saving signal: {e}")
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM signals 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"❌ Error fetching signals: {e}")
            return []
    
    def save_trade(self, trade: Dict):
        """Save executed trade"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    signal_id, entry_time, exit_time, market, side,
                    entry_price, exit_price, sl_price, tp_price,
                    pnl_inr, pnl_percent, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('signal_id'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade['market'],
                trade['side'],
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('sl_price'),
                trade.get('tp_price'),
                trade.get('pnl_inr'),
                trade.get('pnl_percent'),
                trade.get('status', 'OPEN'),
                trade.get('notes', '')
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"💾 Trade saved: {trade['market']}")
        
        except Exception as e:
            logger.error(f"❌ Error saving trade: {e}")
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total signals
            cursor.execute('SELECT COUNT(*) FROM signals')
            total_signals = cursor.fetchone()[0]
            
            # Total trades
            cursor.execute('SELECT COUNT(*) FROM trades')
            total_trades = cursor.fetchone()[0]
            
            # Win rate
            cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl_inr > 0')
            winning_trades = cursor.fetchone()[0]
            
            # Total PNL
            cursor.execute('SELECT SUM(pnl_inr) FROM trades')
            total_pnl = cursor.fetchone()[0] or 0
            
            conn.close()
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_signals': total_signals,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl_inr': round(total_pnl, 2)
            }
        
        except Exception as e:
            logger.error(f"❌ Error fetching stats: {e}")
            return {}

# ==================== TECHNICAL INDICATORS ====================
class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int, column: str = 'close'):
        """Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14):
        """Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(df: pd.DataFrame):
        """MACD Indicator"""
        ema12 = TechnicalIndicators.ema(df, 12)
        ema26 = TechnicalIndicators.ema(df, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14):
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2):
        """Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    @staticmethod
    def vwap(df: pd.DataFrame):
        """Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def obv(df: pd.DataFrame):
        """On Balance Volume"""
        obv_values = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_values.append(obv_values[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_values.append(obv_values[-1] - df['volume'].iloc[i])
            else:
                obv_values.append(obv_values[-1])
        return pd.Series(obv_values, index=df.index)