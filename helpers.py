"""
Helper functions - FIXED for CoinDCX
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

# ==================== COINDCX API (FIXED) ====================
class CoinDCXAPI:
    """CoinDCX Exchange API - FIXED"""
    
    BASE_URL = "https://api.coindcx.com"
    PUBLIC_URL = "https://public.coindcx.com"
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _generate_signature(self, payload: dict) -> str:
        json_payload = json.dumps(payload, separators=(',', ':'))
        signature = hmac.new(
            self.secret.encode(),
            json_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _request(self, method: str, endpoint: str, payload: dict = None, authenticated: bool = False):
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
        """Get all markets - FIXED LOGIC"""
        try:
            data = await self._request('GET', '/market_data/markets')
            
            logger.info(f"Markets API response type: {type(data)}")
            
            if not data:
                logger.error("Empty response from markets API")
                return []
            
            # Sample first market to see structure
            if len(data) > 0:
                logger.info(f"Sample market data: {data[0]}")
            
            logger.info(f"Successfully fetched {len(data)} markets from API")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            return []
    
    async def get_ticker(self, market: str) -> Dict:
        """Get ticker - FIXED"""
        try:
            data = await self._request('GET', '/market_data/ticker')
            
            if not data:
                return {}
            
            # CoinDCX returns array of tickers
            for ticker in data:
                # Match market symbol
                if ticker.get('market') == market or ticker.get('pair') == market:
                    return ticker
            
            logger.warning(f"Ticker not found for {market}")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {market}: {e}")
            return {}
    
    async def get_candles(self, market: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Get candles - FIXED"""
        try:
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '1d': '1d',
                '1w': '1w',
                '1M': '1M'
            }
            
            dcx_interval = interval_map.get(interval, '15m')
            
            # CoinDCX candles endpoint
            payload = {
                'pair': market,
                'interval': dcx_interval,
                'limit': limit
            }
            
            data = await self._request('GET', '/market_data/candles', payload)
            
            if not data or len(data) == 0:
                logger.warning(f"No candle data for {market}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Check columns
            logger.debug(f"Candle columns: {df.columns.tolist()}")
            
            # Rename columns based on what CoinDCX returns
            column_mapping = {
                'time': 'timestamp',
                't': 'timestamp',
                'open': 'open',
                'o': 'open',
                'high': 'high',
                'h': 'high',
                'low': 'low',
                'l': 'low',
                'close': 'close',
                'c': 'close',
                'volume': 'volume',
                'v': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                logger.error(f"Missing required columns. Have: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN
            df = df.dropna()
            
            return df.sort_values('timestamp').reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Error fetching candles for {market}: {e}")
            return pd.DataFrame()
    
    async def get_orderbook(self, market: str, depth: int = 20) -> Dict:
        """Get orderbook - FIXED"""
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
            
            # CoinDCX format
            if 'bids' in data:
                if isinstance(data['bids'], dict):
                    # Format: {"price": "quantity"}
                    for price, qty in list(data['bids'].items())[:depth]:
                        bids.append([float(price), float(qty)])
                elif isinstance(data['bids'], list):
                    # Format: [["price", "quantity"], ...]
                    for item in data['bids'][:depth]:
                        if isinstance(item, dict):
                            bids.append([float(item.get('price', 0)), float(item.get('quantity', 0))])
                        elif isinstance(item, list) and len(item) >= 2:
                            bids.append([float(item[0]), float(item[1])])
            
            if 'asks' in data:
                if isinstance(data['asks'], dict):
                    for price, qty in list(data['asks'].items())[:depth]:
                        asks.append([float(price), float(qty)])
                elif isinstance(data['asks'], list):
                    for item in data['asks'][:depth]:
                        if isinstance(item, dict):
                            asks.append([float(item.get('price', 0)), float(item.get('quantity', 0))])
                        elif isinstance(item, list) and len(item) >= 2:
                            asks.append([float(item[0]), float(item[1])])
            
            return {
                'bids': bids,
                'asks': asks
            }
        
        except Exception as e:
            logger.error(f"Error fetching orderbook for {market}: {e}")
            return {}
    
    async def close(self):
        if self.session:
            await self.session.close()

# ==================== TELEGRAM (SAME) ====================
class TelegramNotifier:
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
    
    async def send_message(self, message: str):
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def send_signal(self, signal: Dict):
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}[signal['confidence']]
        
        message = f"""🚨 *{signal['mode']} MODE SIGNAL* 🚨

📌 *Pair:* {signal['market']}
📊 *TF:* {signal['timeframe']}
{side_emoji} *Side:* *{signal['side']}*

💰 *Entry:* ₹{signal['entry']:,.2f}
🛑 *SL:* ₹{signal['sl']:,.2f}
🎯 *TP:* ₹{signal['tp']:,.2f}

📐 *R:R:* 1:{signal['rr_ratio']:.1f}
🧠 *Score:* {signal['logic_score']}%
{conf_emoji} *Confidence:* {signal['confidence']}

⏱️ *Mode:* {signal['mode']}
⚠️ *Trade manually*

🕐 _{datetime.now().strftime("%d-%b %I:%M %p")}_
"""
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Signal sent: {signal['market']}")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

# ==================== DATABASE (SAME) ====================
class DatabaseManager:
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        except Exception as e:
            logger.error(f"❌ DB save error: {e}")

# ==================== TECHNICAL INDICATORS (SAME) ====================
class TechnicalIndicators:
    
    @staticmethod
    def ema(df: pd.DataFrame, period: int, column: str = 'close'):
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(df: pd.DataFrame):
        ema12 = TechnicalIndicators.ema(df, 12)
        ema26 = TechnicalIndicators.ema(df, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2):
        sma = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, sma, lower
    
    @staticmethod
    def vwap(df: pd.DataFrame):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def obv(df: pd.DataFrame):
        obv_values = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_values.append(obv_values[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_values.append(obv_values[-1] - df['volume'].iloc[i])
            else:
                obv_values.append(obv_values[-1])
        return pd.Series(obv_values, index=df.index)