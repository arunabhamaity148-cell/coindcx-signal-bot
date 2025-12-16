"""
CoinDCX API - COMPLETE FIX
Direct endpoint testing and fallback strategies
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

# ==================== COINDCX API FIXED ====================
class CoinDCXAPI:
    """CoinDCX API with multiple endpoint strategies"""
    
    BASE_URL = "https://api.coindcx.com"
    PUBLIC_URL = "https://public.coindcx.com"
    
    # Alternative endpoint for markets
    ALT_MARKETS_URL = "https://api.coindcx.com/exchange/ticker"
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
        self.session = None
        self.markets_cache = []  # Cache markets
        self.last_cache_time = 0
    
    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
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
            
            try:
                if method == 'GET':
                    async with session.get(url, params=payload) as response:
                        text = await response.text()
                        logger.debug(f"Response status: {response.status}")
                        logger.debug(f"Response text: {text[:500]}")  # First 500 chars
                        return await response.json()
                else:
                    async with session.post(url, json=payload) as response:
                        return await response.json()
            except Exception as e:
                logger.error(f"Request error for {url}: {e}")
                return None
    
    async def get_markets_v1(self) -> List[Dict]:
        """Strategy 1: /market_data/markets"""
        try:
            data = await self._request('GET', '/market_data/markets')
            
            if isinstance(data, dict):
                if 'error' in data or 'message' in data:
                    logger.error(f"API Error: {data}")
                    return []
                
                if 'markets' in data:
                    data = data['markets']
                elif 'data' in data:
                    data = data['data']
            
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Strategy 1 success: {len(data)} markets")
                return data
            
            return []
            
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")
            return []
    
    async def get_markets_v2(self) -> List[Dict]:
        """Strategy 2: Alternative ticker endpoint"""
        try:
            session = await self._get_session()
            
            async with session.get(self.ALT_MARKETS_URL) as response:
                data = await response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"✅ Strategy 2 success: {len(data)} tickers")
                    
                    # Convert ticker format to market format
                    markets = []
                    for ticker in data:
                        markets.append({
                            'symbol': ticker.get('market', ''),
                            'pair': ticker.get('market', ''),
                            'base_currency': ticker.get('market', '').split('_')[0] if '_' in ticker.get('market', '') else '',
                            'target_currency': ticker.get('market', '').split('_')[1] if '_' in ticker.get('market', '') else ''
                        })
                    
                    return markets
                
                return []
        
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")
            return []
    
    async def get_markets_v3(self) -> List[Dict]:
        """Strategy 3: Hardcoded popular INR markets"""
        logger.info("📋 Using hardcoded popular markets")
        
        popular_coins = [
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC',
            'DOT', 'AVAX', 'LINK', 'LTC', 'ATOM', 'UNI', 'ETC',
            'NEAR', 'FTM', 'SAND', 'MANA', 'TRX', 'AAVE'
        ]
        
        # Test which format CoinDCX uses
        formats = [
            lambda c: f"{c}INR",
            lambda c: f"{c}INRT",
            lambda c: f"I-{c}_INRT",
            lambda c: f"B-{c}_INRT",
            lambda c: f"{c}_INR",
        ]
        
        markets = []
        for coin in popular_coins:
            for fmt in formats:
                symbol = fmt(coin)
                markets.append({
                    'symbol': symbol,
                    'pair': symbol,
                    'base_currency': coin,
                    'target_currency': 'INR'
                })
        
        logger.info(f"📋 Created {len(markets)} market variations")
        return markets
    
    async def get_markets(self) -> List[Dict]:
        """Get markets with multiple strategies"""
        
        # Use cache if recent (5 minutes)
        current_time = time.time()
        if self.markets_cache and (current_time - self.last_cache_time) < 300:
            logger.info(f"📦 Using cached markets: {len(self.markets_cache)}")
            return self.markets_cache
        
        logger.info("🔍 Fetching markets with multiple strategies...")
        
        # Try Strategy 1
        markets = await self.get_markets_v1()
        if markets:
            self.markets_cache = markets
            self.last_cache_time = current_time
            return markets
        
        # Try Strategy 2
        markets = await self.get_markets_v2()
        if markets:
            self.markets_cache = markets
            self.last_cache_time = current_time
            return markets
        
        # Fallback Strategy 3
        logger.warning("⚠️ Using fallback hardcoded markets")
        markets = await self.get_markets_v3()
        self.markets_cache = markets
        self.last_cache_time = current_time
        return markets
    
    async def get_ticker(self, market: str) -> Dict:
        """Get ticker with error handling"""
        try:
            # Try public ticker endpoint
            session = await self._get_session()
            
            async with session.get(f"{self.PUBLIC_URL}/market_data/ticker") as response:
                data = await response.json()
                
                if isinstance(data, list):
                    for ticker in data:
                        ticker_market = ticker.get('market', '') or ticker.get('pair', '')
                        if ticker_market == market:
                            return ticker
                
                # Fallback: return basic structure
                logger.warning(f"Ticker not found for {market}, using fallback")
                return {'market': market, 'last_price': 0}
        
        except Exception as e:
            logger.error(f"Ticker error {market}: {e}")
            return {'market': market, 'last_price': 0}
    
    async def get_candles(self, market: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Get candles with fallback"""
        try:
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d'
            }
            
            dcx_interval = interval_map.get(interval, '15m')
            
            # Try to get candles
            payload = {
                'pair': market,
                'interval': dcx_interval,
                'limit': limit
            }
            
            data = await self._request('GET', '/market_data/candles', payload)
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.warning(f"No candles for {market}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Try different column name variations
            column_mappings = [
                {'time': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'},
                {'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'},
                {'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'vol': 'volume'}
            ]
            
            for mapping in column_mappings:
                try:
                    df_renamed = df.rename(columns=mapping)
                    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                    if all(col in df_renamed.columns for col in required):
                        df = df_renamed
                        break
                except:
                    continue
            
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                logger.error(f"Missing columns. Have: {df.columns.tolist()}")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Only {len(df)} candles for {market}")
                return pd.DataFrame()
            
            return df.sort_values('timestamp').reset_index(drop=True)
        
        except Exception as e:
            logger.error(f"Candles error {market}: {e}")
            return pd.DataFrame()
    
    async def get_orderbook(self, market: str, depth: int = 20) -> Dict:
        """Get orderbook"""
        try:
            payload = {'pair': market}
            data = await self._request('GET', '/market_data/orderbook', payload)
            
            if not data:
                return {'bids': [], 'asks': []}
            
            bids = []
            asks = []
            
            if 'bids' in data:
                if isinstance(data['bids'], dict):
                    for price, qty in list(data['bids'].items())[:depth]:
                        bids.append([float(price), float(qty)])
                elif isinstance(data['bids'], list):
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
            
            return {'bids': bids, 'asks': asks}
        
        except Exception as e:
            logger.error(f"Orderbook error {market}: {e}")
            return {'bids': [], 'asks': []}
    
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
            logger.error(f"Telegram error: {e}")
    
    async def send_signal(self, signal: Dict):
        side_emoji = "📈" if signal['side'] == "BUY" else "📉"
        conf_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}[signal['confidence']]
        
        message = f"""🚨 *{signal['mode']} SIGNAL* 🚨

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
⚠️ *Trade CoinDCX manually*

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
            logger.error(f"Telegram error: {e}")

# ==================== DATABASE ====================
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
        
        conn.commit()
        conn.close()
        logger.info(f"📁 Database ready: {self.db_path}")
    
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
            logger.error(f"DB error: {e}")

# ==================== TECHNICAL INDICATORS ====================
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