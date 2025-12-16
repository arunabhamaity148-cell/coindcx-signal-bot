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
        try:
            data = await self._request('GET', '/market_data/markets')
            if not isinstance(data, dict) or 'markets' not in data:
                logger.error("❌ Markets response invalid or missing 'markets' key")
                return []
            markets = data['markets']
            if not isinstance(markets, list):
                logger.error("❌ 'markets' is not a list")
                return []
            inr_pairs = []
            for m in markets:
                if isinstance(m, dict) and m.get('symbol', '').endswith('INR'):
                    inr_pairs.append(m)
            return inr_pairs
        except Exception as e:
            logger.error(f"❌ Error fetching markets: {e}")
            return []

    async def get_ticker(self, market: str) -> Dict:
        try:
            data = await self._request('GET', '/market_data/ticker')
            if isinstance(data, list):
                for ticker in data:
                    if isinstance(ticker, dict) and ticker.get('market') == market:
                        return ticker
            return {}
        except Exception as e:
            logger.error(f"❌ Error fetching ticker for {market}: {e}")
            return {}

    async def get_candles(self, market: str, interval: str, limit: int = 500) -> pd.DataFrame:
        try:
            interval_map = {'5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
            dcx_interval = interval_map.get(interval, '15m')
            payload = {'pair': market, 'interval': dcx_interval, 'limit': limit}
            data = await self._request('GET', '/market_data/candles', payload)
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            if 'time' in df.columns:
                df = df.rename(columns={
                    'time': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            logger.error(f"❌ Error fetching candles for {market}: {e}")
            return pd.DataFrame()

    async def get_orderbook(self, market: str, depth: int = 20) -> Dict:
        try:
            payload = {'pair': market}
            data = await self._request('GET', '/market_data/orderbook', payload)
            if not data:
                return {}
            bids = []
            asks = []
            if 'bids' in data:
                for bid in data['bids'][:depth]:
                    bids.append([float(bid['price']), float(bid['quantity'])])
            if 'asks' in data:
                for ask in data['asks'][:depth]:
                    asks.append([float(ask['price']), float(ask['quantity'])])
            return {'bids': bids, 'asks': asks}
        except Exception as e:
            logger.error(f"❌ Error fetching orderbook for {market}: {e}")
            return {}

    async def close(self):
        if self.session:
            await self.session.close()

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id

    async def send_message(self, message: str):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")

# ==================== DATABASE MANAGER ====================
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
            logger.info(f"💾 Signal saved to database: {signal['market']}")
        except Exception as e:
            logger.error(f"❌ Error saving signal: {e}")
