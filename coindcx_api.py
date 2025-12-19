import hmac
import hashlib
import time
import json
import requests
import pandas as pd
from typing import Dict, Optional
from config import config

class CoinDCXAPI:
    """CoinDCX Futures API Handler"""
    
    BASE_URL = config.COINDCX_BASE_URL
    
    @staticmethod
    def _generate_signature(secret: str, payload: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def _get_headers(payload: str) -> Dict:
        """Generate headers with signature"""
        signature = CoinDCXAPI._generate_signature(config.COINDCX_SECRET, payload)
        
        return {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': config.COINDCX_API_KEY,
            'X-AUTH-SIGNATURE': signature
        }
    
    @staticmethod
    def get_candles(pair: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical candle data
        
        Args:
            pair: Trading pair (e.g., 'F-BTC_INR')
            interval: Timeframe ('5m', '15m', '1h', etc.)
            limit: Number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{CoinDCXAPI.BASE_URL}/market_data/candles"
        
        params = {
            'pair': pair,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"⚠️ No candle data for {pair}")
                return pd.DataFrame()
            
            # Parse candle data
            df = pd.DataFrame(
                data,
                columns=['time', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.astype({
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            })
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching candles for {pair}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_ticker(pair: str) -> Optional[Dict]:
        """
        Get current ticker price
        
        Args:
            pair: Trading pair
        
        Returns:
            Dict with bid, ask, last price
        """
        endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/ticker"
        
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            
            for ticker in tickers:
                if ticker.get('market') == pair:
                    return {
                        'bid': float(ticker.get('bid', 0)),
                        'ask': float(ticker.get('ask', 0)),
                        'last': float(ticker.get('last_price', 0))
                    }
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching ticker for {pair}: {e}")
            return None
    
    @staticmethod
    def place_order(pair: str, side: str, price: float, quantity: float, leverage: int) -> Dict:
        """
        Place futures order
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            price: Entry price
            quantity: Order quantity
            leverage: Leverage (10-15x)
        
        Returns:
            Order response
        """
        
        # DRY RUN mode check
        if not config.AUTO_TRADE:
            print(f"\n{'='*50}")
            print(f"🔸 DRY RUN MODE - NO REAL ORDER PLACED")
            print(f"{'='*50}")
            print(f"Pair: {pair}")
            print(f"Side: {side.upper()}")
            print(f"Price: ₹{price}")
            print(f"Quantity: {quantity}")
            print(f"Leverage: {leverage}x")
            print(f"{'='*50}\n")
            
            return {
                'status': 'dry_run',
                'message': 'Order not placed (AUTO_TRADE=false)'
            }
        
        # REAL ORDER placement
        endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/v1/orders/create"
        timestamp = int(time.time() * 1000)
        
        body = {
            "side": side.lower(),
            "order_type": "limit_order",
            "market": pair,
            "price_per_unit": price,
            "total_quantity": quantity,
            "leverage": leverage,
            "timestamp": timestamp
        }
        
        payload = json.dumps(body, separators=(',', ':'))
        headers = CoinDCXAPI._get_headers(payload)
        
        try:
            response = requests.post(endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Order placed: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Order placement failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def get_account_balance() -> Optional[Dict]:
        """Get account balance and margin info"""
        endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/v1/users/balances"
        timestamp = int(time.time() * 1000)
        
        body = {"timestamp": timestamp}
        payload = json.dumps(body, separators=(',', ':'))
        headers = CoinDCXAPI._get_headers(payload)
        
        try:
            response = requests.post(endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching balance: {e}")
            return None
    
    @staticmethod
    def get_open_positions() -> list:
        """Get all open futures positions"""
        endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/v1/futures/positions"
        timestamp = int(time.time() * 1000)
        
        body = {"timestamp": timestamp}
        payload = json.dumps(body, separators=(',', ':'))
        headers = CoinDCXAPI._get_headers(payload)
        
        try:
            response = requests.post(endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching positions: {e}")
            return []
    
    @staticmethod
    def cancel_order(order_id: str) -> Dict:
        """Cancel an open order"""
        endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/v1/orders/cancel"
        timestamp = int(time.time() * 1000)
        
        body = {
            "id": order_id,
            "timestamp": timestamp
        }
        
        payload = json.dumps(body, separators=(',', ':'))
        headers = CoinDCXAPI._get_headers(payload)
        
        try:
            response = requests.post(endpoint, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error canceling order: {e}")
            return {'status': 'error', 'message': str(e)}
    
    @staticmethod
    def test_connection() -> bool:
        """Test API connection"""
        try:
            endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/ticker"
            response = requests.get(endpoint, timeout=5)
            response.raise_for_status()
            print("✅ CoinDCX API connection successful")
            return True
        except:
            print("❌ CoinDCX API connection failed")
            return False