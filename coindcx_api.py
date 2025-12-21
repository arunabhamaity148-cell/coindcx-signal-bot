import hmac
import hashlib
import time
import json
import requests
import pandas as pd
from typing import Dict, Optional
from config import config

class CoinDCXAPI:
    """CoinDCX API with Binance Historical Data"""
    
    BASE_URL = config.COINDCX_BASE_URL
    
    @staticmethod
    def _generate_signature(secret: str, payload: str) -> str:
        """Generate HMAC SHA256 signature"""
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
    def get_candles(pair: str, interval: str, limit: int = 250) -> pd.DataFrame:
        """
        Fetch real Binance candles + adjust to CoinDCX price
        
        Returns:
            DataFrame with adjusted OHLCV candles
        """
        
        print(f"🔄 Fetching data for {pair}...")
        
        try:
            # Step 1: Get CoinDCX current price
            print(f"   Step 1: Getting CoinDCX price...")
            ticker_url = f"{CoinDCXAPI.BASE_URL}/exchange/ticker"
            response = requests.get(ticker_url, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            
            coindcx_price = None
            for ticker in tickers:
                if ticker.get('market') == pair:
                    coindcx_price = float(ticker.get('last_price', 0))
                    print(f"   ✅ CoinDCX price: ₹{coindcx_price:,.2f}")
                    break
            
            if not coindcx_price or coindcx_price <= 0:
                print(f"   ❌ CoinDCX price not found")
                return pd.DataFrame()
            
            # Step 2: Get Binance candles
            print(f"   Step 2: Fetching {limit} Binance candles...")
            binance_url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(binance_url, params=params, timeout=10)
            response.raise_for_status()
            binance_data = response.json()
            
            if not binance_data or len(binance_data) < 50:
                print(f"   ❌ Binance data insufficient ({len(binance_data) if binance_data else 0} candles)")
                return pd.DataFrame()
            
            print(f"   ✅ Got {len(binance_data)} Binance candles")
            
            # Step 3: Parse candles
            print(f"   Step 3: Parsing candles...")
            candles = []
            for candle in binance_data:
                candles.append({
                    'time': pd.Timestamp(int(candle[0]), unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            df = pd.DataFrame(candles)
            
            # Step 4: Adjust prices
            print(f"   Step 4: Adjusting to CoinDCX price...")
            binance_last = df['close'].iloc[-1]
            ratio = coindcx_price / binance_last
            
            df['open'] = df['open'] * ratio
            df['high'] = df['high'] * ratio
            df['low'] = df['low'] * ratio
            df['close'] = df['close'] * ratio
            
            final_price = df['close'].iloc[-1]
            
            print(f"   ✅ Adjusted: Binance ${binance_last:,.2f} → CoinDCX ₹{final_price:,.2f}")
            print(f"✅ {pair}: Ready with {len(df)} candles")
            
            return df
            
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout fetching data for {pair}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Network error for {pair}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"   ❌ Unexpected error for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    @staticmethod
    def get_ticker(pair: str) -> Optional[Dict]:
        """Get current ticker price"""
        try:
            endpoint = f"{CoinDCXAPI.BASE_URL}/exchange/ticker"
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
            
        except:
            return None
    
    @staticmethod
    def place_order(pair: str, side: str, price: float, quantity: float, leverage: int) -> Dict:
        """Place futures order"""
        
        if not config.AUTO_TRADE:
            print(f"\n{'='*50}")
            print(f"🔸 DRY RUN MODE - NO REAL ORDER")
            print(f"{'='*50}")
            print(f"Pair: {pair}")
            print(f"Side: {side.upper()}")
            print(f"Price: ₹{price:,.2f}")
            print(f"Quantity: {quantity}")
            print(f"Leverage: {leverage}x")
            print(f"{'='*50}\n")
            
            return {'status': 'dry_run', 'message': 'Order not placed'}
        
        # Real order code here...
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
            return response.json()
        except Exception as e:
            print(f"❌ Order failed: {e}")
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