import requests
import pandas as pd
from datetime import datetime, timedelta

class AlternativeDataSource:
    """
    Alternative data sources if CoinDCX API fails
    Uses public APIs as fallback
    """
    
    @staticmethod
    def get_binance_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Get candles from Binance (more reliable)
        
        Args:
            symbol: e.g., 'BTCUSDT'
            interval: '5m', '15m', '1h'
            limit: Number of candles
        
        Returns:
            DataFrame with OHLCV
        """
        
        # Convert CoinDCX symbol to Binance format
        # B-BTC_USDT -> BTCUSDT
        if symbol.startswith('B-'):
            symbol = symbol.replace('B-', '').replace('_', '')
        
        url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Parse Binance format
            df = pd.DataFrame(data, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Keep only needed columns
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.astype({
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            })
            
            print(f"✅ Fetched {len(df)} candles from Binance for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Binance API error for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_coingecko_price(coin_id: str) -> float:
        """
        Get current price from CoinGecko
        
        Args:
            coin_id: 'bitcoin', 'ethereum', etc.
        
        Returns:
            Current price in USD
        """
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data[coin_id]['usd']
            
        except Exception as e:
            print(f"❌ CoinGecko error: {e}")
            return 0.0
    
    @staticmethod
    def convert_pair_to_binance(coindcx_pair: str) -> str:
        """
        Convert CoinDCX pair format to Binance
        
        Args:
            coindcx_pair: 'B-BTC_USDT'
        
        Returns:
            'BTCUSDT'
        """
        
        mapping = {
            'B-BTC_USDT': 'BTCUSDT',
            'B-ETH_USDT': 'ETHUSDT',
            'B-SOL_USDT': 'SOLUSDT',
            'B-MATIC_USDT': 'MATICUSDT',
            'B-ADA_USDT': 'ADAUSDT',
            'B-DOGE_USDT': 'DOGEUSDT'
        }
        
        return mapping.get(coindcx_pair, coindcx_pair.replace('B-', '').replace('_', ''))


# Usage example
if __name__ == "__main__":
    alt = AlternativeDataSource()
    
    # Test Binance
    candles = alt.get_binance_candles('BTCUSDT', '5m', 10)
    print(candles.tail())
    
    # Test CoinGecko
    btc_price = alt.get_coingecko_price('bitcoin')
    print(f"BTC Price: ${btc_price:,.2f}")