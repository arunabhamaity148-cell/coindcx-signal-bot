"""
Data Collection Layer - Multi-Exchange Support
Collects real-time and historical data from Bybit, OKX, Binance
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        for name, settings in self.config.items():
            try:
                if name == 'bybit':
                    exchange = ccxt.bybit({
                        'apiKey': settings['api_key'],
                        'secret': settings['secret'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                elif name == 'okx':
                    exchange = ccxt.okx({
                        'apiKey': settings['api_key'],
                        'secret': settings['secret'],
                        'password': settings['password'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'swap'}
                    })
                elif name == 'binance':
                    exchange = ccxt.binance({
                        'apiKey': settings['api_key'],
                        'secret': settings['secret'],
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'}
                    })
                
                self.exchanges[name] = exchange
                logger.info(f"✅ {name.upper()} connected successfully")
            except Exception as e:
                logger.error(f"❌ Failed to connect {name}: {e}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, 
                     exchange_name: str = 'bybit', limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            exchange = self.exchanges[exchange_name]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✅ Fetched {len(df)} candles from {exchange_name}")
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching OHLCV: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                               years: int = 5, exchange_name: str = 'bybit') -> pd.DataFrame:
        """Fetch large historical dataset (5 years)"""
        logger.info(f"📊 Fetching {years} years of data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        all_data = []
        current_date = start_date
        exchange = self.exchanges[exchange_name]
        
        while current_date < end_date:
            try:
                since = int(current_date.timestamp() * 1000)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                
                logger.info(f"📈 Progress: {current_date.strftime('%Y-%m-%d')}")
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                break
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.drop_duplicates(subset='timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Total {len(df)} candles fetched")
        return df
    
    def fetch_orderbook(self, symbol: str, exchange_name: str = 'bybit', 
                        limit: int = 20) -> Dict:
        """Fetch orderbook data"""
        try:
            exchange = self.exchanges[exchange_name]
            orderbook = exchange.fetch_order_book(symbol, limit=limit)
            
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            bid_volume = sum([bid[1] for bid in bids])
            ask_volume = sum([ask[1] for ask in asks])
            
            return {
                'bids': bids,
                'asks': asks,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance': bid_volume / ask_volume if ask_volume > 0 else 0,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"❌ Orderbook error: {e}")
            return {}
    
    def fetch_funding_rate(self, symbol: str, exchange_name: str = 'bybit') -> float:
        """Fetch current funding rate"""
        try:
            exchange = self.exchanges[exchange_name]
            ticker = exchange.fetch_ticker(symbol)
            funding_rate = ticker.get('info', {}).get('fundingRate', 0)
            return float(funding_rate) if funding_rate else 0
        except Exception as e:
            logger.error(f"❌ Funding rate error: {e}")
            return 0
    
    def fetch_open_interest(self, symbol: str, exchange_name: str = 'bybit') -> float:
        """Fetch open interest - only for contract markets"""
        try:
            # Skip spot markets (no OI)
            if not symbol.endswith(':USDT'):
                logger.debug("OI skipped for spot %s", symbol)
                return 0.0
            
            exchange = self.exchanges[exchange_name]
            oi = exchange.fetch_open_interest(symbol)
            return float(oi.get('openInterest', 0))
        except Exception as e:
            logger.debug("OI error for %s: %s", symbol, e)
            return 0.0
    
    def fetch_liquidations(self, symbol: str, exchange_name: str = 'bybit') -> List:
        """Fetch recent liquidations (if available)"""
        try:
            logger.warning("Liquidation data not available via standard API")
            return []
        except Exception as e:
            logger.error(f"❌ Liquidation error: {e}")
            return []
    
    def detect_large_orders(self, symbol: str, threshold: float = 100000, 
                            exchange_name: str = 'bybit') -> List[Dict]:
        """Detect large orders in orderbook"""
        try:
            orderbook = self.fetch_orderbook(symbol, exchange_name, limit=50)
            large_orders = []
            
            for bid in orderbook.get('bids', []):
                price, size = bid[0], bid[1]
                value = price * size
                if value >= threshold:
                    large_orders.append({
                        'side': 'bid',
                        'price': price,
                        'size': size,
                        'value': value
                    })
            
            for ask in orderbook.get('asks', []):
                price, size = ask[0], ask[1]
                value = price * size
                if value >= threshold:
                    large_orders.append({
                        'side': 'ask',
                        'price': price,
                        'size': size,
                        'value': value
                    })
            
            return large_orders
        except Exception as e:
            logger.error(f"❌ Large order detection error: {e}")
            return []
    
    def fetch_cross_exchange_prices(self, symbol: str) -> Dict[str, float]:
        """Get current price from multiple exchanges"""
        prices = {}
        for name, exchange in self.exchanges.items():
            try:
                ticker = exchange.fetch_ticker(symbol)
                prices[name] = ticker['last']
            except:
                pass
        return prices
    
    def calculate_price_divergence(self, symbol: str) -> Dict:
        """Calculate price divergence across exchanges"""
        prices = self.fetch_cross_exchange_prices(symbol)
        
        if len(prices) < 2:
            return {'divergence': 0, 'opportunity': False}
        
        max_price = max(prices.values())
        min_price = min(prices.values())
        divergence = (max_price - min_price) / min_price
        
        return {
            'prices': prices,
            'divergence': divergence,
            'opportunity': divergence > 0.003,  # 0.3%
            'buy_exchange': min(prices, key=prices.get),
            'sell_exchange': max(prices, key=prices.get)
        }


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    from config.settings import EXCHANGES
    
    collector = DataCollector(EXCHANGES)
    
    # Test data collection
    df = collector.fetch_ohlcv('BTC/USDT:USDT', '15m', 'bybit', limit=100)
    print(df.head())
    
    # Test orderbook
    ob = collector.fetch_orderbook('BTC/USDT:USDT', 'bybit')
    print(f"Orderbook imbalance: {ob.get('imbalance', 0):.2f}")
    
    # Test cross-exchange
    divergence = collector.calculate_price_divergence('BTC/USDT:USDT')
    print(f"Price divergence: {divergence}")
