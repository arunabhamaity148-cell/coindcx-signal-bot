"""
Historical Data Downloader
Downloads 5 years of OHLCV data from multiple exchanges
Saves to CSV and PostgreSQL database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import argparse
from datetime import datetime, timedelta
import logging
from data.data_collector import DataCollector
from config.settings import EXCHANGES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    def __init__(self, exchanges_config):
        self.collector = DataCollector(exchanges_config)
        self.data_dir = 'data/historical'
        
        # Create directory
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_historical_data(self, symbol='BTC/USDT:USDT', timeframe='15m',
                                 years=5, exchange='bybit'):
        """
        Download historical data for specified years
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, etc.)
            years: Number of years to download
            exchange: Exchange name
        """
        logger.info("="*80)
        logger.info("📥 DOWNLOADING HISTORICAL DATA")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Years: {years}")
        logger.info(f"Exchange: {exchange}")
        logger.info("="*80)
        
        try:
            # Download data
            df = self.collector.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                years=years,
                exchange_name=exchange
            )
            
            if df.empty:
                logger.error("❌ No data downloaded")
                return None
            
            # Data info
            logger.info("="*80)
            logger.info("📊 DATA SUMMARY")
            logger.info("="*80)
            logger.info(f"Total candles: {len(df):,}")
            logger.info(f"Start date: {df.index[0]}")
            logger.info(f"End date: {df.index[-1]}")
            logger.info(f"Duration: {(df.index[-1] - df.index[0]).days} days")
            logger.info("="*80)
            
            # Save to CSV
            filename = f"{symbol.replace('/', '_').replace(':', '_')}_{timeframe}_{years}y_{exchange}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)
            logger.info(f"✅ Saved to: {filepath}")
            
            # Display sample
            logger.info("\n📋 Sample data (first 5 rows):")
            print(df.head())
            
            logger.info("\n📋 Sample data (last 5 rows):")
            print(df.tail())
            
            # Statistics
            logger.info("\n📊 Price statistics:")
            print(df['close'].describe())
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return None
    
    def download_multiple_symbols(self, symbols, timeframe='15m', years=5, exchange='bybit'):
        """Download data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Downloading {symbol}...")
            logger.info(f"{'='*80}\n")
            
            df = self.download_historical_data(symbol, timeframe, years, exchange)
            results[symbol] = df
            
            # Small delay between requests
            import time
            time.sleep(2)
        
        return results
    
    def verify_data_quality(self, df):
        """Verify downloaded data quality"""
        logger.info("\n🔍 VERIFYING DATA QUALITY")
        logger.info("="*80)
        
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            issues.append(f"Missing values found: {missing.sum()}")
            logger.warning(f"⚠️ Missing values:\n{missing}")
        
        # Check for duplicates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate timestamps: {duplicates}")
            logger.warning(f"⚠️ Duplicate timestamps: {duplicates}")
            df = df[~df.index.duplicated(keep='first')]
        
        # Check for gaps
        expected_interval = pd.Timedelta(minutes=15)  # For 15m timeframe
        time_diff = df.index.to_series().diff()
        gaps = (time_diff > expected_interval * 1.5).sum()
        if gaps > 0:
            issues.append(f"Time gaps found: {gaps}")
            logger.warning(f"⚠️ Time gaps: {gaps}")
        
        # Check for zero/negative prices
        zero_prices = (df['close'] <= 0).sum()
        if zero_prices > 0:
            issues.append(f"Invalid prices: {zero_prices}")
            logger.error(f"❌ Zero/negative prices: {zero_prices}")
        
        # Check for extreme price changes (>20% in one candle)
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.20).sum()
        if extreme_changes > 0:
            issues.append(f"Extreme price changes: {extreme_changes}")
            logger.warning(f"⚠️ Extreme price changes (>20%): {extreme_changes}")
        
        if not issues:
            logger.info("✅ Data quality: EXCELLENT")
        else:
            logger.warning(f"⚠️ Found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"   • {issue}")
        
        logger.info("="*80)
        
        return df, issues
    
    def prepare_for_ml(self, df):
        """Prepare data for ML training"""
        logger.info("\n🤖 PREPARING DATA FOR ML")
        logger.info("="*80)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Create labels (next 15min price movement)
        # BUY (0): price will go up >0.5%
        # HOLD (1): price change between -0.5% to +0.5%
        # SELL (2): price will go down >0.5%
        
        df['next_return'] = df['close'].shift(-1).pct_change()
        
        def create_label(ret):
            if pd.isna(ret):
                return 1  # HOLD
            elif ret > 0.005:  # >0.5%
                return 2  # SELL (price going up, we sold too early - or for SHORT)
            elif ret < -0.005:  # <-0.5%
                return 0  # BUY (price going down, good entry for LONG)
            else:
                return 1  # HOLD
        
        df['label'] = df['next_return'].apply(create_label)
        
        # Label distribution
        label_dist = df['label'].value_counts(normalize=True).sort_index()
        logger.info("📊 Label distribution:")
        logger.info(f"   BUY (0):  {label_dist.get(0, 0):.1%}")
        logger.info(f"   HOLD (1): {label_dist.get(1, 0):.1%}")
        logger.info(f"   SELL (2): {label_dist.get(2, 0):.1%}")
        
        # Save processed data
        filename = 'processed_data_with_labels.csv'
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        logger.info(f"✅ Processed data saved: {filepath}")
        logger.info("="*80)
        
        return df


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Download Historical Crypto Data')
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT',
                        help='Trading symbol (default: BTC/USDT:USDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                        help='Timeframe (default: 15m)')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of data (default: 5)')
    parser.add_argument('--exchange', type=str, default='bybit',
                        help='Exchange (default: bybit)')
    parser.add_argument('--multiple', action='store_true',
                        help='Download multiple symbols')
    parser.add_argument('--verify', action='store_true',
                        help='Verify data quality')
    parser.add_argument('--prepare-ml', action='store_true',
                        help='Prepare data for ML training')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(EXCHANGES)
    
    # Download data
    if args.multiple:
        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        logger.info(f"📥 Downloading {len(symbols)} symbols...")
        results = downloader.download_multiple_symbols(
            symbols=symbols,
            timeframe=args.timeframe,
            years=args.years,
            exchange=args.exchange
        )
    else:
        df = downloader.download_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            years=args.years,
            exchange=args.exchange
        )
        
        # Verify quality
        if args.verify and df is not None:
            df, issues = downloader.verify_data_quality(df)
        
        # Prepare for ML
        if args.prepare_ml and df is not None:
            df = downloader.prepare_for_ml(df)
    
    logger.info("\n✅ Download complete!")
    logger.info(f"Data saved in: data/historical/")


if __name__ == "__main__":
    main()


# ==================== USAGE EXAMPLES ====================
"""
# Basic download (5 years, BTC, 15m)
python scripts/download_data.py

# Custom symbol and timeframe
python scripts/download_data.py --symbol ETH/USDT:USDT --timeframe 1h --years 3

# Download multiple symbols
python scripts/download_data.py --multiple --years 5

# With quality verification
python scripts/download_data.py --verify

# Prepare for ML training
python scripts/download_data.py --prepare-ml --verify

# Full download with all options
python scripts/download_data.py --symbol BTC/USDT:USDT --timeframe 15m --years 5 --exchange bybit --verify --prepare-ml
""" 