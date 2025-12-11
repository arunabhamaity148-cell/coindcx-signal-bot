"""
Database Setup Script
Creates PostgreSQL database schema for storing trades, signals, and performance data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2 import sql
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    def __init__(self, db_config):
        """
        Initialize database setup
        
        Args:
            db_config: Dict with host, port, database, user, password
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            self.cursor = self.conn.cursor()
            logger.info("✅ Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create all required tables"""
        logger.info("📊 Creating database tables...")
        
        tables = {
            'trades': """
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    trade_id VARCHAR(100) UNIQUE NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price DECIMAL(20, 8) NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price DECIMAL(20, 8),
                    size DECIMAL(20, 8) NOT NULL,
                    leverage INTEGER NOT NULL,
                    stop_loss DECIMAL(20, 8),
                    take_profit DECIMAL(20, 8),
                    pnl DECIMAL(20, 8),
                    pnl_percent DECIMAL(10, 4),
                    fee_open DECIMAL(20, 8),
                    fee_close DECIMAL(20, 8),
                    duration_hours DECIMAL(10, 2),
                    close_reason VARCHAR(50),
                    ml_confidence DECIMAL(5, 4),
                    logic_score DECIMAL(5, 2),
                    status VARCHAR(20) DEFAULT 'open',
                    exchange VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'signals': """
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    signal INTEGER NOT NULL,
                    confidence DECIMAL(5, 4) NOT NULL,
                    logic_score DECIMAL(5, 2),
                    price DECIMAL(20, 8) NOT NULL,
                    executed BOOLEAN DEFAULT FALSE,
                    trade_id VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'daily_performance': """
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    starting_capital DECIMAL(20, 8) NOT NULL,
                    ending_capital DECIMAL(20, 8) NOT NULL,
                    daily_pnl DECIMAL(20, 8) NOT NULL,
                    daily_return DECIMAL(10, 4) NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate DECIMAL(5, 4),
                    max_drawdown DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'equity_curve': """
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    equity DECIMAL(20, 8) NOT NULL,
                    drawdown DECIMAL(10, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'errors': """
                CREATE TABLE IF NOT EXISTS errors (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    error_type VARCHAR(100),
                    error_message TEXT,
                    stack_trace TEXT,
                    severity VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'bot_status': """
                CREATE TABLE IF NOT EXISTS bot_status (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    is_running BOOLEAN NOT NULL,
                    mode VARCHAR(20),
                    capital DECIMAL(20, 8),
                    open_positions INTEGER DEFAULT 0,
                    daily_trades INTEGER DEFAULT 0,
                    daily_pnl DECIMAL(20, 8) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        for table_name, create_sql in tables.items():
            try:
                self.cursor.execute(create_sql)
                self.conn.commit()
                logger.info(f"✅ Created table: {table_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create table {table_name}: {e}")
                self.conn.rollback()
        
        logger.info("✅ All tables created successfully")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        logger.info("🔍 Creating indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_curve(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_daily_perf_date ON daily_performance(date)",
        ]
        
        for index_sql in indexes:
            try:
                self.cursor.execute(index_sql)
                self.conn.commit()
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
                self.conn.rollback()
        
        logger.info("✅ Indexes created")
    
    def insert_sample_data(self):
        """Insert sample data for testing"""
        logger.info("📝 Inserting sample data...")
        
        # Sample trade
        sample_trade = """
            INSERT INTO trades (
                trade_id, symbol, side, entry_time, entry_price,
                size, leverage, stop_loss, take_profit,
                ml_confidence, logic_score, status, exchange
            ) VALUES (
                'SAMPLE_001', 'BTC/USDT:USDT', 'LONG', NOW(),
                50000.00, 1000.00, 5, 49000.00, 51500.00,
                0.7500, 75.50, 'open', 'bybit'
            ) ON CONFLICT (trade_id) DO NOTHING
        """
        
        # Sample signal
        sample_signal = """
            INSERT INTO signals (
                timestamp, symbol, signal, confidence,
                logic_score, price
            ) VALUES (
                NOW(), 'BTC/USDT:USDT', 0, 0.7800,
                78.50, 50000.00
            )
        """
        
        # Sample daily performance
        sample_daily = """
            INSERT INTO daily_performance (
                date, starting_capital, ending_capital,
                daily_pnl, daily_return, total_trades,
                winning_trades, losing_trades, win_rate
            ) VALUES (
                CURRENT_DATE, 10000.00, 10500.00,
                500.00, 0.05, 5, 3, 2, 0.60
            ) ON CONFLICT (date) DO NOTHING
        """
        
        samples = [sample_trade, sample_signal, sample_daily]
        
        for sample_sql in samples:
            try:
                self.cursor.execute(sample_sql)
                self.conn.commit()
            except Exception as e:
                logger.warning(f"⚠️ Sample data warning: {e}")
                self.conn.rollback()
        
        logger.info("✅ Sample data inserted")
    
    def verify_setup(self):
        """Verify database setup"""
        logger.info("\n🔍 Verifying database setup...")
        
        # Check tables
        self.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = self.cursor.fetchall()
        logger.info(f"✅ Found {len(tables)} tables:")
        for table in tables:
            logger.info(f"   • {table[0]}")
        
        # Count records
        tables_to_check = ['trades', 'signals', 'daily_performance']
        for table_name in tables_to_check:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            logger.info(f"   {table_name}: {count} records")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✅ Database connection closed")


def main():
    """Main setup function"""
    logger.info("="*80)
    logger.info("🗄️ DATABASE SETUP")
    logger.info("="*80)
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    logger.info(f"Host: {db_config['host']}")
    logger.info(f"Port: {db_config['port']}")
    logger.info(f"Database: {db_config['database']}")
    logger.info(f"User: {db_config['user']}")
    logger.info("="*80)
    
    # Setup database
    setup = DatabaseSetup(db_config)
    
    try:
        # Connect
        setup.connect()
        
        # Create tables
        setup.create_tables()
        
        # Create indexes
        setup.create_indexes()
        
        # Insert sample data
        setup.insert_sample_data()
        
        # Verify
        setup.verify_setup()
        
        logger.info("\n" + "="*80)
        logger.info("✅ DATABASE SETUP COMPLETE")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Update config/api_keys.env with database credentials")
        logger.info("2. Run data download: python scripts/download_data.py")
        logger.info("3. Train models: python scripts/train_models.py")
        logger.info("4. Start bot: python main.py")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
    finally:
        setup.close()


if __name__ == "__main__":
    main()


# ==================== SETUP INSTRUCTIONS ====================
"""
1. Install PostgreSQL:
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   
   # Windows
   Download from: https://www.postgresql.org/download/windows/

2. Create Database:
   sudo -u postgres psql
   CREATE DATABASE trading_bot;
   CREATE USER trading_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
   \q

3. Set Environment Variables:
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=trading_bot
   export DB_USER=trading_user
   export DB_PASSWORD=your_password

4. Run Setup:
   python scripts/setup_database.py

5. Verify:
   psql -U trading_user -d trading_bot
   \dt  # List tables
   SELECT * FROM trades LIMIT 5;
"""