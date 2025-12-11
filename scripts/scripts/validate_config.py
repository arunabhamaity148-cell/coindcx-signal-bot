"""
Configuration Validator
Checks if everything is properly configured before running the bot
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
import importlib.util
import ccxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def check(self, condition, success_msg, error_msg, is_warning=False):
        """Generic check function"""
        self.checks_total += 1
        
        if condition:
            logger.info(f"✅ {success_msg}")
            self.checks_passed += 1
            return True
        else:
            if is_warning:
                logger.warning(f"⚠️  {error_msg}")
                self.warnings.append(error_msg)
            else:
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
            return False
    
    def check_python_version(self):
        """Check Python version"""
        logger.info("\n🐍 Checking Python Version...")
        
        version = sys.version_info
        is_valid = version.major == 3 and version.minor >= 10
        
        self.check(
            is_valid,
            f"Python {version.major}.{version.minor}.{version.micro}",
            f"Python 3.10+ required, found {version.major}.{version.minor}.{version.micro}"
        )
    
    def check_required_packages(self):
        """Check if required Python packages are installed"""
        logger.info("\n📦 Checking Required Packages...")
        
        required_packages = [
            'pandas', 'numpy', 'ccxt', 'tensorflow', 'xgboost',
            'sklearn', 'psycopg2', 'telegram', 'matplotlib', 'seaborn'
        ]
        
        for package in required_packages:
            spec = importlib.util.find_spec(package)
            self.check(
                spec is not None,
                f"{package} installed",
                f"{package} not installed - run: pip install {package}"
            )
    
    def check_directory_structure(self):
        """Check if all required directories exist"""
        logger.info("\n📁 Checking Directory Structure...")
        
        required_dirs = [
            'config', 'data', 'data/historical', 'ml', 'strategy',
            'risk', 'execution', 'monitoring', 'backtest', 'scripts',
            'visualization', 'models', 'logs', 'charts', 'backtest_results'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            self.check(
                path.exists() and path.is_dir(),
                f"{dir_path}/ exists",
                f"{dir_path}/ missing - run: mkdir -p {dir_path}"
            )
    
    def check_config_files(self):
        """Check if configuration files exist"""
        logger.info("\n⚙️  Checking Configuration Files...")
        
        config_file = Path('config/api_keys.env')
        
        if config_file.exists():
            logger.info("✅ config/api_keys.env exists")
            self.checks_passed += 1
            
            # Check if it's still the example file
            with open(config_file, 'r') as f:
                content = f.read()
                
            self.check(
                'your_bybit_api_key_here' not in content.lower(),
                "API keys configured (not example file)",
                "config/api_keys.env still contains placeholder values!",
                is_warning=True
            )
        else:
            logger.error("❌ config/api_keys.env missing")
            logger.info("   Run: cp config/api_keys.env.example config/api_keys.env")
            self.errors.append("config/api_keys.env missing")
        
        self.checks_total += 1
        
        # Check settings.py
        settings_file = Path('config/settings.py')
        self.check(
            settings_file.exists(),
            "config/settings.py exists",
            "config/settings.py missing"
        )
    
    def check_api_keys(self):
        """Check if API keys are valid format"""
        logger.info("\n🔑 Checking API Keys...")
        
        try:
            from config.settings import EXCHANGES
            
            for exchange_name, config in EXCHANGES.items():
                api_key = config.get('api_key')
                secret = config.get('secret')
                
                # Check if keys exist
                has_key = api_key and api_key != 'YOUR_API_KEY'
                has_secret = secret and secret != 'YOUR_SECRET'
                
                if exchange_name != 'binance':  # Binance is optional (data only)
                    self.check(
                        has_key and has_secret,
                        f"{exchange_name.upper()} API keys present",
                        f"{exchange_name.upper()} API keys missing or not configured",
                        is_warning=(exchange_name == 'okx')  # OKX is secondary
                    )
                
                # Test connection (if keys present)
                if has_key and has_secret and exchange_name == 'bybit':
                    logger.info(f"   Testing {exchange_name.upper()} connection...")
                    try:
                        exchange = ccxt.bybit({
                            'apiKey': api_key,
                            'secret': secret,
                            'enableRateLimit': True
                        })
                        balance = exchange.fetch_balance()
                        logger.info(f"   ✅ {exchange_name.upper()} connection successful")
                    except Exception as e:
                        logger.warning(f"   ⚠️  {exchange_name.upper()} connection failed: {e}")
                        self.warnings.append(f"{exchange_name} API connection failed")
        
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            self.errors.append("Failed to load config/settings.py")
    
    def check_database(self):
        """Check database connection"""
        logger.info("\n🗄️  Checking Database...")
        
        try:
            import psycopg2
            from config.settings import DATA_CONFIG
            
            conn = psycopg2.connect(
                host=DATA_CONFIG['db_host'],
                port=DATA_CONFIG['db_port'],
                database=DATA_CONFIG['db_name'],
                user=DATA_CONFIG['db_user'],
                password=DATA_CONFIG['db_password']
            )
            conn.close()
            
            logger.info("✅ PostgreSQL connection successful")
            self.checks_passed += 1
        
        except Exception as e:
            logger.warning(f"⚠️  PostgreSQL connection failed: {e}")
            logger.warning("   Database is optional but recommended")
            self.warnings.append("PostgreSQL not configured (optional)")
        
        self.checks_total += 1
    
    def check_ml_models(self):
        """Check if ML models are trained"""
        logger.info("\n🤖 Checking ML Models...")
        
        model_files = [
            'models/lstm_model.h5',
            'models/xgboost_model.pkl',
            'models/random_forest.pkl'
        ]
        
        for model_file in model_files:
            path = Path(model_file)
            self.check(
                path.exists(),
                f"{model_file} found",
                f"{model_file} missing - run: python scripts/train_models.py",
                is_warning=True
            )
    
    def check_historical_data(self):
        """Check if historical data is downloaded"""
        logger.info("\n📊 Checking Historical Data...")
        
        data_dir = Path('data/historical')
        csv_files = list(data_dir.glob('*.csv'))
        
        self.check(
            len(csv_files) > 0,
            f"Found {len(csv_files)} data file(s)",
            "No historical data found - run: python scripts/download_data.py",
            is_warning=True
        )
    
    def check_telegram(self):
        """Check Telegram configuration"""
        logger.info("\n📱 Checking Telegram Bot...")
        
        try:
            from config.settings import TELEGRAM_CONFIG
            
            bot_token = TELEGRAM_CONFIG.get('bot_token')
            chat_id = TELEGRAM_CONFIG.get('chat_id')
            
            has_token = bot_token and bot_token != 'YOUR_BOT_TOKEN'
            has_chat = chat_id and chat_id != 'YOUR_CHAT_ID'
            
            self.check(
                has_token and has_chat,
                "Telegram bot configured",
                "Telegram bot not configured (optional)",
                is_warning=True
            )
        
        except Exception as e:
            logger.warning("⚠️  Telegram config check failed (optional)")
    
    def check_permissions(self):
        """Check file permissions"""
        logger.info("\n🔒 Checking Permissions...")
        
        # Check if logs directory is writable
        logs_dir = Path('logs')
        self.check(
            os.access(logs_dir, os.W_OK),
            "logs/ is writable",
            "logs/ is not writable - check permissions"
        )
        
        # Check if models directory is writable
        models_dir = Path('models')
        self.check(
            os.access(models_dir, os.W_OK),
            "models/ is writable",
            "models/ is not writable - check permissions"
        )
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "="*80)
        logger.info("📋 VALIDATION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nChecks passed: {self.checks_passed}/{self.checks_total}")
        
        if self.warnings:
            logger.info(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"   • {warning}")
        
        if self.errors:
            logger.info(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                logger.info(f"   • {error}")
        
        logger.info("\n" + "="*80)
        
        if self.errors:
            logger.error("❌ VALIDATION FAILED")
            logger.error("Please fix the errors above before running the bot")
            return False
        elif self.warnings:
            logger.warning("⚠️  VALIDATION PASSED WITH WARNINGS")
            logger.warning("Bot can run but some features may not work")
            return True
        else:
            logger.info("✅ VALIDATION PASSED")
            logger.info("All systems ready! You can start the bot.")
            return True
    
    def run_all_checks(self):
        """Run all validation checks"""
        logger.info("="*80)
        logger.info("🔍 CONFIGURATION VALIDATION")
        logger.info("="*80)
        
        self.check_python_version()
        self.check_required_packages()
        self.check_directory_structure()
        self.check_config_files()
        self.check_api_keys()
        self.check_database()
        self.check_ml_models()
        self.check_historical_data()
        self.check_telegram()
        self.check_permissions()
        
        return self.print_summary()


def main():
    """Main validation"""
    validator = ConfigValidator()
    success = validator.run_all_checks()
    
    if success:
        logger.info("\n🚀 Next Steps:")
        logger.info("   1. If models not trained: python scripts/train_models.py")
        logger.info("   2. Run backtest: python backtest/backtester.py --data your_data.csv")
        logger.info("   3. Start paper trading: python main.py --mode paper")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


# ==================== USAGE ====================
"""
# Run validation before starting bot
python scripts/validate_config.py

# Check validation status
echo $?  # 0 = passed, 1 = failed

# Use in scripts
if python scripts/validate_config.py; then
    python main.py --mode paper
else
    echo "Configuration validation failed"
fi
"""