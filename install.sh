#!/bin/bash

# Crypto Trading Bot - Auto Installation Script
# This script automates the entire setup process

set -e  # Exit on error

echo "================================================================================"
echo "🤖 CRYPTO TRADING BOT - AUTOMATED INSTALLATION"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check if running on supported OS
check_os() {
    print_info "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_success "Detected: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="mac"
        print_success "Detected: macOS"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    if [ "$OS" == "linux" ]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            python3-dev \
            python3-pip \
            python3-venv \
            libpq-dev \
            postgresql \
            postgresql-contrib
        
        # TA-Lib
        print_info "Installing TA-Lib..."
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
        tar -xzf ta-lib-0.4.0-src.tar.gz
        cd ta-lib/
        ./configure --prefix=/usr
        make
        sudo make install
        cd ..
        rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
        
    elif [ "$OS" == "mac" ]; then
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Please install: https://brew.sh"
            exit 1
        fi
        
        brew install \
            python@3.10 \
            postgresql \
            ta-lib
    fi
    
    print_success "System dependencies installed"
}

# Create virtual environment
setup_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping..."
    else
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install Python packages
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Activate venv
    source venv/bin/activate
    
    # Install from requirements.txt
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Create directory structure
create_directories() {
    print_info "Creating directory structure..."
    
    mkdir -p data/historical
    mkdir -p models
    mkdir -p logs
    mkdir -p charts
    mkdir -p backtest_results
    mkdir -p scripts
    mkdir -p monitoring
    mkdir -p visualization
    
    # Create __init__.py files
    touch config/__init__.py
    touch data/__init__.py
    touch features/__init__.py
    touch ml/__init__.py
    touch strategy/__init__.py
    touch risk/__init__.py
    touch execution/__init__.py
    touch monitoring/__init__.py
    touch backtest/__init__.py
    touch scripts/__init__.py
    touch visualization/__init__.py
    
    print_success "Directory structure created"
}

# Setup configuration
setup_config() {
    print_info "Setting up configuration..."
    
    if [ ! -f "config/api_keys.env" ]; then
        cp config/api_keys.env.example config/api_keys.env
        print_warning "Created config/api_keys.env - PLEASE EDIT IT WITH YOUR API KEYS!"
    else
        print_warning "config/api_keys.env already exists. Skipping..."
    fi
}

# Setup PostgreSQL
setup_database() {
    print_info "Setting up PostgreSQL database..."
    
    read -p "Do you want to setup PostgreSQL database? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Start PostgreSQL
        if [ "$OS" == "linux" ]; then
            sudo systemctl start postgresql
            sudo systemctl enable postgresql
        elif [ "$OS" == "mac" ]; then
            brew services start postgresql
        fi
        
        # Create database and user
        print_info "Creating database and user..."
        sudo -u postgres psql << EOF
CREATE DATABASE trading_bot;
CREATE USER trading_user WITH PASSWORD 'trading_password_123';
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_user;
\q
EOF
        
        # Run setup script
        source venv/bin/activate
        $PYTHON_CMD scripts/setup_database.py
        
        print_success "Database setup complete"
        print_warning "Default password: trading_password_123"
        print_warning "CHANGE THIS PASSWORD in production!"
    else
        print_warning "Skipping database setup"
    fi
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Activate venv
    source venv/bin/activate
    
    # Check Python packages
    $PYTHON_CMD -c "import pandas, numpy, ccxt, tensorflow, xgboost" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Core packages verified"
    else
        print_error "Some packages failed to import"
    fi
    
    # Check directory structure
    if [ -d "models" ] && [ -d "logs" ] && [ -d "data/historical" ]; then
        print_success "Directory structure verified"
    else
        print_error "Directory structure incomplete"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "================================================================================"
    echo "✅ INSTALLATION COMPLETE!"
    echo "================================================================================"
    echo ""
    echo "📝 NEXT STEPS:"
    echo ""
    echo "1. Edit API keys:"
    echo "   nano config/api_keys.env"
    echo ""
    echo "2. Activate virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "3. Download historical data:"
    echo "   python scripts/download_data.py --years 5 --verify --prepare-ml"
    echo ""
    echo "4. Train ML models:"
    echo "   python scripts/train_models.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv"
    echo ""
    echo "5. Run backtest:"
    echo "   python backtest/backtester.py --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv --save"
    echo ""
    echo "6. Start paper trading:"
    echo "   python main.py --mode paper --capital 10000"
    echo ""
    echo "7. Create visualizations:"
    echo "   python visualization/chart_plotter.py --type dashboard \\"
    echo "       --data data/historical/BTC_USDT_USDT_15m_5y_bybit.csv \\"
    echo "       --trades backtest_results/trades.csv \\"
    echo "       --equity backtest_results/equity_curve.csv"
    echo ""
    echo "================================================================================"
    echo "⚠️  IMPORTANT REMINDERS:"
    echo "================================================================================"
    echo ""
    echo "• NEVER commit api_keys.env to GitHub"
    echo "• Start with PAPER TRADING first"
    echo "• Only invest what you can afford to lose"
    echo "• Monitor the bot regularly"
    echo "• Keep software updated"
    echo ""
    echo "📚 Documentation: README.md"
    echo "🐛 Issues: https://github.com/yourusername/crypto-trading-bot/issues"
    echo ""
    echo "Good luck and trade safely! 🚀"
    echo ""
}

# Main installation flow
main() {
    check_os
    check_python
    
    read -p "Install system dependencies? (requires sudo) (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_deps
    fi
    
    setup_venv
    install_python_deps
    create_directories
    setup_config
    setup_database
    verify_installation
    print_next_steps
}

# Run installation
main