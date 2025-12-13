"""
🤖 CoinDCX Futures Trading Bot - Main Orchestrator
Tomar Best Friend er Jonno Banano 🇧🇩
Daily ₹2000 Target with 80% Win Rate

Created with ❤️ by Claude
"""

import time
from datetime import datetime, timedelta
import sys

# Import all modules
from config import *
from market_scanner import MarketScanner
from signal_engine import SignalEngine
from risk_manager import RiskManager
from telegram_bot import TelegramBot
from performance_tracker import PerformanceTracker

class CoinDCXTradingBot:
    def __init__(self):
        print("🚀 Initializing CoinDCX Futures Trading Bot...")
        
        self.scanner = MarketScanner()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.telegram = TelegramBot()
        self.tracker = PerformanceTracker()
        
        self.signals_today = 0
        self.last_report_time = datetime.now()
        self.is_running = True
        
        print("✅ All modules loaded successfully!")
        
    def check_daily_limits(self):
        """Check if we hit daily signal limits"""
        if self.signals_today >= MAX_DAILY_SIGNALS:
            print(f"⚠️ Daily signal limit reached: {self.signals_today}/{MAX_DAILY_SIGNALS}")
            return False
        
        # Check if target achieved
        stats = self.tracker.get_daily_summary()
        if stats['total_pnl'] >= TARGET_DAILY_PROFIT:
            print(f"🎉 Daily target achieved! PnL: ₹{stats['total_pnl']:.2f}")
            self.telegram.send_message(f"🎉 **TARGET ACHIEVED!** 🎉\n\nDaily PnL: ₹{stats['total_pnl']:.2f}")
            return False
        
        return True
    
    def send_hourly_report(self):
        """Send performance report every hour"""
        time_diff = (datetime.now() - self.last_report_time).seconds
        
        if time_diff >= PERFORMANCE_REPORT_INTERVAL:
            stats = self.tracker.get_daily_summary()
            self.telegram.send_performance_report(stats)
            
            # Also send breakdown
            breakdown = self.tracker.format_breakdown_message()
            if breakdown:
                self.telegram.send_message(breakdown)
            
            self.last_report_time = datetime.now()
    
    def process_signal_mode(self, mode, tradeable_pairs):
        """Process signals for a specific mode (quick/mid/trend)"""
        print(f"\n🔍 Scanning {mode.upper()} mode...")
        
        for pair_data in tradeable_pairs:
            if self.signals_today >= MAX_DAILY_SIGNALS:
                break
            
            try:
                # Generate signal
                signal = self.signal_engine.generate_signal(pair_data, mode)
                
                if signal is None:
                    continue
                
                # Calculate TP/SL levels
                levels = self.risk_manager.calculate_tp_sl(signal)
                
                if levels is None:
                    continue
                
                # Valid signal found!
                print(f"\n🎯 SIGNAL GENERATED!")
                print(f"Symbol: {signal['symbol']}")
                print(f"Direction: {signal['direction']}")
                print(f"Entry: ₹{levels['entry']:.2f}")
                print(f"TP1: ₹{levels['tp1']:.2f}")
                print(f"TP2: ₹{levels['tp2']:.2f}")
                print(f"SL: ₹{levels['sl']:.2f}")
                print(f"Score: {signal['score']}/15")
                
                # Send to Telegram
                chart_df = pair_data['data']
                self.telegram.send_signal_alert(signal, levels, chart_df)
                
                # Track signal
                self.tracker.add_signal(signal, levels)
                
                self.signals_today += 1
                
                # Simulated trade execution (Manual trading ke liye)
                print(f"\n📝 Manual Trade Setup:")
                print(f"1. Go to CoinDCX Futures")
                print(f"2. Open {signal['symbol']} chart")
                print(f"3. Place {signal['direction']} order at ₹{levels['entry']:.2f}")
                print(f"4. Set TP1: ₹{levels['tp1']:.2f}")
                print(f"5. Set TP2: ₹{levels['tp2']:.2f}")
                print(f"6. Set SL: ₹{levels['sl']:.2f}")
                print(f"7. Use {levels['leverage']}x leverage")
                print(f"8. Margin: ₹{levels['margin']:,.0f}\n")
                
            except Exception as e:
                print(f"⚠️ Error processing signal for {pair_data['symbol']}: {e}")
    
    def run_scan_cycle(self):
        """Main scanning cycle"""
        try:
            print("\n" + "="*60)
            print(f"🔄 Starting Scan Cycle at {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            # Step 1: Check Market Health
            print("\n📊 Checking Market Health...")
            health_score = self.scanner.calculate_market_health()
            
            if health_score < 4:  # Lowered from 6
                print("⚠️ Market health too low. Skipping this cycle.")
                return
            
            # Step 2: Scan All Pairs
            print(f"\n🔍 Scanning pairs...")
            tradeable_pairs = self.scanner.scan_all_pairs()
            
            if not tradeable_pairs:
                print("⚠️ No tradeable pairs found in this cycle.")
                return
            
            # Step 3: Generate Signals for Each Mode
            for mode in ["quick", "mid", "trend"]:
                if not self.check_daily_limits():
                    break
                self.process_signal_mode(mode, tradeable_pairs)
            
            print(f"\n✅ Scan cycle completed. Signals today: {self.signals_today}/{MAX_DAILY_SIGNALS}")
            
        except Exception as e:
            print(f"⚠️ Error in scan cycle: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start the bot"""
        print("\n" + "="*60)
        print("🤖 COINDCX FUTURES TRADING BOT")
        print("="*60)
        print(f"💰 Target: ₹{TARGET_DAILY_PROFIT:,.0f}/day")
        print(f"📊 Signals: {MIN_DAILY_SIGNALS}-{MAX_DAILY_SIGNALS}/day")
        print(f"⚡ Leverage: {LEVERAGE}x")
        print(f"💵 Margin: ₹{MARGIN_PER_TRADE:,.0f}/trade")
        print(f"🎯 Win Target: 80%+")
        print("="*60)
        
        # Send startup message to Telegram
        self.telegram.send_startup_message()
        
        print("\n🚀 Bot is now running. Press Ctrl+C to stop.\n")
        
        try:
            while self.is_running:
                # Check daily limits
                if not self.check_daily_limits():
                    print("\n⏸️ Daily limits reached. Waiting for next day...")
                    time.sleep(3600)  # Wait 1 hour
                    continue
                
                # Run scan cycle
                self.run_scan_cycle()
                
                # Send hourly report
                self.send_hourly_report()
                
                # Wait for next cycle
                print(f"\n⏳ Waiting {SCAN_INTERVAL} seconds before next scan...")
                time.sleep(SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Bot stopped by user")
            self.shutdown()
        except Exception as e:
            print(f"\n\n❌ Fatal error: {e}")
            self.telegram.send_message(f"🚨 **BOT ERROR**\n\n{str(e)}")
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        print("\n🛑 Shutting down bot...")
        
        # Send final report
        stats = self.tracker.get_daily_summary()
        self.telegram.send_performance_report(stats)
        
        # Save trade history
        self.tracker.save_to_file()
        
        print("\n✅ Bot shutdown complete. Trade safe! 🙏")
        sys.exit(0)

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     🇧🇩 CoinDCX FUTURES TRADING BOT 🇧🇩               ║
    ║                                                       ║
    ║     Tomar Best Friend er Jonno Banano                ║
    ║     Daily ₹2000 Target | 80% Win Rate                ║
    ║                                                       ║
    ║     Created with ❤️ by Claude                        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Validate config
    if COINDCX_API_KEY == "your_api_key_here":
        print("⚠️ WARNING: Please set your CoinDCX API credentials in config.py")
        print("⚠️ Bot will run in simulation mode only.\n")
    
    if TELEGRAM_BOT_TOKEN == "your_telegram_token":
        print("⚠️ WARNING: Please set your Telegram bot token in config.py")
        print("⚠️ Alerts will not be sent.\n")
    
    # Initialize and start bot
    bot = CoinDCXTradingBot()
    bot.start()