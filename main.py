import asyncio
import time
from datetime import datetime
from config import config
from coindcx_api import CoinDCXAPI
from websocket_feed import ws_feed
from signal_generator import SignalGenerator
from telegram_notifier import TelegramNotifier
from chatgpt_advisor import ChatGPTAdvisor
from news_guard import news_guard

class ArunBot:
    """
    ARUN Trading Bot - Main Controller
    Orchestrates all components for automated trading
    """
    
    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.ai_advisor = ChatGPTAdvisor()
        self.running = False
    
    async def startup_checks(self):
        """Perform startup validation and tests"""
        
        print("\n" + "="*50)
        print("🚀 ARUN BOT STARTING...")
        print("="*50 + "\n")
        
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return False
        
        # Test API connections
        print("\n📡 Testing connections...")
        
        if not CoinDCXAPI.test_connection():
            print("❌ CoinDCX API connection failed")
            return False
        
        if not TelegramNotifier.test_connection():
            print("⚠️ Telegram connection failed (continuing anyway)")
        
        # Start WebSocket feed
        print("\n🔌 Starting WebSocket price feed...")
        ws_feed.start()
        
        if not ws_feed.wait_for_connection(timeout=10):
            print("⚠️ WebSocket connection timeout (will retry)")
        
        print("\n✅ Startup checks complete")
        print(f"📊 Mode: {config.MODE}")
        print(f"🤖 Auto Trade: {'ON' if config.AUTO_TRADE else 'OFF'}")
        print(f"📈 Max Signals: {config.MAX_SIGNALS_PER_DAY}/day")
        print(f"🎯 Tracking {len(config.PAIRS)} pairs")
        
        # Show news guard status
        news_status = news_guard.get_status()
        if news_status['blocked']:
            print(f"\n🚫 NEWS GUARD ACTIVE: {news_status['reason']}")
        else:
            print(f"\n✅ News Guard: Clear (No major news)")
        
        if news_status['upcoming_24h'] > 0:
            print(f"📰 Upcoming events (24h): {news_status['upcoming_24h']}")
            news_guard.print_upcoming_events(hours=24)
        
        # Send startup notification
        TelegramNotifier.send_startup_message()
        
        return True
    
    async def scan_markets(self):
        """Scan all pairs for trading opportunities"""
        
        print(f"\n🔍 Scanning markets... ({datetime.now().strftime('%H:%M:%S')})")
        
        # Check news guard
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"🚫 NEWS GUARD ACTIVE: {reason}")
            print("⏸️ Trading paused until news window clears")
            return
        
        for pair in config.PAIRS:
            try:
                # Get candle data
                interval = config.MODE_CONFIG[config.MODE]['timeframe']
                candles = CoinDCXAPI.get_candles(pair, interval, limit=100)
                
                if candles.empty:
                    print(f"⚠️ No data for {pair}")
                    continue
                
                # Analyze for signals
                signal = self.signal_gen.analyze(pair, candles)
                
                if signal:
                    print(f"\n✅ SIGNAL GENERATED: {signal['direction']} {pair}")
                    print(f"   Score: {signal['score']}/100")
                    print(f"   Entry: ₹{signal['entry']}")
                    
                    # Optional: Get AI validation (minimal usage)
                    if signal['score'] < 70:
                        print("   🤖 Requesting AI validation...")
                        ai_advice = self.ai_advisor.validate_signal(signal)
                        print(f"   AI: {ai_advice['advice']}")
                    
                    # Send to Telegram
                    TelegramNotifier.send_signal(signal)
                    
                    # Place order if auto-trade enabled
                    if config.AUTO_TRADE:
                        self.place_order(signal)
                    
                    # Brief pause after signal
                    await asyncio.sleep(5)
                
            except Exception as e:
                print(f"❌ Error analyzing {pair}: {e}")
            
            # Small delay between pairs
            await asyncio.sleep(2)
    
    def place_order(self, signal: dict):
        """
        Place order on CoinDCX
        
        Args:
            signal: Signal dictionary
        """
        
        try:
            # Calculate position size (example: fixed quantity)
            quantity = 0.01  # Adjust based on your capital
            
            result = CoinDCXAPI.place_order(
                pair=signal['pair'],
                side='buy' if signal['direction'] == 'LONG' else 'sell',
                price=signal['entry'],
                quantity=quantity,
                leverage=signal['leverage']
            )
            
            if result.get('status') == 'success':
                print(f"✅ Order placed successfully")
                TelegramNotifier.send_alert(
                    "Order Placed",
                    f"{signal['direction']} {signal['pair']} @ ₹{signal['entry']}"
                )
            else:
                print(f"⚠️ Order placement issue: {result}")
                
        except Exception as e:
            print(f"❌ Order placement error: {e}")
            TelegramNotifier.send_alert(
                "Order Error",
                f"Failed to place order: {str(e)}"
            )
    
    async def monitor_positions(self):
        """Monitor open positions (if any)"""
        
        try:
            positions = CoinDCXAPI.get_open_positions()
            
            if positions:
                print(f"\n📊 Open positions: {len(positions)}")
                for pos in positions:
                    print(f"   {pos.get('market')}: {pos.get('side')} @ {pos.get('entry_price')}")
            
        except Exception as e:
            print(f"⚠️ Position monitoring error: {e}")
    
    async def run(self):
        """Main bot loop"""
        
        # Startup checks
        if not await self.startup_checks():
            print("❌ Startup failed. Exiting.")
            return
        
        self.running = True
        scan_interval = 60  # seconds (scan every 1 minute)
        
        print(f"\n🟢 Bot is now running (scan interval: {scan_interval}s)")
        print("="*50 + "\n")
        
        try:
            while self.running:
                # Get current stats
                stats = self.signal_gen.get_stats()
                print(f"📊 Signals today: {stats['signals_today']}/{config.MAX_SIGNALS_PER_DAY}")
                
                # Scan markets
                await self.scan_markets()
                
                # Monitor positions (if auto-trade enabled)
                if config.AUTO_TRADE:
                    await self.monitor_positions()
                
                # Wait before next scan
                print(f"\n⏳ Next scan in {scan_interval}s...")
                await asyncio.sleep(scan_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            TelegramNotifier.send_alert("Bot Error", str(e))
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        
        print("\n🛑 Shutting down...")
        
        self.running = False
        ws_feed.stop()
        
        # Send summary
        stats = self.signal_gen.get_stats()
        TelegramNotifier.send_daily_summary(stats)
        
        print("✅ Bot stopped cleanly")


async def main():
    """Entry point"""
    
    bot = ArunBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())