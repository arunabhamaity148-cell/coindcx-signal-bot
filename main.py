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
    ARUN Trading Bot - Multi-Mode Controller
    Orchestrates trading across QUICK (5m), MID (15m), and TREND (1h) timeframes
    """

    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.ai_advisor = ChatGPTAdvisor()
        self.running = False

    async def startup_checks(self):
        """Perform startup validation and tests"""

        print("\n" + "="*50)
        print("🚀 ARUN BOT STARTING (MULTI-MODE)...")
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

        # Start WebSocket feed (optional - not critical)
        print("\n🔌 Starting WebSocket price feed (optional)...")
        try:
            ws_feed.start()
            if not ws_feed.wait_for_connection(timeout=5):
                print("⚠️ WebSocket timeout (will use REST API instead)")
        except Exception as e:
            print(f"⚠️ WebSocket failed (using REST API): {e}")

        print("\n✅ Startup checks complete")
        if config.MULTI_MODE_ENABLED:
            print(f"🎯 Multi-Mode Active: {', '.join(config.ACTIVE_MODES)}")
        else:
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
        """
        Scan all pairs in ALL modes (QUICK, MID, TREND)
        This enables 3× more signal opportunities
        """

        print(f"\n🔍 Multi-Mode Scan ({datetime.now().strftime('%H:%M:%S')})")
        print("="*60)

        # Check news guard
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"🚫 NEWS GUARD ACTIVE: {reason}")
            print("⏸️ Trading paused until news window clears")
            return

        # ✅ Scan each pair in ALL modes
        for pair in config.PAIRS:
            
            # ✅ Analyze in QUICK, MID, and TREND modes
            for mode in config.ACTIVE_MODES:
                try:
                    mode_config = config.MODE_CONFIG[mode]
                    timeframe = mode_config['timeframe']
                    
                    print(f"\n{'─'*60}")
                    print(f"📊 {mode} Mode: {pair} ({timeframe})")
                    print(f"{'─'*60}")
                    
                    # Get candles for this specific timeframe
                    candles = CoinDCXAPI.get_candles(pair, timeframe, limit=250)

                    if candles.empty:
                        print(f"❌ No data available")
                        continue

                    print(f"🔍 Analyzing {len(candles)} candles...")

                    # ✅ Analyze with specific mode
                    signal = self.signal_gen.analyze(pair, candles, mode)

                    if signal:
                        print(f"\n{'🎉'*20}")
                        print(f"✅ {mode} SIGNAL: {signal['direction']} {pair}")
                        print(f"   Timeframe: {timeframe}")
                        print(f"   Score: {signal['score']}/100")
                        print(f"   Entry: ₹{signal['entry']:,.2f}")
                        print(f"   SL: ₹{signal['sl']:,.2f}")
                        print(f"   TP1: ₹{signal['tp1']:,.2f}")
                        print(f"   TP2: ₹{signal['tp2']:,.2f}")
                        print(f"{'🎉'*20}\n")

                        # Send to Telegram
                        TelegramNotifier.send_signal(signal)

                        # Place order if auto-trade enabled
                        if config.AUTO_TRADE:
                            self.place_order(signal)

                        # Brief pause after signal
                        await asyncio.sleep(3)
                    else:
                        print(f"❌ No {mode} signal")

                except Exception as e:
                    print(f"❌ {mode} mode error for {pair}: {e}")
                    import traceback
                    traceback.print_exc()

                # Small delay between modes
                await asyncio.sleep(1)
            
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
                    f"{signal['mode']} {signal['direction']} {signal['pair']} @ ₹{signal['entry']}"
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

        print(f"\n🟢 Multi-Mode Bot is now running (scan interval: {scan_interval}s)")
        print("="*50 + "\n")

        try:
            while self.running:
                # Get current stats
                stats = self.signal_gen.get_stats()
                print(f"\n📊 Signals today: {stats['signals_today']}/{config.MAX_SIGNALS_PER_DAY}")
                if 'mode_breakdown' in stats:
                    print(f"   Breakdown: ", end="")
                    for mode, count in stats['mode_breakdown'].items():
                        print(f"{mode}={count} ", end="")
                    print()

                # Scan markets (all modes)
                await self.scan_markets()

                # Monitor positions (if auto-trade enabled)
                if config.AUTO_TRADE:
                    await self.monitor_positions()

                # Wait before next scan
                print(f"\n⏳ Next multi-mode scan in {scan_interval}s...")
                await asyncio.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            TelegramNotifier.send_alert("Bot Error", str(e))
            import traceback
            traceback.print_exc()
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