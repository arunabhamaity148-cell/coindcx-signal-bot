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
from signal_explainer import SignalExplainer
from position_manager import PositionManager
from exit_strategy import ExitStrategy
from risk_manager import RiskManager
from performance_tracker import PerformanceTracker


class ArunBot:
    """
    ARUN Trading Bot - Enhanced with Exit Management
    """

    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.ai_advisor = ChatGPTAdvisor()
        self.position_manager = PositionManager()
        self.exit_strategy = ExitStrategy()
        self.risk_manager = RiskManager(base_capital=10000)  # Set your capital
        self.performance_tracker = PerformanceTracker()
        self.running = False

    async def startup_checks(self):
        """Perform startup validation and tests"""

        print("\n" + "="*50)
        print("🚀 ARUN BOT STARTING (ENHANCED)...")
        print("="*50 + "\n")

        try:
            config.validate()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return False

        print("\n📡 Testing connections...")

        if not CoinDCXAPI.test_connection():
            print("❌ CoinDCX API connection failed")
            return False

        if not TelegramNotifier.test_connection():
            print("⚠️ Telegram connection failed (continuing anyway)")

        print("\n🔌 Starting WebSocket feed...")
        try:
            ws_feed.start()
            if not ws_feed.wait_for_connection(timeout=5):
                print("⚠️ WebSocket timeout (will use REST API)")
        except Exception as e:
            print(f"⚠️ WebSocket failed: {e}")

        print("\n✅ Startup checks complete")
        if config.MULTI_MODE_ENABLED:
            print(f"🎯 Multi-Mode: {', '.join(config.ACTIVE_MODES)}")
        else:
            print(f"📊 Mode: {config.MODE}")
        print(f"🤖 Auto Trade: {'ON' if config.AUTO_TRADE else 'OFF'}")
        print(f"📈 Max Signals: {config.MAX_SIGNALS_PER_DAY}/day")
        print(f"🎯 Tracking {len(config.PAIRS)} pairs")

        # Show risk limits
        print(f"\n🛡️ RISK MANAGEMENT:")
        print(f"   Daily Loss Limit: {config.RISK_LIMITS['daily_loss_percent']}%")
        print(f"   Max Positions: {config.RISK_LIMITS['max_positions']}/day")
        print(f"   Risk Per Trade: {config.RISK_LIMITS['capital_per_trade']}%")

        # News guard
        news_status = news_guard.get_status()
        if news_status['blocked']:
            print(f"\n🚫 NEWS GUARD ACTIVE: {news_status['reason']}")
        else:
            print(f"\n✅ News Guard: Clear")

        if news_status['upcoming_24h'] > 0:
            print(f"📰 Upcoming events (24h): {news_status['upcoming_24h']}")
            news_guard.print_upcoming_events(hours=24)

        TelegramNotifier.send_startup_message()

        return True

    async def scan_markets(self):
        """Scan all pairs in all modes"""

        print(f"\n🔍 Multi-Mode Scan ({datetime.now().strftime('%H:%M:%S')})")
        print("="*60)

        # Check news guard
        is_blocked, reason = news_guard.is_blocked()
        if is_blocked:
            print(f"🚫 NEWS GUARD: {reason}")
            return

        # Check daily risk limits
        limits_ok, limit_reason = self.risk_manager.check_daily_limits()
        if not limits_ok:
            print(f"🛑 RISK LIMIT: {limit_reason}")
            TelegramNotifier.send_risk_alert("Daily Limit", limit_reason)
            return

        # Scan each pair in all modes
        for pair in config.PAIRS:

            for mode in config.ACTIVE_MODES:
                try:
                    mode_config = config.MODE_CONFIG[mode]
                    timeframe = mode_config['timeframe']

                    print(f"\n{'─'*60}")
                    print(f"📊 {mode}: {pair} ({timeframe})")
                    print(f"{'─'*60}")

                    candles = CoinDCXAPI.get_candles(pair, timeframe, limit=250)

                    if candles.empty:
                        print(f"❌ No data")
                        continue

                    print(f"🔍 Analyzing {len(candles)} candles...")

                    # Analyze
                    signal = self.signal_gen.analyze(pair, candles, mode)

                    if signal:
                        print(f"\n{'🎉'*20}")
                        print(f"✅ {mode} SIGNAL: {signal['direction']} {pair}")
                        print(f"   Score: {signal['score']}/100")
                        print(f"   Entry: ₹{signal['entry']:,.2f}")
                        print(f"{'🎉'*20}\n")

                        # Send to Telegram
                        TelegramNotifier.send_signal(signal)

                        # Generate explanation
                        try:
                            explanation_data = SignalExplainer.explain_signal(signal, candles)
                            
                            if explanation_data['chart_path']:
                                TelegramNotifier.send_chart(explanation_data['chart_path'])
                            
                            if explanation_data['explanation']:
                                TelegramNotifier.send_explanation(explanation_data['explanation'])
                        except Exception as e:
                            print(f"⚠️ Explanation failed: {e}")

                        # Add position to tracking
                        self.position_manager.add_position(pair, signal)
                        self.risk_manager.record_position_opened()

                        # Place order if auto-trade
                        if config.AUTO_TRADE:
                            self.place_order(signal)

                        await asyncio.sleep(3)
                    else:
                        print(f"❌ No {mode} signal")

                except Exception as e:
                    print(f"❌ {mode} error for {pair}: {e}")
                    import traceback
                    traceback.print_exc()

                await asyncio.sleep(1)

            await asyncio.sleep(2)

    async def monitor_positions(self):
        """
        ✅ NEW: Monitor active positions for exits
        Runs every 15 minutes (or configure as needed)
        """

        active_positions = self.position_manager.get_active_positions()

        if not active_positions:
            return

        print(f"\n📊 Monitoring {len(active_positions)} active positions...")

        for pair, position in list(active_positions.items()):
            try:
                signal = position['signal']
                mode_config = config.MODE_CONFIG[signal['mode']]
                timeframe = mode_config['timeframe']

                # Get current candles
                candles = CoinDCXAPI.get_candles(pair, timeframe, limit=50)

                if candles.empty:
                    continue

                current_price = float(candles['close'].iloc[-1])

                # Check trailing stop update
                update_result = self.position_manager.update_position(
                    pair, current_price, candles
                )

                if update_result['action'] == 'UPDATE_SL':
                    # Send trailing stop notification
                    TelegramNotifier.send_trailing_update(
                        pair, 
                        update_result['new_sl'],
                        update_result['message']
                    )

                elif update_result['action'] == 'CLOSE':
                    # Trailing stop hit
                    self._close_position_logic(
                        pair, signal, current_price, 
                        update_result['reason'], 100
                    )
                    continue

                # Check dynamic exits
                should_exit, exit_reason, booking_pct = self.exit_strategy.check_all_exits(
                    pair, signal, candles
                )

                if should_exit:
                    print(f"🚪 EXIT: {pair} - {exit_reason}")
                    
                    self._close_position_logic(
                        pair, signal, current_price,
                        exit_reason, booking_pct
                    )

            except Exception as e:
                print(f"⚠️ Monitor error {pair}: {e}")

    def _close_position_logic(self, pair: str, signal: Dict, 
                             exit_price: float, exit_reason: str, booking_pct: int):
        """
        ✅ NEW: Handle position closing logic
        """

        # Close in position manager
        exit_record = self.position_manager.close_position(
            pair, exit_price, exit_reason, booking_pct
        )

        # Calculate profit
        entry = signal['entry']
        direction = signal['direction']
        
        if direction == "LONG":
            profit_pct = ((exit_price - entry) / entry) * 100
        else:
            profit_pct = ((entry - exit_price) / entry) * 100

        # Log to performance tracker
        self.performance_tracker.log_trade_result(
            signal, exit_price, exit_reason, 
            datetime.now(), booking_pct
        )

        # Update risk manager
        profit_amount = 0  # Calculate based on your position size
        self.risk_manager.record_trade_result(
            pair, signal['mode'], profit_pct, profit_amount
        )

        # Send exit notification
        TelegramNotifier.send_exit_alert(
            pair, exit_reason, booking_pct, 
            profit_pct, exit_price
        )

        print(f"✅ Position closed: {pair} | {profit_pct:.2f}%")

    def place_order(self, signal: dict):
        """Place order on CoinDCX"""

        try:
            quantity = 0.01  # Adjust

            result = CoinDCXAPI.place_order(
                pair=signal['pair'],
                side='buy' if signal['direction'] == 'LONG' else 'sell',
                price=signal['entry'],
                quantity=quantity,
                leverage=signal['leverage']
            )

            if result.get('status') == 'success':
                print(f"✅ Order placed")
                TelegramNotifier.send_alert(
                    "Order Placed",
                    f"{signal['mode']} {signal['direction']} {signal['pair']}"
                )
            else:
                print(f"⚠️ Order issue: {result}")

        except Exception as e:
            print(f"❌ Order error: {e}")
            TelegramNotifier.send_alert("Order Error", str(e))

    async def run(self):
        """Main bot loop"""

        if not await self.startup_checks():
            print("❌ Startup failed")
            return

        self.running = True
        scan_interval = config.SCAN_INTERVAL_SECONDS
        monitor_interval = 900  # 15 minutes

        print(f"\n🟢 Bot running (scan: {scan_interval}s, monitor: {monitor_interval}s)")
        print("="*50 + "\n")

        last_monitor = time.time()

        try:
            while self.running:
                # Stats
                stats = self.signal_gen.get_stats()
                risk_stats = self.risk_manager.get_daily_stats()
                
                print(f"\n📊 Signals: {stats['signals_today']}/{config.MAX_SIGNALS_PER_DAY}")
                print(f"💰 Daily P&L: ₹{risk_stats['daily_pnl']:,.2f} ({risk_stats['daily_pnl_pct']:.2f}%)")
                print(f"📈 Active Positions: {self.position_manager.get_position_count()}")
                
                if 'mode_breakdown' in stats:
                    print(f"   Mode: ", end="")
                    for mode, count in stats['mode_breakdown'].items():
                        print(f"{mode}={count} ", end="")
                    print()

                # Scan markets
                await self.scan_markets()

                # Monitor positions (every 15 min)
                current_time = time.time()
                if current_time - last_monitor >= monitor_interval:
                    await self.monitor_positions()
                    last_monitor = current_time

                # Wait
                print(f"\n⏳ Next scan in {scan_interval}s...")
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

        # Send daily summary
        risk_stats = self.risk_manager.get_daily_stats()
        TelegramNotifier.send_daily_report(risk_stats)

        # Send performance report
        try:
            report = self.performance_tracker.generate_daily_report()
            print(f"\n{report}")
        except:
            pass

        print("✅ Bot stopped cleanly")


async def main():
    """Entry point"""
    bot = ArunBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())