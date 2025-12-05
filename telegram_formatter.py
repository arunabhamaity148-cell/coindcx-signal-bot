# ================================================================
# telegram_formatter.py — Professional Signal Formatting
# ================================================================

from datetime import datetime


class TelegramFormatter:
    """Format signals for Telegram with rich information"""
    
    @staticmethod
    def format_signal_alert(signal: dict, levels: dict = None, volume: dict = None) -> str:
        """
        Create comprehensive signal message
        
        Args:
            signal: Enhanced signal with confidence data
            levels: Price levels data
            volume: Volume analysis data
        """
        
        sym = signal["symbol"]
        side = signal["side"].upper()
        score = signal["score"]
        confidence = signal.get("confidence", 0)
        quality = signal.get("quality", "UNKNOWN")
        risk = signal.get("risk_level", "UNKNOWN")
        last = signal["last"]
        strat = signal["strategy"]
        
        # Emoji based on quality
        quality_emoji = {
            "EXCELLENT": "🟢",
            "GOOD": "🟡",
            "FAIR": "🟠",
            "WEAK": "🔴"
        }
        
        side_emoji = "🟢" if side == "LONG" else "🔴"
        
        # Build message
        msg = f"""
╔══════════════════════════════
║ {quality_emoji.get(quality, '⚪')} <b>{sym}</b> — {side_emoji} <b>{side}</b> SIGNAL
╠══════════════════════════════

📊 <b>SIGNAL METRICS</b>
├ Strategy: <code>{strat}</code>
├ Confidence: <b>{confidence}%</b> ({quality})
├ Score: <b>{score:.1f}</b>/17
├ Risk Level: <b>{risk}</b>
└ Price: <code>${last:.6f}</code>

"""
        
        # Add price levels if available
        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]
            msg += f"""📈 <b>KEY LEVELS</b>
├ Resistance: <code>${sr.get('nearest_resistance', 'N/A'):.6f}</code>
├ Support: <code>${sr.get('nearest_support', 'N/A'):.6f}</code>
"""
            
            if levels.get("analysis", {}).get("risk_reward_ratio"):
                rr = levels["analysis"]["risk_reward_ratio"]
                msg += f"└ R:R Ratio: <b>{rr:.2f}</b>\n\n"
            else:
                msg += "\n"
        
        # Add pivots
        if levels and levels.get("pivots"):
            p = levels["pivots"]
            msg += f"""🎯 <b>PIVOT POINTS</b>
├ R2: <code>{p['r2']:.6f}</code>
├ R1: <code>{p['r1']:.6f}</code>
├ PP: <code>{p['pivot']:.6f}</code>
├ S1: <code>{p['s1']:.6f}</code>
└ S2: <code>{p['s2']:.6f}</code>

"""
        
        # Add volume insights
        if volume:
            if volume.get("smart_money"):
                sm = volume["smart_money"]
                msg += f"""💰 <b>SMART MONEY</b>
├ Direction: <b>{sm['smart_money_direction']}</b>
├ Large Buys: {sm['large_buys']}
└ Large Sells: {sm['large_sells']}

"""
            
            if volume.get("absorption") and volume["absorption"].get("is_absorbing"):
                abs_type = volume["absorption"]["absorption_type"]
                msg += f"⚠️ <b>{abs_type} DETECTED</b>\n\n"
        
        # Add key logic points
        passed = signal.get("passed", [])[:5]
        if passed:
            msg += "✅ <b>KEY FACTORS</b>\n"
            for p in passed:
                msg += f"├ {p.replace('_', ' ').title()}\n"
            msg += "\n"
        
        # Trading suggestions
        msg += f"""💡 <b>SUGGESTED ENTRY</b>
├ Entry Zone: <code>${last * 0.999:.6f} - ${last * 1.001:.6f}</code>
"""
        
        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]
            if side == "LONG" and sr.get("nearest_support"):
                stop = sr["nearest_support"] * 0.998
                msg += f"├ Stop Loss: <code>${stop:.6f}</code>\n"
                if sr.get("nearest_resistance"):
                    target = sr["nearest_resistance"] * 0.998
                    msg += f"└ Target: <code>${target:.6f}</code>\n"
            elif side == "SHORT" and sr.get("nearest_resistance"):
                stop = sr["nearest_resistance"] * 1.002
                msg += f"├ Stop Loss: <code>${stop:.6f}</code>\n"
                if sr.get("nearest_support"):
                    target = sr["nearest_support"] * 1.002
                    msg += f"└ Target: <code>${target:.6f}</code>\n"
        
        msg += f"\n╚══════════════════════════════\n"
        msg += f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
        
        return msg
    
    @staticmethod
    def format_summary_report(signals: list, period_hours: int = 24) -> str:
        """Create daily/hourly summary report"""
        
        if not signals:
            return "📊 <b>No signals in this period</b>"
        
        total = len(signals)
        longs = sum(1 for s in signals if s["side"] == "long")
        shorts = total - longs
        
        avg_confidence = sum(s.get("confidence", 0) for s in signals) / total
        
        excellent = sum(1 for s in signals if s.get("quality") == "EXCELLENT")
        good = sum(1 for s in signals if s.get("quality") == "GOOD")
        
        msg = f"""
📊 <b>SIGNAL SUMMARY</b> ({period_hours}h)
━━━━━━━━━━━━━━━━━━━━━━

📈 Total Signals: <b>{total}</b>
├ 🟢 Long: {longs} ({longs/total*100:.0f}%)
└ 🔴 Short: {shorts} ({shorts/total*100:.0f}%)

⭐ Quality Distribution:
├ EXCELLENT: {excellent}
└ GOOD: {good}

📊 Avg Confidence: <b>{avg_confidence:.1f}%</b>

━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
        
        return msg
    
    @staticmethod
    def format_market_overview(btc_data: dict, market_data: dict) -> str:
        """Create market conditions overview"""
        
        msg = f"""
🌍 <b>MARKET OVERVIEW</b>
━━━━━━━━━━━━━━━━━━━━━━

₿ <b>BTC Status</b>
├ Price: <code>${btc_data.get('price', 0):.2f}</code>
├ 24h Change: <b>{btc_data.get('change_24h', 0):.2f}%</b>
└ Volatility: <b>{btc_data.get('volatility', 'NORMAL')}</b>

📊 <b>Market Conditions</b>
├ Trend: <b>{market_data.get('trend', 'NEUTRAL')}</b>
├ Volume: <b>{market_data.get('volume_state', 'NORMAL')}</b>
└ Sentiment: <b>{market_data.get('sentiment', 'NEUTRAL')}</b>

━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return msg


class AlertManager:
    """Manage different types of alerts"""
    
    def __init__(self):
        self.alert_types = {
            "SIGNAL": "🎯",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
    
    def create_alert(self, alert_type: str, title: str, message: str) -> str:
        """Create formatted alert message"""
        
        emoji = self.alert_types.get(alert_type, "📢")
        
        return f"""
{emoji} <b>{title}</b>
━━━━━━━━━━━━━━━━━━━━━━

{message}

━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}
"""


# Quick formatting functions
def quick_signal(sym: str, side: str, price: float, confidence: float) -> str:
    """Quick signal format for rapid alerts"""
    emoji = "🟢" if side.upper() == "LONG" else "🔴"
    return f"{emoji} <b>{sym}</b> {side.upper()} | ${price:.6f} | {confidence:.0f}%"


def format_error_alert(error_msg: str) -> str:
    """Format error messages"""
    return f"❌ <b>ERROR</b>\n\n<code>{error_msg}</code>"