# ================================================================
# telegram_formatter.py — Professional Signal Formatting (NO $)
# ================================================================

from datetime import datetime


class TelegramFormatter:
    """Format signals for Telegram with rich information"""

    @staticmethod
    def format_signal_alert(signal: dict, levels: dict = None, volume: dict = None) -> str:
        """Create comprehensive signal message"""

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
└ Price: <code>{last:.6f}</code>

"""

        # Add price levels if available
        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]
            nr = sr.get('nearest_resistance')
            ns = sr.get('nearest_support')

            msg += "📈 <b>KEY LEVELS</b>\n"
            msg += f"├ Resistance: <code>{nr:.6f}</code>\n" if nr else "├ Resistance: <code>N/A</code>\n"
            msg += f"├ Support: <code>{ns:.6f}</code>\n" if ns else "├ Support: <code>N/A</code>\n"

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
        entry_low = last * 0.999
        entry_high = last * 1.001

        msg += f"""💡 <b>SUGGESTED ENTRY</b>
├ Entry Zone: <code>{entry_low:.6f} - {entry_high:.6f}</code>
"""

        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]

            if side == "LONG" and sr.get("nearest_support"):
                stop = sr["nearest_support"] * 0.998
                msg += f"├ Stop Loss: <code>{stop:.6f}</code>\n"
                if sr.get("nearest_resistance"):
                    target = sr["nearest_resistance"] * 0.998
                    msg += f"└ Target: <code>{target:.6f}</code>\n"

            elif side == "SHORT" and sr.get("nearest_resistance"):
                stop = sr["nearest_resistance"] * 1.002
                msg += f"├ Stop Loss: <code>{stop:.6f}</code>\n"
                if sr.get("nearest_support"):
                    target = sr["nearest_support"] * 1.002
                    msg += f"└ Target: <code>{target:.6f}</code>\n"

        msg += f"\n╚══════════════════════════════\n"
        msg += f"⏰ {datetime.utcnow().strftime('%H:%M:%S UTC')}"
        return msg


# Quick formatting functions
def quick_signal(sym: str, side: str, price: float, confidence: float) -> str:
    emoji = "🟢" if side.upper() == "LONG" else "🔴"
    return f"{emoji} <b>{sym}</b> {side.upper()} | {price:.6f} | {confidence:.0f}%"


def format_error_alert(error_msg: str) -> str:
    return f"❌ <b>ERROR</b>\n\n<code>{error_msg}</code>"