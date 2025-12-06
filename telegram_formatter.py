# ================================================================
# telegram_formatter.py — FINAL (No $, Raw Price, Await Fixed)
# ================================================================

from datetime import datetime, timezone
from chart_image import generate_chart_image


class TelegramFormatter:
    """Format signals for Telegram with charts and correlation"""

    @staticmethod
    async def format_signal_alert(
        signal: dict,
        levels: dict = None,
        volume: dict = None,
        include_chart: bool = True
    ) -> tuple:
        sym = signal["symbol"]
        side = signal["side"].upper()
        score = signal["score"]
        confidence = signal.get("confidence", 0)
        quality = signal.get("quality", "UNKNOWN")
        risk = signal.get("risk_level", "UNKNOWN")
        last = signal["last"]
        strat = signal["strategy"]

        corr_data = signal.get("correlation", {})
        corr = corr_data.get("price_corr", 0)
        corr_strength = corr_data.get("strength", "UNKNOWN")

        quality_emoji = {"EXCELLENT": "🟢", "GOOD": "🟡", "FAIR": "🟠", "WEAK": "🔴"}
        side_emoji = "🟢" if side == "LONG" else "🔴"

        msg = f"""
╔══════════════════════════════
║ {quality_emoji.get(quality, '⚪')} <b>{sym}</b> — {side_emoji} <b>{side}</b>
╠══════════════════════════════

📊 <b>SIGNAL QUALITY</b>
├ Strategy: <code>{strat}</code>
├ Confidence: <b>{confidence:.1f}%</b> ({quality})
├ Score: <b>{score:.1f}/17</b>
├ Risk: <b>{risk}</b>
└ Price: <code>{last:.8f}</code>
"""

        if corr != 0:
            corr_emoji = "📈" if corr > 0 else "📉"
            msg += f"""
🔗 <b>BTC CORRELATION</b>
├ Correlation: {corr_emoji} <b>{corr:+.2f}</b>
├ Strength: <b>{corr_strength}</b>
└ {"Independent" if abs(corr) < 0.3 else "Follows BTC" if corr > 0 else "Inverse BTC"}

"""

        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]
            msg += f"""
📈 <b>KEY LEVELS</b>
├ Resistance: <code>{sr.get('nearest_resistance', 'N/A'):.8f}</code>
├ Support: <code>{sr.get('nearest_support', 'N/A'):.8f}</code>
"""
            if levels.get("analysis", {}).get("risk_reward_ratio"):
                rr = levels["analysis"]["risk_reward_ratio"]
                msg += f"└ R:R Ratio: <b>{rr:.2f}</b>\n\n"
            else:
                msg += "\n"

        if levels and levels.get("pivots"):
            p = levels["pivots"]
            msg += f"""
🎯 <b>PIVOT POINTS</b>
├ R2: <code>{p['r2']:.8f}</code>
├ R1: <code>{p['r1']:.8f}</code>
├ PP: <code>{p['pivot']:.8f}</code>
├ S1: <code>{p['s1']:.8f}</code>
└ S2: <code>{p['s2']:.8f}</code>

"""

        if volume:
            if volume.get("smart_money"):
                sm = volume["smart_money"]
                msg += f"""
💰 <b>SMART MONEY</b>
├ Direction: <b>{sm['smart_money_direction']}</b>
├ Large Buys: {sm['large_buys']}
└ Large Sells: {sm['large_sells']}

"""
            if volume.get("absorption") and volume["absorption"].get("is_absorbing"):
                abs_type = volume["absorption"]["absorption_type"]
                msg += f"⚠️ <b>{abs_type} DETECTED</b>\n\n"

        passed = signal.get("passed", [])[:5]
        if passed:
            msg += "✅ <b>KEY FACTORS</b>\n"
            for p in passed:
                msg += f"├ {p.replace('_', ' ').title()}\n"
            msg += "\n"

        msg += f"""
💡 <b>SUGGESTED ENTRY</b>
├ Entry Zone: <code>{last * 0.999:.8f} - {last * 1.001:.8f}</code>
"""

        if levels and levels.get("support_resistance"):
            sr = levels["support_resistance"]
            if side == "LONG" and sr.get("nearest_support"):
                stop = sr["nearest_support"] * 0.998
                msg += f"├ Stop Loss: <code>{stop:.8f}</code>\n"
                if sr.get("nearest_resistance"):
                    target = sr["nearest_resistance"] * 0.998
                    msg += f"└ Target: <code>{target:.8f}</code>\n"
            elif side == "SHORT" and sr.get("nearest_resistance"):
                stop = sr["nearest_resistance"] * 1.002
                msg += f"├ Stop Loss: <code>{stop:.8f}</code>\n"
                if sr.get("nearest_support"):
                    target = sr["nearest_support"] * 1.002
                    msg += f"└ Target: <code>{target:.8f}</code>\n"

        msg += f"\n╚══════════════════════════════\n"
        msg += f"⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"

        # ✅ Chart URL await fixed
        chart_url = None
        if include_chart:
            try:
                chart_url = await generate_chart_image(sym, "15m")
            except Exception as e:
                pass

        return msg.strip(), chart_url

    @staticmethod
    def format_summary_report(signals: list, period_hours: int = 24) -> str:
        if not signals:
            return "📊 <b>No signals in this period</b>"
        total = len(signals)
        longs = sum(1 for s in signals if s.get("side") == "long")
        shorts = total - longs
        avg_confidence = sum(s.get("confidence", 0) for s in signals) / total
        excellent = sum(1 for s in signals if s.get("quality") == "EXCELLENT")
        good = sum(1 for s in signals if s.get("quality") == "GOOD")
        correlations = [s.get("correlation", {}).get("price_corr", 0) for s in signals]
        avg_corr = sum(correlations) / len(correlations) if correlations else 0

        return f"""
📊 <b>SIGNAL SUMMARY</b> ({period_hours}h)
━━━━━━━━━━━━━━━━━━━━━━
📈 Total Signals: <b>{total}</b>
├ 🟢 Long: {longs} ({longs/total*100:.0f}%)
└ 🔴 Short: {shorts} ({shorts/total*100:.0f}%)
⭐ Quality Distribution:
├ EXCELLENT: {excellent}
└ GOOD: {good}
📊 Average Confidence: <b>{avg_confidence:.1f}%</b>
🔗 Average BTC Corr: <b>{avg_corr:+.2f}</b>
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
""".strip()


# Quick helpers
def quick_signal(sym: str, side: str, price: float, confidence: float, corr: float = 0) -> str:
    emoji = "🟢" if side.upper() == "LONG" else "🔴"
    corr_str = f" | BTC: {corr:+.2f}" if corr != 0 else ""
    return f"{emoji} <b>{sym}</b> {side.upper()} | {price:.8f} | {confidence:.0f}%{corr_str}"


def format_error_alert(error_msg: str) -> str:
    return f"❌ <b>ERROR</b>\n\n<code>{error_msg}</code>"
