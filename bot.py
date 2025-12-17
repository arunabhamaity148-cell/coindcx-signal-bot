"""
CoinDCX Smart Trading Bot - Main Entry Point
"""
import logging
import time
import schedule
import requests
import pandas as pd
from telegram import Bot
from config import *
from signal_logic import analyze_market, can_send_signal, mark_signal_sent
from smart_logic import get_smart_signals

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

bot = Bot(token=TELEGRAM_BOT_TOKEN)

def send_telegram(msg: str):
    """Send message to Telegram"""
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
        logging.info("✅ Message sent")
    except Exception as e:
        logging.error(f"❌ Telegram error: {e}")

def get_candles(market: str) -> pd.DataFrame:
    """Fetch candles from CoinDCX"""
    url = "https://public.coindcx.com/market_data/candles"
    params = {"pair": market, "interval": CANDLE_INTERVAL, "limit": CANDLE_LIMIT}
    
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        for col in ["close", "open", "high", "low", "volume"]:
            df[col] = df[col].astype(float)
        
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"❌ Fetch error {market}: {e}")
        return None

def process_market(market: str):
    """Process single market"""
    logging.info(f"🔍 Analyzing {market}...")
    
    if not can_send_signal(market):
        logging.info(f"⏳ {market} in cooldown/limit")
        return
    
    df = get_candles(market)
    if df is None:
        return
    
    # Regular analysis
    result = analyze_market(df, market)
    
    # Smart logic overlay
    smart = None
    if ENABLE_SMART_LOGIC:
        smart = get_smart_signals(df, market)
    
    # Combine scores
    if result and smart:
        final_score = (result['score'] * WEIGHT_REGULAR_SCORE + 
                      smart['smart_score'] * WEIGHT_SMART_SCORE)
    elif result:
        final_score = result['score']
    elif smart:
        final_score = smart['smart_score']
    else:
        logging.info(f"📊 {market}: No signal")
        return
    
    if final_score < MIN_SIGNAL_SCORE:
        logging.info(f"📊 {market}: Score {final_score:.0f} < threshold")
        return
    
    # Build message
    direction = smart['direction'] if smart else result['direction']
    emoji = "🟢" if direction == "LONG" else "🔴"
    
    msg = f"{emoji} *{direction} SIGNAL*

"
    msg += f"💎 Market: `{market}`
"
    
    if smart:
        msg += f"🧠 Smart: *{smart['smart_score']:.0f}/100*
"
        msg += f"📊 Technical: *{result['score']:.0f}/100*
" if result else ""
        msg += f"🎯 Combined: *{final_score:.0f}/100*
"
        msg += f"⚡ Confidence: *{smart['confidence']}*
"
        msg += f"📈 Regime: `{smart['regime']}`

"
    else:
        msg += f"📊 Score: *{final_score:.0f}/100*

"
    
    if result:
        msg += f"📍 Entry: `{result['entry']:.4f}`
"
        msg += f"🛑 Stop Loss: `{result['sl']:.4f}`
"
        msg += f"🎯 TP1 (50%): `{result['tp1']:.4f}`
"
        msg += f"🎯 TP2 (50%): `{result['tp2']:.4f}`
"
        msg += f"⚖️ R:R: `1:{result['rr']:.2f}`

"
        
        msg += f"📊 *Indicators:*
"
        msg += f"RSI: `{result['rsi']:.1f}`
"
        msg += f"MACD: `{result['macd_hist']:.6f}`
"
        msg += f"ADX: `{result['adx']:.1f}`
"
        msg += f"EMA: `{result['ema_fast']:.2f}`/`{result['ema_slow']:.2f}`
"
        if result['pattern']:
            msg += f"Pattern: `{result['pattern']}`
"
        if result['divergence']:
            msg += f"Div: `{result['divergence']}`
"
    
    if smart and smart['signals']:
        msg += f"
🧠 *Smart Signals:*
"
        for sig in smart['signals'][:6]:
            msg += f"• {sig}
"
    
    msg += f"
⏰ {time.strftime('%H:%M:%S IST')}"
    
    send_telegram(msg)
    mark_signal_sent(market)
    logging.info(f"✅ {market}: {direction} signal sent (score: {final_score:.0f})")

def job():
    """Scheduled scan job"""
    logging.info("=" * 60)
    logging.info("🚀 Market scan started")
    
    for market in MARKETS:
        try:
            process_market(market.strip())
            time.sleep(2)
        except Exception as e:
            logging.error(f"❌ Error {market}: {e}")
    
    logging.info("✅ Scan complete")
    logging.info("=" * 60)

def main():
    """Main bot loop"""
    startup = (
        "🤖 *CoinDCX Smart Bot Started*

"
        f"📊 Markets: `{len(MARKETS)}`
"
        f"⏱️ Interval: `{CANDLE_INTERVAL}`
"
        f"📈 Min Score: `{MIN_SIGNAL_SCORE}`
"
        f"🔄 Check: Every `{CHECK_INTERVAL_MINUTES}min`
"
        f"⏳ Cooldown: `{COOLDOWN_MINUTES}min`
"
        f"📅 Daily Limit: `{MAX_SIGNALS_PER_DAY}`
"
        f"🧠 Smart Logic: `{'ON' if ENABLE_SMART_LOGIC else 'OFF'}`

"
        f"Pairs: {', '.join([f'`{m}`' for m in MARKETS[:4]])}"
    )
    
    send_telegram(startup)
    logging.info("🤖 Bot initialized")
    
    # First run
    job()
    
    # Schedule
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()