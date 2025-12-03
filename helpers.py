import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from config import *

logger = logging.getLogger(__name__)

# ==========================================
# COOLDOWN MANAGER
# ==========================================
class CooldownManager:
    def __init__(self):
        self.last_sent = {}
        self.signal_hashes = {}
    
    def can_send(self, symbol, mode, cooldown_seconds):
        """Check if enough time passed since last signal"""
        key = f"{symbol}_{mode}"
        now = datetime.utcnow()
        
        if key in self.last_sent:
            elapsed = (now - self.last_sent[key]).total_seconds()
            if elapsed < cooldown_seconds:
                return False
        
        self.last_sent[key] = now
        return True
    
    def ensure_single_alert(self, rule_key, triggers, price, mode):
        """Dedupe identical signals"""
        signal_hash = hashlib.md5(f"{rule_key}_{triggers}_{price}_{mode}".encode()).hexdigest()
        now = datetime.utcnow()
        
        if signal_hash in self.signal_hashes:
            last_time = self.signal_hashes[signal_hash]
            if (now - last_time).total_seconds() < 1800:  # 30 min
                return False
        
        self.signal_hashes[signal_hash] = now
        return True

cooldown_manager = CooldownManager()

# ==========================================
# MARKET DATA CLASS
# ==========================================
class MarketData:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'test': USE_TESTNET
            }
        })
    
    async def fetch_with_retry(self, async_fn, retries=3):
        """Fetch with exponential backoff retry"""
        for attempt in range(retries):
            try:
                return await async_fn()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait_time = FETCH_RETRY_BACKOFF ** attempt
                logger.warning(f"Retry {attempt+1}/{retries} after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def get_all_data(self, symbol):
        """Fetch all market data with retry"""
        try:
            ticker = await self.fetch_with_retry(lambda: self.exchange.fetch_ticker(symbol))
            ohlcv_1m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(symbol, '1m', limit=100))
            ohlcv_5m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(symbol, '5m', limit=100))
            ohlcv_15m = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(symbol, '15m', limit=100))
            ohlcv_1h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(symbol, '1h', limit=100))
            ohlcv_4h = await self.fetch_with_retry(lambda: self.exchange.fetch_ohlcv(symbol, '4h', limit=100))
            orderbook = await self.fetch_with_retry(lambda: self.exchange.fetch_order_book(symbol, limit=20))
            
            volume = ticker.get('baseVolume', ticker.get('volume', 0))
            
            # Validate OHLCV data
            if not validate_ohlcv(ohlcv_1m, 50):
                logger.error(f"Insufficient 1m data for {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'volume': volume,
                'ohlcv_1m': ohlcv_1m,
                'ohlcv_5m': ohlcv_5m,
                'ohlcv_15m': ohlcv_15m,
                'ohlcv_1h': ohlcv_1h,
                'ohlcv_4h': ohlcv_4h,
                'orderbook': orderbook,
                'spread': ((orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0]) * 100
            }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def validate_ohlcv(ohlcv, required_length):
    """Validate OHLCV has sufficient data"""
    if not ohlcv or len(ohlcv) < required_length:
        return False
    return True

def format_price_usd(price):
    """Format price with USD precision"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"

def round_prices_for_display(price, mode):
    """Round prices based on mode and value"""
    if price >= 1000:
        return round(price, 2)
    elif price >= 1:
        return round(price, 2)
    else:
        return round(price, 4)

def normalize_scores_to_10(raw_score, max_possible):
    """Normalize score to 0-10 scale"""
    normalized = (raw_score / max_possible) * 10
    return min(10, max(0, normalized))

def score_to_confidence(score):
    """Convert score to confidence percentage"""
    confidence = int(score * 10)
    return min(100, max(0, confidence))

def compute_weighted_score(triggers_dict, weights):
    """Compute weighted score from triggers"""
    total = 0
    for trigger_name, value in triggers_dict.items():
        weight = weights.get(trigger_name, 1.0)
        total += value * weight
    return total

def calc_single_tp_sl(entry_price, direction, mode):
    """Calculate single TP and SL"""
    config = TP_SL_CONFIG.get(mode, TP_SL_CONFIG['MID'])
    tp_pct = config['tp']
    sl_pct = config['sl']
    
    if direction == 'long':
        tp = entry_price * (1 + tp_pct / 100)
        sl = entry_price * (1 - sl_pct / 100)
    else:
        tp = entry_price * (1 - tp_pct / 100)
        sl = entry_price * (1 + sl_pct / 100)
    
    leverage = suggest_leverage(mode)
    
    code = f"""```python
# {mode} {direction.upper()} Setup
ENTRY = {format_price_usd(entry_price)}
TP = {format_price_usd(tp)}
SL = {format_price_usd(sl)}
LEVERAGE = {leverage}x
RISK = {sl_pct}%
REWARD = {tp_pct}%
```"""
    
    return {
        'tp': tp,
        'sl': sl,
        'leverage': leverage,
        'code': code
    }

def calc_liquidation(entry, sl, leverage, side, position_size=100):
    """Calculate liquidation price and distance"""
    if side == 'long':
        liq_price = entry * (1 - 1/leverage)
    else:
        liq_price = entry * (1 + 1/leverage)
    
    dist_usd = abs(liq_price - sl)
    dist_pct = (dist_usd / entry) * 100
    
    return {
        'liq_price': liq_price,
        'dist_usd': dist_usd,
        'dist_pct': dist_pct
    }

def safety_check_sl_vs_liq(entry, sl, leverage):
    """Check if SL is safe from liquidation"""
    side = 'long' if sl < entry else 'short'
    liq_info = calc_liquidation(entry, sl, leverage, side)
    
    if liq_info['dist_pct'] < MIN_SAFE_LIQ_DISTANCE_PCT:
        logger.warning(f"⚠️ SL too close to liquidation! Distance: {liq_info['dist_pct']:.2f}%")
        return False
    return True

def suggest_leverage(mode):
    """Suggest leverage based on mode"""
    return SUGGESTED_LEVERAGE.get(mode, 30)

def spread_and_depth_check(data):
    """Check spread and orderbook depth"""
    if data['spread'] > MAX_SPREAD_PCT:
        logger.warning(f"Spread too wide: {data['spread']:.3f}%")
        return False
    
    orderbook = data['orderbook']
    bid_depth = sum([b[0] * b[1] for b in orderbook['bids'][:5]])
    ask_depth = sum([a[0] * a[1] for a in orderbook['asks'][:5]])
    
    if bid_depth < MIN_ORDERBOOK_DEPTH or ask_depth < MIN_ORDERBOOK_DEPTH:
        logger.warning(f"Insufficient orderbook depth")
        return False
    
    return True

async def btc_calm_check(market):
    """Check if BTC is calm enough to trade"""
    try:
        btc_data = await market.get_all_data('BTCUSDT')
        if not btc_data:
            return True  # Allow if can't fetch BTC
        
        # Check 1m volatility
        ohlcv_1m = btc_data['ohlcv_1m'][-2:]
        price_change_1m = abs(ohlcv_1m[-1][4] - ohlcv_1m[-2][4]) / ohlcv_1m[-2][4] * 100
        
        # Check 5m volatility
        ohlcv_5m = btc_data['ohlcv_5m'][-2:]
        price_change_5m = abs(ohlcv_5m[-1][4] - ohlcv_5m[-2][4]) / ohlcv_5m[-2][4] * 100
        
        if price_change_1m > BTC_VOLATILITY_THRESHOLDS['1m']:
            logger.warning(f"⚠️ BTC too volatile (1m): {price_change_1m:.2f}%")
            return False
        
        if price_change_5m > BTC_VOLATILITY_THRESHOLDS['5m']:
            logger.warning(f"⚠️ BTC too volatile (5m): {price_change_5m:.2f}%")
            return False
        
        return True
    except Exception as e:
        logger.error(f"BTC calm check error: {e}")
        return True  # Allow on error

# ==========================================
# SAFE INDICATOR WRAPPERS
# ==========================================
def safe_calculate_rsi(ohlcv, period=14):
    """Safe RSI calculation"""
    try:
        if not validate_ohlcv(ohlcv, period + 10):
            return 50  # Neutral default
        closes = [x[4] for x in ohlcv]
        df = pd.DataFrame({'close': closes})
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

def safe_calculate_macd(ohlcv):
    """Safe MACD calculation"""
    try:
        if not validate_ohlcv(ohlcv, 35):
            return 0, 0, 0, 0
        closes = [x[4] for x in ohlcv]
        df = pd.DataFrame({'close': closes})
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1], macd.iloc[-2], signal.iloc[-2]
    except:
        return 0, 0, 0, 0

def safe_calculate_ema(ohlcv, period):
    """Safe EMA calculation"""
    try:
        if not validate_ohlcv(ohlcv, period + 10):
            return 0
        closes = [x[4] for x in ohlcv]
        df = pd.DataFrame({'close': closes})
        ema = df['close'].ewm(span=period, adjust=False).mean().iloc[-1]
        return ema if not pd.isna(ema) else 0
    except:
        return 0

def safe_calculate_adx(ohlcv, period=14):
    """Safe ADX calculation"""
    try:
        if not validate_ohlcv(ohlcv, period + 20):
            return 0
        
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
        
        df['+dm'] = df['high'].diff()
        df['-dm'] = -df['low'].diff()
        df['+dm'] = df['+dm'].where((df['+dm'] > df['-dm']) & (df['+dm'] > 0), 0)
        df['-dm'] = df['-dm'].where((df['-dm'] > df['+dm']) & (df['-dm'] > 0), 0)
        
        atr = df['tr'].rolling(period).mean()
        di_plus = 100 * (df['+dm'].rolling(period).mean() / atr)
        di_minus = 100 * (df['-dm'].rolling(period).mean() / atr)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
    except:
        return 0

def safe_calculate_bollinger_bands(ohlcv, period=20, std=2):
    """Safe Bollinger Bands calculation"""
    try:
        if not validate_ohlcv(ohlcv, period + 10):
            price = ohlcv[-1][4]
            return price, price, price
        closes = [x[4] for x in ohlcv]
        df = pd.DataFrame({'close': closes})
        sma = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
    except:
        price = ohlcv[-1][4]
        return price, price, price

def safe_calculate_atr(ohlcv, period=14):
    """Safe ATR calculation"""
    try:
        if not validate_ohlcv(ohlcv, period + 10):
            return 0
        highs = [x[2] for x in ohlcv]
        lows = [x[3] for x in ohlcv]
        closes = [x[4] for x in ohlcv]
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
        atr = df['tr'].rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0
    except:
        return 0

def safe_calculate_vwap(ohlcv):
    """Safe VWAP calculation"""
    try:
        if not validate_ohlcv(ohlcv, 10):
            return ohlcv[-1][4]
        typical_prices = [(x[2] + x[3] + x[4]) / 3 for x in ohlcv]
        volumes = [x[5] for x in ohlcv]
        total_vol = sum(volumes)
        if total_vol == 0:
            return ohlcv[-1][4]
        return sum([tp * v for tp, v in zip(typical_prices, volumes)]) / total_vol
    except:
        return ohlcv[-1][4]

# ==========================================
# QUICK SIGNALS (10 logics)
# ==========================================
def calculate_quick_signals(data):
    """Calculate Quick signal score with weighted logic"""
    score = 0
    triggers = []
    trigger_weights = {}
    direction = 'none'
    long_signals = 0
    short_signals = 0
    
    try:
        price = data['price']
        ohlcv_1m = data['ohlcv_1m']
        ohlcv_5m = data['ohlcv_5m']
        
        # 1. RSI_oversold_breakout
        rsi = safe_calculate_rsi(ohlcv_1m)
        if 25 < rsi < 35:
            trigger_weights['RSI_oversold_breakout'] = 1
            triggers.append("✅ RSI_oversold_breakout")
            long_signals += 1
        elif 65 < rsi < 75:
            trigger_weights['RSI_overbought_breakdown'] = 1
            triggers.append("✅ RSI_overbought_breakdown")
            short_signals += 1
        
        # 2. MACD_bullish_cross
        macd, signal, macd_prev, signal_prev = safe_calculate_macd(ohlcv_5m)
        if macd > signal and macd_prev <= signal_prev:
            trigger_weights['MACD_bullish_cross'] = 1
            triggers.append("✅ MACD_bullish_cross")
            long_signals += 1
        elif macd < signal and macd_prev >= signal_prev:
            trigger_weights['MACD_bearish_cross'] = 1
            triggers.append("✅ MACD_bearish_cross")
            short_signals += 1
        
        # 3. Volume_spike_support
        volumes = [x[5] for x in ohlcv_1m[-20:]]
        avg_vol = np.mean(volumes[:-1])
        current_vol = volumes[-1]
        if current_vol > avg_vol * 2:
            trigger_weights['Volume_spike_support'] = 1
            triggers.append("✅ Volume_spike")
            long_signals += 1
        
        # 4. VWAP_reclaim_long
        vwap = safe_calculate_vwap(ohlcv_5m)
        if price > vwap * 1.001:
            trigger_weights['VWAP_reclaim_long'] = 1
            triggers.append("✅ VWAP_reclaim_long")
            long_signals += 1
        elif price < vwap * 0.999:
            trigger_weights['VWAP_below_short'] = 1
            triggers.append("✅ VWAP_below_short")
            short_signals += 1
        
        # 5. EMA_9_21_bull_cross
        ema9 = safe_calculate_ema(ohlcv_5m, 9)
        ema21 = safe_calculate_ema(ohlcv_5m, 21)
        if ema9 > 0 and ema21 > 0:
            if price > ema9 > ema21:
                trigger_weights['EMA_9_21_bull_cross'] = 1
                triggers.append("✅ EMA_9_21_bull")
                long_signals += 1
            elif price < ema9 < ema21:
                trigger_weights['EMA_9_21_bear'] = 1
                triggers.append("✅ EMA_9_21_bear")
                short_signals += 1
        
        # 6. Orderblock_retest_long
        candles = ohlcv_1m[-10:]
        for i in range(len(candles)-3):
            if candles[i][5] > np.mean([c[5] for c in candles]) * 1.5:
                trigger_weights['Orderblock_retest_long'] = 1
                triggers.append("✅ Orderblock")
                long_signals += 1
                break
        
        # 7. Liquidity_sweep_long
        lows = [x[3] for x in ohlcv_1m[-5:]]
        if len(lows) >= 3 and price < min(lows[:-1]) and price > lows[-2]:
            trigger_weights['Liquidity_sweep_long'] = 1
            triggers.append("✅ Liquidity_sweep")
            long_signals += 1
        
        # 8. Bollinger_band_squeeze_break
        upper, mid, lower = safe_calculate_bollinger_bands(ohlcv_5m)
        if mid > 0:
            bb_width = (upper - lower) / mid
            if bb_width < 0.02:
                trigger_weights['Bollinger_band_squeeze_break'] = 1
                triggers.append("✅ BB_squeeze")
                if price > mid:
                    long_signals += 1
                else:
                    short_signals += 1
        
        # 9. Spread_tight_low_latency
        if data['spread'] < 0.03:
            trigger_weights['Spread_tight_low_latency'] = 1
            triggers.append("✅ Tight_spread")
        
        # 10. Market_structure_HH_HL
        highs = [x[2] for x in ohlcv_5m[-5:]]
        if len(highs) >= 3 and highs[-1] > highs[-2] > highs[-3]:
            trigger_weights['Market_structure_HH_HL'] = 1
            triggers.append("✅ Higher_highs")
            long_signals += 1
        
        # Compute weighted score
        raw_score = compute_weighted_score(trigger_weights, LOGIC_WEIGHTS)
        score = normalize_scores_to_10(raw_score, 15)  # Max possible weighted ~15
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        logger.error(f"Quick signal error: {e}")
    
    return {
        'score': round(score, 1),
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

# ==========================================
# MID SIGNALS (10 logics)
# ==========================================
def calculate_mid_signals(data):
    """Calculate Mid signal score with weighted logic"""
    score = 0
    triggers = []
    trigger_weights = {}
    direction = 'none'
    long_signals = 0
    short_signals = 0
    
    try:
        price = data['price']
        ohlcv_15m = data['ohlcv_15m']
        ohlcv_1h = data['ohlcv_1h']
        
        # 1. RSI_overbought_reversal
        rsi = safe_calculate_rsi(ohlcv_15m)
        if rsi > 70:
            trigger_weights['RSI_overbought_reversal'] = 1
            triggers.append("✅ RSI_overbought")
            short_signals += 1
        elif rsi < 30:
            trigger_weights['RSI_oversold'] = 1
            triggers.append("✅ RSI_oversold")
            long_signals += 1
        
        # 2. MACD_hidden_bullish
        macd, signal, _, _ = safe_calculate_macd(ohlcv_15m)
        if macd > 0 and macd > signal:
            trigger_weights['MACD_hidden_bullish'] = 1
            triggers.append("✅ MACD_hidden_bull")
            long_signals += 1
        
        # 3. MACD_divergence_support
        if macd < 0 and macd < signal:
            trigger_weights['MACD_divergence_support'] = 1
            triggers.append("✅ MACD_divergence")
            long_signals += 1
# 4. ADX_trend_strength_up
        adx = safe_calculate_adx(ohlcv_1h)
        if adx > 25:
            trigger_weights['ADX_trend_strength_up'] = 1
            triggers.append("✅ ADX_strong_trend")
            long_signals += 1
        
        # 5. Volume_delta_buy_pressure
        volumes = [x[5] for x in ohlcv_15m[-10:]]
        if len(volumes) > 0 and volumes[-1] > np.mean(volumes) * 1.5:
            trigger_weights['Volume_delta_buy_pressure'] = 1
            triggers.append("✅ Volume_delta_buy")
            long_signals += 1
        
        # 6. EMA_50_bounce
        ema50 = safe_calculate_ema(ohlcv_1h, 50)
        if ema50 > 0 and abs(price - ema50) / ema50 < 0.01:
            trigger_weights['EMA_50_bounce'] = 1
            triggers.append("✅ EMA50_bounce")
            if price > ema50:
                long_signals += 1
            else:
                short_signals += 1
        
        # 7. EMA_200_bounce
        ema200 = safe_calculate_ema(ohlcv_1h, 200)
        if ema200 > 0 and abs(price - ema200) / ema200 < 0.015:
            trigger_weights['EMA_200_bounce'] = 1
            triggers.append("✅ EMA200_bounce")
            if price > ema200:
                long_signals += 1
        
        # 8. FVG_immediate_fill
        candles = ohlcv_15m[-5:]
        for i in range(len(candles)-2):
            gap = candles[i+2][3] - candles[i][2]
            if gap > 0:
                trigger_weights['FVG_immediate_fill'] = 1
                triggers.append("✅ FVG_detected")
                long_signals += 1
                break
        
        # 9. Keltner_breakout_up
        atr = safe_calculate_atr(ohlcv_1h)
        ema20 = safe_calculate_ema(ohlcv_1h, 20)
        if ema20 > 0 and atr > 0:
            upper_keltner = ema20 + (2 * atr)
            if price > upper_keltner:
                trigger_weights['Keltner_breakout_up'] = 1
                triggers.append("✅ Keltner_breakout")
                long_signals += 1
        
        # 10. Trendline_break_retest
        closes = [x[4] for x in ohlcv_1h[-10:]]
        if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
            trigger_weights['Trendline_break_retest'] = 1
            triggers.append("✅ Trendline_break")
            long_signals += 1
        
        # Compute weighted score
        raw_score = compute_weighted_score(trigger_weights, LOGIC_WEIGHTS)
        score = normalize_scores_to_10(raw_score, 15)
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        logger.error(f"Mid signal error: {e}")
    
    return {
        'score': round(score, 1),
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

# ==========================================
# TREND SIGNALS (10 logics)
# ==========================================
def calculate_trend_signals(data):
    """Calculate Trend signal score with weighted logic"""
    score = 0
    triggers = []
    trigger_weights = {}
    direction = 'none'
    long_signals = 0
    short_signals = 0
    
    try:
        price = data['price']
        ohlcv_1h = data['ohlcv_1h']
        ohlcv_4h = data['ohlcv_4h']
        
        # 1. Breaker_block_retest
        candles = ohlcv_4h[-10:]
        highs = [x[2] for x in candles]
        if price > max(highs[:-1]):
            trigger_weights['Breaker_block_retest'] = 1
            triggers.append("✅ Breaker_block")
            long_signals += 1
        
        # 2. Chop_zone_exit_long
        adx = safe_calculate_adx(ohlcv_4h)
        if adx > 20:
            trigger_weights['Chop_zone_exit_long'] = 1
            triggers.append("✅ Chop_zone_exit")
            long_signals += 1
        
        # 3. Bollinger_midband_reject_flip
        upper, mid, lower = safe_calculate_bollinger_bands(ohlcv_4h)
        if mid > 0:
            if price > mid:
                trigger_weights['Bollinger_midband_reject_flip'] = 1
                triggers.append("✅ BB_midband_above")
                long_signals += 1
            elif price < mid:
                trigger_weights['BB_midband_below'] = 1
                triggers.append("✅ BB_midband_below")
                short_signals += 1
        
        # 4. Supertrend_flip_bull
        ema50 = safe_calculate_ema(ohlcv_4h, 50)
        ema200 = safe_calculate_ema(ohlcv_4h, 200)
        if ema50 > 0 and ema200 > 0:
            if ema50 > ema200 and price > ema50:
                trigger_weights['Supertrend_flip_bull'] = 1
                triggers.append("✅ Supertrend_bull")
                long_signals += 1
            elif ema50 < ema200 and price < ema50:
                trigger_weights['Supertrend_flip_bear'] = 1
                triggers.append("✅ Supertrend_bear")
                short_signals += 1
        
        # 5. ATR_volatility_drop_entry
        atr = safe_calculate_atr(ohlcv_4h)
        atr_prev = safe_calculate_atr(ohlcv_4h[:-10])
        if atr > 0 and atr_prev > 0 and atr < atr_prev * 0.8:
            trigger_weights['ATR_volatility_drop_entry'] = 1
            triggers.append("✅ ATR_volatility_drop")
            long_signals += 1
        
        # 6. Pullback_0_382_fib_entry
        highs = [x[2] for x in ohlcv_4h[-20:]]
        lows = [x[3] for x in ohlcv_4h[-20:]]
        high = max(highs)
        low = min(lows)
        if high > low:
            fib_382 = high - (high - low) * 0.382
            if abs(price - fib_382) / price < 0.01:
                trigger_weights['Pullback_0_382_fib_entry'] = 1
                triggers.append("✅ Fib_0.382")
                long_signals += 1
        
        # 7. Pullback_0_5_fib_entry
        if high > low:
            fib_50 = high - (high - low) * 0.5
            if abs(price - fib_50) / price < 0.01:
                trigger_weights['Pullback_0_5_fib_entry'] = 1
                triggers.append("✅ Fib_0.5")
                long_signals += 1
        
        # 8. Pullback_0_618_fib_entry
        if high > low:
            fib_618 = high - (high - low) * 0.618
            if abs(price - fib_618) / price < 0.01:
                trigger_weights['Pullback_0_618_fib_entry'] = 1
                triggers.append("✅ Fib_0.618")
                long_signals += 1
        
        # 9. Support_demand_zone_reaction
        recent_lows = [x[3] for x in ohlcv_4h[-10:]]
        support = min(recent_lows)
        if abs(price - support) / price < 0.02:
            trigger_weights['Support_demand_zone_reaction'] = 1
            triggers.append("✅ Support_zone")
            long_signals += 1
        
        # 10. Imbalance_fill_continuation
        closes = [x[4] for x in ohlcv_4h[-5:]]
        if len(closes) >= 5 and all(closes[i] < closes[i+1] for i in range(len(closes)-1)):
            trigger_weights['Imbalance_fill_continuation'] = 1
            triggers.append("✅ Imbalance_continuation")
            long_signals += 1
        
        # Compute weighted score
        raw_score = compute_weighted_score(trigger_weights, LOGIC_WEIGHTS)
        score = normalize_scores_to_10(raw_score, 18)
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        logger.error(f"Trend signal error: {e}")
    
    return {
        'score': round(score, 1),
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

# ==========================================
# TELEGRAM FORMATTING
# ==========================================
def telegram_formatter_style_c(symbol, mode, direction, result, data):
    """Format Telegram message with style C (emoji + bold + codeblock)"""
    
    emoji_map = {
        'QUICK': '⚡',
        'MID': '🔵',
        'TREND': '🟣'
    }
    
    dir_emoji = "🟢 LONG" if direction == 'long' else "🔴 SHORT"
    emoji = emoji_map.get(mode, '📊')
    
    price = data['price']
    tp_sl = calc_single_tp_sl(price, direction, mode)
    
    # Calculate liquidation info
    liq_info = calc_liquidation(price, tp_sl['sl'], tp_sl['leverage'], direction)
    
    # Format message
    message = f"""{emoji} <b>{mode} {dir_emoji} SIGNAL</b>
━━━━━━━━━━━━━━━━━━
<b>Pair:</b> {symbol}
<b>Entry:</b> {format_price_usd(price)}
<b>Target:</b> {format_price_usd(tp_sl['tp'])}
<b>Stop Loss:</b> {format_price_usd(tp_sl['sl'])}
<b>Leverage:</b> {tp_sl['leverage']}x

<b>Score:</b> {result['score']}/10
<b>Confidence:</b> {score_to_confidence(result['score'])}%

<b>Liquidation:</b> {format_price_usd(liq_info['liq_price'])}
<b>Liq Distance:</b> {liq_info['dist_pct']:.2f}%

<b>Triggers:</b>
{result['triggers']}

<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC
━━━━━━━━━━━━━━━━━━"""
    
    return message, tp_sl['code']

async def send_telegram_message(token, chat_id, message):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if not result.get('ok'):
                    logger.error(f"Telegram send failed: {result}")
                return result
    except Exception as e:
        logger.error(f"Telegram error: {e}")

async def send_copy_block(token, chat_id, code_block):
    """Send separate code block for easy copying"""
    await send_telegram_message(token, chat_id, code_block)