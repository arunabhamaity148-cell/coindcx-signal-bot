import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import aiohttp

class MarketData:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    async def get_all_data(self, symbol):
        """Fetch all market data"""
        ticker = await self.exchange.fetch_ticker(symbol)
        ohlcv_1m = await self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
        ohlcv_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
        ohlcv_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
        ohlcv_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
        ohlcv_4h = await self.exchange.fetch_ohlcv(symbol, '4h', limit=100)
        orderbook = await self.exchange.fetch_order_book(symbol, limit=20)
        
        return {
            'symbol': symbol,
            'price': ticker['last'],
            'volume': ticker['volume'],
            'ohlcv_1m': ohlcv_1m,
            'ohlcv_5m': ohlcv_5m,
            'ohlcv_15m': ohlcv_15m,
            'ohlcv_1h': ohlcv_1h,
            'ohlcv_4h': ohlcv_4h,
            'orderbook': orderbook,
            'spread': ((orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['bids'][0][0]) * 100
        }

def calculate_rsi(ohlcv, period=14):
    """Calculate RSI"""
    closes = [x[4] for x in ohlcv]
    df = pd.DataFrame({'close': closes})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(ohlcv):
    """Calculate MACD"""
    closes = [x[4] for x in ohlcv]
    df = pd.DataFrame({'close': closes})
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1], macd.iloc[-2], signal.iloc[-2]

def calculate_ema(ohlcv, period):
    """Calculate EMA"""
    closes = [x[4] for x in ohlcv]
    df = pd.DataFrame({'close': closes})
    return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_adx(ohlcv, period=14):
    """Calculate ADX"""
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

def calculate_bollinger_bands(ohlcv, period=20, std=2):
    """Calculate Bollinger Bands"""
    closes = [x[4] for x in ohlcv]
    df = pd.DataFrame({'close': closes})
    sma = df['close'].rolling(period).mean()
    std_dev = df['close'].rolling(period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]

def calculate_atr(ohlcv, period=14):
    """Calculate ATR"""
    highs = [x[2] for x in ohlcv]
    lows = [x[3] for x in ohlcv]
    closes = [x[4] for x in ohlcv]
    
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
    return df['tr'].rolling(period).mean().iloc[-1]

def calculate_vwap(ohlcv):
    """Calculate VWAP"""
    typical_prices = [(x[2] + x[3] + x[4]) / 3 for x in ohlcv]
    volumes = [x[5] for x in ohlcv]
    return sum([tp * v for tp, v in zip(typical_prices, volumes)]) / sum(volumes)

# ==========================================
# QUICK SIGNALS (10 logics)
# ==========================================
def calculate_quick_signals(data):
    """Calculate Quick signal score"""
    score = 0
    triggers = []
    direction = 'none'
    long_signals = 0
    short_signals = 0
    
    try:
        price = data['price']
        ohlcv_1m = data['ohlcv_1m']
        ohlcv_5m = data['ohlcv_5m']
        
        # 1. RSI_oversold_breakout
        rsi = calculate_rsi(ohlcv_1m)
        if 25 < rsi < 35:
            score += 1
            triggers.append("✅ RSI_oversold_breakout")
            long_signals += 1
        elif 65 < rsi < 75:
            score += 1
            triggers.append("✅ RSI_overbought_breakdown")
            short_signals += 1
        
        # 2. MACD_bullish_cross
        macd, signal, macd_prev, signal_prev = calculate_macd(ohlcv_5m)
        if macd > signal and macd_prev <= signal_prev:
            score += 1
            triggers.append("✅ MACD_bullish_cross")
            long_signals += 1
        elif macd < signal and macd_prev >= signal_prev:
            score += 1
            triggers.append("✅ MACD_bearish_cross")
            short_signals += 1
        
        # 3. Volume_spike_support
        volumes = [x[5] for x in ohlcv_1m[-20:]]
        avg_vol = np.mean(volumes[:-1])
        current_vol = volumes[-1]
        if current_vol > avg_vol * 2:
            score += 1
            triggers.append("✅ Volume_spike")
            long_signals += 1
        
        # 4. VWAP_reclaim_long
        vwap = calculate_vwap(ohlcv_5m)
        if price > vwap * 1.001:
            score += 1
            triggers.append("✅ VWAP_reclaim_long")
            long_signals += 1
        elif price < vwap * 0.999:
            score += 1
            triggers.append("✅ VWAP_below_short")
            short_signals += 1
        
        # 5. EMA_9_21_bull_cross
        ema9 = calculate_ema(ohlcv_5m, 9)
        ema21 = calculate_ema(ohlcv_5m, 21)
        if price > ema9 > ema21:
            score += 1
            triggers.append("✅ EMA_9_21_bull")
            long_signals += 1
        elif price < ema9 < ema21:
            score += 1
            triggers.append("✅ EMA_9_21_bear")
            short_signals += 1
        
        # 6. Orderblock_retest_long (simplified)
        candles = ohlcv_1m[-10:]
        for i in range(len(candles)-3):
            if candles[i][5] > np.mean([c[5] for c in candles]) * 1.5:
                score += 1
                triggers.append("✅ Orderblock_detected")
                long_signals += 1
                break
        
        # 7. Liquidity_sweep_long
        lows = [x[3] for x in ohlcv_1m[-5:]]
        if price < min(lows[:-1]) and price > lows[-2]:
            score += 1
            triggers.append("✅ Liquidity_sweep")
            long_signals += 1
        
        # 8. Bollinger_band_squeeze_break
        upper, mid, lower = calculate_bollinger_bands(ohlcv_5m)
        bb_width = (upper - lower) / mid
        if bb_width < 0.02:
            score += 1
            triggers.append("✅ BB_squeeze")
            if price > mid:
                long_signals += 1
            else:
                short_signals += 1
        
        # 9. Spread_tight_low_latency
        if data['spread'] < 0.03:
            score += 1
            triggers.append("✅ Tight_spread")
        
        # 10. Market_structure_HH_HL
        highs = [x[2] for x in ohlcv_5m[-5:]]
        if highs[-1] > highs[-2] > highs[-3]:
            score += 1
            triggers.append("✅ Higher_highs")
            long_signals += 1
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        print(f"Quick signal error: {e}")
    
    return {
        'score': score,
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

# ==========================================
# MID SIGNALS (10 logics)
# ==========================================
def calculate_mid_signals(data):
    """Calculate Mid signal score"""
    score = 0
    triggers = []
    direction = 'none'
    long_signals = 0
    short_signals = 0
    
    try:
        price = data['price']
        ohlcv_15m = data['ohlcv_15m']
        ohlcv_1h = data['ohlcv_1h']
        
        # 1. RSI_overbought_reversal
        rsi = calculate_rsi(ohlcv_15m)
        if rsi > 70:
            score += 1
            triggers.append("✅ RSI_overbought")
            short_signals += 1
        elif rsi < 30:
            score += 1
            triggers.append("✅ RSI_oversold")
            long_signals += 1
        
        # 2. MACD_hidden_bullish
        macd, signal, _, _ = calculate_macd(ohlcv_15m)
        if macd > 0 and macd > signal:
            score += 1
            triggers.append("✅ MACD_hidden_bull")
            long_signals += 1
        
        # 3. MACD_divergence_support (simplified)
        if macd < 0 and macd < signal:
            score += 1
            triggers.append("✅ MACD_divergence")
            long_signals += 1
        
        # 4. ADX_trend_strength_up
        adx = calculate_adx(ohlcv_1h)
        if adx > 25:
            score += 1
            triggers.append("✅ ADX_strong_trend")
            long_signals += 1
        
        # 5. Volume_delta_buy_pressure
        volumes = [x[5] for x in ohlcv_15m[-10:]]
        if volumes[-1] > np.mean(volumes) * 1.5:
            score += 1
            triggers.append("✅ Volume_delta_buy")
            long_signals += 1
        
        # 6. EMA_50_bounce
        ema50 = calculate_ema(ohlcv_1h, 50)
        if abs(price - ema50) / ema50 < 0.01:
            score += 1
            triggers.append("✅ EMA50_bounce")
            if price > ema50:
                long_signals += 1
            else:
                short_signals += 1
        
        # 7. EMA_200_bounce
        ema200 = calculate_ema(ohlcv_1h, 200)
        if abs(price - ema200) / ema200 < 0.015:
            score += 1
            triggers.append("✅ EMA200_bounce")
            if price > ema200:
                long_signals += 1
        
        # 8. FVG_immediate_fill
        candles = ohlcv_15m[-5:]
        for i in range(len(candles)-2):
            gap = candles[i+2][3] - candles[i][2]
            if gap > 0:
                score += 1
                triggers.append("✅ FVG_detected")
                long_signals += 1
                break
        
        # 9. Keltner_breakout_up
        atr = calculate_atr(ohlcv_1h)
        ema20 = calculate_ema(ohlcv_1h, 20)
        upper_keltner = ema20 + (2 * atr)
        if price > upper_keltner:
            score += 1
            triggers.append("✅ Keltner_breakout")
            long_signals += 1
        
        # 10. Trendline_break_retest
        closes = [x[4] for x in ohlcv_1h[-10:]]
        if closes[-1] > closes[-2] > closes[-3]:
            score += 1
            triggers.append("✅ Trendline_break")
            long_signals += 1
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        print(f"Mid signal error: {e}")
    
    return {
        'score': score,
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

# ==========================================
# TREND SIGNALS (10 logics)
# ==========================================
def calculate_trend_signals(data):
    """Calculate Trend signal score"""
    score = 0
    triggers = []
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
            score += 1
            triggers.append("✅ Breaker_block")
            long_signals += 1
        
        # 2. Chop_zone_exit_long
        adx = calculate_adx(ohlcv_4h)
        if adx > 20:
            score += 1
            triggers.append("✅ Chop_zone_exit")
            long_signals += 1
        
        # 3. Bollinger_midband_reject_flip
        upper, mid, lower = calculate_bollinger_bands(ohlcv_4h)
        if price > mid:
            score += 1
            triggers.append("✅ BB_midband_above")
            long_signals += 1
        elif price < mid:
            score += 1
            triggers.append("✅ BB_midband_below")
            short_signals += 1
        
        # 4. Supertrend_flip_bull (simplified using EMA)
        ema50 = calculate_ema(ohlcv_4h, 50)
        ema200 = calculate_ema(ohlcv_4h, 200)
        if ema50 > ema200 and price > ema50:
            score += 1
            triggers.append("✅ Supertrend_bull")
            long_signals += 1
        elif ema50 < ema200 and price < ema50:
            score += 1
            triggers.append("✅ Supertrend_bear")
            short_signals += 1
        
        # 5. ATR_volatility_drop_entry
        atr = calculate_atr(ohlcv_4h)
        atr_prev = calculate_atr(ohlcv_4h[:-10])
        if atr < atr_prev * 0.8:
            score += 1
            triggers.append("✅ ATR_volatility_drop")
            long_signals += 1
        
        # 6. Pullback_0_382_fib_entry
        highs = [x[2] for x in ohlcv_4h[-20:]]
        lows = [x[3] for x in ohlcv_4h[-20:]]
        high = max(highs)
        low = min(lows)
        fib_382 = high - (high - low) * 0.382
        if abs(price - fib_382) / price < 0.01:
            score += 1
            triggers.append("✅ Fib_0.382_level")
            long_signals += 1
        
        # 7. Pullback_0_5_fib_entry
        fib_50 = high - (high - low) * 0.5
        if abs(price - fib_50) / price < 0.01:
            score += 1
            triggers.append("✅ Fib_0.5_level")
            long_signals += 1
        
        # 8. Pullback_0_618_fib_entry
        fib_618 = high - (high - low) * 0.618
        if abs(price - fib_618) / price < 0.01:
            score += 1
            triggers.append("✅ Fib_0.618_level")
            long_signals += 1
        
        # 9. Support_demand_zone_reaction
        recent_lows = [x[3] for x in ohlcv_4h[-10:]]
        support = min(recent_lows)
        if abs(price - support) / price < 0.02:
            score += 1
            triggers.append("✅ Support_zone")
            long_signals += 1
        
        # 10. Imbalance_fill_continuation
        closes = [x[4] for x in ohlcv_4h[-5:]]
        if all(closes[i] < closes[i+1] for i in range(len(closes)-1)):
            score += 1
            triggers.append("✅ Imbalance_continuation")
            long_signals += 1
        
        # Determine direction
        if long_signals > short_signals:
            direction = 'long'
        elif short_signals > long_signals:
            direction = 'short'
        
    except Exception as e:
        print(f"Trend signal error: {e}")
    
    return {
        'score': score,
        'triggers': '\n'.join(triggers) if triggers else 'No triggers',
        'direction': direction
    }

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
                return await response.json()
    except Exception as e:
        print(f"Telegram error: {e}")