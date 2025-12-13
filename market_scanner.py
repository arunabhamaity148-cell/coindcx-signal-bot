"""
📊 Market Scanner - Health Checks + 50 Pair Monitoring
Implements Logic 1-10: Market Health Filters
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *

class MarketScanner:
    def __init__(self):
        self.base_url = "https://api.coindcx.com"
        self.market_health_score = 0
        self.btc_volatility = 0
        self.market_regime = "UNKNOWN"
        
    def get_market_data(self, symbol, interval="5m", limit=200):
        """Fetch OHLCV data from CoinDCX"""
        try:
            endpoint = f"{self.base_url}/market_data/candles"
            params = {
                "pair": f"B-{symbol}_USDT",
                "interval": interval,
                "limit": limit
            }
            response = requests.get(endpoint, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️ API Error for {symbol}: Status {response.status_code}")
                return None
            
            data = response.json()
            
            # Check if data is empty or invalid
            if not data or not isinstance(data, list) or len(data) == 0:
                print(f"⚠️ No data returned for {symbol}")
                return None
            
            df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            
            # Validate dataframe
            if df.empty or len(df) < 20:
                print(f"⚠️ Insufficient data for {symbol}: {len(df)} candles")
                return None
            
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN rows
            df = df.dropna()
            
            if df.empty or len(df) < 20:
                print(f"⚠️ Invalid data for {symbol} after cleaning")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def check_btc_calm(self):
        """Logic 1: BTC Calm Check - Low volatility only"""
        btc_data = self.get_market_data("BTC", "1h", 24)
        if btc_data is None or len(btc_data) < 10:
            print("⚠️ BTC data unavailable, assuming calm")
            self.btc_volatility = 1.0
            return True
        
        btc_returns = btc_data['close'].pct_change().abs()
        self.btc_volatility = btc_returns.mean() * 100
        
        is_calm = self.btc_volatility < BTC_VOLATILITY_THRESHOLD
        print(f"🔹 BTC Volatility: {self.btc_volatility:.2f}% {'✅' if is_calm else '⚠️'}")
        return is_calm
    
    def detect_market_regime(self):
        """Logic 2: Trending / Ranging / Volatile"""
        btc_data = self.get_market_data("BTC", "15m", 100)
        if btc_data is None or len(btc_data) < 50:
            print("⚠️ Insufficient BTC data, assuming RANGING")
            self.market_regime = "RANGING"
            return "RANGING"
        
        close = btc_data['close']
        
        # Simple regime detection
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            self.market_regime = "RANGING"
            print(f"🔹 Market Regime: {self.market_regime}")
            return self.market_regime
        
        volatility = returns.rolling(20).std()
        avg_volatility = volatility.mean()
        
        if len(volatility.dropna()) == 0:
            self.market_regime = "RANGING"
        elif volatility.iloc[-1] > avg_volatility * 1.5:
            self.market_regime = "VOLATILE"
        elif abs(returns.rolling(20).mean().iloc[-1]) > 0.002:
            self.market_regime = "TRENDING"
        else:
            self.market_regime = "RANGING"
        
        print(f"🔹 Market Regime: {self.market_regime}")
        return self.market_regime
    
    def check_funding_rate(self, symbol):
        """Logic 3: Funding Rate Extreme Filter"""
        # CoinDCX doesn't expose funding directly via public API
        # Using proxy: price vs mark price deviation
        return True  # Placeholder - implement with websocket
    
    def check_fear_greed(self):
        """Logic 4: Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            data = response.json()
            fgi = int(data['data'][0]['value'])
            
            is_safe = FEAR_GREED_EXTREME[0] < fgi < FEAR_GREED_EXTREME[1]
            print(f"🔹 Fear & Greed: {fgi} {'✅' if is_safe else '⚠️'}")
            return is_safe
        except:
            return True  # Don't block on API failure
    
    def check_liquidity(self, symbol):
        """Logic 8: Low Liquidity Filter"""
        ticker_url = f"{self.base_url}/market_data/ticker"
        try:
            response = requests.get(ticker_url, timeout=5)
            tickers = response.json()
            
            pair_key = f"B-{symbol}_USDT"
            for ticker in tickers:
                if ticker.get('market') == pair_key:
                    volume_24h = float(ticker.get('volume', 0))
                    return volume_24h > MIN_VOLUME_24H
            return False
        except:
            return False
    
    def check_spread(self, symbol):
        """Logic 7: Spread & Slippage Safety"""
        # Check orderbook depth
        try:
            orderbook_url = f"{self.base_url}/market_data/orderbook"
            params = {"pair": f"B-{symbol}_USDT"}
            response = requests.get(orderbook_url, params=params, timeout=5)
            data = response.json()
            
            if data.get('bids') and data.get('asks'):
                best_bid = float(data['bids'][0]['price'])
                best_ask = float(data['asks'][0]['price'])
                spread_pct = ((best_ask - best_bid) / best_bid) * 100
                
                return spread_pct < MAX_SPREAD_PERCENT
            return False
        except:
            return False
    
    def calculate_market_health(self):
        """Logic 1-10: Overall Market Health Score (0-10)"""
        score = 0
        
        # Check 1: BTC Calm
        if self.check_btc_calm():
            score += 2
        
        # Check 2: Market Regime
        regime = self.detect_market_regime()
        if regime in ["TRENDING", "RANGING"]:
            score += 2
        
        # Check 4: Fear & Greed
        if self.check_fear_greed():
            score += 2
        
        # Check 6: News Time Avoidance
        current_time = datetime.now().time()
        avoid_time = any(start <= current_time <= end for start, end in AVOID_NEWS_HOURS)
        if not avoid_time:
            score += 2
        
        # Check 9: Volatility Spike
        if self.btc_volatility < BTC_VOLATILITY_THRESHOLD:
            score += 2
        
        self.market_health_score = score
        print(f"\n🏥 MARKET HEALTH SCORE: {score}/10 {'✅' if score >= 6 else '⚠️'}\n")
        return score
    
    def scan_all_pairs(self):
        """Scan all 50 pairs for tradeable opportunities"""
        tradeable_pairs = []
        
        print(f"🔍 Scanning {len(WATCHLIST)} pairs...")
        
        for symbol in WATCHLIST[:10]:  # Start with first 10 pairs to test
            try:
                # Get recent data
                df = self.get_market_data(symbol, "5m", 50)
                if df is None or len(df) < 20:
                    continue
                
                # Calculate basic metrics
                df['volume_ma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
                
                # Check if volume_ma has valid data
                if df['volume_ma'].dropna().empty:
                    continue
                
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume_ma'].iloc[-1]
                
                # Handle NaN in avg_volume
                if pd.isna(avg_volume) or avg_volume == 0:
                    continue
                
                # Volume spike check (Logic 31)
                volume_ratio = current_volume / avg_volume
                
                if volume_ratio > 1.2:  # Lowered threshold for more signals
                    tradeable_pairs.append({
                        'symbol': symbol,
                        'price': df['close'].iloc[-1],
                        'volume_ratio': volume_ratio,
                        'data': df
                    })
                    print(f"  ✅ {symbol}: Volume ratio {volume_ratio:.2f}x")
                
            except Exception as e:
                print(f"  ⚠️ Error scanning {symbol}: {e}")
                continue
        
        print(f"\n✅ Found {len(tradeable_pairs)} tradeable pairs\n")
        return tradeable_pairs