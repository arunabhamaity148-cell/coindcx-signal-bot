"""
Quick script to find correct CoinDCX pair names
Run this once to see available pairs
"""

import requests
import json

def find_coindcx_pairs():
    """Find all available CoinDCX trading pairs"""
    
    print("🔍 Fetching CoinDCX pairs...\n")
    
    try:
        # Method 1: Ticker endpoint
        url = "https://api.coindcx.com/exchange/ticker"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tickers = response.json()
        
        print(f"✅ Found {len(tickers)} pairs\n")
        
        # Filter for major coins
        major_coins = ['BTC', 'ETH', 'SOL', 'MATIC', 'ADA', 'DOGE']
        
        print("📊 MAJOR PAIRS (USDT):")
        print("=" * 60)
        
        usdt_pairs = []
        for ticker in tickers:
            market = ticker.get('market', '')
            
            # Check if it's a USDT pair
            if 'USDT' in market:
                for coin in major_coins:
                    if coin in market:
                        last_price = float(ticker.get('last_price', 0))
                        volume = float(ticker.get('volume', 0))
                        
                        print(f"Market: {market:15} | Price: ${last_price:>10,.2f} | Vol: {volume:>12,.0f}")
                        usdt_pairs.append(market)
                        break
        
        print("\n" + "=" * 60)
        print("\n📋 RECOMMENDED PAIRS FOR BOT:")
        print("=" * 60)
        
        for pair in usdt_pairs[:6]:  # Top 6
            print(f"'{pair}',")
        
        print("\n" + "=" * 60)
        
        # Also check INR pairs
        print("\n📊 MAJOR PAIRS (INR):")
        print("=" * 60)
        
        inr_pairs = []
        for ticker in tickers:
            market = ticker.get('market', '')
            
            if 'INR' in market:
                for coin in major_coins:
                    if coin in market:
                        last_price = float(ticker.get('last_price', 0))
                        volume = float(ticker.get('volume', 0))
                        
                        print(f"Market: {market:15} | Price: ₹{last_price:>10,.2f} | Vol: {volume:>12,.0f}")
                        inr_pairs.append(market)
                        break
        
        print("\n" + "=" * 60)
        
        return usdt_pairs, inr_pairs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []


def check_futures_pairs():
    """Check if futures pairs are available"""
    
    print("\n\n🔮 CHECKING FUTURES PAIRS:")
    print("=" * 60)
    
    try:
        url = "https://api.coindcx.com/exchange/v1/markets_details"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        markets = response.json()
        
        futures_pairs = []
        for market in markets:
            symbol = market.get('symbol', '')
            
            # Check for futures indicators
            if any(indicator in symbol for indicator in ['B-', 'F-', 'PERP', 'FUTURE']):
                pair = market.get('pair', symbol)
                base = market.get('base_currency_short_name', '')
                target = market.get('target_currency_short_name', '')
                
                print(f"Symbol: {symbol:20} | Pair: {pair:15} | {base}/{target}")
                futures_pairs.append(symbol)
        
        if not futures_pairs:
            print("⚠️ No futures pairs found in this endpoint")
        else:
            print(f"\n✅ Found {len(futures_pairs)} potential futures pairs")
        
        return futures_pairs
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


if __name__ == "__main__":
    print("\n" + "="*60)
    print("    COINDCX PAIR FINDER")
    print("="*60 + "\n")
    
    usdt_pairs, inr_pairs = find_coindcx_pairs()
    futures_pairs = check_futures_pairs()
    
    print("\n\n✅ SUMMARY:")
    print("=" * 60)
    print(f"USDT Pairs Found: {len(usdt_pairs)}")
    print(f"INR Pairs Found: {len(inr_pairs)}")
    print(f"Futures Pairs Found: {len(futures_pairs)}")
    print("=" * 60)
    
    print("\n💡 TIP: Copy the pair names exactly as shown above")
    print("      and update config.py PAIRS list\n")