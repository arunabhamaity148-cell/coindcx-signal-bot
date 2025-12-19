# 📰 News Guard - Usage Guide

Complete guide to News Guard protection system.

## 🎯 Purpose

Protect trades from major economic news that cause 2-5% sudden spikes, avoiding stop loss hits despite perfect technical analysis.

## 🚫 What Gets Blocked

### High-Impact News Events
1. **US CPI** (Consumer Price Index)
   - Usually 2nd week of month
   - Time: 8:30 PM IST
   - Impact: 2-5% spike in crypto

2. **FOMC Meeting** (Federal Reserve Rate Decision)
   - Usually mid-month
   - Time: 2:00 AM IST (next day)
   - Impact: 3-7% volatility

3. **NFP** (Non-Farm Payroll)
   - First Friday of month
   - Time: 7:00 PM IST
   - Impact: 2-4% spike

4. **Fed Chair Major Speech**
   - Announced in advance
   - Variable timing
   - Impact: 1-3% spike

5. **Bitcoin Major Upgrade**
   - Hard forks, major updates
   - Announced weeks ahead
   - Impact: 5-10% volatility

6. **Ethereum Major Upgrade**
   - Network upgrades (e.g., Shanghai, Dencun)
   - Announced in advance
   - Impact: 3-8% volatility

## ⏱️ Block Window

**30 minutes before news**
**+**
**30 minutes after news**
**=**
**1 hour total protection**

### Why 30 Minutes?

- **Before**: Pre-positioning, fake breakouts
- **After**: Initial spike, reversal, stabilization
- **Total**: Avoid entire volatile period

## 📅 Auto-Loading Events

News Guard automatically loads:
- Current month's scheduled events
- Next month (if last week of current month)

### December 2025 Example
```
US CPI: Dec 11, 8:30 PM IST
NFP: Dec 5, 7:00 PM IST
FOMC: Dec 18, 2:00 AM IST
```

## 🔧 Manual Event Addition

### Add One-Time Event
```python
from news_guard import news_guard
from datetime import datetime
import pytz

ist = pytz.timezone('Asia/Kolkata')

# Add Fed Chair speech
news_guard.add_news_event(
    'FED_SPEECH',
    datetime(2025, 12, 20, 21, 0, tzinfo=ist),
    'Jerome Powell Speech on Inflation'
)
```

### Add Crypto Event
```python
# Bitcoin upgrade
news_guard.add_crypto_event(
    'BTC',
    datetime(2026, 1, 15, 12, 0, tzinfo=ist),
    'Bitcoin Taproot Upgrade'
)

# Ethereum upgrade
news_guard.add_crypto_event(
    'ETH',
    datetime(2026, 2, 10, 14, 30, tzinfo=ist),
    'Ethereum Dencun Upgrade'
)
```

## 📊 Check News Status

### In Code
```python
from news_guard import news_guard

# Check if blocked now
is_blocked, reason = news_guard.is_blocked()

if is_blocked:
    print(f"🚫 Trading blocked: {reason}")
else:
    print("✅ Clear to trade")
```

### View Upcoming Events
```python
# Get events in next 24 hours
upcoming = news_guard.get_upcoming_events(hours=24)

for event in upcoming:
    print(f"{event['name']} at {event['time']}")
```

### Print Full Schedule
```python
# Print all upcoming events
news_guard.print_upcoming_events(hours=48)
```

## 🔄 Monthly Update Process

**Important**: Update news schedule monthly!

### Method 1: Auto-Load (Recommended)
Bot automatically loads current month on startup.

### Method 2: Manual Update
Edit `news_guard.py` → `load_monthly_events()`:

```python
def load_monthly_events(self, year: int, month: int):
    
    # Add your month here
    if year == 2026 and month == 3:
        
        self.add_news_event(
            'US_CPI',
            datetime(2026, 3, 12, 20, 30, tzinfo=self.ist),
            'US Consumer Price Index'
        )
        
        self.add_news_event(
            'FOMC',
            datetime(2026, 3, 19, 2, 0, tzinfo=self.ist),
            'Federal Reserve Rate Decision'
        )
        
        self.add_news_event(
            'NFP',
            datetime(2026, 3, 6, 19, 0, tzinfo=self.ist),
            'Non-Farm Payroll Report'
        )
```

## 📈 Real-World Example

### Without News Guard
```
Dec 11, 8:25 PM - Bot generates LONG signal
Dec 11, 8:30 PM - US CPI released (bad data)
Dec 11, 8:31 PM - BTC drops 3% in 2 minutes
Dec 11, 8:32 PM - Stop loss hit ❌
```

### With News Guard
```
Dec 11, 8:00 PM - News Guard activates (30 min before CPI)
Dec 11, 8:25 PM - Bot tries to signal → BLOCKED 🚫
Dec 11, 9:00 PM - News window ends, trading resumes ✅
Dec 11, 9:15 PM - Market stable, bot generates safe signal
```

## 🎯 Impact on Trading

### Signal Reduction
- **Without News Guard**: ~10-12 signals/day (3-4 might hit SL due to news)
- **With News Guard**: ~8-10 signals/day (all safer, better accuracy)

### Accuracy Improvement
- **Estimated**: +10-15% win rate
- **Reason**: Avoid sudden spikes that break perfect setups

## 📱 Telegram Notifications

When news blocks trading:
```
🚫 NEWS GUARD ACTIVE

Trading paused due to upcoming:
US CPI (⏳ 15 min before)

Signals will resume after news window.
Stay safe! 🛡️
```

## ⚙️ Configuration

### Adjust Block Window
In `news_guard.py`:
```python
BLOCK_WINDOW_MINUTES = 30  # Change to 45 or 60 if needed
```

### Disable News Guard (NOT RECOMMENDED)
In `signal_generator.py`, comment out:
```python
# is_blocked, reason = news_guard.is_blocked()
# if is_blocked:
#     return None
```

## 🔍 Where to Find News Schedules

### Economic Calendar
- [Investing.com Economic Calendar](https://www.investing.com/economic-calendar/)
- [ForexFactory Calendar](https://www.forexfactory.com/calendar)
- [TradingView Calendar](https://www.tradingview.com/economic-calendar/)

### Crypto Events
- [CoinMarketCal](https://coinmarketcal.com/)
- [CryptoCalendar](https://www.cryptocompare.com/coins/guides/what-are-the-top-crypto-calendars/)
- Official project Twitter/Discord

## 📊 Statistics Tracking

### View News Guard Stats
```python
status = news_guard.get_status()

print(f"Blocked: {status['blocked']}")
print(f"Reason: {status['reason']}")
print(f"Total events: {status['total_events']}")
print(f"Upcoming (24h): {status['upcoming_24h']}")
```

## 🧹 Cleanup

Old events auto-cleanup:
- Events older than 1 hour are removed
- Keeps system memory clean
- Runs automatically

## ⚠️ Important Notes

### 1. Manual Updates Required
Bot can't predict news dates automatically. Update monthly schedule manually.

### 2. Surprise Events
Some events are announced suddenly (emergency Fed meetings). Add them immediately:
```python
news_guard.add_news_event(
    'FOMC',
    datetime(2025, 12, 25, 2, 0, tzinfo=ist),
    'Emergency Fed Meeting'
)
```

### 3. Time Zone Accuracy
All times stored in IST (Indian Standard Time). Automatically converts from US Eastern.

### 4. Don't Over-Block
Only block HIGH-IMPACT events. Small news releases don't need blocking.

## 🎓 Best Practices

1. **Update Monthly**: First day of each month, update schedule
2. **Check Calendar**: Review economic calendar weekly
3. **Add Crypto Events**: Monitor major upgrade announcements
4. **Test Block Windows**: Verify timing accuracy
5. **Monitor Impact**: Track if news guard improves results

## 🔄 Integration Flow

```
Bot scans market
    ↓
News Guard checks time
    ↓
Is major news within 30 min? → YES → BLOCK signal 🚫
    ↓ NO
Continue with signal generation ✅
```

## 📈 Expected Results

**Before News Guard**:
- 10-12 signals/day
- 65-70% accuracy
- 3-4 news-related SL hits/month

**After News Guard**:
- 8-10 signals/day
- 75-80% accuracy
- 0-1 news-related SL hits/month

**Trade-off**: Fewer signals, but higher quality and safety.

---

## 🆘 Troubleshooting

### "No events loaded"
```python
# Manually load current month
news_guard.load_monthly_events(2025, 12)
```

### "Wrong time zone"
Check IST timezone:
```python
from datetime import datetime
import pytz

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
print(now_ist)
```

### "Events not blocking"
Verify event time:
```python
news_guard.print_upcoming_events(hours=48)
```

---

**News Guard = Your Trading Shield! 🛡️**

Protect profits from unexpected spikes. Update schedule monthly for best results.