from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pytz

class NewsGuard:
    """
    News Guard - Protects trades from major economic events
    Blocks signals 30 min before and after high-impact news
    """
    
    # Major news events that cause 2-5% spikes
    MAJOR_NEWS_EVENTS = {
        'US_CPI': 'US CPI (Inflation Data)',
        'FOMC': 'FOMC Meeting / Rate Decision',
        'NFP': 'Non-Farm Payroll',
        'FED_SPEECH': 'Fed Chair Major Speech',
        'BTC_UPGRADE': 'Bitcoin Major Upgrade',
        'ETH_UPGRADE': 'Ethereum Major Upgrade',
        'CRYPTO_REGULATION': 'Major Crypto Regulation News'
    }
    
    # Block window (minutes before and after news)
    BLOCK_WINDOW_MINUTES = 30
    
    def __init__(self):
        # Timezone for news events (US Eastern Time)
        self.us_eastern = pytz.timezone('US/Eastern')
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Scheduled news events (manually updated or from API)
        self.scheduled_events: List[Dict] = []
    
    def add_news_event(self, event_type: str, event_time: datetime, description: str = ""):
        """
        Add a scheduled news event
        
        Args:
            event_type: Type from MAJOR_NEWS_EVENTS
            event_time: Event datetime in IST
            description: Optional description
        """
        
        if event_type not in self.MAJOR_NEWS_EVENTS:
            print(f"⚠️ Unknown event type: {event_type}")
            return
        
        event = {
            'type': event_type,
            'name': self.MAJOR_NEWS_EVENTS[event_type],
            'time': event_time,
            'description': description,
            'block_start': event_time - timedelta(minutes=self.BLOCK_WINDOW_MINUTES),
            'block_end': event_time + timedelta(minutes=self.BLOCK_WINDOW_MINUTES)
        }
        
        self.scheduled_events.append(event)
        print(f"✅ News event added: {event['name']} at {event_time.strftime('%Y-%m-%d %H:%M IST')}")
    
    def is_blocked(self, check_time: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
        """
        Check if trading is blocked due to news
        
        Args:
            check_time: Time to check (default: now)
        
        Returns:
            (is_blocked, reason)
        """
        
        if check_time is None:
            check_time = datetime.now()
        
        # Make timezone-aware if not already
        if check_time.tzinfo is None:
            check_time = self.ist.localize(check_time)
        
        # Check against all scheduled events
        for event in self.scheduled_events:
            if event['block_start'] <= check_time <= event['block_end']:
                time_to_event = event['time'] - check_time
                
                if time_to_event.total_seconds() > 0:
                    status = f"⏳ {int(time_to_event.total_seconds() / 60)} min before"
                else:
                    status = f"⏳ {int(abs(time_to_event.total_seconds()) / 60)} min after"
                
                reason = f"{event['name']} ({status})"
                return True, reason
        
        return False, None
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """
        Get upcoming news events in next N hours
        
        Args:
            hours: Look-ahead window
        
        Returns:
            List of upcoming events
        """
        
        now = datetime.now()
        if now.tzinfo is None:
            now = self.ist.localize(now)
        
        cutoff = now + timedelta(hours=hours)
        
        upcoming = [
            event for event in self.scheduled_events
            if now <= event['time'] <= cutoff
        ]
        
        # Sort by time
        upcoming.sort(key=lambda x: x['time'])
        
        return upcoming
    
    def clean_old_events(self):
        """Remove past events to keep list clean"""
        
        now = datetime.now()
        if now.tzinfo is None:
            now = self.ist.localize(now)
        
        # Keep only events within last 1 hour (for logging)
        cutoff = now - timedelta(hours=1)
        
        self.scheduled_events = [
            event for event in self.scheduled_events
            if event['time'] >= cutoff
        ]
    
    def load_monthly_events(self, year: int, month: int):
        """
        Load common monthly events (US CPI, NFP, FOMC)
        Must be manually updated each month
        
        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)
        """
        
        # Example: December 2025 events
        if year == 2025 and month == 12:
            
            # US CPI - Usually 2nd week, 8:30 PM IST
            self.add_news_event(
                'US_CPI',
                datetime(2025, 12, 11, 20, 30, tzinfo=self.ist),
                'US Consumer Price Index'
            )
            
            # FOMC Meeting - Usually mid-month, 2:00 AM IST next day
            self.add_news_event(
                'FOMC',
                datetime(2025, 12, 18, 2, 0, tzinfo=self.ist),
                'Federal Reserve Rate Decision'
            )
            
            # NFP - First Friday, 7:00 PM IST
            self.add_news_event(
                'NFP',
                datetime(2025, 12, 5, 19, 0, tzinfo=self.ist),
                'Non-Farm Payroll Report'
            )
        
        # January 2026 events
        elif year == 2026 and month == 1:
            
            self.add_news_event(
                'US_CPI',
                datetime(2026, 1, 14, 20, 30, tzinfo=self.ist),
                'US Consumer Price Index'
            )
            
            self.add_news_event(
                'FOMC',
                datetime(2026, 1, 29, 2, 0, tzinfo=self.ist),
                'Federal Reserve Rate Decision'
            )
            
            self.add_news_event(
                'NFP',
                datetime(2026, 1, 9, 19, 0, tzinfo=self.ist),
                'Non-Farm Payroll Report'
            )
        
        print(f"📅 Loaded news events for {year}-{month:02d}")
    
    def add_crypto_event(self, coin: str, event_time: datetime, description: str):
        """
        Add cryptocurrency-specific event
        
        Args:
            coin: 'BTC' or 'ETH'
            event_time: Event time in IST
            description: Event description
        """
        
        event_type = 'BTC_UPGRADE' if coin == 'BTC' else 'ETH_UPGRADE'
        self.add_news_event(event_type, event_time, description)
    
    def get_status(self) -> Dict:
        """Get current news guard status"""
        
        is_blocked, reason = self.is_blocked()
        upcoming = self.get_upcoming_events(hours=24)
        
        return {
            'blocked': is_blocked,
            'reason': reason,
            'total_events': len(self.scheduled_events),
            'upcoming_24h': len(upcoming),
            'next_event': upcoming[0] if upcoming else None
        }
    
    def print_upcoming_events(self, hours: int = 48):
        """Print upcoming news events"""
        
        upcoming = self.get_upcoming_events(hours=hours)
        
        if not upcoming:
            print(f"📅 No major news in next {hours} hours")
            return
        
        print(f"\n📰 UPCOMING NEWS EVENTS (Next {hours}h):")
        print("=" * 60)
        
        for event in upcoming:
            time_str = event['time'].strftime('%Y-%m-%d %H:%M IST')
            print(f"🔔 {event['name']}")
            print(f"   Time: {time_str}")
            print(f"   Block: {event['block_start'].strftime('%H:%M')} - {event['block_end'].strftime('%H:%M')}")
            if event['description']:
                print(f"   Info: {event['description']}")
            print("-" * 60)


# Singleton instance
news_guard = NewsGuard()


# Auto-load current month events
def auto_load_current_month():
    """Automatically load current month's events"""
    now = datetime.now()
    news_guard.load_monthly_events(now.year, now.month)
    
    # Also load next month if we're in last week
    if now.day >= 25:
        next_month = now.month + 1 if now.month < 12 else 1
        next_year = now.year if now.month < 12 else now.year + 1
        news_guard.load_monthly_events(next_year, next_month)


# Auto-load on import
auto_load_current_month()