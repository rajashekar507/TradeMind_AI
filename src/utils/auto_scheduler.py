"""
TradeMind_AI: Holiday-Aware Auto Scheduler
Automatically skips trading on Indian stock market holidays
"""

import schedule
import time
import subprocess
import sys
import os
import requests
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List
import threading
import signal
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class HolidayAwareScheduler:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.is_running = True
        self.last_heartbeat = datetime.now()
        
        # Indian Stock Market Holidays 2025 (NSE/BSE Official)
        self.market_holidays = {
            # Format: "YYYY-MM-DD": "Holiday Name"
            "2025-01-26": "Republic Day",
            "2025-03-14": "Holi",
            "2025-03-31": "Ram Navami", 
            "2025-04-14": "Mahavir Jayanti",
            "2025-04-18": "Good Friday",
            "2025-05-01": "Maharashtra Day",
            "2025-05-12": "Buddha Purnima",
            "2025-06-16": "Eid ul-Adha",
            "2025-08-15": "Independence Day",
            "2025-08-27": "Janmashtami",
            "2025-09-07": "Ganesh Chaturthi",
            "2025-10-02": "Gandhi Jayanti",
            "2025-10-21": "Dussehra",
            "2025-11-01": "Diwali Balipratipada",
            "2025-11-04": "Diwali",
            "2025-11-05": "Govardhan Puja",
            "2025-11-24": "Guru Nanak Jayanti",
            "2025-12-25": "Christmas"
        }
        
        # Special trading sessions (Muhurat trading, etc.)
        self.special_sessions = {
            "2025-11-01": {
                "type": "Muhurat Trading",
                "timings": "18:00-19:00",
                "enabled": False  # Disable by default
            }
        }
        
        self.scheduler_health = {
            'started_at': datetime.now(),
            'total_trades_executed': 0,
            'holidays_skipped': 0,
            'successful_notifications': 0,
            'failed_notifications': 0,
            'last_error': None,
            'uptime_hours': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info("STOP - Shutdown signal received. Stopping scheduler...")
        self.is_running = False
        self.send_notification_safe("SCHEDULER STOPPED - Graceful shutdown completed")
        sys.exit(0)
        
    def is_market_holiday(self, check_date: date = None) -> tuple[bool, str]:
        """
        Check if given date is a market holiday
        Returns: (is_holiday, holiday_name)
        """
        if check_date is None:
            check_date = date.today()
            
        date_str = check_date.strftime("%Y-%m-%d")
        
        # Check if it's a weekend
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return True, "Weekend"
            
        # Check official holidays
        if date_str in self.market_holidays:
            return True, self.market_holidays[date_str]
            
        return False, ""
        
    def is_trading_day(self) -> tuple[bool, str]:
        """
        Check if today is a trading day
        Returns: (is_trading_day, reason)
        """
        today = date.today()
        is_holiday, holiday_name = self.is_market_holiday(today)
        
        if is_holiday:
            return False, holiday_name
        else:
            return True, "Regular Trading Day"
            
    def get_next_trading_day(self) -> tuple[date, int]:
        """
        Get the next trading day
        Returns: (next_trading_date, days_away)
        """
        current_date = date.today()
        days_checked = 0
        max_days_to_check = 10
        
        while days_checked < max_days_to_check:
            current_date += timedelta(days=1)
            days_checked += 1
            
            is_holiday, _ = self.is_market_holiday(current_date)
            if not is_holiday:
                return current_date, days_checked
                
        return current_date, days_checked
        
    def send_notification_safe(self, message: str, max_retries: int = 3) -> bool:
        """Send Telegram notification with robust error handling"""
        if not self.bot_token or not self.chat_id:
            logging.warning("Telegram credentials missing - notification skipped")
            return False
            
        for attempt in range(max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                payload = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    logging.info(f"NOTIFICATION SENT: {message[:50]}...")
                    self.scheduler_health['successful_notifications'] += 1
                    return True
                else:
                    logging.warning(f"Telegram API error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logging.warning(f"Telegram timeout on attempt {attempt + 1}")
            except Exception as e:
                logging.error(f"Telegram error on attempt {attempt + 1}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        self.scheduler_health['failed_notifications'] += 1
        return False
        
    def trading_session_wrapper(self, session_name: str):
        """
        Wrapper for trading sessions that checks for holidays
        """
        # Check if today is a trading day
        is_trading, reason = self.is_trading_day()
        
        if not is_trading:
            # It's a holiday - skip trading
            self.scheduler_health['holidays_skipped'] += 1
            
            logging.info(f"HOLIDAY SKIP - {session_name} skipped: {reason}")
            
            # Send holiday notification
            next_trading_day, days_away = self.get_next_trading_day()
            holiday_message = (
                f"<b>TRADING SKIPPED - MARKET HOLIDAY</b>\\n"
                f"Today: {reason}\\n"
                f"Session: {session_name}\\n" 
                f"Next Trading Day: {next_trading_day.strftime('%Y-%m-%d')} ({days_away} days)\\n"
                f"Enjoy the holiday! 🎉"
            )
            
            self.send_notification_safe(holiday_message)
            return
        
        # It's a trading day - proceed with trading
        logging.info(f"TRADING DAY CONFIRMED - Proceeding with {session_name}")
        return self.run_master_trader_safe(session_name)
        
    def run_master_trader_safe(self, session_name: str) -> bool:
        """Execute master trader with comprehensive error handling"""
        try:
            logging.info(f"AI TRADER - Starting {session_name}...")
            
            if not os.path.exists('master_trader.py'):
                logging.error("ERROR - master_trader.py not found!")
                self.send_notification_safe("<b>ERROR:</b> master_trader.py not found!")
                return False
                
            result = subprocess.run([
                sys.executable, 'master_trader.py'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logging.info(f"SUCCESS - {session_name} completed successfully")
                self.scheduler_health['total_trades_executed'] += 1
                
                output_lines = result.stdout.strip().split('\\n')
                last_few_lines = '\\n'.join(output_lines[-5:]) if output_lines else "No output"
                
                self.send_notification_safe(
                    f"<b>TRADE EXECUTED</b>\\n"
                    f"Session: {session_name}\\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}\\n"
                    f"Output:\\n<code>{last_few_lines}</code>"
                )
                return True
            else:
                logging.error(f"FAILED - {session_name} failed: {result.stderr}")
                self.send_notification_safe(
                    f"<b>TRADE FAILED</b>\\n"
                    f"Session: {session_name}\\n"
                    f"Error: {result.stderr[:200]}"
                )
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"TIMEOUT - {session_name} execution timeout!")
            self.send_notification_safe(f"<b>TIMEOUT:</b> {session_name} took too long")
            return False
        except Exception as e:
            logging.error(f"ERROR - Unexpected error in {session_name}: {e}")
            self.send_notification_safe(f"<b>ERROR:</b> {session_name}\\n{str(e)[:200]}")
            return False
            
    def morning_session(self):
        """Morning trading session - 9:15 AM"""
        return self.trading_session_wrapper("Morning Session (9:15 AM)")
        
    def mid_morning_session(self):
        """Mid-morning trading session - 11:00 AM"""
        return self.trading_session_wrapper("Mid-Morning Session (11:00 AM)")
        
    def midday_session(self):
        """Midday trading session - 12:30 PM"""
        return self.trading_session_wrapper("Midday Session (12:30 PM)")
        
    def closing_session(self):
        """Pre-closing trading session - 3:00 PM"""
        return self.trading_session_wrapper("Closing Session (3:00 PM)")
        
    def send_daily_summary(self):
        """Send daily trading summary"""
        is_trading, reason = self.is_trading_day()
        
        uptime = datetime.now() - self.scheduler_health['started_at']
        self.scheduler_health['uptime_hours'] = round(uptime.total_seconds() / 3600, 2)
        
        summary = (
            f"<b>DAILY SUMMARY</b>\\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\\n"
            f"Market Status: {reason}\\n"
            f"Trades Today: {self.scheduler_health['total_trades_executed']}\\n"
            f"Holidays Skipped: {self.scheduler_health['holidays_skipped']}\\n"
            f"System Uptime: {self.scheduler_health['uptime_hours']} hours"
        )
        
        if not is_trading:
            next_trading_day, days_away = self.get_next_trading_day()
            summary += f"\\nNext Trading: {next_trading_day.strftime('%Y-%m-%d')} ({days_away} days)"
            
        self.send_notification_safe(summary)
        
    def setup_schedule(self):
        """Setup holiday-aware trading schedule"""
        try:
            schedule.clear()
            
            # BALANCED STRATEGY: 4 Sessions with holiday awareness
            # All sessions now use trading_session_wrapper for holiday checking
            
            # Session 1: Market Opening - 9:15 AM
            schedule.every().monday.at("09:15").do(self.morning_session)
            schedule.every().tuesday.at("09:15").do(self.morning_session)
            schedule.every().wednesday.at("09:15").do(self.morning_session)
            schedule.every().thursday.at("09:15").do(self.morning_session)
            schedule.every().friday.at("09:15").do(self.morning_session)
            
            # Session 2: Mid-Morning - 11:00 AM
            schedule.every().monday.at("11:00").do(self.mid_morning_session)
            schedule.every().tuesday.at("11:00").do(self.mid_morning_session)
            schedule.every().wednesday.at("11:00").do(self.mid_morning_session)
            schedule.every().thursday.at("11:00").do(self.mid_morning_session)
            schedule.every().friday.at("11:00").do(self.mid_morning_session)
            
            # Session 3: Post-Lunch - 12:30 PM
            schedule.every().monday.at("12:30").do(self.midday_session)
            schedule.every().tuesday.at("12:30").do(self.midday_session)
            schedule.every().wednesday.at("12:30").do(self.midday_session)
            schedule.every().thursday.at("12:30").do(self.midday_session)
            schedule.every().friday.at("12:30").do(self.midday_session)
            
            # Session 4: Pre-Closing - 3:00 PM
            schedule.every().monday.at("15:00").do(self.closing_session)
            schedule.every().tuesday.at("15:00").do(self.closing_session)
            schedule.every().wednesday.at("15:00").do(self.closing_session)
            schedule.every().thursday.at("15:00").do(self.closing_session)
            schedule.every().friday.at("15:00").do(self.closing_session)
            
            # Daily summary at 4:00 PM (runs even on holidays for status)
            schedule.every().monday.at("16:00").do(self.send_daily_summary)
            schedule.every().tuesday.at("16:00").do(self.send_daily_summary)
            schedule.every().wednesday.at("16:00").do(self.send_daily_summary)
            schedule.every().thursday.at("16:00").do(self.send_daily_summary)
            schedule.every().friday.at("16:00").do(self.send_daily_summary)
            
            logging.info("SUCCESS - Holiday-aware schedule configured!")
            logging.info("HOLIDAY PROTECTION - System will skip trading on market holidays")
            return True
            
        except Exception as e:
            logging.error(f"ERROR - Schedule setup failed: {e}")
            return False
            
    def run(self):
        """Main scheduler loop"""
        print("TradeMind_AI: Holiday-Aware Auto Scheduler")
        print("Automatically skips trading on Indian stock market holidays")
        print("=" * 60)
        
        # Check today's market status
        is_trading, reason = self.is_trading_day()
        if is_trading:
            logging.info(f"TRADING DAY - {reason}")
        else:
            logging.info(f"MARKET HOLIDAY - {reason}")
            next_trading_day, days_away = self.get_next_trading_day()
            logging.info(f"NEXT TRADING DAY - {next_trading_day} ({days_away} days away)")
        
        if not self.setup_schedule():
            logging.error("ERROR - Failed to setup schedule. Exiting...")
            return
            
        # Send startup notification
        startup_message = (
            f"<b>HOLIDAY-AWARE SCHEDULER STARTED!</b>\\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
            f"Today: {reason}\\n"
            f"Sessions: 9:15 AM, 11:00 AM, 12:30 PM, 3:00 PM\\n"
            f"Holiday Protection: ENABLED ✅\\n"
            f"Total Holidays in 2025: {len(self.market_holidays)}"
        )
        
        if not is_trading:
            next_trading_day, days_away = self.get_next_trading_day()
            startup_message += f"\\nNext Trading: {next_trading_day} ({days_away} days)"
            
        self.send_notification_safe(startup_message)
        
        logging.info("STARTED - Holiday-Aware Auto Scheduler running!")
        logging.info("PROTECTION - Automatic holiday detection enabled")
        logging.info("CONTROL - Press Ctrl+C to stop")
        
        # Main loop
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                schedule.run_pending()
                consecutive_errors = 0
                time.sleep(30)
                
            except KeyboardInterrupt:
                logging.info("INTERRUPT - Keyboard interrupt received")
                break
            except Exception as e:
                consecutive_errors += 1
                logging.error(f"ERROR - Scheduler error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logging.critical("CRITICAL - Too many errors. Stopping...")
                    break
                else:
                    time.sleep(60)
                    
        self.is_running = False
        logging.info("STOPPED - Holiday-Aware Scheduler stopped")

if __name__ == "__main__":
    try:
        scheduler = HolidayAwareScheduler()
        scheduler.run()
    except Exception as e:
        logging.critical(f"CRITICAL - Scheduler failure: {e}")
        print(f"Critical Error: {e}")
        sys.exit(1)