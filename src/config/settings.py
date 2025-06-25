"""
Settings configuration for institutional-grade trading system
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Configuration settings for the trading system"""
    
    def __init__(self):
        self.KITE_API_KEY = os.getenv('KITE_API_KEY')
        self.KITE_API_SECRET = os.getenv('KITE_API_SECRET')
        self.KITE_USER_ID = os.getenv('KITE_USER_ID')
        self.KITE_PASSWORD = os.getenv('KITE_PASSWORD')
        self.KITE_TOTP_SECRET = os.getenv('KITE_TOTP_SECRET')
        
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        
        self.PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
        
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        self.ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
        
        self.DHAN_CLIENT_ID = os.getenv('DHAN_CLIENT_ID')
        self.DHAN_ACCESS_TOKEN = os.getenv('DHAN_ACCESS_TOKEN')
        
        self.CONFIDENCE_THRESHOLD = 60.0
        self.MAX_DAILY_LOSS = 50000
        self.MAX_POSITION_SIZE = 100000
        
        self.MARKET_OPEN_TIME = "09:15"
        self.MARKET_CLOSE_TIME = "15:30"
        
        self.ANALYSIS_INTERVAL = 30
        self.DATA_REFRESH_INTERVAL = 30
        
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = "trading_system.log"
