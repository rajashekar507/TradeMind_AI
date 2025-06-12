"""
TradeMind_AI: Centralized Configuration Constants
All trading constants and configurations in one place
Last Updated: June 2025
"""

from datetime import datetime
from typing import Dict, List

# ====================
# TRADING CONSTANTS
# ====================

# Lot sizes as per SEBI guidelines (Updated June 2025)
LOT_SIZES = {
    'NIFTY': 75,      # Changed from 25 on Dec 26, 2024
    'BANKNIFTY': 30,  # Changed from 15 on Dec 24, 2024
    'FINNIFTY': 40,   # For future expansion
    'MIDCPNIFTY': 75  # For future expansion
}

# Strike price gaps
STRIKE_GAPS = {
    'NIFTY': 50,
    'BANKNIFTY': 100,
    'FINNIFTY': 50,
    'MIDCPNIFTY': 25
}

# Market identifiers for Dhan API
SECURITY_IDS = {
    'NIFTY': 13,
    'BANKNIFTY': 25,
    'FINNIFTY': 27,
    'MIDCPNIFTY': 36
}

# Exchange segments
EXCHANGE_SEGMENTS = {
    'INDEX': 'IDX_I',
    'STOCK': 'NSE_EQ',
    'FNO': 'NSE_FNO'
}

# ====================
# RISK MANAGEMENT
# ====================

RISK_MANAGEMENT = {
    'MAX_RISK_PER_TRADE': 0.01,          # 1% risk per trade
    'MAX_DAILY_LOSS': 0.02,              # 2% daily loss limit
    'MAX_PORTFOLIO_RISK': 0.06,          # 6% total portfolio risk
    'DEFAULT_STOP_LOSS_PERCENT': 0.02,   # 2% default stop loss
    'TRAILING_STOP_PERCENT': 0.01,       # 1% trailing stop
    'MAX_TRADES_PER_DAY': 8,             # Maximum trades per day
    'MAX_CONCURRENT_POSITIONS': 4,        # Max open positions
    'MIN_RISK_REWARD_RATIO': 1.5         # Minimum 1:1.5 risk reward
}

# ====================
# API RATE LIMITING
# ====================

API_RATE_LIMITS = {
    'DHAN_API': {
        'calls_per_second': 10,
        'calls_per_minute': 200,
        'calls_per_hour': 5000,
        'retry_after': 60  # seconds
    },
    'NEWS_API': {
        'calls_per_minute': 30,
        'calls_per_day': 1000
    }
}

# ====================
# TRADING HOURS
# ====================

MARKET_HOURS = {
    'PRE_OPEN_START': '09:00',
    'PRE_OPEN_END': '09:15',
    'MARKET_OPEN': '09:15',
    'MARKET_CLOSE': '15:30',
    'POST_MARKET_START': '15:40',
    'POST_MARKET_END': '16:00'
}

# ====================
# TECHNICAL INDICATORS
# ====================

INDICATOR_SETTINGS = {
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'BOLLINGER': {
        'period': 20,
        'std_dev': 2
    },
    'STOCHASTIC': {
        'period': 14,
        'smooth_k': 3,
        'smooth_d': 3
    },
    'EMA': {
        'short': 9,
        'medium': 21,
        'long': 55
    }
}

# ====================
# OPTION GREEKS THRESHOLDS
# ====================

GREEKS_THRESHOLDS = {
    'MIN_DELTA': 0.20,      # Minimum delta for option selection
    'MAX_DELTA': 0.80,      # Maximum delta for option selection
    'MAX_THETA': -50,       # Maximum theta decay acceptable
    'MIN_VEGA': 0.5,        # Minimum vega for volatility plays
    'MAX_GAMMA': 0.10       # Maximum gamma risk
}

# ====================
# PAPER TRADING SETTINGS
# ====================

PAPER_TRADING = {
    'INITIAL_CAPITAL': 100000,  # 1 Lakh starting capital
    'BROKERAGE_PER_LOT': 20,   # Brokerage charges
    'ENABLE_SLIPPAGE': True,
    'SLIPPAGE_PERCENT': 0.0005, # 0.05% slippage
    'LOG_DIRECTORY': 'logs/paper_trades/',
    'SEPARATE_FROM_LIVE': True
}

# ====================
# LIVE TRADING SETTINGS
# ====================

LIVE_TRADING = {
    'REQUIRE_2FA': True,              # Two-factor authentication
    'CONFIRM_ORDERS': True,           # Manual confirmation for orders
    'MAX_ORDER_VALUE': 500000,        # 5 Lakh max order value
    'CIRCUIT_BREAKER_LOSS': 0.05,    # 5% portfolio loss triggers halt
    'AUDIT_LOG_DIRECTORY': 'logs/live_trades/',
    'ENABLE_ENCRYPTION': True
}

# ====================
# NOTIFICATION SETTINGS
# ====================

NOTIFICATIONS = {
    'TELEGRAM': {
        'ENABLED': True,
        'RATE_LIMIT': 30,  # messages per minute
        'PRIORITY_ALERTS_ONLY': False
    },
    'EMAIL': {
        'ENABLED': False,
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT': 587
    },
    'WEBHOOK': {
        'ENABLED': False,
        'URL': '',
        'SECRET_KEY': ''
    }
}

# ====================
# DATABASE SETTINGS
# ====================

DATABASE = {
    'TRADES_DB': 'data/trades_database.json',
    'BALANCE_DB': 'data/balance_history.json',
    'PATTERNS_DB': 'data/patterns_database.json',
    'PERFORMANCE_DB': 'data/performance_metrics.json',
    'BACKUP_ENABLED': True,
    'BACKUP_INTERVAL': 3600  # seconds
}

# ====================
# STRATEGY SETTINGS
# ====================

STRATEGIES = {
    'MOMENTUM': {
        'enabled': True,
        'weight': 0.3,
        'min_volume': 10000,
        'min_price_change': 0.005
    },
    'MEAN_REVERSION': {
        'enabled': True,
        'weight': 0.25,
        'bollinger_deviation': 2,
        'rsi_threshold': 30
    },
    'OPTIONS_FLOW': {
        'enabled': True,
        'weight': 0.25,
        'min_oi_change': 1000,
        'min_volume_spike': 2.0
    },
    'NEWS_BASED': {
        'enabled': True,
        'weight': 0.2,
        'sentiment_threshold': 0.7
    }
}

# ====================
# MULTI-LEG STRATEGIES
# ====================

MULTI_LEG_STRATEGIES = {
    'BULL_CALL_SPREAD': {
        'enabled': True,
        'max_spread_width': 200,
        'min_credit': 30
    },
    'BEAR_PUT_SPREAD': {
        'enabled': True,
        'max_spread_width': 200,
        'min_credit': 30
    },
    'IRON_CONDOR': {
        'enabled': False,  # Advanced strategy
        'wing_width': 300,
        'min_credit': 50
    },
    'STRADDLE': {
        'enabled': True,
        'delta_neutral': True,
        'max_days_to_expiry': 7
    }
}

# ====================
# NSE HOLIDAYS 2025
# ====================

NSE_HOLIDAYS_2025 = [
    datetime(2025, 1, 26),   # Republic Day
    datetime(2025, 3, 14),   # Holi
    datetime(2025, 3, 31),   # Ram Navami  
    datetime(2025, 4, 10),   # Mahavir Jayanti
    datetime(2025, 4, 18),   # Good Friday
    datetime(2025, 5, 1),    # Maharashtra Day
    datetime(2025, 8, 15),   # Independence Day
    datetime(2025, 8, 27),   # Ganesh Chaturthi
    datetime(2025, 10, 2),   # Gandhi Jayanti
    datetime(2025, 10, 20),  # Dussehra
    datetime(2025, 11, 1),   # Diwali
    datetime(2025, 11, 2),   # Diwali (Balipratipada)
    datetime(2025, 11, 17),  # Guru Nanak Jayanti
    datetime(2025, 12, 25),  # Christmas
]

# ====================
# MONITORING & ALERTS
# ====================

MONITORING = {
    'HEARTBEAT_INTERVAL': 60,        # seconds
    'METRICS_COLLECTION': True,
    'PERFORMANCE_TRACKING': True,
    'ERROR_THRESHOLD': 5,            # errors per hour before alert
    'LATENCY_THRESHOLD': 1000,       # milliseconds
    'MEMORY_THRESHOLD': 80,          # percent
    'CPU_THRESHOLD': 75              # percent
}

# ====================
# VALIDATION RULES
# ====================

VALIDATION_RULES = {
    'MIN_ORDER_VALUE': 100,
    'MAX_ORDER_VALUE': 10000000,
    'VALID_SYMBOLS': ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'],
    'VALID_OPTION_TYPES': ['CE', 'PE'],
    'MAX_STRIKE_DISTANCE': 1000,
    'MIN_DAYS_TO_EXPIRY': 0,
    'MAX_DAYS_TO_EXPIRY': 45
}

# ====================
# GLOBAL MARKET INDICATORS
# ====================

GLOBAL_MARKETS = {
    'SGX_NIFTY': {
        'symbol': 'SGXNIFTY',
        'weight': 0.7,
        'api_endpoint': ''  # To be configured
    },
    'DOW_FUTURES': {
        'symbol': 'YM=F',
        'weight': 0.15
    },
    'VIX': {
        'symbol': 'INDIAVIX',
        'weight': 0.15,
        'fear_threshold': 20,
        'extreme_fear': 30
    }
}

# ====================
# HELPER FUNCTIONS
# ====================

def get_lot_size(symbol: str) -> int:
    """Get lot size for a symbol"""
    return LOT_SIZES.get(symbol.upper(), 75)

def get_strike_gap(symbol: str) -> int:
    """Get strike gap for a symbol"""
    return STRIKE_GAPS.get(symbol.upper(), 50)

def is_market_open() -> bool:
    """Check if market is currently open"""
    from datetime import datetime
    now = datetime.now().time()
    market_open = datetime.strptime(MARKET_HOURS['MARKET_OPEN'], '%H:%M').time()
    market_close = datetime.strptime(MARKET_HOURS['MARKET_CLOSE'], '%H:%M').time()
    return market_open <= now <= market_close

def is_trading_holiday(date: datetime = None) -> bool:
    """Check if given date is a trading holiday"""
    if date is None:
        date = datetime.now()
    return date.date() in [holiday.date() for holiday in NSE_HOLIDAYS_2025]

def get_risk_per_trade(capital: float) -> float:
    """Calculate risk amount per trade based on capital"""
    return capital * RISK_MANAGEMENT['MAX_RISK_PER_TRADE']

def validate_order_params(symbol: str, quantity: int, price: float) -> tuple:
    """Validate order parameters"""
    errors = []
    
    if symbol.upper() not in VALIDATION_RULES['VALID_SYMBOLS']:
        errors.append(f"Invalid symbol: {symbol}")
    
    if quantity <= 0:
        errors.append(f"Invalid quantity: {quantity}")
    
    if price < VALIDATION_RULES['MIN_ORDER_VALUE'] or price > VALIDATION_RULES['MAX_ORDER_VALUE']:
        errors.append(f"Price {price} outside valid range")
    
    return len(errors) == 0, errors

# Export all constants for easy access
__all__ = [
    'LOT_SIZES', 'STRIKE_GAPS', 'SECURITY_IDS', 'EXCHANGE_SEGMENTS',
    'RISK_MANAGEMENT', 'API_RATE_LIMITS', 'MARKET_HOURS', 'INDICATOR_SETTINGS',
    'GREEKS_THRESHOLDS', 'PAPER_TRADING', 'LIVE_TRADING', 'NOTIFICATIONS',
    'DATABASE', 'STRATEGIES', 'MULTI_LEG_STRATEGIES', 'NSE_HOLIDAYS_2025',
    'MONITORING', 'VALIDATION_RULES', 'GLOBAL_MARKETS',
    'get_lot_size', 'get_strike_gap', 'is_market_open', 
    'is_trading_holiday', 'get_risk_per_trade', 'validate_order_params'
]