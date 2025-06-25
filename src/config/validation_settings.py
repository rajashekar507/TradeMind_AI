"""
Validation settings for institutional-grade options trading system
Configurable thresholds for price validation, volume/OI filters, and performance tracking
"""

class ValidationSettings:
    """Configurable validation settings for options trading"""
    
    MAX_PRICE_THRESHOLD_PCT = 10.0  # 10% max price threshold
    NIFTY_OTM_LIMIT = 300.0  # ₹300 NIFTY OTM limit
    BANKNIFTY_OTM_LIMIT = 500.0  # ₹500 BANKNIFTY OTM limit
    
    STRICT_GREEKS_REJECTION = True  # No calculated fallbacks
    MIN_IV_THRESHOLD = 5.0  # Minimum 5% IV
    MAX_IV_THRESHOLD = 150.0  # Maximum 150% IV for normal conditions
    MAX_IV_EARNINGS = 120.0  # Maximum 120% IV during earnings
    MAX_IV_CRASH = 130.0  # Maximum 130% IV during market crashes
    MAX_IV_WEEKLY_EXPIRY = 100.0  # Maximum 100% IV on weekly expiry day
    
    NIFTY_MIN_VOLUME = 100
    NIFTY_MIN_OI = 1000
    BANKNIFTY_MIN_VOLUME = 200
    BANKNIFTY_MIN_OI = 2000
    MAX_BID_ASK_SPREAD_PCT = 10.0  # Max 10% of LTP
    LAST_TRADE_TIMEOUT_MINUTES = 5  # Last trade within 5 minutes
    
    FREE_VALIDATION_ENABLED = True
    FREE_SOURCES_PRIORITY = ['yahoo_finance', 'moneycontrol_scraping', 'google_finance_scraping']
    FREE_PRICE_TOLERANCE = 2.0  # ₹2 maximum difference tolerance
    FREE_SOURCE_TIMEOUT = 10  # 10 seconds timeout for free sources
    NSE_HARD_STOP_THRESHOLD = 2.0  # >2% difference triggers hard stop
    
    HISTORICAL_ANALYSIS_ENABLED = True
    HISTORICAL_DATA_MONTHS = 3  # Analyze last 3 months for realistic modeling
    PERFORMANCE_METRICS_ENABLED = True
    TIME_BASED_ANALYSIS_ENABLED = True
    
    USE_REALISTIC_MODELING = True  # NO random simulation
    EVIDENCE_BASED_PROJECTIONS = True
    FACTOR_BID_ASK_SPREADS = True
    FACTOR_SLIPPAGE = True
    
    @classmethod
    def get_min_volume(cls, instrument: str) -> int:
        """Get minimum volume requirement for instrument"""
        return cls.NIFTY_MIN_VOLUME if instrument == 'NIFTY' else cls.BANKNIFTY_MIN_VOLUME
    
    @classmethod
    def get_min_oi(cls, instrument: str) -> int:
        """Get minimum OI requirement for instrument"""
        return cls.NIFTY_MIN_OI if instrument == 'NIFTY' else cls.BANKNIFTY_MIN_OI
    
    @classmethod
    def get_otm_limit(cls, instrument: str) -> float:
        """Get OTM price limit for instrument"""
        return cls.NIFTY_OTM_LIMIT if instrument == 'NIFTY' else cls.BANKNIFTY_OTM_LIMIT
    
    @classmethod
    def get_max_iv_for_context(cls, market_context: str = 'normal') -> float:
        """Get maximum IV threshold based on market context"""
        context_thresholds = {
            'earnings': cls.MAX_IV_EARNINGS,
            'crash': cls.MAX_IV_CRASH,
            'weekly_expiry': cls.MAX_IV_WEEKLY_EXPIRY,
            'normal': cls.MAX_IV_THRESHOLD
        }
        return context_thresholds.get(market_context, cls.MAX_IV_THRESHOLD)
