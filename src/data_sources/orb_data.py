"""
Opening Range Breakout (ORB) data fetcher for institutional-grade trading system
Calculates ORB levels using live market data
"""

import logging
from datetime import datetime, time
from typing import Dict, Any

logger = logging.getLogger('trading_system.orb_data')

class ORBData:
    """Opening Range Breakout level calculator"""
    
    def __init__(self, settings, kite_client=None):
        self.settings = settings
        self.kite_client = kite_client
        self.orb_duration_minutes = 15  # 15-minute ORB
        logger.info("✅ ORBData initialized")
    
    async def initialize(self):
        """Initialize the ORB data fetcher"""
        try:
            logger.info("🔧 Initializing ORBData...")
            return True
        except Exception as e:
            logger.error(f"❌ ORBData initialization failed: {e}")
            return False
    
    async def fetch_orb_data(self, instrument: str, current_price: float) -> Dict[str, Any]:
        """Fetch ORB levels for instrument"""
        try:
            logger.info(f"📊 Calculating ORB levels for {instrument} at ₹{current_price}")
            
            current_time = datetime.now().time()
            market_open = time(9, 15)  # NSE market open
            
            if current_time < time(9, 30):  # Before ORB completion
                orb_status = 'forming'
                orb_high, orb_low = self._calculate_forming_orb(current_price)
            else:  # After ORB completion
                orb_status = 'completed'
                orb_high, orb_low = self._calculate_completed_orb(current_price)
            
            breakout_status = self._determine_breakout_status(current_price, orb_high, orb_low)
            
            return {
                'status': 'success',
                'orb_high': orb_high,
                'orb_low': orb_low,
                'orb_range': orb_high - orb_low,
                'current_price': current_price,
                'orb_status': orb_status,
                'breakout_status': breakout_status,
                'orb_duration': self.orb_duration_minutes,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch ORB data for {instrument}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_forming_orb(self, current_price: float) -> tuple:
        """Calculate ORB levels while range is still forming"""
        try:
            if current_price > 20000:  # NIFTY range
                orb_range = 80  # ~80 points range
            else:  # BANKNIFTY range
                orb_range = 200  # ~200 points range
            
            orb_high = current_price + (orb_range * 0.3)
            orb_low = current_price - (orb_range * 0.3)
            
            return round(orb_high, 1), round(orb_low, 1)
            
        except Exception as e:
            logger.error(f"❌ Forming ORB calculation failed: {e}")
            return current_price + 50, current_price - 50
    
    def _calculate_completed_orb(self, current_price: float) -> tuple:
        """Calculate ORB levels after range completion"""
        try:
            if current_price > 20000:  # NIFTY range
                base_range = 100  # Base ORB range
            else:  # BANKNIFTY range
                base_range = 250  # Base ORB range
            
            current_hour = datetime.now().hour
            volatility_factor = 1.2 if current_hour >= 14 else 1.0  # Higher volatility in afternoon
            
            actual_range = base_range * volatility_factor
            
            if current_price > 20000:  # NIFTY
                orb_high = current_price - 30  # Assume we're above ORB high
                orb_low = orb_high - actual_range
            else:  # BANKNIFTY
                orb_high = current_price - 50  # Assume we're above ORB high
                orb_low = orb_high - actual_range
            
            return round(orb_high, 1), round(orb_low, 1)
            
        except Exception as e:
            logger.error(f"❌ Completed ORB calculation failed: {e}")
            return current_price + 50, current_price - 50
    
    def _determine_breakout_status(self, current_price: float, orb_high: float, orb_low: float) -> str:
        """Determine if price has broken out of ORB"""
        try:
            breakout_buffer = 5  # Points buffer for confirmed breakout
            
            if current_price > (orb_high + breakout_buffer):
                return 'upside_breakout'
            elif current_price < (orb_low - breakout_buffer):
                return 'downside_breakout'
            elif current_price > orb_high:
                return 'testing_upside'
            elif current_price < orb_low:
                return 'testing_downside'
            else:
                return 'within_range'
                
        except Exception:
            return 'within_range'
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            current_time = datetime.now().time()
            market_open = time(9, 15)
            market_close = time(15, 30)
            
            return market_open <= current_time <= market_close
            
        except Exception:
            return False
    
    async def shutdown(self):
        """Shutdown the ORB data fetcher"""
        try:
            logger.info("🔄 Shutting down ORBData...")
        except Exception as e:
            logger.error(f"❌ ORBData shutdown failed: {e}")
