"""
Support and Resistance data fetcher for institutional-grade trading system
Calculates dynamic S/R levels using live price data
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger('trading_system.support_resistance_data')

class SupportResistanceData:
    """Support and Resistance level calculator"""
    
    def __init__(self, settings, kite_client=None):
        self.settings = settings
        self.kite_client = kite_client
        logger.info("✅ SupportResistanceData initialized")
    
    async def initialize(self):
        """Initialize the S/R data fetcher"""
        try:
            logger.info("🔧 Initializing SupportResistanceData...")
            return True
        except Exception as e:
            logger.error(f"❌ SupportResistanceData initialization failed: {e}")
            return False
    
    async def fetch_sr_data(self, instrument: str, current_price: float) -> Dict[str, Any]:
        """Fetch support and resistance levels for instrument"""
        try:
            logger.info(f"📊 Calculating S/R levels for {instrument} at ₹{current_price}")
            
            support_levels = self._calculate_support_levels(current_price)
            resistance_levels = self._calculate_resistance_levels(current_price)
            
            current_level = self._determine_current_level(current_price, support_levels, resistance_levels)
            
            return {
                'status': 'success',
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_level': current_level,
                'current_price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch S/R data for {instrument}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_support_levels(self, current_price: float) -> List[Dict[str, Any]]:
        """Calculate support levels below current price"""
        try:
            support_levels = []
            
            support_percentages = [0.5, 1.0, 1.5, 2.0, 3.0]  # % below current price
            
            for pct in support_percentages:
                level = current_price * (1 - pct / 100)
                distance = abs(current_price - level) / current_price * 100
                
                support_levels.append({
                    'level': round(level, 1),
                    'distance': round(distance, 2),
                    'strength': self._calculate_level_strength(level, current_price),
                    'type': 'percentage_support'
                })
            
            psychological_supports = self._get_psychological_levels(current_price, 'support')
            support_levels.extend(psychological_supports)
            
            support_levels.sort(key=lambda x: x['distance'])
            
            return support_levels[:5]  # Return top 5 closest levels
            
        except Exception as e:
            logger.error(f"❌ Support level calculation failed: {e}")
            return []
    
    def _calculate_resistance_levels(self, current_price: float) -> List[Dict[str, Any]]:
        """Calculate resistance levels above current price"""
        try:
            resistance_levels = []
            
            resistance_percentages = [0.5, 1.0, 1.5, 2.0, 3.0]  # % above current price
            
            for pct in resistance_percentages:
                level = current_price * (1 + pct / 100)
                distance = abs(level - current_price) / current_price * 100
                
                resistance_levels.append({
                    'level': round(level, 1),
                    'distance': round(distance, 2),
                    'strength': self._calculate_level_strength(level, current_price),
                    'type': 'percentage_resistance'
                })
            
            psychological_resistances = self._get_psychological_levels(current_price, 'resistance')
            resistance_levels.extend(psychological_resistances)
            
            resistance_levels.sort(key=lambda x: x['distance'])
            
            return resistance_levels[:5]  # Return top 5 closest levels
            
        except Exception as e:
            logger.error(f"❌ Resistance level calculation failed: {e}")
            return []
    
    def _get_psychological_levels(self, current_price: float, level_type: str) -> List[Dict[str, Any]]:
        """Get psychological support/resistance levels (round numbers)"""
        try:
            levels = []
            
            if current_price > 20000:  # NIFTY range
                intervals = [50, 100, 250]
            else:  # Other instruments
                intervals = [100, 500, 1000]
            
            for interval in intervals:
                if level_type == 'support':
                    level = (int(current_price / interval)) * interval
                    if level < current_price:
                        distance = abs(current_price - level) / current_price * 100
                        if distance <= 3.0:  # Within 3%
                            levels.append({
                                'level': float(level),
                                'distance': round(distance, 2),
                                'strength': 0.8,  # High strength for psychological levels
                                'type': 'psychological_support'
                            })
                else:  # resistance
                    level = (int(current_price / interval) + 1) * interval
                    if level > current_price:
                        distance = abs(level - current_price) / current_price * 100
                        if distance <= 3.0:  # Within 3%
                            levels.append({
                                'level': float(level),
                                'distance': round(distance, 2),
                                'strength': 0.8,  # High strength for psychological levels
                                'type': 'psychological_resistance'
                            })
            
            return levels
            
        except Exception as e:
            logger.error(f"❌ Psychological level calculation failed: {e}")
            return []
    
    def _calculate_level_strength(self, level: float, current_price: float) -> float:
        """Calculate strength of S/R level (0.0 to 1.0)"""
        try:
            distance_pct = abs(level - current_price) / current_price * 100
            distance_strength = max(0.2, 1.0 - (distance_pct / 5.0))  # Stronger if within 5%
            
            round_bonus = 0.0
            if level % 100 == 0:  # Round hundreds
                round_bonus = 0.3
            elif level % 50 == 0:  # Round fifties
                round_bonus = 0.2
            
            strength = min(1.0, distance_strength + round_bonus)
            return round(strength, 2)
            
        except Exception:
            return 0.5  # Default strength
    
    def _determine_current_level(self, current_price: float, support_levels: List[Dict], resistance_levels: List[Dict]) -> str:
        """Determine current position relative to S/R levels"""
        try:
            for support in support_levels:
                if support['distance'] <= 0.5:
                    return 'near_support'
            
            for resistance in resistance_levels:
                if resistance['distance'] <= 0.5:
                    return 'near_resistance'
            
            closest_support = min(support_levels, key=lambda x: x['distance']) if support_levels else None
            closest_resistance = min(resistance_levels, key=lambda x: x['distance']) if resistance_levels else None
            
            if closest_support and closest_resistance:
                if closest_support['distance'] < closest_resistance['distance']:
                    return 'above_support'
                else:
                    return 'below_resistance'
            
            return 'neutral'
            
        except Exception:
            return 'neutral'
    
    async def shutdown(self):
        """Shutdown the S/R data fetcher"""
        try:
            logger.info("🔄 Shutting down SupportResistanceData...")
        except Exception as e:
            logger.error(f"❌ SupportResistanceData shutdown failed: {e}")
