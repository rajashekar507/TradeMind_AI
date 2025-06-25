"""
Dynamic support and resistance level calculation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('trading_system.support_resistance')

class SupportResistanceCalculator:
    """Calculate dynamic support and resistance levels"""
    
    def __init__(self, kite_client=None):
        self.kite = kite_client
        self.lookback_periods = {
            '5minute': 288,  # 1 day
            '15minute': 96,  # 1 day
            '60minute': 168, # 1 week
            'day': 50        # 50 days
        }
    
    async def calculate_levels(self, symbol: str, timeframe: str = 'day') -> Dict:
        """Calculate support and resistance levels"""
        sr_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'failed',
            'support_levels': [],
            'resistance_levels': [],
            'current_level': 'neutral',
            'strength': 0
        }
        
        try:
            if not self.kite:
                logger.error("❌ STRICT ENFORCEMENT: No Kite client available - CANNOT CALCULATE S/R")
                sr_data['error'] = 'No Kite client - strict enforcement mode'
                return sr_data
            
            df = await self._fetch_historical_data(symbol, timeframe)
            if df is None or len(df) < 20:
                logger.error(f"❌ STRICT ENFORCEMENT: Insufficient data for S/R calculation - {symbol}")
                sr_data['error'] = 'Insufficient historical data'
                return sr_data
            
            current_price = df['close'].iloc[-1]
            
            pivot_levels = self._calculate_pivot_points(df)
            volume_levels = self._calculate_volume_profile_levels(df)
            fibonacci_levels = self._calculate_fibonacci_levels(df)
            
            all_support = []
            all_resistance = []
            
            all_support.extend(pivot_levels['support'])
            all_support.extend(volume_levels['support'])
            all_support.extend(fibonacci_levels['support'])
            
            all_resistance.extend(pivot_levels['resistance'])
            all_resistance.extend(volume_levels['resistance'])
            all_resistance.extend(fibonacci_levels['resistance'])
            
            support_levels = self._filter_and_rank_levels(all_support, current_price, 'support')
            resistance_levels = self._filter_and_rank_levels(all_resistance, current_price, 'resistance')
            
            current_level, strength = self._determine_current_position(
                current_price, support_levels, resistance_levels
            )
            
            sr_data.update({
                'status': 'success',
                'support_levels': support_levels[:5],
                'resistance_levels': resistance_levels[:5],
                'current_level': current_level,
                'strength': strength
            })
            
            logger.info(f"✅ S/R levels calculated for {symbol}: {len(support_levels)} support, {len(resistance_levels)} resistance")
            
        except Exception as e:
            logger.error(f"❌ STRICT ENFORCEMENT: S/R calculation failed for {symbol}: {e}")
            sr_data['error'] = str(e)
        
        return sr_data
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for S/R analysis"""
        try:
            instruments = self.kite.instruments("NSE")
            instrument_token = None
            
            for instrument in instruments:
                if instrument['name'] == f"{symbol} 50" if symbol == 'NIFTY' else f"{symbol} BANK":
                    if instrument['segment'] == 'INDICES':
                        instrument_token = instrument['instrument_token']
                        break
            
            if not instrument_token:
                logger.error(f"❌ Instrument token not found for {symbol}")
                return None
            
            lookback = self.lookback_periods.get(timeframe, 50)
            from_date = datetime.now() - timedelta(days=lookback * 2)
            to_date = datetime.now()
            
            historical_data = self.kite.historical_data(
                instrument_token,
                from_date,
                to_date,
                timeframe
            )
            
            if not historical_data:
                return None
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df.tail(lookback)
            
        except Exception as e:
            logger.error(f"❌ Historical data fetch failed: {e}")
            return None
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot point based support and resistance"""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            support_levels = []
            resistance_levels = []
            
            for i in range(2, len(df) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                    lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                    support_levels.append({
                        'level': lows[i],
                        'type': 'pivot_low',
                        'strength': self._calculate_level_strength(df, lows[i], 'support'),
                        'touches': self._count_touches(df, lows[i], tolerance=0.5)
                    })
                
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                    resistance_levels.append({
                        'level': highs[i],
                        'type': 'pivot_high',
                        'strength': self._calculate_level_strength(df, highs[i], 'resistance'),
                        'touches': self._count_touches(df, highs[i], tolerance=0.5)
                    })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Pivot point calculation failed: {e}")
            return {'support': [], 'resistance': []}
    
    def _calculate_volume_profile_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate volume profile based levels"""
        try:
            if 'volume' not in df.columns:
                return {'support': [], 'resistance': []}
            
            price_range = df['high'].max() - df['low'].min()
            num_bins = min(50, len(df) // 2)
            
            if num_bins < 5:
                return {'support': [], 'resistance': []}
            
            bins = np.linspace(df['low'].min(), df['high'].max(), num_bins)
            volume_profile = np.zeros(len(bins) - 1)
            
            for i, row in df.iterrows():
                typical_price = (row['high'] + row['low'] + row['close']) / 3
                bin_idx = np.digitize(typical_price, bins) - 1
                if 0 <= bin_idx < len(volume_profile):
                    volume_profile[bin_idx] += row['volume']
            
            high_volume_indices = np.argsort(volume_profile)[-5:]
            
            support_levels = []
            resistance_levels = []
            current_price = df['close'].iloc[-1]
            
            for idx in high_volume_indices:
                level_price = (bins[idx] + bins[idx + 1]) / 2
                
                if level_price < current_price:
                    support_levels.append({
                        'level': level_price,
                        'type': 'volume_support',
                        'strength': volume_profile[idx] / volume_profile.max(),
                        'touches': self._count_touches(df, level_price, tolerance=1.0)
                    })
                else:
                    resistance_levels.append({
                        'level': level_price,
                        'type': 'volume_resistance',
                        'strength': volume_profile[idx] / volume_profile.max(),
                        'touches': self._count_touches(df, level_price, tolerance=1.0)
                    })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Volume profile calculation failed: {e}")
            return {'support': [], 'resistance': []}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            recent_data = df.tail(20)
            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            
            price_range = high_price - low_price
            current_price = df['close'].iloc[-1]
            
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            support_levels = []
            resistance_levels = []
            
            for fib in fib_levels:
                level = high_price - (price_range * fib)
                
                if level < current_price:
                    support_levels.append({
                        'level': level,
                        'type': f'fibonacci_{fib}',
                        'strength': 0.7,
                        'touches': self._count_touches(df, level, tolerance=1.0)
                    })
                else:
                    resistance_levels.append({
                        'level': level,
                        'type': f'fibonacci_{fib}',
                        'strength': 0.7,
                        'touches': self._count_touches(df, level, tolerance=1.0)
                    })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Fibonacci calculation failed: {e}")
            return {'support': [], 'resistance': []}
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float, level_type: str) -> float:
        """Calculate strength of a support/resistance level"""
        try:
            touches = self._count_touches(df, level, tolerance=0.5)
            age_factor = 1.0
            
            recent_data = df.tail(10)
            if level_type == 'support':
                recent_bounces = sum(1 for low in recent_data['low'] if abs(low - level) <= level * 0.01)
            else:
                recent_bounces = sum(1 for high in recent_data['high'] if abs(high - level) <= level * 0.01)
            
            strength = min((touches * 0.3 + recent_bounces * 0.4 + age_factor * 0.3), 1.0)
            return strength
            
        except Exception:
            return 0.5
    
    def _count_touches(self, df: pd.DataFrame, level: float, tolerance: float = 0.5) -> int:
        """Count how many times price touched a level"""
        try:
            tolerance_pct = tolerance / 100
            upper_bound = level * (1 + tolerance_pct)
            lower_bound = level * (1 - tolerance_pct)
            
            touches = 0
            for _, row in df.iterrows():
                if (lower_bound <= row['low'] <= upper_bound or
                    lower_bound <= row['high'] <= upper_bound or
                    (row['low'] <= level <= row['high'])):
                    touches += 1
            
            return touches
            
        except Exception:
            return 0
    
    def _filter_and_rank_levels(self, levels: List[Dict], current_price: float, level_type: str) -> List[Dict]:
        """Filter and rank support/resistance levels"""
        try:
            if not levels:
                return []
            
            filtered_levels = []
            
            for level_data in levels:
                level = level_data['level']
                
                if level_type == 'support' and level < current_price:
                    distance_pct = ((current_price - level) / current_price) * 100
                    if distance_pct <= 10:
                        level_data['distance'] = distance_pct
                        filtered_levels.append(level_data)
                
                elif level_type == 'resistance' and level > current_price:
                    distance_pct = ((level - current_price) / current_price) * 100
                    if distance_pct <= 10:
                        level_data['distance'] = distance_pct
                        filtered_levels.append(level_data)
            
            filtered_levels.sort(key=lambda x: (x['strength'] * x['touches']), reverse=True)
            
            return filtered_levels
            
        except Exception:
            return []
    
    def _determine_current_position(self, current_price: float, support_levels: List[Dict], 
                                  resistance_levels: List[Dict]) -> Tuple[str, float]:
        """Determine current position relative to S/R levels"""
        try:
            if not support_levels and not resistance_levels:
                return 'neutral', 0
            
            nearest_support = None
            nearest_resistance = None
            
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: x['distance'])
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: x['distance'])
            
            if nearest_support and nearest_resistance:
                support_distance = nearest_support['distance']
                resistance_distance = nearest_resistance['distance']
                
                if support_distance < resistance_distance:
                    return 'near_support', nearest_support['strength']
                else:
                    return 'near_resistance', nearest_resistance['strength']
            
            elif nearest_support:
                return 'near_support', nearest_support['strength']
            
            elif nearest_resistance:
                return 'near_resistance', nearest_resistance['strength']
            
            return 'neutral', 0
            
        except Exception:
            return 'neutral', 0
