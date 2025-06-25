"""
Candlestick pattern recognition for institutional trading
"""

import logging
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger('trading_system.pattern_detection')

class PatternDetector:
    """Advanced candlestick pattern recognition"""
    
    def __init__(self, kite_client=None):
        self.kite = kite_client
        self.patterns = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing_bullish': self._detect_bullish_engulfing,
            'engulfing_bearish': self._detect_bearish_engulfing,
            'morning_star': self._detect_morning_star,
            'evening_star': self._detect_evening_star,
            'piercing_line': self._detect_piercing_line,
            'dark_cloud': self._detect_dark_cloud
        }
    
    async def detect_patterns(self, symbol: str, timeframe: str = 'day') -> Dict:
        """Detect candlestick patterns for given symbol"""
        pattern_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'failed',
            'patterns': {},
            'signal': 'neutral',
            'confidence': 0
        }
        
        try:
            if not self.kite:
                logger.error("❌ STRICT ENFORCEMENT: No Kite client available - CANNOT DETECT PATTERNS")
                pattern_data['error'] = 'No Kite client - strict enforcement mode'
                return pattern_data
            
            df = await self._fetch_historical_data(symbol, timeframe)
            if df is None or len(df) < 10:
                logger.error(f"❌ STRICT ENFORCEMENT: Insufficient data for pattern detection - {symbol}")
                pattern_data['error'] = 'Insufficient historical data'
                return pattern_data
            
            detected_patterns = {}
            for pattern_name, detector_func in self.patterns.items():
                try:
                    result = detector_func(df)
                    if result:
                        detected_patterns[pattern_name] = result
                except Exception as e:
                    logger.warning(f"⚠️ Pattern detection failed for {pattern_name}: {e}")
            
            pattern_data['patterns'] = detected_patterns
            pattern_data['signal'], pattern_data['confidence'] = self._calculate_pattern_signal(detected_patterns)
            pattern_data['status'] = 'success'
            
            logger.info(f"✅ Pattern detection completed for {symbol}: {len(detected_patterns)} patterns found")
            
        except Exception as e:
            logger.error(f"❌ STRICT ENFORCEMENT: Pattern detection failed for {symbol}: {e}")
            pattern_data['error'] = str(e)
        
        return pattern_data
    
    async def _fetch_historical_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for pattern analysis"""
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
            
            from_date = datetime.now() - timedelta(days=30)
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
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Historical data fetch failed: {e}")
            return None
    
    def _detect_doji(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Doji patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            body_ratio = body_size / total_range
            
            if body_ratio <= 0.1:
                return {
                    'type': 'doji',
                    'signal': 'reversal',
                    'strength': 1 - body_ratio,
                    'description': 'Market indecision, potential reversal'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_hammer(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Hammer patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            if (lower_shadow >= 2 * body_size and 
                upper_shadow <= body_size * 0.5 and
                body_size / total_range >= 0.1):
                
                return {
                    'type': 'hammer',
                    'signal': 'bullish',
                    'strength': lower_shadow / total_range,
                    'description': 'Bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Shooting Star patterns"""
        try:
            latest = df.iloc[-1]
            open_price = latest['open']
            close_price = latest['close']
            high_price = latest['high']
            low_price = latest['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return None
            
            if (upper_shadow >= 2 * body_size and 
                lower_shadow <= body_size * 0.5 and
                body_size / total_range >= 0.1):
                
                return {
                    'type': 'shooting_star',
                    'signal': 'bearish',
                    'strength': upper_shadow / total_range,
                    'description': 'Bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_bullish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Bullish Engulfing patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bearish = prev_candle['close'] < prev_candle['open']
            curr_bullish = curr_candle['close'] > curr_candle['open']
            
            if (prev_bearish and curr_bullish and
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > prev_candle['open']):
                
                engulfing_ratio = (curr_candle['close'] - curr_candle['open']) / (prev_candle['open'] - prev_candle['close'])
                
                return {
                    'type': 'bullish_engulfing',
                    'signal': 'bullish',
                    'strength': min(engulfing_ratio, 2.0) / 2.0,
                    'description': 'Strong bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_bearish_engulfing(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Bearish Engulfing patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bullish = prev_candle['close'] > prev_candle['open']
            curr_bearish = curr_candle['close'] < curr_candle['open']
            
            if (prev_bullish and curr_bearish and
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < prev_candle['open']):
                
                engulfing_ratio = (curr_candle['open'] - curr_candle['close']) / (prev_candle['close'] - prev_candle['open'])
                
                return {
                    'type': 'bearish_engulfing',
                    'signal': 'bearish',
                    'strength': min(engulfing_ratio, 2.0) / 2.0,
                    'description': 'Strong bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_morning_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Morning Star patterns"""
        try:
            if len(df) < 3:
                return None
            
            first = df.iloc[-3]
            second = df.iloc[-2]
            third = df.iloc[-1]
            
            first_bearish = first['close'] < first['open']
            second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
            third_bullish = third['close'] > third['open']
            
            gap_down = second['high'] < first['close']
            gap_up = third['open'] > second['high']
            
            if (first_bearish and second_small and third_bullish and gap_down and gap_up):
                return {
                    'type': 'morning_star',
                    'signal': 'bullish',
                    'strength': 0.8,
                    'description': 'Strong bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_evening_star(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Evening Star patterns"""
        try:
            if len(df) < 3:
                return None
            
            first = df.iloc[-3]
            second = df.iloc[-2]
            third = df.iloc[-1]
            
            first_bullish = first['close'] > first['open']
            second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
            third_bearish = third['close'] < third['open']
            
            gap_up = second['low'] > first['close']
            gap_down = third['open'] < second['low']
            
            if (first_bullish and second_small and third_bearish and gap_up and gap_down):
                return {
                    'type': 'evening_star',
                    'signal': 'bearish',
                    'strength': 0.8,
                    'description': 'Strong bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_piercing_line(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Piercing Line patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bearish = prev_candle['close'] < prev_candle['open']
            curr_bullish = curr_candle['close'] > curr_candle['open']
            
            midpoint = (prev_candle['open'] + prev_candle['close']) / 2
            
            if (prev_bearish and curr_bullish and
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > midpoint):
                
                penetration = (curr_candle['close'] - midpoint) / (prev_candle['open'] - prev_candle['close'])
                
                return {
                    'type': 'piercing_line',
                    'signal': 'bullish',
                    'strength': min(penetration, 1.0),
                    'description': 'Bullish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_dark_cloud(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect Dark Cloud Cover patterns"""
        try:
            if len(df) < 2:
                return None
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            prev_bullish = prev_candle['close'] > prev_candle['open']
            curr_bearish = curr_candle['close'] < curr_candle['open']
            
            midpoint = (prev_candle['open'] + prev_candle['close']) / 2
            
            if (prev_bullish and curr_bearish and
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < midpoint):
                
                penetration = (midpoint - curr_candle['close']) / (prev_candle['close'] - prev_candle['open'])
                
                return {
                    'type': 'dark_cloud',
                    'signal': 'bearish',
                    'strength': min(penetration, 1.0),
                    'description': 'Bearish reversal pattern'
                }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_signal(self, patterns: Dict) -> tuple:
        """Calculate overall pattern signal and confidence"""
        if not patterns:
            return 'neutral', 0
        
        bullish_strength = 0
        bearish_strength = 0
        total_patterns = 0
        
        for pattern_data in patterns.values():
            strength = pattern_data.get('strength', 0)
            signal = pattern_data.get('signal', 'neutral')
            
            if signal == 'bullish':
                bullish_strength += strength
            elif signal == 'bearish':
                bearish_strength += strength
            
            total_patterns += 1
        
        if total_patterns == 0:
            return 'neutral', 0
        
        if bullish_strength > bearish_strength:
            confidence = (bullish_strength / total_patterns) * 100
            return 'bullish', min(confidence, 100)
        elif bearish_strength > bullish_strength:
            confidence = (bearish_strength / total_patterns) * 100
            return 'bearish', min(confidence, 100)
        else:
            return 'neutral', 0
