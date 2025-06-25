"""
Opening Range Breakout (ORB) strategy implementation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional

logger = logging.getLogger('trading_system.orb_strategy')

class ORBStrategy:
    """Opening Range Breakout strategy for intraday trading"""
    
    def __init__(self, kite_client=None):
        self.kite = kite_client
        self.orb_duration = 15  # 15 minutes opening range
        self.market_open = time(9, 15)
        self.orb_end = time(9, 30)
        self.market_close = time(15, 30)
        
        self.orb_data = {}
        self.active_signals = {}
    
    async def analyze_orb(self, symbol: str) -> Dict:
        """Analyze Opening Range Breakout for given symbol"""
        orb_analysis = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'status': 'failed',
            'orb_high': 0,
            'orb_low': 0,
            'current_price': 0,
            'signal': 'neutral',
            'breakout_type': None,
            'target': 0,
            'stop_loss': 0,
            'confidence': 0
        }
        
        try:
            if not self.kite:
                logger.error("❌ STRICT ENFORCEMENT: No Kite client available - CANNOT ANALYZE ORB")
                orb_analysis['error'] = 'No Kite client - strict enforcement mode'
                return orb_analysis
            
            current_time = datetime.now().time()
            
            if not self._is_market_hours(current_time):
                orb_analysis['error'] = 'Market closed'
                return orb_analysis
            
            if current_time < self.orb_end:
                orb_analysis = await self._build_opening_range(symbol)
            else:
                orb_analysis = await self._check_breakout(symbol)
            
            logger.info(f"✅ ORB analysis completed for {symbol}: {orb_analysis['signal']}")
            
        except Exception as e:
            logger.error(f"❌ STRICT ENFORCEMENT: ORB analysis failed for {symbol}: {e}")
            orb_analysis['error'] = str(e)
        
        return orb_analysis
    
    async def _build_opening_range(self, symbol: str) -> Dict:
        """Build opening range during first 15 minutes"""
        try:
            df = await self._fetch_intraday_data(symbol, '5minute')
            if df is None or len(df) == 0:
                return {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'No intraday data available'
                }
            
            today = datetime.now().date()
            today_data = df[df.index.date == today]
            
            if len(today_data) == 0:
                return {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'No today data available'
                }
            
            opening_range_data = today_data[
                (today_data.index.time >= self.market_open) & 
                (today_data.index.time <= self.orb_end)
            ]
            
            if len(opening_range_data) == 0:
                return {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'status': 'building',
                    'orb_high': 0,
                    'orb_low': 0,
                    'current_price': today_data['close'].iloc[-1] if len(today_data) > 0 else 0
                }
            
            orb_high = opening_range_data['high'].max()
            orb_low = opening_range_data['low'].min()
            current_price = today_data['close'].iloc[-1]
            
            self.orb_data[symbol] = {
                'orb_high': orb_high,
                'orb_low': orb_low,
                'orb_range': orb_high - orb_low,
                'date': today
            }
            
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'status': 'building',
                'orb_high': orb_high,
                'orb_low': orb_low,
                'current_price': current_price,
                'orb_range': orb_high - orb_low,
                'signal': 'building_range'
            }
            
        except Exception as e:
            logger.error(f"❌ Opening range building failed: {e}")
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _check_breakout(self, symbol: str) -> Dict:
        """Check for ORB breakout after opening range"""
        try:
            if symbol not in self.orb_data:
                return await self._build_opening_range(symbol)
            
            orb_info = self.orb_data[symbol]
            
            if orb_info['date'] != datetime.now().date():
                return await self._build_opening_range(symbol)
            
            df = await self._fetch_intraday_data(symbol, '5minute')
            if df is None or len(df) == 0:
                return {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'No current price data'
                }
            
            current_price = df['close'].iloc[-1]
            orb_high = orb_info['orb_high']
            orb_low = orb_info['orb_low']
            orb_range = orb_info['orb_range']
            
            signal = 'neutral'
            breakout_type = None
            target = 0
            stop_loss = 0
            confidence = 0
            
            if current_price > orb_high:
                signal = 'bullish'
                breakout_type = 'upside_breakout'
                target = orb_high + orb_range
                stop_loss = orb_low
                confidence = self._calculate_breakout_confidence(df, orb_high, 'bullish')
                
            elif current_price < orb_low:
                signal = 'bearish'
                breakout_type = 'downside_breakout'
                target = orb_low - orb_range
                stop_loss = orb_high
                confidence = self._calculate_breakout_confidence(df, orb_low, 'bearish')
            
            volume_confirmation = self._check_volume_confirmation(df)
            if volume_confirmation:
                confidence *= 1.2
            
            confidence = min(confidence, 100)
            
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'status': 'success',
                'orb_high': orb_high,
                'orb_low': orb_low,
                'current_price': current_price,
                'signal': signal,
                'breakout_type': breakout_type,
                'target': target,
                'stop_loss': stop_loss,
                'confidence': round(confidence, 1),
                'volume_confirmation': volume_confirmation,
                'orb_range': orb_range
            }
            
        except Exception as e:
            logger.error(f"❌ Breakout check failed: {e}")
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _fetch_intraday_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch intraday data for ORB analysis"""
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
            
            from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
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
            logger.error(f"❌ Intraday data fetch failed: {e}")
            return None
    
    def _calculate_breakout_confidence(self, df: pd.DataFrame, breakout_level: float, direction: str) -> float:
        """Calculate confidence in breakout signal"""
        try:
            confidence = 50
            
            current_price = df['close'].iloc[-1]
            
            if direction == 'bullish':
                breakout_strength = ((current_price - breakout_level) / breakout_level) * 100
            else:
                breakout_strength = ((breakout_level - current_price) / breakout_level) * 100
            
            confidence += min(breakout_strength * 10, 30)
            
            recent_data = df.tail(6)
            if len(recent_data) >= 3:
                if direction == 'bullish':
                    consecutive_higher = all(
                        recent_data['close'].iloc[i] >= recent_data['close'].iloc[i-1] 
                        for i in range(1, min(3, len(recent_data)))
                    )
                else:
                    consecutive_higher = all(
                        recent_data['close'].iloc[i] <= recent_data['close'].iloc[i-1] 
                        for i in range(1, min(3, len(recent_data)))
                    )
                
                if consecutive_higher:
                    confidence += 15
            
            return confidence
            
        except Exception:
            return 50
    
    def _check_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if breakout is confirmed by volume"""
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return False
            
            recent_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(10).mean()
            
            return recent_volume > avg_volume * 1.5
            
        except Exception:
            return False
    
    def _is_market_hours(self, current_time: time) -> bool:
        """Check if current time is within market hours"""
        return self.market_open <= current_time <= self.market_close
    
    async def get_session_analysis(self, symbol: str) -> Dict:
        """Get session-based market analysis"""
        try:
            current_time = datetime.now().time()
            
            if current_time < time(9, 15):
                session = 'pre_market'
            elif current_time < time(10, 0):
                session = 'opening'
            elif current_time < time(14, 30):
                session = 'mid_session'
            elif current_time < time(15, 30):
                session = 'closing'
            else:
                session = 'post_market'
            
            df = await self._fetch_intraday_data(symbol, '5minute')
            if df is None:
                return {
                    'session': session,
                    'status': 'failed',
                    'error': 'No data available'
                }
            
            session_analysis = {
                'session': session,
                'status': 'success',
                'current_price': df['close'].iloc[-1],
                'session_high': df['high'].max(),
                'session_low': df['low'].min(),
                'session_volume': df['volume'].sum() if 'volume' in df.columns else 0,
                'price_action': self._analyze_price_action(df, session)
            }
            
            return session_analysis
            
        except Exception as e:
            logger.error(f"❌ Session analysis failed: {e}")
            return {
                'session': 'unknown',
                'status': 'failed',
                'error': str(e)
            }
    
    def _analyze_price_action(self, df: pd.DataFrame, session: str) -> str:
        """Analyze price action for current session"""
        try:
            if len(df) < 3:
                return 'insufficient_data'
            
            recent_data = df.tail(6)
            price_change = ((recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / 
                          recent_data['close'].iloc[0]) * 100
            
            if session == 'opening':
                if abs(price_change) > 0.5:
                    return 'volatile_opening'
                else:
                    return 'stable_opening'
            
            elif session == 'mid_session':
                if price_change > 0.3:
                    return 'trending_up'
                elif price_change < -0.3:
                    return 'trending_down'
                else:
                    return 'sideways'
            
            elif session == 'closing':
                if abs(price_change) > 0.2:
                    return 'active_closing'
                else:
                    return 'quiet_closing'
            
            return 'normal'
            
        except Exception:
            return 'unknown'
