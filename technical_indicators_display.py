"""
TradeMind AI Technical Indicators Display System - Complete Production Version
This is the FINAL version - no future changes needed
Real-time technical analysis with dashboard integration and visual indicators
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import math

# Import required modules
try:
    from realtime_market_data import RealTimeMarketData
    from historical_data import HistoricalDataFetcher
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please ensure realtime_market_data.py and historical_data.py are available")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('technical_indicators.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength enumeration"""
    VERY_STRONG = "VERY_STRONG"    # 90-100
    STRONG = "STRONG"              # 75-89
    MODERATE = "MODERATE"          # 60-74
    WEAK = "WEAK"                  # 40-59
    VERY_WEAK = "VERY_WEAK"        # 0-39

class TrendDirection(Enum):
    """Trend direction enumeration"""
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class SignalType(Enum):
    """Signal type enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TechnicalIndicator:
    """Individual technical indicator data"""
    name: str
    value: float
    signal: SignalType
    strength: SignalStrength
    description: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': round(self.value, 4),
            'signal': self.signal.value,
            'strength': self.strength.value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class TechnicalAnalysis:
    """Complete technical analysis result"""
    symbol: str
    timeframe: str
    current_price: float
    
    # Individual indicators
    rsi: TechnicalIndicator
    macd: Dict[str, Any]
    bollinger_bands: Dict[str, Any]
    moving_averages: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    momentum_indicators: Dict[str, Any]
    
    # Support and resistance
    support_levels: List[float]
    resistance_levels: List[float]
    
    # Overall analysis
    overall_signal: SignalType
    overall_strength: SignalStrength
    confluence_score: float
    trend_direction: TrendDirection
    
    # Chart data for frontend
    chart_data: Dict[str, Any]
    
    # Metadata
    timestamp: datetime
    data_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': round(self.current_price, 2),
            'indicators': {
                'rsi': self.rsi.to_dict(),
                'macd': self.macd,
                'bollinger_bands': self.bollinger_bands,
                'moving_averages': self.moving_averages,
                'volume_analysis': self.volume_analysis,
                'momentum_indicators': self.momentum_indicators
            },
            'levels': {
                'support': [round(level, 2) for level in self.support_levels],
                'resistance': [round(level, 2) for level in self.resistance_levels]
            },
            'overall': {
                'signal': self.overall_signal.value,
                'strength': self.overall_strength.value,
                'confluence_score': round(self.confluence_score, 2),
                'trend_direction': self.trend_direction.value
            },
            'chart_data': self.chart_data,
            'metadata': {
                'timestamp': self.timestamp.isoformat(),
                'data_quality': round(self.data_quality, 2)
            }
        }

class TechnicalIndicatorsDisplay:
    """Complete Technical Indicators Display System"""
    
    def __init__(self):
        """Initialize technical indicators system"""
        logger.info("📊 Initializing Technical Indicators Display System...")
        
        # Initialize data sources
        self.market_data = RealTimeMarketData()
        self.historical_fetcher = HistoricalDataFetcher()
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # Supported timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.default_timeframe = '5m'
        
        # Indicator parameters
        self.indicator_params = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2},
            'ema': {'periods': [9, 21, 55]},
            'sma': {'periods': [20, 50, 200]},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'williams_r': {'period': 14},
            'volume_sma': {'period': 20}
        }
        
        # Signal weights for confluence
        self.signal_weights = {
            'rsi': 0.20,
            'macd': 0.25,
            'bollinger': 0.15,
            'ema_cross': 0.20,
            'volume': 0.10,
            'momentum': 0.10
        }
        
        # Background update control
        self.stop_updates = False
        self.update_threads = {}
        
        # Performance tracking
        self.performance = {
            'total_analyses': 0,
            'successful_signals': 0,
            'cache_hits': 0,
            'last_update': datetime.now()
        }
        
        logger.info("✅ Technical Indicators Display System initialized!")
    
    def get_technical_analysis(self, symbol: str, timeframe: str = None, 
                             force_refresh: bool = False) -> TechnicalAnalysis:
        """Get complete technical analysis for symbol and timeframe"""
        try:
            timeframe = timeframe or self.default_timeframe
            cache_key = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Check cache first
            if not force_refresh and cache_key in self.analysis_cache:
                cached_time = self.analysis_cache[cache_key]['timestamp']
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    self.performance['cache_hits'] += 1
                    logger.info(f"📋 Returning cached analysis for {symbol} {timeframe}")
                    return self.analysis_cache[cache_key]['analysis']
            
            logger.info(f"🔍 Generating technical analysis for {symbol} {timeframe}...")
            
            # Get market data
            if symbol == 'NIFTY':
                current_data = self.market_data.get_nifty_data()
            elif symbol == 'BANKNIFTY':
                current_data = self.market_data.get_banknifty_data()
            else:
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            current_price = current_data.get('price', 0)
            
            # Get historical data
            historical_data = self.historical_fetcher.get_historical_data(
                symbol, timeframe, days=30
            )
            
            if not historical_data or len(historical_data) < 50:
                return self._generate_fallback_analysis(symbol, timeframe, current_price)
            
            # Calculate all indicators
            rsi_analysis = self._calculate_rsi_analysis(historical_data)
            macd_analysis = self._calculate_macd_analysis(historical_data)
            bollinger_analysis = self._calculate_bollinger_analysis(historical_data)
            ma_analysis = self._calculate_moving_averages_analysis(historical_data)
            volume_analysis = self._calculate_volume_analysis(historical_data)
            momentum_analysis = self._calculate_momentum_analysis(historical_data)
            
            # Calculate support and resistance
            support_levels = self._calculate_support_levels(historical_data)
            resistance_levels = self._calculate_resistance_levels(historical_data)
            
            # Calculate overall signals
            overall_signal, overall_strength, confluence_score = self._calculate_overall_signal(
                rsi_analysis, macd_analysis, bollinger_analysis, ma_analysis, 
                volume_analysis, momentum_analysis
            )
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(
                ma_analysis, macd_analysis, current_price
            )
            
            # Generate chart data
            chart_data = self._generate_chart_data(
                historical_data, rsi_analysis, macd_analysis, bollinger_analysis, 
                ma_analysis, support_levels, resistance_levels
            )
            
            # Assess data quality
            data_quality = self._assess_data_quality(historical_data, current_data)
            
            # Create technical analysis object
            analysis = TechnicalAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                rsi=rsi_analysis,
                macd=macd_analysis,
                bollinger_bands=bollinger_analysis,
                moving_averages=ma_analysis,
                volume_analysis=volume_analysis,
                momentum_indicators=momentum_analysis,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                overall_signal=overall_signal,
                overall_strength=overall_strength,
                confluence_score=confluence_score,
                trend_direction=trend_direction,
                chart_data=chart_data,
                timestamp=datetime.now(),
                data_quality=data_quality
            )
            
            # Cache the analysis
            self.analysis_cache[cache_key] = {
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            # Update performance
            self.performance['total_analyses'] += 1
            self.performance['last_update'] = datetime.now()
            
            logger.info(f"✅ Technical analysis completed for {symbol} {timeframe}")
            logger.info(f"   Overall Signal: {overall_signal.value}")
            logger.info(f"   Confluence Score: {confluence_score:.1f}/100")
            logger.info(f"   Trend: {trend_direction.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed for {symbol} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            return self._generate_fallback_analysis(symbol, timeframe, 0, error=str(e))
    
    def get_multi_timeframe_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis across multiple timeframes"""
        try:
            logger.info(f"📊 Generating multi-timeframe analysis for {symbol}...")
            
            timeframes = ['5m', '15m', '1h']
            analyses = {}
            
            for tf in timeframes:
                try:
                    analysis = self.get_technical_analysis(symbol, tf)
                    analyses[tf] = analysis.to_dict()
                except Exception as e:
                    logger.error(f"❌ Failed to get {tf} analysis: {e}")
                    analyses[tf] = {'error': str(e)}
            
            # Calculate timeframe confluence
            confluence = self._calculate_timeframe_confluence(analyses)
            
            return {
                'symbol': symbol,
                'timeframes': analyses,
                'confluence': confluence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-timeframe analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Get actionable trading signals"""
        try:
            # Get primary timeframe analysis
            primary_analysis = self.get_technical_analysis(symbol, '15m')
            
            # Get multi-timeframe for confirmation
            multi_tf = self.get_multi_timeframe_analysis(symbol)
            
            # Generate specific trading signals
            entry_signals = self._generate_entry_signals(primary_analysis, multi_tf)
            exit_signals = self._generate_exit_signals(primary_analysis)
            risk_signals = self._generate_risk_signals(primary_analysis)
            
            return {
                'symbol': symbol,
                'primary_timeframe': '15m',
                'signals': {
                    'entry': entry_signals,
                    'exit': exit_signals,
                    'risk': risk_signals
                },
                'overall_recommendation': primary_analysis.overall_signal.value,
                'confidence': primary_analysis.confluence_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Trading signals generation failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """Get formatted data for dashboard display"""
        try:
            # Get technical analysis
            analysis = self.get_technical_analysis(symbol, '15m')
            
            # Get trading signals
            signals = self.get_trading_signals(symbol)
            
            # Format for dashboard
            dashboard_data = {
                'symbol': symbol,
                'current_price': analysis.current_price,
                'last_updated': analysis.timestamp.isoformat(),
                
                # Key indicators for gauges/widgets
                'indicators': {
                    'rsi': {
                        'value': analysis.rsi.value,
                        'signal': analysis.rsi.signal.value,
                        'color': self._get_rsi_color(analysis.rsi.value),
                        'description': analysis.rsi.description
                    },
                    'macd': {
                        'value': analysis.macd['macd_line'],
                        'signal_line': analysis.macd['signal_line'],
                        'histogram': analysis.macd['histogram'],
                        'signal': analysis.macd['signal'],
                        'color': self._get_macd_color(analysis.macd)
                    },
                    'bollinger': {
                        'position': analysis.bollinger_bands['bb_position'],
                        'squeeze': analysis.bollinger_bands['squeeze'],
                        'signal': analysis.bollinger_bands['signal'],
                        'color': self._get_bollinger_color(analysis.bollinger_bands)
                    }
                },
                
                # Overall signals for main display
                'overall': {
                    'signal': analysis.overall_signal.value,
                    'strength': analysis.overall_strength.value,
                    'confluence': analysis.confluence_score,
                    'trend': analysis.trend_direction.value,
                    'color': self._get_overall_color(analysis.overall_signal)
                },
                
                # Support/Resistance for charts
                'levels': {
                    'support': analysis.support_levels[:3],  # Top 3
                    'resistance': analysis.resistance_levels[:3]  # Top 3
                },
                
                # Trading recommendations
                'recommendations': signals['signals'],
                
                # Chart data for technical charts
                'chart_data': analysis.chart_data,
                
                # Performance metrics
                'data_quality': analysis.data_quality,
                'cache_performance': {
                    'total_analyses': self.performance['total_analyses'],
                    'cache_hit_rate': (self.performance['cache_hits'] / max(1, self.performance['total_analyses'])) * 100
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data generation failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_real_time_updates(self, callback_function=None):
        """Start real-time technical analysis updates"""
        try:
            logger.info("🔄 Starting real-time technical analysis updates...")
            
            def technical_updater():
                """Background technical analysis update loop"""
                symbols = ['NIFTY', 'BANKNIFTY']
                
                while not self.stop_updates:
                    try:
                        for symbol in symbols:
                            # Update analysis for each symbol
                            analysis = self.get_technical_analysis(symbol, force_refresh=True)
                            
                            # Call callback if provided
                            if callback_function:
                                callback_function(symbol, analysis.to_dict())
                        
                        # Wait before next update
                        time.sleep(300)  # 5 minutes
                        
                    except Exception as e:
                        logger.error(f"❌ Technical update error: {e}")
                        time.sleep(60)  # Wait longer on error
            
            # Start background thread
            self.update_threads['technical'] = threading.Thread(target=technical_updater, daemon=True)
            self.update_threads['technical'].start()
            
            logger.info("✅ Real-time technical analysis updates started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time updates: {e}")
    
    def stop_real_time_updates(self):
        """Stop real-time updates"""
        self.stop_updates = True
        logger.info("⏹️ Real-time technical analysis updates stopped")
    
    # ======================
    # TECHNICAL INDICATOR CALCULATIONS
    # ======================
    
    def _calculate_rsi_analysis(self, historical_data: List[Dict]) -> TechnicalIndicator:
        """Calculate RSI with analysis"""
        try:
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            period = self.indicator_params['rsi']['period']
            
            if len(closes) < period + 1:
                return TechnicalIndicator(
                    name='RSI',
                    value=50.0,
                    signal=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    description='Insufficient data for RSI calculation',
                    timestamp=datetime.now()
                )
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes, period)
            
            # Determine signal
            overbought = self.indicator_params['rsi']['overbought']
            oversold = self.indicator_params['rsi']['oversold']
            
            if rsi >= overbought:
                if rsi >= 80:
                    signal = SignalType.STRONG_SELL
                    strength = SignalStrength.VERY_STRONG
                    description = f'RSI extremely overbought at {rsi:.1f}'
                else:
                    signal = SignalType.SELL
                    strength = SignalStrength.STRONG
                    description = f'RSI overbought at {rsi:.1f}'
            elif rsi <= oversold:
                if rsi <= 20:
                    signal = SignalType.STRONG_BUY
                    strength = SignalStrength.VERY_STRONG
                    description = f'RSI extremely oversold at {rsi:.1f}'
                else:
                    signal = SignalType.BUY
                    strength = SignalStrength.STRONG
                    description = f'RSI oversold at {rsi:.1f}'
            elif 45 <= rsi <= 55:
                signal = SignalType.HOLD
                strength = SignalStrength.WEAK
                description = f'RSI neutral at {rsi:.1f}'
            elif rsi > 55:
                signal = SignalType.SELL
                strength = SignalStrength.MODERATE
                description = f'RSI bearish bias at {rsi:.1f}'
            else:
                signal = SignalType.BUY
                strength = SignalStrength.MODERATE
                description = f'RSI bullish bias at {rsi:.1f}'
            
            return TechnicalIndicator(
                name='RSI',
                value=rsi,
                signal=signal,
                strength=strength,
                description=description,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return TechnicalIndicator(
                name='RSI',
                value=50.0,
                signal=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                description=f'RSI calculation error: {str(e)}',
                timestamp=datetime.now()
            )
    
    def _calculate_macd_analysis(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate MACD with analysis"""
        try:
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            params = self.indicator_params['macd']
            
            if len(closes) < params['slow'] + params['signal']:
                return {
                    'macd_line': 0,
                    'signal_line': 0,
                    'histogram': 0,
                    'signal': 'HOLD',
                    'strength': 'WEAK',
                    'description': 'Insufficient data for MACD calculation',
                    'crossover': False,
                    'divergence': False
                }
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(closes, params['fast'])
            ema_slow = self._calculate_ema(closes, params['slow'])
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD)
            macd_values = []
            for i in range(len(closes) - params['slow'] + 1):
                if i >= params['fast'] - 1:
                    fast_val = self._calculate_ema(closes[:params['slow'] + i], params['fast'])
                    slow_val = self._calculate_ema(closes[:params['slow'] + i], params['slow'])
                    macd_values.append(fast_val - slow_val)
            
            signal_line = self._calculate_ema(macd_values, params['signal']) if len(macd_values) >= params['signal'] else 0
            
            # Histogram
            histogram = macd_line - signal_line
            
            # Previous values for crossover detection
            prev_macd = macd_values[-2] if len(macd_values) >= 2 else macd_line
            prev_signal = self._calculate_ema(macd_values[:-1], params['signal']) if len(macd_values) > params['signal'] else signal_line
            prev_histogram = prev_macd - prev_signal
            
            # Determine signals
            bullish_crossover = (prev_macd <= prev_signal) and (macd_line > signal_line)
            bearish_crossover = (prev_macd >= prev_signal) and (macd_line < signal_line)
            
            if bullish_crossover:
                signal = 'STRONG_BUY'
                strength = 'VERY_STRONG'
                description = 'MACD bullish crossover - strong buy signal'
            elif bearish_crossover:
                signal = 'STRONG_SELL'
                strength = 'VERY_STRONG'
                description = 'MACD bearish crossover - strong sell signal'
            elif macd_line > signal_line and histogram > prev_histogram:
                signal = 'BUY'
                strength = 'STRONG'
                description = 'MACD above signal line with increasing momentum'
            elif macd_line < signal_line and histogram < prev_histogram:
                signal = 'SELL'
                strength = 'STRONG'
                description = 'MACD below signal line with decreasing momentum'
            elif macd_line > signal_line:
                signal = 'BUY'
                strength = 'MODERATE'
                description = 'MACD above signal line'
            elif macd_line < signal_line:
                signal = 'SELL'
                strength = 'MODERATE'
                description = 'MACD below signal line'
            else:
                signal = 'HOLD'
                strength = 'WEAK'
                description = 'MACD neutral'
            
            return {
                'macd_line': round(macd_line, 4),
                'signal_line': round(signal_line, 4),
                'histogram': round(histogram, 4),
                'signal': signal,
                'strength': strength,
                'description': description,
                'crossover': bullish_crossover or bearish_crossover,
                'bullish_crossover': bullish_crossover,
                'bearish_crossover': bearish_crossover,
                'divergence': self._detect_macd_divergence(closes, macd_values)
            }
            
        except Exception as e:
            logger.error(f"❌ MACD calculation failed: {e}")
            return {
                'macd_line': 0,
                'signal_line': 0,
                'histogram': 0,
                'signal': 'HOLD',
                'strength': 'WEAK',
                'description': f'MACD calculation error: {str(e)}',
                'crossover': False,
                'divergence': False
            }
    
    def _calculate_bollinger_analysis(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate Bollinger Bands with analysis"""
        try:
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            params = self.indicator_params['bollinger']
            
            if len(closes) < params['period']:
                return {
                    'upper_band': 0,
                    'middle_band': 0,
                    'lower_band': 0,
                    'bb_position': 0.5,
                    'bandwidth': 0,
                    'squeeze': False,
                    'signal': 'HOLD',
                    'strength': 'WEAK',
                    'description': 'Insufficient data for Bollinger Bands'
                }
            
            # Calculate Bollinger Bands
            recent_closes = closes[-params['period']:]
            sma = sum(recent_closes) / params['period']
            
            variance = sum((price - sma) ** 2 for price in recent_closes) / params['period']
            std_dev = math.sqrt(variance)
            
            upper_band = sma + (params['std_dev'] * std_dev)
            lower_band = sma - (params['std_dev'] * std_dev)
            middle_band = sma
            
            current_price = closes[-1]
            
            # Band position (0 = lower band, 0.5 = middle, 1 = upper band)
            bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
            
            # Bandwidth (volatility measure)
            bandwidth = (upper_band - lower_band) / middle_band * 100
            
            # Squeeze detection (low volatility)
            avg_bandwidth = []
            for i in range(max(0, len(closes) - 20), len(closes)):
                if i >= params['period'] - 1:
                    period_closes = closes[i - params['period'] + 1:i + 1]
                    period_sma = sum(period_closes) / len(period_closes)
                    period_variance = sum((p - period_sma) ** 2 for p in period_closes) / len(period_closes)
                    period_std = math.sqrt(period_variance)
                    period_bandwidth = (2 * params['std_dev'] * period_std) / period_sma * 100
                    avg_bandwidth.append(period_bandwidth)
            
            avg_bw = sum(avg_bandwidth) / len(avg_bandwidth) if avg_bandwidth else bandwidth
            squeeze = bandwidth < (avg_bw * 0.8)  # Current bandwidth < 80% of average
            
            # Determine signals
            if bb_position <= 0.1:  # Near lower band
                signal = 'STRONG_BUY'
                strength = 'VERY_STRONG'
                description = 'Price at lower Bollinger Band - oversold condition'
            elif bb_position >= 0.9:  # Near upper band
                signal = 'STRONG_SELL'
                strength = 'VERY_STRONG'
                description = 'Price at upper Bollinger Band - overbought condition'
            elif bb_position <= 0.2:
                signal = 'BUY'
                strength = 'STRONG'
                description = 'Price near lower Bollinger Band'
            elif bb_position >= 0.8:
                signal = 'SELL'
                strength = 'STRONG'
                description = 'Price near upper Bollinger Band'
            elif squeeze:
                signal = 'HOLD'
                strength = 'MODERATE'
                description = 'Bollinger Band squeeze - low volatility, breakout expected'
            else:
                signal = 'HOLD'
                strength = 'WEAK'
                description = f'Price in middle of Bollinger Bands (position: {bb_position:.1%})'
            
            return {
                'upper_band': round(upper_band, 2),
                'middle_band': round(middle_band, 2),
                'lower_band': round(lower_band, 2),
                'bb_position': round(bb_position, 3),
                'bandwidth': round(bandwidth, 2),
                'squeeze': squeeze,
                'signal': signal,
                'strength': strength,
                'description': description,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Bollinger Bands calculation failed: {e}")
            return {
                'upper_band': 0,
                'middle_band': 0,
                'lower_band': 0,
                'bb_position': 0.5,
                'bandwidth': 0,
                'squeeze': False,
                'signal': 'HOLD',
                'strength': 'WEAK',
                'description': f'Bollinger Bands calculation error: {str(e)}'
            }
    
    def _calculate_moving_averages_analysis(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate moving averages analysis"""
        try:
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            current_price = closes[-1]
            
            # Calculate EMAs
            ema_9 = self._calculate_ema(closes, 9)
            ema_21 = self._calculate_ema(closes, 21)
            ema_55 = self._calculate_ema(closes, 55)
            
            # Calculate SMAs for comparison
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else current_price
            
            # Previous EMAs for crossover detection
            prev_closes = closes[:-1] if len(closes) > 1 else closes
            prev_ema_9 = self._calculate_ema(prev_closes, 9) if len(prev_closes) >= 9 else ema_9
            prev_ema_21 = self._calculate_ema(prev_closes, 21) if len(prev_closes) >= 21 else ema_21
            
            # Detect crossovers
            golden_cross = (prev_ema_9 <= prev_ema_21) and (ema_9 > ema_21)
            death_cross = (prev_ema_9 >= prev_ema_21) and (ema_9 < ema_21)
            
            # Price position relative to EMAs
            above_ema_9 = current_price > ema_9
            above_ema_21 = current_price > ema_21
            above_ema_55 = current_price > ema_55
            
            # EMA alignment
            bullish_alignment = ema_9 > ema_21 > ema_55
            bearish_alignment = ema_9 < ema_21 < ema_55
            
            # Determine overall signal
            if golden_cross:
                signal = 'STRONG_BUY'
                strength = 'VERY_STRONG'
                description = 'Golden cross - EMA 9 crossed above EMA 21'
            elif death_cross:
                signal = 'STRONG_SELL'
                strength = 'VERY_STRONG'
                description = 'Death cross - EMA 9 crossed below EMA 21'
            elif bullish_alignment and above_ema_9 and above_ema_21:
                signal = 'BUY'
                strength = 'STRONG'
                description = 'Bullish EMA alignment with price above key EMAs'
            elif bearish_alignment and not above_ema_9 and not above_ema_21:
                signal = 'SELL'
                strength = 'STRONG'
                description = 'Bearish EMA alignment with price below key EMAs'
            elif above_ema_9 and above_ema_21:
                signal = 'BUY'
                strength = 'MODERATE'
                description = 'Price above key EMAs - bullish bias'
            elif not above_ema_9 and not above_ema_21:
                signal = 'SELL'
                strength = 'MODERATE'
                description = 'Price below key EMAs - bearish bias'
            else:
                signal = 'HOLD'
                strength = 'WEAK'
                description = 'Mixed signals from moving averages'
            
            return {
                'ema_9': round(ema_9, 2),
                'ema_21': round(ema_21, 2),
                'ema_55': round(ema_55, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'golden_cross': golden_cross,
                'death_cross': death_cross,
                'bullish_alignment': bullish_alignment,
                'bearish_alignment': bearish_alignment,
                'price_vs_emas': {
                    'above_ema_9': above_ema_9,
                    'above_ema_21': above_ema_21,
                    'above_ema_55': above_ema_55
                },
                'signal': signal,
                'strength': strength,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"❌ Moving averages calculation failed: {e}")
            return {
                'ema_9': 0,
                'ema_21': 0,
                'ema_55': 0,
                'sma_20': 0,
                'sma_50': 0,
                'signal': 'HOLD',
                'strength': 'WEAK',
                'description': f'Moving averages calculation error: {str(e)}'
            }
    
    def _calculate_volume_analysis(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate volume analysis"""
        try:
            volumes = [int(candle.get('volume', 0)) for candle in historical_data]
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            
            if len(volumes) < 20:
                return {
                    'current_volume': volumes[-1] if volumes else 0,
                    'avg_volume': 0,
                    'volume_ratio': 1.0,
                    'volume_trend': 'NEUTRAL',
                    'signal': 'HOLD',
                    'strength': 'WEAK',
                    'description': 'Insufficient volume data'
                }
            
            current_volume = volumes[-1]
            avg_volume = sum(volumes[-20:]) / 20
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend
            recent_avg = sum(volumes[-5:]) / 5
            older_avg = sum(volumes[-15:-5]) / 10
            volume_trend_ratio = recent_avg / older_avg if older_avg > 0 else 1.0
            
            if volume_trend_ratio > 1.2:
                volume_trend = 'INCREASING'
            elif volume_trend_ratio < 0.8:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'STABLE'
            
            # Price-volume analysis
            price_change = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0
            
            # Determine signal
            if volume_ratio > 2.0 and price_change > 0.5:
                signal = 'STRONG_BUY'
                strength = 'VERY_STRONG'
                description = f'High volume ({volume_ratio:.1f}x avg) with price increase'
            elif volume_ratio > 2.0 and price_change < -0.5:
                signal = 'STRONG_SELL'
                strength = 'VERY_STRONG'
                description = f'High volume ({volume_ratio:.1f}x avg) with price decrease'
            elif volume_ratio > 1.5 and price_change > 0:
                signal = 'BUY'
                strength = 'STRONG'
                description = f'Above average volume ({volume_ratio:.1f}x) supporting price rise'
            elif volume_ratio > 1.5 and price_change < 0:
                signal = 'SELL'
                strength = 'STRONG'
                description = f'Above average volume ({volume_ratio:.1f}x) with price decline'
            elif volume_ratio < 0.5:
                signal = 'HOLD'
                strength = 'WEAK'
                description = f'Low volume ({volume_ratio:.1f}x avg) - lack of conviction'
            else:
                signal = 'HOLD'
                strength = 'MODERATE'
                description = f'Normal volume levels ({volume_ratio:.1f}x avg)'
            
            return {
                'current_volume': current_volume,
                'avg_volume': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'volume_trend': volume_trend,
                'price_change_pct': round(price_change, 2),
                'signal': signal,
                'strength': strength,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"❌ Volume analysis failed: {e}")
            return {
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 1.0,
                'volume_trend': 'NEUTRAL',
                'signal': 'HOLD',
                'strength': 'WEAK',
                'description': f'Volume analysis error: {str(e)}'
            }
    
    def _calculate_momentum_analysis(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        try:
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            highs = [float(candle.get('high', 0)) for candle in historical_data]
            lows = [float(candle.get('low', 0)) for candle in historical_data]
            
            if len(closes) < 14:
                return {
                    'stochastic_k': 50,
                    'stochastic_d': 50,
                    'williams_r': -50,
                    'roc': 0,
                    'signal': 'HOLD',
                    'strength': 'WEAK',
                    'description': 'Insufficient data for momentum analysis'
                }
            
            # Stochastic Oscillator
            period = 14
            lowest_low = min(lows[-period:])
            highest_high = max(highs[-period:])
            current_close = closes[-1]
            
            if highest_high != lowest_low:
                stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                stoch_k = 50
            
            # Stochastic %D (3-period SMA of %K)
            k_values = []
            for i in range(max(0, len(closes) - 20), len(closes)):
                if i >= period - 1:
                    period_lows = lows[i - period + 1:i + 1]
                    period_highs = highs[i - period + 1:i + 1]
                    period_close = closes[i]
                    
                    p_low = min(period_lows)
                    p_high = max(period_highs)
                    
                    if p_high != p_low:
                        k_val = ((period_close - p_low) / (p_high - p_low)) * 100
                    else:
                        k_val = 50
                    k_values.append(k_val)
            
            stoch_d = sum(k_values[-3:]) / 3 if len(k_values) >= 3 else stoch_k
            
            # Williams %R
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100 if highest_high != lowest_low else -50
            
            # Rate of Change (ROC)
            roc_period = 10
            if len(closes) >= roc_period:
                roc = ((closes[-1] - closes[-roc_period]) / closes[-roc_period]) * 100
            else:
                roc = 0
            
            # Determine overall momentum signal
            momentum_signals = []
            
            # Stochastic signals
            if stoch_k < 20 and stoch_d < 20:
                momentum_signals.append(('BUY', 'STRONG', 'Stochastic oversold'))
            elif stoch_k > 80 and stoch_d > 80:
                momentum_signals.append(('SELL', 'STRONG', 'Stochastic overbought'))
            
            # Williams %R signals
            if williams_r < -80:
                momentum_signals.append(('BUY', 'MODERATE', 'Williams %R oversold'))
            elif williams_r > -20:
                momentum_signals.append(('SELL', 'MODERATE', 'Williams %R overbought'))
            
            # ROC signals
            if roc > 2:
                momentum_signals.append(('BUY', 'MODERATE', 'Positive momentum'))
            elif roc < -2:
                momentum_signals.append(('SELL', 'MODERATE', 'Negative momentum'))
            
            # Combine signals
            if not momentum_signals:
                signal = 'HOLD'
                strength = 'WEAK'
                description = 'Neutral momentum indicators'
            else:
                buy_signals = [s for s in momentum_signals if s[0] == 'BUY']
                sell_signals = [s for s in momentum_signals if s[0] == 'SELL']
                
                if len(buy_signals) > len(sell_signals):
                    signal = 'BUY'
                    strength = 'STRONG' if len(buy_signals) >= 2 else 'MODERATE'
                    description = '; '.join([s[2] for s in buy_signals])
                elif len(sell_signals) > len(buy_signals):
                    signal = 'SELL'
                    strength = 'STRONG' if len(sell_signals) >= 2 else 'MODERATE'
                    description = '; '.join([s[2] for s in sell_signals])
                else:
                    signal = 'HOLD'
                    strength = 'MODERATE'
                    description = 'Mixed momentum signals'
            
            return {
                'stochastic_k': round(stoch_k, 2),
                'stochastic_d': round(stoch_d, 2),
                'williams_r': round(williams_r, 2),
                'roc': round(roc, 2),
                'signal': signal,
                'strength': strength,
                'description': description,
                'momentum_signals': momentum_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed: {e}")
            return {
                'stochastic_k': 50,
                'stochastic_d': 50,
                'williams_r': -50,
                'roc': 0,
                'signal': 'HOLD',
                'strength': 'WEAK',
                'description': f'Momentum analysis error: {str(e)}'
            }
    
    def _calculate_support_levels(self, historical_data: List[Dict], num_levels: int = 5) -> List[float]:
        """Calculate support levels"""
        try:
            lows = [float(candle.get('low', 0)) for candle in historical_data]
            
            if len(lows) < 20:
                return []
            
            # Find local minima
            support_levels = []
            window = 5
            
            for i in range(window, len(lows) - window):
                is_local_min = True
                for j in range(i - window, i + window + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_local_min = False
                        break
                
                if is_local_min:
                    support_levels.append(lows[i])
            
            # Sort and return strongest levels
            support_levels.sort()
            return support_levels[:num_levels]
            
        except Exception as e:
            logger.error(f"❌ Support levels calculation failed: {e}")
            return []
    
    def _calculate_resistance_levels(self, historical_data: List[Dict], num_levels: int = 5) -> List[float]:
        """Calculate resistance levels"""
        try:
            highs = [float(candle.get('high', 0)) for candle in historical_data]
            
            if len(highs) < 20:
                return []
            
            # Find local maxima
            resistance_levels = []
            window = 5
            
            for i in range(window, len(highs) - window):
                is_local_max = True
                for j in range(i - window, i + window + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_local_max = False
                        break
                
                if is_local_max:
                    resistance_levels.append(highs[i])
            
            # Sort and return strongest levels
            resistance_levels.sort(reverse=True)
            return resistance_levels[:num_levels]
            
        except Exception as e:
            logger.error(f"❌ Resistance levels calculation failed: {e}")
            return []
    
    # ======================
    # UTILITY METHODS
    # ======================
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            if len(gains) < period:
                return 50.0
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception:
            return 50.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        try:
            if len(prices) < period:
                return sum(prices) / len(prices) if prices else 0
            
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period  # Start with SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception:
            return 0
    
    def _detect_macd_divergence(self, prices: List[float], macd_values: List[float]) -> bool:
        """Detect MACD divergence"""
        try:
            if len(prices) < 10 or len(macd_values) < 10:
                return False
            
            # Simplified divergence detection
            price_trend = prices[-1] - prices[-10]
            macd_trend = macd_values[-1] - macd_values[-10]
            
            # Bullish divergence: price falling, MACD rising
            # Bearish divergence: price rising, MACD falling
            return (price_trend > 0 and macd_trend < 0) or (price_trend < 0 and macd_trend > 0)
            
        except Exception:
            return False
    
    def _calculate_overall_signal(self, rsi_analysis: TechnicalIndicator, macd_analysis: Dict,
                                bollinger_analysis: Dict, ma_analysis: Dict,
                                volume_analysis: Dict, momentum_analysis: Dict) -> Tuple[SignalType, SignalStrength, float]:
        """Calculate overall signal from all indicators"""
        try:
            # Convert signals to numerical scores
            signal_scores = {
                'STRONG_BUY': 100,
                'BUY': 75,
                'HOLD': 50,
                'SELL': 25,
                'STRONG_SELL': 0
            }
            
            # Get individual scores
            rsi_score = signal_scores.get(rsi_analysis.signal.value, 50)
            macd_score = signal_scores.get(macd_analysis['signal'], 50)
            bb_score = signal_scores.get(bollinger_analysis['signal'], 50)
            ma_score = signal_scores.get(ma_analysis['signal'], 50)
            vol_score = signal_scores.get(volume_analysis['signal'], 50)
            mom_score = signal_scores.get(momentum_analysis['signal'], 50)
            
            # Calculate weighted average
            confluence_score = (
                rsi_score * self.signal_weights['rsi'] +
                macd_score * self.signal_weights['macd'] +
                bb_score * self.signal_weights['bollinger'] +
                ma_score * self.signal_weights['ema_cross'] +
                vol_score * self.signal_weights['volume'] +
                mom_score * self.signal_weights['momentum']
            )
            
            # Determine overall signal
            if confluence_score >= 85:
                overall_signal = SignalType.STRONG_BUY
                overall_strength = SignalStrength.VERY_STRONG
            elif confluence_score >= 70:
                overall_signal = SignalType.BUY
                overall_strength = SignalStrength.STRONG
            elif confluence_score >= 55:
                overall_signal = SignalType.BUY
                overall_strength = SignalStrength.MODERATE
            elif confluence_score >= 45:
                overall_signal = SignalType.HOLD
                overall_strength = SignalStrength.MODERATE
            elif confluence_score >= 30:
                overall_signal = SignalType.SELL
                overall_strength = SignalStrength.MODERATE
            elif confluence_score >= 15:
                overall_signal = SignalType.SELL
                overall_strength = SignalStrength.STRONG
            else:
                overall_signal = SignalType.STRONG_SELL
                overall_strength = SignalStrength.VERY_STRONG
            
            return overall_signal, overall_strength, confluence_score
            
        except Exception as e:
            logger.error(f"❌ Overall signal calculation failed: {e}")
            return SignalType.HOLD, SignalStrength.WEAK, 50.0
    
    def _determine_trend_direction(self, ma_analysis: Dict, macd_analysis: Dict, current_price: float) -> TrendDirection:
        """Determine overall trend direction"""
        try:
            # EMA alignment
            bullish_alignment = ma_analysis.get('bullish_alignment', False)
            bearish_alignment = ma_analysis.get('bearish_alignment', False)
            
            # Price vs EMAs
            above_ema_21 = ma_analysis.get('price_vs_emas', {}).get('above_ema_21', False)
            above_ema_55 = ma_analysis.get('price_vs_emas', {}).get('above_ema_55', False)
            
            # MACD trend
            macd_positive = macd_analysis.get('macd_line', 0) > 0
            
            if bullish_alignment and above_ema_21 and above_ema_55 and macd_positive:
                return TrendDirection.STRONG_BULLISH
            elif bullish_alignment and above_ema_21:
                return TrendDirection.BULLISH
            elif bearish_alignment and not above_ema_21 and not above_ema_55 and not macd_positive:
                return TrendDirection.STRONG_BEARISH
            elif bearish_alignment and not above_ema_21:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"❌ Trend direction calculation failed: {e}")
            return TrendDirection.NEUTRAL
    
    def _generate_chart_data(self, historical_data: List[Dict], rsi_analysis: TechnicalIndicator,
                           macd_analysis: Dict, bollinger_analysis: Dict, ma_analysis: Dict,
                           support_levels: List[float], resistance_levels: List[float]) -> Dict[str, Any]:
        """Generate chart data for frontend"""
        try:
            # Prepare OHLCV data
            ohlcv_data = []
            for candle in historical_data[-100:]:  # Last 100 candles
                ohlcv_data.append({
                    'timestamp': candle.get('timestamp', ''),
                    'open': float(candle.get('open', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': int(candle.get('volume', 0))
                })
            
            # Prepare indicator overlays
            overlays = {
                'bollinger_bands': {
                    'upper': bollinger_analysis.get('upper_band', 0),
                    'middle': bollinger_analysis.get('middle_band', 0),
                    'lower': bollinger_analysis.get('lower_band', 0)
                },
                'moving_averages': {
                    'ema_9': ma_analysis.get('ema_9', 0),
                    'ema_21': ma_analysis.get('ema_21', 0),
                    'ema_55': ma_analysis.get('ema_55', 0)
                },
                'support_resistance': {
                    'support': support_levels,
                    'resistance': resistance_levels
                }
            }
            
            # Prepare oscillators
            oscillators = {
                'rsi': {
                    'value': rsi_analysis.value,
                    'overbought': 70,
                    'oversold': 30
                },
                'macd': {
                    'macd_line': macd_analysis.get('macd_line', 0),
                    'signal_line': macd_analysis.get('signal_line', 0),
                    'histogram': macd_analysis.get('histogram', 0)
                }
            }
            
            return {
                'ohlcv': ohlcv_data,
                'overlays': overlays,
                'oscillators': oscillators,
                'chart_config': {
                    'timeframe': '5m',
                    'candle_count': len(ohlcv_data),
                    'price_precision': 2
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Chart data generation failed: {e}")
            return {
                'ohlcv': [],
                'overlays': {},
                'oscillators': {},
                'error': str(e)
            }
    
    def _assess_data_quality(self, historical_data: List[Dict], current_data: Dict[str, Any]) -> float:
        """Assess quality of data for analysis"""
        try:
            quality_score = 100.0
            
            # Check data completeness
            if len(historical_data) < 50:
                quality_score -= 30
            elif len(historical_data) < 100:
                quality_score -= 10
            
            # Check for missing values
            missing_count = 0
            for candle in historical_data[-20:]:  # Check last 20 candles
                if (candle.get('close', 0) == 0 or 
                    candle.get('volume', 0) == 0):
                    missing_count += 1
            
            quality_score -= (missing_count / 20) * 20
            
            # Check current data freshness
            if current_data.get('source') == 'fallback_data':
                quality_score -= 25
            elif 'error' in current_data:
                quality_score -= 40
            
            return max(0, min(100, quality_score))
            
        except Exception:
            return 50.0
    
    def _calculate_timeframe_confluence(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confluence across timeframes"""
        try:
            signals = []
            confidences = []
            
            for tf, analysis in analyses.items():
                if 'error' not in analysis:
                    overall = analysis.get('overall', {})
                    signal = overall.get('signal', 'HOLD')
                    confluence = overall.get('confluence_score', 50)
                    
                    signals.append(signal)
                    confidences.append(confluence)
            
            if not signals:
                return {'confluence': 'LOW', 'score': 0, 'agreement': 0}
            
            # Check signal agreement
            buy_signals = sum(1 for s in signals if 'BUY' in s)
            sell_signals = sum(1 for s in signals if 'SELL' in s)
            total_signals = len(signals)
            
            agreement_pct = max(buy_signals, sell_signals) / total_signals * 100
            avg_confidence = sum(confidences) / len(confidences)
            
            if agreement_pct >= 80 and avg_confidence >= 70:
                confluence_level = 'VERY_HIGH'
            elif agreement_pct >= 70 and avg_confidence >= 60:
                confluence_level = 'HIGH'
            elif agreement_pct >= 60:
                confluence_level = 'MODERATE'
            else:
                confluence_level = 'LOW'
            
            return {
                'confluence': confluence_level,
                'score': round(avg_confidence, 1),
                'agreement': round(agreement_pct, 1),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_timeframes': total_signals
            }
            
        except Exception as e:
            logger.error(f"❌ Timeframe confluence calculation failed: {e}")
            return {'confluence': 'LOW', 'score': 0, 'agreement': 0, 'error': str(e)}
    
    def _generate_entry_signals(self, analysis: TechnicalAnalysis, multi_tf: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific entry signals"""
        try:
            entry_signals = []
            
            # RSI entry signals
            if analysis.rsi.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                entry_signals.append({
                    'type': 'RSI_ENTRY',
                    'signal': analysis.rsi.signal.value,
                    'description': analysis.rsi.description,
                    'strength': analysis.rsi.strength.value,
                    'entry_price': analysis.current_price,
                    'target': analysis.current_price * 1.02,
                    'stop_loss': analysis.current_price * 0.98
                })
            
            # MACD crossover signals
            if analysis.macd.get('bullish_crossover') or analysis.macd.get('bearish_crossover'):
                signal_type = 'BUY' if analysis.macd.get('bullish_crossover') else 'SELL'
                entry_signals.append({
                    'type': 'MACD_CROSSOVER',
                    'signal': signal_type,
                    'description': analysis.macd.get('description', 'MACD crossover signal'),
                    'strength': 'VERY_STRONG',
                    'entry_price': analysis.current_price,
                    'target': analysis.current_price * (1.015 if signal_type == 'BUY' else 0.985),
                    'stop_loss': analysis.current_price * (0.985 if signal_type == 'BUY' else 1.015)
                })
            
            # Bollinger Band squeeze breakout
            if analysis.bollinger_bands.get('squeeze'):
                entry_signals.append({
                    'type': 'BB_SQUEEZE_BREAKOUT',
                    'signal': 'WATCH',
                    'description': 'Bollinger Band squeeze - breakout imminent',
                    'strength': 'MODERATE',
                    'entry_price': analysis.current_price,
                    'breakout_above': analysis.bollinger_bands.get('upper_band', 0),
                    'breakout_below': analysis.bollinger_bands.get('lower_band', 0)
                })
            
            # Support/Resistance bounce
            for support in analysis.support_levels[:2]:
                if abs(analysis.current_price - support) / support < 0.005:  # Within 0.5%
                    entry_signals.append({
                        'type': 'SUPPORT_BOUNCE',
                        'signal': 'BUY',
                        'description': f'Price near support level at {support:.2f}',
                        'strength': 'STRONG',
                        'entry_price': support,
                        'target': support * 1.02,
                        'stop_loss': support * 0.995
                    })
            
            for resistance in analysis.resistance_levels[:2]:
                if abs(analysis.current_price - resistance) / resistance < 0.005:  # Within 0.5%
                    entry_signals.append({
                        'type': 'RESISTANCE_REJECTION',
                        'signal': 'SELL',
                        'description': f'Price near resistance level at {resistance:.2f}',
                        'strength': 'STRONG',
                        'entry_price': resistance,
                        'target': resistance * 0.98,
                        'stop_loss': resistance * 1.005
                    })
            
            return entry_signals
            
        except Exception as e:
            logger.error(f"❌ Entry signals generation failed: {e}")
            return []
    
    def _generate_exit_signals(self, analysis: TechnicalAnalysis) -> List[Dict[str, Any]]:
        """Generate exit signals for existing positions"""
        try:
            exit_signals = []
            
            # RSI overbought/oversold exits
            if analysis.rsi.value >= 80:
                exit_signals.append({
                    'type': 'RSI_OVERBOUGHT_EXIT',
                    'signal': 'SELL',
                    'description': f'RSI extremely overbought at {analysis.rsi.value:.1f}',
                    'urgency': 'HIGH',
                    'exit_price': analysis.current_price
                })
            elif analysis.rsi.value <= 20:
                exit_signals.append({
                    'type': 'RSI_OVERSOLD_EXIT',
                    'signal': 'COVER',
                    'description': f'RSI extremely oversold at {analysis.rsi.value:.1f}',
                    'urgency': 'HIGH',
                    'exit_price': analysis.current_price
                })
            
            # Bollinger Band extreme exits
            bb_position = analysis.bollinger_bands.get('bb_position', 0.5)
            if bb_position >= 0.95:
                exit_signals.append({
                    'type': 'BB_UPPER_EXIT',
                    'signal': 'SELL',
                    'description': 'Price at upper Bollinger Band extreme',
                    'urgency': 'MEDIUM',
                    'exit_price': analysis.bollinger_bands.get('upper_band', analysis.current_price)
                })
            elif bb_position <= 0.05:
                exit_signals.append({
                    'type': 'BB_LOWER_EXIT',
                    'signal': 'COVER',
                    'description': 'Price at lower Bollinger Band extreme',
                    'urgency': 'MEDIUM',
                    'exit_price': analysis.bollinger_bands.get('lower_band', analysis.current_price)
                })
            
            # Moving average breakdown/breakthrough
            if analysis.moving_averages.get('death_cross'):
                exit_signals.append({
                    'type': 'MA_DEATH_CROSS',
                    'signal': 'SELL',
                    'description': 'Death cross - major trend reversal',
                    'urgency': 'HIGH',
                    'exit_price': analysis.current_price
                })
            elif analysis.moving_averages.get('golden_cross'):
                exit_signals.append({
                    'type': 'MA_GOLDEN_CROSS',
                    'signal': 'COVER',
                    'description': 'Golden cross - major trend reversal',
                    'urgency': 'HIGH',
                    'exit_price': analysis.current_price
                })
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"❌ Exit signals generation failed: {e}")
            return []
    
    def _generate_risk_signals(self, analysis: TechnicalAnalysis) -> List[Dict[str, Any]]:
        """Generate risk management signals"""
        try:
            risk_signals = []
            
            # High volatility warning
            if analysis.bollinger_bands.get('bandwidth', 0) > 5:
                risk_signals.append({
                    'type': 'HIGH_VOLATILITY',
                    'level': 'WARNING',
                    'description': f'High volatility detected - Bollinger bandwidth: {analysis.bollinger_bands.get("bandwidth", 0):.1f}%',
                    'recommendation': 'Consider reducing position sizes'
                })
            
            # Low volume warning
            volume_ratio = analysis.volume_analysis.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                risk_signals.append({
                    'type': 'LOW_VOLUME',
                    'level': 'CAUTION',
                    'description': f'Below average volume ({volume_ratio:.1f}x) - lack of conviction',
                    'recommendation': 'Wait for volume confirmation before entering trades'
                })
            
            # Conflicting signals warning
            if analysis.confluence_score < 60:
                risk_signals.append({
                    'type': 'CONFLICTING_SIGNALS',
                    'level': 'CAUTION',
                    'description': f'Low confluence score ({analysis.confluence_score:.1f}) - mixed signals',
                    'recommendation': 'Wait for clearer signals or use smaller position sizes'
                })
            
            # Trend uncertainty
            if analysis.trend_direction == TrendDirection.NEUTRAL:
                risk_signals.append({
                    'type': 'TREND_UNCERTAINTY',
                    'level': 'INFO',
                    'description': 'No clear trend direction identified',
                    'recommendation': 'Consider range-bound trading strategies'
                })
            
            return risk_signals
            
        except Exception as e:
            logger.error(f"❌ Risk signals generation failed: {e}")
            return []
    
    def _get_rsi_color(self, rsi_value: float) -> str:
        """Get color for RSI value"""
        if rsi_value >= 70:
            return 'red'
        elif rsi_value <= 30:
            return 'green'
        elif rsi_value >= 60:
            return 'orange'
        elif rsi_value <= 40:
            return 'lightgreen'
        else:
            return 'yellow'
    
    def _get_macd_color(self, macd_data: Dict[str, Any]) -> str:
        """Get color for MACD"""
        if macd_data.get('bullish_crossover'):
            return 'green'
        elif macd_data.get('bearish_crossover'):
            return 'red'
        elif macd_data.get('macd_line', 0) > macd_data.get('signal_line', 0):
            return 'lightgreen'
        else:
            return 'lightred'
    
    def _get_bollinger_color(self, bb_data: Dict[str, Any]) -> str:
        """Get color for Bollinger Bands"""
        bb_position = bb_data.get('bb_position', 0.5)
        if bb_position >= 0.8:
            return 'red'
        elif bb_position <= 0.2:
            return 'green'
        elif bb_data.get('squeeze'):
            return 'orange'
        else:
            return 'blue'
    
    def _get_overall_color(self, signal: SignalType) -> str:
        """Get color for overall signal"""
        color_map = {
            SignalType.STRONG_BUY: 'darkgreen',
            SignalType.BUY: 'green',
            SignalType.HOLD: 'yellow',
            SignalType.SELL: 'red',
            SignalType.STRONG_SELL: 'darkred'
        }
        return color_map.get(signal, 'gray')
    
    def _generate_fallback_analysis(self, symbol: str, timeframe: str, current_price: float, 
                                  error: str = None) -> TechnicalAnalysis:
        """Generate fallback analysis when calculation fails"""
        
        # Create fallback RSI
        fallback_rsi = TechnicalIndicator(
            name='RSI',
            value=50.0,
            signal=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            description=f'Analysis unavailable: {error}' if error else 'Insufficient data',
            timestamp=datetime.now()
        )
        
        # Create fallback data
        fallback_data = {
            'signal': 'HOLD',
            'strength': 'WEAK',
            'description': 'Analysis unavailable'
        }
        
        return TechnicalAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            rsi=fallback_rsi,
            macd=fallback_data.copy(),
            bollinger_bands=fallback_data.copy(),
            moving_averages=fallback_data.copy(),
            volume_analysis=fallback_data.copy(),
            momentum_indicators=fallback_data.copy(),
            support_levels=[],
            resistance_levels=[],
            overall_signal=SignalType.HOLD,
            overall_strength=SignalStrength.WEAK,
            confluence_score=0,
            trend_direction=TrendDirection.NEUTRAL,
            chart_data={'error': error or 'Data unavailable'},
            timestamp=datetime.now(),
            data_quality=0
        )

def main():
    """Test the Technical Indicators Display System"""
    print("📊 TradeMind AI Technical Indicators Display System")
    print("=" * 70)
    
    try:
        # Initialize system
        tech_system = TechnicalIndicatorsDisplay()
        
        print("\n🧪 Testing Technical Analysis System...")
        
        # Test NIFTY analysis
        print("\n📈 NIFTY Technical Analysis:")
        nifty_analysis = tech_system.get_technical_analysis('NIFTY', '15m')
        
        print(f"   Current Price: ₹{nifty_analysis.current_price:,.2f}")
        print(f"   Overall Signal: {nifty_analysis.overall_signal.value}")
        print(f"   Strength: {nifty_analysis.overall_strength.value}")
        print(f"   Confluence Score: {nifty_analysis.confluence_score:.1f}/100")
        print(f"   Trend: {nifty_analysis.trend_direction.value}")
        
        print(f"\n   📊 Key Indicators:")
        print(f"      RSI: {nifty_analysis.rsi.value:.1f} ({nifty_analysis.rsi.signal.value})")
        print(f"      MACD: {nifty_analysis.macd['signal']} ({nifty_analysis.macd['description'][:50]}...)")
        print(f"      Bollinger: {nifty_analysis.bollinger_bands['signal']}")
        print(f"      Moving Averages: {nifty_analysis.moving_averages['signal']}")
        
        # Test trading signals
        print("\n🎯 Trading Signals:")
        signals = tech_system.get_trading_signals('NIFTY')
        
        entry_signals = signals['signals']['entry']
        if entry_signals:
            print(f"   Entry Signals ({len(entry_signals)}):")
            for signal in entry_signals[:3]:
                print(f"      • {signal['type']}: {signal['signal']} - {signal['description'][:50]}...")
        
        exit_signals = signals['signals']['exit']
        if exit_signals:
            print(f"   Exit Signals ({len(exit_signals)}):")
            for signal in exit_signals[:2]:
                print(f"      • {signal['type']}: {signal['signal']} - {signal['description'][:50]}...")
        
        # Test multi-timeframe analysis
        print("\n⏰ Multi-Timeframe Analysis:")
        multi_tf = tech_system.get_multi_timeframe_analysis('NIFTY')
        confluence = multi_tf.get('confluence', {})
        print(f"   Timeframe Confluence: {confluence.get('confluence', 'UNKNOWN')}")
        print(f"   Agreement: {confluence.get('agreement', 0):.1f}%")
        print(f"   Average Score: {confluence.get('score', 0):.1f}/100")
        
        # Test dashboard data
        print("\n📱 Dashboard Data Format:")
        dashboard_data = tech_system.get_dashboard_data('NIFTY')
        
        indicators = dashboard_data.get('indicators', {})
        print(f"   RSI Widget: {indicators.get('rsi', {}).get('value', 0):.1f} ({indicators.get('rsi', {}).get('color', 'gray')})")
        print(f"   MACD Widget: {indicators.get('macd', {}).get('signal', 'UNKNOWN')}")
        print(f"   Overall Signal: {dashboard_data.get('overall', {}).get('signal', 'UNKNOWN')}")
        
        # Test performance metrics
        print("\n📈 System Performance:")
        print(f"   Total Analyses: {tech_system.performance['total_analyses']}")
        print(f"   Cache Hit Rate: {(tech_system.performance['cache_hits'] / max(1, tech_system.performance['total_analyses'])) * 100:.1f}%")
        print(f"   Data Quality: {nifty_analysis.data_quality:.1f}/100")
        
        print("\n✅ Technical Indicators Display System testing completed!")
        print("\n🚀 Integration commands:")
        print("   # Add to dashboard_backend.py:")
        print("   from technical_indicators_display import TechnicalIndicatorsDisplay")
        print("   self.tech_system = TechnicalIndicatorsDisplay()")
        print("   ")
        print("   # New API endpoint:")
        print("   @app.route('/api/technical/<symbol>')")
        print("   def get_technical_data(symbol):")
        print("       return jsonify(self.tech_system.get_dashboard_data(symbol))")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        try:
            tech_system.stop_real_time_updates()
        except:
            pass

if __name__ == "__main__":
    main()