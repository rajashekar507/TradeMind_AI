"""
TradeMind AI Decision Engine - Complete Master Orchestrator
This is the FINAL version - no future changes needed
Combines all AI components into unified trading decisions
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
from dataclasses import dataclass
from enum import Enum

# Import all TradeMind AI components
try:
    from ml_trader import SelfLearningTrader, integrate_ml_trader
    from global_market_analyzer import GlobalMarketAnalyzer, integrate_global_markets
    from real_news_analyzer import RealNewsAnalyzer
    from realtime_market_data import RealTimeMarketData
    from portfolio_manager import PortfolioManager
    from historical_data import HistoricalDataFetcher
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please ensure all TradeMind modules are in the same directory")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_decision_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    """Trade direction enumeration"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    AVOID = "AVOID"

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_HIGH = "VERY_HIGH"  # 90-100%
    HIGH = "HIGH"            # 75-89%
    MODERATE = "MODERATE"    # 60-74%
    LOW = "LOW"              # 40-59%
    VERY_LOW = "VERY_LOW"    # 0-39%

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class TradingDecision:
    """Complete trading decision data structure"""
    # Basic recommendation
    action: str                    # BUY, SELL, HOLD, AVOID
    symbol: str                   # NIFTY, BANKNIFTY
    option_type: str              # CE, PE
    strike_price: float           # Recommended strike
    
    # Confidence and risk
    overall_confidence: float     # 0-100
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    
    # AI component scores
    ml_confidence: float          # ML trader confidence
    technical_score: float        # Technical analysis score
    global_sentiment_score: float # Global market sentiment
    news_sentiment_score: float   # News sentiment score
    risk_score: float            # Risk assessment score
    
    # Position sizing
    recommended_lots: int         # Number of lots
    capital_allocation: float     # Amount to allocate
    
    # Timing and targets
    entry_price_range: Tuple[float, float]  # (min, max) entry price
    target_price: float           # Profit target
    stop_loss: float             # Stop loss price
    max_hold_time: int           # Maximum hold time in minutes
    
    # Reasoning and explanation
    reasoning: List[str]          # Decision reasoning
    warnings: List[str]           # Risk warnings
    market_conditions: Dict[str, Any]  # Current market state
    
    # Metadata
    timestamp: datetime
    expiry_date: str
    days_to_expiry: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'action': self.action,
            'symbol': self.symbol,
            'option_type': self.option_type,
            'strike_price': self.strike_price,
            'overall_confidence': round(self.overall_confidence, 2),
            'confidence_level': self.confidence_level.value,
            'risk_level': self.risk_level.value,
            'scores': {
                'ml_confidence': round(self.ml_confidence, 2),
                'technical_score': round(self.technical_score, 2),
                'global_sentiment': round(self.global_sentiment_score, 2),
                'news_sentiment': round(self.news_sentiment_score, 2),
                'risk_score': round(self.risk_score, 2)
            },
            'position': {
                'recommended_lots': self.recommended_lots,
                'capital_allocation': round(self.capital_allocation, 2),
                'entry_range': [round(self.entry_price_range[0], 2), round(self.entry_price_range[1], 2)],
                'target_price': round(self.target_price, 2),
                'stop_loss': round(self.stop_loss, 2),
                'max_hold_time': self.max_hold_time
            },
            'explanation': {
                'reasoning': self.reasoning,
                'warnings': self.warnings,
                'market_conditions': self.market_conditions
            },
            'timing': {
                'timestamp': self.timestamp.isoformat(),
                'expiry_date': self.expiry_date,
                'days_to_expiry': self.days_to_expiry
            }
        }

class AIDecisionEngine:
    """Master AI Decision Engine - Orchestrates all AI components"""
    
    def __init__(self):
        """Initialize the AI Decision Engine"""
        logger.info("🧠 Initializing TradeMind AI Decision Engine...")
        
        # Initialize all AI components
        self._initialize_components()
        
        # Decision cache and history
        self.decision_cache = {}
        self.decision_history = []
        self.max_history = 1000
        
        # Configuration
        self.config = {
            'min_confidence_threshold': 60.0,  # Minimum confidence to recommend trades
            'max_risk_per_trade': 0.01,       # Maximum 1% risk per trade
            'max_daily_trades': 8,            # Maximum trades per day
            'max_concurrent_positions': 4,     # Maximum open positions
            'preferred_expiry_days': [7, 14, 21],  # Preferred expiry days
            'update_interval': 30,            # Decision update interval in seconds
        }
        
        # Component weights for final decision
        self.component_weights = {
            'ml_trader': 0.30,          # 30% weight for ML predictions
            'technical_analysis': 0.25,  # 25% weight for technical indicators
            'global_sentiment': 0.20,   # 20% weight for global markets
            'news_sentiment': 0.15,     # 15% weight for news sentiment
            'risk_assessment': 0.10     # 10% weight for risk factors
        }
        
        # Market state tracking
        self.current_market_state = {}
        self.last_decision_time = {}
        
        # Background update control
        self.stop_updates = False
        self.update_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'average_confidence': 0,
            'last_update': datetime.now()
        }
        
        logger.info("✅ AI Decision Engine initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all AI and data components"""
        try:
            # Core AI components
            self.ml_trader = SelfLearningTrader()
            self.global_analyzer = GlobalMarketAnalyzer()
            self.news_analyzer = RealNewsAnalyzer()
            
            # Market data and analysis
            self.market_data = RealTimeMarketData()
            self.historical_fetcher = HistoricalDataFetcher()
            self.portfolio_manager = PortfolioManager()
            
            # Start real-time market data updates
            self.market_data.start_real_time_updates(self._on_market_data_update)
            
            logger.info("✅ All AI components initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_trading_decision(self, symbol: str = 'NIFTY', force_refresh: bool = False) -> TradingDecision:
        """Get comprehensive trading decision for given symbol"""
        try:
            logger.info(f"🔍 Generating trading decision for {symbol}...")
            
            # Check cache first (unless force refresh)
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if not force_refresh and cache_key in self.decision_cache:
                cached_time = self.decision_cache[cache_key]['timestamp']
                if (datetime.now() - cached_time).seconds < 300:  # 5 minutes cache
                    logger.info("📋 Returning cached decision")
                    return self.decision_cache[cache_key]['decision']
            
            # Step 1: Gather market data and conditions
            market_conditions = self._gather_market_conditions(symbol)
            
            # Step 2: Get individual component analyses
            ml_analysis = self._get_ml_analysis(symbol, market_conditions)
            technical_analysis = self._get_technical_analysis(symbol, market_conditions)
            global_analysis = self._get_global_analysis(market_conditions)
            news_analysis = self._get_news_analysis(market_conditions)
            risk_analysis = self._get_risk_analysis(symbol, market_conditions)
            
            # Step 3: Combine all analyses into unified decision
            decision = self._synthesize_decision(
                symbol, market_conditions, ml_analysis, technical_analysis,
                global_analysis, news_analysis, risk_analysis
            )
            
            # Step 4: Apply final filters and validations
            decision = self._apply_decision_filters(decision, market_conditions)
            
            # Step 5: Cache and store decision
            self.decision_cache[cache_key] = {
                'decision': decision,
                'timestamp': datetime.now()
            }
            
            # Add to history
            self.decision_history.append(decision)
            if len(self.decision_history) > self.max_history:
                self.decision_history.pop(0)
            
            # Update performance metrics
            self.performance_metrics['total_decisions'] += 1
            self.performance_metrics['last_update'] = datetime.now()
            
            logger.info(f"✅ Decision generated: {decision.action} {symbol} {decision.option_type} {decision.strike_price} (Confidence: {decision.overall_confidence:.1f}%)")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Decision generation failed for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return self._generate_fallback_decision(symbol, error=str(e))
    
    def get_real_time_signals(self) -> Dict[str, Any]:
        """Get real-time trading signals for dashboard"""
        try:
            # Get decisions for both NIFTY and BANKNIFTY
            nifty_decision = self.get_trading_decision('NIFTY')
            banknifty_decision = self.get_trading_decision('BANKNIFTY')
            
            # Calculate overall market bias
            market_bias = self._calculate_market_bias([nifty_decision, banknifty_decision])
            
            # Get market conditions summary
            market_summary = self._get_market_summary()
            
            return {
                'signals': {
                    'NIFTY': nifty_decision.to_dict(),
                    'BANKNIFTY': banknifty_decision.to_dict()
                },
                'market_bias': market_bias,
                'market_summary': market_summary,
                'performance': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Real-time signals generation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_real_time_updates(self, callback_function=None):
        """Start background real-time decision updates"""
        try:
            logger.info("🔄 Starting real-time AI decision updates...")
            
            def decision_updater():
                """Background decision update loop"""
                while not self.stop_updates:
                    try:
                        # Generate fresh decisions
                        signals = self.get_real_time_signals()
                        
                        # Call callback if provided
                        if callback_function:
                            callback_function(signals)
                        
                        # Wait for next update
                        time.sleep(self.config['update_interval'])
                        
                    except Exception as e:
                        logger.error(f"❌ Decision update error: {e}")
                        time.sleep(60)  # Wait longer on error
            
            # Start background thread
            self.update_thread = threading.Thread(target=decision_updater, daemon=True)
            self.update_thread.start()
            
            logger.info("✅ Real-time AI updates started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time updates: {e}")
    
    def stop_real_time_updates(self):
        """Stop background updates"""
        self.stop_updates = True
        if self.market_data:
            self.market_data.stop_real_time_updates()
        logger.info("⏹️ Real-time AI updates stopped")
    
    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        try:
            recent_decisions = self.decision_history[-limit:] if limit else self.decision_history
            return [decision.to_dict() for decision in recent_decisions]
        except Exception as e:
            logger.error(f"❌ Decision history retrieval failed: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get AI performance summary"""
        try:
            if not self.decision_history:
                return {'message': 'No decisions made yet'}
            
            # Calculate performance metrics
            total_decisions = len(self.decision_history)
            high_confidence_decisions = sum(1 for d in self.decision_history if d.overall_confidence >= 75)
            
            # Calculate average confidence by time period
            today_decisions = [d for d in self.decision_history if d.timestamp.date() == datetime.now().date()]
            avg_confidence_today = np.mean([d.overall_confidence for d in today_decisions]) if today_decisions else 0
            
            # Risk distribution
            risk_distribution = {}
            for decision in self.decision_history:
                risk = decision.risk_level.value
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
            
            return {
                'total_decisions': total_decisions,
                'high_confidence_decisions': high_confidence_decisions,
                'high_confidence_percentage': (high_confidence_decisions / total_decisions) * 100 if total_decisions > 0 else 0,
                'average_confidence_today': round(avg_confidence_today, 2),
                'decisions_today': len(today_decisions),
                'risk_distribution': risk_distribution,
                'component_weights': self.component_weights,
                'last_update': self.performance_metrics['last_update'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Performance summary generation failed: {e}")
            return {'error': str(e)}
    
    # ======================
    # PRIVATE ANALYSIS METHODS
    # ======================
    
    def _gather_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Gather comprehensive market conditions"""
        try:
            # Get current market data
            if symbol == 'NIFTY':
                index_data = self.market_data.get_nifty_data()
            else:
                index_data = self.market_data.get_banknifty_data()
            
            # Get VIX data
            vix_data = self.market_data.get_vix_data()
            
            # Get historical data for technical analysis
            historical_data = self.historical_fetcher.get_historical_data(symbol, '5m', 5)
            
            # Get option chain data
            option_data = self.market_data.get_option_chain_data(symbol)
            
            # Get global markets snapshot
            global_snapshot = self.market_data.get_global_markets_data()
            
            # Calculate key metrics
            current_price = index_data.get('price', 0)
            price_change_pct = index_data.get('change_percent', 0)
            volume = index_data.get('volume', 0)
            vix_level = vix_data.get('price', 15)
            
            # Market timing analysis
            now = datetime.now()
            market_session = self._get_market_session(now)
            time_score = self._calculate_time_score(now)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'volume': volume,
                'vix_level': vix_level,
                'market_session': market_session,
                'time_score': time_score,
                'historical_data': historical_data,
                'option_data': option_data,
                'global_sentiment': global_snapshot.get('market_sentiment', {}),
                'timestamp': datetime.now(),
                'raw_data': {
                    'index': index_data,
                    'vix': vix_data,
                    'global': global_snapshot
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Market conditions gathering failed: {e}")
            return {'symbol': symbol, 'error': str(e), 'timestamp': datetime.now()}
    
    def _get_ml_analysis(self, symbol: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML trader analysis"""
        try:
            # Prepare features for ML model
            trade_features = {
                'rsi': self._calculate_rsi(market_conditions.get('historical_data', [])),
                'macd_signal': self._calculate_macd_signal(market_conditions.get('historical_data', [])),
                'oi_ratio': self._calculate_oi_ratio(market_conditions.get('option_data', {})),
                'iv_skew': self._calculate_iv_skew(market_conditions.get('option_data', {})),
                'moneyness': 1.0,  # Will be calculated per strike
                'days_to_expiry': 7,  # Default
                'vix_level': market_conditions.get('vix_level', 15),
                'pcr': self._calculate_pcr(market_conditions.get('option_data', {})),
                'delta': 0.4,  # Default
                'ai_confidence': 80,  # Base confidence
                'timestamp': datetime.now(),
                'price_change_pct': market_conditions.get('price_change_pct', 0),
                'volume_ratio': self._calculate_volume_ratio(market_conditions)
            }
            
            # Get ML recommendation
            ml_decision = self.ml_trader.should_take_trade(trade_features)
            
            # Get strategy recommendations
            strategy_recommendations = self.ml_trader.get_strategy_recommendations()
            
            return {
                'decision': ml_decision['decision'],
                'confidence': ml_decision['ml_confidence'],
                'recommendation': ml_decision['recommendation'],
                'insights': ml_decision.get('insights', []),
                'position_size': ml_decision.get('adjusted_position_size', 1),
                'strategy_recommendations': strategy_recommendations,
                'features_used': trade_features
            }
            
        except Exception as e:
            logger.error(f"❌ ML analysis failed: {e}")
            return {
                'decision': False,
                'confidence': 50,
                'recommendation': 'NEUTRAL',
                'error': str(e)
            }
    
    def _get_technical_analysis(self, symbol: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical analysis"""
        try:
            historical_data = market_conditions.get('historical_data', [])
            
            if not historical_data or len(historical_data) < 20:
                return {
                    'rsi': 50,
                    'macd_signal': 0,
                    'bollinger_signal': 'NEUTRAL',
                    'trend': 'SIDEWAYS',
                    'score': 50,
                    'signals': ['Insufficient data for technical analysis']
                }
            
            # Calculate technical indicators
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            highs = [float(candle.get('high', 0)) for candle in historical_data]
            lows = [float(candle.get('low', 0)) for candle in historical_data]
            
            rsi = self._calculate_rsi(historical_data)
            macd_signal = self._calculate_macd_signal(historical_data)
            bollinger_signal = self._calculate_bollinger_signal(historical_data)
            trend = self._calculate_trend(historical_data)
            
            # Generate technical signals
            signals = []
            score = 50  # Base score
            
            # RSI signals
            if rsi < 30:
                signals.append("RSI oversold - potential bullish reversal")
                score += 15
            elif rsi > 70:
                signals.append("RSI overbought - potential bearish reversal")
                score -= 15
            elif 30 <= rsi <= 45:
                signals.append("RSI in bullish zone")
                score += 8
            elif 55 <= rsi <= 70:
                signals.append("RSI in bearish zone")
                score -= 8
            
            # MACD signals
            if macd_signal > 0:
                signals.append("MACD bullish crossover")
                score += 10
            elif macd_signal < 0:
                signals.append("MACD bearish crossover")
                score -= 10
            
            # Bollinger Band signals
            if bollinger_signal == 'OVERSOLD':
                signals.append("Price near lower Bollinger Band")
                score += 12
            elif bollinger_signal == 'OVERBOUGHT':
                signals.append("Price near upper Bollinger Band")
                score -= 12
            
            # Trend signals
            if trend == 'STRONG_BULLISH':
                signals.append("Strong bullish trend confirmed")
                score += 20
            elif trend == 'BULLISH':
                signals.append("Bullish trend in progress")
                score += 10
            elif trend == 'STRONG_BEARISH':
                signals.append("Strong bearish trend confirmed")
                score -= 20
            elif trend == 'BEARISH':
                signals.append("Bearish trend in progress")
                score -= 10
            
            # Clamp score between 0-100
            score = max(0, min(100, score))
            
            return {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'bollinger_signal': bollinger_signal,
                'trend': trend,
                'score': score,
                'signals': signals,
                'support_levels': self._calculate_support_levels(historical_data),
                'resistance_levels': self._calculate_resistance_levels(historical_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Technical analysis failed: {e}")
            return {
                'rsi': 50,
                'macd_signal': 0,
                'bollinger_signal': 'NEUTRAL',
                'trend': 'SIDEWAYS',
                'score': 50,
                'error': str(e)
            }
    
    def _get_global_analysis(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get global market analysis"""
        try:
            # Get global market bias
            global_bias = self.global_analyzer.get_trading_bias()
            
            # Get trading decision
            should_trade = self.global_analyzer.should_trade_today()
            
            # Calculate global score
            direction = global_bias.get('direction', 'NEUTRAL')
            strength = global_bias.get('strength', 50)
            
            if direction == 'BULLISH':
                score = 50 + (strength * 0.5)
            elif direction == 'BEARISH':
                score = 50 - (strength * 0.5)
            else:
                score = 50
            
            return {
                'direction': direction,
                'strength': strength,
                'score': score,
                'should_trade': should_trade.get('trade_today', True),
                'confidence': should_trade.get('confidence', 50),
                'global_factors': global_bias.get('reasons', []),
                'warnings': should_trade.get('warnings', []),
                'opportunities': should_trade.get('opportunities', [])
            }
            
        except Exception as e:
            logger.error(f"❌ Global analysis failed: {e}")
            return {
                'direction': 'NEUTRAL',
                'strength': 50,
                'score': 50,
                'should_trade': True,
                'error': str(e)
            }
    
    def _get_news_analysis(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get news sentiment analysis"""
        try:
            # Fetch and analyze recent news
            news_items = self.news_analyzer.fetch_market_news("NIFTY BANKNIFTY", hours_back=6)
            analyzed_news = self.news_analyzer.analyze_news_sentiment(news_items)
            market_mood = self.news_analyzer.calculate_market_mood(analyzed_news)
            
            # Convert market mood to score
            mood = market_mood.get('market_mood', 'NEUTRAL/MIXED')
            bullish_pct = market_mood.get('bullish_percentage', 33)
            bearish_pct = market_mood.get('bearish_percentage', 33)
            
            if 'EXTREMELY BULLISH' in mood:
                score = 85 + (bullish_pct - 70) * 0.5
            elif 'BULLISH' in mood:
                score = 60 + (bullish_pct - 55) * 1.5
            elif 'EXTREMELY BEARISH' in mood:
                score = 15 - (bearish_pct - 70) * 0.5
            elif 'BEARISH' in mood:
                score = 40 - (bearish_pct - 55) * 1.5
            else:
                score = 50
            
            # Generate news insights
            insights = []
            if bullish_pct > 60:
                insights.append(f"Strong positive news sentiment ({bullish_pct:.1f}% bullish)")
            elif bearish_pct > 60:
                insights.append(f"Strong negative news sentiment ({bearish_pct:.1f}% bearish)")
            else:
                insights.append("Mixed news sentiment - no clear bias")
            
            return {
                'market_mood': mood,
                'score': max(0, min(100, score)),
                'bullish_percentage': bullish_pct,
                'bearish_percentage': bearish_pct,
                'sentiment_score': market_mood.get('average_sentiment_score', 0),
                'insights': insights,
                'top_headlines': [news.get('title', '') for news in analyzed_news[:3]]
            }
            
        except Exception as e:
            logger.error(f"❌ News analysis failed: {e}")
            return {
                'market_mood': 'NEUTRAL/MIXED',
                'score': 50,
                'bullish_percentage': 33,
                'bearish_percentage': 33,
                'error': str(e)
            }
    
    def _get_risk_analysis(self, symbol: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk assessment"""
        try:
            # Get current portfolio exposure
            current_balance = self.portfolio_manager.fetch_current_balance()
            current_positions = len(self.portfolio_manager.get_current_positions())
            daily_pnl = self.portfolio_manager.calculate_daily_pnl()
            
            # Calculate risk factors
            vix_level = market_conditions.get('vix_level', 15)
            price_volatility = abs(market_conditions.get('price_change_pct', 0))
            
            # Risk scoring
            risk_score = 50  # Base risk
            risk_factors = []
            
            # VIX risk
            if vix_level > 25:
                risk_score += 20
                risk_factors.append(f"High volatility (VIX: {vix_level:.1f})")
            elif vix_level > 20:
                risk_score += 10
                risk_factors.append(f"Elevated volatility (VIX: {vix_level:.1f})")
            elif vix_level < 12:
                risk_score -= 5
                risk_factors.append(f"Low volatility environment (VIX: {vix_level:.1f})")
            
            # Price movement risk
            if price_volatility > 2:
                risk_score += 15
                risk_factors.append(f"High price volatility ({price_volatility:.1f}%)")
            elif price_volatility > 1:
                risk_score += 8
                risk_factors.append(f"Moderate price volatility ({price_volatility:.1f}%)")
            
            # Position concentration risk
            if current_positions >= 4:
                risk_score += 15
                risk_factors.append(f"High position concentration ({current_positions} positions)")
            elif current_positions >= 2:
                risk_score += 5
                risk_factors.append(f"Moderate position count ({current_positions} positions)")
            
            # Daily P&L risk
            daily_loss_pct = (daily_pnl / current_balance) * 100 if current_balance > 0 else 0
            if daily_loss_pct < -2:
                risk_score += 25
                risk_factors.append(f"Significant daily loss ({daily_loss_pct:.1f}%)")
            elif daily_loss_pct < -1:
                risk_score += 10
                risk_factors.append(f"Daily loss threshold breached ({daily_loss_pct:.1f}%)")
            
            # Market session risk
            market_session = market_conditions.get('market_session', 'UNKNOWN')
            if market_session == 'OPENING':
                risk_score += 5
                risk_factors.append("Opening session - higher volatility expected")
            elif market_session == 'CLOSING':
                risk_score += 8
                risk_factors.append("Closing session - position squaring expected")
            
            # Clamp risk score
            risk_score = max(0, min(100, risk_score))
            
            # Determine risk level
            if risk_score >= 80:
                risk_level = RiskLevel.VERY_HIGH
            elif risk_score >= 65:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 45:
                risk_level = RiskLevel.MODERATE
            elif risk_score >= 25:
                risk_level = RiskLevel.LOW
            else:
                risk_level = RiskLevel.VERY_LOW
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'current_positions': current_positions,
                'daily_pnl_pct': daily_loss_pct,
                'vix_level': vix_level,
                'max_additional_positions': max(0, 4 - current_positions),
                'recommended_position_size': self._calculate_risk_adjusted_size(risk_score, current_balance)
            }
            
        except Exception as e:
            logger.error(f"❌ Risk analysis failed: {e}")
            return {
                'risk_score': 50,
                'risk_level': RiskLevel.MODERATE,
                'risk_factors': [],
                'error': str(e)
            }
    
    def _synthesize_decision(self, symbol: str, market_conditions: Dict[str, Any], 
                           ml_analysis: Dict[str, Any], technical_analysis: Dict[str, Any],
                           global_analysis: Dict[str, Any], news_analysis: Dict[str, Any],
                           risk_analysis: Dict[str, Any]) -> TradingDecision:
        """Synthesize all analyses into unified trading decision"""
        try:
            # Extract individual scores
            ml_score = ml_analysis.get('confidence', 50)
            tech_score = technical_analysis.get('score', 50)
            global_score = global_analysis.get('score', 50)
            news_score = news_analysis.get('score', 50)
            risk_score = risk_analysis.get('risk_score', 50)
            
            # Calculate weighted overall confidence
            overall_confidence = (
                ml_score * self.component_weights['ml_trader'] +
                tech_score * self.component_weights['technical_analysis'] +
                global_score * self.component_weights['global_sentiment'] +
                news_score * self.component_weights['news_sentiment'] +
                (100 - risk_score) * self.component_weights['risk_assessment']
            )
            
            # Determine confidence level
            if overall_confidence >= 90:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif overall_confidence >= 75:
                confidence_level = ConfidenceLevel.HIGH
            elif overall_confidence >= 60:
                confidence_level = ConfidenceLevel.MODERATE
            elif overall_confidence >= 40:
                confidence_level = ConfidenceLevel.LOW
            else:
                confidence_level = ConfidenceLevel.VERY_LOW
            
            # Determine trade direction
            if overall_confidence >= 65:
                if (ml_score + tech_score + global_score + news_score) / 4 > 55:
                    action = 'BUY'
                    option_type = 'CE'
                    direction = TradeDirection.BULLISH
                else:
                    action = 'BUY'
                    option_type = 'PE'
                    direction = TradeDirection.BEARISH
            elif overall_confidence >= 45:
                action = 'HOLD'
                option_type = 'CE' if tech_score > 50 else 'PE'
                direction = TradeDirection.NEUTRAL
            else:
                action = 'AVOID'
                option_type = 'CE'
                direction = TradeDirection.AVOID
            
            # Calculate strike price
            current_price = market_conditions.get('current_price', 25000)
            strike_gap = 50 if symbol == 'NIFTY' else 100
            
            if option_type == 'CE':
                # For calls, use slightly OTM strikes
                strike_price = current_price + strike_gap
            else:
                # For puts, use slightly OTM strikes
                strike_price = current_price - strike_gap
            
            # Round strike to nearest strike gap
            strike_price = round(strike_price / strike_gap) * strike_gap
            
            # Calculate position sizing
            recommended_lots = risk_analysis.get('recommended_position_size', 1)
            lot_size = 75 if symbol == 'NIFTY' else 30
            
            # Estimate option premium (simplified)
            estimated_premium = self._estimate_option_premium(
                current_price, strike_price, option_type, 
                market_conditions.get('vix_level', 15)
            )
            
            capital_allocation = estimated_premium * recommended_lots * lot_size
            
            # Calculate targets and stop loss
            target_price = estimated_premium * 1.5  # 50% profit target
            stop_loss = estimated_premium * 0.7     # 30% stop loss
            
            # Entry price range
            entry_range = (estimated_premium * 0.95, estimated_premium * 1.05)
            
            # Generate reasoning
            reasoning = []
            if ml_analysis.get('confidence', 0) > 70:
                reasoning.append(f"ML model shows {ml_analysis.get('recommendation', 'NEUTRAL')} with {ml_score:.0f}% confidence")
            
            if tech_score > 60:
                reasoning.extend(technical_analysis.get('signals', [])[:2])
            elif tech_score < 40:
                reasoning.append("Technical indicators suggest caution")
            
            if global_score > 60:
                reasoning.append(f"Global markets are {global_analysis.get('direction', 'NEUTRAL').lower()}")
            
            if news_score > 60:
                reasoning.append("Positive news sentiment supports the move")
            elif news_score < 40:
                reasoning.append("Negative news sentiment creates headwinds")
            
            # Generate warnings
            warnings = []
            if risk_score > 70:
                warnings.extend(risk_analysis.get('risk_factors', []))
            
            if market_conditions.get('vix_level', 15) > 20:
                warnings.append("High volatility environment - consider smaller position sizes")
            
            if overall_confidence < 60:
                warnings.append("Low confidence signal - proceed with caution")
            
            # Expiry calculation
            expiry_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            
            return TradingDecision(
                action=action,
                symbol=symbol,
                option_type=option_type,
                strike_price=strike_price,
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                risk_level=risk_analysis.get('risk_level', RiskLevel.MODERATE),
                ml_confidence=ml_score,
                technical_score=tech_score,
                global_sentiment_score=global_score,
                news_sentiment_score=news_score,
                risk_score=risk_score,
                recommended_lots=recommended_lots,
                capital_allocation=capital_allocation,
                entry_price_range=entry_range,
                target_price=target_price,
                stop_loss=stop_loss,
                max_hold_time=240,  # 4 hours max
                reasoning=reasoning,
                warnings=warnings,
                market_conditions=market_conditions,
                timestamp=datetime.now(),
                expiry_date=expiry_date,
                days_to_expiry=7
            )
            
        except Exception as e:
            logger.error(f"❌ Decision synthesis failed: {e}")
            return self._generate_fallback_decision(symbol, error=str(e))
    
    def _apply_decision_filters(self, decision: TradingDecision, market_conditions: Dict[str, Any]) -> TradingDecision:
        """Apply final filters and validations to decision"""
        try:
            # Check minimum confidence threshold
            if decision.overall_confidence < self.config['min_confidence_threshold']:
                decision.action = 'AVOID'
                decision.warnings.append(f"Confidence below threshold ({decision.overall_confidence:.1f}% < {self.config['min_confidence_threshold']}%)")
            
            # Check risk limits
            if decision.risk_level == RiskLevel.VERY_HIGH:
                decision.action = 'AVOID'
                decision.warnings.append("Risk level too high for trading")
            
            # Check market hours
            market_session = market_conditions.get('market_session', 'UNKNOWN')
            if market_session == 'CLOSED':
                decision.action = 'AVOID'
                decision.warnings.append("Market is closed")
            
            # Check position limits
            current_positions = len(self.portfolio_manager.get_current_positions())
            if current_positions >= self.config['max_concurrent_positions']:
                decision.action = 'AVOID'
                decision.warnings.append(f"Maximum positions limit reached ({current_positions})")
            
            # Adjust position size based on available capital
            available_capital = self.portfolio_manager.fetch_current_balance()
            max_allocation = available_capital * self.config['max_risk_per_trade']
            
            if decision.capital_allocation > max_allocation:
                # Reduce position size
                reduction_factor = max_allocation / decision.capital_allocation
                decision.recommended_lots = max(1, int(decision.recommended_lots * reduction_factor))
                decision.capital_allocation = max_allocation
                decision.warnings.append("Position size reduced due to capital constraints")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Decision filtering failed: {e}")
            decision.warnings.append(f"Filter application failed: {str(e)}")
            return decision
    
    # ======================
    # UTILITY METHODS
    # ======================
    
    def _calculate_rsi(self, historical_data: List[Dict], period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if not historical_data or len(historical_data) < period + 1:
                return 50.0
            
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            
            gains = []
            losses = []
            
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
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
            
            return round(rsi, 2)
            
        except Exception:
            return 50.0
    
    def _calculate_macd_signal(self, historical_data: List[Dict]) -> int:
        """Calculate MACD signal (-1, 0, 1)"""
        try:
            if not historical_data or len(historical_data) < 26:
                return 0
            
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            
            # Simple MACD calculation
            ema_12 = sum(closes[-12:]) / 12
            ema_26 = sum(closes[-26:]) / 26
            macd = ema_12 - ema_26
            
            # Signal line (simplified)
            signal = sum(closes[-9:]) / 9 - sum(closes[-18:]) / 18
            
            if macd > signal:
                return 1  # Bullish
            elif macd < signal:
                return -1  # Bearish
            else:
                return 0  # Neutral
                
        except Exception:
            return 0
    
    def _calculate_bollinger_signal(self, historical_data: List[Dict], period: int = 20) -> str:
        """Calculate Bollinger Band signal"""
        try:
            if not historical_data or len(historical_data) < period:
                return 'NEUTRAL'
            
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            recent_closes = closes[-period:]
            
            sma = sum(recent_closes) / period
            variance = sum((price - sma) ** 2 for price in recent_closes) / period
            std_dev = variance ** 0.5
            
            upper_band = sma + (2 * std_dev)
            lower_band = sma - (2 * std_dev)
            current_price = closes[-1]
            
            if current_price <= lower_band:
                return 'OVERSOLD'
            elif current_price >= upper_band:
                return 'OVERBOUGHT'
            else:
                return 'NEUTRAL'
                
        except Exception:
            return 'NEUTRAL'
    
    def _calculate_trend(self, historical_data: List[Dict]) -> str:
        """Calculate trend direction"""
        try:
            if not historical_data or len(historical_data) < 10:
                return 'SIDEWAYS'
            
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            
            # Simple trend calculation using moving averages
            short_ma = sum(closes[-5:]) / 5
            long_ma = sum(closes[-10:]) / 10
            
            diff_pct = ((short_ma - long_ma) / long_ma) * 100
            
            if diff_pct > 1:
                return 'STRONG_BULLISH'
            elif diff_pct > 0.3:
                return 'BULLISH'
            elif diff_pct < -1:
                return 'STRONG_BEARISH'
            elif diff_pct < -0.3:
                return 'BEARISH'
            else:
                return 'SIDEWAYS'
                
        except Exception:
            return 'SIDEWAYS'
    
    def _calculate_support_levels(self, historical_data: List[Dict]) -> List[float]:
        """Calculate support levels"""
        try:
            if not historical_data:
                return []
            
            lows = [float(candle.get('low', 0)) for candle in historical_data]
            lows.sort()
            
            # Return bottom 3 levels
            return lows[:3]
            
        except Exception:
            return []
    
    def _calculate_resistance_levels(self, historical_data: List[Dict]) -> List[float]:
        """Calculate resistance levels"""
        try:
            if not historical_data:
                return []
            
            highs = [float(candle.get('high', 0)) for candle in historical_data]
            highs.sort(reverse=True)
            
            # Return top 3 levels
            return highs[:3]
            
        except Exception:
            return []
    
    def _calculate_oi_ratio(self, option_data: Dict[str, Any]) -> float:
        """Calculate OI ratio from option data"""
        try:
            strikes = option_data.get('option_data', [])
            if not strikes:
                return 1.0
            
            total_ce_oi = sum(strike.get('ce_oi', 0) for strike in strikes)
            total_pe_oi = sum(strike.get('pe_oi', 0) for strike in strikes)
            
            if total_pe_oi == 0:
                return 2.0
            
            return total_ce_oi / total_pe_oi
            
        except Exception:
            return 1.0
    
    def _calculate_iv_skew(self, option_data: Dict[str, Any]) -> float:
        """Calculate IV skew"""
        try:
            # Simplified IV skew calculation
            # In real implementation, this would use actual IV data
            return np.random.uniform(-5, 5)
            
        except Exception:
            return 0.0
    
    def _calculate_pcr(self, option_data: Dict[str, Any]) -> float:
        """Calculate Put-Call Ratio"""
        try:
            strikes = option_data.get('option_data', [])
            if not strikes:
                return 1.0
            
            total_pe_volume = sum(strike.get('pe_volume', 0) for strike in strikes)
            total_ce_volume = sum(strike.get('ce_volume', 0) for strike in strikes)
            
            if total_ce_volume == 0:
                return 2.0
            
            return total_pe_volume / total_ce_volume
            
        except Exception:
            return 1.0
    
    def _calculate_volume_ratio(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate volume ratio"""
        try:
            current_volume = market_conditions.get('volume', 0)
            historical_data = market_conditions.get('historical_data', [])
            
            if not historical_data:
                return 1.0
            
            avg_volume = sum(candle.get('volume', 0) for candle in historical_data) / len(historical_data)
            
            if avg_volume == 0:
                return 1.0
            
            return current_volume / avg_volume
            
        except Exception:
            return 1.0
    
    def _get_market_session(self, now: datetime) -> str:
        """Get current market session"""
        try:
            hour = now.hour
            minute = now.minute
            
            if hour < 9 or (hour == 9 and minute < 15):
                return 'PRE_MARKET'
            elif hour == 9 and minute < 30:
                return 'OPENING'
            elif hour < 15 or (hour == 15 and minute < 15):
                return 'REGULAR'
            elif hour == 15 and minute < 30:
                return 'CLOSING'
            else:
                return 'CLOSED'
                
        except Exception:
            return 'UNKNOWN'
    
    def _calculate_time_score(self, now: datetime) -> float:
        """Calculate time-based trading score"""
        try:
            hour = now.hour
            minute = now.minute
            
            # Best trading times: 9:30-11:00 and 14:00-15:15
            if (9 <= hour <= 10) or (14 <= hour <= 15 and minute <= 15):
                return 0.9
            elif 11 <= hour <= 13:
                return 0.6  # Lunch time - lower activity
            elif hour == 15 and minute > 15:
                return 0.3  # After market close
            else:
                return 0.7  # Regular trading time
                
        except Exception:
            return 0.5
    
    def _estimate_option_premium(self, spot_price: float, strike_price: float, 
                                option_type: str, vix_level: float) -> float:
        """Estimate option premium (simplified Black-Scholes approximation)"""
        try:
            # Simplified premium calculation
            moneyness = spot_price / strike_price if option_type == 'CE' else strike_price / spot_price
            
            # Base premium based on moneyness
            if moneyness > 1.02:  # Deep ITM
                base_premium = spot_price * 0.02
            elif moneyness > 1.005:  # Slightly ITM
                base_premium = spot_price * 0.015
            elif moneyness > 0.995:  # ATM
                base_premium = spot_price * 0.012
            elif moneyness > 0.98:  # Slightly OTM
                base_premium = spot_price * 0.008
            else:  # Deep OTM
                base_premium = spot_price * 0.004
            
            # Adjust for volatility
            vix_multiplier = (vix_level / 15)  # Normalize around VIX 15
            adjusted_premium = base_premium * vix_multiplier
            
            return max(1.0, adjusted_premium)  # Minimum premium of 1
            
        except Exception:
            return 50.0  # Fallback premium
    
    def _calculate_risk_adjusted_size(self, risk_score: float, available_capital: float) -> int:
        """Calculate risk-adjusted position size"""
        try:
            # Base position size
            base_lots = 2
            
            # Adjust based on risk score
            if risk_score > 80:
                return 1  # High risk - minimum size
            elif risk_score > 60:
                return max(1, base_lots - 1)
            elif risk_score < 30:
                return min(4, base_lots + 1)  # Low risk - can increase
            else:
                return base_lots
                
        except Exception:
            return 1
    
    def _calculate_market_bias(self, decisions: List[TradingDecision]) -> Dict[str, Any]:
        """Calculate overall market bias from multiple decisions"""
        try:
            if not decisions:
                return {'bias': 'NEUTRAL', 'strength': 50}
            
            bullish_signals = sum(1 for d in decisions if d.option_type == 'CE' and d.action == 'BUY')
            bearish_signals = sum(1 for d in decisions if d.option_type == 'PE' and d.action == 'BUY')
            total_signals = len(decisions)
            
            avg_confidence = sum(d.overall_confidence for d in decisions) / total_signals
            
            if bullish_signals > bearish_signals:
                bias = 'BULLISH'
                strength = (bullish_signals / total_signals) * avg_confidence
            elif bearish_signals > bullish_signals:
                bias = 'BEARISH'
                strength = (bearish_signals / total_signals) * avg_confidence
            else:
                bias = 'NEUTRAL'
                strength = avg_confidence
            
            return {
                'bias': bias,
                'strength': round(strength, 1),
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'average_confidence': round(avg_confidence, 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Market bias calculation failed: {e}")
            return {'bias': 'NEUTRAL', 'strength': 50, 'error': str(e)}
    
    def _get_market_summary(self) -> Dict[str, Any]:
        """Get market summary for dashboard"""
        try:
            # Get current market data
            nifty_data = self.market_data.get_nifty_data()
            banknifty_data = self.market_data.get_banknifty_data()
            vix_data = self.market_data.get_vix_data()
            
            return {
                'nifty': {
                    'price': nifty_data.get('price', 0),
                    'change_pct': nifty_data.get('change_percent', 0)
                },
                'banknifty': {
                    'price': banknifty_data.get('price', 0),
                    'change_pct': banknifty_data.get('change_percent', 0)
                },
                'vix': {
                    'level': vix_data.get('price', 15),
                    'change_pct': vix_data.get('change_percent', 0)
                },
                'market_status': self.market_data._get_market_status(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Market summary generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_fallback_decision(self, symbol: str, error: str = None) -> TradingDecision:
        """Generate fallback decision when analysis fails"""
        return TradingDecision(
            action='AVOID',
            symbol=symbol,
            option_type='CE',
            strike_price=25000 if symbol == 'NIFTY' else 55000,
            overall_confidence=0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            risk_level=RiskLevel.VERY_HIGH,
            ml_confidence=0,
            technical_score=0,
            global_sentiment_score=0,
            news_sentiment_score=0,
            risk_score=100,
            recommended_lots=0,
            capital_allocation=0,
            entry_price_range=(0, 0),
            target_price=0,
            stop_loss=0,
            max_hold_time=0,
            reasoning=['System error occurred during analysis'],
            warnings=[f'Analysis failed: {error}'] if error else ['System temporarily unavailable'],
            market_conditions={'error': error},
            timestamp=datetime.now(),
            expiry_date=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            days_to_expiry=7
        )
    
    def _on_market_data_update(self, data_type: str, data: Dict[str, Any]):
        """Callback for market data updates"""
        try:
            # Update current market state
            self.current_market_state[data_type] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            # Trigger decision refresh if significant change
            if data_type == 'indices':
                nifty_change = abs(data.get('nifty', {}).get('change_percent', 0))
                banknifty_change = abs(data.get('banknifty', {}).get('change_percent', 0))
                
                if nifty_change > 0.5 or banknifty_change > 0.5:
                    # Significant price movement - refresh decisions
                    logger.info(f"📊 Significant market movement detected - refreshing decisions")
                    
        except Exception as e:
            logger.error(f"❌ Market data update callback failed: {e}")

def main():
    """Test the AI Decision Engine"""
    print("🧠 TradeMind AI Decision Engine")
    print("=" * 60)
    
    try:
        # Initialize decision engine
        ai_engine = AIDecisionEngine()
        
        print("\n🧪 Testing AI Decision Engine...")
        
        # Test NIFTY decision
        print("\n📈 NIFTY Trading Decision:")
        nifty_decision = ai_engine.get_trading_decision('NIFTY')
        print(f"   Action: {nifty_decision.action}")
        print(f"   Symbol: {nifty_decision.symbol}")
        print(f"   Option: {nifty_decision.option_type}")
        print(f"   Strike: {nifty_decision.strike_price}")
        print(f"   Confidence: {nifty_decision.overall_confidence:.1f}%")
        print(f"   Risk Level: {nifty_decision.risk_level.value}")
        print(f"   Recommended Lots: {nifty_decision.recommended_lots}")
        
        if nifty_decision.reasoning:
            print(f"   Reasoning:")
            for reason in nifty_decision.reasoning[:3]:
                print(f"      • {reason}")
        
        # Test BANKNIFTY decision
        print("\n🏦 BANKNIFTY Trading Decision:")
        banknifty_decision = ai_engine.get_trading_decision('BANKNIFTY')
        print(f"   Action: {banknifty_decision.action}")
        print(f"   Option: {banknifty_decision.option_type}")
        print(f"   Strike: {banknifty_decision.strike_price}")
        print(f"   Confidence: {banknifty_decision.overall_confidence:.1f}%")
        
        # Test real-time signals
        print("\n📡 Real-Time Signals:")
        signals = ai_engine.get_real_time_signals()
        market_bias = signals.get('market_bias', {})
        print(f"   Market Bias: {market_bias.get('bias', 'UNKNOWN')}")
        print(f"   Bias Strength: {market_bias.get('strength', 0):.1f}%")
        
        # Test performance summary
        print("\n📊 Performance Summary:")
        performance = ai_engine.get_performance_summary()
        print(f"   Total Decisions: {performance.get('total_decisions', 0)}")
        print(f"   High Confidence %: {performance.get('high_confidence_percentage', 0):.1f}%")
        
        print("\n✅ AI Decision Engine testing completed!")
        print("\n🚀 To start real-time updates, call: ai_engine.start_real_time_updates()")
        print("⏹️ To stop updates, call: ai_engine.stop_real_time_updates()")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        try:
            ai_engine.stop_real_time_updates()
        except:
            pass

if __name__ == "__main__":
    main()