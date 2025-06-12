"""
TradeMind_AI: Enhanced Master Trader with All Features
Integrates indicators, historical data, news, and exact recommendations
Updated to use centralized constants and enhanced error handling
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import centralized constants
from config.constants import (
    LOT_SIZES, 
    get_lot_size, 
    RISK_MANAGEMENT,
    SECURITY_IDS,
    EXCHANGE_SEGMENTS,
    MARKET_HOURS
)

# Import shared balance utility
from src.utils.balance_utils import (
    fetch_current_balance,
    send_balance_alert,
    update_balance_after_trade
)

# Import all modules with enhanced error handling
try:
    from src.analysis.exact_recommender import ExactStrikeRecommender
    EXACT_RECOMMENDER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ExactStrikeRecommender not available: {e}")
    EXACT_RECOMMENDER_AVAILABLE = False

try:
    from src.portfolio.portfolio_manager import PortfolioManager
    PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PortfolioManager not available: {e}")
    PORTFOLIO_MANAGER_AVAILABLE = False

try:
    from src.data.market_data import MarketDataEngine
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MarketDataEngine not available: {e}")
    MARKET_DATA_AVAILABLE = False

try:
    from src.analysis.technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TechnicalIndicators not available: {e}")
    TECHNICAL_INDICATORS_AVAILABLE = False

try:
    from src.analysis.historical_data import HistoricalDataFetcher
    HISTORICAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ HistoricalDataFetcher not available: {e}")
    HISTORICAL_DATA_AVAILABLE = False

try:
    from src.analysis.news_sentiment import NewsSentimentAnalyzer
    NEWS_SENTIMENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NewsSentimentAnalyzer not available: {e}")
    NEWS_SENTIMENT_AVAILABLE = False

try:
    from src.core.unified_trading_engine import UnifiedTradingEngine, TradingMode
    UNIFIED_TRADING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ UnifiedTradingEngine not available: {e}")
    UNIFIED_TRADING_AVAILABLE = False

class EnhancedMasterTrader:
    """The Ultimate TradeMind_AI Master Trading System - ENHANCED with Error Handling"""
    
    def __init__(self):
        """Initialize the enhanced master trading system"""
        self.logger = logging.getLogger('EnhancedMasterTrader')
        
        print("🚀 Initializing ENHANCED TradeMind_AI MASTER TRADER...")
        print("🎯 Now with Technical Indicators, Historical Data & News!")
        print("🛡️ Enhanced Error Handling & Shared Utilities!")
        print("="*70)
        
        # Load environment
        load_dotenv()
        
        # Initialize ALL components with error handling
        self.market_data_engine = None
        self.smart_trader = None
        self.strike_recommender = None
        self.portfolio_manager = None
        self.technical_indicators = None
        self.historical_data = None
        self.news_analyzer = None
        
        # Master trader settings
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Trading session settings
        self.session_active = False
        self.trades_today = 0
        self.max_trades_per_day = RISK_MANAGEMENT['MAX_TRADES_PER_DAY']
        self.session_start_time = None
        
        # Performance tracking
        self.session_pnl = 0
        self.best_trade_today = 0
        self.worst_trade_today = 0
        
        # Use centralized lot sizes
        self.lot_sizes = LOT_SIZES
        
        # Current balance using shared utility
        try:
            self.current_balance = fetch_current_balance()
            print(f"💰 Current Balance: ₹{self.current_balance:,.2f}")
        except Exception as e:
            print(f"⚠️ Could not fetch balance: {e}")
            self.current_balance = 100000  # Default
        
        print("✅ Enhanced Master Trader initialized!")
        self.send_master_alert("🚀 ENHANCED TradeMind_AI MASTER TRADER is ONLINE!")
    
    def initialize_all_systems(self) -> bool:
        """Initialize all trading subsystems with comprehensive error handling"""
        print("\n🔧 Initializing ALL Trading Systems...")
        
        success_count = 0
        total_systems = 7
        
        try:
            # 1. Market Data Engine
            if MARKET_DATA_AVAILABLE:
                try:
                    print("📊 Starting Market Data Engine...")
                    self.market_data_engine = MarketDataEngine()
                    print("✅ Market Data Engine ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Market Data Engine failed: {e}")
            else:
                print("⚠️ Market Data Engine not available")
            
            # 2. Unified Trading Engine
            if UNIFIED_TRADING_AVAILABLE:
                try:
                    print("🤖 Starting Unified Trading Engine...")
                    self.smart_trader = UnifiedTradingEngine(TradingMode.PAPER)
                    print("✅ Unified Trading Engine ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Unified Trading Engine failed: {e}")
            else:
                print("⚠️ Unified Trading Engine not available")
            
            # 3. Strike Recommender
            if EXACT_RECOMMENDER_AVAILABLE:
                try:
                    print("🎯 Starting Strike Recommender...")
                    self.strike_recommender = ExactStrikeRecommender()
                    print("✅ Strike Recommender ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Strike Recommender failed: {e}")
            else:
                print("⚠️ Strike Recommender not available")
            
            # 4. Portfolio Manager
            if PORTFOLIO_MANAGER_AVAILABLE:
                try:
                    print("💼 Starting Portfolio Manager...")
                    self.portfolio_manager = PortfolioManager()
                    print("✅ Portfolio Manager ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Portfolio Manager failed: {e}")
            else:
                print("⚠️ Portfolio Manager not available")
            
            # 5. Technical Indicators
            if TECHNICAL_INDICATORS_AVAILABLE:
                try:
                    print("📈 Starting Technical Indicators...")
                    self.technical_indicators = TechnicalIndicators()
                    print("✅ Technical Indicators ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Technical Indicators failed: {e}")
            else:
                print("⚠️ Technical Indicators not available")
            
            # 6. Historical Data
            if HISTORICAL_DATA_AVAILABLE:
                try:
                    print("📚 Starting Historical Data Fetcher...")
                    self.historical_data = HistoricalDataFetcher()
                    print("✅ Historical Data ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Historical Data failed: {e}")
            else:
                print("⚠️ Historical Data not available")
            
            # 7. News Analyzer
            if NEWS_SENTIMENT_AVAILABLE:
                try:
                    print("📰 Starting News Sentiment Analyzer...")
                    self.news_analyzer = NewsSentimentAnalyzer()
                    print("✅ News Analyzer ready!")
                    success_count += 1
                except Exception as e:
                    print(f"❌ News Analyzer failed: {e}")
            else:
                print("⚠️ News Analyzer not available")
            
            # Summary
            print(f"\n📊 SYSTEM INITIALIZATION SUMMARY:")
            print(f"✅ Successfully initialized: {success_count}/{total_systems} systems")
            print(f"⚠️ Failed/Unavailable: {total_systems - success_count}/{total_systems} systems")
            
            if success_count >= 3:  # At least 3 systems working
                print("🎉 SUFFICIENT SYSTEMS INITIALIZED - Ready to trade!")
                return True
            else:
                print("❌ INSUFFICIENT SYSTEMS - Need at least 3 working systems")
                return False
            
        except Exception as e:
            self.logger.error(f"Critical error during system initialization: {e}")
            print(f"❌ Critical initialization error: {e}")
            return False
    
    def send_master_alert(self, message: str) -> bool:
        """Send alert via Telegram with error handling"""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                print(f"📱 Alert (Telegram disabled): {message}")
                return False
            
            # Use shared utility for sending alerts
            return send_balance_alert(message)
            
        except Exception as e:
            print(f"❌ Alert error: {e}")
            return False
    
    def get_comprehensive_market_analysis(self, symbol: str) -> Dict:
        """Get analysis from ALL available systems"""
        print(f"\n🔍 Running COMPREHENSIVE analysis for {symbol}...")
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'recommendations': [],
            'errors': [],
            'systems_used': []
        }
        
        # 1. Market Data Analysis
        if self.market_data_engine:
            try:
                print("📊 Getting real-time market data...")
                market_data = self.market_data_engine.get_option_chain(
                    SECURITY_IDS[symbol], symbol
                )
                if market_data:
                    analysis['market_data'] = market_data
                    analysis['spot_price'] = market_data.get('underlying_price', 0)
                    analysis['systems_used'].append('Market Data')
                    print(f"✅ Spot Price: ₹{analysis['spot_price']}")
                else:
                    analysis['errors'].append('Market data fetch failed')
            except Exception as e:
                analysis['errors'].append(f'Market data error: {str(e)}')
                print(f"❌ Market data error: {e}")
        
        # 2. Technical Analysis
        if self.technical_indicators:
            try:
                print("📈 Running technical analysis...")
                tech_signals = self.technical_indicators.get_indicator_signals(symbol)
                analysis['technical'] = tech_signals
                analysis['systems_used'].append('Technical Analysis')
                
                # Add to recommendations
                if tech_signals['signal'] != 'NEUTRAL':
                    analysis['recommendations'].append({
                        'source': 'Technical',
                        'signal': tech_signals['signal'],
                        'confidence': tech_signals['confidence']
                    })
                print(f"✅ Technical Signal: {tech_signals['signal']}")
            except Exception as e:
                analysis['errors'].append(f'Technical analysis error: {str(e)}')
                print(f"❌ Technical analysis error: {e}")
        
        # 3. Historical Pattern Analysis
        if self.historical_data:
            try:
                print("📚 Analyzing historical patterns...")
                patterns = self.historical_data.analyze_patterns(symbol)
                analysis['patterns'] = patterns
                analysis['systems_used'].append('Historical Analysis')
                print(f"✅ Found {len(patterns)} historical patterns")
            except Exception as e:
                analysis['errors'].append(f'Historical analysis error: {str(e)}')
                print(f"❌ Pattern analysis error: {e}")
        
        # 4. News Sentiment
        if self.news_analyzer:
            try:
                print("📰 Analyzing news sentiment...")
                sentiment = self.news_analyzer.get_market_sentiment(symbol)
                analysis['sentiment'] = sentiment
                analysis['systems_used'].append('News Sentiment')
                
                if sentiment['score'] > 0.5:
                    analysis['recommendations'].append({
                        'source': 'News',
                        'signal': 'BULLISH',
                        'confidence': int(sentiment['score'] * 100)
                    })
                elif sentiment['score'] < -0.5:
                    analysis['recommendations'].append({
                        'source': 'News',
                        'signal': 'BEARISH',
                        'confidence': int(abs(sentiment['score']) * 100)
                    })
                print(f"✅ News Sentiment: {sentiment['mood']}")
            except Exception as e:
                analysis['errors'].append(f'News analysis error: {str(e)}')
                print(f"❌ News analysis error: {e}")
        
        # 5. Smart AI Analysis
        if self.smart_trader:
            try:
                print("🤖 Getting Smart AI recommendation...")
                ai_signal = self.smart_trader.get_trading_signal(symbol)
                analysis['ai_signal'] = ai_signal
                analysis['systems_used'].append('Smart AI')
                
                if ai_signal.get('action') != 'WAIT':
                    analysis['recommendations'].append({
                        'source': 'Smart AI',
                        'signal': ai_signal['action'],
                        'confidence': ai_signal.get('confidence', 70)
                    })
                print(f"✅ AI Signal: {ai_signal.get('action', 'ANALYZING')}")
            except Exception as e:
                analysis['errors'].append(f'Smart AI error: {str(e)}')
                print(f"❌ Smart AI error: {e}")
        
        # 6. Strike Recommendations
        if self.strike_recommender and analysis.get('spot_price'):
            try:
                print("🎯 Getting exact strike recommendations...")
                strikes = self.strike_recommender.get_best_strikes(
                    symbol, 
                    analysis['spot_price']
                )
                analysis['recommended_strikes'] = strikes
                analysis['systems_used'].append('Strike Recommender')
                print(f"✅ Recommended {len(strikes)} strikes")
            except Exception as e:
                analysis['errors'].append(f'Strike recommendation error: {str(e)}')
                print(f"❌ Strike recommendation error: {e}")
        
        # Summary
        print(f"\n📊 Analysis Summary:")
        print(f"   Systems Used: {', '.join(analysis['systems_used'])}")
        print(f"   Recommendations: {len(analysis['recommendations'])}")
        print(f"   Errors: {len(analysis['errors'])}")
        
        return analysis
    
    def calculate_master_score(self, recommendations: List[Dict]) -> tuple:
        """Calculate overall score from all recommendations with error handling"""
        try:
            if not recommendations:
                return 0, 'NEUTRAL'
            
            total_score = 0
            total_weight = 0
            
            # Weight for each source
            weights = {
                'Technical': 0.25,
                'News': 0.20,
                'Smart AI': 0.30,
                'Historical': 0.25
            }
            
            for rec in recommendations:
                source = rec['source']
                signal = rec['signal']
                confidence = rec['confidence']
                weight = weights.get(source, 0.20)
                
                # Convert signal to score
                if signal in ['BUY', 'BULLISH', 'LONG']:
                    score = 1
                elif signal in ['SELL', 'BEARISH', 'SHORT']:
                    score = -1
                else:
                    score = 0
                
                total_score += score * confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0
            
            # Determine action
            if final_score > 30:
                action = 'STRONG BUY'
            elif final_score > 15:
                action = 'BUY'
            elif final_score < -30:
                action = 'STRONG SELL'
            elif final_score < -15:
                action = 'SELL'
            else:
                action = 'NEUTRAL'
            
            return final_score, action
            
        except Exception as e:
            print(f"❌ Error calculating master score: {e}")
            return 0, 'ERROR'
    
    def execute_master_trade(self, analysis: Dict) -> Optional[Dict]:
        """Execute trade based on master analysis with error handling"""
        try:
            master_score, action = self.calculate_master_score(analysis['recommendations'])
            
            if action == 'NEUTRAL' or action == 'ERROR':
                print("🔸 No clear trading signal - Staying neutral")
                return None
            
            print(f"\n🎯 MASTER DECISION: {action} (Score: {master_score:.2f})")
            
            # Get best strike
            if 'recommended_strikes' in analysis and analysis['recommended_strikes']:
                best_strike = analysis['recommended_strikes'][0]
                
                # Calculate position size using centralized lot size
                symbol = analysis['symbol']
                lot_size = get_lot_size(symbol)
                
                trade_params = {
                    'symbol': symbol,
                    'strike': best_strike['strike'],
                    'option_type': 'CE' if 'BUY' in action else 'PE',
                    'action': 'BUY',
                    'quantity': 1,  # Number of lots
                    'lot_size': lot_size,  # Use centralized lot size
                    'entry_price': best_strike.get('price', 100),
                    'confidence': abs(master_score),
                    'strategy': 'Master AI'
                }
                
                # Add to portfolio
                if self.portfolio_manager:
                    try:
                        self.portfolio_manager.add_trade(trade_params)
                        self.trades_today += 1
                        
                        print(f"✅ Trade executed: {symbol} {best_strike['strike']} {trade_params['option_type']}")
                        print(f"📊 Quantity: {trade_params['quantity']} lot ({lot_size} shares)")
                        
                        # Send alert using shared utility
                        alert_msg = f"""
🎯 <b>MASTER TRADE EXECUTED</b>

📊 {symbol} {best_strike['strike']} {trade_params['option_type']}
🔢 Quantity: {trade_params['quantity']} lot ({lot_size} shares)
💰 Entry: ₹{trade_params['entry_price']}
💪 Confidence: {abs(master_score):.1f}%
🎪 Strategy: Master AI

📈 Master Score: {master_score:.2f}
🎯 Action: {action}
                        """
                        self.send_master_alert(alert_msg)
                        
                        return trade_params
                    except Exception as e:
                        print(f"❌ Error adding trade to portfolio: {e}")
                        return None
                else:
                    print("⚠️ Portfolio manager not available")
                    return None
            else:
                print("⚠️ No strike recommendations available")
                return None
            
        except Exception as e:
            print(f"❌ Error executing master trade: {e}")
            return None
    
    def run_enhanced_trading_cycle(self) -> bool:
        """Run one complete enhanced trading cycle with error handling"""
        print(f"\n{'='*70}")
        print(f"🔄 ENHANCED TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        try:
            cycle_start = datetime.now()
            success_count = 0
            
            # Analyze both NIFTY and BANKNIFTY
            for symbol in ['NIFTY', 'BANKNIFTY']:
                try:
                    print(f"\n📊 Analyzing {symbol}...")
                    
                    # Get comprehensive analysis
                    analysis = self.get_comprehensive_market_analysis(symbol)
                    
                    # Display recommendations
                    print(f"\n📋 Recommendations for {symbol}:")
                    for rec in analysis['recommendations']:
                        print(f"  • {rec['source']}: {rec['signal']} (Confidence: {rec['confidence']}%)")
                    
                    # Display errors if any
                    if analysis['errors']:
                        print(f"⚠️ Analysis errors: {analysis['errors']}")
                    
                    # Calculate master score
                    master_score, action = self.calculate_master_score(analysis['recommendations'])
                    print(f"\n🎯 Master Score: {master_score:.2f} - Action: {action}")
                    
                    # Execute trade if conditions are met
                    if abs(master_score) > 20 and self.trades_today < self.max_trades_per_day:
                        trade_result = self.execute_master_trade(analysis)
                        if trade_result:
                            success_count += 1
                    
                    # Small delay between symbols
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"❌ Error analyzing {symbol}: {e}")
                    continue
            
            # Portfolio update
            if self.portfolio_manager:
                try:
                    print("\n💼 Portfolio Update:")
                    summary = self.portfolio_manager.get_portfolio_summary()
                    print(f"  • Open Positions: {summary['open_positions']}")
                    print(f"  • Today's P&L: ₹{summary['daily_pnl']:,.2f}")
                    print(f"  • Total P&L: ₹{summary['total_pnl']:,.2f}")
                except Exception as e:
                    print(f"❌ Error getting portfolio summary: {e}")
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            print(f"\n⏱️ Cycle completed in {cycle_duration:.1f} seconds")
            print(f"✅ Successful operations: {success_count}")
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            self.send_master_alert(f"❌ Trading cycle error: {str(e)[:100]}")
            return False
    
    def run_enhanced_session(self, duration_minutes: int = 30) -> None:
        """Run enhanced trading session with comprehensive error handling"""
        print(f"\n🚀 STARTING ENHANCED MASTER TRADING SESSION")
        print(f"⏰ Duration: {duration_minutes} minutes")
        print("="*70)
        
        # Initialize all systems
        if not self.initialize_all_systems():
            print("❌ Failed to initialize sufficient systems. Aborting session.")
            return
        
        try:
            self.session_active = True
            self.session_start_time = datetime.now()
            session_end_time = self.session_start_time + timedelta(minutes=duration_minutes)
            
            # Send session start alert
            self.send_master_alert(f"""
🚀 <b>ENHANCED TRADING SESSION STARTED</b>

⏰ Duration: {duration_minutes} minutes
🎯 Max Trades: {self.max_trades_per_day}
📊 Systems: Active systems initialized

💪 Let's make profitable trades!
            """)
            
            # Run trading cycles
            cycle_count = 0
            successful_cycles = 0
            
            while datetime.now() < session_end_time and self.session_active:
                cycle_count += 1
                print(f"\n🔄 Cycle {cycle_count}")
                
                # Check if market is open
                current_time = datetime.now().time()
                market_open = datetime.strptime(MARKET_HOURS['MARKET_OPEN'], '%H:%M').time()
                market_close = datetime.strptime(MARKET_HOURS['MARKET_CLOSE'], '%H:%M').time()
                
                if current_time < market_open or current_time > market_close:
                    print("🔸 Market is closed. Waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Run trading cycle
                if self.run_enhanced_trading_cycle():
                    successful_cycles += 1
                
                # Check trade limit
                if self.trades_today >= self.max_trades_per_day:
                    print(f"\n⚠️ Daily trade limit reached ({self.max_trades_per_day})")
                    break
                
                # Wait between cycles
                if datetime.now() < session_end_time:
                    wait_time = 120  # 2 minutes
                    print(f"\n⏳ Waiting {wait_time} seconds for next cycle...")
                    time.sleep(wait_time)
            
            # Session summary
            self.session_active = False
            session_duration = datetime.now() - self.session_start_time
            
            summary_msg = f"""
🏁 <b>ENHANCED SESSION COMPLETED</b>

⏱️ Duration: {session_duration}
🔄 Total Cycles: {cycle_count}
✅ Successful Cycles: {successful_cycles}
📊 Trades Executed: {self.trades_today}

💰 Session P&L: ₹{self.session_pnl:,.2f}
📈 Best Trade: ₹{self.best_trade_today:,.2f}
📉 Worst Trade: ₹{self.worst_trade_today:,.2f}

✅ Session completed successfully!
            """
            
            print(summary_msg.replace('<b>', '').replace('</b>', ''))
            self.send_master_alert(summary_msg)
            
        except Exception as e:
            print(f"❌ Session error: {e}")
            self.send_master_alert(f"❌ Session ended with error: {str(e)}")
        finally:
            self.session_active = False
    
    def stop_session(self) -> None:
        """Stop the trading session"""
        print("\n🛑 Stopping trading session...")
        self.session_active = False
        
        # Close all positions if needed
        if self.portfolio_manager:
            try:
                open_positions = self.portfolio_manager.get_open_positions()
                if open_positions:
                    print(f"⚠️ Warning: {len(open_positions)} positions still open")
            except Exception as e:
                print(f"❌ Error checking open positions: {e}")
    
    def get_system_status(self) -> Dict:
        """Get status of all systems"""
        try:
            status = {
                'master_trader': 'ACTIVE' if self.session_active else 'IDLE',
                'market_data': 'ACTIVE' if self.market_data_engine else 'INACTIVE',
                'smart_ai': 'ACTIVE' if self.smart_trader else 'INACTIVE',
                'strike_recommender': 'ACTIVE' if self.strike_recommender else 'INACTIVE',
                'portfolio_manager': 'ACTIVE' if self.portfolio_manager else 'INACTIVE',
                'technical_indicators': 'ACTIVE' if self.technical_indicators else 'INACTIVE',
                'historical_data': 'ACTIVE' if self.historical_data else 'INACTIVE',
                'news_analyzer': 'ACTIVE' if self.news_analyzer else 'INACTIVE',
                'trades_today': self.trades_today,
                'session_pnl': self.session_pnl,
                'current_balance': self.current_balance,
                'available_modules': {
                    'exact_recommender': EXACT_RECOMMENDER_AVAILABLE,
                    'portfolio_manager': PORTFOLIO_MANAGER_AVAILABLE,
                    'market_data': MARKET_DATA_AVAILABLE,
                    'technical_indicators': TECHNICAL_INDICATORS_AVAILABLE,
                    'historical_data': HISTORICAL_DATA_AVAILABLE,
                    'news_sentiment': NEWS_SENTIMENT_AVAILABLE,
                    'unified_trading': UNIFIED_TRADING_AVAILABLE
                }
            }
            return status
        except Exception as e:
            return {'error': str(e), 'status': 'ERROR'}

def main():
    """Main function to run Enhanced Master Trader"""
    print("🌟 TradeMind_AI ENHANCED MASTER TRADER")
    print("🚀 Now with Technical Indicators, Historical Data & News!")
    print("🎯 The Most Advanced Trading System!")
    print("🛡️ Enhanced Error Handling & Shared Utilities!")
    print("="*70)
    
    try:
        # Create master trader instance
        master = EnhancedMasterTrader()
        
        while True:
            print("\n📊 MASTER TRADER MENU:")
            print("1. Run Quick Analysis (Both symbols)")
            print("2. Start 30-min Trading Session")
            print("3. Start 2-hour Trading Session")
            print("4. Get System Status")
            print("5. View Portfolio")
            print("6. Manual Trade Entry")
            print("0. Exit")
            
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                # Quick analysis
                if master.initialize_all_systems():
                    for symbol in ['NIFTY', 'BANKNIFTY']:
                        analysis = master.get_comprehensive_market_analysis(symbol)
                        master_score, action = master.calculate_master_score(analysis['recommendations'])
                        print(f"\n{symbol}: {action} (Score: {master_score:.2f})")
                else:
                    print("❌ Failed to initialize systems for analysis")
            
            elif choice == '2':
                # 30-minute session
                master.run_enhanced_session(30)
            
            elif choice == '3':
                # 2-hour session
                master.run_enhanced_session(120)
            
            elif choice == '4':
                # System status
                status = master.get_system_status()
                print("\n📊 SYSTEM STATUS:")
                for system, state in status.items():
                    if isinstance(state, dict):
                        print(f"  • {system}:")
                        for k, v in state.items():
                            print(f"    - {k}: {v}")
                    else:
                        print(f"  • {system}: {state}")
            
            elif choice == '5':
                # View portfolio
                if master.portfolio_manager:
                    try:
                        master.portfolio_manager.send_daily_portfolio_report()
                        print("✅ Portfolio report sent!")
                    except Exception as e:
                        print(f"❌ Portfolio error: {e}")
                else:
                    print("❌ Portfolio manager not initialized")
            
            elif choice == '6':
                # Manual trade
                print("\n📝 Manual Trade Entry")
                try:
                    symbol = input("Symbol (NIFTY/BANKNIFTY): ").upper()
                    if symbol in ['NIFTY', 'BANKNIFTY']:
                        strike = float(input("Strike Price: "))
                        option_type = input("Option Type (CE/PE): ").upper()
                        quantity = int(input("Quantity (lots): "))
                        
                        lot_size = get_lot_size(symbol)
                        
                        trade_params = {
                            'symbol': symbol,
                            'strike': strike,
                            'option_type': option_type,
                            'action': 'BUY',
                            'quantity': quantity,
                            'lot_size': lot_size,
                            'entry_price': float(input("Entry Price: ")),
                            'confidence': 75,
                            'strategy': 'Manual'
                        }
                        
                        if master.portfolio_manager:
                            master.portfolio_manager.add_trade(trade_params)
                            print(f"✅ Trade added: {quantity} lot(s) of {symbol} {strike} {option_type}")
                            print(f"📊 Total quantity: {quantity * lot_size} shares")
                        else:
                            print("❌ Portfolio manager not available")
                    else:
                        print("❌ Invalid symbol")
                except ValueError:
                    print("❌ Invalid input")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif choice == '0':
                print("\n👋 Goodbye! Happy Trading!")
                master.stop_session()
                break
            
            else:
                print("❌ Invalid choice")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")

if __name__ == "__main__":
    main()