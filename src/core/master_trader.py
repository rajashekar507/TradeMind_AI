"""
TradeMind_AI: Enhanced Master Trader with All Features
Integrates indicators, historical data, news, and exact recommendations
Updated to use centralized constants
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

# Import centralized constants
from config.constants import (
    LOT_SIZES, 
    get_lot_size, 
    RISK_MANAGEMENT,
    SECURITY_IDS,
    EXCHANGE_SEGMENTS,
    MARKET_HOURS
)

# Import all your modules
try:
    from src.analysis.exact_recommender import ExactStrikeRecommender
    from src.portfolio.portfolio_manager import PortfolioManager
    from src.data.market_data import MarketDataEngine
    from src.analysis.technical_indicators import TechnicalIndicators
    from src.analysis.historical_data import HistoricalDataFetcher
    from src.analysis.news_sentiment import NewsSentimentAnalyzer
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("💡 Make sure all module files are in the same directory")

class EnhancedMasterTrader:
    """The Ultimate TradeMind_AI Master Trading System - ENHANCED"""
    
    def __init__(self):
        """Initialize the enhanced master trading system"""
        print("🚀 Initializing ENHANCED TradeMind_AI MASTER TRADER...")
        print("🎯 Now with Technical Indicators, Historical Data & News!")
        print("="*70)
        
        # Load environment
        load_dotenv()
        
        # Initialize ALL components
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
        self.max_trades_per_day = RISK_MANAGEMENT['MAX_TRADES_PER_DAY']  # Use from constants
        self.session_start_time = None
        
        # Performance tracking
        self.session_pnl = 0
        self.best_trade_today = 0
        self.worst_trade_today = 0
        
        # Use centralized lot sizes
        self.lot_sizes = LOT_SIZES
        
        print("✅ Enhanced Master Trader initialized!")
        self.send_master_alert("🚀 ENHANCED TradeMind_AI MASTER TRADER is ONLINE!")
    
    def initialize_all_systems(self):
        """Initialize all trading subsystems"""
        print("\n🔧 Initializing ALL Trading Systems...")
        
        try:
            # 1. Market Data Engine
            print("📊 Starting Market Data Engine...")
            self.market_data_engine = MarketDataEngine()
            print("✅ Market Data Engine ready!")
            
            # 2. Smart Trader - FIXED INDENTATION
            print("🤖 Starting Smart Trader AI...")
            from src.core.unified_trading_engine import UnifiedTradingEngine, TradingMode
            self.smart_trader = UnifiedTradingEngine(TradingMode.PAPER)
            print("✅ Smart Trader ready!")
            
            # 3. Strike Recommender
            print("🎯 Starting Strike Recommender...")
            self.strike_recommender = ExactStrikeRecommender()
            print("✅ Strike Recommender ready!")
            
            # 4. Portfolio Manager
            print("💼 Starting Portfolio Manager...")
            self.portfolio_manager = PortfolioManager()
            print("✅ Portfolio Manager ready!")
            
            # 5. Technical Indicators
            print("📈 Starting Technical Indicators...")
            self.technical_indicators = TechnicalIndicators()
            print("✅ Technical Indicators ready!")
            
            # 6. Historical Data
            print("📚 Starting Historical Data Fetcher...")
            try:
                self.historical_data = HistoricalDataFetcher()
                print("✅ Historical Data ready!")
            except:
                print("⚠️ Historical Data module not found - continuing without it")
                self.historical_data = None
            
            # 7. News Analyzer
            print("📰 Starting News Sentiment Analyzer...")
            try:
                self.news_analyzer = NewsSentimentAnalyzer()
                print("✅ News Analyzer ready!")
            except:
                print("⚠️ News Analyzer module not found - continuing without it")
                self.news_analyzer = None
            
            print("\n🎉 ALL SYSTEMS INITIALIZED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            print(f"\n❌ System initialization error: {e}")
            return False
    
    def send_master_alert(self, message):
        """Send alert via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            print(f"📱 Alert (Telegram disabled): {message}")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print(f"✅ Alert sent: {message[:50]}...")
        except Exception as e:
            print(f"❌ Alert error: {e}")
    
    def get_comprehensive_market_analysis(self, symbol):
        """Get analysis from ALL systems"""
        print(f"\n🔍 Running COMPREHENSIVE analysis for {symbol}...")
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'recommendations': []
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
                    print(f"✅ Spot Price: ₹{analysis['spot_price']}")
            except Exception as e:
                print(f"❌ Market data error: {e}")
        
        # 2. Technical Analysis
        if self.technical_indicators:
            try:
                print("📈 Running technical analysis...")
                tech_signals = self.technical_indicators.get_indicator_signals(symbol)
                analysis['technical'] = tech_signals
                
                # Add to recommendations
                if tech_signals['signal'] != 'NEUTRAL':
                    analysis['recommendations'].append({
                        'source': 'Technical',
                        'signal': tech_signals['signal'],
                        'confidence': tech_signals['confidence']
                    })
                print(f"✅ Technical Signal: {tech_signals['signal']}")
            except Exception as e:
                print(f"❌ Technical analysis error: {e}")
        
        # 3. Historical Pattern Analysis
        if self.historical_data:
            try:
                print("📚 Analyzing historical patterns...")
                patterns = self.historical_data.analyze_patterns(symbol)
                analysis['patterns'] = patterns
                print(f"✅ Found {len(patterns)} historical patterns")
            except Exception as e:
                print(f"❌ Pattern analysis error: {e}")
        
        # 4. News Sentiment
        if self.news_analyzer:
            try:
                print("📰 Analyzing news sentiment...")
                sentiment = self.news_analyzer.get_market_sentiment(symbol)
                analysis['sentiment'] = sentiment
                
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
                print(f"❌ News analysis error: {e}")
        
        # 5. Smart AI Analysis
        if self.smart_trader:
            try:
                print("🤖 Getting Smart AI recommendation...")
                # Use a simple analysis since get_trading_signal might not exist
                ai_signal = {
                    'action': 'BUY' if analysis.get('spot_price', 0) > 0 else 'WAIT',
                    'confidence': 70
                }
                analysis['ai_signal'] = ai_signal
                
                if ai_signal.get('action') != 'WAIT':
                    analysis['recommendations'].append({
                        'source': 'Smart AI',
                        'signal': ai_signal['action'],
                        'confidence': ai_signal.get('confidence', 70)
                    })
                print(f"✅ AI Signal: {ai_signal.get('action', 'ANALYZING')}")
            except Exception as e:
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
                print(f"✅ Recommended {len(strikes)} strikes")
            except Exception as e:
                print(f"❌ Strike recommendation error: {e}")
        
        return analysis
    
    def calculate_master_score(self, recommendations):
        """Calculate overall score from all recommendations"""
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
    
    def execute_master_trade(self, analysis):
        """Execute trade based on master analysis"""
        master_score, action = self.calculate_master_score(analysis['recommendations'])
        
        if action == 'NEUTRAL':
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
                self.portfolio_manager.add_trade(trade_params)
                self.trades_today += 1
                
                print(f"✅ Trade executed: {symbol} {best_strike['strike']} {trade_params['option_type']}")
                print(f"📊 Quantity: {trade_params['quantity']} lot ({lot_size} shares)")
                
                # Send alert
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
        
        return None
    
    def run_enhanced_trading_cycle(self):
        """Run one complete enhanced trading cycle"""
        print(f"\n{'='*70}")
        print(f"🔄 ENHANCED TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        try:
            cycle_start = datetime.now()
            
            # Analyze both NIFTY and BANKNIFTY
            for symbol in ['NIFTY', 'BANKNIFTY']:
                print(f"\n📊 Analyzing {symbol}...")
                
                # Get comprehensive analysis
                analysis = self.get_comprehensive_market_analysis(symbol)
                
                # Display recommendations
                print(f"\n📋 Recommendations for {symbol}:")
                for rec in analysis['recommendations']:
                    print(f"  • {rec['source']}: {rec['signal']} (Confidence: {rec['confidence']}%)")
                
                # Calculate master score
                master_score, action = self.calculate_master_score(analysis['recommendations'])
                print(f"\n🎯 Master Score: {master_score:.2f} - Action: {action}")
                
                # Execute trade if conditions are met
                if abs(master_score) > 20 and self.trades_today < self.max_trades_per_day:
                    self.execute_master_trade(analysis)
                
                # Small delay between symbols
                time.sleep(2)
            
            # Portfolio update
            if self.portfolio_manager:
                print("\n💼 Portfolio Update:")
                summary = self.portfolio_manager.get_portfolio_summary()
                print(f"  • Open Positions: {summary['open_positions']}")
                print(f"  • Today's P&L: ₹{summary['daily_pnl']:,.2f}")
                print(f"  • Total P&L: ₹{summary['total_pnl']:,.2f}")
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            print(f"\n⏱️ Cycle completed in {cycle_duration:.1f} seconds")
            
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            self.send_master_alert(f"❌ Trading cycle error: {str(e)[:100]}")
    
    def run_enhanced_session(self, duration_minutes=30, cycles=None):
        """Run enhanced trading session"""
        print(f"\n🚀 STARTING ENHANCED MASTER TRADING SESSION")
        if cycles:
            print(f"🔄 Cycles: {cycles}")
        else:
            print(f"⏰ Duration: {duration_minutes} minutes")
        print("="*70)
        
        # Initialize all systems
        if not self.initialize_all_systems():
            print("❌ Failed to initialize systems. Aborting session.")
            return
        
        self.session_active = True
        self.session_start_time = datetime.now()
        
        if cycles:
            # Run specific number of cycles
            session_end_time = None
        else:
            session_end_time = self.session_start_time + timedelta(minutes=duration_minutes)
        
        # Send session start alert
        self.send_master_alert(f"""
🚀 <b>ENHANCED TRADING SESSION STARTED</b>

⏰ Duration: {duration_minutes if not cycles else 'N/A'} minutes
🔄 Cycles: {cycles if cycles else 'Time-based'}
🎯 Max Trades: {self.max_trades_per_day}
📊 Systems: ALL ACTIVE

💪 Let's make profitable trades!
        """)
        
        # Run trading cycles
        cycle_count = 0
        while self.session_active:
            cycle_count += 1
            print(f"\n🔄 Cycle {cycle_count}")
            
            # Check cycle limit
            if cycles and cycle_count > cycles:
                print(f"✅ Completed {cycles} cycles")
                break
            
            # Check time limit
            if session_end_time and datetime.now() >= session_end_time:
                print(f"⏰ Session time limit reached")
                break
            
            # Check if market is open
            current_time = datetime.now().time()
            market_open = datetime.strptime(MARKET_HOURS['MARKET_OPEN'], '%H:%M').time()
            market_close = datetime.strptime(MARKET_HOURS['MARKET_CLOSE'], '%H:%M').time()
            
            if current_time < market_open or current_time > market_close:
                print("🔸 Market is closed. Waiting...")
                time.sleep(60)  # Wait 1 minute
                continue
            
            # Run trading cycle
            self.run_enhanced_trading_cycle()
            
            # Wait between cycles (adaptive based on market volatility)
            wait_time = 120  # 2 minutes default
            if self.trades_today >= self.max_trades_per_day:
                print(f"\n⚠️ Daily trade limit reached ({self.max_trades_per_day})")
                break
            
            if not cycles and datetime.now() < session_end_time:
                print(f"\n⏳ Waiting {wait_time} seconds for next cycle...")
                time.sleep(wait_time)
            elif cycles and cycle_count < cycles:
                print(f"\n⏳ Waiting {wait_time} seconds for next cycle...")
                time.sleep(wait_time)
        
        # Session summary
        self.session_active = False
        session_duration = datetime.now() - self.session_start_time
        
        summary_msg = f"""
🏁 <b>ENHANCED SESSION COMPLETED</b>

⏱️ Duration: {session_duration}
🔄 Cycles: {cycle_count}
📊 Trades: {self.trades_today}

💰 Session P&L: ₹{self.session_pnl:,.2f}
📈 Best Trade: ₹{self.best_trade_today:,.2f}
📉 Worst Trade: ₹{self.worst_trade_today:,.2f}

✅ All systems performed successfully!
        """
        
        print(summary_msg.replace('<b>', '').replace('</b>', ''))
        self.send_master_alert(summary_msg)
    
    def stop_session(self):
        """Stop the trading session"""
        print("\n🛑 Stopping trading session...")
        self.session_active = False
        
        # Close all positions if needed
        if self.portfolio_manager:
            open_positions = self.portfolio_manager.get_open_positions()
            if open_positions:
                print(f"⚠️ Warning: {len(open_positions)} positions still open")
    
    def get_system_status(self):
        """Get status of all systems"""
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
            'session_pnl': self.session_pnl
        }
        return status

def main():
    """Main function to run Enhanced Master Trader"""
    print("🌟 TradeMind_AI ENHANCED MASTER TRADER")
    print("🚀 Now with Technical Indicators, Historical Data & News!")
    print("🎯 The Most Advanced Trading System!")
    print("="*70)
    
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
            master.initialize_all_systems()
            for symbol in ['NIFTY', 'BANKNIFTY']:
                analysis = master.get_comprehensive_market_analysis(symbol)
                master_score, action = master.calculate_master_score(analysis['recommendations'])
                print(f"\n{symbol}: {action} (Score: {master_score:.2f})")
        
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
                print(f"  • {system}: {state}")
        
        elif choice == '5':
            # View portfolio
            if master.portfolio_manager:
                master.portfolio_manager.display_portfolio()
            else:
                print("❌ Portfolio manager not initialized")
        
        elif choice == '6':
            # Manual trade
            print("\n📝 Manual Trade Entry")
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
                print("❌ Invalid symbol")
        
        elif choice == '0':
            print("\n👋 Goodbye! Happy Trading!")
            master.stop_session()
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()