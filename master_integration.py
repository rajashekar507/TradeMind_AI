"""
Fixed Master Integration Controller for TradeMind AI
Working version with proper imports
"""

import os
import sys
import json
import time
from datetime import datetime
import logging

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MasterIntegrationController:
    """The brain that orchestrates all TradeMind AI modules"""
    
    def __init__(self):
        """Initialize all components"""
        self.logger = logging.getLogger('MasterController')
        self.logger.info("🧠 Initializing Master Integration Controller...")
        
        # System state
        self.is_running = False
        self.trading_enabled = False
        self.paper_trading = True  # Start with paper trading
        
        # Initialize all modules
        self.modules_loaded = self.initialize_modules()
        
        # Signal weights for master decision
        self.signal_weights = {
            'multi_timeframe': 0.25,
            'technical_indicators': 0.20,
            'greeks': 0.15,
            'news_sentiment': 0.15,
            'oi_analysis': 0.15,
            'exact_recommender': 0.10
        }
        
        self.logger.info("✅ Master Controller Initialized!")
    
    def initialize_modules(self):
        """Initialize all trading modules with error handling"""
        modules_status = {}
        
        # 1. Multi-timeframe Analyzer
        try:
            # Try different import methods and class names
            mtf_loaded = False
            
            # Method 1: Try from src.analysis
            try:
                from src.analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
                self.mtf_analyzer = MultiTimeframeAnalyzer()
                mtf_loaded = True
            except:
                pass
            
            # Method 2: Try different class name
            if not mtf_loaded:
                try:
                    from src.analysis.multi_timeframe_analyzer import EnhancedMultiTimeframeAnalyzer
                    self.mtf_analyzer = EnhancedMultiTimeframeAnalyzer()
                    mtf_loaded = True
                except:
                    pass
            
            # Method 3: Try MultiTimeFrameAnalyzer (with capital F)
            if not mtf_loaded:
                try:
                    from src.analysis.multi_timeframe_analyzer import MultiTimeFrameAnalyzer
                    self.mtf_analyzer = MultiTimeFrameAnalyzer()
                    mtf_loaded = True
                except:
                    pass
            
            if mtf_loaded:
                modules_status['multi_timeframe'] = True
                self.logger.info("✅ Multi-timeframe analyzer loaded successfully")
            else:
                self.mtf_analyzer = None
                modules_status['multi_timeframe'] = False
                self.logger.warning("⚠️ Multi-timeframe analyzer not loaded")
                
        except Exception as e:
            self.logger.error(f"MTF error: {e}")
            self.mtf_analyzer = None
            modules_status['multi_timeframe'] = False
            
        # 2. Exact Recommender
        try:
            from src.analysis.exact_recommender import ExactStrikeRecommender
            self.exact_recommender = ExactStrikeRecommender()
            modules_status['exact_recommender'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Exact recommender not loaded: {e}")
            self.exact_recommender = None
            modules_status['exact_recommender'] = False
            
        # 3. Greeks Calculator
        try:
            from src.analysis.greeks_calculator import GreeksCalculator
            self.greeks_calc = GreeksCalculator()
            modules_status['greeks'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Greeks calculator not loaded: {e}")
            self.greeks_calc = None
            modules_status['greeks'] = False
            
        # 4. Technical Indicators
        try:
            from src.analysis.technical_indicators import TechnicalIndicators
            self.tech_indicators = TechnicalIndicators()
            modules_status['technical'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicators not loaded: {e}")
            self.tech_indicators = None
            modules_status['technical'] = False
            
        # 5. News Analyzer
        try:
            from src.analysis.news_sentiment import NewsSentimentAnalyzer
            self.news_analyzer = NewsSentimentAnalyzer()
            modules_status['news'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ News analyzer not loaded: {e}")
            self.news_analyzer = None
            modules_status['news'] = False
            
        # 6. OI Tracker
        try:
            from src.analysis.oi_tracker import OIChangeTracker
            self.oi_tracker = OIChangeTracker()
            modules_status['oi'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ OI tracker not loaded: {e}")
            self.oi_tracker = None
            modules_status['oi'] = False
            
        # 7. Market Data
        try:
            from src.data.market_data import MarketDataEngine
            self.market_data = MarketDataEngine()
            modules_status['market_data'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Market data engine not loaded: {e}")
            self.market_data = None
            modules_status['market_data'] = False
            
        # 8. Portfolio Manager
        try:
            from src.portfolio.portfolio_manager import PortfolioManager
            self.portfolio_manager = PortfolioManager()
            modules_status['portfolio'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Portfolio manager not loaded: {e}")
            self.portfolio_manager = None
            modules_status['portfolio'] = False
            
        # 9. Excel Reporter
        try:
            from src.utils.excel_reporter import ExcelReporter
            self.excel_reporter = ExcelReporter()
            modules_status['excel'] = True
        except Exception as e:
            self.logger.warning(f"⚠️ Excel reporter not loaded: {e}")
            self.excel_reporter = None
            modules_status['excel'] = False
        
        # Display module status
        self.logger.info("\n📊 Module Status:")
        for module, status in modules_status.items():
            self.logger.info(f"   {module}: {'✅' if status else '❌'}")
        
        return modules_status
    
    def analyze_market(self, symbol='NIFTY'):
        """Run complete market analysis using available modules"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔍 Running Analysis for {symbol}")
        self.logger.info(f"{'='*60}")
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signals': {},
            'master_signal': None,
            'confidence': 0,
            'recommendations': []
        }
        
        # 1. Multi-Timeframe Analysis
        if self.mtf_analyzer:
            try:
                self.logger.info("📊 Running Multi-Timeframe Analysis...")
                # Check which method exists
                if hasattr(self.mtf_analyzer, 'analyze'):
                    mtf_result = self.mtf_analyzer.analyze(symbol)
                else:
                    mtf_result = self.mtf_analyzer.analyze_all_timeframes(symbol)
                    
                if isinstance(mtf_result, dict):
                    analysis_results['signals']['multi_timeframe'] = {
                        'signal': mtf_result.get('overall_signal', 'NEUTRAL'),
                        'confidence': mtf_result.get('overall_confidence', 50),
                        'trend': mtf_result.get('overall_trend', 'SIDEWAYS')
                    }
            except Exception as e:
                self.logger.error(f"MTF Analysis failed: {e}")
        
        # 2. Technical Indicators
        if self.tech_indicators:
            try:
                self.logger.info("📈 Running Technical Indicators...")
                tech_result = self.tech_indicators.get_indicator_signals(symbol)
                analysis_results['signals']['technical'] = {
                    'signal': tech_result['signal'],
                    'confidence': tech_result['confidence'],
                    'rsi': tech_result['indicators'].get('RSI', 50)
                }
            except Exception as e:
                self.logger.error(f"Technical Analysis failed: {e}")
        
        # 3. News Sentiment
        if self.news_analyzer:
            try:
                self.logger.info("📰 Running News Sentiment Analysis...")
                sentiment_report = self.news_analyzer.generate_sentiment_report(symbol)
                analysis_results['signals']['sentiment'] = {
                    'market_mood': sentiment_report['market_sentiment'],
                    'score': sentiment_report['sentiment_score'],
                    'confidence': sentiment_report['confidence']
                }
            except Exception as e:
                self.logger.error(f"Sentiment Analysis failed: {e}")
        
        # 4. If we have market data, get current price
        if self.market_data:
            try:
                self.logger.info("📊 Fetching Market Data...")
                option_data = self.market_data.get_option_chain(
                    self.market_data.NIFTY_ID if symbol == 'NIFTY' else self.market_data.BANKNIFTY_ID,
                    symbol
                )
                if option_data:
                    analysis = self.market_data.analyze_option_data(option_data)
                    if analysis:
                        analysis_results['current_price'] = analysis['underlying_price']
                        analysis_results['atm_strike'] = analysis['atm_strike']
            except Exception as e:
                self.logger.error(f"Market data fetch failed: {e}")
        
        # Calculate Master Signal
        master_signal = self.calculate_master_signal(analysis_results['signals'])
        analysis_results['master_signal'] = master_signal
        
        # Generate recommendations
        if master_signal['action'] != 'WAIT':
            recommendations = self.generate_trade_recommendations(symbol, master_signal, analysis_results)
            analysis_results['recommendations'] = recommendations
        
        return analysis_results
    
    def calculate_master_signal(self, signals):
        """Calculate master signal from available signals"""
        if not signals:
            return {
                'score': 50,
                'action': 'WAIT',
                'option_type': None,
                'confidence': 0
            }
        
        total_score = 0
        total_weight = 0
        
        # Map signals to scores
        signal_scores = {
            'STRONG BUY': 100, 'BUY': 75, 'NEUTRAL': 50, 'SELL': 25, 'STRONG SELL': 0,
            'EXTREMELY BULLISH': 100, 'STRONGLY BULLISH': 90, 'BULLISH': 75,
            'BEARISH': 25, 'STRONGLY BEARISH': 10, 'EXTREMELY BEARISH': 0
        }
        
        # Calculate weighted score from available signals
        for signal_type, signal_data in signals.items():
            weight = 1.0 / len(signals)  # Equal weight for available signals
            
            # Get signal score
            signal_text = signal_data.get('signal') or signal_data.get('market_mood', 'NEUTRAL')
            score = signal_scores.get(signal_text, 50)
            
            # Apply confidence if available
            if 'confidence' in signal_data:
                score = score * (signal_data['confidence'] / 100)
            
            total_score += score * weight
            total_weight += weight
        
        # Calculate final score
        final_score = total_score / total_weight if total_weight > 0 else 50
        
        # Determine action
        if final_score >= 75:
            action = 'STRONG BUY'
            option_type = 'CE'
        elif final_score >= 60:
            action = 'BUY'
            option_type = 'CE'
        elif final_score <= 25:
            action = 'STRONG SELL'
            option_type = 'PE'
        elif final_score <= 40:
            action = 'SELL'
            option_type = 'PE'
        else:
            action = 'WAIT'
            option_type = None
        
        return {
            'score': final_score,
            'action': action,
            'option_type': option_type,
            'confidence': min(95, max(final_score, 100 - final_score))
        }
    
    def generate_trade_recommendations(self, symbol, master_signal, analysis_results):
        """Generate specific trade recommendations"""
        recommendations = []
        
        # Use current market data if available
        current_price = analysis_results.get('current_price', 25500 if symbol == 'NIFTY' else 57000)
        atm_strike = analysis_results.get('atm_strike', round(current_price / 50) * 50)
        
        # Simple recommendation
        recommendation = {
            'symbol': symbol,
            'strike': atm_strike,
            'option_type': master_signal['option_type'],
            'action': master_signal['action'],
            'confidence': master_signal['confidence'],
            'entry_price': 100,  # Placeholder - would get from exact recommender
            'target_price': 130,
            'stop_loss': 80
        }
        recommendations.append(recommendation)
        
        return recommendations
    
    def display_analysis_results(self, analysis):
        """Display analysis results"""
        print(f"\n{'='*70}")
        print(f"🎯 MASTER ANALYSIS - {analysis['symbol']}")
        print(f"{'='*70}")
        
        # Current price if available
        if 'current_price' in analysis:
            print(f"\n💰 Current Price: ₹{analysis['current_price']}")
        
        # Individual signals
        print("\n📊 Individual Signals:")
        for module, signal in analysis['signals'].items():
            print(f"   {module}: {signal}")
        
        # Master decision
        master = analysis['master_signal']
        print(f"\n🧠 MASTER DECISION:")
        print(f"   Action: {master['action']}")
        print(f"   Confidence: {master['confidence']:.1f}%")
        print(f"   Score: {master['score']:.1f}")
        
        # Recommendations
        if analysis['recommendations']:
            print(f"\n💡 TRADE RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"   {rec['symbol']} {rec['strike']} {rec['option_type']}")
                print(f"   Entry: ₹{rec['entry_price']} | Target: ₹{rec['target_price']}")
        
        print(f"{'='*70}")
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            # Analyze both NIFTY and BANKNIFTY
            for symbol in ['NIFTY', 'BANKNIFTY']:
                analysis = self.analyze_market(symbol)
                
                # Display results
                self.display_analysis_results(analysis)
                
                # Execute trades if confident
                if analysis['master_signal']['confidence'] > 70 and self.trading_enabled:
                    self.execute_trades(analysis['recommendations'])
                
                time.sleep(2)  # Delay between symbols
            
            # Generate reports if available
            if self.excel_reporter:
                try:
                    self.excel_reporter.generate_daily_report()
                    print("\n✅ Excel report generated")
                except:
                    pass
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
    
    def execute_trades(self, recommendations):
        """Execute trades based on recommendations"""
        if not self.trading_enabled:
            self.logger.info("⚠️ Trading disabled - Logging trades only")
            return
        
        for rec in recommendations:
            try:
                if self.paper_trading and self.portfolio_manager:
                    self.logger.info(f"📝 PAPER TRADE: {rec}")
                    self.portfolio_manager.simulate_paper_trade(rec)
                else:
                    self.logger.info(f"💰 LIVE TRADE: {rec}")
                    # Live trading execution would go here
                    
            except Exception as e:
                self.logger.error(f"Trade execution failed: {e}")

def main():
    """Main entry point"""
    print("🧠 TradeMind AI - Master Integration Controller")
    print("="*70)
    
    controller = MasterIntegrationController()
    
    # Menu
    while True:
        print("\n📊 MASTER CONTROL MENU:")
        print("1. Run Single Analysis Cycle")
        print("2. Start Automated Paper Trading")
        print("3. Generate Reports")
        print("4. View Module Status")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == '1':
            controller.run_trading_cycle()
        elif choice == '2':
            controller.trading_enabled = True
            print("\n📊 Starting paper trading...")
            print("Press Ctrl+C to stop")
            try:
                while True:
                    controller.run_trading_cycle()
                    print("\n⏳ Waiting 5 minutes for next cycle...")
                    time.sleep(300)  # 5 minutes
            except KeyboardInterrupt:
                print("\n⛔ Paper trading stopped")
        elif choice == '3':
            if controller.excel_reporter:
                controller.excel_reporter.generate_daily_report()
                print("✅ Reports generated")
            else:
                print("❌ Excel reporter not available")
        elif choice == '4':
            print("\n📊 Module Status:")
            for module, status in controller.modules_loaded.items():
                print(f"   {module}: {'✅ Loaded' if status else '❌ Not loaded'}")
        elif choice == '0':
            break

if __name__ == "__main__":
    main()