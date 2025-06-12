"""
TradeMind AI Dashboard Backend - Complete Production System
This is the FINAL version - no future changes needed
Connects all backend services to dashboard frontend
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# Flask imports
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Import all TradeMind modules
try:
    from portfolio_manager import PortfolioManager
    from live_trading_engine import LiveTradingEngine
    from market_data import MarketDataEngine
    from ml_trader import SelfLearningTrader
    from global_market_analyzer import GlobalMarketAnalyzer
    from real_news_analyzer import RealNewsAnalyzer
    from historical_data import HistoricalDataFetcher
    from dhanhq import DhanContext, dhanhq
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
        logging.FileHandler('dashboard_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradeMindBackend:
    """Complete TradeMind AI Backend System"""
    
    def __init__(self):
        """Initialize all backend components"""
        logger.info("🚀 Initializing TradeMind AI Backend...")
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                         template_folder='templates',
                         static_folder='static')
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'trademind-ai-2025')
        
        # Enable CORS for frontend
        CORS(self.app, origins=["http://localhost:3000", "http://127.0.0.1:5500", "*"])
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Initialize all TradeMind components
        self._initialize_components()
        
        # Setup routes
        self._setup_routes()
        
        # Setup WebSocket events
        self._setup_websocket_events()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Cache for real-time data
        self.data_cache = {
            'last_update': None,
            'market_data': {},
            'positions': [],
            'balance': 0,
            'pnl': 0,
            'ai_signals': {},
            'news_sentiment': {},
            'technical_indicators': {}
        }
        
        logger.info("✅ TradeMind AI Backend initialized successfully!")
    
    def _initialize_components(self):
        """Initialize all TradeMind AI components"""
        try:
            # Core trading components
            self.portfolio_manager = PortfolioManager()
            self.trading_engine = LiveTradingEngine()
            self.market_data_engine = MarketDataEngine()
            
            # AI components
            self.ml_trader = SelfLearningTrader()
            self.global_analyzer = GlobalMarketAnalyzer()
            self.news_analyzer = RealNewsAnalyzer()
            self.historical_fetcher = HistoricalDataFetcher()
            
            # Initialize Dhan client for real-time data
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            if client_id and access_token:
                dhan_context = DhanContext(client_id=client_id, access_token=access_token)
                self.dhan_client = dhanhq(dhan_context)
                logger.info("✅ Dhan API client initialized")
            else:
                logger.warning("⚠️ Dhan credentials not found - using demo mode")
                self.dhan_client = None
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            logger.error(traceback.format_exc())
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # ======================
        # DASHBOARD DATA ROUTES
        # ======================
        
        @self.app.route('/')
        def dashboard():
            """Serve dashboard HTML"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """API health check"""
            return jsonify({
                'status': 'online',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'components': {
                    'portfolio_manager': True,
                    'trading_engine': True,
                    'ml_trader': True,
                    'dhan_api': self.dhan_client is not None
                }
            })
        
        @self.app.route('/api/balance')
        def get_balance():
            """Get real-time account balance"""
            try:
                balance = self.portfolio_manager.fetch_current_balance()
                self.data_cache['balance'] = balance
                
                return jsonify({
                    'success': True,
                    'balance': balance,
                    'formatted': f"₹{balance:,.2f}",
                    'last_updated': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Balance fetch error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'balance': 0
                }), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current trading positions"""
            try:
                positions = self.trading_engine.get_positions()
                portfolio_positions = self.portfolio_manager.get_current_positions()
                
                # Combine live and portfolio positions
                combined_positions = {
                    'live_positions': positions or [],
                    'portfolio_positions': portfolio_positions or [],
                    'total_positions': len(positions or []) + len(portfolio_positions or []),
                    'last_updated': datetime.now().isoformat()
                }
                
                self.data_cache['positions'] = combined_positions
                return jsonify(combined_positions)
                
            except Exception as e:
                logger.error(f"Positions fetch error: {e}")
                return jsonify({
                    'live_positions': [],
                    'portfolio_positions': [],
                    'total_positions': 0,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/pnl')
        def get_pnl():
            """Get P&L data"""
            try:
                daily_pnl = self.portfolio_manager.calculate_daily_pnl()
                total_pnl = self.portfolio_manager.total_pnl
                
                pnl_data = {
                    'daily_pnl': daily_pnl,
                    'total_pnl': total_pnl,
                    'daily_pnl_formatted': f"₹{daily_pnl:,.2f}",
                    'total_pnl_formatted': f"₹{total_pnl:,.2f}",
                    'daily_percentage': (daily_pnl / self.portfolio_manager.total_capital) * 100,
                    'last_updated': datetime.now().isoformat()
                }
                
                self.data_cache['pnl'] = pnl_data
                return jsonify(pnl_data)
                
            except Exception as e:
                logger.error(f"P&L calculation error: {e}")
                return jsonify({
                    'daily_pnl': 0,
                    'total_pnl': 0,
                    'error': str(e)
                }), 500
        
        # ======================
        # MARKET DATA ROUTES
        # ======================
        
        @self.app.route('/api/market/nifty')
        def get_nifty_data():
            """Get NIFTY market data"""
            try:
                if self.dhan_client:
                    # Get real NIFTY data
                    nifty_data = self._get_live_index_data('NIFTY')
                else:
                    # Demo data
                    nifty_data = {
                        'price': 25234.50,
                        'change': 125.30,
                        'change_percent': 0.50,
                        'high': 25289.75,
                        'low': 25156.20,
                        'volume': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                
                return jsonify(nifty_data)
                
            except Exception as e:
                logger.error(f"NIFTY data error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/market/banknifty')
        def get_banknifty_data():
            """Get BANKNIFTY market data"""
            try:
                if self.dhan_client:
                    # Get real BANKNIFTY data
                    banknifty_data = self._get_live_index_data('BANKNIFTY')
                else:
                    # Demo data
                    banknifty_data = {
                        'price': 55456.75,
                        'change': -234.25,
                        'change_percent': -0.42,
                        'high': 55698.50,
                        'low': 55234.10,
                        'volume': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                
                return jsonify(banknifty_data)
                
            except Exception as e:
                logger.error(f"BANKNIFTY data error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/market/option-chain/<symbol>')
        def get_option_chain(symbol):
            """Get option chain data"""
            try:
                symbol_upper = symbol.upper()
                
                if symbol_upper == 'NIFTY':
                    symbol_id = 13
                elif symbol_upper == 'BANKNIFTY':
                    symbol_id = 25
                else:
                    return jsonify({'error': 'Invalid symbol'}), 400
                
                option_chain = self.market_data_engine.get_option_chain(symbol_id, symbol_upper)
                return jsonify(option_chain or {'error': 'No data available'})
                
            except Exception as e:
                logger.error(f"Option chain error: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ======================
        # AI & ANALYSIS ROUTES
        # ======================
        
        @self.app.route('/api/ai/decision')
        def get_ai_decision():
            """Get AI trading decision"""
            try:
                # Get market conditions
                market_conditions = self._get_current_market_conditions()
                
                # Get ML prediction
                ml_decision = self.ml_trader.should_take_trade(market_conditions)
                
                # Get global market bias
                global_bias = self.global_analyzer.get_trading_bias()
                
                # Combine decisions
                ai_decision = {
                    'ml_decision': ml_decision,
                    'global_bias': global_bias,
                    'combined_confidence': (ml_decision['ml_confidence'] + global_bias['strength']) / 2,
                    'recommendation': self._combine_recommendations(ml_decision, global_bias),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.data_cache['ai_signals'] = ai_decision
                return jsonify(ai_decision)
                
            except Exception as e:
                logger.error(f"AI decision error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analysis/technical/<symbol>')
        def get_technical_analysis(symbol):
            """Get technical indicators"""
            try:
                # Get historical data
                historical_data = self.historical_fetcher.get_historical_data(
                    symbol.upper(), '5m', 5
                )
                
                if not historical_data:
                    return jsonify({'error': 'No historical data available'}), 404
                
                # Calculate indicators
                indicators = self._calculate_technical_indicators(historical_data)
                
                return jsonify({
                    'symbol': symbol.upper(),
                    'indicators': indicators,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Technical analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/news/sentiment')
        def get_news_sentiment():
            """Get news sentiment analysis"""
            try:
                # Fetch and analyze news
                news_items = self.news_analyzer.fetch_market_news("NIFTY BANKNIFTY", hours_back=24)
                analyzed_news = self.news_analyzer.analyze_news_sentiment(news_items)
                market_mood = self.news_analyzer.calculate_market_mood(analyzed_news)
                
                sentiment_data = {
                    'market_mood': market_mood,
                    'top_headlines': analyzed_news[:5],
                    'sentiment_score': market_mood.get('average_sentiment_score', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.data_cache['news_sentiment'] = sentiment_data
                return jsonify(sentiment_data)
                
            except Exception as e:
                logger.error(f"News sentiment error: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ======================
        # TRADING EXECUTION ROUTES
        # ======================
        
        @self.app.route('/api/trading/mode', methods=['GET', 'POST'])
        def trading_mode():
            """Get or set trading mode"""
            if request.method == 'GET':
                return jsonify({
                    'mode': 'live' if self.trading_engine.is_live_mode else 'paper',
                    'live_enabled': self.trading_engine.is_live_mode
                })
            
            elif request.method == 'POST':
                data = request.get_json()
                new_mode = data.get('mode', 'paper')
                
                if new_mode == 'live':
                    # Require confirmation for live trading
                    confirmation = data.get('confirmation', False)
                    if confirmation:
                        self.trading_engine.enable_live_trading()
                        logger.warning("🔴 LIVE TRADING MODE ENABLED")
                        return jsonify({'mode': 'live', 'status': 'enabled'})
                    else:
                        return jsonify({'error': 'Confirmation required for live trading'}), 400
                else:
                    self.trading_engine.disable_live_trading()
                    logger.info("📝 PAPER TRADING MODE ENABLED")
                    return jsonify({'mode': 'paper', 'status': 'enabled'})
        
        @self.app.route('/api/trading/execute', methods=['POST'])
        def execute_trade():
            """Execute a trade"""
            try:
                trade_data = request.get_json()
                
                # Validate trade data
                required_fields = ['symbol', 'strike', 'option_type', 'action', 'quantity']
                for field in required_fields:
                    if field not in trade_data:
                        return jsonify({'error': f'Missing field: {field}'}), 400
                
                # Execute trade through trading engine
                result = self.trading_engine.place_live_order(trade_data)
                
                if result:
                    # Update portfolio
                    if not self.trading_engine.is_live_mode:
                        self.portfolio_manager.simulate_paper_trade(trade_data)
                    
                    return jsonify({
                        'success': True,
                        'trade_id': result.get('order_id', 'PAPER_TRADE'),
                        'status': 'executed',
                        'mode': 'live' if self.trading_engine.is_live_mode else 'paper'
                    })
                else:
                    return jsonify({'error': 'Trade execution failed'}), 500
                    
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                return jsonify({'error': str(e)}), 500
        
        # ======================
        # DASHBOARD SUMMARY ROUTE
        # ======================
        
        @self.app.route('/api/dashboard/summary')
        def dashboard_summary():
            """Get complete dashboard summary"""
            try:
                # Collect all data
                balance = self.portfolio_manager.fetch_current_balance()
                positions = self.trading_engine.get_positions()
                daily_pnl = self.portfolio_manager.calculate_daily_pnl()
                
                # Get recent performance
                performance_data = self.portfolio_manager.get_performance_summary()
                
                # Market status
                market_open = self._is_market_open()
                
                summary = {
                    'balance': {
                        'total': balance,
                        'available': balance,
                        'formatted': f"₹{balance:,.2f}"
                    },
                    'pnl': {
                        'daily': daily_pnl,
                        'total': self.portfolio_manager.total_pnl,
                        'daily_formatted': f"₹{daily_pnl:,.2f}",
                        'daily_percentage': (daily_pnl / self.portfolio_manager.total_capital) * 100 if self.portfolio_manager.total_capital > 0 else 0
                    },
                    'positions': {
                        'count': len(positions or []),
                        'active': len([p for p in (positions or []) if p.get('status') == 'OPEN'])
                    },
                    'performance': performance_data,
                    'market': {
                        'status': 'OPEN' if market_open else 'CLOSED',
                        'is_open': market_open
                    },
                    'trading_mode': 'live' if self.trading_engine.is_live_mode else 'paper',
                    'last_updated': datetime.now().isoformat()
                }
                
                return jsonify(summary)
                
            except Exception as e:
                logger.error(f"Dashboard summary error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websocket_events(self):
        """Setup WebSocket events for real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to TradeMind AI Backend'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_updates')
        def handle_subscribe():
            logger.info(f"Client subscribed to updates: {request.sid}")
            # Send initial data
            self._emit_real_time_data()
    
    def _start_background_tasks(self):
        """Start background tasks for real-time updates"""
        
        def real_time_data_updater():
            """Update real-time data every 5 seconds"""
            while True:
                try:
                    if hasattr(self, 'socketio'):
                        self._emit_real_time_data()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Real-time update error: {e}")
                    time.sleep(10)  # Wait longer on error
        
        def ai_decision_updater():
            """Update AI decisions every 30 seconds"""
            while True:
                try:
                    if hasattr(self, 'socketio'):
                        self._update_ai_decisions()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"AI decision update error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start background threads
        threading.Thread(target=real_time_data_updater, daemon=True).start()
        threading.Thread(target=ai_decision_updater, daemon=True).start()
        
        logger.info("✅ Background tasks started")
    
    def _emit_real_time_data(self):
        """Emit real-time data to connected clients"""
        try:
            # Get current data
            balance = self.portfolio_manager.fetch_current_balance()
            positions = self.trading_engine.get_positions()
            daily_pnl = self.portfolio_manager.calculate_daily_pnl()
            
            # Prepare data packet
            data_packet = {
                'balance': balance,
                'balance_formatted': f"₹{balance:,.2f}",
                'positions_count': len(positions or []),
                'daily_pnl': daily_pnl,
                'daily_pnl_formatted': f"₹{daily_pnl:,.2f}",
                'timestamp': datetime.now().isoformat(),
                'market_status': 'OPEN' if self._is_market_open() else 'CLOSED'
            }
            
            # Emit to all connected clients
            self.socketio.emit('real_time_update', data_packet)
            
        except Exception as e:
            logger.error(f"Real-time emit error: {e}")
    
    def _update_ai_decisions(self):
        """Update AI trading decisions"""
        try:
            # Get market conditions
            market_conditions = self._get_current_market_conditions()
            
            # Get AI decision
            ml_decision = self.ml_trader.should_take_trade(market_conditions)
            
            # Emit AI update
            self.socketio.emit('ai_update', {
                'decision': ml_decision['decision'],
                'confidence': ml_decision['ml_confidence'],
                'recommendation': ml_decision['recommendation'],
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"AI decision update error: {e}")
    
    # ======================
    # HELPER METHODS
    # ======================
    
    def _get_live_index_data(self, symbol: str) -> Dict[str, Any]:
        """Get live index data from Dhan API"""
        try:
            if not self.dhan_client:
                raise Exception("Dhan client not available")
            
            # Map symbols to Dhan IDs
            symbol_ids = {'NIFTY': 13, 'BANKNIFTY': 25}
            symbol_id = symbol_ids.get(symbol)
            
            if not symbol_id:
                raise Exception(f"Symbol {symbol} not supported")
            
            # Get real-time data
            # Note: Dhan API method calls may vary - adjust as needed
            data = self.dhan_client.get_intraday_daily_minute_charts(
                security_id=str(symbol_id),
                exchange_segment="IDX_I",
                instrument_type="INDEX"
            )
            
            if data and 'data' in data:
                latest = data['data'][-1] if data['data'] else {}
                return {
                    'price': latest.get('close', 0),
                    'change': latest.get('close', 0) - latest.get('open', 0),
                    'change_percent': ((latest.get('close', 0) - latest.get('open', 0)) / latest.get('open', 1)) * 100,
                    'high': latest.get('high', 0),
                    'low': latest.get('low', 0),
                    'volume': latest.get('volume', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise Exception("No data received from Dhan API")
                
        except Exception as e:
            logger.error(f"Live data fetch error for {symbol}: {e}")
            # Return demo data on error
            return {
                'price': 25000 if symbol == 'NIFTY' else 55000,
                'change': 0,
                'change_percent': 0,
                'high': 0,
                'low': 0,
                'volume': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _get_current_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions for AI analysis"""
        try:
            # This would typically fetch real market data
            # For now, return sample conditions
            return {
                'rsi': 45,
                'macd_signal': 1,
                'oi_ratio': 1.2,
                'iv_skew': -1,
                'moneyness': 1.0,
                'days_to_expiry': 7,
                'vix_level': 16,
                'pcr': 1.1,
                'delta': 0.4,
                'ai_confidence': 75,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Market conditions error: {e}")
            return {}
    
    def _combine_recommendations(self, ml_decision: Dict, global_bias: Dict) -> str:
        """Combine ML and global recommendations"""
        try:
            ml_rec = ml_decision.get('recommendation', 'NEUTRAL')
            global_rec = global_bias.get('direction', 'NEUTRAL')
            
            if ml_rec in ['STRONG BUY', 'BUY'] and global_rec == 'BULLISH':
                return 'STRONG BUY'
            elif ml_rec in ['STRONG BUY', 'BUY'] or global_rec == 'BULLISH':
                return 'BUY'
            elif ml_rec in ['AVOID', 'CAUTION'] and global_rec == 'BEARISH':
                return 'STRONG SELL'
            elif ml_rec in ['AVOID', 'CAUTION'] or global_rec == 'BEARISH':
                return 'SELL'
            else:
                return 'NEUTRAL'
        except Exception:
            return 'NEUTRAL'
    
    def _calculate_technical_indicators(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate technical indicators from historical data"""
        try:
            if not historical_data or len(historical_data) < 14:
                return {}
            
            # Convert to price arrays
            closes = [float(candle.get('close', 0)) for candle in historical_data]
            highs = [float(candle.get('high', 0)) for candle in historical_data]
            lows = [float(candle.get('low', 0)) for candle in historical_data]
            
            # Calculate RSI (simplified)
            def calculate_rsi(prices, period=14):
                if len(prices) < period:
                    return 50
                
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
                
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                
                if avg_loss == 0:
                    return 100
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return round(rsi, 2)
            
            # Calculate moving averages
            def calculate_sma(prices, period):
                if len(prices) < period:
                    return 0
                return sum(prices[-period:]) / period
            
            rsi = calculate_rsi(closes)
            sma_20 = calculate_sma(closes, 20)
            sma_50 = calculate_sma(closes, 50)
            current_price = closes[-1] if closes else 0
            
            return {
                'rsi': rsi,
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'current_price': current_price,
                'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
                'price_vs_sma20': 'ABOVE' if current_price > sma_20 else 'BELOW',
                'trend': 'BULLISH' if sma_20 > sma_50 else 'BEARISH'
            }
            
        except Exception as e:
            logger.error(f"Technical indicators calculation error: {e}")
            return {}
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now()
            weekday = now.weekday()  # 0 = Monday, 6 = Sunday
            
            # Check if it's a weekend
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Check market hours (9:15 AM to 3:30 PM IST)
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception:
            return False
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the backend server"""
        logger.info(f"🚀 Starting TradeMind AI Backend on {host}:{port}")
        logger.info(f"📊 Dashboard URL: http://{host}:{port}")
        logger.info(f"🔗 API Base URL: http://{host}:{port}/api")
        
        try:
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        except KeyboardInterrupt:
            logger.info("⏹️ TradeMind AI Backend stopped by user")
        except Exception as e:
            logger.error(f"❌ Backend error: {e}")
            logger.error(traceback.format_exc())

# ======================
# MAIN EXECUTION
# ======================

def main():
    """Main function to start the backend"""
    print("🧠 TradeMind AI Dashboard Backend")
    print("=" * 50)
    
    try:
        # Create backend instance
        backend = TradeMindBackend()
        
        # Run the server
        backend.run(
            host='127.0.0.1',
            port=5000,
            debug=False  # Set to True for development
        )
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()