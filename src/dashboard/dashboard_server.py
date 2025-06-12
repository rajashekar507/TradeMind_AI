"""
TradeMind_AI: Professional Dashboard Server
Separated backend logic with proper template usage
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import threading
import time
from typing import Dict, List, Any, Optional

# Import our modules
from config.constants import (
    LOT_SIZES, MARKET_HOURS, PAPER_TRADING, LIVE_TRADING,
    is_market_open, is_trading_holiday
)
from core.unified_trading_engine import UnifiedTradingEngine, TradingMode, OrderDetails
from utils.rate_limiter import rate_limiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DashboardServer')

# Flask app setup
app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static'
)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardServer:
    """
    Professional dashboard server with real-time updates
    """
    
    def __init__(self):
        """Initialize dashboard server"""
        logger.info("🌐 Initializing TradeMind AI Dashboard Server...")
        
        # Initialize trading engine
        self.trading_engine = UnifiedTradingEngine(TradingMode.PAPER)
        
        # Dashboard state
        self.connected_clients = set()
        self.update_interval = 5  # seconds
        self.is_running = False
        
        # Cache for performance
        self.data_cache = {
            'account': {},
            'positions': [],
            'pnl': {},
            'stats': {},
            'market_data': {},
            'pnl_history': {
                'timestamps': [],
                'values': []
            },
            'last_update': None
        }
        
        # Start background tasks
        self.start_background_tasks()
        
        logger.info("✅ Dashboard Server initialized")
    
    def start_background_tasks(self):
        """Start background update tasks"""
        self.is_running = True
        
        # Data updater thread
        update_thread = threading.Thread(target=self._background_updater, daemon=True)
        update_thread.start()
        
        # Position monitor thread
        monitor_thread = threading.Thread(target=self._position_monitor, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ Background tasks started")
    
    def _background_updater(self):
        """Background task to update dashboard data"""
        while self.is_running:
            try:
                # Update all data
                self.update_account_data()
                self.update_positions()
                self.update_pnl()
                self.update_stats()
                
                # Broadcast to all connected clients
                if self.connected_clients:
                    socketio.emit('data_update', self.get_dashboard_data())
                
            except Exception as e:
                logger.error(f"Background update error: {e}")
            
            time.sleep(self.update_interval)
    
    def _position_monitor(self):
        """Monitor positions for changes"""
        last_positions = []
        
        while self.is_running:
            try:
                current_positions = self.trading_engine.get_positions()
                
                # Check for changes
                if current_positions != last_positions:
                    # Position change detected
                    socketio.emit('position_update', {
                        'positions': current_positions,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Check for closed positions
                    for old_pos in last_positions:
                        if not any(p['order_id'] == old_pos['order_id'] for p in current_positions):
                            # Position closed
                            self._notify_position_closed(old_pos)
                    
                    last_positions = current_positions.copy()
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            
            time.sleep(2)  # Check every 2 seconds
    
    def update_account_data(self):
        """Update account balance data"""
        try:
            account_data = self.trading_engine.get_account_balance()
            self.data_cache['account'] = account_data
            self.data_cache['last_update'] = datetime.now()
        except Exception as e:
            logger.error(f"Account update error: {e}")
    
    def update_positions(self):
        """Update positions data"""
        try:
            positions = self.trading_engine.get_positions()
            
            # Update current prices if available
            # In production, this would fetch real-time prices
            for position in positions:
                if position.get('status') == 'OPEN':
                    # Simulate price movement for demo
                    import random
                    price_change = random.uniform(-2, 2) / 100
                    position['current_price'] = position['entry_price'] * (1 + price_change)
                    
                    # Update P&L
                    if position['action'] == 'BUY':
                        position['pnl'] = (
                            (position['current_price'] - position['entry_price']) * 
                            position['quantity'] * position['lot_size']
                        )
                    else:
                        position['pnl'] = (
                            (position['entry_price'] - position['current_price']) * 
                            position['quantity'] * position['lot_size']
                        )
            
            self.data_cache['positions'] = positions
            
            # Update paper positions if in paper mode
            if self.trading_engine.mode == TradingMode.PAPER:
                market_prices = {
                    f"{p['symbol']}_{p['strike']}_{p['option_type']}": p['current_price']
                    for p in positions if 'current_price' in p
                }
                self.trading_engine.update_paper_positions(market_prices)
                
        except Exception as e:
            logger.error(f"Position update error: {e}")
    
    def update_pnl(self):
        """Update P&L data"""
        try:
            pnl = self.trading_engine.get_pnl()
            self.data_cache['pnl'] = pnl
            
            # Update P&L history
            if pnl['total_pnl'] != 0:
                history = self.data_cache['pnl_history']
                history['timestamps'].append(datetime.now().strftime('%H:%M:%S'))
                history['values'].append(pnl['total_pnl'])
                
                # Keep last 50 points
                if len(history['timestamps']) > 50:
                    history['timestamps'] = history['timestamps'][-50:]
                    history['values'] = history['values'][-50:]
                    
        except Exception as e:
            logger.error(f"P&L update error: {e}")
    
    def update_stats(self):
        """Update trading statistics"""
        try:
            stats = {
                'trades_today': self.trading_engine.trades_today,
                'winning_trades': self.trading_engine.winning_trades,
                'losing_trades': self.trading_engine.losing_trades,
                'market_open': is_market_open(),
                'is_holiday': is_trading_holiday()
            }
            self.data_cache['stats'] = stats
        except Exception as e:
            logger.error(f"Stats update error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data"""
        return {
            'account': self.data_cache['account'],
            'positions': self.data_cache['positions'],
            'pnl': self.data_cache['pnl'],
            'stats': self.data_cache['stats'],
            'pnl_history': self.data_cache['pnl_history'],
            'mode': self.trading_engine.mode.value,
            'timestamp': datetime.now().isoformat()
        }
    
    def place_order(self, order_data: Dict) -> Dict[str, Any]:
        """Place order through trading engine"""
        try:
            # Create OrderDetails object
            order = OrderDetails(
                symbol=order_data['symbol'],
                strike=float(order_data['strike']),
                option_type=order_data['option_type'],
                action=order_data['action'],
                quantity=int(order_data['quantity']),
                order_type=order_data.get('order_type', 'MARKET'),
                price=float(order_data.get('price', 0))
            )
            
            # Place order
            result = self.trading_engine.place_order(order)
            
            # Send notification
            if result.success:
                self._notify_trade_placed(order, result)
            
            return {
                'success': result.success,
                'order_id': result.order_id,
                'message': result.message,
                'details': result.details
            }
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def close_position(self, order_id: str) -> Dict[str, Any]:
        """Close an open position"""
        try:
            # Find position
            position = next(
                (p for p in self.data_cache['positions'] if p['order_id'] == order_id),
                None
            )
            
            if not position:
                return {'success': False, 'message': 'Position not found'}
            
            # Create reverse order
            close_order = OrderDetails(
                symbol=position['symbol'],
                strike=position['strike'],
                option_type=position['option_type'],
                action='SELL' if position['action'] == 'BUY' else 'BUY',
                quantity=position['quantity'],
                order_type='MARKET',
                price=0
            )
            
            # Place closing order
            result = self.trading_engine.place_order(close_order)
            
            if result.success:
                # Update position status
                position['status'] = 'CLOSED'
                position['exit_time'] = datetime.now().isoformat()
                position['exit_price'] = position.get('current_price', position['entry_price'])
                
                # Update win/loss stats
                if position.get('pnl', 0) > 0:
                    self.trading_engine.winning_trades += 1
                else:
                    self.trading_engine.losing_trades += 1
                
                return {
                    'success': True,
                    'message': 'Position closed successfully',
                    'pnl': position.get('pnl', 0)
                }
            else:
                return {
                    'success': False,
                    'message': result.message
                }
                
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def set_trading_mode(self, mode: str) -> Dict[str, Any]:
        """Set trading mode (PAPER/LIVE)"""
        try:
            mode_enum = TradingMode[mode.upper()]
            success = self.trading_engine.switch_mode(mode_enum)
            
            if success:
                # Broadcast mode change
                socketio.emit('mode_changed', {
                    'mode': mode,
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': True,
                    'message': f'Trading mode set to {mode}'
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to switch mode. Check requirements.'
                }
                
        except Exception as e:
            logger.error(f"Mode switch error: {e}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def _notify_trade_placed(self, order: OrderDetails, result):
        """Send trade notification"""
        socketio.emit('trade_alert', {
            'type': 'success',
            'message': f'Order placed: {order.symbol} {order.strike} {order.option_type} - {order.action}',
            'order_id': result.order_id,
            'timestamp': datetime.now().isoformat()
        })
    
    def _notify_position_closed(self, position: Dict):
        """Send position closed notification"""
        pnl = position.get('pnl', 0)
        socketio.emit('trade_alert', {
            'type': 'success' if pnl > 0 else 'warning',
            'message': f'Position closed: {position["symbol"]} - P&L: ₹{pnl:,.2f}',
            'timestamp': datetime.now().isoformat()
        })
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API rate limit statistics"""
        return rate_limiter.get_all_stats()


# Create global dashboard instance
dashboard = DashboardServer()


# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Get all dashboard data"""
    return jsonify(dashboard.get_dashboard_data())


@app.route('/api/place_order', methods=['POST'])
def place_order():
    """Place order endpoint"""
    order_data = request.json
    result = dashboard.place_order(order_data)
    return jsonify(result)


@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Close position endpoint"""
    order_id = request.json.get('order_id')
    result = dashboard.close_position(order_id)
    return jsonify(result)


@app.route('/api/set_trading_mode', methods=['POST'])
def set_trading_mode():
    """Set trading mode endpoint"""
    mode = request.json.get('mode')
    result = dashboard.set_trading_mode(mode)
    return jsonify(result)


@app.route('/api/positions')
def get_positions():
    """Get current positions"""
    return jsonify({
        'positions': dashboard.data_cache['positions'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/api_stats')
def get_api_stats():
    """Get API rate limit statistics"""
    return jsonify(dashboard.get_api_stats())


# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    dashboard.connected_clients.add(client_id)
    logger.info(f'Client connected: {client_id}')
    
    # Send initial data
    emit('connected', {
        'message': 'Connected to TradeMind AI Dashboard',
        'mode': dashboard.trading_engine.mode.value
    })
    emit('data_update', dashboard.get_dashboard_data())


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    dashboard.connected_clients.discard(client_id)
    logger.info(f'Client disconnected: {client_id}')


@socketio.on('request_data')
def handle_data_request():
    """Handle data request from client"""
    emit('data_update', dashboard.get_dashboard_data())


@socketio.on('request_positions')
def handle_positions_request():
    """Handle positions request"""
    emit('position_update', {
        'positions': dashboard.data_cache['positions'],
        'timestamp': datetime.now().isoformat()
    })


def run_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server"""
    logger.info(f"🚀 Starting TradeMind AI Dashboard on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Run in development mode
    run_dashboard(debug=True)