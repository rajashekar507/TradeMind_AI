"""
TradeMind_AI: Unified Web Dashboard
Real-time monitoring and one-click trading interface
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Any

# Import your existing modules
try:
    from core.master_trader import EnhancedMasterTrader
    from portfolio.portfolio_manager import PortfolioManager
    from data.market_data import MarketDataEngine
    from analysis.ml_trader import SelfLearningTrader
    from analysis.global_market_analyzer import GlobalMarketAnalyzer
except ImportError:
    print("⚠️ Some modules not found. Dashboard will run with limited features.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardServer:
    def __init__(self):
        """Initialize Dashboard Server"""
        print("🌐 Initializing TradeMind_AI Dashboard...")
        
        # Initialize components
        self.portfolio_manager = None
        self.market_engine = None
        self.ml_trader = None
        self.global_analyzer = None
        self.master_trader = None
        
        # Dashboard data
        self.dashboard_data = {
            'portfolio': {},
            'market_data': {},
            'active_trades': [],
            'trade_history': [],
            'global_markets': {},
            'ml_insights': {},
            'system_status': {
                'trading_active': False,
                'auto_trading': False,
                'last_update': None
            }
        }
        
        # Initialize components
        self.initialize_components()
        
        # Start background updates
        self.start_background_updates()
        
        print("✅ Dashboard Server ready!")
    
    def initialize_components(self):
        """Initialize all trading components"""
        try:
            self.portfolio_manager = PortfolioManager()
            self.market_engine = MarketDataEngine()
            self.ml_trader = SelfLearningTrader()
            self.global_analyzer = GlobalMarketAnalyzer()
            print("✅ All components initialized")
        except Exception as e:
            print(f"⚠️ Component initialization error: {e}")
    
    def start_background_updates(self):
        """Start background data updates"""
        def update_loop():
            while True:
                try:
                    # Update portfolio
                    self.update_portfolio_data()
                    
                    # Update market data
                    self.update_market_data()
                    
                    # Update global markets
                    self.update_global_markets()
                    
                    # Emit updates to connected clients
                    socketio.emit('data_update', self.dashboard_data)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    print(f"❌ Update error: {e}")
                    time.sleep(10)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def update_portfolio_data(self):
        """Update portfolio data"""
        if self.portfolio_manager:
            summary = self.portfolio_manager.get_portfolio_summary()
            self.dashboard_data['portfolio'] = summary
            
            # Get active trades
            if hasattr(self.portfolio_manager, 'trades_database'):
                trades = self.portfolio_manager.trades_database.get('trades', [])
                self.dashboard_data['active_trades'] = [
                    t for t in trades if t.get('status') == 'OPEN'
                ]
                self.dashboard_data['trade_history'] = [
                    t for t in trades if t.get('status') == 'CLOSED'
                ][-10:]  # Last 10 closed trades
    
    def update_market_data(self):
        """Update market data"""
        if self.market_engine:
            # Get NIFTY data
            nifty_data = self.market_engine.get_option_chain(
                self.market_engine.NIFTY_ID, "NIFTY"
            )
            
            # Get BANKNIFTY data
            banknifty_data = self.market_engine.get_option_chain(
                self.market_engine.BANKNIFTY_ID, "BANKNIFTY"
            )
            
            # Process data
            if nifty_data:
                analysis = self.market_engine.analyze_option_data(nifty_data)
                if analysis:
                    self.dashboard_data['market_data']['NIFTY'] = analysis
            
            if banknifty_data:
                analysis = self.market_engine.analyze_option_data(banknifty_data)
                if analysis:
                    self.dashboard_data['market_data']['BANKNIFTY'] = analysis
    
    def update_global_markets(self):
        """Update global market data"""
        if self.global_analyzer:
            # Get market snapshot
            global_data = self.global_analyzer.fetch_global_market_data()
            
            # Get trading bias
            bias = self.global_analyzer.get_trading_bias()
            
            self.dashboard_data['global_markets'] = {
                'data': global_data,
                'bias': bias,
                'last_update': datetime.now().isoformat()
            }
    
    def execute_trade(self, trade_params):
        """Execute trade from dashboard"""
        try:
            # Validate parameters
            required_fields = ['symbol', 'strike', 'option_type', 'quantity']
            for field in required_fields:
                if field not in trade_params:
                    return {'success': False, 'error': f'Missing {field}'}
            
            # Add trade to portfolio
            if self.portfolio_manager:
                self.portfolio_manager.add_trade(trade_params)
                
                return {
                    'success': True,
                    'message': 'Trade executed successfully',
                    'trade_id': trade_params.get('trade_id')
                }
            else:
                return {'success': False, 'error': 'Portfolio manager not initialized'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_ml_recommendations(self):
        """Get ML trading recommendations"""
        if self.ml_trader:
            recommendations = self.ml_trader.get_strategy_recommendations()
            
            # Get current market setup
            market_setup = {
                'rsi': 45,  # Would come from technical indicators
                'macd_signal': 1,
                'oi_ratio': 1.2,
                'iv_skew': -1.5,
                'moneyness': 0.99,
                'days_to_expiry': 5,
                'vix_level': 16,
                'pcr': 1.1,
                'delta': 0.4,
                'ai_confidence': 80,
                'timestamp': datetime.now()
            }
            
            # Get ML decision
            ml_decision = self.ml_trader.should_take_trade(market_setup)
            
            return {
                'recommendations': recommendations,
                'ml_decision': ml_decision,
                'timestamp': datetime.now().isoformat()
            }
        
        return None

# Dashboard instance
dashboard = DashboardServer()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio data"""
    return jsonify(dashboard.dashboard_data['portfolio'])

@app.route('/api/market_data')
def get_market_data():
    """Get market data"""
    return jsonify(dashboard.dashboard_data['market_data'])

@app.route('/api/global_markets')
def get_global_markets():
    """Get global market data"""
    return jsonify(dashboard.dashboard_data['global_markets'])

@app.route('/api/active_trades')
def get_active_trades():
    """Get active trades"""
    return jsonify(dashboard.dashboard_data['active_trades'])

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history"""
    return jsonify(dashboard.dashboard_data['trade_history'])

@app.route('/api/ml_insights')
def get_ml_insights():
    """Get ML insights"""
    insights = dashboard.get_ml_recommendations()
    return jsonify(insights)

@app.route('/api/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    trade_params = request.json
    result = dashboard.execute_trade(trade_params)
    return jsonify(result)

@app.route('/api/toggle_auto_trading', methods=['POST'])
def toggle_auto_trading():
    """Toggle auto trading on/off"""
    dashboard.dashboard_data['system_status']['auto_trading'] = \
        not dashboard.dashboard_data['system_status']['auto_trading']
    
    status = "ENABLED" if dashboard.dashboard_data['system_status']['auto_trading'] else "DISABLED"
    
    return jsonify({
        'success': True,
        'auto_trading': dashboard.dashboard_data['system_status']['auto_trading'],
        'message': f'Auto trading {status}'
    })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to TradeMind_AI Dashboard'})
    
    # Send initial data
    emit('data_update', dashboard.dashboard_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request"""
    dashboard.update_portfolio_data()
    dashboard.update_market_data()
    dashboard.update_global_markets()
    emit('data_update', dashboard.dashboard_data)

# Create templates directory and HTML file
def create_dashboard_html():
    """Create the dashboard HTML template"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradeMind_AI Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .header {
            background: #1a1a2e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #16213e;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .status {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid #16213e;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 212, 255, 0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #16213e;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #00d4ff;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
        }
        
        .metric-label {
            color: #888;
            font-size: 0.9rem;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .positive { color: #00ff88; }
        .negative { color: #ff3366; }
        .neutral { color: #ffaa00; }
        
        .btn {
            background: #00d4ff;
            color: #0f0f23;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: #00a8cc;
            transform: translateY(-1px);
        }
        
        .btn-danger {
            background: #ff3366;
            color: white;
        }
        
        .btn-danger:hover {
            background: #cc0033;
        }
        
        .trade-form {
            display: grid;
            gap: 1rem;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .form-group label {
            color: #888;
            font-size: 0.9rem;
        }
        
        .form-group input, .form-group select {
            background: #0f0f23;
            border: 1px solid #16213e;
            color: #e0e0e0;
            padding: 0.5rem;
            border-radius: 5px;
        }
        
        .trades-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .trade-item {
            background: #0f0f23;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .trade-info {
            flex: 1;
        }
        
        .trade-pnl {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        #pnlChart {
            max-height: 300px;
        }
        
        .alert {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 1.5rem;
            border-radius: 5px;
            background: #00d4ff;
            color: #0f0f23;
            font-weight: 600;
            animation: slideIn 0.3s ease-out;
            z-index: 1000;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🚀 TradeMind_AI Dashboard</div>
        <div class="status">
            <div class="status-indicator"></div>
            <span>Connected</span>
            <button class="btn" id="autoTradeBtn">Auto Trading: OFF</button>
        </div>
    </div>
    
    <div class="container">
        <!-- Portfolio Summary -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">💰 Portfolio Summary</h3>
            </div>
            <div class="metric">
                <span class="metric-label">Portfolio Value</span>
                <span class="metric-value" id="portfolioValue">₹0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Available Capital</span>
                <span class="metric-value" id="availableCapital">₹0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total P&L</span>
                <span class="metric-value" id="totalPnl">₹0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Win Rate</span>
                <span class="metric-value" id="winRate">0%</span>
            </div>
        </div>
        
        <!-- Market Data -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">📊 Market Data</h3>
            </div>
            <div class="metric">
                <span class="metric-label">NIFTY</span>
                <span class="metric-value" id="niftyPrice">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">BANKNIFTY</span>
                <span class="metric-value" id="bankniftyPrice">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">India VIX</span>
                <span class="metric-value" id="indiaVix">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Global Sentiment</span>
                <span class="metric-value" id="globalSentiment">-</span>
            </div>
        </div>
        
        <!-- Quick Trade -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">⚡ Quick Trade</h3>
            </div>
            <form class="trade-form" id="tradeForm">
                <div class="form-group">
                    <label>Symbol</label>
                    <select name="symbol" required>
                        <option value="NIFTY">NIFTY</option>
                        <option value="BANKNIFTY">BANKNIFTY</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Strike Price</label>
                    <input type="number" name="strike" required>
                </div>
                <div class="form-group">
                    <label>Option Type</label>
                    <select name="option_type" required>
                        <option value="CE">Call (CE)</option>
                        <option value="PE">Put (PE)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Quantity (Lots)</label>
                    <input type="number" name="quantity" value="1" min="1" required>
                </div>
                <button type="submit" class="btn">Execute Trade</button>
            </form>
        </div>
        
        <!-- Active Trades -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">📈 Active Trades</h3>
            </div>
            <div class="trades-list" id="activeTrades">
                <!-- Active trades will be populated here -->
            </div>
        </div>
        
        <!-- ML Insights -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">🧠 AI Insights</h3>
            </div>
            <div id="mlInsights">
                <!-- ML insights will be populated here -->
            </div>
        </div>
        
        <!-- P&L Chart -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">📊 P&L Chart</h3>
            </div>
            <canvas id="pnlChart"></canvas>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Chart.js setup
        const ctx = document.getElementById('pnlChart').getContext('2d');
        const pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#16213e'
                        }
                    },
                    x: {
                        grid: {
                            color: '#16213e'
                        }
                    }
                }
            }
        });
        
        // Socket event handlers
        socket.on('data_update', (data) => {
            updateDashboard(data);
        });
        
        // Update dashboard with new data
        function updateDashboard(data) {
            // Update portfolio
            if (data.portfolio) {
                document.getElementById('portfolioValue').textContent = 
                    `₹${(data.portfolio.portfolio_value || 0).toLocaleString('en-IN')}`;
                document.getElementById('availableCapital').textContent = 
                    `₹${(data.portfolio.available_capital || 0).toLocaleString('en-IN')}`;
                
                const pnl = data.portfolio.total_pnl || 0;
                const pnlElement = document.getElementById('totalPnl');
                pnlElement.textContent = `₹${Math.abs(pnl).toLocaleString('en-IN')}`;
                pnlElement.className = pnl >= 0 ? 'metric-value positive' : 'metric-value negative';
                
                document.getElementById('winRate').textContent = 
                    `${(data.portfolio.win_rate || 0).toFixed(1)}%`;
            }
            
            // Update market data
            if (data.market_data) {
                if (data.market_data.NIFTY) {
                    document.getElementById('niftyPrice').textContent = 
                        `₹${data.market_data.NIFTY.underlying_price.toLocaleString('en-IN')}`;
                }
                if (data.market_data.BANKNIFTY) {
                    document.getElementById('bankniftyPrice').textContent = 
                        `₹${data.market_data.BANKNIFTY.underlying_price.toLocaleString('en-IN')}`;
                }
            }
            
            // Update global markets
            if (data.global_markets && data.global_markets.bias) {
                const sentiment = data.global_markets.bias.direction;
                const sentimentElement = document.getElementById('globalSentiment');
                sentimentElement.textContent = sentiment;
                sentimentElement.className = 
                    sentiment === 'BULLISH' ? 'metric-value positive' :
                    sentiment === 'BEARISH' ? 'metric-value negative' :
                    'metric-value neutral';
            }
            
            // Update active trades
            updateActiveTrades(data.active_trades || []);
            
            // Update P&L chart
            updatePnLChart(data.trade_history || []);
        }
        
        // Update active trades list
        function updateActiveTrades(trades) {
            const container = document.getElementById('activeTrades');
            container.innerHTML = '';
            
            trades.forEach(trade => {
                const tradeElement = document.createElement('div');
                tradeElement.className = 'trade-item';
                
                const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
                
                tradeElement.innerHTML = `
                    <div class="trade-info">
                        <div>${trade.symbol} ${trade.strike} ${trade.option_type}</div>
                        <div style="font-size: 0.9rem; color: #888;">
                            Entry: ₹${trade.entry_price} | Current: ₹${trade.current_price}
                        </div>
                    </div>
                    <div class="trade-pnl ${pnlClass}">
                        ₹${Math.abs(trade.pnl).toFixed(0)}
                    </div>
                `;
                
                container.appendChild(tradeElement);
            });
        }
        
        // Update P&L chart
        function updatePnLChart(tradeHistory) {
            const labels = [];
            const data = [];
            let cumulativePnl = 0;
            
            tradeHistory.forEach((trade, index) => {
                cumulativePnl += trade.pnl || 0;
                labels.push(`Trade ${index + 1}`);
                data.push(cumulativePnl);
            });
            
            pnlChart.data.labels = labels;
            pnlChart.data.datasets[0].data = data;
            pnlChart.update();
        }
        
        // Handle trade form submission
        document.getElementById('tradeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const tradeData = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/execute_trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(tradeData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showAlert('Trade executed successfully!', 'success');
                    e.target.reset();
                } else {
                    showAlert(result.error || 'Trade failed', 'error');
                }
            } catch (error) {
                showAlert('Network error', 'error');
            }
        });
        
        // Handle auto trading toggle
        document.getElementById('autoTradeBtn').addEventListener('click', async () => {
            try {
                const response = await fetch('/api/toggle_auto_trading', {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const btn = document.getElementById('autoTradeBtn');
                    btn.textContent = `Auto Trading: ${result.auto_trading ? 'ON' : 'OFF'}`;
                    btn.className = result.auto_trading ? 'btn btn-danger' : 'btn';
                }
            } catch (error) {
                showAlert('Failed to toggle auto trading', 'error');
            }
        });
        
        // Show alert
        function showAlert(message, type) {
            const alert = document.createElement('div');
            alert.className = 'alert';
            alert.textContent = message;
            
            if (type === 'error') {
                alert.style.background = '#ff3366';
                alert.style.color = 'white';
            }
            
            document.body.appendChild(alert);
            
            setTimeout(() => {
                alert.remove();
            }, 3000);
        }
        
        // Request initial update
        socket.emit('request_update');
    </script>
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
        f.write(html_content)

def run_dashboard(port=5000):
    """Run the dashboard server"""
    create_dashboard_html()
    print(f"\n🌐 Starting TradeMind_AI Dashboard on http://localhost:{port}")
    print("📊 Open your browser to view the dashboard")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    run_dashboard()