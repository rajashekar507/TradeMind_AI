from flask import Flask, jsonify
from flask_cors import CORS
import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('src/portfolio')

try:
    from portfolio.portfolio_manager import PortfolioManager
    pm = PortfolioManager()
except ImportError as e:
    print(f"Import error: {e}")
    pm = None

app = Flask(__name__)
CORS(app)

@app.route('/api/balance')
def get_balance():
    try:
        if pm:
            balance = pm.fetch_current_balance()
            return jsonify({'balance': balance, 'status': 'success'})
        else:
            return jsonify({'balance': 0, 'status': 'error', 'message': 'Portfolio manager not available'})
    except Exception as e:
        return jsonify({'balance': 0, 'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Balance Server...")
    print("📊 URL: http://localhost:5001/api/balance")
    app.run(host='0.0.0.0', port=5001, debug=True)