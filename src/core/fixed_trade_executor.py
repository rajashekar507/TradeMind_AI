"""
Fixed Trade Execution System for TradeMind AI
Handles proper order execution with correct lot sizes and balance updates
"""

import os
import json
import time
from datetime import datetime
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv
import requests

# Import centralized constants - FIXED DUPLICATION
try:
    from config.constants import LOT_SIZES, get_lot_size
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config.constants import LOT_SIZES, get_lot_size

class FixedTradeExecutor:
    def __init__(self):
        """Initialize with proper lot sizes and balance tracking"""
        load_dotenv()
        
        # Dhan API setup
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        dhan_context = DhanContext(self.client_id, self.access_token)
        self.dhan = dhanhq(dhan_context)
        
        # REMOVED DUPLICATE LOT_SIZES - Now using centralized version from config.constants
        
        # Files
        self.trades_file = "data/trades_database.json"
        self.balance_file = "data/balance_history.json"
        
        # Load current balance
        self.current_balance = self.fetch_live_balance()
        
        print(f"✅ Fixed Trade Executor initialized with centralized lot sizes!")
        print(f"📦 NIFTY: {get_lot_size('NIFTY')}, BANKNIFTY: {get_lot_size('BANKNIFTY')}")
        
    def fetch_live_balance(self):
        """Get real balance from Dhan"""
        try:
            funds = self.dhan.get_fund_limits()
            if funds and 'data' in funds:
                balance_data = funds['data']
                available = balance_data.get('availabelBalance', 0)
                if available == 0:
                    available = balance_data.get('availableBalance', 10000)
                return float(available)
        except:
            return 10000.0  # Default
    
    def calculate_position_size(self, signal_data):
        """Calculate correct position size based on balance and risk"""
        symbol = signal_data['symbol']
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        
        # Get lot size using centralized function - FIXED
        lot_size = get_lot_size(symbol)
        
        # Risk per trade (1% of capital)
        risk_amount = self.current_balance * 0.01
        
        # Risk per lot
        risk_per_lot = (entry_price - stop_loss) * lot_size
        
        # Calculate lots (max 5 lots)
        if risk_per_lot > 0:
            lots = min(int(risk_amount / risk_per_lot), 5)
        else:
            lots = 1
            
        # Check affordability
        capital_needed = entry_price * lot_size * lots
        if capital_needed > self.current_balance * 0.8:  # Max 80% capital
            lots = int((self.current_balance * 0.8) / (entry_price * lot_size))
            
        return max(1, lots)  # At least 1 lot
    
    def execute_trade(self, signal_data):
        """Execute trade with proper logging"""
        # Calculate position size
        lots = self.calculate_position_size(signal_data)
        lot_size = get_lot_size(signal_data['symbol'])  # FIXED - using centralized function
        
        # Calculate required capital
        required_capital = signal_data['entry_price'] * lots * lot_size
        
        # Check if we have sufficient balance
        if required_capital > self.current_balance:
            print(f"❌ Insufficient Balance!")
            print(f"   Required: ₹{required_capital:,.2f}")
            print(f"   Available: ₹{self.current_balance:,.2f}")
            print(f"   Shortfall: ₹{required_capital - self.current_balance:,.2f}")
            
            # Try with 1 lot
            if signal_data['entry_price'] * lot_size <= self.current_balance:
                lots = 1
                required_capital = signal_data['entry_price'] * lots * lot_size
                print(f"📊 Adjusting to 1 lot. New capital: ₹{required_capital:,.2f}")
            else:
                print(f"❌ Cannot afford even 1 lot. Skipping trade.")
                return None
        
        # Create trade record
        trade = {
            'trade_id': f"TM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'symbol': signal_data['symbol'],
            'strike': signal_data['strike'],
            'option_type': signal_data['option_type'],
            'action': 'BUY',
            'quantity': lots,
            'lot_size': lot_size,
            'total_quantity': lots * lot_size,
            'entry_price': signal_data['entry_price'],
            'current_price': signal_data['entry_price'],
            'target_price': signal_data['target_price'],
            'stop_loss': signal_data['stop_loss'],
            'status': 'OPEN',
            'pnl': 0,
            'capital_allocated': required_capital,
            'execution_type': 'PAPER'  # Change to 'LIVE' for real trading
        }
        
        # Update balance
        old_balance = self.current_balance
        self.current_balance -= trade['capital_allocated']
        
        # Log balance change
        balance_update = {
            'timestamp': datetime.now().isoformat(),
            'trade_type': f"TRADE_OPENED_{signal_data['symbol']}",
            'amount': -trade['capital_allocated'],
            'balance_before': old_balance,
            'balance_after': self.current_balance,
            'change': -trade['capital_allocated']
        }
        
        # Save trade
        self._save_trade(trade)
        
        # Save balance update
        self._save_balance_update(balance_update)
        
        return trade
    
    def _save_trade(self, trade):
        """Save trade to database"""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {'trades': [], 'summary': {}}
                
            data['trades'].append(trade)
            
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"✅ Trade saved: {trade['trade_id']}")
            
        except Exception as e:
            print(f"❌ Error saving trade: {e}")
    
    def _save_balance_update(self, update):
        """Save balance update"""
        try:
            if os.path.exists(self.balance_file):
                with open(self.balance_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            history.append(update)
            
            with open(self.balance_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            print(f"💰 Balance updated: ₹{update['balance_after']:,.2f}")
            
        except Exception as e:
            print(f"❌ Error saving balance: {e}")

# Test the fixed executor
if __name__ == "__main__":
    executor = FixedTradeExecutor()
    
    # Display current balance
    print(f"\n💰 Current Balance: ₹{executor.current_balance:,.2f}")
    
    # Test trade with lower price that fits balance
    test_signal = {
        'symbol': 'NIFTY',
        'strike': 25500,
        'option_type': 'CE',
        'entry_price': 100,  # Reduced from 150 to fit balance
        'target_price': 130,
        'stop_loss': 80
    }
    
    print(f"\n📊 Attempting trade:")
    print(f"   Strike: {test_signal['strike']} {test_signal['option_type']}")
    print(f"   Entry Price: ₹{test_signal['entry_price']}")
    print(f"   Lot Size: {get_lot_size(test_signal['symbol'])}")
    
    trade = executor.execute_trade(test_signal)
    
    if trade:
        print(f"\n✅ Trade Executed:")
        print(f"   ID: {trade['trade_id']}")
        print(f"   Quantity: {trade['quantity']} lots x {trade['lot_size']} = {trade['total_quantity']}")
        print(f"   Capital: ₹{trade['capital_allocated']:,.2f}")
        print(f"   Remaining Balance: ₹{executor.current_balance:,.2f}")
        print(f"   Using Centralized Lot Size: {get_lot_size(test_signal['symbol'])} units")
    else:
        print(f"\n❌ Trade failed to execute")