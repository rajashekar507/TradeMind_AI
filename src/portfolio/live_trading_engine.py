"""
Live Trading Engine - Converts simulation to real trading
Fixed version with proper imports and execution
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import dhanhq with proper error handling
try:
    from dhanhq import DhanContext, dhanhq
    print("✅ dhanhq v2.1.0+ library loaded")
    NEW_VERSION = True
except ImportError:
    try:
        from dhanhq import dhanhq
        print("✅ dhanhq legacy version loaded")
        NEW_VERSION = False
    except ImportError:
        print("❌ dhanhq library not found!")
        print("📦 Install with: pip install dhanhq")
        sys.exit(1)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded")
except ImportError:
    print("⚠️ python-dotenv not found. Using os.getenv() directly")

class LiveTradingEngine:
    def __init__(self):
        """Initialize with proper dhanhq setup"""
        print("🔄 Initializing Live Trading Engine...")
        
        # Get credentials
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not self.client_id or not self.access_token:
            print("❌ DHAN CREDENTIALS NOT FOUND!")
            print("📝 Please add to your .env file:")
            print("DHAN_CLIENT_ID=your_client_id")
            print("DHAN_ACCESS_TOKEN=your_access_token")
            sys.exit(1)
        
        # Initialize dhanhq client
        self.dhan = self._initialize_dhan_client()
        self.is_live_mode = False
        
        print("✅ Live Trading Engine initialized")
        
    def _initialize_dhan_client(self):
        """Initialize dhanhq client with correct method"""
        if NEW_VERSION:
            try:
                print("🔄 Using dhanhq v2.1.0+ with DhanContext...")
                dhan_context = DhanContext(self.client_id, self.access_token)
                client = dhanhq(dhan_context)
                print("✅ DhanContext initialization successful")
                return client
            except Exception as e:
                print(f"❌ DhanContext failed: {e}")
                return None
        else:
            try:
                print("🔄 Using legacy dhanhq...")
                client = dhanhq(self.client_id, self.access_token)
                print("✅ Legacy dhanhq initialization successful")
                return client
            except Exception as e:
                print(f"❌ Legacy dhanhq failed: {e}")
                return None
        
    def enable_live_trading(self):
        """Enable live trading mode"""
        self.is_live_mode = True
        logging.info("🔴 LIVE TRADING MODE ENABLED")
        print("🔴 LIVE TRADING MODE ENABLED")
        
    def disable_live_trading(self):
        """Disable live trading mode"""
        self.is_live_mode = False
        logging.info("📝 PAPER TRADING MODE ENABLED")
        print("📝 PAPER TRADING MODE ENABLED")
        
    def get_account_funds(self):
        """Get real-time account balance"""
        if not self.dhan:
            print("❌ No Dhan client available")
            return None
            
        try:
            print("📊 Fetching account funds...")
            funds = self.dhan.get_fund_limits()
            
            if funds:
                # Handle different response formats and field names
                if isinstance(funds, dict):
                    if 'data' in funds:
                        balance_data = funds['data']
                    else:
                        balance_data = funds
                else:
                    balance_data = funds
                
                # Use correct field names from Dhan API v2
                available = balance_data.get('availabelBalance', 0)  # Note: Dhan's API typo
                if available == 0:
                    available = balance_data.get('availableBalance', 0)  # Fallback
                
                total = balance_data.get('sodLimit', 0)
                used = balance_data.get('utilizedAmount', 0)
                
                fund_info = {
                    'available_balance': available,
                    'margin_used': used,
                    'total_balance': total,
                    'free_balance': available - used
                }
                
                print(f"💰 Available Balance: ₹{available:,.2f}")
                print(f"📊 Utilized Amount: ₹{used:,.2f}")
                print(f"💳 Total Balance: ₹{total:,.2f}")
                
                return fund_info
            else:
                print("❌ No fund data received")
                return None
                
        except Exception as e:
            logging.error(f"Fund fetch error: {e}")
            print(f"❌ Fund fetch error: {e}")
            return None
            
    def place_live_order(self, order_details: Dict[str, Any]):
        """Place actual live order"""
        if not self.is_live_mode:
            logging.info("📝 PAPER TRADE MODE - Order logged only")
            print("📝 PAPER TRADE MODE - Order logged only")
            return self._paper_trade(order_details)
        
        if not self.dhan:
            print("❌ No Dhan client available for live trading")
            return None
            
        try:
            print(f"🎯 Placing LIVE ORDER: {order_details}")
            
            response = self.dhan.place_order(
                security_id=order_details['security_id'],
                exchange_segment=order_details.get('exchange_segment', 'NSE_FNO'),
                transaction_type=order_details['transaction_type'],
                quantity=order_details['quantity'],
                order_type=order_details['order_type'],
                price=order_details.get('price', 0),
                product_type=order_details.get('product_type', 'INTRADAY'),
                validity=order_details.get('validity', 'DAY')
            )
            
            logging.info(f"🎯 LIVE ORDER PLACED: {response}")
            print(f"✅ LIVE ORDER PLACED: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Order placement failed: {e}")
            print(f"❌ Order placement failed: {e}")
            return None
    
    def _paper_trade(self, order_details: Dict[str, Any]):
        """Simulate paper trading"""
        paper_order = {
            'order_id': f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'status': 'COMPLETE',
            'message': 'Paper trade executed',
            'details': order_details
        }
        
        print(f"📝 PAPER TRADE: {paper_order}")
        return paper_order
            
    def modify_order(self, order_id: str, new_price: float):
        """Modify existing order"""
        if not self.dhan:
            print("❌ No Dhan client available")
            return None
            
        try:
            response = self.dhan.modify_order(
                order_id=order_id,
                price=new_price
            )
            logging.info(f"📝 ORDER MODIFIED: {order_id} to {new_price}")
            print(f"✅ ORDER MODIFIED: {order_id} to ₹{new_price}")
            return response
        except Exception as e:
            logging.error(f"Order modification failed: {e}")
            print(f"❌ Order modification failed: {e}")
            return None
            
    def cancel_order(self, order_id: str):
        """Cancel existing order"""
        if not self.dhan:
            print("❌ No Dhan client available")
            return None
            
        try:
            response = self.dhan.cancel_order(order_id=order_id)
            logging.info(f"❌ ORDER CANCELLED: {order_id}")
            print(f"✅ ORDER CANCELLED: {order_id}")
            return response
        except Exception as e:
            logging.error(f"Order cancellation failed: {e}")
            print(f"❌ Order cancellation failed: {e}")
            return None
            
    def get_positions(self):
        """Get current positions"""
        if not self.dhan:
            print("❌ No Dhan client available")
            return []
            
        try:
            print("📊 Fetching current positions...")
            positions = self.dhan.get_positions()
            
            if positions:
                print(f"📈 Found {len(positions)} positions")
                for pos in positions:
                    print(f"   • {pos}")
            else:
                print("📭 No active positions")
                
            return positions or []
            
        except Exception as e:
            logging.error(f"Position fetch error: {e}")
            print(f"❌ Position fetch error: {e}")
            return []
    
    def get_order_book(self):
        """Get order book"""
        if not self.dhan:
            print("❌ No Dhan client available")
            return []
            
        try:
            print("📋 Fetching order book...")
            orders = self.dhan.get_order_list()
            
            if orders:
                print(f"📋 Found {len(orders)} orders")
                for order in orders:
                    print(f"   • Order ID: {order.get('order_id', 'N/A')} - Status: {order.get('status', 'N/A')}")
            else:
                print("📭 No orders found")
                
            return orders or []
            
        except Exception as e:
            logging.error(f"Order book fetch error: {e}")
            print(f"❌ Order book fetch error: {e}")
            return []
    
    def test_connection(self):
        """Test API connection"""
        print("🔍 Testing API connection...")
        funds = self.get_account_funds()
        
        if funds:
            print("✅ API connection test successful!")
            return True
        else:
            print("❌ API connection test failed!")
            return False

def main():
    """Main function to test the trading engine"""
    print("🚀 LIVE TRADING ENGINE")
    print("=" * 50)
    
    try:
        # Create trading engine
        engine = LiveTradingEngine()
        
        # Test connection
        if engine.test_connection():
            print("\n📊 Testing various functions...")
            
            # Get account funds
            funds = engine.get_account_funds()
            
            # Get positions
            positions = engine.get_positions()
            
            # Get order book
            orders = engine.get_order_book()
            
            # Example paper trade
            sample_order = {
                'security_id': '52175',  # Example: Nifty option
                'transaction_type': 'BUY',
                'quantity': 25,
                'order_type': 'MARKET',
                'price': 0
            }
            
            print(f"\n📝 Testing paper trade...")
            engine.place_live_order(sample_order)
            
            print(f"\n🎯 To enable live trading, call: engine.enable_live_trading()")
            print(f"⚠️ WARNING: Only enable live trading when you're ready to place real orders!")
            
        else:
            print("❌ Connection test failed. Check your credentials.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Trading engine stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()