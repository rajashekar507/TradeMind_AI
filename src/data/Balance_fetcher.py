"""
Fixed Dhan Account Balance Fetcher for dhanhq v2.1.0+
Compatible with the latest DhanContext initialization method
"""

import os
import sys
from datetime import datetime
import json

# Try to import dhanhq with new v2.1.0+ imports
try:
    from dhanhq import DhanContext, dhanhq
    print("✅ dhanhq v2.1.0+ library found")
    NEW_VERSION = True
except ImportError:
    try:
        from dhanhq import dhanhq
        print("✅ dhanhq legacy version found")
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

class DhanBalanceFetcher:
    def __init__(self):
        """Initialize Dhan connection with proper context"""
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not self.client_id or not self.access_token:
            print("❌ DHAN CREDENTIALS NOT FOUND!")
            print("📝 Please add to your .env file:")
            print("DHAN_CLIENT_ID=your_client_id")
            print("DHAN_ACCESS_TOKEN=your_access_token")
            print("\n🔗 Get credentials from: https://web.dhan.co")
            print("💡 Go to Profile > DhanHQ Trading APIs")
            sys.exit(1)
            
        # Initialize Dhan client based on version
        self.dhan = self._initialize_dhan_client()
    
    def _initialize_dhan_client(self):
        """Initialize Dhan client with correct method for version"""
        print(f"🔄 Initializing Dhan client...")
        print(f"🆔 Client ID: {self.client_id}")
        print(f"🔑 Access Token: {self.access_token[:20]}...")
        
        if NEW_VERSION:
            # dhanhq v2.1.0+ with DhanContext
            try:
                print("🔄 Using dhanhq v2.1.0+ with DhanContext...")
                dhan_context = DhanContext(self.client_id, self.access_token)
                client = dhanhq(dhan_context)
                print("✅ Success: DhanContext initialization")
                return client
            except Exception as e:
                print(f"❌ DhanContext failed: {e}")
                # Fall back to direct API call
                return self._try_direct_api_call()
        else:
            # Legacy dhanhq versions
            try:
                print("🔄 Using legacy dhanhq version...")
                client = dhanhq(self.client_id, self.access_token)
                print("✅ Success: Legacy dhanhq initialization")
                return client
            except Exception as e:
                print(f"❌ Legacy dhanhq failed: {e}")
                # Fall back to direct API call
                return self._try_direct_api_call()
    
    def _try_direct_api_call(self):
        """Try direct REST API call to Dhan with correct headers"""
        import requests
        
        try:
            print("🌐 Attempting direct API call to Dhan...")
            
            # Correct headers as per v2 documentation
            headers = {
                'access-token': self.access_token,
                'client-id': self.client_id,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Correct v2 API endpoint for fund limits
            url = 'https://api.dhan.co/v2/fundlimit'
            
            response = requests.get(url, headers=headers, timeout=10)
            
            print(f"📋 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Direct API call successful!")
                data = response.json()
                
                print("\n💰 ACCOUNT BALANCE (Direct API):")
                print("=" * 50)
                
                if 'data' in data:
                    balance_data = data['data']
                    self._display_balance_data(balance_data)
                    
                    # Save data with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"dhan_balance_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"💾 Data saved to: {filename}")
                    
                else:
                    print("📊 Raw API Response:")
                    print(json.dumps(data, indent=2))
                    
                return "DIRECT_API_SUCCESS"
                
            elif response.status_code == 401:
                print("❌ Authentication failed! Check your access token")
                print("💡 Token may have expired. Generate new token from Dhan web app")
                return None
                
            elif response.status_code == 400:
                print("❌ Bad request! Check your client ID and headers")
                print(f"📋 Response: {response.text}")
                return None
                
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"📋 Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Direct API call failed: {e}")
            return None
    
    def _display_balance_data(self, balance_data):
        """Display balance data in formatted way"""
        # Note: Dhan API has a typo - 'availabelBalance' (missing 'i')
        available = balance_data.get('availabelBalance', 0)  # Dhan's API typo
        if available == 0:
            available = balance_data.get('availableBalance', 0)  # Fallback
        total = balance_data.get('sodLimit', 0)
        used = balance_data.get('utilizedAmount', 0)
        
        print(f"💵 Available Balance: ₹{available:,.2f}")
        print(f"💳 Start of Day Limit: ₹{total:,.2f}")
        print(f"📊 Utilized Amount: ₹{used:,.2f}")
        print(f"🔓 Free Balance: ₹{available - used:,.2f}")
        
        # Additional balance details
        collateral = balance_data.get('collateralAmount', 0)
        receivable = balance_data.get('receiveableAmount', 0)  # Another API typo
        if receivable == 0:
            receivable = balance_data.get('receivableAmount', 0)  # Fallback
        withdrawable = balance_data.get('withdrawableBalance', 0)
        blocked = balance_data.get('blockedPayoutAmount', 0)
        
        print(f"💰 Collateral Amount: ₹{collateral:,.2f}")
        print(f"📥 Receivable Amount: ₹{receivable:,.2f}")
        print(f"💸 Withdrawable Balance: ₹{withdrawable:,.2f}")
        print(f"🚫 Blocked Payout: ₹{blocked:,.2f}")
        
        # Trading capacity analysis
        if available > 0:
            print(f"\n🎯 TRADING CAPACITY:")
            print(f"   • Conservative (0.5% risk): ₹{available * 0.005:,.2f} per trade")
            print(f"   • Options Premium: ₹{available:,.2f}")
            print(f"   • Intraday (5x leverage): ₹{available * 5:,.2f}")
            
        # Trading readiness status
        if available > 10000:
            print(f"\n🟢 TRADING READY: Sufficient balance for trading")
        elif available > 1000:
            print(f"\n🟡 LIMITED TRADING: Low balance - consider adding funds")
        else:
            print(f"\n🔴 INSUFFICIENT BALANCE: Add funds to start trading")
            
        print("=" * 50)
    
    def fetch_fund_limits(self):
        """Fetch fund limits using working method"""
        if self.dhan == "DIRECT_API_SUCCESS":
            print("✅ Balance already fetched via direct API")
            return True
        elif self.dhan:
            try:
                print("\n📊 Fetching fund limits via dhanhq...")
                funds = self.dhan.get_fund_limits()
                
                if funds:
                    print("✅ Fund limits fetched successfully!")
                    self.display_fund_summary(funds)
                    return funds
                else:
                    print("❌ No fund data received")
                    return None
                    
            except Exception as e:
                print(f"❌ Error fetching fund limits: {e}")
                print("💡 Falling back to direct API call...")
                return self._try_direct_api_call()
        else:
            print("❌ No working Dhan connection available")
            return None
    
    def display_fund_summary(self, funds):
        """Display fund summary in a readable format"""
        if not funds:
            return
            
        print("\n" + "="*60)
        print("💰 DHAN ACCOUNT BALANCE SUMMARY")
        print("="*60)
        
        # Handle different response formats
        if isinstance(funds, dict):
            if 'data' in funds:
                balance_data = funds['data']
            else:
                balance_data = funds
        else:
            balance_data = funds
        
        print(f"\n🔍 Raw API Response Keys: {list(balance_data.keys())}")
        self._display_balance_data(balance_data)
    
    def test_api_connection(self):
        """Test API connection with profile endpoint"""
        import requests
        
        try:
            print("🔍 Testing API connection with profile endpoint...")
            
            headers = {
                'access-token': self.access_token,
                'client-id': self.client_id,
                'Content-Type': 'application/json'
            }
            
            url = 'https://api.dhan.co/v2/profile'
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ API connection test successful!")
                print(f"🆔 Client ID: {data.get('dhanClientId', 'N/A')}")
                print(f"⏰ Token Validity: {data.get('tokenValidity', 'N/A')}")
                print(f"📈 Active Segments: {data.get('activeSegment', 'N/A')}")
                return True
            else:
                print(f"❌ API connection test failed: {response.status_code}")
                print(f"📋 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API connection test error: {e}")
            return False
    
    def run_balance_check(self):
        """Run complete balance check with connection test"""
        print("🚀 DHAN ACCOUNT BALANCE CHECKER v2.1.0")
        print("=" * 50)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test API connection first
        if self.test_api_connection():
            print("\n📊 Proceeding to fetch balance...")
            # Fetch and display balance
            result = self.fetch_fund_limits()
            
            if result:
                print(f"\n✅ Balance check completed successfully!")
            else:
                print(f"\n❌ Balance check failed!")
        else:
            print(f"\n❌ API connection failed. Please check credentials.")
        
        print(f"\n💡 If you see auth errors, regenerate your access token from:")
        print(f"   https://web.dhan.co -> Profile -> DhanHQ Trading APIs")

def main():
    """Main function"""
    try:
        # Create balance fetcher
        fetcher = DhanBalanceFetcher()
        
        # Run balance check
        fetcher.run_balance_check()
        
    except KeyboardInterrupt:
        print("\n⏹️ Balance check interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check your Dhan credentials and internet connection")
        print("🔄 Consider updating dhanhq: pip install --upgrade dhanhq")

if __name__ == "__main__":
    main()