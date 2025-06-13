"""
Test Dhan API Initialization Methods
Find the correct way to initialize your Dhan client
"""

import os
from dotenv import load_dotenv

def test_dhan_initialization():
    """Test different ways to initialize Dhan client"""
    
    load_dotenv()
    
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        print("❌ Missing Dhan credentials in .env file")
        return
    
    print(f"🔑 Client ID: {client_id[:10]}...")
    print(f"🔑 Access Token: {access_token[:20]}...")
    print()
    
    # Method 1: Try dhanhq with access token only
    print("🧪 Method 1: dhanhq(access_token)")
    try:
        from dhanhq import dhanhq
        dhan = dhanhq(access_token)
        print("✅ SUCCESS - dhanhq(access_token) works!")
        
        # Test if client works
        try:
            # Test basic functionality
            if hasattr(dhan, 'get_fund_limits'):
                fund_limits = dhan.get_fund_limits()
                print(f"   ✅ Fund limits call successful: {type(fund_limits)}")
            else:
                print("   ⚠️ get_fund_limits method not available")
        except Exception as e:
            print(f"   ⚠️ Fund limits call failed: {e}")
        
        return dhan
        
    except Exception as e:
        print(f"❌ FAILED - dhanhq(access_token): {e}")
    
    # Method 2: Try dhanhq with both parameters
    print("\n🧪 Method 2: dhanhq(client_id, access_token)")
    try:
        from dhanhq import dhanhq
        dhan = dhanhq(client_id, access_token)
        print("✅ SUCCESS - dhanhq(client_id, access_token) works!")
        return dhan
    except Exception as e:
        print(f"❌ FAILED - dhanhq(client_id, access_token): {e}")
    
    # Method 3: Try dhanhq with named parameters
    print("\n🧪 Method 3: dhanhq(client_id=..., access_token=...)")
    try:
        from dhanhq import dhanhq
        dhan = dhanhq(client_id=client_id, access_token=access_token)
        print("✅ SUCCESS - Named parameters work!")
        return dhan
    except Exception as e:
        print(f"❌ FAILED - Named parameters: {e}")
    
    # Method 4: Try DhanHQ class
    print("\n🧪 Method 4: DhanHQ(client_id, access_token)")
    try:
        from dhanhq import DhanHQ
        dhan = DhanHQ(client_id, access_token)
        print("✅ SUCCESS - DhanHQ class works!")
        return dhan
    except Exception as e:
        print(f"❌ FAILED - DhanHQ class: {e}")
    
    # Method 5: Try DhanContext
    print("\n🧪 Method 5: DhanContext approach")
    try:
        from dhanhq import DhanContext, dhanhq
        context = DhanContext(client_id=client_id, access_token=access_token)
        dhan = dhanhq(context)
        print("✅ SUCCESS - DhanContext approach works!")
        return dhan
    except Exception as e:
        print(f"❌ FAILED - DhanContext: {e}")
    
    print("\n❌ All initialization methods failed!")
    return None

def test_working_client(dhan):
    """Test the working client"""
    if not dhan:
        print("❌ No working client to test")
        return
    
    print("\n" + "="*50)
    print("🧪 Testing Working Client")
    print("="*50)
    
    # Get all available methods
    methods = [method for method in dir(dhan) if not method.startswith('_')]
    print(f"📋 Available methods ({len(methods)}):")
    
    # Categorize methods
    data_methods = []
    trading_methods = []
    
    for method in methods:
        method_lower = method.lower()
        if any(keyword in method_lower for keyword in ['data', 'historical', 'intraday', 'quote', 'ltp', 'ohlc']):
            data_methods.append(method)
        elif any(keyword in method_lower for keyword in ['order', 'position', 'holding', 'fund', 'balance']):
            trading_methods.append(method)
    
    print(f"\n📊 DATA METHODS ({len(data_methods)}):")
    for method in data_methods:
        print(f"   ✅ {method}")
    
    print(f"\n💰 TRADING METHODS ({len(trading_methods)}):")
    for method in trading_methods:
        print(f"   ✅ {method}")
    
    # Test key methods
    print(f"\n🚀 Testing Key Methods:")
    
    # Test fund limits
    if hasattr(dhan, 'get_fund_limits'):
        try:
            result = dhan.get_fund_limits()
            print(f"✅ get_fund_limits: {type(result)}")
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
        except Exception as e:
            print(f"❌ get_fund_limits failed: {e}")
    
    # Test historical data if available
    if hasattr(dhan, 'historical_daily_data'):
        try:
            result = dhan.historical_daily_data
            print(f"✅ historical_daily_data method exists: {type(result)}")
        except Exception as e:
            print(f"❌ historical_daily_data error: {e}")

if __name__ == "__main__":
    print("🧠 TradeMind AI - Dhan Initialization Test")
    print("="*50)
    
    working_client = test_dhan_initialization()
    test_working_client(working_client)
    
    print("\n" + "="*50)
    print("✅ Test Complete!")
    print("Use the successful method in your dashboard.")