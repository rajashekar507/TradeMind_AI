"""
Discover Real Dhan API Methods - Test Script
This will find the actual methods available in your dhanhq library
"""

import os
import sys
from dotenv import load_dotenv

def discover_dhan_methods():
    """Discover all available methods in the dhanhq library"""
    
    load_dotenv()
    
    try:
        from dhanhq import dhanhq
        
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not client_id or not access_token:
            print("❌ Missing Dhan credentials!")
            return
        
        # Initialize Dhan client with correct syntax
        try:
            # Method 1: Try access token only
            dhan = dhanhq(access_token)
            print("✅ Dhan client initialized with access token only")
        except:
            try:
                # Method 2: Try named parameters
                dhan = dhanhq(client_id=client_id, access_token=access_token)
                print("✅ Dhan client initialized with named parameters")
            except:
                try:
                    # Method 3: Try DhanHQ class
                    from dhanhq import DhanHQ
                    dhan = DhanHQ(client_id, access_token)
                    print("✅ Dhan client initialized with DhanHQ class")
                except Exception as e:
                    print(f"❌ All initialization methods failed: {e}")
                    return
        
        print("🔍 Discovering all available Dhan API methods...")
        print("=" * 60)
        
        # Get all public methods (excluding private methods starting with _)
        all_methods = [method for method in dir(dhan) if not method.startswith('_')]
        
        print(f"📋 Found {len(all_methods)} public methods:")
        print("-" * 40)
        
        # Categorize methods
        data_methods = []
        trading_methods = []
        other_methods = []
        
        for method in all_methods:
            method_lower = method.lower()
            
            if any(keyword in method_lower for keyword in ['data', 'historical', 'intraday', 'quote', 'ltp', 'ohlc', 'market', 'feed']):
                data_methods.append(method)
            elif any(keyword in method_lower for keyword in ['order', 'position', 'holding', 'trade', 'fund', 'balance']):
                trading_methods.append(method)
            else:
                other_methods.append(method)
        
        print("📊 DATA/MARKET METHODS:")
        for method in data_methods:
            print(f"   ✅ {method}")
        
        print(f"\n💰 TRADING METHODS:")
        for method in trading_methods:
            print(f"   ✅ {method}")
        
        print(f"\n🔧 OTHER METHODS:")
        for method in other_methods:
            print(f"   ✅ {method}")
        
        print("\n" + "=" * 60)
        print("🧪 Testing Key Methods for Market Data...")
        print("-" * 40)
        
        # Test methods that might work for market data
        test_methods = [
            'intraday_minute_data',
            'historical_daily_data', 
            'quote_data',
            'ohlc_data',
            'ltp_data',
            'get_ltp_data',
            'get_ohlc_data',
            'get_quote_data',
            'market_quote',
            'ticker_data'
        ]
        
        working_methods = []
        
        for method_name in test_methods:
            if hasattr(dhan, method_name):
                print(f"✅ {method_name} - Method exists")
                working_methods.append(method_name)
                
                # Try to get method signature
                try:
                    method = getattr(dhan, method_name)
                    print(f"   📝 Type: {type(method)}")
                    
                    # Try to get docstring
                    if hasattr(method, '__doc__') and method.__doc__:
                        doc = method.__doc__.strip()
                        if doc:
                            print(f"   📖 Doc: {doc[:100]}...")
                    
                except Exception as e:
                    print(f"   ⚠️ Error getting info: {e}")
            else:
                print(f"❌ {method_name} - Method not found")
        
        print(f"\n🎯 WORKING METHODS FOR TESTING: {working_methods}")
        
        # Test the working methods with actual API calls
        if working_methods:
            print("\n" + "=" * 60)
            print("🚀 Testing Working Methods with Real API Calls...")
            print("-" * 40)
            
            # Test parameters for NIFTY
            test_params = {
                'symbol': '13',
                'security_id': '13',
                'exchange_segment': 'IDX_I',
                'instrument_type': 'INDEX',
                'from_date': '2025-06-12',
                'to_date': '2025-06-13',
                'interval': '1'
            }
            
            for method_name in working_methods[:3]:  # Test first 3 methods to avoid rate limits
                try:
                    print(f"\n🔄 Testing {method_name}...")
                    method = getattr(dhan, method_name)
                    
                    # Try different parameter combinations
                    if 'historical' in method_name.lower():
                        result = method(
                            symbol=test_params['symbol'],
                            exchange_segment=test_params['exchange_segment'],
                            instrument_type=test_params['instrument_type'],
                            expiry_code=0,
                            from_date=test_params['from_date'],
                            to_date=test_params['to_date']
                        )
                    elif 'intraday' in method_name.lower():
                        result = method(
                            symbol=test_params['symbol'],
                            exchange_segment=test_params['exchange_segment'],
                            instrument_type=test_params['instrument_type'],
                            expiry_code=0,
                            interval=test_params['interval'],
                            from_date=test_params['from_date'] + ' 09:15:00',
                            to_date=test_params['to_date'] + ' 15:30:00'
                        )
                    else:
                        # Try simple call
                        try:
                            result = method()
                        except:
                            # Try with symbol parameter
                            result = method(symbol=test_params['symbol'])
                    
                    if result:
                        print(f"✅ {method_name} - SUCCESS!")
                        print(f"   📊 Response type: {type(result)}")
                        if isinstance(result, dict):
                            print(f"   🔑 Keys: {list(result.keys())}")
                            if 'status' in result:
                                print(f"   📈 Status: {result['status']}")
                            if 'data' in result:
                                print(f"   📄 Data type: {type(result['data'])}")
                                if isinstance(result['data'], list) and len(result['data']) > 0:
                                    print(f"   📄 Sample data: {result['data'][0]}")
                    else:
                        print(f"⚠️ {method_name} - No response")
                
                except Exception as e:
                    print(f"❌ {method_name} - Error: {str(e)[:100]}")
        
        print("\n" + "=" * 60)
        print("✅ Discovery Complete!")
        print("Copy the working methods to use in your dashboard.")
        
    except Exception as e:
        print(f"❌ Error in discovery: {e}")

if __name__ == "__main__":
    discover_dhan_methods()