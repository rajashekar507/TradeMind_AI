"""
Test Market Data API Parameters
Find the correct parameter names for Dhan API methods
"""

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

def test_market_data_api():
    """Test different parameter formats for market data APIs"""
    
    load_dotenv()
    
    client_id = os.getenv('DHAN_CLIENT_ID')
    access_token = os.getenv('DHAN_ACCESS_TOKEN')
    
    if not client_id or not access_token:
        print("❌ Missing Dhan credentials")
        return
    
    try:
        # Initialize Dhan client
        from dhanhq import DhanContext, dhanhq
        dhan_context = DhanContext(client_id=client_id, access_token=access_token)
        dhan = dhanhq(dhan_context)
        print("✅ Dhan client initialized successfully")
        
        # Test parameters for NIFTY
        nifty_params = {
            'security_id': '13',
            'exchange_segment': 'IDX_I',
            'instrument_type': 'INDEX'
        }
        
        print("\n" + "="*60)
        print("🧪 Testing historical_daily_data with different parameters")
        print("="*60)
        
        # Test 1: With security_id parameter
        print("\n🔄 Test 1: historical_daily_data(security_id=...)")
        try:
            from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            response = dhan.historical_daily_data(
                security_id=nifty_params['security_id'],
                exchange_segment=nifty_params['exchange_segment'],
                instrument_type=nifty_params['instrument_type'],
                expiry_code=0,
                from_date=from_date,
                to_date=to_date
            )
            
            print(f"✅ SUCCESS: {type(response)}")
            if isinstance(response, dict):
                print(f"   Keys: {list(response.keys())}")
                if 'data' in response and response['data']:
                    print(f"   Data points: {len(response['data'])}")
                    print(f"   Sample: {response['data'][0] if response['data'] else 'No data'}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        # Test 2: With symbol parameter
        print("\n🔄 Test 2: historical_daily_data(symbol=...)")
        try:
            response = dhan.historical_daily_data(
                symbol=nifty_params['security_id'],
                exchange_segment=nifty_params['exchange_segment'],
                instrument_type=nifty_params['instrument_type'],
                expiry_code=0,
                from_date=from_date,
                to_date=to_date
            )
            print(f"✅ SUCCESS: {type(response)}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        print("\n" + "="*60)
        print("🧪 Testing quote_data with securities array")
        print("="*60)
        
        # Test 3: Quote data with securities array
        print("\n🔄 Test 3: quote_data(securities=[...])")
        try:
            securities = [{
                'security_id': nifty_params['security_id'],
                'exchange_segment': nifty_params['exchange_segment']
            }]
            
            response = dhan.quote_data(securities)
            print(f"✅ SUCCESS: {type(response)}")
            if isinstance(response, dict):
                print(f"   Keys: {list(response.keys())}")
                if 'data' in response and response['data']:
                    print(f"   Data points: {len(response['data'])}")
                    print(f"   Sample: {response['data'][0] if response['data'] else 'No data'}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        print("\n" + "="*60)
        print("🧪 Testing ohlc_data with securities array")
        print("="*60)
        
        # Test 4: OHLC data with securities array
        print("\n🔄 Test 4: ohlc_data(securities=[...])")
        try:
            securities = [{
                'security_id': nifty_params['security_id'],
                'exchange_segment': nifty_params['exchange_segment']
            }]
            
            response = dhan.ohlc_data(securities)
            print(f"✅ SUCCESS: {type(response)}")
            if isinstance(response, dict):
                print(f"   Keys: {list(response.keys())}")
                if 'data' in response and response['data']:
                    print(f"   Data points: {len(response['data'])}")
                    print(f"   Sample: {response['data'][0] if response['data'] else 'No data'}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        print("\n" + "="*60)
        print("🧪 Testing intraday_minute_data")
        print("="*60)
        
        # Test 5: Intraday minute data
        print("\n🔄 Test 5: intraday_minute_data(security_id=...)")
        try:
            from_time = datetime.now().strftime('%Y-%m-%d') + ' 09:15:00'
            to_time = datetime.now().strftime('%Y-%m-%d') + ' 15:30:00'
            
            response = dhan.intraday_minute_data(
                security_id=nifty_params['security_id'],
                exchange_segment=nifty_params['exchange_segment'],
                instrument_type=nifty_params['instrument_type'],
                expiry_code=0,
                from_date=from_time,
                to_date=to_time,
                interval='1'
            )
            
            print(f"✅ SUCCESS: {type(response)}")
            if isinstance(response, dict):
                print(f"   Keys: {list(response.keys())}")
                if 'data' in response and response['data']:
                    print(f"   Data points: {len(response['data'])}")
                    print(f"   Sample: {response['data'][0] if response['data'] else 'No data'}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        print("\n" + "="*60)
        print("✅ Test Complete!")
        print("Use the successful parameter format in your dashboard.")
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")

if __name__ == "__main__":
    test_market_data_api()