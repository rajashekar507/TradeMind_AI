#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('/home/ubuntu/trading_system')

from src.analysis.backtesting import BacktestingEngine
from src.auth.kite_auth import KiteAuthenticator
from src.config.settings import Settings
from datetime import datetime, timedelta

async def debug_historical_data():
    """Debug historical data fetching for backtesting"""
    print("🔍 DEBUGGING HISTORICAL DATA FETCHING")
    print("=" * 50)
    
    try:
        settings = Settings()
        kite_auth = KiteAuthenticator(settings)
        kite_client = kite_auth.get_authenticated_kite()
        
        if not kite_client:
            print("❌ Failed to get Kite client")
            return
        
        print("✅ Kite client authenticated")
        
        engine = BacktestingEngine(kite_client=kite_client)
        
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now() - timedelta(days=1)
        
        print(f"📅 Fetching historical data from {start_date.date()} to {end_date.date()}")
        
        historical_data = await engine._fetch_historical_options_data('NIFTY', start_date, end_date)
        
        if historical_data is None:
            print("❌ No historical data returned")
        elif len(historical_data) == 0:
            print("❌ Empty historical data returned")
        else:
            print(f"✅ Historical data fetched: {len(historical_data)} records")
            print(f"   Date range: {historical_data['date'].min()} to {historical_data['date'].max()}")
            print(f"   Columns: {list(historical_data.columns)}")
            print(f"   Sample data:")
            print(historical_data.head())
        
        print("\n🔄 Testing full backtest...")
        result = await engine.run_backtest('test_strategy', 'NIFTY', start_date, end_date)
        
        print(f"📊 Backtest result:")
        print(f"   Status: {result.get('status')}")
        print(f"   Total trades: {result.get('total_trades', 0)}")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_historical_data())
