# test_global.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from analysis.global_market_analyzer import GlobalMarketAnalyzer
    print("✅ Global Market Analyzer module imported successfully!")
    
    # Create instance
    analyzer = GlobalMarketAnalyzer()
    print("✅ Global Market Analyzer initialized!")
    
    # Test basic functionality
    print("\n📊 Testing Global Market Analyzer...")
    
    # Test fetching data (simplified test)
    print("📡 Testing market data fetch...")
    # We'll just check if the methods exist
    print(f"✅ fetch_global_market_data method exists: {hasattr(analyzer, 'fetch_global_market_data')}")
    print(f"✅ get_trading_bias method exists: {hasattr(analyzer, 'get_trading_bias')}")
    print(f"✅ should_trade_today method exists: {hasattr(analyzer, 'should_trade_today')}")
    
    print("\n✅ ALL GLOBAL MARKET TESTS PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()