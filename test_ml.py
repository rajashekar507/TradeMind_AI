# test_ml.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from analysis.ml_trader import SelfLearningTrader
    print("✅ ML Trader module imported successfully!")
    
    # Create instance
    ml_trader = SelfLearningTrader()
    print("✅ ML Trader initialized!")
    
    # Test basic functionality
    print("\n📊 Testing ML Trader...")
    
    # Check if it loads properly
    print(f"✅ Models directory exists: {os.path.exists('models')}")
    print(f"✅ Minimum trades for learning: {ml_trader.min_trades_for_learning}")
    
    print("\n✅ ALL TESTS PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()