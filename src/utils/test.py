# Test TradeMind_AI Setup - Fixed Version
print("🚀 TradeMind_AI VSCode Setup Test")
print("✅ Python is working!")

# Test core imports
try:
    import pandas as pd
    print("✅ Pandas imported successfully")
    
    import numpy as np
    print("✅ Numpy imported successfully")
    
    from dhanhq import dhanhq
    print("✅ Dhan API imported successfully")
    
    import requests
    print("✅ Requests imported successfully")
    
    import yfinance as yf
    print("✅ Yahoo Finance imported successfully")
    
    # Test technical analysis import (with fallback)
    ta_available = False
    try:
        import finta
        print("✅ Technical Analysis (Finta) imported successfully")
        ta_available = True
    except ImportError:
        try:
            import pandas_ta as ta
            print("✅ Technical Analysis (pandas_ta) imported successfully")
            ta_available = True
        except ImportError:
            print("⚠️ Technical Analysis library not available")
    
    # Test other essential libraries
    import json
    print("✅ JSON imported successfully")
    
    import time
    print("✅ Time imported successfully")
    
    from datetime import datetime
    print("✅ DateTime imported successfully")
    
    print(f"\n🎉 ALL CORE SYSTEMS READY FOR TRADING AI!")
    print(f"📊 Technical Analysis: {'✅ Available' if ta_available else '⚠️ Will install later'}")
    print("🚀 Ready to build profitable trading strategies!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")