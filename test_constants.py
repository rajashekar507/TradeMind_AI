# test_constants.py
try:
    from config.constants import LOT_SIZES
    print("✅ Import successful!")
    print(f"NIFTY lot size: {LOT_SIZES['NIFTY']}")
    print(f"BANKNIFTY lot size: {LOT_SIZES['BANKNIFTY']}")
except Exception as e:
    print(f"❌ Import failed: {e}")