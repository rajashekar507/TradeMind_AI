# test_debug.py
import sys
import os

print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}")

# Check if config folder exists
if os.path.exists('config'):
    print("✅ config folder exists")
    if os.path.exists('config/__init__.py'):
        print("✅ config/__init__.py exists")
    if os.path.exists('config/constants.py'):
        print("✅ config/constants.py exists")
else:
    print("❌ config folder NOT found")

# Try import
try:
    from config.constants import LOT_SIZES
    print("✅ Import successful!")
    print(f"NIFTY: {LOT_SIZES['NIFTY']}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()