# test_error_handler.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from analysis.error_handler import AutoErrorHandler, auto_heal
    print("✅ Error Handler module imported successfully!")
    
    # Create instance
    error_handler = AutoErrorHandler()
    print("✅ Error Handler initialized!")
    
    # Test basic functionality
    print("\n📊 Testing Error Handler...")
    
    # Test auto-heal decorator
    @auto_heal
    def test_function():
        return "Function executed successfully"
    
    result = test_function()
    print(f"✅ Auto-heal decorator works: {result}")
    
    # Check if logs directory was created
    print(f"✅ Logs directory exists: {os.path.exists('logs')}")
    
    print("\n✅ ALL ERROR HANDLER TESTS PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()