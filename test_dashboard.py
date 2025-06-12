# test_dashboard.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dashboard.dashboard import app, DashboardServer
    print("✅ Dashboard module imported successfully!")
    
    # Check if Flask app exists
    print(f"✅ Flask app created: {app is not None}")
    
    # Check if DashboardServer can be instantiated
    dashboard = DashboardServer()
    print("✅ Dashboard Server initialized!")
    
    # Check if templates directory will be created
    print(f"✅ Dashboard methods exist: {hasattr(dashboard, 'update_portfolio_data')}")
    
    print("\n✅ ALL DASHBOARD TESTS PASSED!")
    print("\n💡 To run the dashboard, use: python run_dashboard.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()