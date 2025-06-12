"""
Fix master_trader.py to use unified_trading_engine instead of old imports
"""

import os

def fix_master_trader():
    """Fix imports in master_trader.py"""
    
    # Find master_trader.py in src/core
    master_trader_path = os.path.join('src', 'core', 'master_trader.py')
    
    if not os.path.exists(master_trader_path):
        print(f"❌ Could not find {master_trader_path}")
        return False
    
    print(f"📄 Fixing {master_trader_path}...")
    
    try:
        # Read the file
        with open(master_trader_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Store original for comparison
        original_content = content
        
        # Fix the imports
        # Remove old imports
        old_imports = [
            '# Removed smart_trader import - using unified_trading_engine
            '# Removed smart_trader import - using unified_trading_engine
            '# Removed ai_trader import - using unified_trading_engine
            '# Removed ai_trader import - using unified_trading_engine
        ]
        
        for old_import in old_imports:
            if old_import in content:
                content = content.replace(old_import, '# ' + old_import + ' # Removed - using unified engine')
                print(f"  ✓ Commented out: {old_import}")
        
        # Update the class initialization
        # Replace smart_trader initialization
        content = content.replace(
            'self.smart_trader = SmartUnifiedTradingEngine(TradingMode.PAPER)',
            'from src.core.unified_trading_engine import UnifiedTradingEngine, TradingMode\n        self.smart_trader = UnifiedTradingEngine(TradingMode.PAPER)'
        )
        
        # Fix any remaining SmartUnifiedTradingEngine references
        content = content.replace('SmartUnifiedTradingEngine(TradingMode.PAPER)', 'UnifiedTradingEngine(TradingMode.PAPER)')
        
        # Write back only if changed
        if content != original_content:
            with open(master_trader_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed master_trader.py successfully!")
            return True
        else:
            print("ℹ️ No changes needed in master_trader.py")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔧 Fixing Master Trader imports...")
    print("=" * 50)
    
    if fix_master_trader():
        print("\n✅ SUCCESS! Master trader is now using unified_trading_engine")
        print("\n💡 Next step: Test if everything works by running:")
        print("   python run_trading.py")
    else:
        print("\n❌ Failed to fix master_trader.py")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()