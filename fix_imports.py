"""
Fix all import errors in TradeMind_AI project
Run this script to automatically fix all import paths
"""

import os
import re

def fix_imports_in_file(filepath):
    """Fix imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Dictionary of import fixes
        import_fixes = {
            # Fix imports from root files to src files
            'from smart_trader import': 'from src.core.smart_trader import',
            'from ai_trader import': 'from src.core.ai_trader import',
            'from master_trader import': 'from src.core.master_trader import',
            'from unified_trading_engine import': 'from src.core.unified_trading_engine import',
            
            # Fix analysis imports
            'from exact_recommender import': 'from src.analysis.exact_recommender import',
            'from technical_indicators import': 'from src.analysis.technical_indicators import',
            'from greeks_calculator import': 'from src.analysis.greeks_calculator import',
            'from multi_timeframe_analyzer import': 'from src.analysis.multi_timeframe_analyzer import',
            'from ml_trader import': 'from src.analysis.ml_trader import',
            'from oi_tracker import': 'from src.analysis.oi_tracker import',
            'from news_sentiment import': 'from src.analysis.news_sentiment import',
            'from historical_data import': 'from src.analysis.historical_data import',
            'from real_news_analyzer import': 'from src.analysis.real_news_analyzer import',
            
            # Fix data imports
            'from market_data import': 'from src.data.market_data import',
            'from Balance_fetcher import': 'from src.data.Balance_fetcher import',
            'from real_historical_data import': 'from src.data.real_historical_data import',
            
            # Fix portfolio imports
            'from portfolio_manager import': 'from src.portfolio.portfolio_manager import',
            'from live_trading_engine import': 'from src.portfolio.live_trading_engine import',
            
            # Fix utils imports
            'from auto_scheduler import': 'from src.utils.auto_scheduler import',
            'from excel_reporter import': 'from src.utils.excel_reporter import',
            'from rate_limiter import': 'from src.utils.rate_limiter import',
            'from test import': 'from src.utils.test import',
            
            # Fix config imports - special handling
            'from config.constants import': 'from config.constants import',
            'import config.constants': 'import config.constants',
            
            # Fix error_handler - it's in analysis folder
            'from error_handler import': 'from src.analysis.error_handler import',
            
            # Fix global_market_analyzer
            'from global_market_analyzer import': 'from src.analysis.global_market_analyzer import',
            
            # Fix dashboard imports
            'from dashboard_server import': 'from src.dashboard.dashboard_server import',
            'from dashboard_template import': 'from src.dashboard.dashboard_template import',
        }
        
        # Apply all fixes
        for old_import, new_import in import_fixes.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                print(f"  ✓ Fixed: {old_import} → {new_import}")
        
        # Special case: Fix relative imports in files that are already in src
        if 'src' in filepath:
            # For files in src/core trying to import from analysis
            content = re.sub(
                r'from \.\. import (\w+)',
                r'from src.analysis import \1',
                content
            )
            
            # For files in src folders trying to import from same folder
            content = re.sub(
                r'from \. import (\w+)',
                lambda m: f'from {get_module_path(filepath)} import {m.group(1)}',
                content
            )
        
        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {filepath}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def get_module_path(filepath):
    """Get the module path from filepath"""
    # Convert filepath to module path
    parts = filepath.replace('\\', '/').split('/')
    
    # Find where 'src' is in the path
    if 'src' in parts:
        src_index = parts.index('src')
        module_parts = parts[src_index:-1]  # Exclude the filename
        return '.'.join(module_parts)
    
    return ''

def main():
    print("🔧 TradeMind_AI Import Fixer")
    print("=" * 50)
    
    # Count files
    total_files = 0
    fixed_files = 0
    
    # Process all Python files
    for root, dirs, files in os.walk('.'):
        # Skip __pycache__ and backup directories
        if '__pycache__' in root or 'TradeMind_AI_Backup' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file != 'fix_imports.py':
                filepath = os.path.join(root, file)
                total_files += 1
                
                print(f"\n📄 Processing: {filepath}")
                if fix_imports_in_file(filepath):
                    fixed_files += 1
    
    print("\n" + "=" * 50)
    print(f"✅ COMPLETE!")
    print(f"📊 Total files processed: {total_files}")
    print(f"🔧 Files fixed: {fixed_files}")
    print("\n💡 Next steps:")
    print("1. Run 'python run_trading.py' to test if imports work")
    print("2. If you see any remaining import errors, tell me!")
    
if __name__ == "__main__":
    main()