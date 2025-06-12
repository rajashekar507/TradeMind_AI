"""
Find Code Duplications in TradeMind AI
Scans all Python files for duplicate definitions
"""

import os
import re
from pathlib import Path

def find_duplications():
    """Find duplicate code patterns across the project"""
    print("🔍 Scanning for Code Duplications...")
    print("="*60)
    
    # Patterns to search for
    patterns = {
        'LOT_SIZES': r'LOT_SIZES\s*=\s*{',
        'NIFTY.*75': r'["\']NIFTY["\'].*75',
        'BANKNIFTY.*30': r'["\']BANKNIFTY["\'].*30',
        'get_lot_size': r'def get_lot_size',
        'telegram_token': r'telegram_token\s*=',
        'send.*alert': r'def send.*alert',
        'fetch.*balance': r'def fetch.*balance'
    }
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.env', 'venv', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(python_files)} Python files to scan")
    print()
    
    # Track duplications
    duplications = {}
    for pattern_name, pattern in patterns.items():
        duplications[pattern_name] = []
    
    # Scan each file
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern_name, pattern in patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    duplications[pattern_name].append({
                        'file': file_path,
                        'line': line_num,
                        'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group()
                    })
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    
    # Report duplications
    print("📊 DUPLICATION ANALYSIS RESULTS:")
    print("="*60)
    
    for pattern_name, occurrences in duplications.items():
        if len(occurrences) > 1:  # Only show if duplicated
            print(f"\n🔴 DUPLICATE: {pattern_name} ({len(occurrences)} occurrences)")
            for i, occ in enumerate(occurrences, 1):
                print(f"   {i}. {occ['file']}:{occ['line']} - {occ['match']}")
        elif len(occurrences) == 1:
            print(f"\n🟢 UNIQUE: {pattern_name} (1 occurrence)")
            print(f"   ✅ {occurrences[0]['file']}:{occurrences[0]['line']}")
        else:
            print(f"\n⚪ NOT FOUND: {pattern_name}")
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDED ACTIONS:")
    
    # Analyze LOT_SIZES specifically
    lot_sizes_files = [occ['file'] for occ in duplications['LOT_SIZES']]
    if len(lot_sizes_files) > 1:
        print(f"\n1️⃣ LOT_SIZES CONSOLIDATION:")
        print(f"   📍 Keep: config/constants.py (master definition)")
        print(f"   🗑️ Remove from:")
        for file in lot_sizes_files:
            if 'constants.py' not in file:
                print(f"      - {file}")
        print(f"   🔧 Add imports: from config.constants import LOT_SIZES, get_lot_size")
    
    # Check for other duplications
    duplicate_count = sum(1 for occs in duplications.values() if len(occs) > 1)
    if duplicate_count > 0:
        print(f"\n📈 TOTAL DUPLICATIONS FOUND: {duplicate_count}")
        print("💡 Each duplication should be consolidated to improve maintainability")
    else:
        print(f"\n✅ NO MAJOR DUPLICATIONS FOUND!")
    
    return duplications

def show_lot_sizes_consolidation_plan():
    """Show specific plan for LOT_SIZES consolidation"""
    print("\n" + "="*60)
    print("🎯 LOT_SIZES CONSOLIDATION PLAN")
    print("="*60)
    
    print("""
📍 MASTER LOCATION (Keep):
   ✅ config/constants.py
   
🔧 FILES TO UPDATE (Remove LOT_SIZES, Add Import):
   1. src/analysis/exact_recommender.py
   2. src/portfolio/portfolio_manager.py  
   3. src/core/master_trader.py
   4. Any other files with duplicate definitions
   
📝 CHANGES NEEDED:
   1. Remove: LOT_SIZES = {'NIFTY': 75, 'BANKNIFTY': 30}
   2. Add: from config.constants import LOT_SIZES, get_lot_size
   3. Replace: self.lot_sizes with LOT_SIZES
   4. Replace: custom lot size logic with get_lot_size(symbol)
   
✅ BENEFITS:
   - Single source of truth for lot sizes
   - Easy to update when SEBI changes lot sizes
   - Consistent across all modules
   - Reduced maintenance overhead
    """)

if __name__ == "__main__":
    print("🌟 TradeMind AI - Code Duplication Scanner")
    print("🔍 Finding duplicate code patterns for optimization")
    
    duplications = find_duplications()
    show_lot_sizes_consolidation_plan()
    
    print(f"\n🚀 Ready to start consolidation? (Y/n)")
    response = input().lower()
    if response in ['y', 'yes', '']:
        print("✅ Great! Let's start with LOT_SIZES consolidation...")
        print("📋 First, run this script to see current duplications")
        print("💡 Then I'll provide specific file fixes one by one")
    else:
        print("📋 Analysis complete. Run again when ready to consolidate.")