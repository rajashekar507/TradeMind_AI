"""
Fix final import issues - find and fix all references to ai_trader
CORRECTED VERSION
"""

import os
import re

def find_and_fix_ai_trader_imports():
    """Find all files that import ai_trader and fix them"""
    
    files_fixed = []
    files_checked = 0
    
    # Walk through all Python files
    for root, dirs, files in os.walk('.'):
        # Skip backup and __pycache__ directories
        if 'old_files_backup' in root or '__pycache__' in root or '.git' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file != 'fix_final_imports.py':
                filepath = os.path.join(root, file)
                files_checked += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Find and fix various import patterns
                    patterns_to_fix = [
                        # Pattern 1: from core.ai_trader import ...
                        (r'from core\.ai_trader import [^\n]*', '# Removed ai_trader import - using unified_trading_engine'),
                        
                        # Pattern 2: from src.core.ai_trader import ...
                        (r'from src\.core\.ai_trader import [^\n]*', '# Removed ai_trader import - using unified_trading_engine'),
                        
                        # Pattern 3: import core.ai_trader
                        (r'import core\.ai_trader', '# Removed ai_trader import - using unified_trading_engine'),
                        
                        # Pattern 4: from ai_trader import ...
                        (r'from ai_trader import [^\n]*', '# Removed ai_trader import - using unified_trading_engine'),
                        
                        # Pattern 5: import ai_trader
                        (r'import ai_trader', '# Removed ai_trader import - using unified_trading_engine'),
                    ]
                    
                    # Apply all fixes
                    for pattern, replacement in patterns_to_fix:
                        if re.search(pattern, content):
                            content = re.sub(pattern, replacement, content)
                            print(f"  ✓ Fixed pattern in {filepath}: {pattern[:30]}...")
                    
                    # Write back if changed
                    if content != original_content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        files_fixed.append(filepath)
                        
                except Exception as e:
                    if 'fix_final_imports.py' not in str(e):
                        print(f"  ⚠️ Error processing {filepath}: {e}")
    
    return files_checked, files_fixed

def main():
    print("🔍 Searching for remaining ai_trader imports...")
    print("=" * 50)
    
    files_checked, files_fixed = find_and_fix_ai_trader_imports()
    
    print("\n" + "=" * 50)
    print(f"✅ COMPLETE!")
    print(f"📊 Files checked: {files_checked}")
    print(f"🔧 Files fixed: {len(files_fixed)}")
    
    if files_fixed:
        print("\n📝 Fixed files:")
        for file in files_fixed:
            print(f"   - {file}")
    else:
        print("\n✨ No files needed fixing!")
    
    print("\n💡 Now test: Run 'python run_trading.py' and try option 6")

if __name__ == "__main__":
    main()