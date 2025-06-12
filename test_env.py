"""
Test Environment Variables Loading
Run this to check if your .env file is loading correctly
"""

import os
from dotenv import load_dotenv

print("🔍 Testing Environment Variables...")
print("="*50)

# Load .env file
env_loaded = load_dotenv()
print(f"📁 .env file loaded: {env_loaded}")

# Check for .env file existence
env_file_exists = os.path.exists('.env')
print(f"📄 .env file exists: {env_file_exists}")

if env_file_exists:
    print(f"📍 .env file location: {os.path.abspath('.env')}")

print("\n🔑 Environment Variables Check:")
print("-" * 30)

# Check critical variables (without showing values)
critical_vars = [
    'DHAN_CLIENT_ID',
    'DHAN_ACCESS_TOKEN', 
    'TOTAL_CAPITAL',
    'TELEGRAM_BOT_TOKEN',
    'TELEGRAM_CHAT_ID'
]

for var in critical_vars:
    value = os.getenv(var)
    if value:
        # Show only first 5 and last 3 characters for security
        if len(value) > 8:
            masked_value = value[:5] + "***" + value[-3:]
        else:
            masked_value = "***"
        print(f"✅ {var}: {masked_value}")
    else:
        print(f"❌ {var}: NOT FOUND")

print("\n🌍 Current Working Directory:")
print(f"📂 {os.getcwd()}")

print("\n📝 .env File Content Structure (if exists):")
if env_file_exists:
    try:
        with open('.env', 'r') as f:
            lines = f.readlines()
        print(f"📊 Total lines: {len(lines)}")
        print("📋 Variables found:")
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                var_name = line.split('=')[0]
                print(f"   {i}. {var_name}")
            elif line.startswith('#'):
                print(f"   {i}. # (comment)")
            elif not line:
                print(f"   {i}. (empty line)")
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")

print("\n" + "="*50)
print("🎯 Next Steps:")
print("1. If variables show 'NOT FOUND', add them to .env")
print("2. If .env file doesn't exist, create it")
print("3. Make sure no spaces around = signs")
print("4. Ensure file is in correct directory")