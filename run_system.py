#!/usr/bin/env python3
"""
VLR_AI Institutional-Grade Trading System
Main Entry Point for Live Trading Operation
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.core.system_manager import TradingSystemManager
    from src.config.settings import Settings
    print("✅ VLR_AI institutional-grade modules loaded successfully!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you're running from the trading_system root directory")
    sys.exit(1)

async def main():
    """Main entry point for VLR_AI institutional-grade trading system"""
    try:
        print("\n" + "="*80)
        print("🚀 VLR_AI INSTITUTIONAL-GRADE TRADING SYSTEM")
        print("="*80)
        print("🎯 7 Comprehensive Enhancements Active:")
        print("   🧠 Multi-timeframe Technical Analysis")
        print("   🕯️ Candlestick Pattern Recognition") 
        print("   📊 Dynamic Support/Resistance Calculation")
        print("   🎯 Opening Range Breakout (ORB) Strategy")
        print("   📈 Comprehensive Backtesting Framework")
        print("   💼 Live Trade Execution Engine")
        print("   🛡️ Advanced Risk Management System")
        print("="*80)
        
        settings = Settings()
        
        system_manager = TradingSystemManager(settings)
        
        print("\n🔄 Initializing institutional-grade trading system...")
        await system_manager.initialize()
        
        print("\n🚀 Starting live trading operation during market hours...")
        print("⏰ Market Hours: 09:15-15:30 IST")
        print("🔄 Analysis Cycle: Every 30 seconds")
        print("📊 Data Sources: 8 live APIs (Kite, NSE, Perplexity, Yahoo Finance)")
        print("🎯 Signal Threshold: 60% confidence")
        print("📱 Telegram Notifications: Active")
        print("\n" + "="*80)
        
        await system_manager.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
        print("👋 VLR_AI trading session ended")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Check your .env file and API credentials")
    finally:
        try:
            await system_manager.shutdown()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
