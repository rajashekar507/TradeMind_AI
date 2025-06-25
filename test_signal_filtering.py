#!/usr/bin/env python3

import os
import asyncio
from dotenv import load_dotenv
from src.core.data_manager import DataManager
from src.analysis.trade_signal_engine import TradeSignalEngine
from src.config.settings import Settings
from src.auth.kite_auth import KiteAuthenticator
from src.utils.telegram_notifier import TelegramNotifier

async def test_signal_filtering():
    """Test signal filtering with lower confidence threshold"""
    
    load_dotenv()
    
    print("🔍 Testing Signal Filtering with Lower Threshold...")
    
    settings = Settings()
    
    kite_auth = KiteAuthenticator(settings)
    await kite_auth.initialize()
    kite_client = kite_auth.get_authenticated_kite()
    
    data_manager = DataManager(settings, kite_client)
    signal_engine = TradeSignalEngine(settings)
    telegram = TelegramNotifier(settings)
    
    await data_manager.initialize()
    await signal_engine.initialize()
    await telegram.initialize()
    
    original_threshold = signal_engine.confidence_threshold
    signal_engine.confidence_threshold = 15.0  # Lower threshold for testing
    
    print(f"\n📊 Confidence threshold lowered: {original_threshold}% → {signal_engine.confidence_threshold}%")
    
    print("\n📡 Fetching live market data...")
    market_data = await data_manager.fetch_all_data()
    
    print(f"\n📊 Market Data Status:")
    for source, data in market_data.items():
        if isinstance(data, dict) and 'status' in data:
            status = data['status']
            print(f"   {source}: {status}")
    
    print("\n🎯 Testing Signal Generation with Filtering...")
    
    for cycle in range(1, 4):
        print(f"\n--- Cycle {cycle} ---")
        
        signals = await signal_engine.generate_signals(market_data)
        
        if signals:
            print(f"✅ Generated {len(signals)} signals:")
            for signal in signals:
                print(f"   📊 {signal.get('instrument')} {signal.get('strike')} {signal.get('direction')}")
                print(f"      Entry: ₹{signal.get('entry_price')} (Technical: ₹{signal.get('entry_level')})")
                print(f"      Signal Type: {signal.get('signal_type', 'N/A')}")
                print(f"      Confidence: {signal.get('confidence')}%")
                print(f"      Signal ID: {signal.get('signal_id')}")
                
                await telegram.send_trade_signal(signal)
        else:
            print("❌ No signals generated")
        
        print(f"\n⏰ Cooldown Status:")
        for instrument in ['NIFTY', 'BANKNIFTY']:
            if instrument in signal_engine.last_signal_time:
                last_time = signal_engine.last_signal_time[instrument]
                print(f"   {instrument}: Last signal at {last_time.strftime('%H:%M:%S')}")
            else:
                print(f"   {instrument}: No previous signals")
        
        if cycle < 3:
            print("⏳ Waiting 5 seconds...")
            await asyncio.sleep(5)
    
    signal_engine.confidence_threshold = original_threshold
    
    await data_manager.shutdown()
    await signal_engine.shutdown()
    await kite_auth.shutdown()

if __name__ == "__main__":
    asyncio.run(test_signal_filtering())
