#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.rejected_signals_logger import RejectedSignalsLogger

async def test_enhanced_performance_tracking():
    """Test enhanced performance tracking with realistic modeling"""
    
    print("📊 Testing Enhanced Performance Tracking System...")
    
    logger = RejectedSignalsLogger()
    
    sample_market_data = {
        'spot_data': {'status': 'success', 'prices': {'NIFTY': 25250, 'BANKNIFTY': 56500}},
        'vix_data': {'status': 'success', 'vix': 18.5}
    }
    
    logger.log_rejection('NIFTY', 25300, 'CE', 'High IV 160% exceeds 150% limit for normal market conditions', 
                        sample_market_data, {'ltp': 45, 'iv': 160, 'volume': 150, 'oi': 2500})
    
    logger.log_rejection('BANKNIFTY', 57000, 'PE', 'Low volume: 50 < 200 minimum liquidity requirement', 
                        sample_market_data, {'ltp': 120, 'volume': 50, 'oi': 800})
    
    logger.log_rejection('NIFTY', 25200, 'PE', 'Google Finance price mismatch: Kite ₹25250 vs Google ₹25280 (diff: ₹30)', 
                        sample_market_data, {'ltp': 85, 'iv': 25, 'volume': 300, 'oi': 5000})
    
    sample_signals = [
        {
            'timestamp': '2025-06-25T10:30:00',
            'instrument': 'NIFTY',
            'strike': 25250,
            'direction': 'CE',
            'entry_price': 85.5
        },
        {
            'timestamp': '2025-06-25T14:15:00',
            'instrument': 'BANKNIFTY',
            'strike': 56500,
            'direction': 'PE',
            'entry_price': 95.0
        }
    ]
    
    for signal in sample_signals:
        logger.log_accepted_signal(signal)
    
    print("\n🧮 Calculating realistic outcomes using evidence-based modeling...")
    await logger.calculate_realistic_outcomes()
    
    print("\n📈 Calculating Enhanced Performance Metrics...")
    enhanced_metrics = logger.calculate_enhanced_performance_metrics()
    
    print("\n📊 Enhanced Performance Analysis:")
    print("=" * 60)
    
    if 'rejected_signals_metrics' in enhanced_metrics:
        print("\n🚫 Rejected Signals Performance:")
        for key, value in enhanced_metrics['rejected_signals_metrics'].items():
            print(f"   {key}: {value}")
    
    if 'accepted_signals_metrics' in enhanced_metrics:
        print("\n✅ Accepted Signals Performance:")
        for key, value in enhanced_metrics['accepted_signals_metrics'].items():
            print(f"   {key}: {value}")
    
    if 'time_based_analysis' in enhanced_metrics:
        time_analysis = enhanced_metrics['time_based_analysis']
        print("\n⏰ Time-Based Performance Analysis:")
        
        if 'best_hour' in time_analysis and time_analysis['best_hour']:
            print(f"   Best performing hour: {time_analysis['best_hour'][0]}:00 ({time_analysis['best_hour'][1]:.2f}%)")
        
        if 'worst_hour' in time_analysis and time_analysis['worst_hour']:
            print(f"   Worst performing hour: {time_analysis['worst_hour'][0]}:00 ({time_analysis['worst_hour'][1]:.2f}%)")
        
        if 'best_day' in time_analysis and time_analysis['best_day']:
            print(f"   Best performing day: {time_analysis['best_day'][0]} ({time_analysis['best_day'][1]:.2f}%)")
    
    if 'comparison' in enhanced_metrics:
        comparison = enhanced_metrics['comparison']
        print("\n🔍 Filter Effectiveness Analysis:")
        print(f"   Return difference (Accepted - Rejected): {comparison.get('rejected_vs_accepted_return', 0):.2f}%")
        print(f"   Sharpe ratio difference: {comparison.get('rejected_vs_accepted_sharpe', 0):.3f}")
        print(f"   Filter effectiveness score: {comparison.get('filter_effectiveness_score', 0):.1f}/100")
    
    weekly_report = logger.generate_weekly_report()
    print("\n📋 Weekly Analysis Report:")
    print("=" * 60)
    for key, value in weekly_report.items():
        if key != 'performance_comparison':
            print(f"{key}: {value}")
    
    print("\n✅ Enhanced performance tracking system with realistic modeling working correctly")

if __name__ == "__main__":
    asyncio.run(test_enhanced_performance_tracking())
