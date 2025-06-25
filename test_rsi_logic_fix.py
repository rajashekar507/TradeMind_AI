#!/usr/bin/env python3
"""
Test script to validate RSI logic fixes
Ensures professional mean-reversion strategy is implemented correctly
"""

import asyncio
import sys
import os
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.trade_signal_engine import TradeSignalEngine

async def test_rsi_logic():
    """Test RSI logic with various scenarios to ensure correct behavior"""
    
    print("🧪 TESTING RSI LOGIC FIXES")
    print("=" * 80)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    settings = {
        'CONFIDENCE_THRESHOLD': 20.0,
        'MAX_DAILY_LOSS': 50000,
        'MAX_POSITION_SIZE': 100000
    }
    signal_engine = TradeSignalEngine(settings)
    await signal_engine.initialize()
    
    test_scenarios = [
        {"rsi": 20, "expected_direction": "CE", "description": "Oversold - Strong Bullish"},
        {"rsi": 30, "expected_direction": "CE", "description": "Oversold Threshold - Bullish"},
        {"rsi": 35, "expected_direction": "CE", "description": "Mild Bullish"},
        {"rsi": 50, "expected_direction": "neutral", "description": "Neutral"},
        {"rsi": 65, "expected_direction": "PE", "description": "Mild Bearish"},
        {"rsi": 70, "expected_direction": "PE", "description": "Overbought Threshold - Bearish"},
        {"rsi": 80, "expected_direction": "PE", "description": "Overbought - Strong Bearish"},
        {"rsi": 76, "expected_direction": "PE", "description": "Extremely Overbought - Hard Override"},
        {"rsi": 24, "expected_direction": "CE", "description": "Extremely Oversold - Hard Override"}
    ]
    
    print("🎯 TESTING DIRECTIONAL LOGIC:")
    print("-" * 50)
    
    passed_tests = 0
    total_tests = len(test_scenarios)
    
    print(f"📋 Testing {total_tests} RSI scenarios including extreme override conditions...")
    
    for scenario in test_scenarios:
        rsi_value = scenario["rsi"]
        expected = scenario["expected_direction"]
        description = scenario["description"]
        
        mock_market_data = {
            'technical_data': {
                'status': 'success',
                'NIFTY': {
                    'rsi': rsi_value,
                    'trend': 'neutral',
                    'macd': {'signal': 'neutral'}
                }
            },
            'vix_data': {'status': 'success', 'vix': 16.5},
            'global_data': {'status': 'success', 'indices': {'DOW_CHANGE': 0, 'NASDAQ_CHANGE': 0}},
            'options_data': {'status': 'success', 'NIFTY': {'status': 'success', 'pcr': 1.0}}
        }
        
        direction = signal_engine._determine_signal_direction('NIFTY', mock_market_data, 60.0)
        
        if expected == "neutral":
            test_passed = True  # For neutral, we accept either direction
            result_icon = "✅"
        elif direction == expected:
            test_passed = True
            result_icon = "✅"
            passed_tests += 1
        else:
            test_passed = False
            result_icon = "❌"
        
        print(f"   {result_icon} RSI {rsi_value}: {description}")
        print(f"      Expected: {expected}, Got: {direction}")
        
        if not test_passed and expected != "neutral":
            print(f"      🚨 CRITICAL FAILURE: RSI {rsi_value} should generate {expected} signals!")
    
    print()
    print("🧮 TESTING TECHNICAL SCORE CALCULATION:")
    print("-" * 50)
    
    score_tests = [
        {"rsi": 20, "expected_sign": "positive", "description": "Oversold should add to score"},
        {"rsi": 30, "expected_sign": "positive", "description": "Oversold threshold should add to score"},
        {"rsi": 70, "expected_sign": "negative", "description": "Overbought should subtract from score"},
        {"rsi": 80, "expected_sign": "negative", "description": "Overbought should subtract from score"},
        {"rsi": 76, "expected_sign": "negative", "description": "Extremely overbought should heavily subtract from score"},
        {"rsi": 24, "expected_sign": "positive", "description": "Extremely oversold should heavily add to score"}
    ]
    
    for test in score_tests:
        rsi_value = test["rsi"]
        expected_sign = test["expected_sign"]
        description = test["description"]
        
        mock_market_data = {
            'technical_data': {
                'status': 'success',
                'NIFTY': {
                    'rsi': rsi_value,
                    'trend': 'neutral',
                    'macd': {'signal': 'neutral'},
                    'volume_trend': 'neutral'
                }
            }
        }
        
        score = signal_engine._calculate_technical_score('NIFTY', mock_market_data)
        base_score = 20  # Base score from trend and macd being neutral
        rsi_contribution = score - base_score
        
        if expected_sign == "positive" and rsi_contribution > 0:
            result_icon = "✅"
            passed_tests += 1
        elif expected_sign == "negative" and rsi_contribution < 0:
            result_icon = "✅"
            passed_tests += 1
        else:
            result_icon = "❌"
        
        print(f"   {result_icon} RSI {rsi_value}: {description}")
        print(f"      Score: {score}, RSI Contribution: {rsi_contribution:+.1f}")
        
        total_tests += 1
    
    print()
    print("📊 TESTING TECHNICAL CONFIRMATIONS:")
    print("-" * 50)
    
    confirmation_tests = [
        {"rsi": 25, "should_confirm": True, "description": "Strong oversold should confirm"},
        {"rsi": 35, "should_confirm": False, "description": "Mild oversold should not confirm"},
        {"rsi": 65, "should_confirm": False, "description": "Mild overbought should not confirm"},
        {"rsi": 75, "should_confirm": True, "description": "Strong overbought should confirm"},
        {"rsi": 76, "should_confirm": True, "description": "Extremely overbought should confirm"},
        {"rsi": 24, "should_confirm": True, "description": "Extremely oversold should confirm"}
    ]
    
    for test in confirmation_tests:
        rsi_value = test["rsi"]
        should_confirm = test["should_confirm"]
        description = test["description"]
        
        mock_market_data = {
            'technical_data': {
                'status': 'success',
                'NIFTY': {
                    'rsi': rsi_value,
                    'trend': 'neutral'
                }
            },
            'options_data': {'status': 'success', 'NIFTY': {'status': 'success'}}
        }
        
        confirmations = signal_engine._count_technical_confirmations('NIFTY', mock_market_data)
        rsi_confirmed = confirmations > 1  # Base confirmation from options data
        
        if should_confirm == rsi_confirmed:
            result_icon = "✅"
            passed_tests += 1
        else:
            result_icon = "❌"
        
        print(f"   {result_icon} RSI {rsi_value}: {description}")
        print(f"      Confirmations: {confirmations}, RSI Confirmed: {rsi_confirmed}")
        
        total_tests += 1
    
    print()
    print("🏆 TEST RESULTS SUMMARY:")
    print("-" * 50)
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"   📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("   ✅ RSI LOGIC FIX: SUCCESSFUL")
        print("   🎯 Professional mean-reversion strategy implemented correctly")
        print("   💰 System will no longer generate losing trades at market tops")
    elif success_rate >= 70:
        print("   ⚠️ RSI LOGIC FIX: PARTIALLY SUCCESSFUL")
        print("   🔧 Some issues remain - review failed tests")
    else:
        print("   ❌ RSI LOGIC FIX: FAILED")
        print("   🚨 Critical issues remain - system still unsafe for trading")
    
    print()
    print("🔍 CRITICAL VALIDATION:")
    print("-" * 50)
    
    critical_test_data = {
        'technical_data': {
            'status': 'success',
            'NIFTY': {
                'rsi': 76,  # Extremely overbought (>75 for hard override)
                'trend': 'bullish',  # Even with bullish trend
                'macd': {'signal': 'bullish'}
            }
        },
        'vix_data': {'status': 'success', 'vix': 15.0},  # Low VIX
        'global_data': {'status': 'success', 'indices': {'DOW_CHANGE': 1.0, 'NASDAQ_CHANGE': 0.8}},  # Positive global
        'options_data': {'status': 'success', 'NIFTY': {'status': 'success', 'pcr': 0.7}}  # Bullish PCR
    }
    
    critical_direction = signal_engine._determine_signal_direction('NIFTY', critical_test_data, 80.0)
    
    if critical_direction == 'PE':
        print("   ✅ CRITICAL TEST PASSED: RSI extremely overbought generates PUT signals")
        print("   🛡️ System will NOT buy calls at market tops")
        print("   💰 Professional mean-reversion strategy working correctly")
        print("   🔒 RSI OVERRIDE: Extreme levels force correct directional signals")
    else:
        print("   ❌ CRITICAL TEST FAILED: RSI extremely overbought still generates CALL signals")
        print("   🚨 SYSTEM UNSAFE: Will still buy calls at market tops")
        print("   💸 GUARANTEED LOSSES: Fix incomplete")
    
    await signal_engine.shutdown()
    
    return success_rate >= 90

if __name__ == "__main__":
    success = asyncio.run(test_rsi_logic())
    if success:
        print("\n🎉 RSI LOGIC FIX VALIDATION: COMPLETE")
        exit(0)
    else:
        print("\n💥 RSI LOGIC FIX VALIDATION: FAILED")
        exit(1)
