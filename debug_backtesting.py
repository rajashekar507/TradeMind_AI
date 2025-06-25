#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('/home/ubuntu/trading_system')

from src.analysis.backtesting import BacktestingEngine
from src.analysis.trade_signal_engine import TradeSignalEngine
from src.config.settings import Settings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

async def debug_backtesting():
    """Debug why backtesting isn't generating trades"""
    print("🔍 DEBUGGING BACKTESTING TRADE GENERATION")
    print("=" * 50)
    
    settings = Settings()
    signal_engine = TradeSignalEngine(settings)
    signal_engine.confidence_threshold = 45  # Lower threshold
    
    print(f"✅ Signal engine initialized with {signal_engine.confidence_threshold}% threshold")
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
    historical_data = pd.DataFrame({
        'date': dates,
        'open': [24000, 24100, 24200, 24300, 24400],
        'high': [24100, 24200, 24300, 24400, 24500],
        'low': [23900, 24000, 24100, 24200, 24300],
        'close': [24050, 24150, 24250, 24350, 24450],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    })
    
    print(f"✅ Created mock historical data: {len(historical_data)} days")
    
    for i, row in historical_data.iterrows():
        print(f"\n📅 Testing day {i+1}: {row['date'].strftime('%Y-%m-%d')}")
        
        spot_price = row['close']
        
        mock_data = {
            'spot_data': {
                'status': 'success',
                'prices': {
                    'NIFTY': spot_price,
                    'BANKNIFTY': spot_price * 2.1  # Approximate ratio
                }
            },
            'options_data': {
                'NIFTY': {
                    'status': 'success',
                    'chain': [
                        {'strike': spot_price - 100, 'option_type': 'PE', 'open_interest': 1000000},
                        {'strike': spot_price + 100, 'option_type': 'CE', 'open_interest': 800000}
                    ],
                    'pcr': np.random.choice([0.6, 0.9, 1.1, 1.4]),
                    'max_pain': spot_price * (1 + np.random.normal(0, 0.01))
                }
            },
            'technical_data': {
                'NIFTY': {
                    'status': 'success',
                    'indicators': {
                        'rsi': np.random.choice([25, 35, 65, 75]),
                        'macd': np.random.normal(0, 1.0),
                        'macd_signal': np.random.normal(0, 1.0),
                        'current_price': spot_price,
                        'ema_9': spot_price * 1.001,
                        'ema_21': spot_price * 0.999,
                        'ema_50': spot_price * 0.998,
                        'trend_signal': np.random.choice(['bullish', 'bearish', 'neutral']),
                        'momentum_signal': np.random.choice(['oversold', 'overbought', 'neutral'])
                    }
                }
            },
            'vix_data': {'status': 'success', 'vix': np.random.choice([12, 18, 22, 26])},
            'fii_dii_data': {'status': 'success', 'net_flow': np.random.choice([-1500, -500, 500, 1500])},
            'news_data': {
                'status': 'success', 
                'sentiment': np.random.choice(['positive', 'negative', 'neutral']),
                'sentiment_score': np.random.choice([-0.4, -0.2, 0.2, 0.4])
            },
            'global_data': {
                'status': 'success', 
                'indices': {
                    'SGX_NIFTY': np.random.choice([-1.5, -0.5, 0.5, 1.5]),
                    'DOW': np.random.choice([-1.2, -0.3, 0.3, 1.2])
                }
            }
        }
        
        print(f"   📊 Spot: {spot_price:.0f}")
        print(f"   📈 RSI: {mock_data['technical_data']['NIFTY']['indicators']['rsi']}")
        print(f"   📉 PCR: {mock_data['options_data']['NIFTY']['pcr']:.1f}")
        print(f"   🌍 VIX: {mock_data['vix_data']['vix']}")
        
        signals = await signal_engine.generate_signals(mock_data)
        
        if signals:
            for signal in signals:
                print(f"   ✅ SIGNAL GENERATED!")
                print(f"      Instrument: {signal['instrument']}")
                print(f"      Strike: {signal['strike']} {signal['option_type']}")
                print(f"      Confidence: {signal['confidence']:.1f}%")
                print(f"      Reason: {signal['reason']}")
                print(f"      Scores: T:{signal['scores']['technical']:.0f} O:{signal['scores']['options']:.0f} S:{signal['scores']['sentiment']:.0f} G:{signal['scores']['global']:.0f}")
        else:
            print(f"   ❌ No signals generated")
            
            try:
                tech_score = signal_engine._calculate_technical_score('NIFTY', mock_data)
                options_score = signal_engine._calculate_options_score('NIFTY', mock_data)
                sentiment_score = signal_engine._calculate_sentiment_score(mock_data)
                global_score = signal_engine._calculate_global_score(mock_data)
                
                confidence = (
                    tech_score * signal_engine.weights['technical'] +
                    options_score * signal_engine.weights['options_flow'] +
                    sentiment_score * signal_engine.weights['sentiment_vix'] +
                    global_score * signal_engine.weights['global_cues']
                )
                
                print(f"      Debug scores: T:{tech_score:.0f} O:{options_score:.0f} S:{sentiment_score:.0f} G:{global_score:.0f}")
                print(f"      Weighted confidence: {confidence:.1f}% (threshold: {signal_engine.confidence_threshold}%)")
                
                if confidence < signal_engine.confidence_threshold:
                    print(f"      ❌ Below threshold by {signal_engine.confidence_threshold - confidence:.1f}%")
                
            except Exception as e:
                print(f"      ❌ Error calculating scores: {e}")

if __name__ == "__main__":
    asyncio.run(debug_backtesting())
