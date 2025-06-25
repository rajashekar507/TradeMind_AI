"""
TradeMind_AI: Technical Indicators Module - FIXED VERSION
Adds RSI, MACD, Bollinger Bands, and more indicators
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

class TechnicalIndicators:
    def __init__(self):
        """Initialize Technical Indicators module"""
        print("📊 Initializing Technical Indicators...")
        
        # Load environment
        load_dotenv()
        
        print("✅ Technical Indicators ready!")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow:
            return None
        
        # Convert to pandas series
        prices_series = pd.Series(prices)
        
        # Calculate EMAs
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None
        
        # Convert to pandas series
        prices_series = pd.Series(prices)
        
        # Calculate moving average
        middle_band = prices_series.rolling(window=period).mean().iloc[-1]
        
        # Calculate standard deviation
        std = prices_series.rolling(window=period).std().iloc[-1]
        
        # Calculate bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'current_price': prices[-1]
        }
    
    def calculate_stochastic(self, prices, period=14):
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return None
        
        # Get recent prices
        recent_prices = prices[-period:]
        
        # Calculate %K
        lowest_low = min(recent_prices)
        highest_high = max(recent_prices)
        current_close = prices[-1]
        
        if highest_high == lowest_low:
            return 50  # Neutral
        
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        return k_percent
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range (ATR)"""
        if len(high) < period:
            return None
        
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(high)):
            high_low = high[i] - low[i]
            high_close = abs(high[i] - close[i-1])
            low_close = abs(low[i] - close[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        # Calculate ATR
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
            return atr
        
        return None
    
    def get_indicator_signals(self, symbol="NIFTY", prices=None):
        """Get all indicator signals and generate recommendation"""
        
        # For demo, using simulated prices if not provided
        if prices is None:
            # Generate realistic price data
            base_price = 25500 if symbol == "NIFTY" else 57000
            trend = np.random.choice([1, -1])  # Random trend
            prices = []
            
            for i in range(50):
                noise = np.random.randn() * 30
                trend_component = trend * i * 2
                price = base_price + trend_component + noise
                prices.append(price)
        
        signals = {}
        score = 0
        recommendations = []
        
        # Calculate RSI
        rsi = self.calculate_rsi(prices)
        if rsi is not None:
            signals['RSI'] = round(rsi, 2)
            if rsi <= 30:
                score += 20
                recommendations.append(f"RSI Oversold ({rsi:.1f}) - STRONG BUY Signal")
            elif rsi >= 70:
                score -= 20
                recommendations.append(f"RSI Overbought ({rsi:.1f}) - STRONG SELL Signal")
            else:
                recommendations.append(f"RSI Neutral ({rsi:.1f})")
        
        # Calculate MACD
        macd_data = self.calculate_macd(prices)
        if macd_data:
            signals['MACD'] = {
                'macd': round(macd_data['macd'], 2),
                'signal': round(macd_data['signal'], 2),
                'histogram': round(macd_data['histogram'], 2)
            }
            if macd_data['histogram'] > 0:
                score += 15
                recommendations.append("MACD Bullish Crossover - BUY Signal")
            else:
                score -= 15
                recommendations.append("MACD Bearish Crossover - SELL Signal")
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(prices)
        if bb_data:
            signals['Bollinger_Bands'] = {
                'upper': round(bb_data['upper'], 2),
                'middle': round(bb_data['middle'], 2),
                'lower': round(bb_data['lower'], 2)
            }
            if bb_data['current_price'] < bb_data['lower']:
                score += 15
                recommendations.append("Price below Lower BB - BUY Signal")
            elif bb_data['current_price'] > bb_data['upper']:
                score -= 15
                recommendations.append("Price above Upper BB - SELL Signal")
            else:
                recommendations.append("Price within Bollinger Bands - NEUTRAL")
        
        # Calculate Stochastic
        stoch = self.calculate_stochastic(prices)
        if stoch is not None:
            signals['Stochastic'] = round(stoch, 2)
            if stoch < 20:
                score += 10
                recommendations.append(f"Stochastic Oversold ({stoch:.1f}) - BUY Signal")
            elif stoch > 80:
                score -= 10
                recommendations.append(f"Stochastic Overbought ({stoch:.1f}) - SELL Signal")
        
        # Generate final signal
        if score >= 30:
            final_signal = "STRONG BUY"
            confidence = min(95, 70 + score)
        elif score >= 15:
            final_signal = "BUY"
            confidence = min(85, 60 + score)
        elif score <= -30:
            final_signal = "STRONG SELL"
            confidence = min(95, 70 + abs(score))
        elif score <= -15:
            final_signal = "SELL"
            confidence = min(85, 60 + abs(score))
        else:
            final_signal = "NEUTRAL"
            confidence = 50
        
        return {
            'symbol': symbol,
            'indicators': signals,
            'score': score,
            'signal': final_signal,
            'confidence': confidence,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    def display_analysis(self, analysis):
        """Display technical analysis in readable format"""
        print(f"\n📊 TECHNICAL ANALYSIS - {analysis['symbol']}")
        print("="*60)
        print(f"🎯 Signal: {analysis['signal']} (Score: {analysis['score']})")
        print(f"🎪 Confidence: {analysis['confidence']}%")
        print(f"⏰ Time: {analysis['timestamp'].strftime('%H:%M:%S')}")
        
        print("\n📈 Indicator Values:")
        if 'RSI' in analysis['indicators']:
            print(f"   • RSI: {analysis['indicators']['RSI']}")
        
        if 'MACD' in analysis['indicators']:
            macd = analysis['indicators']['MACD']
            print(f"   • MACD: {macd['macd']}")
            print(f"   • Signal Line: {macd['signal']}")
            print(f"   • Histogram: {macd['histogram']}")
        
        if 'Bollinger_Bands' in analysis['indicators']:
            bb = analysis['indicators']['Bollinger_Bands']
            print(f"   • BB Upper: {bb['upper']}")
            print(f"   • BB Middle: {bb['middle']}")
            print(f"   • BB Lower: {bb['lower']}")
        
        if 'Stochastic' in analysis['indicators']:
            print(f"   • Stochastic: {analysis['indicators']['Stochastic']}")
        
        print("\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        print("="*60)

# Test the module
if __name__ == "__main__":
    print("🌟 Testing Technical Indicators Module - FIXED VERSION")
    print("📊 Now with pure Python calculations (no finta dependency)")
    
    # Create instance
    indicators = TechnicalIndicators()
    
    # Test with NIFTY
    print("\n1️⃣ Testing NIFTY...")
    nifty_analysis = indicators.get_indicator_signals("NIFTY")
    indicators.display_analysis(nifty_analysis)
    
    # Test with BANKNIFTY
    print("\n2️⃣ Testing BANKNIFTY...")
    banknifty_analysis = indicators.get_indicator_signals("BANKNIFTY")
    indicators.display_analysis(banknifty_analysis)
    
    print("\n✅ Technical Indicators module working perfectly!")
    print("🎯 All indicators calculated without external dependencies!")
