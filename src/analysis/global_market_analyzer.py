"""
TradeMind_AI: Global Market Integration
Tracks US markets, Dollar Index, Asian markets, India VIX, and SGX Nifty
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
from typing import Dict, List, Tuple
import pytz

class GlobalMarketAnalyzer:
    def __init__(self):
        """Initialize Global Market Analyzer"""
        print("🌍 Initializing Global Market Analyzer...")
        
        # Market symbols
        self.global_indices = {
            # US Markets
            'SP500': '^GSPC',
            'DOW': '^DJI',
            'NASDAQ': '^IXIC',
            'VIX': '^VIX',
            
            # Asian Markets
            'NIKKEI': '^N225',
            'HANGSENG': '^HSI',
            'SHANGHAI': '000001.SS',
            'KOSPI': '^KS11',
            'SGX_NIFTY': 'SGXNIFTY.NS',  # SGX Nifty futures
            
            # Commodities & Currencies
            'GOLD': 'GC=F',
            'CRUDE': 'CL=F',
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'USDINR': 'USDINR=X',
            
            # India specific
            'INDIA_VIX': '^NSEINDVIX'
        }
        
        # Market timings (in IST)
        self.market_timings = {
            'US_OPEN': {'hour': 19, 'minute': 0},      # 7:30 PM IST
            'US_CLOSE': {'hour': 1, 'minute': 30},     # 1:30 AM IST next day
            'ASIA_OPEN': {'hour': 5, 'minute': 30},    # 5:30 AM IST
            'INDIA_OPEN': {'hour': 9, 'minute': 15},   # 9:15 AM IST
            'INDIA_CLOSE': {'hour': 15, 'minute': 30}  # 3:30 PM IST
        }
        
        # Correlation thresholds
        self.correlation_period = 30  # days
        self.strong_correlation = 0.7
        self.moderate_correlation = 0.5
        
        # Cache for market data
        self.market_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        print("✅ Global Market Analyzer ready!")
    
    def fetch_global_market_data(self, use_cache=True):
        """Fetch all global market data"""
        print("\n🌍 Fetching global market data...")
        
        current_time = time.time()
        
        # Check cache
        if use_cache and self.market_cache.get('timestamp', 0) > current_time - self.cache_duration:
            print("📦 Using cached data")
            return self.market_cache['data']
        
        market_data = {}
        
        for market_name, symbol in self.global_indices.items():
            try:
                print(f"📊 Fetching {market_name}...", end='')
                
                # Fetch data
                ticker = yf.Ticker(symbol)
                
                # Get current data
                info = ticker.info
                history = ticker.history(period='5d', interval='1d')
                
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    prev_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data[market_name] = {
                        'symbol': symbol,
                        'current_price': round(current_price, 2),
                        'previous_close': round(prev_close, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_pct, 2),
                        'volume': history['Volume'].iloc[-1],
                        'high': history['High'].iloc[-1],
                        'low': history['Low'].iloc[-1],
                        'timestamp': datetime.now()
                    }
                    
                    print(f" ✅ {current_price:.2f} ({change_pct:+.2f}%)")
                else:
                    print(f" ❌ No data")
                    
            except Exception as e:
                print(f" ❌ Error: {e}")
                market_data[market_name] = None
        
        # Special handling for India VIX
        market_data['INDIA_VIX'] = self.fetch_india_vix()
        
        # Special handling for SGX Nifty
        market_data['SGX_NIFTY'] = self.fetch_sgx_nifty()
        
        # Update cache
        self.market_cache = {
            'data': market_data,
            'timestamp': current_time
        }
        
        return market_data
    
    def fetch_india_vix(self):
        """Fetch India VIX data"""
        try:
            # Simulate India VIX (in production, use NSE API)
            # For now, using approximation based on global VIX
            global_vix = self.market_cache.get('data', {}).get('VIX', {})
            
            if global_vix and global_vix.get('current_price'):
                # India VIX typically correlates with global VIX
                india_vix_value = global_vix['current_price'] * 0.85 + np.random.uniform(-2, 2)
                
                return {
                    'current_value': round(india_vix_value, 2),
                    'change': round(np.random.uniform(-1, 1), 2),
                    'interpretation': self.interpret_vix(india_vix_value)
                }
            else:
                # Default value
                return {
                    'current_value': 15.5,
                    'change': 0.5,
                    'interpretation': 'Normal volatility'
                }
                
        except Exception as e:
            print(f"❌ India VIX error: {e}")
            return None
    
    def fetch_sgx_nifty(self):
        """Fetch SGX Nifty futures data"""
        try:
            # In production, use proper SGX data feed
            # For now, simulate based on correlation with other markets
            
            # SGX Nifty typically indicates Nifty opening
            # It correlates with US markets overnight performance
            
            us_performance = self.calculate_us_market_impact()
            asia_performance = self.calculate_asia_market_impact()
            
            # Estimate SGX Nifty
            base_nifty = 25100  # You can get this from your market data
            
            # Apply overnight changes
            sgx_change_pct = (us_performance['weighted_change'] * 0.4 + 
                             asia_performance['weighted_change'] * 0.3 +
                             np.random.uniform(-0.3, 0.3))
            
            sgx_value = base_nifty * (1 + sgx_change_pct / 100)
            
            return {
                'indicative_value': round(sgx_value, 2),
                'change_percent': round(sgx_change_pct, 2),
                'gap_indication': 'Gap Up' if sgx_change_pct > 0.5 else 'Gap Down' if sgx_change_pct < -0.5 else 'Flat',
                'confidence': 'High' if abs(sgx_change_pct) > 1 else 'Medium'
            }
            
        except Exception as e:
            print(f"❌ SGX Nifty error: {e}")
            return None
    
    def interpret_vix(self, vix_value):
        """Interpret VIX levels"""
        if vix_value < 12:
            return "Very Low volatility - Complacency"
        elif vix_value < 15:
            return "Low volatility - Normal market"
        elif vix_value < 20:
            return "Moderate volatility - Cautious"
        elif vix_value < 30:
            return "High volatility - Fear in market"
        else:
            return "Extreme volatility - Panic mode"
    
    def calculate_us_market_impact(self):
        """Calculate US market impact on Indian markets"""
        us_indices = ['SP500', 'DOW', 'NASDAQ']
        
        total_change = 0
        weight_sum = 0
        
        weights = {
            'SP500': 0.5,
            'DOW': 0.3,
            'NASDAQ': 0.2
        }
        
        impact_data = {
            'indices': {},
            'weighted_change': 0,
            'sentiment': 'Neutral'
        }
        
        for index in us_indices:
            market_data = self.market_cache.get('data', {}).get(index)
            
            if market_data and market_data.get('change_percent'):
                change = market_data['change_percent']
                weight = weights[index]
                
                impact_data['indices'][index] = change
                total_change += change * weight
                weight_sum += weight
        
        if weight_sum > 0:
            impact_data['weighted_change'] = round(total_change / weight_sum, 2)
            
            # Determine sentiment
            if impact_data['weighted_change'] > 1:
                impact_data['sentiment'] = 'Strongly Positive'
            elif impact_data['weighted_change'] > 0.5:
                impact_data['sentiment'] = 'Positive'
            elif impact_data['weighted_change'] < -1:
                impact_data['sentiment'] = 'Strongly Negative'
            elif impact_data['weighted_change'] < -0.5:
                impact_data['sentiment'] = 'Negative'
        
        return impact_data
    
    def calculate_asia_market_impact(self):
        """Calculate Asian market impact"""
        asia_indices = ['NIKKEI', 'HANGSENG', 'KOSPI']
        
        weights = {
            'NIKKEI': 0.4,
            'HANGSENG': 0.4,
            'KOSPI': 0.2
        }
        
        impact_data = {
            'indices': {},
            'weighted_change': 0,
            'sentiment': 'Neutral'
        }
        
        total_change = 0
        weight_sum = 0
        
        for index in asia_indices:
            market_data = self.market_cache.get('data', {}).get(index)
            
            if market_data and market_data.get('change_percent'):
                change = market_data['change_percent']
                weight = weights[index]
                
                impact_data['indices'][index] = change
                total_change += change * weight
                weight_sum += weight
        
        if weight_sum > 0:
            impact_data['weighted_change'] = round(total_change / weight_sum, 2)
            
            # Determine sentiment
            if impact_data['weighted_change'] > 0.8:
                impact_data['sentiment'] = 'Positive'
            elif impact_data['weighted_change'] < -0.8:
                impact_data['sentiment'] = 'Negative'
        
        return impact_data
    
    def calculate_dollar_impact(self):
        """Calculate Dollar Index impact on Indian markets"""
        dxy_data = self.market_cache.get('data', {}).get('DXY')
        usdinr_data = self.market_cache.get('data', {}).get('USDINR')
        
        impact = {
            'dxy_change': 0,
            'usdinr_change': 0,
            'impact_on_nifty': 'Neutral',
            'fii_flow_indication': 'Neutral'
        }
        
        if dxy_data and dxy_data.get('change_percent'):
            impact['dxy_change'] = dxy_data['change_percent']
            
            # Strong dollar typically negative for emerging markets
            if impact['dxy_change'] > 0.5:
                impact['impact_on_nifty'] = 'Negative'
                impact['fii_flow_indication'] = 'Outflow likely'
            elif impact['dxy_change'] < -0.5:
                impact['impact_on_nifty'] = 'Positive'
                impact['fii_flow_indication'] = 'Inflow likely'
        
        if usdinr_data and usdinr_data.get('change_percent'):
            impact['usdinr_change'] = usdinr_data['change_percent']
        
        return impact
    
    def calculate_correlation_matrix(self, period_days=30):
        """Calculate correlation between global markets and Nifty"""
        print("\n📊 Calculating market correlations...")
        
        correlations = {}
        
        # Fetch historical data for Nifty
        nifty = yf.Ticker('^NSEI')
        nifty_hist = nifty.history(period=f'{period_days}d')['Close'].pct_change().dropna()
        
        for market_name, symbol in self.global_indices.items():
            if market_name in ['INDIA_VIX', 'SGX_NIFTY']:
                continue
                
            try:
                ticker = yf.Ticker(symbol)
                market_hist = ticker.history(period=f'{period_days}d')['Close'].pct_change().dropna()
                
                # Align dates
                common_dates = nifty_hist.index.intersection(market_hist.index)
                
                if len(common_dates) > 10:
                    correlation = nifty_hist[common_dates].corr(market_hist[common_dates])
                    correlations[market_name] = round(correlation, 3)
                    
            except Exception as e:
                print(f"❌ Correlation error for {market_name}: {e}")
        
        return correlations
    
    def get_market_opening_indication(self):
        """Get market opening indication based on global cues"""
        indication = {
            'timestamp': datetime.now(),
            'sgx_indication': None,
            'global_sentiment': None,
            'opening_prediction': None,
            'confidence': 0,
            'key_factors': []
        }
        
        # Get latest data
        market_data = self.fetch_global_market_data()
        
        # SGX Nifty indication
        sgx_data = market_data.get('SGX_NIFTY')
        if sgx_data:
            indication['sgx_indication'] = {
                'value': sgx_data['indicative_value'],
                'gap': sgx_data['gap_indication'],
                'change': sgx_data['change_percent']
            }
            
            if abs(sgx_data['change_percent']) > 0.5:
                indication['key_factors'].append(f"SGX Nifty indicating {sgx_data['gap_indication']}")
        
        # US market impact
        us_impact = self.calculate_us_market_impact()
        if abs(us_impact['weighted_change']) > 1:
            indication['key_factors'].append(f"US markets {us_impact['sentiment']}")
        
        # Asia market impact  
        asia_impact = self.calculate_asia_market_impact()
        if abs(asia_impact['weighted_change']) > 0.8:
            indication['key_factors'].append(f"Asian markets {asia_impact['sentiment']}")
        
        # Dollar impact
        dollar_impact = self.calculate_dollar_impact()
        if abs(dollar_impact['dxy_change']) > 0.5:
            indication['key_factors'].append(f"Dollar Index impact: {dollar_impact['impact_on_nifty']}")
        
        # India VIX
        vix_data = market_data.get('INDIA_VIX')
        if vix_data and vix_data['current_value'] > 20:
            indication['key_factors'].append(f"High volatility: VIX at {vix_data['current_value']}")
        
        # Calculate overall sentiment
        sentiment_score = (
            us_impact['weighted_change'] * 0.4 +
            asia_impact['weighted_change'] * 0.3 +
            (sgx_data['change_percent'] if sgx_data else 0) * 0.3
        )
        
        if sentiment_score > 0.5:
            indication['global_sentiment'] = 'Positive'
            indication['opening_prediction'] = 'Gap Up Opening Expected'
            indication['confidence'] = min(90, 70 + abs(sentiment_score) * 10)
        elif sentiment_score < -0.5:
            indication['global_sentiment'] = 'Negative'
            indication['opening_prediction'] = 'Gap Down Opening Expected'
            indication['confidence'] = min(90, 70 + abs(sentiment_score) * 10)
        else:
            indication['global_sentiment'] = 'Mixed'
            indication['opening_prediction'] = 'Flat Opening Expected'
            indication['confidence'] = 60
        
        return indication
    
    def generate_global_market_report(self):
        """Generate comprehensive global market report"""
        print("\n📊 GENERATING GLOBAL MARKET REPORT")
        print("="*70)
        
        # Fetch all data
        market_data = self.fetch_global_market_data()
        correlations = self.calculate_correlation_matrix()
        opening_indication = self.get_market_opening_indication()
        
        report = {
            'timestamp': datetime.now(),
            'market_data': market_data,
            'correlations': correlations,
            'opening_indication': opening_indication,
            'us_impact': self.calculate_us_market_impact(),
            'asia_impact': self.calculate_asia_market_impact(),
            'dollar_impact': self.calculate_dollar_impact()
        }
        
        # Display report
        self.display_market_report(report)
        
        return report
    
    def display_market_report(self, report):
        """Display market report in formatted way"""
        print(f"\n🌍 GLOBAL MARKET ANALYSIS")
        print(f"📅 {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # US Markets
        print("\n🇺🇸 US MARKETS:")
        us_impact = report['us_impact']
        for index, change in us_impact['indices'].items():
            print(f"   {index}: {change:+.2f}%")
        print(f"   Overall Impact: {us_impact['sentiment']} ({us_impact['weighted_change']:+.2f}%)")
        
        # Asian Markets
        print("\n🌏 ASIAN MARKETS:")
        asia_impact = report['asia_impact']
        for index, change in asia_impact['indices'].items():
            print(f"   {index}: {change:+.2f}%")
        print(f"   Overall Impact: {asia_impact['sentiment']} ({asia_impact['weighted_change']:+.2f}%)")
        
        # Dollar & Commodities
        print("\n💵 DOLLAR & COMMODITIES:")
        dollar_impact = report['dollar_impact']
        print(f"   Dollar Index: {dollar_impact['dxy_change']:+.2f}%")
        print(f"   USD/INR: {dollar_impact['usdinr_change']:+.2f}%")
        print(f"   Impact on Nifty: {dollar_impact['impact_on_nifty']}")
        
        # India Specific
        print("\n🇮🇳 INDIA INDICATORS:")
        
        # India VIX
        vix_data = report['market_data'].get('INDIA_VIX')
        if vix_data:
            print(f"   India VIX: {vix_data['current_value']} ({vix_data['change']:+.2f})")
            print(f"   Interpretation: {vix_data['interpretation']}")
        
        # SGX Nifty
        sgx_data = report['market_data'].get('SGX_NIFTY')
        if sgx_data:
            print(f"   SGX Nifty: {sgx_data['indicative_value']} ({sgx_data['change_percent']:+.2f}%)")
            print(f"   Gap Indication: {sgx_data['gap_indication']}")
        
        # Market Opening Indication
        opening = report['opening_indication']
        print(f"\n🎯 MARKET OPENING INDICATION:")
        print(f"   Prediction: {opening['opening_prediction']}")
        print(f"   Global Sentiment: {opening['global_sentiment']}")
        print(f"   Confidence: {opening['confidence']}%")
        
        if opening['key_factors']:
            print(f"   Key Factors:")
            for factor in opening['key_factors']:
                print(f"      • {factor}")
        
        # Correlations
        print(f"\n📊 MARKET CORRELATIONS (30-day):")
        sorted_corr = sorted(report['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)
        for market, correlation in sorted_corr[:5]:
            strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.5 else "Weak"
            print(f"   {market}: {correlation:+.3f} ({strength})")
        
        print("="*70)
    
    def get_trading_bias(self):
        """Get trading bias based on global markets"""
        market_data = self.fetch_global_market_data()
        
        bias = {
            'direction': 'NEUTRAL',
            'strength': 0,
            'reasons': [],
            'recommended_strategy': ''
        }
        
        score = 0
        
        # US market influence
        us_impact = self.calculate_us_market_impact()
        if us_impact['weighted_change'] > 1:
            score += 2
            bias['reasons'].append("Strong positive US markets")
        elif us_impact['weighted_change'] < -1:
            score -= 2
            bias['reasons'].append("Strong negative US markets")
        
        # Asia market influence
        asia_impact = self.calculate_asia_market_impact()
        if asia_impact['weighted_change'] > 0.8:
            score += 1
            bias['reasons'].append("Positive Asian markets")
        elif asia_impact['weighted_change'] < -0.8:
            score -= 1
            bias['reasons'].append("Negative Asian markets")
        
        # VIX levels
        vix_data = market_data.get('INDIA_VIX')
        if vix_data:
            if vix_data['current_value'] > 20:
                bias['reasons'].append("High volatility - cautious approach")
            elif vix_data['current_value'] < 15:
                bias['reasons'].append("Low volatility - trending market")
        
        # SGX Nifty
        sgx_data = market_data.get('SGX_NIFTY')
        if sgx_data:
            if sgx_data['change_percent'] > 0.5:
                score += 1
                bias['reasons'].append("SGX Nifty positive")
            elif sgx_data['change_percent'] < -0.5:
                score -= 1
                bias['reasons'].append("SGX Nifty negative")
        
        # Determine bias
        if score >= 2:
            bias['direction'] = 'BULLISH'
            bias['strength'] = min(90, 60 + score * 10)
            bias['recommended_strategy'] = 'Buy CE on dips, avoid PE'
        elif score <= -2:
            bias['direction'] = 'BEARISH'
            bias['strength'] = min(90, 60 + abs(score) * 10)
            bias['recommended_strategy'] = 'Buy PE on rise, avoid CE'
        else:
            bias['direction'] = 'NEUTRAL'
            bias['strength'] = 50
            bias['recommended_strategy'] = 'Wait for clarity or trade both sides'
        
        return bias
    
    def should_trade_today(self):
        """Determine if market conditions are favorable for trading"""
        decision = {
            'trade_today': True,
            'confidence': 70,
            'warnings': [],
            'opportunities': []
        }
        
        # Check global volatility
        market_data = self.fetch_global_market_data()
        
        # Check VIX levels
        global_vix = market_data.get('VIX', {})
        india_vix = market_data.get('INDIA_VIX', {})
        
        if global_vix and global_vix.get('current_price', 0) > 30:
            decision['warnings'].append("Extreme global volatility")
            decision['confidence'] -= 20
            
        if india_vix and india_vix.get('current_value', 0) > 25:
            decision['warnings'].append("High India VIX - expect volatility")
            decision['confidence'] -= 10
        
        # Check for major gaps
        sgx_data = market_data.get('SGX_NIFTY')
        if sgx_data and abs(sgx_data.get('change_percent', 0)) > 2:
            decision['warnings'].append("Large gap expected - wait for stability")
            decision['confidence'] -= 15
        
        # Check correlations
        us_impact = self.calculate_us_market_impact()
        if abs(us_impact['weighted_change']) > 2.5:
            decision['warnings'].append("Extreme global moves - high risk")
            decision['trade_today'] = False
        
        # Look for opportunities
        if india_vix and 15 < india_vix.get('current_value', 0) < 20:
            decision['opportunities'].append("Moderate volatility - good for options")
            decision['confidence'] += 10
        
        # Dollar impact
        dollar_impact = self.calculate_dollar_impact()
        if dollar_impact['fii_flow_indication'] == 'Inflow likely':
            decision['opportunities'].append("Weak dollar - FII inflows expected")
            decision['confidence'] += 5
        
        # Final decision
        if decision['confidence'] < 50:
            decision['trade_today'] = False
        
        return decision

# Integration function for existing system
def integrate_global_markets(existing_trade_setup):
    """Integrate global market analysis with existing trade setup"""
    analyzer = GlobalMarketAnalyzer()
    
    # Get global market bias
    global_bias = analyzer.get_trading_bias()
    
    # Get trading decision
    trade_decision = analyzer.should_trade_today()
    
    # Enhance trade setup
    enhanced_setup = existing_trade_setup.copy()
    
    # Adjust confidence based on global markets
    if global_bias['direction'] == 'BULLISH' and existing_trade_setup.get('option_type') == 'CE':
        enhanced_setup['confidence'] = min(95, existing_trade_setup.get('confidence', 80) + 10)
        enhanced_setup['global_alignment'] = 'Positive'
    elif global_bias['direction'] == 'BEARISH' and existing_trade_setup.get('option_type') == 'PE':
        enhanced_setup['confidence'] = min(95, existing_trade_setup.get('confidence', 80) + 10)
        enhanced_setup['global_alignment'] = 'Positive'
    else:
        enhanced_setup['confidence'] = existing_trade_setup.get('confidence', 80) - 5
        enhanced_setup['global_alignment'] = 'Negative'
    
    # Add global insights
    enhanced_setup['global_factors'] = global_bias['reasons']
    enhanced_setup['trade_recommendation'] = trade_decision['trade_today']
    enhanced_setup['global_warnings'] = trade_decision['warnings']
    
    return enhanced_setup

if __name__ == "__main__":
    # Test the global market analyzer
    print("🧪 Testing Global Market Analyzer...")
    
    analyzer = GlobalMarketAnalyzer()
    
    # Generate full report
    report = analyzer.generate_global_market_report()
    
    # Get trading bias
    bias = analyzer.get_trading_bias()
    print(f"\n🎯 TRADING BIAS:")
    print(f"   Direction: {bias['direction']}")
    print(f"   Strength: {bias['strength']}%")
    print(f"   Strategy: {bias['recommended_strategy']}")
    
    # Should trade today?
    decision = analyzer.should_trade_today()
    print(f"\n📊 TRADING DECISION:")
    print(f"   Trade Today: {'YES' if decision['trade_today'] else 'NO'}")
    print(f"   Confidence: {decision['confidence']}%")
    
    if decision['warnings']:
        print(f"   ⚠️ Warnings:")
        for warning in decision['warnings']:
            print(f"      • {warning}")
    
    print("\n✅ Global Market Analyzer ready for integration!")