"""
TradeMind_AI: OI Change Tracker
Tracks Open Interest changes and identifies significant shifts
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time

class OIChangeTracker:
    def __init__(self):
        """Initialize OI Change Tracker"""
        print("📊 Initializing OI Change Tracker...")
        
        # Create data directory for OI history
        self.oi_data_dir = os.path.join("data", "oi_history")
        if not os.path.exists(self.oi_data_dir):
            os.makedirs(self.oi_data_dir)
            
        # OI change thresholds for alerts
        self.significant_change = 10  # 10% change is significant
        self.extreme_change = 20     # 20% change is extreme
        
        print("✅ OI Change Tracker ready!")
    
    def save_oi_snapshot(self, symbol: str, option_chain_data: dict):
        """Save current OI data as snapshot"""
        timestamp = datetime.now()
        filename = f"{symbol}_oi_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.oi_data_dir, filename)
        
        snapshot = {
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'underlying_price': option_chain_data.get('underlying_price', 0),
            'strikes': {}
        }
        
        # Extract OI data for each strike
        for strike_key, strike_data in option_chain_data.get('option_chain', {}).items():
            strike_price = float(strike_key)
            ce_data = strike_data.get('ce', {})
            pe_data = strike_data.get('pe', {})
            
            snapshot['strikes'][strike_price] = {
                'ce_oi': ce_data.get('oi', 0),
                'pe_oi': pe_data.get('oi', 0),
                'ce_volume': ce_data.get('volume', 0),
                'pe_volume': pe_data.get('volume', 0),
                'ce_price': ce_data.get('last_price', 0),
                'pe_price': pe_data.get('last_price', 0)
            }
        
        # Save snapshot
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)
            
        return filepath
    
    def load_previous_snapshot(self, symbol: str, hours_back: int = 1) -> dict:
        """Load previous OI snapshot for comparison"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Find all OI files for symbol
        oi_files = []
        for filename in os.listdir(self.oi_data_dir):
            if filename.startswith(f"{symbol}_oi_") and filename.endswith('.json'):
                filepath = os.path.join(self.oi_data_dir, filename)
                # Extract timestamp from filename
                timestamp_str = filename.replace(f"{symbol}_oi_", "").replace(".json", "")
                try:
                    file_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    if file_time <= cutoff_time:
                        oi_files.append((file_time, filepath))
                except:
                    continue
        
        # Get most recent file before cutoff
        if oi_files:
            oi_files.sort(reverse=True)
            _, filepath = oi_files[0]
            
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
    
    def calculate_oi_changes(self, current_data: dict, previous_data: dict) -> Dict:
        """Calculate OI changes between snapshots"""
        if not previous_data:
            return {}
        
        changes = {
            'time_diff': (datetime.fromisoformat(current_data['timestamp']) - 
                         datetime.fromisoformat(previous_data['timestamp'])).total_seconds() / 3600,
            'price_change': current_data['underlying_price'] - previous_data['underlying_price'],
            'price_change_pct': ((current_data['underlying_price'] - previous_data['underlying_price']) / 
                                previous_data['underlying_price'] * 100),
            'strikes': {},
            'summary': {
                'total_ce_oi_change': 0,
                'total_pe_oi_change': 0,
                'bullish_strikes': [],
                'bearish_strikes': [],
                'significant_changes': []
            }
        }
        
        # Calculate changes for each strike
        for strike_str, current_strike in current_data['strikes'].items():
            strike = float(strike_str)
            if str(strike) in previous_data['strikes']:
                prev_strike = previous_data['strikes'][str(strike)]
                
                # Calculate OI changes
                ce_oi_change = current_strike['ce_oi'] - prev_strike['ce_oi']
                pe_oi_change = current_strike['pe_oi'] - prev_strike['pe_oi']
                
                ce_oi_change_pct = 0
                if prev_strike['ce_oi'] > 0:
                    ce_oi_change_pct = (ce_oi_change / prev_strike['ce_oi']) * 100
                    
                pe_oi_change_pct = 0
                if prev_strike['pe_oi'] > 0:
                    pe_oi_change_pct = (pe_oi_change / prev_strike['pe_oi']) * 100
                
                changes['strikes'][strike] = {
                    'ce_oi_change': ce_oi_change,
                    'ce_oi_change_pct': round(ce_oi_change_pct, 2),
                    'pe_oi_change': pe_oi_change,
                    'pe_oi_change_pct': round(pe_oi_change_pct, 2),
                    'ce_price_change': current_strike['ce_price'] - prev_strike['ce_price'],
                    'pe_price_change': current_strike['pe_price'] - prev_strike['pe_price']
                }
                
                # Update summary
                changes['summary']['total_ce_oi_change'] += ce_oi_change
                changes['summary']['total_pe_oi_change'] += pe_oi_change
                
                # Identify significant changes
                if abs(ce_oi_change_pct) >= self.significant_change:
                    changes['summary']['significant_changes'].append({
                        'strike': strike,
                        'type': 'CE',
                        'change_pct': ce_oi_change_pct,
                        'interpretation': self._interpret_oi_change('CE', ce_oi_change_pct)
                    })
                    
                if abs(pe_oi_change_pct) >= self.significant_change:
                    changes['summary']['significant_changes'].append({
                        'strike': strike,
                        'type': 'PE',
                        'change_pct': pe_oi_change_pct,
                        'interpretation': self._interpret_oi_change('PE', pe_oi_change_pct)
                    })
                
                # Identify bullish/bearish strikes
                if ce_oi_change > pe_oi_change and ce_oi_change > 0:
                    changes['summary']['bullish_strikes'].append(strike)
                elif pe_oi_change > ce_oi_change and pe_oi_change > 0:
                    changes['summary']['bearish_strikes'].append(strike)
        
        return changes
    
    def _interpret_oi_change(self, option_type: str, change_pct: float) -> str:
        """Interpret OI change meaning"""
        if option_type == 'CE':
            if change_pct > self.extreme_change:
                return "Heavy CALL writing - Strong RESISTANCE"
            elif change_pct > self.significant_change:
                return "CALL writing - Resistance building"
            elif change_pct < -self.extreme_change:
                return "Heavy CALL unwinding - Resistance breaking"
            elif change_pct < -self.significant_change:
                return "CALL unwinding - Bullish"
        else:  # PE
            if change_pct > self.extreme_change:
                return "Heavy PUT writing - Strong SUPPORT"
            elif change_pct > self.significant_change:
                return "PUT writing - Support building"
            elif change_pct < -self.extreme_change:
                return "Heavy PUT unwinding - Support breaking"
            elif change_pct < -self.significant_change:
                return "PUT unwinding - Bearish"
        
        return "Normal activity"
    
    def identify_key_levels(self, oi_changes: Dict) -> Dict:
        """Identify key support and resistance levels from OI"""
        key_levels = {
            'max_call_oi_strike': None,
            'max_put_oi_strike': None,
            'max_call_addition': None,
            'max_put_addition': None,
            'pcr_trend': 'Neutral',
            'market_sentiment': 'Neutral'
        }
        
        if not oi_changes or 'strikes' not in oi_changes:
            return key_levels
        
        # Find strikes with maximum OI changes
        max_ce_addition = 0
        max_pe_addition = 0
        
        for strike, changes in oi_changes['strikes'].items():
            if changes['ce_oi_change'] > max_ce_addition:
                max_ce_addition = changes['ce_oi_change']
                key_levels['max_call_addition'] = strike
                
            if changes['pe_oi_change'] > max_pe_addition:
                max_pe_addition = changes['pe_oi_change']
                key_levels['max_put_addition'] = strike
        
        # Determine market sentiment
        total_ce_change = oi_changes['summary']['total_ce_oi_change']
        total_pe_change = oi_changes['summary']['total_pe_oi_change']
        
        if total_pe_change > total_ce_change * 1.5:
            key_levels['market_sentiment'] = 'Bullish (PUT writing)'
            key_levels['pcr_trend'] = 'Increasing'
        elif total_ce_change > total_pe_change * 1.5:
            key_levels['market_sentiment'] = 'Bearish (CALL writing)'
            key_levels['pcr_trend'] = 'Decreasing'
        else:
            key_levels['market_sentiment'] = 'Neutral'
            key_levels['pcr_trend'] = 'Stable'
        
        return key_levels
    
    def display_oi_analysis(self, symbol: str, oi_changes: Dict, key_levels: Dict):
        """Display OI change analysis"""
        print(f"\n{'='*70}")
        print(f"📊 OI CHANGE ANALYSIS - {symbol}")
        print(f"{'='*70}")
        
        if not oi_changes:
            print("❌ No previous data available for comparison")
            return
        
        print(f"\n⏰ Time Period: {oi_changes['time_diff']:.1f} hours")
        print(f"📈 Price Change: ₹{oi_changes['price_change']:.2f} ({oi_changes['price_change_pct']:.2f}%)")
        
        print(f"\n📊 Overall OI Changes:")
        print(f"   Total CALL OI Change: {oi_changes['summary']['total_ce_oi_change']:,}")
        print(f"   Total PUT OI Change: {oi_changes['summary']['total_pe_oi_change']:,}")
        
        if oi_changes['summary']['significant_changes']:
            print(f"\n🚨 Significant OI Changes:")
            for change in oi_changes['summary']['significant_changes'][:5]:  # Top 5
                print(f"   {change['strike']} {change['type']}: "
                      f"{change['change_pct']:+.1f}% - {change['interpretation']}")
        
        print(f"\n🎯 Key Levels:")
        if key_levels['max_call_addition']:
            print(f"   Max CALL Addition: {key_levels['max_call_addition']} (Resistance)")
        if key_levels['max_put_addition']:
            print(f"   Max PUT Addition: {key_levels['max_put_addition']} (Support)")
        
        print(f"\n📈 Market Sentiment: {key_levels['market_sentiment']}")
        print(f"   PCR Trend: {key_levels['pcr_trend']}")
        
        if oi_changes['summary']['bullish_strikes']:
            print(f"\n🟢 Bullish Strikes: {oi_changes['summary']['bullish_strikes'][:3]}")
        if oi_changes['summary']['bearish_strikes']:
            print(f"\n🔴 Bearish Strikes: {oi_changes['summary']['bearish_strikes'][:3]}")
        
        print(f"{'='*70}")
    
    def generate_oi_report(self, symbol: str, current_data: dict, previous_data: dict = None) -> str:
        """Generate detailed OI change report"""
        if not previous_data:
            previous_data = self.load_previous_snapshot(symbol, hours_back=1)
        
        oi_changes = self.calculate_oi_changes(current_data, previous_data)
        key_levels = self.identify_key_levels(oi_changes)
        
        # Display analysis
        self.display_oi_analysis(symbol, oi_changes, key_levels)
        
        return oi_changes

# Test function
if __name__ == "__main__":
    tracker = OIChangeTracker()
    
    # Test with sample data
    print("🧪 Testing OI Change Tracker...")
    
    # Simulate current data
    current_data = {
        'symbol': 'NIFTY',
        'timestamp': datetime.now().isoformat(),
        'underlying_price': 25100,
        'strikes': {
            '25000': {'ce_oi': 1000000, 'pe_oi': 800000, 'ce_volume': 50000, 'pe_volume': 40000, 'ce_price': 150, 'pe_price': 50},
            '25100': {'ce_oi': 1200000, 'pe_oi': 1100000, 'ce_volume': 60000, 'pe_volume': 55000, 'ce_price': 80, 'pe_price': 80},
            '25200': {'ce_oi': 900000, 'pe_oi': 600000, 'ce_volume': 45000, 'pe_volume': 30000, 'ce_price': 40, 'pe_price': 140}
        }
    }
    
    # Simulate previous data
    previous_data = {
        'symbol': 'NIFTY',
        'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
        'underlying_price': 25050,
        'strikes': {
            '25000': {'ce_oi': 900000, 'pe_oi': 850000, 'ce_volume': 45000, 'pe_volume': 42000, 'ce_price': 160, 'pe_price': 45},
            '25100': {'ce_oi': 1000000, 'pe_oi': 1000000, 'ce_volume': 50000, 'pe_volume': 50000, 'ce_price': 85, 'pe_price': 75},
            '25200': {'ce_oi': 800000, 'pe_oi': 650000, 'ce_volume': 40000, 'pe_volume': 32000, 'ce_price': 45, 'pe_price': 135}
        }
    }
    
    tracker.generate_oi_report('NIFTY', current_data, previous_data)