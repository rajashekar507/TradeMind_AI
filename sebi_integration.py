"""
TradeMind_AI: SEBI Announcements Integration
Fetches and analyzes SEBI announcements for market impact
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Tuple
import logging

class SEBIAnnouncementsIntegration:
    def __init__(self):
        """Initialize SEBI Announcements Integration"""
        self.logger = logging.getLogger(__name__)
        print("📋 Initializing SEBI Announcements Integration...")
        
        # SEBI URLs
        self.sebi_urls = {
            'circulars': 'https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=4&ssid=47&smid=0',
            'press_releases': 'https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=6&smid=0',
            'orders': 'https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=5&ssid=56&smid=0',
            'board_meetings': 'https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=2&ssid=18&smid=0'
        }
        
        # Keywords for market impact analysis
        self.high_impact_keywords = [
            'margin', 'circuit breaker', 'trading halt', 'ban', 'suspension',
            'penalty', 'investigation', 'fraud', 'manipulation', 'insider',
            'delisting', 'merger', 'acquisition', 'buyback', 'split'
        ]
        
        self.medium_impact_keywords = [
            'disclosure', 'compliance', 'amendment', 'modification', 'clarification',
            'guidelines', 'framework', 'norms', 'regulations', 'consultation'
        ]
        
        self.sector_keywords = {
            'banking': ['bank', 'nbfc', 'financial institution', 'rbi'],
            'it': ['technology', 'software', 'it services', 'digital'],
            'pharma': ['pharmaceutical', 'drug', 'healthcare', 'medical'],
            'auto': ['automobile', 'vehicle', 'auto ancillary', 'ev'],
            'infra': ['infrastructure', 'construction', 'real estate', 'cement']
        }
        
        # Cache for announcements
        self.announcement_cache = []
        self.last_fetch_time = None
        self.cache_duration = 900  # 15 minutes
        
        print("✅ SEBI Announcements Integration ready!")
    
    def fetch_latest_announcements(self, category='circulars', limit=10):
        """Fetch latest SEBI announcements"""
        try:
            # Check cache
            if self._is_cache_valid():
                return self._filter_cached_announcements(category, limit)
            
            print(f"\n📥 Fetching SEBI {category}...")
            
            # Simulate fetching (in production, use actual web scraping)
            announcements = self._simulate_announcements()
            
            # Update cache
            self.announcement_cache = announcements
            self.last_fetch_time = datetime.now()
            
            return announcements[:limit]
            
        except Exception as e:
            self.logger.error(f"Error fetching SEBI announcements: {e}")
            return []
    
    def _simulate_announcements(self):
        """Simulate SEBI announcements for demo"""
        current_time = datetime.now()
        
        simulated_announcements = [
            {
                'id': 'SEBI/HO/MIRSD/001/2025',
                'title': 'Revised Margin Requirements for Equity Derivatives',
                'category': 'circular',
                'date': current_time - timedelta(hours=2),
                'content': 'SEBI has revised margin requirements for equity derivatives trading...',
                'impact': 'high',
                'sectors_affected': ['all'],
                'url': 'https://www.sebi.gov.in/circular/2025/example1.html'
            },
            {
                'id': 'SEBI/PR/002/2025',
                'title': 'SEBI Board approves amendments to Insider Trading Regulations',
                'category': 'press_release',
                'date': current_time - timedelta(hours=5),
                'content': 'The SEBI Board has approved key amendments to strengthen insider trading norms...',
                'impact': 'medium',
                'sectors_affected': ['all'],
                'url': 'https://www.sebi.gov.in/media/2025/example2.html'
            },
            {
                'id': 'SEBI/HO/CFID/003/2025',
                'title': 'New disclosure norms for listed banks',
                'category': 'circular',
                'date': current_time - timedelta(days=1),
                'content': 'Enhanced disclosure requirements for listed banking companies...',
                'impact': 'medium',
                'sectors_affected': ['banking'],
                'url': 'https://www.sebi.gov.in/circular/2025/example3.html'
            },
            {
                'id': 'SEBI/ADJ/004/2025',
                'title': 'Order in matter of XYZ Ltd - Trading suspension',
                'category': 'order',
                'date': current_time - timedelta(days=1),
                'content': 'Trading suspended in shares of XYZ Ltd pending investigation...',
                'impact': 'high',
                'sectors_affected': ['pharma'],
                'url': 'https://www.sebi.gov.in/order/2025/example4.html'
            },
            {
                'id': 'SEBI/HO/MRD/005/2025',
                'title': 'Consultation paper on algorithmic trading framework',
                'category': 'consultation',
                'date': current_time - timedelta(days=2),
                'content': 'SEBI seeks public comments on proposed algo trading regulations...',
                'impact': 'low',
                'sectors_affected': ['all'],
                'url': 'https://www.sebi.gov.in/consultation/2025/example5.html'
            }
        ]
        
        # Analyze impact for each announcement
        for announcement in simulated_announcements:
            announcement['impact_score'] = self._calculate_impact_score(announcement)
            announcement['market_impact'] = self._analyze_market_impact(announcement)
        
        return simulated_announcements
    
    def _is_cache_valid(self):
        """Check if cache is still valid"""
        if not self.last_fetch_time:
            return False
        
        time_since_fetch = (datetime.now() - self.last_fetch_time).total_seconds()
        return time_since_fetch < self.cache_duration
    
    def _filter_cached_announcements(self, category, limit):
        """Filter cached announcements by category"""
        if category == 'all':
            return self.announcement_cache[:limit]
        
        filtered = [a for a in self.announcement_cache if a['category'] == category]
        return filtered[:limit]
    
    def _calculate_impact_score(self, announcement):
        """Calculate impact score (0-100) for an announcement"""
        score = 50  # Base score
        
        title_lower = announcement['title'].lower()
        content_lower = announcement['content'].lower()
        
        # Check high impact keywords
        for keyword in self.high_impact_keywords:
            if keyword in title_lower:
                score += 15
            if keyword in content_lower:
                score += 10
        
        # Check medium impact keywords
        for keyword in self.medium_impact_keywords:
            if keyword in title_lower:
                score += 8
            if keyword in content_lower:
                score += 5
        
        # Category-based scoring
        category_scores = {
            'order': 20,
            'circular': 15,
            'press_release': 10,
            'consultation': 5
        }
        score += category_scores.get(announcement['category'], 0)
        
        # Recency bonus
        hours_old = (datetime.now() - announcement['date']).total_seconds() / 3600
        if hours_old < 6:
            score += 20
        elif hours_old < 24:
            score += 10
        elif hours_old < 72:
            score += 5
        
        return min(100, score)
    
    def _analyze_market_impact(self, announcement):
        """Analyze potential market impact of announcement"""
        impact_analysis = {
            'immediate_impact': 'neutral',
            'sectors_affected': announcement['sectors_affected'],
            'trading_recommendation': 'continue',
            'risk_level': 'normal',
            'key_points': []
        }
        
        title_lower = announcement['title'].lower()
        
        # Check for immediate trading impact
        if any(word in title_lower for word in ['suspension', 'halt', 'ban']):
            impact_analysis['immediate_impact'] = 'negative'
            impact_analysis['trading_recommendation'] = 'avoid affected stocks'
            impact_analysis['risk_level'] = 'high'
            impact_analysis['key_points'].append('Trading restrictions in place')
        
        elif any(word in title_lower for word in ['margin', 'circuit']):
            impact_analysis['immediate_impact'] = 'caution'
            impact_analysis['trading_recommendation'] = 'reduce position sizes'
            impact_analysis['risk_level'] = 'elevated'
            impact_analysis['key_points'].append('Margin requirements changed')
        
        elif any(word in title_lower for word in ['disclosure', 'compliance']):
            impact_analysis['immediate_impact'] = 'neutral'
            impact_analysis['trading_recommendation'] = 'monitor affected stocks'
            impact_analysis['risk_level'] = 'normal'
            impact_analysis['key_points'].append('Regulatory compliance update')
        
        # Sector-specific analysis
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                if sector not in impact_analysis['sectors_affected']:
                    impact_analysis['sectors_affected'].append(sector)
        
        return impact_analysis
    
    def get_trading_impact_summary(self):
        """Get summary of trading impact from recent SEBI announcements"""
        # Fetch latest announcements
        recent_announcements = self.fetch_latest_announcements('all', limit=20)
        
        summary = {
            'timestamp': datetime.now(),
            'high_impact_count': 0,
            'sectors_with_updates': set(),
            'trading_restrictions': [],
            'margin_changes': [],
            'overall_market_impact': 'neutral',
            'key_announcements': [],
            'recommendations': []
        }
        
        for announcement in recent_announcements:
            # Count high impact announcements
            if announcement['impact'] == 'high':
                summary['high_impact_count'] += 1
            
            # Track sectors
            for sector in announcement['sectors_affected']:
                summary['sectors_with_updates'].add(sector)
            
            # Check for trading restrictions
            if 'suspension' in announcement['title'].lower() or 'halt' in announcement['title'].lower():
                summary['trading_restrictions'].append({
                    'title': announcement['title'],
                    'date': announcement['date'],
                    'impact': announcement['market_impact']
                })
            
            # Check for margin changes
            if 'margin' in announcement['title'].lower():
                summary['margin_changes'].append({
                    'title': announcement['title'],
                    'date': announcement['date'],
                    'details': announcement['content'][:100] + '...'
                })
            
            # Add to key announcements if high impact
            if announcement['impact_score'] > 70:
                summary['key_announcements'].append({
                    'id': announcement['id'],
                    'title': announcement['title'],
                    'impact': announcement['impact'],
                    'date': announcement['date']
                })
        
        # Convert set to list for JSON serialization
        summary['sectors_with_updates'] = list(summary['sectors_with_updates'])
        
        # Determine overall market impact
        if summary['high_impact_count'] >= 3:
            summary['overall_market_impact'] = 'high_caution'
            summary['recommendations'].append('Consider reducing position sizes')
        elif summary['trading_restrictions']:
            summary['overall_market_impact'] = 'elevated_caution'
            summary['recommendations'].append('Avoid stocks with trading restrictions')
        elif summary['margin_changes']:
            summary['overall_market_impact'] = 'moderate_caution'
            summary['recommendations'].append('Review margin requirements before trading')
        
        # Add general recommendations
        if summary['sectors_with_updates']:
            summary['recommendations'].append(f"Monitor {', '.join(summary['sectors_with_updates'])} sectors closely")
        
        return summary
    
    def check_stock_specific_announcements(self, stock_symbol):
        """Check if there are any SEBI announcements for a specific stock"""
        announcements = self.fetch_latest_announcements('all', limit=50)
        
        stock_announcements = []
        for announcement in announcements:
            if stock_symbol.lower() in announcement['title'].lower() or \
               stock_symbol.lower() in announcement['content'].lower():
                stock_announcements.append(announcement)
        
        return stock_announcements
    
    def integrate_with_trading_decision(self, existing_trade_setup):
        """Integrate SEBI announcement analysis with trading decision"""
        # Get trading impact summary
        impact_summary = self.get_trading_impact_summary()
        
        # Enhanced trade setup
        enhanced_setup = existing_trade_setup.copy()
        
        # Adjust confidence based on SEBI announcements
        if impact_summary['overall_market_impact'] == 'high_caution':
            enhanced_setup['confidence'] = max(40, existing_trade_setup.get('confidence', 80) - 30)
            enhanced_setup['sebi_warning'] = 'High regulatory activity detected'
        elif impact_summary['overall_market_impact'] == 'elevated_caution':
            enhanced_setup['confidence'] = max(50, existing_trade_setup.get('confidence', 80) - 20)
            enhanced_setup['sebi_warning'] = 'Trading restrictions in place for some stocks'
        elif impact_summary['overall_market_impact'] == 'moderate_caution':
            enhanced_setup['confidence'] = max(60, existing_trade_setup.get('confidence', 80) - 10)
            enhanced_setup['sebi_warning'] = 'Margin requirements updated'
        
        # Add SEBI insights
        enhanced_setup['sebi_updates'] = {
            'high_impact_announcements': impact_summary['high_impact_count'],
            'trading_restrictions': len(impact_summary['trading_restrictions']),
            'sectors_affected': impact_summary['sectors_with_updates'],
            'recommendations': impact_summary['recommendations']
        }
        
        return enhanced_setup
    
    def display_announcement_summary(self, summary):
        """Display SEBI announcement summary in readable format"""
        print("\n" + "="*70)
        print("📋 SEBI ANNOUNCEMENTS SUMMARY")
        print("="*70)
        
        print(f"\n⏰ Generated: {summary['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚠️  High Impact Announcements: {summary['high_impact_count']}")
        print(f"🎯 Overall Market Impact: {summary['overall_market_impact'].upper()}")
        
        if summary['sectors_with_updates']:
            print(f"\n📊 Sectors with Updates:")
            for sector in summary['sectors_with_updates']:
                print(f"   • {sector.capitalize()}")
        
        if summary['trading_restrictions']:
            print(f"\n🚫 Trading Restrictions:")
            for restriction in summary['trading_restrictions']:
                print(f"   • {restriction['title']}")
                print(f"     Date: {restriction['date'].strftime('%Y-%m-%d %H:%M')}")
        
        if summary['margin_changes']:
            print(f"\n💰 Margin Changes:")
            for change in summary['margin_changes']:
                print(f"   • {change['title']}")
        
        if summary['key_announcements']:
            print(f"\n📌 Key Announcements:")
            for announcement in summary['key_announcements'][:3]:
                print(f"   • [{announcement['impact'].upper()}] {announcement['title']}")
        
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        print("="*70)


# Test the module
if __name__ == "__main__":
    print("🧪 Testing SEBI Announcements Integration...")
    
    # Create instance
    sebi = SEBIAnnouncementsIntegration()
    
    # Test 1: Fetch latest circulars
    print("\n1️⃣ Fetching latest SEBI circulars...")
    circulars = sebi.fetch_latest_announcements('circular', limit=5)
    for circular in circulars:
        print(f"   • {circular['title']} (Impact: {circular['impact']})")
    
    # Test 2: Get trading impact summary
    print("\n2️⃣ Getting trading impact summary...")
    summary = sebi.get_trading_impact_summary()
    sebi.display_announcement_summary(summary)
    
    # Test 3: Check stock-specific announcements
    print("\n3️⃣ Checking stock-specific announcements...")
    stock_announcements = sebi.check_stock_specific_announcements('RELIANCE')
    if stock_announcements:
        print(f"   Found {len(stock_announcements)} announcements for RELIANCE")
    else:
        print("   No specific announcements found for RELIANCE")
    
    # Test 4: Integration with trading decision
    print("\n4️⃣ Testing integration with trading decision...")
    sample_trade_setup = {
        'symbol': 'NIFTY',
        'option_type': 'CE',
        'confidence': 80,
        'entry_price': 25000
    }
    
    enhanced_setup = sebi.integrate_with_trading_decision(sample_trade_setup)
    print(f"   Original confidence: {sample_trade_setup['confidence']}%")
    print(f"   Adjusted confidence: {enhanced_setup['confidence']}%")
    if 'sebi_warning' in enhanced_setup:
        print(f"   Warning: {enhanced_setup['sebi_warning']}")
    
    print("\n✅ SEBI Announcements Integration ready for use!")