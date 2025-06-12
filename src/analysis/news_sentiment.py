"""
TradeMind_AI: News & Sentiment Analysis Module
Analyzes market news, social sentiment, and economic events
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

class NewsSentimentAnalyzer:
    def __init__(self):
        """Initialize News & Sentiment Analyzer"""
        print("📰 Initializing News & Sentiment Analyzer...")
        
        # Load environment
        load_dotenv()
        
        # You can add API keys here later for real news APIs
        # For now, we'll simulate the functionality
        self.news_api_key = os.getenv('NEWS_API_KEY', 'demo')
        
        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'gain', 'boost', 'breakthrough', 'record high',
            'bullish', 'positive', 'growth', 'outperform', 'upgrade', 'beat',
            'strong', 'rise', 'climb', 'soar', 'jump', 'advance'
        ]
        
        self.bearish_keywords = [
            'crash', 'fall', 'drop', 'decline', 'plunge', 'slump', 'bearish',
            'negative', 'weakness', 'concern', 'risk', 'downgrade', 'miss',
            'weak', 'sink', 'tumble', 'slide', 'retreat', 'loss'
        ]
        
        self.neutral_keywords = [
            'steady', 'unchanged', 'flat', 'stable', 'mixed', 'consolidate',
            'range', 'sideways', 'hold'
        ]
        
        print("✅ News & Sentiment Analyzer ready!")
    
    def fetch_market_news(self, symbol='NIFTY'):
        """Fetch latest market news (simulated for demo)"""
        print(f"\n📰 Fetching {symbol} market news...")
        
        # Simulated news data (in production, use real news API)
        simulated_news = [
            {
                'title': f'{symbol} surges to new record high on strong FII buying',
                'description': 'Foreign institutional investors pumped in Rs 3,500 crore today',
                'source': 'Economic Times',
                'published': datetime.now() - timedelta(hours=1),
                'sentiment': None
            },
            {
                'title': 'RBI keeps interest rates unchanged, markets react positively',
                'description': 'Central bank maintains accommodative stance to support growth',
                'source': 'Business Standard',
                'published': datetime.now() - timedelta(hours=2),
                'sentiment': None
            },
            {
                'title': 'Global markets mixed ahead of Fed decision',
                'description': 'Asian markets show caution as investors await US Federal Reserve meeting',
                'source': 'Reuters',
                'published': datetime.now() - timedelta(hours=3),
                'sentiment': None
            },
            {
                'title': f'{symbol} options show bullish bias, PCR at 1.3',
                'description': 'Put-Call ratio indicates strong bullish sentiment among traders',
                'source': 'Moneycontrol',
                'published': datetime.now() - timedelta(hours=4),
                'sentiment': None
            },
            {
                'title': 'IT stocks lead market rally on strong Q3 earnings',
                'description': 'TCS and Infosys beat street estimates, guide higher for FY25',
                'source': 'CNBC',
                'published': datetime.now() - timedelta(hours=5),
                'sentiment': None
            }
        ]
        
        # Analyze sentiment for each news
        for news in simulated_news:
            news['sentiment'] = self.analyze_text_sentiment(
                news['title'] + ' ' + news['description']
            )
        
        return simulated_news
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text"""
        text_lower = text.lower()
        
        bullish_score = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_score = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        if bullish_score > bearish_score:
            return 'BULLISH'
        elif bearish_score > bullish_score:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def fetch_economic_events(self):
        """Fetch important economic events"""
        print("\n📅 Fetching economic events...")
        
        # Simulated economic calendar
        events = [
            {
                'event': 'RBI Monetary Policy',
                'date': datetime.now() + timedelta(days=2),
                'importance': 'HIGH',
                'forecast': 'No change in rates',
                'impact': 'MEDIUM'
            },
            {
                'event': 'India GDP Data',
                'date': datetime.now() + timedelta(days=5),
                'importance': 'HIGH',
                'forecast': '7.2% YoY growth',
                'impact': 'HIGH'
            },
            {
                'event': 'US Fed Meeting',
                'date': datetime.now() + timedelta(days=7),
                'importance': 'HIGH',
                'forecast': 'Hawkish tone expected',
                'impact': 'HIGH'
            },
            {
                'event': 'India Inflation Data',
                'date': datetime.now() + timedelta(days=10),
                'importance': 'MEDIUM',
                'forecast': '5.8% YoY',
                'impact': 'MEDIUM'
            }
        ]
        
        return events
    
    def analyze_social_sentiment(self, symbol='NIFTY'):
        """Analyze social media sentiment (simulated)"""
        print(f"\n💬 Analyzing social sentiment for {symbol}...")
        
        # Simulated social media sentiment
        social_data = {
            'twitter_mentions': 15420,
            'reddit_posts': 342,
            'stocktwits_messages': 891,
            'overall_sentiment': {
                'bullish': 62,
                'bearish': 23,
                'neutral': 15
            },
            'trending_topics': [
                '#NiftyToTheMoon',
                'BuyTheDip',
                'OptionsTrading',
                'NiftyPrediction'
            ],
            'influencer_sentiment': 'BULLISH'
        }
        
        return social_data
    
    def calculate_sentiment_score(self, news_data, social_data):
        """Calculate overall sentiment score"""
        score = 0
        
        # News sentiment scoring
        for news in news_data:
            if news['sentiment'] == 'BULLISH':
                score += 10
            elif news['sentiment'] == 'BEARISH':
                score -= 10
        
        # Social sentiment scoring
        social_sentiment = social_data['overall_sentiment']
        bullish_pct = social_sentiment['bullish']
        bearish_pct = social_sentiment['bearish']
        
        if bullish_pct > 60:
            score += 20
        elif bullish_pct > 50:
            score += 10
        
        if bearish_pct > 40:
            score -= 15
        
        # Normalize score to -100 to +100
        score = max(-100, min(100, score))
        
        return score
    
    def generate_sentiment_report(self, symbol='NIFTY'):
        """Generate comprehensive sentiment report"""
        print(f"\n📊 Generating Sentiment Report for {symbol}...")
        
        # Fetch all data
        news_data = self.fetch_market_news(symbol)
        economic_events = self.fetch_economic_events()
        social_data = self.analyze_social_sentiment(symbol)
        
        # Calculate sentiment score
        sentiment_score = self.calculate_sentiment_score(news_data, social_data)
        
        # Determine market sentiment
        if sentiment_score > 30:
            market_sentiment = "STRONGLY BULLISH"
            confidence = min(95, 70 + sentiment_score/2)
        elif sentiment_score > 10:
            market_sentiment = "BULLISH"
            confidence = 60 + sentiment_score
        elif sentiment_score < -30:
            market_sentiment = "STRONGLY BEARISH"
            confidence = min(95, 70 + abs(sentiment_score)/2)
        elif sentiment_score < -10:
            market_sentiment = "BEARISH"
            confidence = 60 + abs(sentiment_score)
        else:
            market_sentiment = "NEUTRAL"
            confidence = 50
        
        report = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'sentiment_score': sentiment_score,
            'market_sentiment': market_sentiment,
            'confidence': confidence,
            'news_summary': news_data[:3],  # Top 3 news
            'upcoming_events': economic_events[:2],  # Next 2 events
            'social_metrics': social_data,
            'trading_recommendation': self.generate_trading_recommendation(
                sentiment_score, market_sentiment
            )
        }
        
        return report
    
    def generate_trading_recommendation(self, score, sentiment):
        """Generate trading recommendation based on sentiment"""
        if sentiment == "STRONGLY BULLISH":
            return {
                'action': 'BUY CE',
                'reasoning': 'Strong positive sentiment across news and social media',
                'risk_level': 'MODERATE',
                'suggested_strategy': 'Buy ATM or slightly OTM Call options'
            }
        elif sentiment == "BULLISH":
            return {
                'action': 'BUY CE (Conservative)',
                'reasoning': 'Positive sentiment but not overwhelming',
                'risk_level': 'LOW',
                'suggested_strategy': 'Bull spread or ITM Call options'
            }
        elif sentiment == "STRONGLY BEARISH":
            return {
                'action': 'BUY PE',
                'reasoning': 'Strong negative sentiment detected',
                'risk_level': 'MODERATE',
                'suggested_strategy': 'Buy ATM or slightly OTM Put options'
            }
        elif sentiment == "BEARISH":
            return {
                'action': 'BUY PE (Conservative)',
                'reasoning': 'Negative sentiment but not extreme',
                'risk_level': 'LOW',
                'suggested_strategy': 'Bear spread or ITM Put options'
            }
        else:
            return {
                'action': 'WAIT/STRADDLE',
                'reasoning': 'Mixed sentiment - market direction unclear',
                'risk_level': 'HIGH',
                'suggested_strategy': 'Wait for clarity or consider straddle/strangle'
            }
    
    def display_sentiment_report(self, report):
        """Display sentiment report in readable format"""
        print("\n" + "="*70)
        print(f"📊 SENTIMENT ANALYSIS REPORT - {report['symbol']}")
        print("="*70)
        
        print(f"\n🎯 Overall Market Sentiment: {report['market_sentiment']}")
        print(f"📈 Sentiment Score: {report['sentiment_score']}/100")
        print(f"🎪 Confidence Level: {report['confidence']:.1f}%")
        print(f"⏰ Generated: {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📰 Top News Headlines:")
        for i, news in enumerate(report['news_summary'], 1):
            print(f"{i}. {news['title']}")
            print(f"   Sentiment: {news['sentiment']} | Source: {news['source']}")
        
        print("\n📅 Upcoming Economic Events:")
        for event in report['upcoming_events']:
            print(f"• {event['event']} - {event['date'].strftime('%Y-%m-%d')}")
            print(f"  Importance: {event['importance']} | Impact: {event['impact']}")
        
        print("\n💬 Social Media Metrics:")
        social = report['social_metrics']
        print(f"• Twitter Mentions: {social['twitter_mentions']:,}")
        print(f"• Sentiment: {social['overall_sentiment']['bullish']}% Bullish, "
              f"{social['overall_sentiment']['bearish']}% Bearish")
        print(f"• Trending: {', '.join(social['trending_topics'][:2])}")
        
        print("\n🎯 TRADING RECOMMENDATION:")
        rec = report['trading_recommendation']
        print(f"Action: {rec['action']}")
        print(f"Reasoning: {rec['reasoning']}")
        print(f"Risk Level: {rec['risk_level']}")
        print(f"Strategy: {rec['suggested_strategy']}")
        
        print("="*70)

# Test the module
if __name__ == "__main__":
    print("🌟 Testing News & Sentiment Analysis Module")
    
    # Create analyzer instance
    analyzer = NewsSentimentAnalyzer()
    
    # Generate sentiment report for NIFTY
    print("\n1️⃣ Analyzing NIFTY sentiment...")
    nifty_report = analyzer.generate_sentiment_report('NIFTY')
    analyzer.display_sentiment_report(nifty_report)
    
    # Generate sentiment report for BANKNIFTY
    print("\n2️⃣ Analyzing BANKNIFTY sentiment...")
    banknifty_report = analyzer.generate_sentiment_report('BANKNIFTY')
    analyzer.display_sentiment_report(banknifty_report)
    
    print("\n✅ News & Sentiment module ready for integration!")