"""
TradeMind_AI: Enhanced News Sentiment & Economic Calendar Analysis
Includes NSE/SEBI announcements, SGX Nifty, and economic events
"""

import os
import json
import logging
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
from bs4 import BeautifulSoup
import pandas as pd

from config.constants import GLOBAL_MARKETS, NOTIFICATIONS
from utils.rate_limiter import rate_limit

logger = logging.getLogger('NewsSentimentAnalyzer')


class SentimentScore(Enum):
    """Sentiment classification"""
    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class NewsItem:
    """News item structure"""
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    sentiment: SentimentScore
    relevance: float
    keywords: List[str]
    impact: str  # HIGH, MEDIUM, LOW


@dataclass
class EconomicEvent:
    """Economic calendar event"""
    event_name: str
    country: str
    importance: str  # HIGH, MEDIUM, LOW
    scheduled_time: datetime
    actual_value: Optional[float]
    forecast_value: Optional[float]
    previous_value: Optional[float]
    impact_on_market: str


class EnhancedNewsSentimentAnalyzer:
    """
    Advanced news sentiment analyzer with multiple sources
    """
    
    def __init__(self):
        """Initialize news sentiment analyzer"""
        self.logger = logging.getLogger('EnhancedNewsSentiment')
        
        # API keys
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'gain', 'rise', 'bullish', 'positive', 'growth',
            'record high', 'breakthrough', 'upgrade', 'beat expectations',
            'strong', 'robust', 'outperform', 'accelerate', 'boost'
        ]
        
        self.bearish_keywords = [
            'fall', 'drop', 'decline', 'bearish', 'negative', 'loss',
            'record low', 'crash', 'downgrade', 'miss expectations',
            'weak', 'slowdown', 'underperform', 'recession', 'concern'
        ]
        
        # Market-specific keywords
        self.market_keywords = {
            'NIFTY': ['nifty', 'nse', 'sensex', 'indian market', 'india stock'],
            'BANKNIFTY': ['bank nifty', 'banking sector', 'hdfc', 'icici', 'sbi', 'kotak'],
            'GLOBAL': ['fed', 'fomc', 'dow', 'nasdaq', 'crude', 'gold', 'dollar']
        }
        
        # News sources
        self.rss_feeds = {
            'moneycontrol': 'https://www.moneycontrol.com/rss/marketreports.xml',
            'economictimes': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'reuters': 'https://www.reuters.com/markets/rss',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
        
        # Cache
        self.news_cache = []
        self.economic_calendar = []
        self.last_update = None
        
        self.logger.info("Enhanced News Sentiment Analyzer initialized")
    
    def fetch_all_news(self) -> List[NewsItem]:
        """Fetch news from all sources"""
        all_news = []
        
        # 1. Fetch from NewsAPI
        if self.news_api_key:
            all_news.extend(self._fetch_newsapi())
        
        # 2. Fetch from RSS feeds
        for source, url in self.rss_feeds.items():
            try:
                news_items = self._fetch_rss_feed(url, source)
                all_news.extend(news_items)
            except Exception as e:
                self.logger.error(f"Error fetching {source}: {e}")
        
        # 3. Fetch NSE announcements
        all_news.extend(self._fetch_nse_announcements())
        
        # 4. Fetch SEBI updates
        all_news.extend(self._fetch_sebi_updates())
        
        # Remove duplicates and sort by time
        unique_news = self._deduplicate_news(all_news)
        unique_news.sort(key=lambda x: x.published, reverse=True)
        
        # Cache results
        self.news_cache = unique_news
        self.last_update = datetime.now()
        
        return unique_news
    
    @rate_limit('NEWS_API')
    def _fetch_newsapi(self) -> List[NewsItem]:
        """Fetch news from NewsAPI"""
        news_items = []
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': 'NIFTY OR BANKNIFTY OR "indian stock market" OR NSE',
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    news_item = self._parse_newsapi_article(article)
                    if news_item:
                        news_items.append(news_item)
                        
        except Exception as e:
            self.logger.error(f"NewsAPI error: {e}")
        
        return news_items
    
    def _fetch_rss_feed(self, feed_url: str, source: str) -> List[NewsItem]:
        """Fetch news from RSS feed"""
        news_items = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:20]:  # Last 20 items
                news_item = self._parse_rss_entry(entry, source)
                if news_item:
                    news_items.append(news_item)
                    
        except Exception as e:
            self.logger.error(f"RSS feed error for {source}: {e}")
        
        return news_items
    
    def _fetch_nse_announcements(self) -> List[NewsItem]:
        """Fetch NSE corporate announcements"""
        news_items = []
        
        try:
            # NSE announcements API endpoint
            url = 'https://www.nseindia.com/api/corporate-announcements'
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for announcement in data.get('data', [])[:10]:
                    news_item = NewsItem(
                        title=f"NSE: {announcement.get('subject', '')}",
                        summary=announcement.get('desc', ''),
                        source='NSE',
                        url=f"https://www.nseindia.com/companies-listing/corporate-filings-announcements",
                        published=datetime.strptime(
                            announcement.get('an_dt', ''),
                            '%d-%b-%Y %H:%M:%S'
                        ) if announcement.get('an_dt') else datetime.now(),
                        sentiment=self._analyze_sentiment(announcement.get('desc', '')),
                        relevance=0.9,  # NSE announcements are highly relevant
                        keywords=self._extract_keywords(announcement.get('desc', '')),
                        impact='HIGH' if 'result' in announcement.get('subject', '').lower() else 'MEDIUM'
                    )
                    news_items.append(news_item)
                    
        except Exception as e:
            self.logger.error(f"NSE announcements error: {e}")
        
        return news_items
    
    def _fetch_sebi_updates(self) -> List[NewsItem]:
        """Fetch SEBI regulatory updates"""
        news_items = []
        
        try:
            # SEBI updates RSS feed
            feed_url = 'https://www.sebi.gov.in/sebiweb/rss/feeds.xml'
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:10]:
                news_item = NewsItem(
                    title=f"SEBI: {entry.title}",
                    summary=entry.get('summary', ''),
                    source='SEBI',
                    url=entry.get('link', ''),
                    published=datetime.fromtimestamp(
                        entry.published_parsed.tm_sec
                    ) if hasattr(entry, 'published_parsed') else datetime.now(),
                    sentiment=SentimentScore.NEUTRAL,  # Regulatory updates are neutral
                    relevance=0.8,
                    keywords=['sebi', 'regulation', 'compliance'],
                    impact='HIGH' if 'circular' in entry.title.lower() else 'MEDIUM'
                )
                news_items.append(news_item)
                
        except Exception as e:
            self.logger.error(f"SEBI updates error: {e}")
        
        return news_items
    
    def fetch_sgx_nifty(self) -> Dict[str, Any]:
        """Fetch SGX Nifty data for pre-market indication"""
        try:
            # In production, use a real data provider
            # This is a mock implementation
            sgx_data = {
                'value': 25100,  # Current value
                'change': 50,    # Change from previous close
                'change_percent': 0.20,
                'timestamp': datetime.now(),
                'indication': 'BULLISH' if 50 > 0 else 'BEARISH'
            }
            
            return sgx_data
            
        except Exception as e:
            self.logger.error(f"SGX Nifty fetch error: {e}")
            return {}
    
    def fetch_economic_calendar(self) -> List[EconomicEvent]:
        """Fetch economic calendar events"""
        events = []
        
        try:
            # Fetch from multiple sources
            # 1. Indian economic events
            events.extend(self._fetch_indian_economic_events())
            
            # 2. Global economic events
            events.extend(self._fetch_global_economic_events())
            
            # Sort by time
            events.sort(key=lambda x: x.scheduled_time)
            
            # Cache results
            self.economic_calendar = events
            
        except Exception as e:
            self.logger.error(f"Economic calendar error: {e}")
        
        return events
    
    def _fetch_indian_economic_events(self) -> List[EconomicEvent]:
        """Fetch Indian economic events"""
        events = []
        
        # Mock data - in production, use real API
        upcoming_events = [
            {
                'name': 'RBI Monetary Policy',
                'importance': 'HIGH',
                'scheduled': datetime.now() + timedelta(days=2),
                'forecast': 6.5,  # Interest rate
                'previous': 6.5
            },
            {
                'name': 'India GDP Growth Rate',
                'importance': 'HIGH',
                'scheduled': datetime.now() + timedelta(days=5),
                'forecast': 7.2,
                'previous': 7.6
            },
            {
                'name': 'India Inflation Rate',
                'importance': 'MEDIUM',
                'scheduled': datetime.now() + timedelta(days=3),
                'forecast': 5.5,
                'previous': 5.4
            }
        ]
        
        for event_data in upcoming_events:
            event = EconomicEvent(
                event_name=event_data['name'],
                country='India',
                importance=event_data['importance'],
                scheduled_time=event_data['scheduled'],
                actual_value=None,
                forecast_value=event_data['forecast'],
                previous_value=event_data['previous'],
                impact_on_market=self._assess_event_impact(event_data)
            )
            events.append(event)
        
        return events
    
    def _fetch_global_economic_events(self) -> List[EconomicEvent]:
        """Fetch global economic events"""
        events = []
        
        # Mock data - in production, use real API
        global_events = [
            {
                'name': 'US Federal Reserve Meeting',
                'country': 'USA',
                'importance': 'HIGH',
                'scheduled': datetime.now() + timedelta(days=1),
                'impact': 'Global market volatility expected'
            },
            {
                'name': 'US Non-Farm Payrolls',
                'country': 'USA',
                'importance': 'HIGH',
                'scheduled': datetime.now() + timedelta(days=4),
                'forecast': 200000,
                'previous': 187000
            }
        ]
        
        for event_data in global_events:
            event = EconomicEvent(
                event_name=event_data['name'],
                country=event_data['country'],
                importance=event_data['importance'],
                scheduled_time=event_data['scheduled'],
                actual_value=None,
                forecast_value=event_data.get('forecast'),
                previous_value=event_data.get('previous'),
                impact_on_market=event_data.get('impact', 'Market volatility expected')
            )
            events.append(event)
        
        return events
    
    def _analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of text"""
        if not text:
            return SentimentScore.NEUTRAL
        
        text_lower = text.lower()
        
        # Count sentiment keywords
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        # Calculate sentiment score
        sentiment_diff = bullish_count - bearish_count
        
        if sentiment_diff >= 3:
            return SentimentScore.VERY_BULLISH
        elif sentiment_diff >= 1:
            return SentimentScore.BULLISH
        elif sentiment_diff <= -3:
            return SentimentScore.VERY_BEARISH
        elif sentiment_diff <= -1:
            return SentimentScore.BEARISH
        else:
            return SentimentScore.NEUTRAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []
        
        if not text:
            return keywords
        
        text_lower = text.lower()
        
        # Check for market-specific keywords
        for market, market_words in self.market_keywords.items():
            for word in market_words:
                if word in text_lower:
                    keywords.append(word)
        
        # Extract numbers (potential price levels)
        numbers = re.findall(r'\b\d{4,5}\b', text)
        keywords.extend([f"level_{num}" for num in numbers[:3]])
        
        return list(set(keywords))[:10]  # Return unique keywords
    
    def _parse_newsapi_article(self, article: Dict) -> Optional[NewsItem]:
        """Parse NewsAPI article"""
        try:
            return NewsItem(
                title=article.get('title', ''),
                summary=article.get('description', ''),
                source=article.get('source', {}).get('name', 'Unknown'),
                url=article.get('url', ''),
                published=datetime.strptime(
                    article.get('publishedAt', ''),
                    '%Y-%m-%dT%H:%M:%SZ'
                ) if article.get('publishedAt') else datetime.now(),
                sentiment=self._analyze_sentiment(
                    article.get('title', '') + ' ' + article.get('description', '')
                ),
                relevance=self._calculate_relevance(article),
                keywords=self._extract_keywords(article.get('description', '')),
                impact=self._assess_impact(article)
            )
        except Exception as e:
            self.logger.error(f"Error parsing article: {e}")
            return None
    
    def _parse_rss_entry(self, entry: Any, source: str) -> Optional[NewsItem]:
        """Parse RSS feed entry"""
        try:
            return NewsItem(
                title=entry.get('title', ''),
                summary=entry.get('summary', ''),
                source=source,
                url=entry.get('link', ''),
                published=datetime.fromtimestamp(
                    entry.published_parsed.tm_sec
                ) if hasattr(entry, 'published_parsed') else datetime.now(),
                sentiment=self._analyze_sentiment(
                    entry.get('title', '') + ' ' + entry.get('summary', '')
                ),
                relevance=0.7,  # Default relevance for RSS
                keywords=self._extract_keywords(entry.get('summary', '')),
                impact='MEDIUM'
            )
        except Exception as e:
            self.logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    def _calculate_relevance(self, article: Dict) -> float:
        """Calculate relevance score for article"""
        relevance = 0.5  # Base relevance
        
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        # Check for specific market mentions
        if 'nifty' in text:
            relevance += 0.3
        if 'banknifty' in text or 'bank nifty' in text:
            relevance += 0.3
        if any(word in text for word in ['option', 'futures', 'derivative']):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def _assess_impact(self, article: Dict) -> str:
        """Assess market impact of news"""
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        
        high_impact_words = ['crash', 'surge', 'plunge', 'rally', 'record', 'crisis']
        medium_impact_words = ['rise', 'fall', 'gain', 'loss', 'increase', 'decrease']
        
        if any(word in text for word in high_impact_words):
            return 'HIGH'
        elif any(word in text for word in medium_impact_words):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _assess_event_impact(self, event: Dict) -> str:
        """Assess impact of economic event"""
        if event['importance'] == 'HIGH':
            if event.get('forecast') and event.get('previous'):
                deviation = abs(event['forecast'] - event['previous']) / event['previous']
                if deviation > 0.05:  # 5% deviation
                    return 'Significant market movement expected'
                else:
                    return 'Moderate market impact expected'
            return 'High market volatility expected'
        else:
            return 'Limited market impact expected'
    
    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items"""
        seen = set()
        unique = []
        
        for item in news_items:
            # Create a hash of title and source
            hash_key = hash(item.title[:50] + item.source)
            
            if hash_key not in seen:
                seen.add(hash_key)
                unique.append(item)
        
        return unique
    
    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        if not self.news_cache or (
            self.last_update and 
            datetime.now() - self.last_update > timedelta(minutes=30)
        ):
            self.fetch_all_news()
        
        # Calculate sentiment distribution
        sentiment_counts = {
            'very_bullish': 0,
            'bullish': 0,
            'neutral': 0,
            'bearish': 0,
            'very_bearish': 0
        }
        
        total_relevance = 0
        weighted_sentiment = 0
        
        for news in self.news_cache[:50]:  # Last 50 news items
            if news.sentiment == SentimentScore.VERY_BULLISH:
                sentiment_counts['very_bullish'] += 1
                weighted_sentiment += 2 * news.relevance
            elif news.sentiment == SentimentScore.BULLISH:
                sentiment_counts['bullish'] += 1
                weighted_sentiment += 1 * news.relevance
            elif news.sentiment == SentimentScore.BEARISH:
                sentiment_counts['bearish'] += 1
                weighted_sentiment += -1 * news.relevance
            elif news.sentiment == SentimentScore.VERY_BEARISH:
                sentiment_counts['very_bearish'] += 1
                weighted_sentiment += -2 * news.relevance
            else:
                sentiment_counts['neutral'] += 1
            
            total_relevance += news.relevance
        
        # Calculate overall sentiment
        overall_sentiment = weighted_sentiment / total_relevance if total_relevance > 0 else 0
        
        # Get SGX Nifty indication
        sgx_data = self.fetch_sgx_nifty()
        
        # Get upcoming events
        upcoming_events = [
            event for event in self.economic_calendar
            if event.scheduled_time > datetime.now() and
            event.scheduled_time < datetime.now() + timedelta(days=1)
        ]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': self._get_sentiment_label(overall_sentiment),
            'sentiment_distribution': sentiment_counts,
            'total_news_analyzed': len(self.news_cache),
            'sgx_nifty': sgx_data,
            'upcoming_events': upcoming_events,
            'high_impact_news': [
                news for news in self.news_cache[:10]
                if news.impact == 'HIGH'
            ],
            'last_update': self.last_update
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 1.5:
            return 'VERY BULLISH'
        elif score >= 0.5:
            return 'BULLISH'
        elif score <= -1.5:
            return 'VERY BEARISH'
        elif score <= -0.5:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_trading_signals(self, symbol: str = 'NIFTY') -> Dict[str, Any]:
        """Generate trading signals based on news sentiment"""
        summary = self.get_market_sentiment_summary()
        
        # Base signal on overall sentiment
        signal = 'NEUTRAL'
        confidence = 50
        
        if summary['overall_sentiment'] >= 1.0:
            signal = 'BUY'
            confidence = min(90, 60 + summary['overall_sentiment'] * 10)
        elif summary['overall_sentiment'] <= -1.0:
            signal = 'SELL'
            confidence = min(90, 60 + abs(summary['overall_sentiment']) * 10)
        
        # Adjust for SGX Nifty
        if summary['sgx_nifty']:
            sgx_indication = summary['sgx_nifty'].get('indication')
            if sgx_indication == 'BULLISH' and signal != 'SELL':
                confidence += 10
            elif sgx_indication == 'BEARISH' and signal != 'BUY':
                confidence += 10
        
        # Check for high-impact events
        high_impact_events = [
            e for e in summary['upcoming_events']
            if e.importance == 'HIGH'
        ]
        
        if high_impact_events:
            # Reduce confidence before major events
            confidence *= 0.7
            signal = 'WAIT'  # Wait for event outcome
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': int(confidence),
            'sentiment_score': summary['overall_sentiment'],
            'reasoning': self._generate_signal_reasoning(summary, signal),
            'timestamp': datetime.now()
        }
    
    def _generate_signal_reasoning(self, summary: Dict, signal: str) -> List[str]:
        """Generate reasoning for trading signal"""
        reasons = []
        
        # Sentiment reason
        sentiment_label = summary['sentiment_label']
        reasons.append(f"Market sentiment is {sentiment_label}")
        
        # SGX Nifty reason
        if summary['sgx_nifty']:
            sgx_change = summary['sgx_nifty'].get('change_percent', 0)
            if abs(sgx_change) > 0.5:
                reasons.append(f"SGX Nifty indicates {sgx_change:+.2f}% move")
        
        # News impact
        high_impact = summary.get('high_impact_news', [])
        if high_impact:
            reasons.append(f"{len(high_impact)} high-impact news events detected")
        
        # Economic events
        events = summary.get('upcoming_events', [])
        if events:
            reasons.append(f"{len(events)} major economic events in next 24 hours")
        
        return reasons


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = EnhancedNewsSentimentAnalyzer()
    
    # Fetch latest news
    print("Fetching latest news...")
    news = analyzer.fetch_all_news()
    print(f"Fetched {len(news)} news items")
    
    # Get sentiment summary
    print("\nMarket Sentiment Summary:")
    summary = analyzer.get_market_sentiment_summary()
    print(f"Overall Sentiment: {summary['sentiment_label']} ({summary['overall_sentiment']:.2f})")
    print(f"Sentiment Distribution: {summary['sentiment_distribution']}")
    
    # Get trading signals
    print("\nTrading Signals:")
    signals = analyzer.get_trading_signals('NIFTY')
    print(f"Signal: {signals['signal']} (Confidence: {signals['confidence']}%)")
    print("Reasoning:")
    for reason in signals['reasoning']:
        print(f"  - {reason}")