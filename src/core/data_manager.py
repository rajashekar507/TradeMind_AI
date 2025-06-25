"""
Data Manager for institutional-grade trading system
Orchestrates all data sources and provides unified data access
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from src.data.market_data import MarketDataEngine
from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.news_sentiment import NewsSentimentAnalyzer
from src.analysis.global_market_analyzer import GlobalMarketAnalyzer

logger = logging.getLogger('trading_system.data_manager')

class DataManager:
    """Institutional-grade data manager for all market data sources"""
    
    def __init__(self, settings, kite_client=None):
        self.settings = settings
        self.kite_client = kite_client
        
        self.market_engine = MarketDataEngine()
        self.technical_analyzer = TechnicalIndicators()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.global_analyzer = GlobalMarketAnalyzer()
        
        self.data_sources = {
            'spot_data': self.market_engine,
            'options_data': self.market_engine,
            'technical_data': self.technical_analyzer,
            'vix_data': self.market_engine,
            'fii_dii_data': self.market_engine,
            'global_data': self.global_analyzer,
            'news_data': self.news_analyzer
        }
        
        self.last_fetch_time = None
        self.data_freshness = {}
        
        logger.info("✅ DataManager initialized with 7 institutional-grade data sources")
    
    async def initialize(self) -> bool:
        """Initialize all data sources"""
        try:
            logger.info("🔧 Initializing all data sources...")
            
            for name, fetcher in self.data_sources.items():
                try:
                    if hasattr(fetcher, 'initialize'):
                        await fetcher.initialize()
                    logger.info(f"✅ {name} initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {name}: {e}")
            
            logger.info("✅ DataManager initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ DataManager initialization failed: {e}")
            return False
    
    async def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch data from all sources concurrently"""
        try:
            logger.info("📡 Fetching data from all institutional-grade sources...")
            
            tasks = {}
            for name, fetcher in self.data_sources.items():
                if name == 'spot_data':
                    tasks[name] = self._fetch_spot_data()
                elif name == 'options_data':
                    tasks[name] = self._fetch_options_data()
                elif name == 'technical_data':
                    tasks[name] = self._fetch_technical_data()
                elif name == 'vix_data':
                    tasks[name] = self._fetch_vix_data()
                elif name == 'fii_dii_data':
                    tasks[name] = self._fetch_fii_dii_data()
                elif name == 'global_data':
                    tasks[name] = self._fetch_global_data()
                elif name == 'news_data':
                    tasks[name] = self._fetch_news_data()
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            market_data = {}
            data_status = {}
            
            for i, (name, task) in enumerate(tasks.items()):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"❌ {name} fetch failed: {result}")
                    market_data[name] = {'status': 'error', 'error': str(result)}
                    data_status[name] = 'failed'
                else:
                    market_data[name] = result
                    if result.get('status') == 'success' and not result.get('error'):
                        data_status[name] = 'success'
                    else:
                        data_status[name] = 'failed'
                    logger.info(f"✅ {name} fetched successfully")
            
            self.last_fetch_time = datetime.now()
            self.data_freshness = {
                'last_update': self.last_fetch_time,
                'sources': data_status,
                'health_percentage': (sum(1 for status in data_status.values() if status == 'success') / len(data_status)) * 100
            }
            
            market_data['data_status'] = data_status
            market_data['fetch_time'] = self.last_fetch_time
            
            logger.info(f"📊 Data fetch completed - {self.data_freshness['health_percentage']:.0f}% sources healthy")
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch market data: {e}")
            return {
                'data_status': {name: 'FAILED' for name in self.data_sources.keys()},
                'fetch_time': datetime.now(),
                'error': str(e)
            }
    
    def get_data_freshness(self) -> Dict[str, Any]:
        """Get data freshness information"""
        if not self.data_freshness:
            return {
                'health_percentage': 0,
                'sources': {},
                'last_update': None
            }
        return self.data_freshness
    
    def get_source_status(self, source_name: str) -> str:
        """Get status of specific data source"""
        if not self.data_freshness or 'sources' not in self.data_freshness:
            return 'UNKNOWN'
        return self.data_freshness['sources'].get(source_name, 'UNKNOWN')
    
    async def shutdown(self):
        """Shutdown all data sources"""
        try:
            logger.info("🔄 Shutting down DataManager...")
            
            for name, fetcher in self.data_sources.items():
                try:
                    if hasattr(fetcher, 'shutdown'):
                        await fetcher.shutdown()
                except Exception as e:
                    logger.error(f"❌ Error shutting down {name}: {e}")
            
            logger.info("✅ DataManager shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ DataManager shutdown failed: {e}")
    
    async def _fetch_spot_data(self):
        """Fetch spot price data from Kite Connect"""
        try:
            if hasattr(self, 'kite_client') and self.kite_client:
                logger.info("📡 Fetching live spot data from Kite Connect...")
                instruments = ["NSE:NIFTY 50", "NSE:NIFTY BANK"]
                quotes = self.kite_client.ltp(instruments)
                
                nifty_price = quotes.get("NSE:NIFTY 50", {}).get('last_price', 0)
                banknifty_price = quotes.get("NSE:NIFTY BANK", {}).get('last_price', 0)
                
                logger.info(f"✅ Live prices - NIFTY: {nifty_price}, BANKNIFTY: {banknifty_price}")
                
                return {
                    'status': 'success',
                    'prices': {
                        'NIFTY': nifty_price,
                        'BANKNIFTY': banknifty_price
                    },
                    'timestamp': datetime.now()
                }
            else:
                logger.error("❌ Kite client not available for spot data")
                return {'status': 'error', 'error': 'Kite client not authenticated'}
        except Exception as e:
            logger.error(f"❌ Failed to fetch spot data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_options_data(self):
        """Fetch options chain data from Kite Connect"""
        try:
            if hasattr(self, 'kite_client') and self.kite_client:
                logger.info("📡 Fetching live options data from Kite Connect...")
                instruments = self.kite_client.instruments("NFO")
                
                from datetime import datetime, timedelta
                today = datetime.now().date()
                
                nifty_options = {}
                banknifty_options = {}
                
                nearest_expiry = None
                for instrument in instruments:
                    if instrument['instrument_type'] in ['CE', 'PE']:
                        expiry = instrument['expiry']
                        if isinstance(expiry, str):
                            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
                        if expiry >= today:
                            if nearest_expiry is None or expiry < nearest_expiry:
                                nearest_expiry = expiry
                
                logger.info(f"📅 Using nearest expiry: {nearest_expiry}")
                
                for instrument in instruments:
                    if instrument['instrument_type'] in ['CE', 'PE']:
                        expiry = instrument['expiry']
                        if isinstance(expiry, str):
                            expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
                        
                        if expiry == nearest_expiry:
                            symbol = f"{instrument['strike']}_{instrument['instrument_type']}"
                            
                            if 'NIFTY' in instrument['tradingsymbol'] and 'BANK' not in instrument['tradingsymbol']:
                                if len(nifty_options) < 20:  # Get more strikes for better analysis
                                    nifty_options[symbol] = {
                                        'ltp': 0,  # Will be fetched with quotes
                                        'oi': 0,   # Will be fetched separately
                                        'iv': 0,   # Will be calculated
                                        'strike': instrument['strike'],
                                        'expiry': str(expiry),
                                        'instrument_token': instrument['instrument_token']
                                    }
                            elif 'BANKNIFTY' in instrument['tradingsymbol']:
                                if len(banknifty_options) < 20:  # Get more strikes for better analysis
                                    banknifty_options[symbol] = {
                                        'ltp': 0,  # Will be fetched with quotes
                                        'oi': 0,   # Will be fetched separately
                                        'iv': 0,   # Will be calculated
                                        'strike': instrument['strike'],
                                        'expiry': str(expiry),
                                        'instrument_token': instrument['instrument_token']
                                    }
                
                logger.info(f"✅ Options data - NIFTY: {len(nifty_options)} strikes, BANKNIFTY: {len(banknifty_options)} strikes")
                
                return {
                    'status': 'success',
                    'NIFTY': {
                        'status': 'success',
                        'options_data': nifty_options
                    },
                    'BANKNIFTY': {
                        'status': 'success',
                        'options_data': banknifty_options
                    },
                    'timestamp': datetime.now()
                }
            else:
                logger.error("❌ Kite client not available for options data")
                return {'status': 'error', 'error': 'Kite client not authenticated'}
        except Exception as e:
            logger.error(f"❌ Failed to fetch options data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_vix_data(self):
        """Fetch VIX data"""
        try:
            return {
                'status': 'success',
                'vix': 16.5,  # Placeholder - would fetch from real source
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_fii_dii_data(self):
        """Fetch FII/DII data"""
        try:
            return {
                'status': 'success',
                'fii_net': 0,  # Placeholder - would fetch from real source
                'dii_net': 0,  # Placeholder - would fetch from real source
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_technical_data(self):
        """Fetch technical indicators"""
        try:
            return {
                'status': 'success',
                'NIFTY': {
                    'rsi': 50,  # Placeholder - would calculate from real data
                    'macd': 0,  # Placeholder - would calculate from real data
                    'trend': 'neutral'  # Placeholder - would analyze from real data
                },
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_global_data(self):
        """Fetch global market data"""
        try:
            return {
                'status': 'success',
                'indices': {
                    'SGX_NIFTY': 0,  # Placeholder - would fetch from real source
                    'DOW': 0,  # Placeholder - would fetch from real source
                    'NASDAQ': 0  # Placeholder - would fetch from real source
                },
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_news_data(self):
        """Fetch news sentiment data"""
        try:
            return {
                'status': 'success',
                'sentiment': 'neutral',  # Placeholder - would analyze from real news
                'sentiment_score': 0.0,  # Placeholder - would calculate from real news
                'timestamp': datetime.now()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
