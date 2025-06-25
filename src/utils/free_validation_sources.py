"""
Free validation sources for cross-checking market data
Uses Yahoo Finance, MoneyControl, and Google Finance scraping
"""

import logging
import asyncio
import aiohttp
from typing import Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup
import re
from datetime import datetime

logger = logging.getLogger('trading_system.free_validation')

class FreeValidationSources:
    """Free data sources for validation (NO PAID APIs)"""
    
    def __init__(self):
        self.session = None
        self.yahoo_cache = {}
        self.cache_timeout = 60  # 1 minute cache
    
    async def initialize(self):
        """Initialize HTTP session for scraping"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(limit=10)
            )
            logger.info("✅ Free validation sources initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Free validation sources initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown HTTP session"""
        try:
            if self.session:
                await self.session.close()
            logger.info("🔄 Free validation sources shutdown")
        except Exception as e:
            logger.error(f"❌ Free validation sources shutdown failed: {e}")
    
    async def _get_yahoo_spot_price(self, instrument: str) -> Optional[float]:
        """Get spot price from Yahoo Finance API"""
        try:
            if not self.session:
                return None
            
            cache_key = f"yahoo_{instrument}"
            if cache_key in self.yahoo_cache:
                cached_data = self.yahoo_cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_timeout:
                    return cached_data['price']
            
            yahoo_symbols = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK'
            }
            
            symbol = yahoo_symbols.get(instrument)
            if not symbol:
                return None
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        if 'meta' in result and 'regularMarketPrice' in result['meta']:
                            price = float(result['meta']['regularMarketPrice'])
                            
                            self.yahoo_cache[cache_key] = {
                                'price': price,
                                'timestamp': datetime.now()
                            }
                            
                            logger.debug(f"📊 Yahoo Finance {instrument}: ₹{price}")
                            return price
            
            return None
            
        except Exception as e:
            logger.debug(f"❌ Yahoo Finance API failed for {instrument}: {e}")
            return None
    
    async def _scrape_moneycontrol_price(self, instrument: str) -> Optional[float]:
        """Scrape spot price from MoneyControl"""
        try:
            if not self.session:
                return None
            
            mc_urls = {
                'NIFTY': 'https://www.moneycontrol.com/indian-indices/nifty-50-9.html',
                'BANKNIFTY': 'https://www.moneycontrol.com/indian-indices/nifty-bank-23.html'
            }
            
            url = mc_urls.get(instrument)
            if not url:
                return None
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    price_selectors = [
                        '.inprice1',
                        '.price',
                        '[data-price]',
                        '.pcnstext'
                    ]
                    
                    for selector in price_selectors:
                        price_element = soup.select_one(selector)
                        if price_element:
                            price_text = price_element.get_text().strip()
                            price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                            if price_match:
                                price = float(price_match.group())
                                logger.debug(f"📊 MoneyControl {instrument}: ₹{price}")
                                return price
            
            return None
            
        except Exception as e:
            logger.debug(f"❌ MoneyControl scraping failed for {instrument}: {e}")
            return None
    
    async def _scrape_google_finance_price(self, instrument: str) -> Optional[float]:
        """Scrape spot price from Google Finance"""
        try:
            if not self.session:
                return None
            
            google_urls = {
                'NIFTY': 'https://www.google.com/finance/quote/NIFTY_50:INDEXNSE',
                'BANKNIFTY': 'https://www.google.com/finance/quote/NIFTY_BANK:INDEXNSE'
            }
            
            url = google_urls.get(instrument)
            if not url:
                return None
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    price_selectors = [
                        '[data-last-price]',
                        '.YMlKec',
                        '.P6K39c',
                        '[jsname="ip75Cb"]'
                    ]
                    
                    for selector in price_selectors:
                        price_element = soup.select_one(selector)
                        if price_element:
                            price_text = price_element.get_text().strip()
                            price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                            if price_match:
                                price = float(price_match.group())
                                logger.debug(f"📊 Google Finance {instrument}: ₹{price}")
                                return price
            
            return None
            
        except Exception as e:
            logger.debug(f"❌ Google Finance scraping failed for {instrument}: {e}")
            return None
    
    async def validate_spot_price_three_sources(self, instrument: str, kite_price: float) -> Tuple[bool, str]:
        """Validate spot price using all three free sources with confidence scoring"""
        try:
            from src.config.validation_settings import ValidationSettings
            
            if not ValidationSettings.FREE_VALIDATION_ENABLED:
                return True, "Free validation disabled"
            
            validation_results = []
            
            yahoo_price = await self._get_yahoo_spot_price(instrument)
            mc_price = await self._scrape_moneycontrol_price(instrument)
            google_price = await self._scrape_google_finance_price(instrument)
            
            sources = [
                ('Yahoo Finance', yahoo_price),
                ('MoneyControl', mc_price),
                ('Google Finance', google_price)
            ]
            
            valid_sources = 0
            total_diff = 0
            max_diff = 0
            
            for source_name, price in sources:
                if price and price > 0:
                    price_diff = abs(kite_price - price)
                    max_diff = max(max_diff, price_diff)
                    if price_diff <= ValidationSettings.FREE_PRICE_TOLERANCE:
                        validation_results.append(f"{source_name}: ✅ (₹{price}, diff: ₹{price_diff:.2f})")
                        valid_sources += 1
                    else:
                        validation_results.append(f"{source_name}: ⚠️ (₹{price}, diff: ₹{price_diff:.2f})")
                        total_diff += price_diff
                else:
                    validation_results.append(f"{source_name}: ❌ (unavailable)")
            
            if valid_sources >= 2:
                confidence = "HIGH"
                status = True
            elif valid_sources == 1:
                confidence = "MEDIUM"
                status = True
            else:
                confidence = "LOW"
                status = True  # Soft warning approach
            
            validation_summary = f"Free validation {confidence} confidence: {valid_sources}/3 sources agree. " + "; ".join(validation_results)
            
            if max_diff > (kite_price * ValidationSettings.NSE_HARD_STOP_THRESHOLD / 100):
                return False, f"CRITICAL: Free sources contradict Kite data by >2% (max diff: ₹{max_diff:.2f})"
            
            return status, validation_summary
            
        except Exception as e:
            logger.warning(f"⚠️ Three-source validation error: {e}")
            return True, f"Free validation skipped due to error: {str(e)}"
    
    async def validate_vix_data(self, kite_vix: float) -> Tuple[bool, str]:
        """Validate VIX data using free sources"""
        try:
            if 5.0 <= kite_vix <= 80.0:
                return True, f"VIX {kite_vix:.2f} within reasonable range"
            else:
                return False, f"VIX {kite_vix:.2f} outside reasonable range (5-80)"
                
        except Exception as e:
            logger.warning(f"⚠️ VIX validation error: {e}")
            return True, f"VIX validation skipped due to error: {str(e)}"
