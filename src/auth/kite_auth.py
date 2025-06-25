"""
Kite Authentication Module for institutional-grade trading system
Automated login with TOTP and access token generation
"""

import logging
import os
import time
import pyotp
from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime

logger = logging.getLogger('trading_system.kite_auth')

class KiteAuthenticator:
    """Automated Kite authentication with TOTP"""
    
    def __init__(self, settings):
        self.settings = settings
        
        self.api_key = os.getenv('KITE_API_KEY')
        self.api_secret = os.getenv('KITE_API_SECRET')
        self.user_id = os.getenv('KITE_USER_ID')
        self.password = os.getenv('KITE_PASSWORD')
        self.totp_secret = os.getenv('KITE_TOTP_SECRET')
        
        self.access_token = None
        self.driver = None
        
        if not all([self.api_key, self.api_secret, self.user_id, self.password, self.totp_secret]):
            logger.error("❌ Missing Kite credentials in environment variables")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ KiteAuthenticator initialized with credentials")
    
    async def initialize(self):
        """Initialize the Kite authenticator"""
        try:
            logger.info("🔧 Initializing KiteAuthenticator...")
            
            if self.enabled:
                success = await self.authenticate()
                if success:
                    logger.info("✅ Kite authentication successful")
                else:
                    logger.warning("⚠️ Kite authentication failed")
            
            return True
        except Exception as e:
            logger.error(f"❌ KiteAuthenticator initialization failed: {e}")
            return False
    
    async def authenticate(self) -> bool:
        """Use existing access token from environment variables"""
        try:
            logger.info("🔐 Using existing Kite access token...")
            
            env_access_token = os.getenv('KITE_ACCESS_TOKEN')
            if env_access_token:
                self.access_token = env_access_token
                logger.info(f"✅ Using access token from environment: {env_access_token[:10]}...")
                
                try:
                    kite = KiteConnect(api_key=self.api_key, timeout=30)
                    kite.set_access_token(self.access_token)
                    
                    profile = kite.profile()
                    logger.info(f"✅ Token validated - User: {profile.get('user_name', 'Unknown')}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Token validation failed: {e}")
                    return await self._perform_automated_login()
            else:
                logger.info("🔐 No access token in environment, performing automated login...")
                return await self._perform_automated_login()
                
        except Exception as e:
            logger.error(f"❌ Kite authentication failed: {e}")
            return False
    
    async def _perform_automated_login(self) -> bool:
        """Perform automated Kite authentication with Selenium"""
        try:
            logger.info("🔐 Starting automated Kite authentication...")
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            login_url = f"https://kite.zerodha.com/connect/login?api_key={self.api_key}"
            self.driver.get(login_url)
            
            wait = WebDriverWait(self.driver, 10)
            
            user_id_field = wait.until(EC.presence_of_element_located((By.ID, "userid")))
            user_id_field.send_keys(self.user_id)
            
            password_field = self.driver.find_element(By.ID, "password")
            password_field.send_keys(self.password)
            
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_button.click()
            
            totp_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='number']")))
            
            totp = pyotp.TOTP(self.totp_secret)
            current_totp = totp.now()
            
            totp_field.send_keys(current_totp)
            
            continue_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            continue_button.click()
            
            time.sleep(3)
            current_url = self.driver.current_url
            
            if "request_token=" in current_url:
                request_token = current_url.split("request_token=")[1].split("&")[0]
                logger.info(f"✅ Request token obtained: {request_token[:10]}...")
                
                kite = KiteConnect(api_key=self.api_key)
                data = kite.generate_session(request_token, api_secret=self.api_secret)
                self.access_token = data["access_token"]
                
                logger.info("✅ Access token generated successfully")
                return True
            else:
                logger.error("❌ Failed to obtain request token")
                return False
                
        except Exception as e:
            logger.error(f"❌ Automated Kite authentication failed: {e}")
            return False
        finally:
            if self.driver:
                self.driver.quit()
    
    def get_access_token(self) -> str:
        """Get the current access token"""
        return self.access_token or ""
    
    def is_authenticated(self) -> bool:
        """Check if authentication is valid"""
        return self.access_token is not None
    
    def get_authenticated_kite(self):
        """Get authenticated Kite client"""
        if self.is_authenticated():
            try:
                kite = KiteConnect(api_key=self.api_key, timeout=30)
                kite.set_access_token(self.access_token)
                logger.info("✅ KiteConnect client created successfully")
                return kite
            except Exception as e:
                logger.error(f"❌ Failed to create KiteConnect client: {e}")
                return None
        return None
    
    async def shutdown(self):
        """Shutdown the Kite authenticator"""
        try:
            logger.info("🔄 Shutting down KiteAuthenticator...")
            
            if self.driver:
                self.driver.quit()
                
        except Exception as e:
            logger.error(f"❌ KiteAuthenticator shutdown failed: {e}")
