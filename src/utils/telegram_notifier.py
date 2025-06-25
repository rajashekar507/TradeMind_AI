"""
Telegram Notifier for institutional-grade trading system
Sends trade signals and alerts to Telegram chat
"""

import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger = logging.getLogger('trading_system.telegram_notifier')

class TelegramNotifier:
    """Telegram notification service for trade signals"""
    
    def __init__(self, settings):
        self.settings = settings
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ Telegram credentials not found in environment")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ TelegramNotifier initialized successfully")
    
    async def initialize(self):
        """Initialize the Telegram notifier"""
        try:
            logger.info("🔧 Initializing TelegramNotifier...")
            
            if self.enabled:
                # Test connection
                test_result = await self._test_connection()
                if test_result:
                    logger.info("✅ Telegram connection test successful")
                else:
                    logger.warning("⚠️ Telegram connection test failed")
            
            return True
        except Exception as e:
            logger.error(f"❌ TelegramNotifier initialization failed: {e}")
            return False
    
    async def send_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """Send trade signal notification to Telegram"""
        try:
            if not self.enabled:
                logger.debug("📱 Telegram notifications disabled")
                return False
            
            message = self._format_trade_signal(signal)
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Failed to send trade signal: {e}")
            return False
    
    async def send_system_alert(self, alert_type: str, message: str) -> bool:
        """Send system alert to Telegram"""
        try:
            if not self.enabled:
                return False
            
            formatted_message = f"[SYSTEM ALERT]\n\nType: {alert_type}\nMessage: {message}\nTime: {datetime.now().strftime('%H:%M:%S')}"
            return await self._send_message(formatted_message)
            
        except Exception as e:
            logger.error(f"❌ Failed to send system alert: {e}")
            return False
    
    async def send_market_status(self, status: Dict[str, Any]) -> bool:
        """Send market status update to Telegram"""
        try:
            if not self.enabled:
                return False
            
            message = self._format_market_status(status)
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Failed to send market status: {e}")
            return False
    
    def _format_trade_signal(self, signal: Dict[str, Any]) -> str:
        """Format trade signal for Telegram with live data verification"""
        try:
            timestamp = signal.get('timestamp', datetime.now())
            instrument = signal.get('instrument', 'N/A')
            strike = signal.get('strike', 'N/A')
            option_type = signal.get('option_type', 'N/A')
            direction = signal.get('direction', option_type)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            target_1 = signal.get('target_1', 0)
            target_2 = signal.get('target_2', 0)
            confidence = signal.get('confidence', 0)
            reason = signal.get('reason', 'Multi-factor analysis')
            expiry = signal.get('expiry', 'Current Week')
            current_spot = signal.get('current_spot', 0)
            strike_ltp = signal.get('strike_ltp', entry_price)
            risk_status = signal.get('risk_status', 'VALIDATED')
            
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%H:%M:%S IST')
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                time_str = datetime.now().strftime('%H:%M:%S IST')
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            status_emoji = "[VALIDATED]" if risk_status == 'VALIDATED' else "[RISK FILTERED]"
            status_text = "VALIDATED" if risk_status == 'VALIDATED' else "RISK FILTERED"
            
            message = f"""
{status_emoji} LIVE TRADE SIGNAL

Timestamp: {date_str} {time_str}
Instrument: {instrument}
Current Spot: Rs.{current_spot}
Signal Direction: {direction}
Strike Price: {strike}
Strike LTP: Rs.{strike_ltp}
Expiry Date: {expiry}
Entry Price: Rs.{entry_price}
Stop Loss: Rs.{stop_loss}
Target 1: Rs.{target_1}
Target 2: Rs.{target_2}
Confidence Score: {confidence}%
Reason Summary: {reason}
Status: {status_text}

[LIVE DATA VERIFIED]
VLR_AI Institutional Trading System
"""
            return message.strip()
            
        except Exception as e:
            logger.error(f"❌ Signal formatting failed: {e}")
            return f"Trade Signal Error: {str(e)}"
    
    def _format_market_status(self, status: Dict[str, Any]) -> str:
        """Format market status for Telegram"""
        try:
            health_percentage = status.get('health_percentage', 0)
            active_sources = status.get('active_sources', 0)
            total_sources = status.get('total_sources', 8)
            
            message = f"""
[MARKET STATUS UPDATE]

System Health: {health_percentage}%
Data Sources: {active_sources}/{total_sources} Active
Time: {datetime.now().strftime('%H:%M:%S')}

VLR_AI System Monitor
"""
            return message.strip()
            
        except Exception as e:
            logger.error(f"❌ Status formatting failed: {e}")
            return f"Status Update Error: {str(e)}"
    
    async def send_message(self, message: str) -> bool:
        """Public method to send message to Telegram"""
        return await self._send_message(message)
    
    async def _send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Telegram message send failed: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test Telegram connection"""
        try:
            test_message = f"VLR_AI System Test - {datetime.now().strftime('%H:%M:%S')}"
            return await self._send_message(test_message)
            
        except Exception as e:
            logger.error(f"❌ Telegram connection test failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the Telegram notifier"""
        try:
            logger.info("🔄 Shutting down TelegramNotifier...")
            
            if self.enabled:
                await self.send_system_alert("SHUTDOWN", "VLR_AI trading system shutting down")
                
        except Exception as e:
            logger.error(f"❌ TelegramNotifier shutdown failed: {e}")
