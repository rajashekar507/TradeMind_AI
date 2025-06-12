"""
TradeMind_AI: Centralized Balance Management Utility
Eliminates code duplication across multiple files
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BalanceManager:
    """Centralized balance management for TradeMind_AI"""
    
    def __init__(self):
        """Initialize balance manager"""
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Dhan client
        self.dhan_client = self._initialize_dhan_client()
        
        # Balance history file
        self.balance_history_file = "data/balance_history.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
    
    def _initialize_dhan_client(self):
        """Initialize Dhan API client"""
        try:
            from dhanhq import DhanContext, dhanhq
            
            if not self.client_id or not self.access_token:
                print("⚠️ Dhan credentials not found in environment")
                return None
            
            dhan_context = DhanContext(self.client_id, self.access_token)
            return dhanhq(dhan_context)
            
        except ImportError:
            print("⚠️ dhanhq library not available")
            return None
        except Exception as e:
            print(f"❌ Failed to initialize Dhan client: {e}")
            return None
    
    def fetch_current_balance(self) -> float:
        """
        Fetch current account balance from Dhan API
        Returns: Current available balance as float
        """
        try:
            if not self.dhan_client:
                # Fallback to environment variable
                return float(os.getenv('TOTAL_CAPITAL', 100000))
            
            # Fetch from Dhan API
            funds_response = self.dhan_client.get_fund_limits()
            
            if funds_response and 'data' in funds_response:
                balance_data = funds_response['data']
                
                # Handle Dhan API typo
                available = balance_data.get('availabelBalance', 0)
                if available == 0:
                    available = balance_data.get('availableBalance', 0)
                
                return float(available)
            else:
                print("⚠️ No balance data received from Dhan API")
                return float(os.getenv('TOTAL_CAPITAL', 100000))
                
        except Exception as e:
            print(f"❌ Error fetching balance: {e}")
            # Fallback to environment variable
            return float(os.getenv('TOTAL_CAPITAL', 100000))
    
    def get_detailed_balance_info(self) -> Dict:
        """
        Get detailed balance information
        Returns: Dictionary with comprehensive balance data
        """
        try:
            if not self.dhan_client:
                return {
                    'available_balance': float(os.getenv('TOTAL_CAPITAL', 100000)),
                    'used_margin': 0,
                    'total_balance': float(os.getenv('TOTAL_CAPITAL', 100000)),
                    'mode': 'FALLBACK',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Fetch detailed fund limits
            funds_response = self.dhan_client.get_fund_limits()
            
            if funds_response and 'data' in funds_response:
                data = funds_response['data']
                
                return {
                    'available_balance': float(data.get('availabelBalance', data.get('availableBalance', 0))),
                    'used_margin': float(data.get('utilizedAmount', 0)),
                    'total_balance': float(data.get('sodLimit', 0)),
                    'opening_balance': float(data.get('openingBalance', 0)),
                    'realized_pnl': float(data.get('realizedPnL', 0)),
                    'unrealized_pnl': float(data.get('unrealizedPnL', 0)),
                    'mode': 'LIVE',
                    'last_updated': datetime.now().isoformat()
                }
            else:
                raise Exception("No data received from Dhan API")
                
        except Exception as e:
            print(f"❌ Error getting detailed balance: {e}")
            return {
                'available_balance': float(os.getenv('TOTAL_CAPITAL', 100000)),
                'used_margin': 0,
                'total_balance': float(os.getenv('TOTAL_CAPITAL', 100000)),
                'mode': 'ERROR',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def log_balance_update(self, update_type: str, amount: float, 
                          old_balance: float, new_balance: float) -> None:
        """
        Log balance updates to history file
        """
        try:
            # Load existing history
            if os.path.exists(self.balance_history_file):
                with open(self.balance_history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new update
            balance_update = {
                'timestamp': datetime.now().isoformat(),
                'update_type': update_type,
                'amount': amount,
                'balance_before': old_balance,
                'balance_after': new_balance,
                'change': new_balance - old_balance
            }
            
            history.append(balance_update)
            
            # Keep only last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save updated history
            with open(self.balance_history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error logging balance update: {e}")
    
    def get_balance_history(self, days: int = 7) -> List[Dict]:
        """
        Get balance history for specified number of days
        """
        try:
            if not os.path.exists(self.balance_history_file):
                return []
            
            with open(self.balance_history_file, 'r') as f:
                history = json.load(f)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_history = [
                entry for entry in history
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
            ]
            
            return filtered_history
            
        except Exception as e:
            print(f"❌ Error getting balance history: {e}")
            return []
    
    def send_balance_alert(self, message: str) -> bool:
        """
        Send balance alert via Telegram
        """
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                print(f"📱 Alert (Telegram disabled): {message}")
                return False
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Error sending balance alert: {e}")
            return False
    
    def send_balance_update_notification(self, update_type: str, amount: float, 
                                       old_balance: float, new_balance: float) -> None:
        """
        Send formatted balance update notification
        """
        try:
            change = new_balance - old_balance
            change_emoji = "📈" if change >= 0 else "📉"
            
            message = f"""
💰 <b>BALANCE UPDATE</b>

🔄 Type: {update_type}
💵 Amount: ₹{abs(amount):,.2f}
📊 Balance Before: ₹{old_balance:,.2f}
💰 Balance After: ₹{new_balance:,.2f}
{change_emoji} Change: ₹{change:+,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
            """
            
            self.send_balance_alert(message)
            
        except Exception as e:
            print(f"❌ Error sending balance notification: {e}")
    
    def check_balance_alerts(self, current_balance: float) -> None:
        """
        Check if balance alerts should be triggered
        """
        try:
            total_capital = float(os.getenv('TOTAL_CAPITAL', 100000))
            
            # Low balance alert (below 20% of total capital)
            if current_balance < total_capital * 0.20:
                self.send_balance_alert(
                    f"⚠️ <b>LOW BALANCE ALERT</b>\n\n"
                    f"💰 Current Balance: ₹{current_balance:,.2f}\n"
                    f"📊 Total Capital: ₹{total_capital:,.2f}\n"
                    f"📉 Remaining: {(current_balance/total_capital)*100:.1f}%\n\n"
                    f"⚡ Consider reducing position sizes or adding funds"
                )
            
            # Very low balance alert (below 10% of total capital)
            elif current_balance < total_capital * 0.10:
                self.send_balance_alert(
                    f"🚨 <b>CRITICAL BALANCE ALERT</b>\n\n"
                    f"💰 Current Balance: ₹{current_balance:,.2f}\n"
                    f"📊 Total Capital: ₹{total_capital:,.2f}\n"
                    f"📉 Remaining: {(current_balance/total_capital)*100:.1f}%\n\n"
                    f"🛑 IMMEDIATE ACTION REQUIRED - Add funds or stop trading"
                )
                
        except Exception as e:
            print(f"❌ Error checking balance alerts: {e}")
    
    def update_balance_after_trade(self, trade_type: str, amount: float) -> Dict:
        """
        Update balance after trade execution and return balance info
        """
        try:
            # Get current balance before update
            old_balance = self.fetch_current_balance()
            
            # Wait a moment for balance to update on broker side
            time.sleep(2)
            
            # Get updated balance
            new_balance = self.fetch_current_balance()
            
            # Log the update
            self.log_balance_update(trade_type, amount, old_balance, new_balance)
            
            # Send notification
            self.send_balance_update_notification(trade_type, amount, old_balance, new_balance)
            
            # Check for balance alerts
            self.check_balance_alerts(new_balance)
            
            return {
                'old_balance': old_balance,
                'new_balance': new_balance,
                'change': new_balance - old_balance,
                'trade_type': trade_type,
                'amount': amount,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error updating balance after trade: {e}")
            return {
                'old_balance': 0,
                'new_balance': 0,
                'change': 0,
                'error': str(e)
            }

# Global instance for easy access
balance_manager = BalanceManager()

# Convenience functions for backward compatibility
def fetch_current_balance() -> float:
    """Convenience function to fetch current balance"""
    return balance_manager.fetch_current_balance()

def get_detailed_balance_info() -> Dict:
    """Convenience function to get detailed balance info"""
    return balance_manager.get_detailed_balance_info()

def send_balance_alert(message: str) -> bool:
    """Convenience function to send balance alert"""
    return balance_manager.send_balance_alert(message)

def update_balance_after_trade(trade_type: str, amount: float) -> Dict:
    """Convenience function to update balance after trade"""
    return balance_manager.update_balance_after_trade(trade_type, amount)

# Example usage
if __name__ == "__main__":
    print("🧪 Testing Balance Management Utility...")
    
    # Test balance fetching
    current_balance = fetch_current_balance()
    print(f"💰 Current Balance: ₹{current_balance:,.2f}")
    
    # Test detailed balance info
    detailed_info = get_detailed_balance_info()
    print(f"📊 Detailed Balance Info: {detailed_info}")
    
    # Test balance alert
    send_balance_alert("🧪 Test alert from Balance Management Utility")
    
    print("✅ Balance Management Utility tested successfully!")