"""
TradeMind_AI: Unified Trading Engine
Consolidates all trading logic from ai_trader.py and smart_trader.py
Includes paper/live trading toggle and proper lot size management
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import configurations
from config.constants import (
    LOT_SIZES, STRIKE_GAPS, SECURITY_IDS, EXCHANGE_SEGMENTS,
    RISK_MANAGEMENT, API_RATE_LIMITS, PAPER_TRADING, LIVE_TRADING,
    get_lot_size, validate_order_params, get_risk_per_trade
)

# Import required modules
try:
    from dhanhq import DhanContext, dhanhq
    DHANHQ_AVAILABLE = True
except ImportError:
    DHANHQ_AVAILABLE = False
    print("⚠️ dhanhq not available - running in simulation mode")

from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "PAPER"
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"

@dataclass
class OrderDetails:
    """Order details structure"""
    symbol: str
    strike: float
    option_type: str  # CE or PE
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET or LIMIT
    price: float
    product_type: str = "INTRADAY"
    validity: str = "DAY"
    
@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class UnifiedTradingEngine:
    """
    Unified trading engine that handles all trading operations
    Consolidates functionality from ai_trader.py and smart_trader.py
    """
    
    def __init__(self, mode: TradingMode = TradingMode.PAPER):
        """Initialize unified trading engine"""
        self.logger = logging.getLogger('UnifiedTradingEngine')
        self.logger.info(f"🚀 Initializing Unified Trading Engine in {mode.value} mode...")
        
        # Trading mode
        self.mode = mode
        
        # Credentials
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Dhan client
        self.dhan_client = None
        if mode != TradingMode.SIMULATION and DHANHQ_AVAILABLE:
            self._initialize_dhan_client()
        
        # Trading statistics
        self.trades_today = 0
        self.daily_pnl = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Rate limiting
        self.last_api_call = {}
        self.api_call_count = {}
        
        # Paper trading capital
        self.paper_capital = PAPER_TRADING['INITIAL_CAPITAL']
        self.paper_positions = []
        
        # Trade logs
        self.trade_log_file = self._get_trade_log_file()
        
        self.logger.info(f"✅ Unified Trading Engine initialized in {mode.value} mode")
        
    def _initialize_dhan_client(self):
        """Initialize Dhan API client"""
        try:
            if not self.client_id or not self.access_token:
                raise ValueError("Dhan credentials not found in environment")
            
            dhan_context = DhanContext(self.client_id, self.access_token)
            self.dhan_client = dhanhq(dhan_context)
            self.logger.info("✅ Dhan API client initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Dhan client: {e}")
            self.dhan_client = None
    
    def _get_trade_log_file(self) -> str:
        """Get appropriate trade log file based on mode"""
        if self.mode == TradingMode.PAPER:
            log_dir = PAPER_TRADING['LOG_DIRECTORY']
        elif self.mode == TradingMode.LIVE:
            log_dir = LIVE_TRADING['AUDIT_LOG_DIRECTORY']
        else:
            log_dir = 'logs/simulation/'
        
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"trades_{datetime.now().strftime('%Y%m%d')}.json")
    
    def switch_mode(self, new_mode: TradingMode) -> bool:
        """Switch trading mode with proper validation"""
        self.logger.info(f"🔄 Switching from {self.mode.value} to {new_mode.value} mode")
        
        # Validate switch to live mode
        if new_mode == TradingMode.LIVE:
            if not self._validate_live_trading_requirements():
                self.logger.error("❌ Live trading requirements not met")
                return False
        
        # Close all paper positions if switching from paper
        if self.mode == TradingMode.PAPER and self.paper_positions:
            self.logger.info("📝 Closing all paper positions...")
            self._close_all_paper_positions()
        
        # Update mode
        self.mode = new_mode
        self.trade_log_file = self._get_trade_log_file()
        
        # Send notification
        self._send_notification(
            f"🔄 Trading mode switched to {new_mode.value}",
            priority=True
        )
        
        return True
    
    def _validate_live_trading_requirements(self) -> bool:
        """Validate requirements for live trading"""
        checks = []
        
        # Check Dhan client
        if not self.dhan_client:
            checks.append("Dhan client not initialized")
        
        # Check 2FA if required
        if LIVE_TRADING['REQUIRE_2FA']:
            # Implement 2FA check here
            pass
        
        # Check risk limits
        if self.daily_pnl < -RISK_MANAGEMENT['MAX_DAILY_LOSS'] * self.paper_capital:
            checks.append("Daily loss limit exceeded")
        
        if checks:
            self.logger.error(f"Live trading validation failed: {checks}")
            return False
        
        return True
    
    def _check_rate_limit(self, api_name: str = 'DHAN_API') -> bool:
        """Check API rate limits"""
        limits = API_RATE_LIMITS.get(api_name, {})
        current_time = time.time()
        
        # Initialize counters
        if api_name not in self.api_call_count:
            self.api_call_count[api_name] = []
        
        # Remove old entries
        self.api_call_count[api_name] = [
            t for t in self.api_call_count[api_name] 
            if current_time - t < 3600  # Keep last hour
        ]
        
        # Check per second limit
        recent_calls = [
            t for t in self.api_call_count[api_name]
            if current_time - t < 1
        ]
        if len(recent_calls) >= limits.get('calls_per_second', float('inf')):
            return False
        
        # Check per minute limit
        recent_calls = [
            t for t in self.api_call_count[api_name]
            if current_time - t < 60
        ]
        if len(recent_calls) >= limits.get('calls_per_minute', float('inf')):
            return False
        
        # Add current call
        self.api_call_count[api_name].append(current_time)
        return True
    
    def place_order(self, order: OrderDetails) -> TradeResult:
        """
        Place order with appropriate mode handling
        Centralizes all order placement logic
        """
        # Validate order parameters
        is_valid, errors = validate_order_params(
            order.symbol, 
            order.quantity, 
            order.price
        )
        
        if not is_valid:
            return TradeResult(
                success=False,
                order_id="",
                message=f"Validation failed: {', '.join(errors)}",
                details={'errors': errors},
                timestamp=datetime.now()
            )
        
        # Check daily trade limit
        if self.trades_today >= RISK_MANAGEMENT['MAX_TRADES_PER_DAY']:
            return TradeResult(
                success=False,
                order_id="",
                message="Daily trade limit reached",
                details={'trades_today': self.trades_today},
                timestamp=datetime.now()
            )
        
        # Route to appropriate handler
        if self.mode == TradingMode.LIVE:
            result = self._place_live_order(order)
        elif self.mode == TradingMode.PAPER:
            result = self._place_paper_order(order)
        else:
            result = self._place_simulation_order(order)
        
        # Log trade
        self._log_trade(order, result)
        
        # Update statistics
        if result.success:
            self.trades_today += 1
        
        return result
    
    def _place_live_order(self, order: OrderDetails) -> TradeResult:
        """Place live order through Dhan API"""
        if not self.dhan_client:
            return TradeResult(
                success=False,
                order_id="",
                message="Dhan client not available",
                details={},
                timestamp=datetime.now()
            )
        
        # Check rate limit
        if not self._check_rate_limit():
            return TradeResult(
                success=False,
                order_id="",
                message="API rate limit exceeded",
                details={'retry_after': API_RATE_LIMITS['DHAN_API']['retry_after']},
                timestamp=datetime.now()
            )
        
        # Confirm order if required
        if LIVE_TRADING['CONFIRM_ORDERS']:
            if not self._confirm_order(order):
                return TradeResult(
                    success=False,
                    order_id="",
                    message="Order cancelled by user",
                    details={},
                    timestamp=datetime.now()
                )
        
        try:
            # Get security ID for the option
            security_id = self._get_option_security_id(
                order.symbol, 
                order.strike, 
                order.option_type
            )
            
            # Place order via Dhan API
            response = self.dhan_client.place_order(
                security_id=security_id,
                exchange_segment=EXCHANGE_SEGMENTS['FNO'],
                transaction_type=order.action,
                quantity=order.quantity * get_lot_size(order.symbol),
                order_type=order.order_type,
                price=order.price if order.order_type == 'LIMIT' else 0,
                product_type=order.product_type,
                validity=order.validity
            )
            
            if response.get('status') == 'success':
                order_id = response.get('data', {}).get('order_id', '')
                self._send_notification(
                    f"✅ LIVE ORDER PLACED\n"
                    f"Symbol: {order.symbol} {order.strike} {order.option_type}\n"
                    f"Action: {order.action}\n"
                    f"Quantity: {order.quantity} lots\n"
                    f"Order ID: {order_id}",
                    priority=True
                )
                
                return TradeResult(
                    success=True,
                    order_id=order_id,
                    message="Live order placed successfully",
                    details=response,
                    timestamp=datetime.now()
                )
            else:
                return TradeResult(
                    success=False,
                    order_id="",
                    message=response.get('message', 'Order failed'),
                    details=response,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            self.logger.error(f"Live order placement error: {e}")
            return TradeResult(
                success=False,
                order_id="",
                message=str(e),
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def _place_paper_order(self, order: OrderDetails) -> TradeResult:
        """Place paper trading order"""
        # Calculate order value
        lot_size = get_lot_size(order.symbol)
        order_value = order.quantity * lot_size * order.price
        
        # Apply slippage if enabled
        if PAPER_TRADING['ENABLE_SLIPPAGE']:
            slippage = order.price * PAPER_TRADING['SLIPPAGE_PERCENT']
            if order.action == 'BUY':
                order.price += slippage
            else:
                order.price -= slippage
        
        # Check capital
        if order.action == 'BUY' and order_value > self.paper_capital:
            return TradeResult(
                success=False,
                order_id="",
                message="Insufficient paper capital",
                details={
                    'required': order_value,
                    'available': self.paper_capital
                },
                timestamp=datetime.now()
            )
        
        # Generate paper order ID
        order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.paper_positions)}"
        
        # Update paper positions
        position = {
            'order_id': order_id,
            'symbol': order.symbol,
            'strike': order.strike,
            'option_type': order.option_type,
            'action': order.action,
            'quantity': order.quantity,
            'lot_size': lot_size,
            'entry_price': order.price,
            'current_price': order.price,
            'pnl': 0,
            'status': 'OPEN',
            'timestamp': datetime.now().isoformat()
        }
        
        self.paper_positions.append(position)
        
        # Update capital
        if order.action == 'BUY':
            self.paper_capital -= order_value
            self.paper_capital -= PAPER_TRADING['BROKERAGE_PER_LOT'] * order.quantity
        
        self._send_notification(
            f"📝 PAPER ORDER EXECUTED\n"
            f"Symbol: {order.symbol} {order.strike} {order.option_type}\n"
            f"Action: {order.action}\n"
            f"Quantity: {order.quantity} lots\n"
            f"Price: ₹{order.price}\n"
            f"Capital: ₹{self.paper_capital:,.0f}"
        )
        
        return TradeResult(
            success=True,
            order_id=order_id,
            message="Paper order executed",
            details=position,
            timestamp=datetime.now()
        )
    
    def _place_simulation_order(self, order: OrderDetails) -> TradeResult:
        """Place simulation order for testing"""
        order_id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return TradeResult(
            success=True,
            order_id=order_id,
            message="Simulation order placed",
            details={
                'mode': 'simulation',
                'order': order.__dict__
            },
            timestamp=datetime.now()
        )
    
    def get_positions(self) -> List[Dict]:
        """Get current positions based on mode"""
        if self.mode == TradingMode.LIVE and self.dhan_client:
            try:
                response = self.dhan_client.get_positions()
                return response.get('data', []) if response else []
            except Exception as e:
                self.logger.error(f"Error fetching live positions: {e}")
                return []
                
        elif self.mode == TradingMode.PAPER:
            return self.paper_positions
            
        else:
            return []
    
    def get_pnl(self) -> Dict[str, float]:
        """Get P&L based on mode"""
        if self.mode == TradingMode.PAPER:
            total_pnl = sum(pos.get('pnl', 0) for pos in self.paper_positions)
            return {
                'realized_pnl': sum(
                    pos.get('pnl', 0) for pos in self.paper_positions 
                    if pos.get('status') == 'CLOSED'
                ),
                'unrealized_pnl': sum(
                    pos.get('pnl', 0) for pos in self.paper_positions 
                    if pos.get('status') == 'OPEN'
                ),
                'total_pnl': total_pnl,
                'capital': self.paper_capital,
                'returns_percent': (total_pnl / PAPER_TRADING['INITIAL_CAPITAL']) * 100
            }
        
        elif self.mode == TradingMode.LIVE and self.dhan_client:
            # Implement live P&L calculation
            return {
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'total_pnl': self.daily_pnl,
                'capital': 0,
                'returns_percent': 0
            }
        
        return {
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'total_pnl': 0,
            'capital': 0,
            'returns_percent': 0
        }
    
    def update_paper_positions(self, market_prices: Dict[str, float]):
        """Update paper trading positions with current prices"""
        for position in self.paper_positions:
            if position['status'] == 'OPEN':
                key = f"{position['symbol']}_{position['strike']}_{position['option_type']}"
                if key in market_prices:
                    position['current_price'] = market_prices[key]
                    
                    # Calculate P&L
                    if position['action'] == 'BUY':
                        position['pnl'] = (
                            (position['current_price'] - position['entry_price']) * 
                            position['quantity'] * position['lot_size']
                        )
                    else:  # SELL
                        position['pnl'] = (
                            (position['entry_price'] - position['current_price']) * 
                            position['quantity'] * position['lot_size']
                        )
    
    def _close_all_paper_positions(self):
        """Close all open paper positions"""
        for position in self.paper_positions:
            if position['status'] == 'OPEN':
                position['status'] = 'CLOSED'
                position['exit_time'] = datetime.now().isoformat()
                
                # Update capital
                if position['action'] == 'BUY':
                    self.paper_capital += (
                        position['current_price'] * 
                        position['quantity'] * 
                        position['lot_size']
                    )
                
                self.paper_capital += position['pnl']
    
    def _confirm_order(self, order: OrderDetails) -> bool:
        """Confirm order with user (for live trading)"""
        # In production, this would show a UI dialog
        # For now, we'll auto-confirm with a log
        self.logger.warning(
            f"⚠️ ORDER CONFIRMATION REQUIRED:\n"
            f"Symbol: {order.symbol} {order.strike} {order.option_type}\n"
            f"Action: {order.action}\n"
            f"Quantity: {order.quantity} lots\n"
            f"Price: ₹{order.price}"
        )
        
        # Auto-confirm for now
        return True
    
    def _get_option_security_id(self, symbol: str, strike: float, option_type: str) -> str:
        """Get security ID for option contract"""
        # This would need to be implemented based on Dhan's option chain API
        # For now, returning a placeholder
        return f"{symbol}_{strike}_{option_type}"
    
    def _log_trade(self, order: OrderDetails, result: TradeResult):
        """Log trade details to file"""
        trade_log = {
            'timestamp': result.timestamp.isoformat(),
            'mode': self.mode.value,
            'order': order.__dict__,
            'result': {
                'success': result.success,
                'order_id': result.order_id,
                'message': result.message
            }
        }
        
        # Load existing logs
        try:
            with open(self.trade_log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
        
        # Append new log
        logs.append(trade_log)
        
        # Save logs
        with open(self.trade_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def _send_notification(self, message: str, priority: bool = False):
        """Send notification via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Telegram notification failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Notification error: {e}")
    
    def execute_strategy_trade(self, signal: Dict[str, Any]) -> TradeResult:
        """Execute trade based on strategy signal"""
        # Convert signal to OrderDetails
        order = OrderDetails(
            symbol=signal['symbol'],
            strike=signal['strike'],
            option_type=signal['option_type'],
            action=signal['action'],
            quantity=signal.get('quantity', 1),
            order_type=signal.get('order_type', 'MARKET'),
            price=signal.get('price', 0)
        )
        
        # Place order
        return self.place_order(order)
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance based on mode"""
        if self.mode == TradingMode.PAPER:
            return {
                'available_balance': self.paper_capital,
                'used_margin': PAPER_TRADING['INITIAL_CAPITAL'] - self.paper_capital,
                'total_balance': PAPER_TRADING['INITIAL_CAPITAL'],
                'mode': 'PAPER'
            }
        
        elif self.mode == TradingMode.LIVE and self.dhan_client:
            try:
                response = self.dhan_client.get_fund_limits()
                if response and response.get('status') == 'success':
                    data = response.get('data', {})
                    return {
                        'available_balance': data.get('availableBalance', 0),
                        'used_margin': data.get('utilizedAmount', 0),
                        'total_balance': data.get('sodLimit', 0),
                        'mode': 'LIVE'
                    }
            except Exception as e:
                self.logger.error(f"Error fetching balance: {e}")
        
        return {
            'available_balance': 0,
            'used_margin': 0,
            'total_balance': 0,
            'mode': self.mode.value
        }


# Factory function for easy initialization
def create_trading_engine(mode: str = "PAPER") -> UnifiedTradingEngine:
    """Create trading engine with specified mode"""
    mode_map = {
        "PAPER": TradingMode.PAPER,
        "LIVE": TradingMode.LIVE,
        "SIMULATION": TradingMode.SIMULATION
    }
    
    trading_mode = mode_map.get(mode.upper(), TradingMode.PAPER)
    return UnifiedTradingEngine(trading_mode)


# Example usage and testing
if __name__ == "__main__":
    # Create paper trading engine
    engine = create_trading_engine("PAPER")
    
    # Check balance
    balance = engine.get_account_balance()
    print(f"Account Balance: {balance}")
    
    # Place a test order
    test_order = OrderDetails(
        symbol="NIFTY",
        strike=25000,
        option_type="CE",
        action="BUY",
        quantity=1,
        order_type="MARKET",
        price=150
    )
    
    result = engine.place_order(test_order)
    print(f"Order Result: {result}")
    
    # Get positions
    positions = engine.get_positions()
    print(f"Current Positions: {positions}")
    
    # Get P&L
    pnl = engine.get_pnl()
    print(f"P&L: {pnl}")