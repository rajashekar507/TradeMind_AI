"""
Live trade execution engine with Kite Connect integration
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger('trading_system.trade_executor')

class TradeExecutor:
    """Live trade execution with SEBI compliance"""
    
    def __init__(self, kite_client=None, settings=None):
        self.kite = kite_client
        self.settings = settings
        self.paper_trading = getattr(settings, 'PAPER_TRADING', True)
        
        self.lot_sizes = {
            'NIFTY': 25,
            'BANKNIFTY': 75
        }
        
        self.active_positions = {}
        self.order_history = []
        self.daily_pnl = 0
        self.max_daily_loss = -5000
        self.max_positions = 5
    
    async def execute_trade(self, signal: Dict) -> Dict:
        """Execute trade based on signal"""
        execution_result = {
            'timestamp': datetime.now(),
            'signal': signal,
            'status': 'failed',
            'order_id': None,
            'execution_price': 0,
            'quantity': 0,
            'message': ''
        }
        
        try:
            if not self.kite:
                logger.error("❌ STRICT ENFORCEMENT: No Kite client available - CANNOT EXECUTE TRADES")
                execution_result['message'] = 'No Kite client - strict enforcement mode'
                return execution_result
            
            if not self._validate_trade_conditions(signal):
                execution_result['message'] = 'Trade validation failed'
                return execution_result
            
            if self.paper_trading:
                execution_result = await self._execute_paper_trade(signal)
            else:
                execution_result = await self._execute_live_trade(signal)
            
            if execution_result['status'] == 'success':
                await self._update_position_tracking(signal, execution_result)
                logger.info(f"✅ Trade executed: {signal['instrument']} {signal['strike']} {signal['option_type']}")
            
        except Exception as e:
            logger.error(f"❌ STRICT ENFORCEMENT: Trade execution failed: {e}")
            execution_result['message'] = str(e)
        
        return execution_result
    
    async def _execute_live_trade(self, signal: Dict) -> Dict:
        """Execute live trade through Kite Connect"""
        try:
            tradingsymbol = self._construct_tradingsymbol(signal)
            quantity = self._calculate_quantity(signal)
            
            order_params = {
                'variety': 'regular',
                'exchange': 'NFO',
                'tradingsymbol': tradingsymbol,
                'transaction_type': 'BUY',
                'quantity': quantity,
                'product': 'MIS',
                'order_type': 'MARKET',
                'validity': 'DAY'
            }
            
            order_id = self.kite.place_order(**order_params)
            
            await asyncio.sleep(2)
            
            order_status = self.kite.order_history(order_id)
            
            if order_status and len(order_status) > 0:
                latest_status = order_status[-1]
                
                if latest_status['status'] == 'COMPLETE':
                    return {
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'status': 'success',
                        'order_id': order_id,
                        'execution_price': float(latest_status['average_price']),
                        'quantity': quantity,
                        'message': 'Live trade executed successfully'
                    }
                else:
                    return {
                        'timestamp': datetime.now(),
                        'signal': signal,
                        'status': 'pending',
                        'order_id': order_id,
                        'execution_price': 0,
                        'quantity': quantity,
                        'message': f"Order status: {latest_status['status']}"
                    }
            
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': order_id,
                'execution_price': 0,
                'quantity': 0,
                'message': 'Order status unknown'
            }
            
        except Exception as e:
            logger.error(f"❌ Live trade execution failed: {e}")
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': None,
                'execution_price': 0,
                'quantity': 0,
                'message': str(e)
            }
    
    async def _execute_paper_trade(self, signal: Dict) -> Dict:
        """Execute paper trade for testing"""
        try:
            quantity = self._calculate_quantity(signal)
            execution_price = signal['entry_price']
            
            paper_order_id = f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'success',
                'order_id': paper_order_id,
                'execution_price': execution_price,
                'quantity': quantity,
                'message': 'Paper trade executed successfully'
            }
            
        except Exception as e:
            logger.error(f"❌ Paper trade execution failed: {e}")
            return {
                'timestamp': datetime.now(),
                'signal': signal,
                'status': 'failed',
                'order_id': None,
                'execution_price': 0,
                'quantity': 0,
                'message': str(e)
            }
    
    def _construct_tradingsymbol(self, signal: Dict) -> str:
        """Construct trading symbol for options"""
        try:
            symbol = signal['instrument']
            strike = int(signal['strike'])
            option_type = signal['option_type']
            
            expiry_date = self._get_nearest_expiry()
            expiry_str = expiry_date.strftime('%y%m%d')
            
            if symbol == 'NIFTY':
                tradingsymbol = f"NIFTY{expiry_str}{strike}{option_type}"
            elif symbol == 'BANKNIFTY':
                tradingsymbol = f"BANKNIFTY{expiry_str}{strike}{option_type}"
            else:
                tradingsymbol = f"{symbol}{expiry_str}{strike}{option_type}"
            
            return tradingsymbol
            
        except Exception as e:
            logger.error(f"❌ Trading symbol construction failed: {e}")
            return ""
    
    def _get_nearest_expiry(self) -> datetime:
        """Get nearest Thursday expiry for options"""
        try:
            today = datetime.now()
            
            days_until_thursday = (3 - today.weekday()) % 7
            if days_until_thursday == 0 and today.hour >= 15:
                days_until_thursday = 7
            
            nearest_thursday = today + timedelta(days=days_until_thursday)
            return nearest_thursday
            
        except Exception:
            return datetime.now() + timedelta(days=1)
    
    def _calculate_quantity(self, signal: Dict) -> int:
        """Calculate SEBI compliant quantity"""
        try:
            symbol = signal['instrument']
            lot_size = self.lot_sizes.get(symbol, 25)
            
            base_lots = 1
            
            confidence = signal.get('confidence', 60)
            if confidence > 80:
                base_lots = 2
            elif confidence > 90:
                base_lots = 3
            
            return lot_size * base_lots
            
        except Exception:
            return self.lot_sizes.get(signal.get('instrument', 'NIFTY'), 25)
    
    def _validate_trade_conditions(self, signal: Dict) -> bool:
        """Validate trade conditions before execution"""
        try:
            if len(self.active_positions) >= self.max_positions:
                logger.warning("⚠️ Maximum positions limit reached")
                return False
            
            if self.daily_pnl <= self.max_daily_loss:
                logger.warning("⚠️ Daily loss limit reached")
                return False
            
            if not self._is_market_hours():
                logger.warning("⚠️ Market is closed")
                return False
            
            required_fields = ['instrument', 'strike', 'option_type', 'entry_price', 'confidence']
            if not all(field in signal for field in required_fields):
                logger.warning("⚠️ Signal missing required fields")
                return False
            
            if signal['confidence'] < 60:
                logger.warning("⚠️ Signal confidence below threshold")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade validation failed: {e}")
            return False
    
    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        try:
            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            is_weekday = now.weekday() < 5
            is_market_time = market_open <= now <= market_close
            
            return is_weekday and is_market_time
            
        except Exception:
            return False
    
    async def _update_position_tracking(self, signal: Dict, execution_result: Dict):
        """Update position tracking after trade execution"""
        try:
            position_key = f"{signal['instrument']}_{signal['strike']}_{signal['option_type']}"
            
            position = {
                'symbol': signal['instrument'],
                'strike': signal['strike'],
                'option_type': signal['option_type'],
                'quantity': execution_result['quantity'],
                'entry_price': execution_result['execution_price'],
                'entry_time': execution_result['timestamp'],
                'stop_loss': signal['stop_loss'],
                'target_1': signal['target_1'],
                'target_2': signal['target_2'],
                'order_id': execution_result['order_id'],
                'status': 'open'
            }
            
            self.active_positions[position_key] = position
            
            self.order_history.append({
                'timestamp': execution_result['timestamp'],
                'action': 'BUY',
                'signal': signal,
                'execution': execution_result
            })
            
        except Exception as e:
            logger.error(f"❌ Position tracking update failed: {e}")
    
    async def monitor_positions(self) -> Dict:
        """Monitor active positions for stop loss and targets"""
        monitoring_result = {
            'timestamp': datetime.now(),
            'active_positions': len(self.active_positions),
            'actions_taken': [],
            'daily_pnl': self.daily_pnl
        }
        
        try:
            if not self.active_positions:
                return monitoring_result
            
            for position_key, position in list(self.active_positions.items()):
                current_price = await self._get_current_option_price(position)
                
                if current_price > 0:
                    action = self._check_exit_conditions(position, current_price)
                    
                    if action:
                        exit_result = await self._exit_position(position, current_price, action['reason'])
                        monitoring_result['actions_taken'].append({
                            'position': position_key,
                            'action': action,
                            'exit_result': exit_result
                        })
            
        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")
            monitoring_result['error'] = str(e)
        
        return monitoring_result
    
    async def _get_current_option_price(self, position: Dict) -> float:
        """Get current market price for option"""
        try:
            if not self.kite:
                return 0
            
            tradingsymbol = self._construct_tradingsymbol(position)
            
            quote = self.kite.quote([f"NFO:{tradingsymbol}"])
            
            if quote and tradingsymbol in quote:
                return float(quote[tradingsymbol]['last_price'])
            
            return 0
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get current price: {e}")
            return 0
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> Optional[Dict]:
        """Check if position should be exited"""
        try:
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            target_1 = position['target_1']
            target_2 = position['target_2']
            
            if current_price <= stop_loss:
                return {
                    'action': 'exit',
                    'reason': 'stop_loss',
                    'price': current_price
                }
            
            if current_price >= target_2:
                return {
                    'action': 'exit',
                    'reason': 'target_2',
                    'price': current_price
                }
            
            if current_price >= target_1:
                return {
                    'action': 'partial_exit',
                    'reason': 'target_1',
                    'price': current_price
                }
            
            entry_time = position['entry_time']
            time_elapsed = datetime.now() - entry_time
            
            if time_elapsed > timedelta(hours=4):
                return {
                    'action': 'exit',
                    'reason': 'time_based',
                    'price': current_price
                }
            
            return None
            
        except Exception:
            return None
    
    async def _exit_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """Exit position"""
        try:
            if self.paper_trading:
                pnl = (exit_price - position['entry_price']) * position['quantity']
                self.daily_pnl += pnl
                
                position_key = f"{position['symbol']}_{position['strike']}_{position['option_type']}"
                if position_key in self.active_positions:
                    del self.active_positions[position_key]
                
                return {
                    'status': 'success',
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'reason': reason,
                    'message': 'Paper position exited'
                }
            else:
                tradingsymbol = self._construct_tradingsymbol(position)
                
                order_params = {
                    'variety': 'regular',
                    'exchange': 'NFO',
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': 'SELL',
                    'quantity': position['quantity'],
                    'product': 'MIS',
                    'order_type': 'MARKET',
                    'validity': 'DAY'
                }
                
                order_id = self.kite.place_order(**order_params)
                
                return {
                    'status': 'success',
                    'exit_price': exit_price,
                    'order_id': order_id,
                    'reason': reason,
                    'message': 'Live position exit order placed'
                }
            
        except Exception as e:
            logger.error(f"❌ Position exit failed: {e}")
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        try:
            total_positions = len(self.active_positions)
            total_value = sum(
                pos['entry_price'] * pos['quantity'] 
                for pos in self.active_positions.values()
            )
            
            return {
                'timestamp': datetime.now(),
                'total_positions': total_positions,
                'total_value': total_value,
                'daily_pnl': self.daily_pnl,
                'max_daily_loss': self.max_daily_loss,
                'paper_trading': self.paper_trading,
                'positions': list(self.active_positions.values())
            }
            
        except Exception as e:
            logger.error(f"❌ Position summary failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def reset_daily_tracking(self):
        """Reset daily P&L tracking"""
        try:
            self.daily_pnl = 0
            self.order_history = []
            logger.info("✅ Daily tracking reset")
            
        except Exception as e:
            logger.error(f"❌ Daily tracking reset failed: {e}")
