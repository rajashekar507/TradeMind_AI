"""
TradeMind_AI: Comprehensive Unit Tests
Tests for unified trading engine and related components
"""

import unittest
import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import (
    LOT_SIZES, RISK_MANAGEMENT, PAPER_TRADING,
    get_lot_size, validate_order_params, is_trading_holiday
)
from core.unified_trading_engine import (
    UnifiedTradingEngine, TradingMode, OrderDetails, TradeResult
)
from utils.rate_limiter import RateLimiter, rate_limiter


class TestConstants(unittest.TestCase):
    """Test configuration constants"""
    
    def test_lot_sizes(self):
        """Test lot size configuration"""
        self.assertEqual(get_lot_size('NIFTY'), 75)
        self.assertEqual(get_lot_size('BANKNIFTY'), 30)
        self.assertEqual(get_lot_size('UNKNOWN'), 75)  # Default
    
    def test_order_validation(self):
        """Test order parameter validation"""
        # Valid order
        valid, errors = validate_order_params('NIFTY', 1, 150)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid symbol
        valid, errors = validate_order_params('INVALID', 1, 150)
        self.assertFalse(valid)
        self.assertIn('Invalid symbol: INVALID', errors[0])
        
        # Invalid quantity
        valid, errors = validate_order_params('NIFTY', 0, 150)
        self.assertFalse(valid)
        self.assertIn('Invalid quantity: 0', errors[0])
        
        # Invalid price
        valid, errors = validate_order_params('NIFTY', 1, -10)
        self.assertFalse(valid)
        self.assertIn('Price -10', errors[0])
    
    def test_holiday_check(self):
        """Test holiday checking"""
        # Test with a known holiday
        republic_day = datetime(2025, 1, 26)
        self.assertTrue(is_trading_holiday(republic_day))
        
        # Test with a regular day
        regular_day = datetime(2025, 1, 27)
        self.assertFalse(is_trading_holiday(regular_day))


class TestUnifiedTradingEngine(unittest.TestCase):
    """Test unified trading engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = UnifiedTradingEngine(TradingMode.SIMULATION)
        
        # Create test order
        self.test_order = OrderDetails(
            symbol='NIFTY',
            strike=25000,
            option_type='CE',
            action='BUY',
            quantity=1,
            order_type='MARKET',
            price=150
        )
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.mode, TradingMode.SIMULATION)
        self.assertEqual(self.engine.trades_today, 0)
        self.assertEqual(self.engine.daily_pnl, 0)
        self.assertIsNone(self.engine.dhan_client)
    
    def test_mode_switching(self):
        """Test trading mode switching"""
        # Switch to paper mode
        result = self.engine.switch_mode(TradingMode.PAPER)
        self.assertTrue(result)
        self.assertEqual(self.engine.mode, TradingMode.PAPER)
        
        # Try to switch to live mode (should fail without credentials)
        result = self.engine.switch_mode(TradingMode.LIVE)
        self.assertFalse(result)  # Should fail due to no Dhan client
    
    def test_paper_order_placement(self):
        """Test paper trading order placement"""
        # Switch to paper mode
        self.engine.switch_mode(TradingMode.PAPER)
        
        # Place order
        result = self.engine.place_order(self.test_order)
        
        self.assertTrue(result.success)
        self.assertIn('PAPER_', result.order_id)
        self.assertEqual(self.engine.trades_today, 1)
        
        # Check paper positions
        positions = self.engine.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'NIFTY')
        
        # Check capital deduction
        initial_capital = PAPER_TRADING['INITIAL_CAPITAL']
        order_value = self.test_order.quantity * LOT_SIZES['NIFTY'] * self.test_order.price
        brokerage = PAPER_TRADING['BROKERAGE_PER_LOT'] * self.test_order.quantity
        expected_capital = initial_capital - order_value - brokerage
        
        self.assertEqual(self.engine.paper_capital, expected_capital)
    
    def test_order_validation(self):
        """Test order validation"""
        # Invalid symbol
        invalid_order = OrderDetails(
            symbol='INVALID',
            strike=25000,
            option_type='CE',
            action='BUY',
            quantity=1,
            order_type='MARKET',
            price=150
        )
        
        result = self.engine.place_order(invalid_order)
        self.assertFalse(result.success)
        self.assertIn('Validation failed', result.message)
    
    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement"""
        self.engine.switch_mode(TradingMode.PAPER)
        
        # Set trades to max
        self.engine.trades_today = RISK_MANAGEMENT['MAX_TRADES_PER_DAY']
        
        # Try to place order
        result = self.engine.place_order(self.test_order)
        self.assertFalse(result.success)
        self.assertIn('Daily trade limit reached', result.message)
    
    def test_insufficient_capital(self):
        """Test insufficient capital handling"""
        self.engine.switch_mode(TradingMode.PAPER)
        
        # Set low capital
        self.engine.paper_capital = 1000
        
        # Try to place order
        result = self.engine.place_order(self.test_order)
        self.assertFalse(result.success)
        self.assertIn('Insufficient paper capital', result.message)
    
    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions"""
        self.engine.switch_mode(TradingMode.PAPER)
        
        # Place order
        self.engine.place_order(self.test_order)
        
        # Update market prices
        market_prices = {
            f"NIFTY_25000_CE": 160  # Price increased
        }
        
        self.engine.update_paper_positions(market_prices)
        
        # Check P&L
        positions = self.engine.get_positions()
        position = positions[0]
        
        expected_pnl = (160 - 150) * 1 * LOT_SIZES['NIFTY']
        self.assertEqual(position['pnl'], expected_pnl)
    
    def test_get_account_balance(self):
        """Test account balance retrieval"""
        self.engine.switch_mode(TradingMode.PAPER)
        
        balance = self.engine.get_account_balance()
        
        self.assertEqual(balance['mode'], 'PAPER')
        self.assertEqual(balance['total_balance'], PAPER_TRADING['INITIAL_CAPITAL'])
        self.assertEqual(balance['available_balance'], self.engine.paper_capital)
    
    def test_trade_logging(self):
        """Test trade logging functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override log directory
            self.engine.trade_log_file = os.path.join(temp_dir, 'test_trades.json')
            
            # Place order
            result = self.engine.place_order(self.test_order)
            
            # Check log file exists
            self.assertTrue(os.path.exists(self.engine.trade_log_file))
            
            # Read log file
            with open(self.engine.trade_log_file, 'r') as f:
                logs = json.load(f)
            
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]['order']['symbol'], 'NIFTY')
            self.assertEqual(logs[0]['result']['success'], True)


class TestRateLimiter(unittest.TestCase):
    """Test API rate limiter"""
    
    def setUp(self):
        """Set up test rate limiter"""
        self.limiter = RateLimiter('TEST_API')
        self.limiter.limits = {
            'calls_per_second': 2,
            'calls_per_minute': 10,
            'calls_per_hour': 100
        }
        self.limiter.max_tokens = 2
        self.limiter.tokens = 2
    
    def test_token_bucket(self):
        """Test token bucket algorithm"""
        # First two requests should succeed immediately
        can_proceed, wait_time = self.limiter.can_make_request()
        self.assertTrue(can_proceed)
        self.assertIsNone(wait_time)
        
        self.limiter.record_request()
        
        can_proceed, wait_time = self.limiter.can_make_request()
        self.assertTrue(can_proceed)
        
        self.limiter.record_request()
        
        # Third request should be rate limited
        can_proceed, wait_time = self.limiter.can_make_request()
        self.assertFalse(can_proceed)
        self.assertIsNotNone(wait_time)
        self.assertGreater(wait_time, 0)
    
    def test_sliding_window(self):
        """Test sliding window rate limiting"""
        # Fill up minute limit
        for _ in range(10):
            self.limiter.record_request()
        
        # Next request should be rate limited
        can_proceed, wait_time = self.limiter.can_make_request()
        self.assertFalse(can_proceed)
        self.assertIsNotNone(wait_time)
    
    def test_statistics(self):
        """Test rate limiter statistics"""
        # Make some requests
        for _ in range(3):
            if self.limiter.can_make_request()[0]:
                self.limiter.record_request()
        
        stats = self.limiter.get_stats()
        
        self.assertEqual(stats['api'], 'TEST_API')
        self.assertGreaterEqual(stats['total_calls'], 2)
        self.assertIn('calls_last_minute', stats)
        self.assertIn('available_tokens', stats)
    
    def test_rate_limit_decorator(self):
        """Test rate limit decorator"""
        call_count = 0
        
        @rate_limiter.rate_limit('TEST_API')
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        # Mock the rate limiter
        with patch.object(rate_limiter, 'wait_if_needed', return_value=True):
            result = test_function()
            self.assertEqual(result, "success")
            self.assertEqual(call_count, 1)


class TestOrderExecution(unittest.TestCase):
    """Test order execution scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = UnifiedTradingEngine(TradingMode.PAPER)
    
    def test_buy_sell_cycle(self):
        """Test complete buy-sell cycle"""
        # Buy order
        buy_order = OrderDetails(
            symbol='NIFTY',
            strike=25000,
            option_type='CE',
            action='BUY',
            quantity=1,
            order_type='MARKET',
            price=150
        )
        
        buy_result = self.engine.place_order(buy_order)
        self.assertTrue(buy_result.success)
        
        # Update price
        self.engine.update_paper_positions({'NIFTY_25000_CE': 160})
        
        # Sell order
        sell_order = OrderDetails(
            symbol='NIFTY',
            strike=25000,
            option_type='CE',
            action='SELL',
            quantity=1,
            order_type='MARKET',
            price=160
        )
        
        sell_result = self.engine.place_order(sell_order)
        self.assertTrue(sell_result.success)
        
        # Check P&L
        pnl = self.engine.get_pnl()
        expected_profit = (160 - 150) * LOT_SIZES['NIFTY'] - (2 * PAPER_TRADING['BROKERAGE_PER_LOT'])
        
        # Allow small difference due to slippage
        self.assertAlmostEqual(pnl['realized_pnl'], expected_profit, delta=100)
    
    def test_multi_position_management(self):
        """Test managing multiple positions"""
        positions_to_create = [
            ('NIFTY', 25000, 'CE', 150),
            ('NIFTY', 25000, 'PE', 100),
            ('BANKNIFTY', 52000, 'CE', 200)
        ]
        
        # Create positions
        for symbol, strike, opt_type, price in positions_to_create:
            order = OrderDetails(
                symbol=symbol,
                strike=strike,
                option_type=opt_type,
                action='BUY',
                quantity=1,
                order_type='MARKET',
                price=price
            )
            result = self.engine.place_order(order)
            self.assertTrue(result.success)
        
        # Check positions
        positions = self.engine.get_positions()
        self.assertEqual(len(positions), 3)
        
        # Check position details
        nifty_ce = next(p for p in positions if p['symbol'] == 'NIFTY' and p['option_type'] == 'CE')
        self.assertEqual(nifty_ce['strike'], 25000)
        self.assertEqual(nifty_ce['entry_price'], 150)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @patch('requests.post')
    def test_telegram_notification(self, mock_post):
        """Test Telegram notification"""
        mock_post.return_value.status_code = 200
        
        engine = UnifiedTradingEngine(TradingMode.PAPER)
        engine.telegram_token = 'test_token'
        engine.telegram_chat_id = 'test_chat'
        
        # Place order to trigger notification
        order = OrderDetails(
            symbol='NIFTY',
            strike=25000,
            option_type='CE',
            action='BUY',
            quantity=1,
            order_type='MARKET',
            price=150
        )
        
        result = engine.place_order(order)
        
        # Check notification was sent
        mock_post.assert_called()
        call_args = mock_post.call_args[1]['data']
        self.assertEqual(call_args['chat_id'], 'test_chat')
        self.assertIn('PAPER ORDER EXECUTED', call_args['text'])


def run_all_tests():
    """Run all unit tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_all_tests()