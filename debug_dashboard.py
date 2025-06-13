"""
Debug Dashboard - Find the exact missing field causing DH-905 error
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DebugDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'debug_key'
        CORS(self.app)
        
        # Initialize Dhan client
        self.dhan_client = None
        self._initialize_dhan()
        self._setup_routes()
    
    def _initialize_dhan(self):
        """Initialize Dhan client"""
        try:
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            from dhanhq import DhanContext, dhanhq
            dhan_context = DhanContext(client_id=client_id, access_token=access_token)
            self.dhan_client = dhanhq(dhan_context)
            
            logger.info("Dhan client initialized for debugging")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Dhan client: {e}")
            return False
    
    def _setup_routes(self):
        """Setup debug routes"""
        
        @self.app.route('/')
        def home():
            return """
            <h1>TradeMind AI Debug Tool</h1>
            <p>Click the links below to test different parameter combinations:</p>
            <ul>
                <li><a href="/test1">Test 1: Basic parameters (like your working test)</a></li>
                <li><a href="/test2">Test 2: With expiry_code=0</a></li>
                <li><a href="/test3">Test 3: With all possible fields</a></li>
                <li><a href="/test4">Test 4: Different security IDs</a></li>
                <li><a href="/test5">Test 5: Copy EXACT working test parameters</a></li>
            </ul>
            """
        
        @self.app.route('/test1')
        def test1():
            """Test 1: Basic parameters"""
            try:
                from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info("=== TEST 1: Basic Parameters ===")
                logger.info(f"Parameters: security_id=13, exchange_segment=IDX_I, instrument_type=INDEX")
                logger.info(f"Date range: {from_date} to {to_date}")
                
                response = self.dhan_client.historical_daily_data(
                    security_id='13',
                    exchange_segment='IDX_I',
                    instrument_type='INDEX',
                    from_date=from_date,
                    to_date=to_date
                )
                
                return jsonify({
                    'test': 'Basic Parameters', 
                    'status': 'success',
                    'parameters': {
                        'security_id': '13',
                        'exchange_segment': 'IDX_I',
                        'instrument_type': 'INDEX',
                        'from_date': from_date,
                        'to_date': to_date
                    },
                    'response': response
                })
                
            except Exception as e:
                return jsonify({
                    'test': 'Basic Parameters',
                    'status': 'error',
                    'error': str(e),
                    'parameters': {
                        'security_id': '13',
                        'exchange_segment': 'IDX_I',
                        'instrument_type': 'INDEX',
                        'from_date': from_date,
                        'to_date': to_date
                    }
                })
        
        @self.app.route('/test2')
        def test2():
            """Test 2: With expiry_code"""
            try:
                from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info("=== TEST 2: With expiry_code ===")
                
                response = self.dhan_client.historical_daily_data(
                    security_id='13',
                    exchange_segment='IDX_I',
                    instrument_type='INDEX',
                    expiry_code=0,
                    from_date=from_date,
                    to_date=to_date
                )
                
                return jsonify({
                    'test': 'With expiry_code',
                    'status': 'success',
                    'parameters': {
                        'security_id': '13',
                        'exchange_segment': 'IDX_I',
                        'instrument_type': 'INDEX',
                        'expiry_code': 0,
                        'from_date': from_date,
                        'to_date': to_date
                    },
                    'response': response
                })
                
            except Exception as e:
                return jsonify({
                    'test': 'With expiry_code',
                    'status': 'error',
                    'error': str(e)
                })
        
        @self.app.route('/test3')
        def test3():
            """Test 3: All possible fields"""
            try:
                from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info("=== TEST 3: All Fields ===")
                
                # Try with all possible fields
                response = self.dhan_client.historical_daily_data(
                    security_id='13',
                    exchange_segment='IDX_I',
                    instrument_type='INDEX',
                    expiry_code=0,
                    from_date=from_date,
                    to_date=to_date,
                    oi=False  # Sometimes needed for options/futures
                )
                
                return jsonify({
                    'test': 'All Fields', 
                    'status': 'success',
                    'response': response
                })
                
            except Exception as e:
                return jsonify({
                    'test': 'All Fields',
                    'status': 'error',
                    'error': str(e)
                })
        
        @self.app.route('/test4')
        def test4():
            """Test 4: Different security IDs"""
            try:
                from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info("=== TEST 4: Different Security IDs ===")
                
                # Test with different indices
                results = {}
                
                # Test NIFTY with different ID formats
                for nifty_id in ['13', '1333', 'NIFTY']:
                    try:
                        response = self.dhan_client.historical_daily_data(
                            security_id=nifty_id,
                            exchange_segment='IDX_I',
                            instrument_type='INDEX',
                            from_date=from_date,
                            to_date=to_date
                        )
                        results[f'NIFTY_{nifty_id}'] = {'status': 'success', 'response_status': response.get('status')}
                    except Exception as e:
                        results[f'NIFTY_{nifty_id}'] = {'status': 'error', 'error': str(e)[:100]}
                
                return jsonify({
                    'test': 'Different Security IDs',
                    'results': results
                })
                
            except Exception as e:
                return jsonify({
                    'test': 'Different Security IDs',
                    'status': 'error',
                    'error': str(e)
                })
        
        @self.app.route('/test5')
        def test5():
            """Test 5: Copy EXACT working test parameters"""
            try:
                logger.info("=== TEST 5: Exact Copy of Working Test ===")
                
                # This is EXACTLY what worked in your test
                from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
                
                logger.info(f"Calling with exact test parameters:")
                logger.info(f"security_id='13'")
                logger.info(f"exchange_segment='IDX_I'")
                logger.info(f"instrument_type='INDEX'")
                logger.info(f"from_date='{from_date}'")
                logger.info(f"to_date='{to_date}'")
                
                response = self.dhan_client.historical_daily_data(
                    security_id='13',
                    exchange_segment='IDX_I',
                    instrument_type='INDEX',
                    from_date=from_date,
                    to_date=to_date
                )
                
                logger.info(f"Response received: {response}")
                
                return jsonify({
                    'test': 'Exact Copy of Working Test',
                    'status': 'success',
                    'note': 'This should work since your test_market_data.py showed success',
                    'parameters_used': {
                        'security_id': '13',
                        'exchange_segment': 'IDX_I', 
                        'instrument_type': 'INDEX',
                        'from_date': from_date,
                        'to_date': to_date
                    },
                    'response': response
                })
                
            except Exception as e:
                logger.error(f"Test 5 failed: {e}")
                return jsonify({
                    'test': 'Exact Copy of Working Test',
                    'status': 'error',
                    'error': str(e),
                    'note': 'This is strange - your test showed this should work!'
                })
        
        @self.app.route('/compare')
        def compare():
            """Compare what worked vs what doesn't"""
            return jsonify({
                'working_test_parameters': {
                    'note': 'From your test_market_data.py - this worked',
                    'method': 'historical_daily_data',
                    'security_id': '13',
                    'exchange_segment': 'IDX_I',
                    'instrument_type': 'INDEX',
                    'from_date': '2025-06-06',
                    'to_date': '2025-06-13',
                    'result': 'SUCCESS'
                },
                'failing_dashboard_parameters': {
                    'note': 'From dashboard - this fails with DH-905',
                    'method': 'historical_daily_data',
                    'security_id': '13',
                    'exchange_segment': 'IDX_I',
                    'instrument_type': 'INDEX',
                    'from_date': '2025-06-06',
                    'to_date': '2025-06-13',
                    'result': 'DH-905 Error'
                },
                'analysis': 'Parameters look identical - something else must be different'
            })
    
    def run(self):
        """Run the debug app"""
        logger.info("Starting Debug Dashboard on http://127.0.0.1:5001")
        self.app.run(host='127.0.0.1', port=5001, debug=True)

if __name__ == "__main__":
    debug = DebugDashboard()
    debug.run()