"""
TradeMind_AI: Configuration Validator
Validates all configuration settings and ensures consistency
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigValidator:
    """Comprehensive configuration validation for TradeMind_AI"""
    
    def __init__(self):
        """Initialize configuration validator"""
        self.validation_results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Import constants for validation
        try:
            from config.constants import (
                LOT_SIZES, STRIKE_GAPS, SECURITY_IDS, EXCHANGE_SEGMENTS,
                RISK_MANAGEMENT, API_RATE_LIMITS, MARKET_HOURS, INDICATOR_SETTINGS,
                GREEKS_THRESHOLDS, PAPER_TRADING, LIVE_TRADING, NOTIFICATIONS,
                DATABASE, STRATEGIES, MULTI_LEG_STRATEGIES, NSE_HOLIDAYS_2025,
                MONITORING, VALIDATION_RULES, GLOBAL_MARKETS
            )
            self.constants_available = True
            self.constants = {
                'LOT_SIZES': LOT_SIZES,
                'STRIKE_GAPS': STRIKE_GAPS,
                'SECURITY_IDS': SECURITY_IDS,
                'EXCHANGE_SEGMENTS': EXCHANGE_SEGMENTS,
                'RISK_MANAGEMENT': RISK_MANAGEMENT,
                'API_RATE_LIMITS': API_RATE_LIMITS,
                'MARKET_HOURS': MARKET_HOURS,
                'INDICATOR_SETTINGS': INDICATOR_SETTINGS,
                'GREEKS_THRESHOLDS': GREEKS_THRESHOLDS,
                'PAPER_TRADING': PAPER_TRADING,
                'LIVE_TRADING': LIVE_TRADING,
                'NOTIFICATIONS': NOTIFICATIONS,
                'DATABASE': DATABASE,
                'STRATEGIES': STRATEGIES,
                'MULTI_LEG_STRATEGIES': MULTI_LEG_STRATEGIES,
                'NSE_HOLIDAYS_2025': NSE_HOLIDAYS_2025,
                'MONITORING': MONITORING,
                'VALIDATION_RULES': VALIDATION_RULES,
                'GLOBAL_MARKETS': GLOBAL_MARKETS
            }
        except ImportError as e:
            self.constants_available = False
            self.validation_results['failed'].append(f"Cannot import constants: {e}")
    
    def validate_environment_variables(self) -> bool:
        """Validate all required environment variables"""
        print("🔍 Validating Environment Variables...")
        
        # Required environment variables
        required_vars = {
            'DHAN_CLIENT_ID': 'Dhan API Client ID',
            'DHAN_ACCESS_TOKEN': 'Dhan API Access Token',
            'TOTAL_CAPITAL': 'Total Trading Capital'
        }
        
        # Optional but recommended variables
        optional_vars = {
            'TELEGRAM_BOT_TOKEN': 'Telegram Bot Token for notifications',
            'TELEGRAM_CHAT_ID': 'Telegram Chat ID for notifications',
            'NEWS_API_KEY': 'News API Key for sentiment analysis',
            'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage API Key for market data'
        }
        
        all_passed = True
        
        # Check required variables
        for var, description in required_vars.items():
            value = os.getenv(var)
            if value:
                if var == 'TOTAL_CAPITAL':
                    try:
                        capital = float(value)
                        if capital > 0:
                            self.validation_results['passed'].append(f"✅ {var}: ₹{capital:,.2f}")
                        else:
                            self.validation_results['failed'].append(f"❌ {var}: Must be positive")
                            all_passed = False
                    except ValueError:
                        self.validation_results['failed'].append(f"❌ {var}: Invalid number format")
                        all_passed = False
                else:
                    # Mask sensitive information
                    masked_value = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
                    self.validation_results['passed'].append(f"✅ {var}: {masked_value}")
            else:
                self.validation_results['failed'].append(f"❌ {var}: Missing ({description})")
                all_passed = False
        
        # Check optional variables
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if value:
                masked_value = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
                self.validation_results['passed'].append(f"✅ {var}: {masked_value}")
            else:
                self.validation_results['warnings'].append(f"⚠️ {var}: Not set ({description})")
        
        return all_passed
    
    def validate_constants_configuration(self) -> bool:
        """Validate constants configuration"""
        print("🔍 Validating Constants Configuration...")
        
        if not self.constants_available:
            return False
        
        all_passed = True
        
        # Validate lot sizes
        try:
            lot_sizes = self.constants['LOT_SIZES']
            expected_lot_sizes = {'NIFTY': 75, 'BANKNIFTY': 30}
            
            for symbol, expected_size in expected_lot_sizes.items():
                if symbol in lot_sizes:
                    if lot_sizes[symbol] == expected_size:
                        self.validation_results['passed'].append(f"✅ {symbol} lot size: {lot_sizes[symbol]} (SEBI 2025)")
                    else:
                        self.validation_results['warnings'].append(f"⚠️ {symbol} lot size: {lot_sizes[symbol]} (expected {expected_size})")
                else:
                    self.validation_results['failed'].append(f"❌ {symbol} lot size: Missing")
                    all_passed = False
        except Exception as e:
            self.validation_results['failed'].append(f"❌ Lot sizes validation error: {e}")
            all_passed = False
        
        # Validate risk management settings
        try:
            risk_mgmt = self.constants['RISK_MANAGEMENT']
            required_risk_params = [
                'MAX_RISK_PER_TRADE', 'MAX_DAILY_LOSS', 'MAX_PORTFOLIO_RISK',
                'DEFAULT_STOP_LOSS_PERCENT', 'MAX_TRADES_PER_DAY'
            ]
            
            for param in required_risk_params:
                if param in risk_mgmt:
                    value = risk_mgmt[param]
                    if isinstance(value, (int, float)) and value > 0:
                        self.validation_results['passed'].append(f"✅ {param}: {value}")
                    else:
                        self.validation_results['failed'].append(f"❌ {param}: Invalid value")
                        all_passed = False
                else:
                    self.validation_results['failed'].append(f"❌ {param}: Missing")
                    all_passed = False
        except Exception as e:
            self.validation_results['failed'].append(f"❌ Risk management validation error: {e}")
            all_passed = False
        
        # Validate market hours
        try:
            market_hours = self.constants['MARKET_HOURS']
            required_hours = ['MARKET_OPEN', 'MARKET_CLOSE', 'PRE_OPEN_START', 'PRE_OPEN_END']
            
            for hour_type in required_hours:
                if hour_type in market_hours:
                    time_str = market_hours[hour_type]
                    try:
                        datetime.strptime(time_str, '%H:%M')
                        self.validation_results['passed'].append(f"✅ {hour_type}: {time_str}")
                    except ValueError:
                        self.validation_results['failed'].append(f"❌ {hour_type}: Invalid time format")
                        all_passed = False
                else:
                    self.validation_results['failed'].append(f"❌ {hour_type}: Missing")
                    all_passed = False
        except Exception as e:
            self.validation_results['failed'].append(f"❌ Market hours validation error: {e}")
            all_passed = False
        
        # Validate security IDs
        try:
            security_ids = self.constants['SECURITY_IDS']
            expected_ids = {'NIFTY': 13, 'BANKNIFTY': 25}
            
            for symbol, expected_id in expected_ids.items():
                if symbol in security_ids:
                    if security_ids[symbol] == expected_id:
                        self.validation_results['passed'].append(f"✅ {symbol} security ID: {security_ids[symbol]}")
                    else:
                        self.validation_results['warnings'].append(f"⚠️ {symbol} security ID: {security_ids[symbol]} (expected {expected_id})")
                else:
                    self.validation_results['failed'].append(f"❌ {symbol} security ID: Missing")
                    all_passed = False
        except Exception as e:
            self.validation_results['failed'].append(f"❌ Security IDs validation error: {e}")
            all_passed = False
        
        return all_passed
    
    def validate_directory_structure(self) -> bool:
        """Validate required directory structure"""
        print("🔍 Validating Directory Structure...")
        
        required_directories = [
            'src', 'src/core', 'src/analysis', 'src/data', 'src/portfolio', 
            'src/utils', 'src/dashboard', 'config', 'data', 'logs', 'reports'
        ]
        
        optional_directories = [
            'models', 'logs/audit', 'data/historical', 'data/news', 'templates'
        ]
        
        all_passed = True
        
        # Check required directories
        for directory in required_directories:
            if os.path.exists(directory):
                self.validation_results['passed'].append(f"✅ Directory: {directory}")
            else:
                self.validation_results['failed'].append(f"❌ Directory missing: {directory}")
                all_passed = False
        
        # Check optional directories
        for directory in optional_directories:
            if os.path.exists(directory):
                self.validation_results['passed'].append(f"✅ Optional directory: {directory}")
            else:
                self.validation_results['warnings'].append(f"⚠️ Optional directory missing: {directory}")
        
        return all_passed
    
    def validate_critical_files(self) -> bool:
        """Validate critical files exist"""
        print("🔍 Validating Critical Files...")
        
        critical_files = [
            'run_trading.py',
            'config/constants.py',
            'config/__init__.py',
            'src/__init__.py',
            'src/core/unified_trading_engine.py',
            'src/core/master_trader.py',
            'src/utils/balance_utils.py',
            'requirements.txt'
        ]
        
        important_files = [
            'src/data/market_data.py',
            'src/portfolio/portfolio_manager.py',
            'src/analysis/technical_indicators.py',
            'src/analysis/exact_recommender.py',
            'src/dashboard/dashboard.py'
        ]
        
        all_passed = True
        
        # Check critical files
        for file_path in critical_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.validation_results['passed'].append(f"✅ Critical file: {file_path} ({file_size} bytes)")
            else:
                self.validation_results['failed'].append(f"❌ Critical file missing: {file_path}")
                all_passed = False
        
        # Check important files
        for file_path in important_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.validation_results['passed'].append(f"✅ Important file: {file_path} ({file_size} bytes)")
            else:
                self.validation_results['warnings'].append(f"⚠️ Important file missing: {file_path}")
        
        return all_passed
    
    def validate_data_files(self) -> bool:
        """Validate data files and their structure"""
        print("🔍 Validating Data Files...")
        
        data_files = [
            'data/trades_database.json',
            'data/balance_history.json'
        ]
        
        all_passed = True
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, (dict, list)):
                        self.validation_results['passed'].append(f"✅ Data file: {file_path} (valid JSON)")
                    else:
                        self.validation_results['warnings'].append(f"⚠️ Data file: {file_path} (unexpected format)")
                        
                except json.JSONDecodeError:
                    self.validation_results['failed'].append(f"❌ Data file: {file_path} (invalid JSON)")
                    all_passed = False
                except Exception as e:
                    self.validation_results['failed'].append(f"❌ Data file: {file_path} (error: {e})")
                    all_passed = False
            else:
                self.validation_results['warnings'].append(f"⚠️ Data file will be created: {file_path}")
        
        return all_passed
    
    def check_api_connectivity(self) -> bool:
        """Test API connectivity"""
        print("🔍 Testing API Connectivity...")
        
        try:
            from dhanhq import DhanContext, dhanhq
            
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            
            if not client_id or not access_token:
                self.validation_results['failed'].append("❌ API Test: Missing credentials")
                return False
            
            # Test Dhan API connection
            dhan_context = DhanContext(client_id, access_token)
            dhan_client = dhanhq(dhan_context)
            
            # Try to get fund limits as a connectivity test
            response = dhan_client.get_fund_limits()
            
            if response and response.get('status') == 'success':
                self.validation_results['passed'].append("✅ Dhan API: Connected successfully")
                return True
            else:
                self.validation_results['failed'].append("❌ Dhan API: Connection failed")
                return False
                
        except ImportError:
            self.validation_results['failed'].append("❌ dhanhq library not installed")
            return False
        except Exception as e:
            self.validation_results['failed'].append(f"❌ API Test error: {e}")
            return False
    
    def generate_recommendations(self) -> None:
        """Generate configuration recommendations"""
        print("🔍 Generating Recommendations...")
        
        # Environment recommendations
        if not os.getenv('TELEGRAM_BOT_TOKEN'):
            self.validation_results['recommendations'].append(
                "📱 Set up Telegram notifications for trade alerts"
            )
        
        if not os.getenv('NEWS_API_KEY'):
            self.validation_results['recommendations'].append(
                "📰 Add News API key for sentiment analysis"
            )
        
        # Capital recommendations
        total_capital = os.getenv('TOTAL_CAPITAL')
        if total_capital:
            try:
                capital = float(total_capital)
                if capital < 50000:
                    self.validation_results['recommendations'].append(
                        "💰 Consider increasing capital above ₹50,000 for better diversification"
                    )
                elif capital < 100000:
                    self.validation_results['recommendations'].append(
                        "💰 Current capital is adequate for basic trading"
                    )
                else:
                    self.validation_results['recommendations'].append(
                        "💰 Excellent capital allocation for advanced strategies"
                    )
            except ValueError:
                pass
        
        # Risk management recommendations
        if self.constants_available:
            risk_mgmt = self.constants.get('RISK_MANAGEMENT', {})
            max_risk = risk_mgmt.get('MAX_RISK_PER_TRADE', 0)
            
            if max_risk > 0.02:  # More than 2%
                self.validation_results['recommendations'].append(
                    "⚠️ Consider reducing MAX_RISK_PER_TRADE to 1-2% for better risk management"
                )
            
            max_daily_loss = risk_mgmt.get('MAX_DAILY_LOSS', 0)
            if max_daily_loss > 0.05:  # More than 5%
                self.validation_results['recommendations'].append(
                    "⚠️ Consider reducing MAX_DAILY_LOSS to 3-5% to preserve capital"
                )
        
        # Performance recommendations
        self.validation_results['recommendations'].append(
            "📊 Enable comprehensive logging for better trade analysis"
        )
        
        self.validation_results['recommendations'].append(
            "🔄 Set up automated backup of trade data"
        )
        
        self.validation_results['recommendations'].append(
            "📈 Consider implementing paper trading before live trading"
        )
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete configuration validation"""
        print("\n🔍 RUNNING COMPREHENSIVE CONFIGURATION VALIDATION")
        print("="*70)
        
        validation_steps = [
            ("Environment Variables", self.validate_environment_variables),
            ("Constants Configuration", self.validate_constants_configuration),
            ("Directory Structure", self.validate_directory_structure),
            ("Critical Files", self.validate_critical_files),
            ("Data Files", self.validate_data_files),
            ("API Connectivity", self.check_api_connectivity)
        ]
        
        overall_status = True
        step_results = {}
        
        for step_name, validation_func in validation_steps:
            print(f"\n📋 {step_name}:")
            try:
                result = validation_func()
                step_results[step_name] = result
                if not result:
                    overall_status = False
                print(f"{'✅' if result else '❌'} {step_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                step_results[step_name] = False
                overall_status = False
                print(f"❌ {step_name}: ERROR - {e}")
                self.validation_results['failed'].append(f"{step_name} validation error: {e}")
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Compile final results
        final_results = {
            'overall_status': overall_status,
            'step_results': step_results,
            'validation_results': self.validation_results,
            'summary': {
                'passed': len(self.validation_results['passed']),
                'failed': len(self.validation_results['failed']),
                'warnings': len(self.validation_results['warnings']),
                'recommendations': len(self.validation_results['recommendations'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display validation results in a formatted way"""
        print("\n" + "="*70)
        print("📊 CONFIGURATION VALIDATION RESULTS")
        print("="*70)
        
        # Overall status
        status_emoji = "✅" if results['overall_status'] else "❌"
        print(f"\n{status_emoji} Overall Status: {'PASSED' if results['overall_status'] else 'FAILED'}")
        
        # Summary
        summary = results['summary']
        print(f"\n📈 Summary:")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⚠️ Warnings: {summary['warnings']}")
        print(f"   💡 Recommendations: {summary['recommendations']}")
        
        # Detailed results
        validation_results = results['validation_results']
        
        if validation_results['passed']:
            print(f"\n✅ PASSED VALIDATIONS:")
            for item in validation_results['passed']:
                print(f"   {item}")
        
        if validation_results['failed']:
            print(f"\n❌ FAILED VALIDATIONS:")
            for item in validation_results['failed']:
                print(f"   {item}")
        
        if validation_results['warnings']:
            print(f"\n⚠️ WARNINGS:")
            for item in validation_results['warnings']:
                print(f"   {item}")
        
        if validation_results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for item in validation_results['recommendations']:
                print(f"   {item}")
        
        print(f"\n⏰ Generated: {results['timestamp']}")
        print("="*70)
    
    def save_results(self, results: Dict[str, Any], filename: str = "config_validation_report.json") -> None:
        """Save validation results to file"""
        try:
            os.makedirs('logs', exist_ok=True)
            filepath = os.path.join('logs', filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Validation report saved: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")

def main():
    """Main function to run configuration validation"""
    print("🛡️ TradeMind_AI Configuration Validator")
    print("🔍 Ensuring system integrity and optimal configuration")
    
    try:
        validator = ConfigValidator()
        results = validator.run_full_validation()
        validator.display_results(results)
        validator.save_results(results)
        
        if results['overall_status']:
            print("\n🎉 Configuration validation completed successfully!")
            print("✅ Your TradeMind_AI system is properly configured!")
        else:
            print("\n⚠️ Configuration issues detected!")
            print("🔧 Please address the failed validations above.")
            
    except Exception as e:
        print(f"❌ Validation error: {e}")

if __name__ == "__main__":
    main()