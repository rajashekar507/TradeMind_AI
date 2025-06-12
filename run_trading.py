#!/usr/bin/env python3
"""
TradeMind_AI - Main Entry Point
Select what you want to run
FIXED VERSION - Import path standardization
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Standardized imports with error handling
try:
    from src.core.unified_trading_engine import UnifiedTradingEngine, TradingMode
    from src.utils.auto_scheduler import HolidayAwareScheduler
    from src.analysis.exact_recommender import ExactStrikeRecommender
    from src.data.Balance_fetcher import DhanBalanceFetcher
    from src.portfolio.portfolio_manager import PortfolioManager
    from src.core.master_trader import EnhancedMasterTrader
    from src.utils.excel_reporter import ExcelReporter
    from src.analysis.greeks_calculator import GreeksCalculator
    from src.data.market_data import MarketDataEngine
    IMPORTS_SUCCESSFUL = True
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you're running from the TradeMind_AI root directory")
    print("📁 Current directory:", os.getcwd())
    print("📁 Expected structure:")
    print("   TradeMind_AI/")
    print("   ├── src/")
    print("   ├── config/")
    print("   ├── data/")
    print("   └── run_trading.py")
    IMPORTS_SUCCESSFUL = False

def check_imports():
    """Check if all imports are successful"""
    if not IMPORTS_SUCCESSFUL:
        print("\n❌ IMPORT ERRORS DETECTED!")
        print("🔧 Please run from TradeMind_AI root directory")
        return False
    return True

def main_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("🚀 TradeMind_AI - Professional Trading System")
    print("="*60)
    print("\n📊 Select Operation:\n")
    print("1. 📈 Run Auto Scheduler (Automated Trading)")
    print("2. 🎯 Get Strike Recommendations")
    print("3. 💰 Check Balance")
    print("4. 📊 View Portfolio")
    print("5. 🧠 Run Master Trader (All Features)")
    print("6. 🔧 Test Connections")
    print("7. 📊 Generate Excel Report")
    print("8. 🔢 Calculate Option Greeks (Live Data)")
    print("0. ❌ Exit")
    print("\n" + "="*60)
    
    choice = input("\nEnter your choice (0-8): ")
    return choice

def run_selection(choice):
    """Run selected module with improved error handling"""
    try:
        if choice == '1':
            print("\n🚀 Starting Auto Scheduler...")
            try:
                scheduler = HolidayAwareScheduler()
                scheduler.run()
            except Exception as e:
                print(f"❌ Auto Scheduler Error: {e}")
                print("💡 Check if scheduler module is properly configured")
            
        elif choice == '2':
            print("\n🎯 Getting Strike Recommendations...")
            try:
                recommender = ExactStrikeRecommender()
                recommender.generate_exact_recommendations()
            except Exception as e:
                print(f"❌ Strike Recommender Error: {e}")
                print("💡 Check Dhan API credentials and connection")
            
        elif choice == '3':
            print("\n💰 Checking Balance...")
            try:
                fetcher = DhanBalanceFetcher()
                fetcher.run_balance_check()
            except Exception as e:
                print(f"❌ Balance Fetcher Error: {e}")
                print("💡 Check Dhan API credentials")
            
        elif choice == '4':
            print("\n📊 Viewing Portfolio...")
            try:
                pm = PortfolioManager()
                pm.send_daily_portfolio_report()
            except Exception as e:
                print(f"❌ Portfolio Manager Error: {e}")
                print("💡 Check if portfolio data files exist")
            
        elif choice == '5':
            print("\n🧠 Running Master Trader...")
            try:
                master = EnhancedMasterTrader()
                master.run_enhanced_session(30)  # 30 minute session
            except Exception as e:
                print(f"❌ Master Trader Error: {e}")
                print("💡 Check all dependencies and API connections")
            
        elif choice == '6':
            print("\n🔧 Testing Connections...")
            try:
                # Test unified trading engine
                print("Testing Unified Trading Engine...")
                ai = UnifiedTradingEngine(TradingMode.PAPER)
                print("✅ Unified Trading Engine initialized")
                
                # Test balance
                balance = ai.get_account_balance()
                print(f"✅ Account Balance: {balance}")
                
                # Test market data
                print("Testing Market Data Engine...")
                market_engine = MarketDataEngine()
                print("✅ Market Data Engine initialized")
                
                print("🎉 All connections working!")
                
            except Exception as e:
                print(f"❌ Connection test error: {e}")
                print("💡 Check API keys in .env file")
            
        elif choice == '7':
            print("\n📊 Generating Excel Report...")
            try:
                # Create reports folder if it doesn't exist
                if not os.path.exists('reports'):
                    os.makedirs('reports')
                    print("📁 Created reports folder")
                    
                reporter = ExcelReporter()
                report_path = reporter.generate_daily_report()
                if report_path:
                    print(f"✅ Report saved at: {report_path}")
                    print(f"📁 Check the 'reports' folder in your TradeMind_AI directory")
                else:
                    print("❌ Failed to generate report")
                    
            except Exception as e:
                print(f"❌ Excel Reporter Error: {e}")
                print("💡 Check if openpyxl is installed: pip install openpyxl")
                
        elif choice == '8':
            print("\n🔢 Option Greeks Calculator with Live Data...")
            try:
                calc = GreeksCalculator()
                market_engine = MarketDataEngine()
                
                print("\n📊 Select Option for Greeks Calculation:")
                print("1. NIFTY Options")
                print("2. BANKNIFTY Options")
                
                symbol_choice = input("\nEnter choice (1 or 2): ")
                
                if symbol_choice == '1':
                    symbol = 'NIFTY'
                    symbol_id = market_engine.NIFTY_ID
                elif symbol_choice == '2':
                    symbol = 'BANKNIFTY'
                    symbol_id = market_engine.BANKNIFTY_ID
                else:
                    print("❌ Invalid choice, using NIFTY")
                    symbol = 'NIFTY'
                    symbol_id = market_engine.NIFTY_ID
                    
                print(f"\n📡 Fetching live {symbol} data...")
                
                # Fetch live option chain
                option_data = market_engine.get_option_chain(symbol_id, symbol)
                
                if option_data:
                    # Analyze the data
                    analysis = market_engine.analyze_option_data(option_data)
                    
                    if analysis:
                        spot_price = analysis['underlying_price']
                        atm_strike = analysis['atm_strike']
                        
                        print(f"\n📊 Live Market Data:")
                        print(f"   Current {symbol} Spot: ₹{spot_price}")
                        print(f"   ATM Strike: ₹{atm_strike}")
                        print(f"   Call Premium: ₹{analysis['call_price']}")
                        print(f"   Put Premium: ₹{analysis['put_price']}")
                        
                        # Get user's preferred strike
                        print(f"\n📌 Available strikes around ATM:")
                        strikes = []
                        strike_gap = 50 if symbol == 'NIFTY' else 100
                        
                        for i in range(-3, 4):
                            strike = atm_strike + (i * strike_gap)
                            strikes.append(strike)
                            print(f"   {i+4}. {strike} {'(ATM)' if i == 0 else ''}")
                        
                        try:
                            strike_choice = int(input("\nSelect strike (1-7): ")) - 1
                            selected_strike = strikes[strike_choice] if 0 <= strike_choice < 7 else atm_strike
                        except ValueError:
                            print("Invalid input, using ATM strike")
                            selected_strike = atm_strike
                        
                        option_type = input("Option type (CE/PE): ").upper()
                        if option_type not in ['CE', 'PE']:
                            option_type = 'CE'
                            print("Invalid input, using CE")
                        
                        # Get option price from chain (simplified - using ATM price)
                        option_price = analysis['call_price'] if option_type == 'CE' else analysis['put_price']
                        
                        # Get expiry
                        print("\nSelect expiry:")
                        print("1. Current Week")
                        print("2. Next Week")
                        expiry_choice = input("Enter choice (1 or 2): ")
                        days_to_expiry = 3 if expiry_choice == '1' else 10
                        
                        print("\n🔄 Calculating Greeks with live data...")
                        
                        # Calculate Greeks
                        greeks = calc.calculate_greeks(
                            spot_price=spot_price,
                            strike_price=selected_strike,
                            time_to_expiry=days_to_expiry,
                            volatility=analysis['call_iv']/100 if option_type == 'CE' else analysis['put_iv']/100,
                            option_type=option_type,
                            option_price=option_price
                        )
                        
                        # Calculate PCR from live data
                        pcr = calc.calculate_pcr(
                            call_oi=analysis['call_oi'],
                            put_oi=analysis['put_oi'],
                            call_volume=analysis['call_volume'],
                            put_volume=analysis['put_volume']
                        )
                        
                        # Calculate probability of profit
                        pop = calc.calculate_probability_of_profit(
                            spot_price=spot_price,
                            strike_price=selected_strike,
                            time_to_expiry=days_to_expiry,
                            volatility=greeks['implied_volatility']/100,
                            option_type=option_type,
                            option_price=option_price
                        )
                        
                        # Display results
                        calc.display_greeks(symbol, selected_strike, option_type, greeks, pcr, pop)
                        
                        # Additional live market insights
                        print(f"\n📊 Live Market Insights:")
                        print(f"   Total Call OI: {analysis['call_oi']:,}")
                        print(f"   Total Put OI: {analysis['put_oi']:,}")
                        print(f"   Call Volume: {analysis['call_volume']:,}")
                        print(f"   Put Volume: {analysis['put_volume']:,}")
                    else:
                        print("❌ Unable to analyze option data")
                        raise Exception("Analysis failed")
                        
                else:
                    print("❌ Unable to fetch live market data")
                    raise Exception("Data fetch failed")
                    
            except Exception as e:
                print(f"❌ Greeks Calculator Error: {e}")
                print("💡 Falling back to manual input mode...")
                
                try:
                    # Fallback to manual input
                    spot = float(input("Enter spot price manually: "))
                    strike = float(input("Enter strike price: "))
                    option_type = input("Enter option type (CE/PE): ").upper()
                    option_price = float(input("Enter option premium: "))
                    days_to_expiry = int(input("Enter days to expiry: "))
                    
                    calc = GreeksCalculator()
                    greeks = calc.calculate_greeks(spot, strike, days_to_expiry, 0.15, option_type, option_price)
                    calc.display_greeks(symbol, strike, option_type, greeks)
                    
                except Exception as manual_error:
                    print(f"❌ Manual input error: {manual_error}")
            
        elif choice == '0':
            print("\n👋 Goodbye! Happy Trading!")
            sys.exit(0)
            
        else:
            print("\n❌ Invalid choice! Please try again.")
            
    except ValueError as e:
        print(f"\n❌ Invalid input: Please enter numeric values where required")
        print(f"Error details: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        print("💡 Check if .env file contains all required API keys")

def main():
    """Main function with import checking"""
    try:
        # Check imports first
        if not check_imports():
            print("\n🔧 To fix import issues:")
            print("1. Make sure you're in the TradeMind_AI root directory")
            print("2. Run: cd /path/to/TradeMind_AI")
            print("3. Then run: python run_trading.py")
            return
        
        print("🌟 TradeMind_AI System Starting...")
        print("✅ All modules loaded successfully!")
        
        while True:
            choice = main_menu()
            run_selection(choice)
            
            if choice != '0':
                input("\n✅ Press Enter to continue...")
                
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Critical error in main: {e}")
        print("💡 Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()