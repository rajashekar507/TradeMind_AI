#!/usr/bin/env python3
from src.core.unified_trading_engine import UnifiedTradingEngine, TradingMode
"""
TradeMind_AI - Main Entry Point
Select what you want to run
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
    """Run selected module"""
    try:
        if choice == '1':
            print("\n🚀 Starting Auto Scheduler...")
            from utils.auto_scheduler import HolidayAwareScheduler
            scheduler = HolidayAwareScheduler()
            scheduler.run()
            
        elif choice == '2':
            print("\n🎯 Getting Strike Recommendations...")
            from analysis.exact_recommender import ExactStrikeRecommender
            recommender = ExactStrikeRecommender()
            recommender.generate_exact_recommendations()
            
        elif choice == '3':
            print("\n💰 Checking Balance...")
            from data.Balance_fetcher import DhanBalanceFetcher
            fetcher = DhanBalanceFetcher()
            fetcher.run_balance_check()
            
        elif choice == '4':
            print("\n📊 Viewing Portfolio...")
            from portfolio.portfolio_manager import PortfolioManager
            pm = PortfolioManager()
            pm.send_daily_portfolio_report()
            
        elif choice == '5':
            print("\n🧠 Running Master Trader...")
            from core.master_trader import EnhancedMasterTrader
            master = EnhancedMasterTrader()
            master.run_enhanced_session(cycles=2)
            
        elif choice == '6':
            print("\n🔧 Testing Connections...")
            # Removed ai_trader import - using unified_trading_engine
            ai = UnifiedTradingEngine(TradingMode.PAPER)
            # Test basic functionality
            print("Testing Dhan API connection...")
            try:
                if hasattr(ai, 'dhan_client') and ai.dhan_client:
                    print("✅ Dhan client initialized")
                else:
                    print("⚠️ Dhan client not initialized - check API keys")
            except Exception as e:
                print(f"❌ Connection test error: {e}")
            
        elif choice == '7':
            print("\n📊 Generating Excel Report...")
            # Create reports folder if it doesn't exist
            if not os.path.exists('reports'):
                os.makedirs('reports')
                print("📁 Created reports folder")
                
            from utils.excel_reporter import ExcelReporter
            reporter = ExcelReporter()
            report_path = reporter.generate_daily_report()
            if report_path:
                print(f"✅ Report saved at: {report_path}")
                print(f"📁 Check the 'reports' folder in your TradeMind_AI directory")
                
        elif choice == '8':
            print("\n🔢 Option Greeks Calculator with Live Data...")
            from analysis.greeks_calculator import GreeksCalculator
            from data.market_data import MarketDataEngine
            import time
            
            calc = GreeksCalculator()
            market_engine = MarketDataEngine()
            
            print("\n📊 Select Option for Greeks Calculation:")
            print("1. NIFTY Options")
            print("2. BANKNIFTY Options")
            
            symbol_choice = input("\nEnter choice (1 or 2): ")
            
            if symbol_choice == '1':
                symbol = 'NIFTY'
                symbol_id = market_engine.NIFTY_ID
            else:
                symbol = 'BANKNIFTY'
                symbol_id = market_engine.BANKNIFTY_ID
                
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
                    
                    strike_choice = int(input("\nSelect strike (1-7): ")) - 1
                    selected_strike = strikes[strike_choice] if 0 <= strike_choice < 7 else atm_strike
                    
                    option_type = input("Option type (CE/PE): ").upper()
                    if option_type not in ['CE', 'PE']:
                        option_type = 'CE'
                    
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
            else:
                print("❌ Unable to fetch live market data")
                print("💡 Falling back to manual input mode...")
                
                # Fallback to manual input
                spot = float(input("Enter spot price manually: "))
                strike = float(input("Enter strike price: "))
                option_type = input("Enter option type (CE/PE): ").upper()
                option_price = float(input("Enter option premium: "))
                days_to_expiry = int(input("Enter days to expiry: "))
                
                greeks = calc.calculate_greeks(spot, strike, days_to_expiry, 0.15, option_type, option_price)
                calc.display_greeks(symbol, strike, option_type, greeks)
            
        elif choice == '0':
            print("\n👋 Goodbye! Happy Trading!")
            sys.exit(0)
            
        else:
            print("\n❌ Invalid choice! Please try again.")
            
    except ValueError as e:
        print(f"\n❌ Invalid input: Please enter numeric values where required")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

def main():
    """Main function"""
    while True:
        choice = main_menu()
        run_selection(choice)
        
        if choice != '0':
            input("\n✅ Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")