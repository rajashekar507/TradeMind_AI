# TradeMind_AI Market Data Engine
# Fetches real NIFTY and BANKNIFTY data

from dhanhq import DhanContext, dhanhq
import time
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class MarketDataEngine:
    def __init__(self):
        # Initialize Dhan
        client_id = os.getenv('DHAN_CLIENT_ID')
        access_token = os.getenv('DHAN_ACCESS_TOKEN')
        dhan_context = DhanContext(client_id=client_id, access_token=access_token)
        self.dhan = dhanhq(dhan_context)
        
        # Market identifiers
        self.NIFTY_ID = 13
        self.BANKNIFTY_ID = 25
        self.IDX_SEGMENT = "IDX_I"
        
        print("📊 Market Data Engine initialized!")

    def get_option_chain(self, symbol_id, symbol_name):
        """Get real option chain data"""
        try:
            print(f"📡 Fetching {symbol_name} option chain...")
            
            # Get expiry list
            expiry_response = self.dhan.expiry_list(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT
            )
            
            if expiry_response.get("status") != "success":
                print(f"❌ Failed to get expiry list for {symbol_name}")
                return None
                
            expiry_list = expiry_response["data"]["data"]
            nearest_expiry = expiry_list[0]
            
            print(f"📅 Using expiry: {nearest_expiry}")
            
            # Wait for rate limiting
            time.sleep(3)
            
            # Get option chain
            option_chain = self.dhan.option_chain(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT,
                expiry=nearest_expiry
            )
            
            if option_chain.get("status") == "success":
                print(f"✅ {symbol_name} option chain fetched successfully!")
                return {
                    'symbol': symbol_name,
                    'expiry': nearest_expiry,
                    'data': option_chain["data"],
                    'timestamp': datetime.now()
                }
            else:
                print(f"❌ Failed to get option chain for {symbol_name}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching {symbol_name} data: {e}")
            return None

    def analyze_option_data(self, option_data):
        """Analyze option chain for trading opportunities"""
        if not option_data or 'data' not in option_data:
            return None
            
        try:
            data = option_data['data']
            
            # Extract key information
            if 'data' in data:
                inner_data = data['data']
                underlying_price = inner_data.get('last_price', 0)
                option_chain = inner_data.get('oc', {})
            else:
                underlying_price = data.get('last_price', 0)
                option_chain = data.get('oc', {})
            
            if not option_chain:
                print("⚠️ No option chain data found")
                return None
            
            # Find ATM (At The Money) options
            strikes = list(option_chain.keys())
            strikes_float = [float(strike) for strike in strikes]
            
            # Find closest strike to underlying price
            atm_strike = min(strikes_float, key=lambda x: abs(x - underlying_price))
            atm_strike_str = f"{atm_strike:.6f}"
            
            if atm_strike_str in option_chain:
                atm_data = option_chain[atm_strike_str]
                
                # Extract Call and Put data
                ce_data = atm_data.get('ce', {})
                pe_data = atm_data.get('pe', {})
                
                analysis = {
                    'symbol': option_data['symbol'],
                    'underlying_price': underlying_price,
                    'atm_strike': atm_strike,
                    'call_price': ce_data.get('last_price', 0),
                    'put_price': pe_data.get('last_price', 0),
                    'call_oi': ce_data.get('oi', 0),
                    'put_oi': pe_data.get('oi', 0),
                    'call_volume': ce_data.get('volume', 0),
                    'put_volume': pe_data.get('volume', 0),
                    'call_iv': ce_data.get('implied_volatility', 0),
                    'put_iv': pe_data.get('implied_volatility', 0),
                    'timestamp': datetime.now()
                }
                
                print(f"📊 {option_data['symbol']} Analysis:")
                print(f"   💰 Underlying: ₹{underlying_price}")
                print(f"   🎯 ATM Strike: ₹{atm_strike}")
                print(f"   📞 Call Price: ₹{analysis['call_price']}")
                print(f"   📞 Put Price: ₹{analysis['put_price']}")
                
                return analysis
            else:
                print(f"⚠️ ATM strike {atm_strike} not found in option chain")
                return None
                
        except Exception as e:
            print(f"❌ Error analyzing option data: {e}")
            return None

    def get_market_snapshot(self):
        """Get complete market snapshot"""
        print("\n🔄 Getting market snapshot...")
        
        # Get NIFTY data
        nifty_data = self.get_option_chain(self.NIFTY_ID, "NIFTY")
        nifty_analysis = self.analyze_option_data(nifty_data) if nifty_data else None
        
        # Get BANKNIFTY data
        banknifty_data = self.get_option_chain(self.BANKNIFTY_ID, "BANKNIFTY")
        banknifty_analysis = self.analyze_option_data(banknifty_data) if banknifty_data else None
        
        return {
            'nifty': nifty_analysis,
            'banknifty': banknifty_analysis,
            'timestamp': datetime.now()
        }

# Test the market data engine
if __name__ == "__main__":
    engine = MarketDataEngine()
    snapshot = engine.get_market_snapshot()
    
    print("\n📊 Market Snapshot Complete!")
    if snapshot['nifty']:
        print("✅ NIFTY data available")
    if snapshot['banknifty']:
        print("✅ BANKNIFTY data available")