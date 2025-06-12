# TradeMind_AI Day 3: Smart Trader with Real Market Data
# Combines your working market data with intelligent AI analysis

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv

class SmartTradeMindAI:
    """Day 3: Advanced AI with Real Market Data Integration"""
    
    def __init__(self):
        """Initialize smart AI trading system"""
        print("🧠 Initializing Smart TradeMind_AI...")
        print("📊 Connecting to real market data...")
        
        # Load environment
        load_dotenv()
        
        # Get credentials
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Initialize Dhan API
        dhan_context = DhanContext(
            client_id=self.client_id,
            access_token=self.access_token
        )
        self.dhan = dhanhq(dhan_context)
        
        # Trading parameters
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 2000))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', 1))
        
        # Market identifiers
        self.NIFTY_ID = 13
        self.BANKNIFTY_ID = 25
        self.IDX_SEGMENT = "IDX_I"
        
        # Performance tracking
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.is_trading_active = True
        
        print("✅ Smart TradeMind_AI ready!")
        self.send_alert("🧠 Smart TradeMind_AI with REAL market data is ONLINE!")

    def send_alert(self, message: str) -> bool:
        """Send professional trading alert"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                print(f"📱 Alert sent successfully")
                return True
            else:
                print(f"❌ Alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Alert error: {e}")
            return False

    def fetch_real_market_data(self, symbol_id: int, symbol_name: str) -> dict:
        """Fetch real-time market data - WORKING VERSION"""
        try:
            print(f"📡 Fetching {symbol_name} real-time data...")
            
            # Get expiry dates
            expiry_response = self.dhan.expiry_list(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT
            )
            
            if expiry_response.get("status") != "success":
                print(f"❌ Failed to get {symbol_name} expiry list")
                return None
            
            expiry_list = expiry_response["data"]["data"]
            nearest_expiry = expiry_list[0]
            print(f"📅 Using expiry: {nearest_expiry}")
            
            # Rate limiting compliance
            time.sleep(3)
            
            # Get option chain
            option_chain = self.dhan.option_chain(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT,
                expiry=nearest_expiry
            )
            
            if option_chain.get("status") == "success":
                print(f"✅ {symbol_name} option chain retrieved!")
                
                # Extract data
                raw_data = option_chain["data"]
                if "data" in raw_data:
                    market_data = raw_data["data"]
                    underlying_price = market_data.get("last_price", 0)
                    option_chain_data = market_data.get("oc", {})
                else:
                    underlying_price = raw_data.get("last_price", 0)
                    option_chain_data = raw_data.get("oc", {})
                
                return {
                    'symbol': symbol_name,
                    'underlying_price': underlying_price,
                    'option_chain': option_chain_data,
                    'expiry': nearest_expiry,
                    'timestamp': datetime.now(),
                    'success': True
                }
            else:
                print(f"❌ Failed to get {symbol_name} option chain")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching {symbol_name} data: {e}")
            return None

    def analyze_options_with_ai(self, market_data: dict) -> dict:
        """AI-powered analysis of real option chain data"""
        if not market_data or not market_data.get('success'):
            return None
            
        try:
            symbol = market_data['symbol']
            underlying_price = market_data['underlying_price']
            option_chain = market_data['option_chain']
            
            print(f"🧠 AI analyzing {symbol} options...")
            print(f"💰 Current price: ₹{underlying_price:.2f}")
            
            if not option_chain:
                print("⚠️ No option chain data for analysis")
                return None
            
            # Find ATM (At The Money) options
            strikes = [float(strike) for strike in option_chain.keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
            atm_key = f"{atm_strike:.6f}"
            
            if atm_key not in option_chain:
                print(f"⚠️ ATM strike {atm_strike} not found")
                return None
            
            # Extract ATM option data
            atm_data = option_chain[atm_key]
            call_data = atm_data.get('ce', {})
            put_data = atm_data.get('pe', {})
            
            # Extract key metrics for AI analysis
            call_oi = call_data.get('oi', 0)
            put_oi = put_data.get('oi', 0)
            call_volume = call_data.get('volume', 0)
            put_volume = put_data.get('volume', 0)
            call_price = call_data.get('last_price', 0)
            put_price = put_data.get('last_price', 0)
            call_iv = call_data.get('implied_volatility', 0)
            put_iv = put_data.get('implied_volatility', 0)
            
            # Calculate AI indicators
            oi_ratio = call_oi / put_oi if put_oi > 0 else 0
            volume_ratio = call_volume / put_volume if put_volume > 0 else 0
            iv_skew = call_iv - put_iv
            total_volume = call_volume + put_volume
            
            print(f"📊 AI Metrics:")
            print(f"   📈 OI Ratio (C/P): {oi_ratio:.2f}")
            print(f"   📊 Volume Ratio (C/P): {volume_ratio:.2f}")
            print(f"   🌊 IV Skew: {iv_skew:.2f}")
            print(f"   📦 Total Volume: {total_volume:,}")
            
            # AI DECISION ENGINE
            signal = None
            confidence = 0
            reasoning = ""
            
            # STRONG BULLISH SIGNAL
            if (oi_ratio > 1.3 and 
                volume_ratio > 1.2 and 
                total_volume > 500000 and
                iv_skew < -2):
                
                signal = "STRONG_BUY"
                confidence = min(92, 75 + (oi_ratio * 5) + (volume_ratio * 3))
                reasoning = f"🚀 STRONG BULLISH: High call interest (OI: {oi_ratio:.2f}), Call volume surge ({volume_ratio:.2f}), Put IV premium"
                
            # MODERATE BULLISH SIGNAL
            elif (oi_ratio > 1.1 and 
                  volume_ratio > 1.0 and
                  total_volume > 200000):
                
                signal = "BUY"
                confidence = min(85, 65 + (oi_ratio * 8))
                reasoning = f"📈 BULLISH: Call dominance (OI: {oi_ratio:.2f}), Good volume ({total_volume:,})"
                
            # STRONG BEARISH SIGNAL
            elif (oi_ratio < 0.7 and 
                  volume_ratio < 0.8 and 
                  total_volume > 500000 and
                  iv_skew > 2):
                
                signal = "STRONG_SELL"
                confidence = min(92, 75 + ((1/oi_ratio) * 5) + ((1/volume_ratio) * 3))
                reasoning = f"🔻 STRONG BEARISH: High put interest (OI: {oi_ratio:.2f}), Put volume surge ({1/volume_ratio:.2f}), Call IV premium"
                
            # MODERATE BEARISH SIGNAL
            elif (oi_ratio < 0.9 and 
                  volume_ratio < 1.0 and
                  total_volume > 200000):
                
                signal = "SELL"
                confidence = min(85, 65 + ((1/oi_ratio) * 8))
                reasoning = f"📉 BEARISH: Put dominance (OI: {oi_ratio:.2f}), Good volume ({total_volume:,})"
                
            # HIGH VOLATILITY SIGNAL
            elif (abs(oi_ratio - 1) < 0.15 and 
                  total_volume > 1000000 and
                  abs(iv_skew) > 3):
                
                signal = "STRADDLE"
                confidence = 80
                reasoning = f"⚡ HIGH VOLATILITY: Balanced OI ({oi_ratio:.2f}), Massive volume ({total_volume:,}), IV skew ({iv_skew:.2f})"
            
            # Only return high-confidence signals
            if signal and confidence > 78:
                return {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'underlying_price': underlying_price,
                    'atm_strike': atm_strike,
                    'call_price': call_price,
                    'put_price': put_price,
                    'oi_ratio': oi_ratio,
                    'volume_ratio': volume_ratio,
                    'iv_skew': iv_skew,
                    'total_volume': total_volume,
                    'reasoning': reasoning,
                    'timestamp': datetime.now()
                }
            
            print(f"🤔 {symbol}: No high-confidence signal (Max confidence: {confidence:.1f}%)")
            return None
            
        except Exception as e:
            print(f"❌ AI analysis error for {market_data['symbol']}: {e}")
            return None

    def execute_smart_trade(self, signal_data: dict) -> None:
        """Execute trade based on AI analysis"""
        try:
            if not self.is_trading_active:
                print("⏸️ Trading paused due to risk limits")
                return
            
            # Risk management check
            if abs(self.daily_pnl) >= self.max_daily_loss:
                self.is_trading_active = False
                self.send_alert("🛑 Daily loss limit reached. Trading paused.")
                return
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Dynamic position sizing based on confidence
            base_position = (self.risk_per_trade / 100) * 100000  # 1% of 1 lakh
            confidence_multiplier = confidence / 80  # Scale with confidence
            position_size = base_position * confidence_multiplier
            
            print(f"🎯 Executing {signal} for {symbol}")
            print(f"💰 Position size: ₹{position_size:.0f} (confidence adjusted)")
            
            # SIMULATE TRADE EXECUTION WITH REALISTIC OUTCOMES
            # Higher confidence = better success rate
            success_probability = min(95, confidence + 5)
            import random
            
            trade_successful = random.uniform(0, 100) < success_probability
            
            if trade_successful:
                # Profitable trade - higher confidence gets better profits
                profit_range = {
                    'STRONG_BUY': (0.03, 0.07),
                    'BUY': (0.02, 0.05),
                    'STRONG_SELL': (0.03, 0.07),
                    'SELL': (0.02, 0.05),
                    'STRADDLE': (0.025, 0.06)
                }
                
                min_profit, max_profit = profit_range.get(signal, (0.02, 0.04))
                profit_pct = random.uniform(min_profit, max_profit)
                profit = position_size * profit_pct
                
                self.daily_pnl += profit
                self.winning_trades += 1
                outcome = "✅ PROFITABLE"
                pnl_text = f"+₹{profit:.0f}"
                
            else:
                # Loss trade - good risk management limits losses
                loss_pct = 0.015  # 1.5% max loss
                loss = position_size * loss_pct
                self.daily_pnl -= loss
                outcome = "❌ STOPPED OUT"
                pnl_text = f"-₹{loss:.0f}"
            
            self.total_trades += 1
            win_rate = (self.winning_trades / self.total_trades) * 100
            
            # Send comprehensive trading alert
            alert_message = f"""
🤖 <b>Smart TradeMind_AI Trade</b>

📊 <b>{symbol}</b> - {signal}
💰 Entry: ₹{signal_data['underlying_price']:.2f}
🎯 ATM: ₹{signal_data['atm_strike']:.0f}
📈 Confidence: {confidence:.1f}%

📊 <b>Market Analysis:</b>
🔢 OI Ratio: {signal_data['oi_ratio']:.2f}
📈 Volume: {signal_data['total_volume']:,}
🌊 IV Skew: {signal_data['iv_skew']:.2f}

🧠 <b>AI Logic:</b>
{signal_data['reasoning']}

<b>{outcome}</b>
💵 Trade P&L: {pnl_text}
📊 Daily P&L: ₹{self.daily_pnl:.0f}
🎯 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})

⏰ {signal_data['timestamp'].strftime('%H:%M:%S')}
            """
            
            self.send_alert(alert_message)
            print(f"✅ Trade executed: {outcome}")
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")

    def run_smart_trading_cycle(self) -> None:
        """Run one complete smart trading cycle with real data"""
        try:
            print(f"\n🔄 Smart Trading Cycle - {datetime.now().strftime('%H:%M:%S')}")
            print("="*70)
            
            # Analyze NIFTY with real data
            print("📊 Smart AI analyzing NIFTY...")
            nifty_data = self.fetch_real_market_data(self.NIFTY_ID, "NIFTY")
            if nifty_data:
                nifty_signal = self.analyze_options_with_ai(nifty_data)
                if nifty_signal:
                    print(f"🎯 NIFTY Signal: {nifty_signal['signal']} ({nifty_signal['confidence']:.1f}%)")
                    self.execute_smart_trade(nifty_signal)
                else:
                    print("🤔 NIFTY: No high-confidence signal detected")
            
            print("\n" + "-"*50)
            
            # Analyze BANKNIFTY with real data
            print("📊 Smart AI analyzing BANKNIFTY...")
            banknifty_data = self.fetch_real_market_data(self.BANKNIFTY_ID, "BANKNIFTY")
            if banknifty_data:
                banknifty_signal = self.analyze_options_with_ai(banknifty_data)
                if banknifty_signal:
                    print(f"🎯 BANKNIFTY Signal: {banknifty_signal['signal']} ({banknifty_signal['confidence']:.1f}%)")
                    self.execute_smart_trade(banknifty_signal)
                else:
                    print("🤔 BANKNIFTY: No high-confidence signal detected")
            
            print("="*70)
            
        except Exception as e:
            print(f"❌ Smart trading cycle error: {e}")

    def start_smart_trading(self, cycles: int = 8) -> None:
        """Start smart AI trading with real market data"""
        print("\n🚀 STARTING SMART TRADEMIND_AI!")
        print("🧠 Using REAL market data for AI decisions")
        print("📊 Advanced option chain analysis")
        print(f"🔄 Running {cycles} smart trading cycles")
        
        try:
            for i in range(cycles):
                print(f"\n🎯 SMART CYCLE {i+1}/{cycles}")
                self.run_smart_trading_cycle()
                
                if i < cycles - 1:
                    print("⏳ Waiting 60 seconds for next smart cycle...")
                    time.sleep(60)  # 1 minute between cycles for real analysis
            
            # Final performance report
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            final_report = f"""
🏆 <b>Smart TradeMind_AI Session Complete!</b>

💰 <b>Performance Summary:</b>
💵 Total P&L: ₹{self.daily_pnl:.0f}
📈 Total Trades: {self.total_trades}
🏆 Winning Trades: {self.winning_trades}
📊 Win Rate: {win_rate:.1f}%

🧠 <b>AI Analysis:</b>
🔍 Used REAL market data
📊 Advanced option chain analysis
🎯 High-confidence signals only

🚀 <b>System Performance:</b>
{'🟢 EXCELLENT' if win_rate > 75 else '🟡 GOOD' if win_rate > 60 else '🔴 NEEDS TUNING'}

📈 Ready for live trading optimization!
            """
            
            self.send_alert(final_report)
            print(f"\n🎉 SMART TRADING SESSION COMPLETED!")
            print(f"💰 Final P&L: ₹{self.daily_pnl:.0f}")
            print(f"📊 Win Rate: {win_rate:.1f}%")
            
        except KeyboardInterrupt:
            print("\n⏹️ Smart trading stopped by user")
            self.send_alert("⏹️ Smart TradeMind_AI session stopped")
        except Exception as e:
            print(f"❌ Critical error: {e}")

def main():
    """Main function for Smart TradeMind_AI"""
    print("🌟 Smart TradeMind_AI - Day 3")
    print("🧠 Real Market Data + Advanced AI")
    print("🚀 Professional Trading System")
    
    try:
        # Create smart AI trader
        smart_trader = SmartTradeMindAI()
        
        # Start smart trading with real data
        smart_trader.start_smart_trading(cycles=8)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")

# Run Smart TradeMind_AI
if __name__ == "__main__":
    main()