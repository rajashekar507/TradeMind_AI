# TradeMind_AI: Exact Strike Recommendation System
# Enhanced with Movement Potential Analysis, Balance Tracking, and Probability Filtering

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv

class ExactStrikeRecommender:
    """AI that recommends EXACT strikes to trade"""
    
    def __init__(self):
        """Initialize the exact strike recommendation system"""
        print("🎯 Initializing Exact Strike Recommender...")
        print("📊 Will recommend specific CE/PE strikes to trade!")
        
        # Load environment
        load_dotenv()
        
        # Initialize Dhan API
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        dhan_context = DhanContext(
            client_id=self.client_id,
            access_token=self.access_token
        )
        self.dhan = dhanhq(dhan_context)
        
        # Market identifiers
        self.NIFTY_ID = 13
        self.BANKNIFTY_ID = 25
        self.IDX_SEGMENT = "IDX_I"
        
        # UPDATED LOT SIZES AS PER SEBI GUIDELINES (June 2025)
        self.lot_sizes = {
            'NIFTY': 75,      # Changed from 25 on Dec 26, 2024
            'BANKNIFTY': 30   # Changed from 15 on Dec 24, 2024
        }
        
        # Trading preferences for better recommendations
        self.trading_preferences = {
            'min_movement_score': 60,      # Minimum movement potential
            'min_total_score': 65,         # Minimum overall score
            'max_otm_distance': 0.10,      # Max 10% OTM
            'min_delta': 0.20,             # Minimum delta for decent movement
            'min_volume': 5000,            # Minimum volume for liquidity
            'min_oi': 50000,               # Minimum OI for liquidity
            'preferred_moneyness_range': (0.95, 1.05)  # 5% ITM to 5% OTM
        }
        
        # Initialize balance tracking
        self.current_balance = self.fetch_current_balance()
        
        print("✅ Exact Strike Recommender ready!")
        print(f"💰 Current Balance: ₹{self.current_balance:,.2f}")
        self.send_alert(f"🎯 TradeMind_AI Strike Recommender is ONLINE!\n💰 Balance: ₹{self.current_balance:,.2f}")

    def fetch_current_balance(self) -> float:
        """Fetch current account balance from Dhan"""
        try:
            funds = self.dhan.get_fund_limits()
            if funds and 'data' in funds:
                balance_data = funds['data']
                available = balance_data.get('availabelBalance', 0)  # Dhan API typo
                if available == 0:
                    available = balance_data.get('availableBalance', 0)
                return float(available)
            return float(os.getenv('TOTAL_CAPITAL', 100000))
        except Exception as e:
            print(f"⚠️ Could not fetch balance: {e}")
            return float(os.getenv('TOTAL_CAPITAL', 100000))

    def calculate_movement_potential(self, price: float, delta: float, gamma: float, 
                                   iv: float, volume: int, oi: int, 
                                   days_to_expiry: int, underlying_price: float,
                                   moneyness: float) -> dict:
        """Calculate movement potential of an option with probability weighting"""
        
        # 1. Delta-based movement (Most important)
        # Higher delta = better movement with underlying
        delta_score = abs(delta) * 100  # 0-100 scale
        
        # 2. Gamma acceleration potential
        # Higher gamma = faster delta change
        gamma_score = min(gamma * 1000, 100)  # Normalize to 0-100
        
        # 3. Liquidity score (Volume + OI)
        liquidity_score = 0
        if volume > 100000:
            liquidity_score += 50
        elif volume > 50000:
            liquidity_score += 30
        elif volume > 10000:
            liquidity_score += 20
        elif volume > 5000:
            liquidity_score += 10
            
        if oi > 1000000:
            liquidity_score += 50
        elif oi > 500000:
            liquidity_score += 30
        elif oi > 100000:
            liquidity_score += 20
        elif oi > 50000:
            liquidity_score += 10
            
        # 4. Time decay factor (Theta impact)
        time_score = 100
        if days_to_expiry < 3:
            time_score = 20  # Very high decay
        elif days_to_expiry < 7:
            time_score = 50  # High decay
        elif days_to_expiry < 15:
            time_score = 80  # Moderate decay
            
        # 5. Volatility opportunity
        iv_score = min(iv * 2, 100)  # Higher IV = more movement potential
        
        # 6. Price leverage factor
        # Lower priced options have higher percentage movement potential
        leverage_factor = (underlying_price * 0.01) / price if price > 0 else 0
        leverage_score = min(leverage_factor * 20, 100)
        
        # 7. PROBABILITY ADJUSTMENT (NEW)
        # Penalize far OTM options with low probability
        probability_score = 100
        if abs(1 - moneyness) > 0.10:  # More than 10% OTM
            probability_score = 30
        elif abs(1 - moneyness) > 0.07:  # More than 7% OTM
            probability_score = 50
        elif abs(1 - moneyness) > 0.05:  # More than 5% OTM
            probability_score = 70
        elif abs(1 - moneyness) > 0.03:  # More than 3% OTM
            probability_score = 85
        
        # 8. Calculate expected daily movement
        # Based on delta and underlying's expected movement (using IV)
        underlying_daily_move = underlying_price * (iv / 100) / np.sqrt(252)
        expected_option_move = underlying_daily_move * abs(delta)
        movement_percentage = (expected_option_move / price * 100) if price > 0 else 0
        
        # Adjust movement percentage by probability
        adjusted_movement_percentage = movement_percentage * (probability_score / 100)
        
        # Calculate composite movement score with probability weighting
        movement_score = (
            delta_score * 0.20 +           # 20% weight
            gamma_score * 0.10 +           # 10% weight
            liquidity_score * 0.20 +       # 20% weight
            time_score * 0.10 +            # 10% weight
            iv_score * 0.10 +              # 10% weight
            leverage_score * 0.10 +        # 10% weight
            probability_score * 0.20       # 20% weight - NEW
        )
        
        return {
            'movement_score': movement_score,
            'expected_daily_move': expected_option_move,
            'movement_percentage': movement_percentage,
            'adjusted_movement_percentage': adjusted_movement_percentage,
            'probability_score': probability_score,
            'delta_score': delta_score,
            'liquidity_score': liquidity_score,
            'leverage_score': leverage_score,
            'details': {
                'delta_contribution': delta_score * 0.20,
                'gamma_contribution': gamma_score * 0.10,
                'liquidity_contribution': liquidity_score * 0.20,
                'time_contribution': time_score * 0.10,
                'iv_contribution': iv_score * 0.10,
                'leverage_contribution': leverage_score * 0.10,
                'probability_contribution': probability_score * 0.20
            }
        }

    def _analyze_individual_option(self, symbol: str, strike: float, underlying: float, 
                                 option_data: dict, option_type: str, expiry: str) -> dict:
        """Analyze individual option for trading opportunity"""
        try:
            # Extract option metrics
            price = option_data.get('last_price', 0)
            oi = option_data.get('oi', 0)
            volume = option_data.get('volume', 0)
            iv = option_data.get('implied_volatility', 0)
            
            # Get Greeks
            greeks = option_data.get('greeks', {})
            delta = greeks.get('delta', 0)
            gamma = greeks.get('gamma', 0)
            theta = greeks.get('theta', 0)
            vega = greeks.get('vega', 0)
            
            # Apply minimum filters
            if price <= 0:
                return None
            
            if volume < self.trading_preferences['min_volume']:
                return None
                
            if oi < self.trading_preferences['min_oi']:
                return None
                
            if abs(delta) < self.trading_preferences['min_delta']:
                return None
            
            # Get lot size
            lot_size = self.lot_sizes.get(symbol, 75)
            
            # Calculate capital requirements
            capital_per_lot = price * lot_size
            max_affordable_lots = int(self.current_balance * 0.8 / capital_per_lot) if capital_per_lot > 0 else 0
            
            # Skip if we can't afford even 1 lot
            if max_affordable_lots < 1:
                return None
            
            # Calculate days to expiry
            days_to_expiry = self._calculate_days_to_expiry(expiry)
            
            # Calculate moneyness
            moneyness = strike / underlying
            
            # Skip far OTM options
            if option_type == 'CE' and moneyness > (1 + self.trading_preferences['max_otm_distance']):
                return None
            elif option_type == 'PE' and moneyness < (1 - self.trading_preferences['max_otm_distance']):
                return None
            
            # CALCULATE MOVEMENT POTENTIAL WITH PROBABILITY
            movement_analysis = self.calculate_movement_potential(
                price, delta, gamma, iv, volume, oi, days_to_expiry, underlying, moneyness
            )
            
            # Skip if movement potential is too low
            if movement_analysis['movement_score'] < self.trading_preferences['min_movement_score']:
                return None
            
            # ENHANCED SCORING WITH MOVEMENT POTENTIAL AND PROBABILITY
            score = 0
            reasons = []
            
            # 1. MOVEMENT POTENTIAL SCORING (35 points max)
            movement_points = (movement_analysis['movement_score'] / 100) * 35
            score += movement_points
            reasons.append(f"Movement potential: {movement_analysis['movement_score']:.1f}/100")
            reasons.append(f"Expected daily move: ₹{movement_analysis['expected_daily_move']:.2f} ({movement_analysis['adjusted_movement_percentage']:.1f}% risk-adjusted)")
            
            # 2. MONEYNESS AND PROBABILITY SCORING (30 points max)
            moneyness_score = 0
            min_pref, max_pref = self.trading_preferences['preferred_moneyness_range']
            
            if option_type == 'CE':
                if min_pref <= moneyness <= max_pref:  # Preferred range
                    moneyness_score = 30
                    reasons.append("Optimal strike selection (high probability)")
                elif 0.93 <= moneyness <= 1.07:  # Acceptable range
                    moneyness_score = 20
                    reasons.append("Good strike selection (decent probability)")
                elif 0.90 <= moneyness <= 1.10:  # Outer range
                    moneyness_score = 10
                    reasons.append("Acceptable strike (moderate probability)")
                else:
                    moneyness_score = 0
                    reasons.append("Far OTM (low probability)")
            else:  # PE
                if min_pref <= moneyness <= max_pref:  # Preferred range
                    moneyness_score = 30
                    reasons.append("Optimal strike selection (high probability)")
                elif 0.93 <= moneyness <= 1.07:  # Acceptable range
                    moneyness_score = 20
                    reasons.append("Good strike selection (decent probability)")
                elif 0.90 <= moneyness <= 1.10:  # Outer range
                    moneyness_score = 10
                    reasons.append("Acceptable strike (moderate probability)")
                else:
                    moneyness_score = 0
                    reasons.append("Far OTM (low probability)")
            
            score += moneyness_score
            
            # 3. GREEK-BASED SCORING (20 points max)
            greek_score = 0
            if abs(delta) >= 0.40:
                greek_score += 10
                reasons.append(f"Excellent delta ({delta:.3f})")
            elif abs(delta) >= 0.30:
                greek_score += 7
                reasons.append(f"Good delta ({delta:.3f})")
            elif abs(delta) >= 0.25:
                greek_score += 5
                reasons.append(f"Acceptable delta ({delta:.3f})")
                
            if gamma > 0.001:
                greek_score += 5
                reasons.append(f"Good gamma acceleration ({gamma:.4f})")
            elif gamma > 0.0005:
                greek_score += 3
                
            score += greek_score
            
            # 4. AFFORDABILITY AND POSITION SIZING (15 points max)
            if max_affordable_lots >= 5:
                score += 15
                reasons.append(f"Excellent affordability ({max_affordable_lots} lots)")
            elif max_affordable_lots >= 3:
                score += 10
                reasons.append(f"Good affordability ({max_affordable_lots} lots)")
            elif max_affordable_lots >= 2:
                score += 7
                reasons.append(f"Fair affordability ({max_affordable_lots} lots)")
            else:
                score += 3
                reasons.append(f"Limited affordability ({max_affordable_lots} lot only)")
            
            # Only recommend if score meets minimum threshold
            if score < self.trading_preferences['min_total_score']:
                return None
            
            # Calculate targets based on movement potential and probability
            expected_move = movement_analysis['expected_daily_move']
            probability_factor = movement_analysis['probability_score'] / 100
            
            # Adjust targets based on probability
            if movement_analysis['probability_score'] >= 80:  # High probability
                target1_multiplier = 1.5
                target2_multiplier = 2.5
                sl_multiplier = 0.80
            elif movement_analysis['probability_score'] >= 60:  # Medium probability
                target1_multiplier = 1.3
                target2_multiplier = 2.0
                sl_multiplier = 0.75
            else:  # Lower probability
                target1_multiplier = 1.2
                target2_multiplier = 1.8
                sl_multiplier = 0.70
            
            # Calculate realistic targets
            target1 = price + (expected_move * target1_multiplier * probability_factor)
            target2 = price + (expected_move * target2_multiplier * probability_factor)
            stop_loss = price * sl_multiplier
            
            return {
                'symbol': symbol,
                'strike': strike,
                'option_type': option_type,
                'direction': "BULLISH" if option_type == 'CE' else "BEARISH",
                'current_price': price,
                'target_price': round(target1, 2),
                'target2_price': round(target2, 2),
                'stop_loss': round(stop_loss, 2),
                'score': score,
                'movement_score': movement_analysis['movement_score'],
                'probability_score': movement_analysis['probability_score'],
                'expected_daily_move': movement_analysis['expected_daily_move'],
                'movement_percentage': movement_analysis['movement_percentage'],
                'adjusted_movement_percentage': movement_analysis['adjusted_movement_percentage'],
                'volume': volume,
                'oi': oi,
                'delta': delta,
                'gamma': gamma,
                'iv': iv,
                'days_to_expiry': days_to_expiry,
                'expiry': expiry,
                'reasons': reasons,
                'lot_size': lot_size,
                'capital_per_lot': capital_per_lot,
                'max_affordable_lots': max_affordable_lots,
                'movement_details': movement_analysis['details'],
                'moneyness': moneyness,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {option_type} {strike}: {e}")
            return None

    def _send_exact_recommendations(self, recommendations: list) -> None:
        """Send exact trading recommendations to Telegram"""
        
        # Update balance before recommendations
        self.current_balance = self.fetch_current_balance()
        
        for i, rec in enumerate(recommendations, 1):
            # Calculate position details
            recommended_lots = min(rec['max_affordable_lots'], 3)  # Max 3 lots for safety
            total_capital = rec['capital_per_lot'] * recommended_lots
            max_profit = (rec['target2_price'] - rec['current_price']) * recommended_lots * rec['lot_size']
            max_loss = (rec['current_price'] - rec['stop_loss']) * recommended_lots * rec['lot_size']
            
            alert_message = f"""
🎯 <b>EXACT TRADING RECOMMENDATION #{i}</b>

📊 <b>{rec['symbol']} {rec['strike']:.0f} {rec['option_type']}</b>
🎪 Direction: {rec['direction']}
💰 Current Price: ₹{rec['current_price']:.2f}
📍 Moneyness: {((1 - rec['moneyness']) * 100):.1f}% {'ITM' if rec['moneyness'] < 1 else 'OTM'} 

🚀 <b>MOVEMENT POTENTIAL:</b>
📈 Movement Score: {rec['movement_score']:.1f}/100
🎯 Probability Score: {rec['probability_score']:.0f}/100
📊 Expected Daily Move: ₹{rec['expected_daily_move']:.2f}
📈 Movement %: {rec['movement_percentage']:.1f}% (Raw)
📈 Risk-Adjusted %: {rec['adjusted_movement_percentage']:.1f}%
⚡ Delta: {rec['delta']:.3f} | Gamma: {rec['gamma']:.4f}

🎯 <b>TRADE PLAN:</b>
🟢 BUY: ₹{rec['current_price']:.2f}
🎯 TARGET 1: ₹{rec['target_price']:.2f} (+{((rec['target_price']/rec['current_price']-1)*100):.1f}%)
🎯 TARGET 2: ₹{rec['target2_price']:.2f} (+{((rec['target2_price']/rec['current_price']-1)*100):.1f}%)
🛡️ STOP LOSS: ₹{rec['stop_loss']:.2f} (-{((1-rec['stop_loss']/rec['current_price'])*100):.1f}%)
📅 EXPIRY: {rec['expiry']} ({rec['days_to_expiry']} days)

💰 <b>POSITION SIZING:</b>
📦 Lot Size: {rec['lot_size']} units
💵 Capital per Lot: ₹{rec['capital_per_lot']:,.2f}
📊 Recommended Lots: {recommended_lots} (of {rec['max_affordable_lots']} affordable)
💰 Total Capital: ₹{total_capital:,.2f}
📈 Max Profit: ₹{max_profit:,.2f}
📉 Max Loss: ₹{max_loss:,.2f}
💰 Current Balance: ₹{self.current_balance:,.2f}

📊 <b>QUALITY METRICS:</b>
🏆 Overall Score: {rec['score']:.0f}/100
📈 Volume: {rec['volume']:,}
🔢 OI: {rec['oi']:,}
📊 IV: {rec['iv']:.1f}%

🧠 <b>AI ANALYSIS:</b>
{chr(10).join(f"• {reason}" for reason in rec['reasons'])}

⏰ {rec['timestamp'].strftime('%H:%M:%S')}
            """
            
            self.send_alert(alert_message)
            print(f"\n🎯 RECOMMENDATION #{i}:")
            print(f"   📊 {rec['symbol']} {rec['strike']:.0f} {rec['option_type']}")
            print(f"   💰 Price: ₹{rec['current_price']:.2f}")
            print(f"   📍 Moneyness: {((1 - rec['moneyness']) * 100):.1f}% {'ITM' if rec['moneyness'] < 1 else 'OTM'}")
            print(f"   🚀 Movement Score: {rec['movement_score']:.1f}/100")
            print(f"   🎯 Probability Score: {rec['probability_score']:.0f}/100")
            print(f"   📈 Risk-Adjusted Move: {rec['adjusted_movement_percentage']:.1f}%")
            print(f"   💵 Capital/Lot: ₹{rec['capital_per_lot']:,.2f}")
            print(f"   📦 Recommended Lots: {recommended_lots}")
            
            time.sleep(2)

    def send_alert(self, message: str) -> bool:
        """Send alert to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            return response.status_code == 200
        except:
            return False

    def fetch_complete_option_chain(self, symbol_id: int, symbol_name: str) -> dict:
        """Fetch COMPLETE option chain with ALL strikes"""
        try:
            print(f"📡 Fetching COMPLETE {symbol_name} option chain...")
            
            # Get expiry dates
            expiry_response = self.dhan.expiry_list(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT
            )
            
            if expiry_response.get("status") != "success":
                return None
            
            expiry_list = expiry_response["data"]["data"]
            nearest_expiry = expiry_list[0]
            
            # Rate limiting
            time.sleep(3)
            
            # Get complete option chain
            option_chain = self.dhan.option_chain(
                under_security_id=symbol_id,
                under_exchange_segment=self.IDX_SEGMENT,
                expiry=nearest_expiry
            )
            
            if option_chain.get("status") == "success":
                raw_data = option_chain["data"]
                
                # Extract data
                if "data" in raw_data:
                    market_data = raw_data["data"]
                    underlying_price = market_data.get("last_price", 0)
                    option_chain_data = market_data.get("oc", {})
                else:
                    underlying_price = raw_data.get("last_price", 0)
                    option_chain_data = raw_data.get("oc", {})
                
                print(f"✅ {symbol_name} complete option chain fetched!")
                print(f"💰 Underlying: ₹{underlying_price:.2f}")
                print(f"📊 Total strikes available: {len(option_chain_data)}")
                
                return {
                    'symbol': symbol_name,
                    'underlying_price': underlying_price,
                    'option_chain': option_chain_data,
                    'expiry': nearest_expiry,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error fetching {symbol_name} option chain: {e}")
            return None

    def analyze_all_strikes_for_opportunities(self, market_data: dict) -> list:
        """Analyze ALL strikes and recommend BEST trades based on MOVEMENT POTENTIAL and PROBABILITY"""
        if not market_data:
            return []
            
        try:
            symbol = market_data['symbol']
            underlying_price = market_data['underlying_price']
            option_chain = market_data['option_chain']
            expiry = market_data['expiry']
            
            print(f"\n🧠 Analyzing ALL {symbol} strikes for QUALITY opportunities...")
            
            recommendations = []
            analyzed_count = 0
            filtered_count = 0
            
            # Analyze each strike
            for strike_str, strike_data in option_chain.items():
                strike_price = float(strike_str)
                
                # Quick moneyness filter - skip very far strikes
                moneyness = strike_price / underlying_price
                if moneyness < 0.85 or moneyness > 1.15:  # Skip beyond 15% range
                    continue
                
                analyzed_count += 1
                
                # Get call and put data
                call_data = strike_data.get('ce', {})
                put_data = strike_data.get('pe', {})
                
                # Skip if no data
                if not call_data and not put_data:
                    continue
                
                # Analyze CALL options
                if call_data:
                    call_analysis = self._analyze_individual_option(
                        symbol, strike_price, underlying_price, call_data, 'CE', expiry
                    )
                    if call_analysis:
                        recommendations.append(call_analysis)
                        filtered_count += 1
                
                # Analyze PUT options
                if put_data:
                    put_analysis = self._analyze_individual_option(
                        symbol, strike_price, underlying_price, put_data, 'PE', expiry
                    )
                    if put_analysis:
                        recommendations.append(put_analysis)
                        filtered_count += 1
            
            # Sort by overall score (which includes movement and probability)
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"📊 Analyzed {analyzed_count} strikes, found {filtered_count} quality opportunities")
            
            # Return top 3 recommendations
            return recommendations[:3]
            
        except Exception as e:
            print(f"❌ Error analyzing strikes: {e}")
            return []

    def _calculate_days_to_expiry(self, expiry_str: str) -> int:
        """Calculate days to expiry"""
        try:
            expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
            today = datetime.now()
            return (expiry_date - today).days
        except:
            return 7  # Default

    def generate_exact_recommendations(self) -> None:
        """Generate exact strike recommendations for both NIFTY and BANKNIFTY"""
        print("\n🚀 GENERATING HIGH-PROBABILITY TRADING RECOMMENDATIONS...")
        print("📊 Filtering for Quality: Movement + Probability + Affordability")
        print("="*80)
        
        # Update balance before analysis
        self.current_balance = self.fetch_current_balance()
        print(f"💰 Current Trading Balance: ₹{self.current_balance:,.2f}")
        
        # Display filtering criteria
        print(f"\n🎯 Quality Filters Active:")
        print(f"   • Minimum Movement Score: {self.trading_preferences['min_movement_score']}/100")
        print(f"   • Minimum Overall Score: {self.trading_preferences['min_total_score']}/100")
        print(f"   • Maximum OTM Distance: {self.trading_preferences['max_otm_distance']*100:.0f}%")
        print(f"   • Minimum Delta: {self.trading_preferences['min_delta']}")
        print(f"   • Minimum Volume: {self.trading_preferences['min_volume']:,}")
        print(f"   • Preferred Strike Range: {self.trading_preferences['preferred_moneyness_range'][0]*100:.0f}%-{self.trading_preferences['preferred_moneyness_range'][1]*100:.0f}%")
        
        all_recommendations = []
        
        # Analyze NIFTY
        print("\n📊 Analyzing NIFTY strikes...")
        nifty_data = self.fetch_complete_option_chain(self.NIFTY_ID, "NIFTY")
        if nifty_data:
            nifty_recs = self.analyze_all_strikes_for_opportunities(nifty_data)
            all_recommendations.extend(nifty_recs)
        
        # Analyze BANKNIFTY
        print("\n📊 Analyzing BANKNIFTY strikes...")
        banknifty_data = self.fetch_complete_option_chain(self.BANKNIFTY_ID, "BANKNIFTY")
        if banknifty_data:
            banknifty_recs = self.analyze_all_strikes_for_opportunities(banknifty_data)
            all_recommendations.extend(banknifty_recs)
        
        # Sort all recommendations by score
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Send top recommendations
        if all_recommendations:
            print(f"\n✅ Found {len(all_recommendations)} high-quality opportunities!")
            self._send_exact_recommendations(all_recommendations[:3])
        else:
            print("\n❌ No opportunities meet the quality criteria right now")
            print("💡 This could mean:")
            print("   • Market is in a low-volatility phase")
            print("   • Options are overpriced")
            print("   • Better to wait for clearer opportunities")
            self.send_alert("🤔 No high-quality trading opportunities found. Quality filters are protecting your capital! 🛡️")

def main():
    """Main function to run exact strike recommendations"""
    print("🌟 TradeMind_AI: Exact Strike Recommender V2.0")
    print("🎯 High-Probability Option Selection System")
    print("📊 Balances Movement Potential with Success Probability")
    print("📦 Using UPDATED lot sizes: NIFTY=75, BANKNIFTY=30")
    print("🛡️ Quality Filters: Protecting Your Capital")
    
    try:
        recommender = ExactStrikeRecommender()
        recommender.generate_exact_recommendations()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()