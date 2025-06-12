"""
TradeMind_AI: Greeks Calculator
Calculates and displays Option Greeks (Delta, Gamma, Theta, Vega, Rho)
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import math

class GreeksCalculator:
    def __init__(self):
        """Initialize Greeks Calculator"""
        print("🔢 Initializing Greeks Calculator...")
        self.risk_free_rate = 0.065  # 6.5% risk-free rate (Indian context)
        
    def calculate_greeks(self, spot_price, strike_price, time_to_expiry, volatility, option_type='CE', option_price=None):
        """
        Calculate all Greeks for an option
        
        Parameters:
        - spot_price: Current price of underlying
        - strike_price: Strike price of option
        - time_to_expiry: Time to expiry in days
        - volatility: Implied volatility (as decimal, e.g., 0.20 for 20%)
        - option_type: 'CE' for Call, 'PE' for Put
        - option_price: Current market price of option
        """
        
        # Convert time to years
        T = time_to_expiry / 365.0
        
        # Avoid division by zero
        if T <= 0:
            return {
                'delta': 0,
                'gamma': 0,
                'theta': 0,
                'vega': 0,
                'rho': 0,
                'theoretical_price': 0
            }
        
        # Calculate d1 and d2
        d1 = (np.log(spot_price / strike_price) + (self.risk_free_rate + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        # Greeks calculations
        if option_type == 'CE':
            # Call option Greeks
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) 
                    - self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)) / 365
            rho = strike_price * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2) / 100
            theoretical_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:
            # Put option Greeks
            delta = norm.cdf(d1) - 1
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(T)) 
                    + self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)) / 365
            rho = -strike_price * T * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) / 100
            theoretical_price = strike_price * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        # Common Greeks
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(T))
        vega = spot_price * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Calculate IV if option price is provided
        implied_volatility = volatility
        if option_price and option_price > 0:
            implied_volatility = self.calculate_implied_volatility(
                spot_price, strike_price, T, option_price, option_type
            )
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(theta, 2),
            'vega': round(vega, 2),
            'rho': round(rho, 2),
            'theoretical_price': round(theoretical_price, 2),
            'implied_volatility': round(implied_volatility * 100, 2)  # Convert to percentage
        }
    
    def calculate_implied_volatility(self, spot_price, strike_price, time_to_expiry, option_price, option_type):
        """Calculate implied volatility using Newton-Raphson method"""
        # Initial guess
        volatility = 0.20  # Start with 20%
        
        for i in range(100):  # Max iterations
            greeks = self.calculate_greeks(spot_price, strike_price, time_to_expiry * 365, volatility, option_type)
            
            price_diff = greeks['theoretical_price'] - option_price
            
            if abs(price_diff) < 0.01:  # Convergence threshold
                break
                
            # Newton-Raphson update
            vega = greeks['vega']
            if vega != 0:
                volatility = volatility - price_diff / (vega * 100)
                volatility = max(0.01, min(volatility, 3.0))  # Keep between 1% and 300%
        
        return volatility
    
    def calculate_pcr(self, call_oi, put_oi, call_volume=None, put_volume=None):
        """Calculate Put-Call Ratio"""
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        pcr_volume = put_volume / call_volume if call_volume and put_volume and call_volume > 0 else 0
        
        return {
            'pcr_oi': round(pcr_oi, 3),
            'pcr_volume': round(pcr_volume, 3),
            'interpretation': self._interpret_pcr(pcr_oi)
        }
    
    def _interpret_pcr(self, pcr):
        """Interpret PCR value"""
        if pcr > 1.3:
            return "Extremely Bullish (Contrarian)"
        elif pcr > 1.0:
            return "Bullish"
        elif pcr > 0.7:
            return "Neutral"
        elif pcr > 0.5:
            return "Bearish"
        else:
            return "Extremely Bearish (Contrarian)"
    
    def calculate_probability_of_profit(self, spot_price, strike_price, time_to_expiry, volatility, option_type, option_price):
        """Calculate probability of profit for an option"""
        # Calculate breakeven
        if option_type == 'CE':
            breakeven = strike_price + option_price
            # Probability that spot > breakeven at expiry
            prob = 1 - norm.cdf((np.log(breakeven / spot_price) - (self.risk_free_rate - 0.5 * volatility ** 2) * time_to_expiry / 365) 
                               / (volatility * np.sqrt(time_to_expiry / 365)))
        else:
            breakeven = strike_price - option_price
            # Probability that spot < breakeven at expiry
            prob = norm.cdf((np.log(breakeven / spot_price) - (self.risk_free_rate - 0.5 * volatility ** 2) * time_to_expiry / 365) 
                           / (volatility * np.sqrt(time_to_expiry / 365)))
        
        return {
            'breakeven': round(breakeven, 2),
            'probability_of_profit': round(prob * 100, 2)
        }
    
    def display_greeks(self, symbol, strike, option_type, greeks_data, pcr_data=None, pop_data=None):
        """Display Greeks in a formatted way"""
        print(f"\n{'='*60}")
        print(f"🔢 OPTION GREEKS - {symbol} {strike} {option_type}")
        print(f"{'='*60}")
        
        print(f"\n📊 Greeks Values:")
        print(f"   Delta (Δ): {greeks_data['delta']} - Price change per ₹1 move")
        print(f"   Gamma (Γ): {greeks_data['gamma']} - Delta change rate")
        print(f"   Theta (Θ): ₹{greeks_data['theta']} - Daily time decay")
        print(f"   Vega (ν): ₹{greeks_data['vega']} - IV sensitivity")
        print(f"   Rho (ρ): ₹{greeks_data['rho']} - Interest rate sensitivity")
        
        print(f"\n💰 Pricing:")
        print(f"   Theoretical Price: ₹{greeks_data['theoretical_price']}")
        print(f"   Implied Volatility: {greeks_data['implied_volatility']}%")
        
        if pcr_data:
            print(f"\n📈 Put-Call Ratio:")
            print(f"   PCR (OI): {pcr_data['pcr_oi']} - {pcr_data['interpretation']}")
            if pcr_data['pcr_volume'] > 0:
                print(f"   PCR (Volume): {pcr_data['pcr_volume']}")
        
        if pop_data:
            print(f"\n🎯 Probability Analysis:")
            print(f"   Breakeven: ₹{pop_data['breakeven']}")
            print(f"   Probability of Profit: {pop_data['probability_of_profit']}%")
        
        print(f"{'='*60}")

# Test the module
if __name__ == "__main__":
    calc = GreeksCalculator()
    
    # Example calculation
    greeks = calc.calculate_greeks(
        spot_price=25000,
        strike_price=25200,
        time_to_expiry=7,
        volatility=0.15,
        option_type='CE',
        option_price=150
    )
    
    pcr = calc.calculate_pcr(call_oi=100000, put_oi=150000, call_volume=5000, put_volume=7000)
    
    pop = calc.calculate_probability_of_profit(
        spot_price=25000,
        strike_price=25200,
        time_to_expiry=7,
        volatility=0.15,
        option_type='CE',
        option_price=150
    )
    
    calc.display_greeks('NIFTY', 25200, 'CE', greeks, pcr, pop)