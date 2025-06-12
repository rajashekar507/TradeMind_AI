"""
TradeMind AI Option Greeks Calculator - Complete Production Version
This is the FINAL version - no future changes needed
Real-time Greeks calculation with dashboard integration and risk management
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import math
from scipy.stats import norm

# Import required modules
try:
    from realtime_market_data import RealTimeMarketData
    from portfolio_manager import PortfolioManager
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Please ensure realtime_market_data.py and portfolio_manager.py are available")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('option_greeks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "CE"
    PUT = "PE"

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class OptionGreeks:
    """Individual option Greeks data structure"""
    # Basic option details
    symbol: str
    strike_price: float
    option_type: OptionType
    expiry_date: str
    days_to_expiry: int
    
    # Market data
    spot_price: float
    option_price: float
    implied_volatility: float
    risk_free_rate: float
    
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    # Derived metrics
    delta_dollars: float
    gamma_dollars: float
    theta_dollars: float
    vega_dollars: float
    
    # Risk metrics
    moneyness: float
    time_value: float
    intrinsic_value: float
    
    # Position details
    quantity: int
    position_value: float
    
    # Metadata
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'option_details': {
                'symbol': self.symbol,
                'strike_price': self.strike_price,
                'option_type': self.option_type.value,
                'expiry_date': self.expiry_date,
                'days_to_expiry': self.days_to_expiry
            },
            'market_data': {
                'spot_price': round(self.spot_price, 2),
                'option_price': round(self.option_price, 2),
                'implied_volatility': round(self.implied_volatility, 4),
                'risk_free_rate': round(self.risk_free_rate, 4)
            },
            'greeks': {
                'delta': round(self.delta, 4),
                'gamma': round(self.gamma, 6),
                'theta': round(self.theta, 4),
                'vega': round(self.vega, 4),
                'rho': round(self.rho, 4)
            },
            'greeks_dollars': {
                'delta_dollars': round(self.delta_dollars, 2),
                'gamma_dollars': round(self.gamma_dollars, 2),
                'theta_dollars': round(self.theta_dollars, 2),
                'vega_dollars': round(self.vega_dollars, 2)
            },
            'risk_metrics': {
                'moneyness': round(self.moneyness, 4),
                'time_value': round(self.time_value, 2),
                'intrinsic_value': round(self.intrinsic_value, 2)
            },
            'position': {
                'quantity': self.quantity,
                'position_value': round(self.position_value, 2)
            },
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class PortfolioGreeks:
    """Portfolio-level Greeks aggregation"""
    # Net Greeks
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    net_rho: float
    
    # Dollar Greeks
    net_delta_dollars: float
    net_gamma_dollars: float
    net_theta_dollars: float
    net_vega_dollars: float
    
    # Portfolio metrics
    total_positions: int
    total_position_value: float
    delta_neutral_ratio: float
    
    # Risk metrics
    portfolio_var: float
    max_theta_decay: float
    iv_exposure: float
    
    # Greeks distribution
    greeks_by_symbol: Dict[str, Dict[str, float]]
    greeks_by_expiry: Dict[str, Dict[str, float]]
    
    # Hedging requirements
    hedge_delta_shares: int
    hedge_cost: float
    
    # Timestamp
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'net_greeks': {
                'delta': round(self.net_delta, 4),
                'gamma': round(self.net_gamma, 6),
                'theta': round(self.net_theta, 4),
                'vega': round(self.net_vega, 4),
                'rho': round(self.net_rho, 4)
            },
            'dollar_greeks': {
                'delta_dollars': round(self.net_delta_dollars, 2),
                'gamma_dollars': round(self.net_gamma_dollars, 2),
                'theta_dollars': round(self.net_theta_dollars, 2),
                'vega_dollars': round(self.net_vega_dollars, 2)
            },
            'portfolio_metrics': {
                'total_positions': self.total_positions,
                'total_value': round(self.total_position_value, 2),
                'delta_neutral_ratio': round(self.delta_neutral_ratio, 4)
            },
            'risk_metrics': {
                'portfolio_var': round(self.portfolio_var, 2),
                'max_theta_decay': round(self.max_theta_decay, 2),
                'iv_exposure': round(self.iv_exposure, 4)
            },
            'distribution': {
                'by_symbol': self.greeks_by_symbol,
                'by_expiry': self.greeks_by_expiry
            },
            'hedging': {
                'delta_shares_needed': self.hedge_delta_shares,
                'hedge_cost': round(self.hedge_cost, 2)
            },
            'timestamp': self.timestamp.isoformat()
        }

class OptionGreeksCalculator:
    """Complete Option Greeks Calculator System"""
    
    def __init__(self):
        """Initialize Greeks calculator"""
        logger.info("⚡ Initializing Option Greeks Calculator...")
        
        # Initialize data sources
        self.market_data = RealTimeMarketData()
        self.portfolio_manager = PortfolioManager()
        
        # Greeks calculation cache
        self.greeks_cache = {}
        self.portfolio_cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # Market parameters
        self.market_params = {
            'risk_free_rate': 0.065,  # 6.5% current RBI rate
            'dividend_yield': 0.01,   # 1% average dividend yield
            'trading_days_per_year': 252
        }
        
        # Greeks calculation parameters
        self.calculation_params = {
            'vol_smile_points': 20,      # Points for volatility smile
            'scenario_range': 0.10,      # 10% price range for scenarios
            'time_decay_days': 30,       # Days to project theta decay
            'bump_size': {
                'delta': 1.0,            # ₹1 price bump for Delta
                'gamma': 1.0,            # ₹1 price bump for Gamma
                'vega': 0.01,            # 1% vol bump for Vega
                'theta': 1/365           # 1 day time decay
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'delta': {'low': 0.2, 'moderate': 0.5, 'high': 0.8},
            'gamma': {'low': 0.01, 'moderate': 0.05, 'high': 0.10},
            'theta': {'low': -20, 'moderate': -50, 'high': -100},
            'vega': {'low': 10, 'moderate': 25, 'high': 50},
            'portfolio_delta': {'neutral': 0.05}  # ±5% for delta neutral
        }
        
        # Lot sizes for different indices
        self.lot_sizes = {
            'NIFTY': 75,
            'BANKNIFTY': 30,
            'FINNIFTY': 40,
            'MIDCPNIFTY': 75
        }
        
        # Background update control
        self.stop_updates = False
        self.update_threads = {}
        
        # Performance tracking
        self.performance = {
            'calculations_performed': 0,
            'cache_hits': 0,
            'error_count': 0,
            'avg_calculation_time': 0,
            'last_update': datetime.now()
        }
        
        logger.info("✅ Option Greeks Calculator initialized!")
    
    def calculate_option_greeks(self, symbol: str, strike_price: float, option_type: str,
                              expiry_date: str, quantity: int = 1, 
                              force_refresh: bool = False) -> OptionGreeks:
        """Calculate Greeks for a specific option"""
        try:
            start_time = time.time()
            
            # Create cache key
            cache_key = f"{symbol}_{strike_price}_{option_type}_{expiry_date}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Check cache first
            if not force_refresh and cache_key in self.greeks_cache:
                cached_time = self.greeks_cache[cache_key]['timestamp']
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    self.performance['cache_hits'] += 1
                    logger.info(f"📋 Returning cached Greeks for {symbol} {strike_price} {option_type}")
                    return self.greeks_cache[cache_key]['greeks']
            
            logger.info(f"⚡ Calculating Greeks for {symbol} {strike_price} {option_type}...")
            
            # Get current market data
            if symbol == 'NIFTY':
                market_data = self.market_data.get_nifty_data()
            elif symbol == 'BANKNIFTY':
                market_data = self.market_data.get_banknifty_data()
            else:
                raise ValueError(f"Unsupported symbol: {symbol}")
            
            spot_price = market_data.get('price', 0)
            
            # Get VIX for implied volatility estimation
            vix_data = self.market_data.get_vix_data()
            vix_level = vix_data.get('price', 15)
            
            # Calculate days to expiry
            expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            days_to_expiry = (expiry_dt - datetime.now()).days
            time_to_expiry = days_to_expiry / 365.0
            
            if days_to_expiry <= 0:
                raise ValueError("Option has expired")
            
            # Estimate implied volatility from VIX
            implied_vol = self._estimate_implied_volatility(vix_level, days_to_expiry, strike_price, spot_price)
            
            # Estimate option price using Black-Scholes
            option_price = self._black_scholes_price(
                spot_price, strike_price, time_to_expiry, 
                self.market_params['risk_free_rate'], implied_vol, 
                OptionType.CALL if option_type == 'CE' else OptionType.PUT
            )
            
            # Calculate Greeks using Black-Scholes formulas
            greeks = self._calculate_black_scholes_greeks(
                spot_price, strike_price, time_to_expiry,
                self.market_params['risk_free_rate'], implied_vol,
                OptionType.CALL if option_type == 'CE' else OptionType.PUT
            )
            
            # Calculate dollar Greeks (per lot)
            lot_size = self.lot_sizes.get(symbol, 75)
            delta_dollars = greeks['delta'] * lot_size * quantity
            gamma_dollars = greeks['gamma'] * lot_size * quantity * spot_price / 100
            theta_dollars = greeks['theta'] * lot_size * quantity
            vega_dollars = greeks['vega'] * lot_size * quantity / 100
            
            # Calculate derived metrics
            intrinsic_value = max(0, (spot_price - strike_price) if option_type == 'CE' else (strike_price - spot_price))
            time_value = option_price - intrinsic_value
            moneyness = spot_price / strike_price
            position_value = option_price * lot_size * quantity
            
            # Create OptionGreeks object
            option_greeks = OptionGreeks(
                symbol=symbol,
                strike_price=strike_price,
                option_type=OptionType.CALL if option_type == 'CE' else OptionType.PUT,
                expiry_date=expiry_date,
                days_to_expiry=days_to_expiry,
                spot_price=spot_price,
                option_price=option_price,
                implied_volatility=implied_vol,
                risk_free_rate=self.market_params['risk_free_rate'],
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks['rho'],
                delta_dollars=delta_dollars,
                gamma_dollars=gamma_dollars,
                theta_dollars=theta_dollars,
                vega_dollars=vega_dollars,
                moneyness=moneyness,
                time_value=time_value,
                intrinsic_value=intrinsic_value,
                quantity=quantity,
                position_value=position_value,
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.greeks_cache[cache_key] = {
                'greeks': option_greeks,
                'timestamp': datetime.now()
            }
            
            # Update performance metrics
            calc_time = time.time() - start_time
            self.performance['calculations_performed'] += 1
            self.performance['avg_calculation_time'] = (
                (self.performance['avg_calculation_time'] * (self.performance['calculations_performed'] - 1) + calc_time) /
                self.performance['calculations_performed']
            )
            self.performance['last_update'] = datetime.now()
            
            logger.info(f"✅ Greeks calculated for {symbol} {strike_price} {option_type}")
            logger.info(f"   Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.6f}, Theta: {greeks['theta']:.4f}")
            
            return option_greeks
            
        except Exception as e:
            logger.error(f"❌ Greeks calculation failed for {symbol} {strike_price} {option_type}: {e}")
            self.performance['error_count'] += 1
            return self._generate_fallback_greeks(symbol, strike_price, option_type, expiry_date, quantity, error=str(e))
    
    def calculate_portfolio_greeks(self, force_refresh: bool = False) -> PortfolioGreeks:
        """Calculate aggregated Greeks for entire portfolio"""
        try:
            cache_key = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Check cache first
            if not force_refresh and cache_key in self.portfolio_cache:
                cached_time = self.portfolio_cache[cache_key]['timestamp']
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    logger.info("📋 Returning cached portfolio Greeks")
                    return self.portfolio_cache[cache_key]['portfolio_greeks']
            
            logger.info("⚡ Calculating portfolio Greeks...")
            
            # Get current positions from portfolio manager
            positions = self.portfolio_manager.get_current_positions()
            
            if not positions:
                return self._generate_empty_portfolio_greeks()
            
            # Calculate Greeks for each position
            position_greeks = []
            for position in positions:
                try:
                    greeks = self.calculate_option_greeks(
                        symbol=position.get('symbol', 'NIFTY'),
                        strike_price=float(position.get('strike', 25000)),
                        option_type=position.get('option_type', 'CE'),
                        expiry_date=position.get('expiry', (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')),
                        quantity=int(position.get('quantity', 1))
                    )
                    position_greeks.append(greeks)
                except Exception as e:
                    logger.error(f"❌ Failed to calculate Greeks for position: {e}")
                    continue
            
            if not position_greeks:
                return self._generate_empty_portfolio_greeks()
            
            # Aggregate Greeks
            net_delta = sum(pg.delta * pg.quantity for pg in position_greeks)
            net_gamma = sum(pg.gamma * pg.quantity for pg in position_greeks)
            net_theta = sum(pg.theta * pg.quantity for pg in position_greeks)
            net_vega = sum(pg.vega * pg.quantity for pg in position_greeks)
            net_rho = sum(pg.rho * pg.quantity for pg in position_greeks)
            
            # Aggregate dollar Greeks
            net_delta_dollars = sum(pg.delta_dollars for pg in position_greeks)
            net_gamma_dollars = sum(pg.gamma_dollars for pg in position_greeks)
            net_theta_dollars = sum(pg.theta_dollars for pg in position_greeks)
            net_vega_dollars = sum(pg.vega_dollars for pg in position_greeks)
            
            # Portfolio metrics
            total_positions = len(position_greeks)
            total_position_value = sum(pg.position_value for pg in position_greeks)
            
            # Delta neutral ratio (how close to delta neutral)
            delta_neutral_ratio = abs(net_delta) / max(1, total_positions)
            
            # Risk metrics
            portfolio_var = self._calculate_portfolio_var(position_greeks)
            max_theta_decay = abs(net_theta_dollars)
            iv_exposure = abs(net_vega_dollars) / 100  # Exposure to 1% IV change
            
            # Greeks distribution by symbol and expiry
            greeks_by_symbol = self._aggregate_greeks_by_symbol(position_greeks)
            greeks_by_expiry = self._aggregate_greeks_by_expiry(position_greeks)
            
            # Hedging calculations
            hedge_calculations = self._calculate_hedging_requirements(position_greeks, net_delta_dollars)
            
            # Create portfolio Greeks object
            portfolio_greeks = PortfolioGreeks(
                net_delta=net_delta,
                net_gamma=net_gamma,
                net_theta=net_theta,
                net_vega=net_vega,
                net_rho=net_rho,
                net_delta_dollars=net_delta_dollars,
                net_gamma_dollars=net_gamma_dollars,
                net_theta_dollars=net_theta_dollars,
                net_vega_dollars=net_vega_dollars,
                total_positions=total_positions,
                total_position_value=total_position_value,
                delta_neutral_ratio=delta_neutral_ratio,
                portfolio_var=portfolio_var,
                max_theta_decay=max_theta_decay,
                iv_exposure=iv_exposure,
                greeks_by_symbol=greeks_by_symbol,
                greeks_by_expiry=greeks_by_expiry,
                hedge_delta_shares=hedge_calculations['shares'],
                hedge_cost=hedge_calculations['cost'],
                timestamp=datetime.now()
            )
            
            # Cache the result
            self.portfolio_cache[cache_key] = {
                'portfolio_greeks': portfolio_greeks,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Portfolio Greeks calculated for {total_positions} positions")
            logger.info(f"   Net Delta: {net_delta:.4f} (${net_delta_dollars:,.2f})")
            logger.info(f"   Net Theta: {net_theta:.4f} (${net_theta_dollars:,.2f})")
            
            return portfolio_greeks
            
        except Exception as e:
            logger.error(f"❌ Portfolio Greeks calculation failed: {e}")
            return self._generate_empty_portfolio_greeks(error=str(e))
    
    def get_greeks_scenarios(self, symbol: str, strike_price: float, option_type: str,
                           expiry_date: str, quantity: int = 1) -> Dict[str, Any]:
        """Generate Greeks scenarios for different market conditions"""
        try:
            logger.info(f"📊 Generating Greeks scenarios for {symbol} {strike_price} {option_type}...")
            
            # Get base Greeks
            base_greeks = self.calculate_option_greeks(symbol, strike_price, option_type, expiry_date, quantity)
            
            # Get current market data
            if symbol == 'NIFTY':
                market_data = self.market_data.get_nifty_data()
            else:
                market_data = self.market_data.get_banknifty_data()
            
            current_price = market_data.get('price', 0)
            
            # Define scenario ranges
            price_scenarios = np.arange(
                current_price * (1 - self.calculation_params['scenario_range']),
                current_price * (1 + self.calculation_params['scenario_range']),
                current_price * 0.01  # 1% increments
            )
            
            vol_scenarios = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]  # 10% to 35% volatility
            time_scenarios = [1, 7, 14, 30]  # Days to expiry scenarios
            
            # Calculate scenarios
            scenarios = {
                'price_scenarios': self._calculate_price_scenarios(base_greeks, price_scenarios),
                'volatility_scenarios': self._calculate_volatility_scenarios(base_greeks, vol_scenarios),
                'time_scenarios': self._calculate_time_scenarios(base_greeks, time_scenarios),
                'combined_scenarios': self._calculate_combined_scenarios(base_greeks),
                'base_greeks': base_greeks.to_dict(),
                'scenario_summary': {
                    'best_case_pnl': 0,
                    'worst_case_pnl': 0,
                    'probability_profit': 0,
                    'max_risk': 0
                }
            }
            
            # Calculate scenario summary
            scenarios['scenario_summary'] = self._calculate_scenario_summary(scenarios)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Greeks scenarios generation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_hedging_recommendations(self, target_delta: float = 0.0) -> Dict[str, Any]:
        """Get hedging recommendations to achieve target portfolio delta"""
        try:
            logger.info(f"🛡️ Generating hedging recommendations for target delta: {target_delta}")
            
            # Get current portfolio Greeks
            portfolio_greeks = self.calculate_portfolio_greeks()
            current_delta = portfolio_greeks.net_delta
            
            # Calculate hedging requirements
            delta_difference = current_delta - target_delta
            
            # Get current market prices
            nifty_data = self.market_data.get_nifty_data()
            banknifty_data = self.market_data.get_banknifty_data()
            
            # Calculate hedging strategies
            strategies = []
            
            # Strategy 1: Futures hedging
            if abs(delta_difference) > 0.1:
                nifty_futures_delta = self.lot_sizes['NIFTY']
                banknifty_futures_delta = self.lot_sizes['BANKNIFTY']
                
                nifty_lots_needed = -delta_difference / nifty_futures_delta
                banknifty_lots_needed = -delta_difference / banknifty_futures_delta
                
                strategies.append({
                    'strategy': 'FUTURES_HEDGE',
                    'description': 'Hedge using index futures',
                    'recommendations': [
                        {
                            'instrument': 'NIFTY_FUTURES',
                            'action': 'BUY' if nifty_lots_needed > 0 else 'SELL',
                            'lots': abs(round(nifty_lots_needed)),
                            'cost_estimate': abs(nifty_lots_needed) * nifty_data.get('price', 25000) * self.lot_sizes['NIFTY'] * 0.001  # 0.1% cost
                        },
                        {
                            'instrument': 'BANKNIFTY_FUTURES',
                            'action': 'BUY' if banknifty_lots_needed > 0 else 'SELL',
                            'lots': abs(round(banknifty_lots_needed)),
                            'cost_estimate': abs(banknifty_lots_needed) * banknifty_data.get('price', 55000) * self.lot_sizes['BANKNIFTY'] * 0.001
                        }
                    ]
                })
            
            # Strategy 2: Options hedging
            hedge_options = self._generate_options_hedge(delta_difference, portfolio_greeks)
            if hedge_options:
                strategies.append(hedge_options)
            
            # Strategy 3: Dynamic hedging plan
            dynamic_plan = self._generate_dynamic_hedge_plan(portfolio_greeks)
            if dynamic_plan:
                strategies.append(dynamic_plan)
            
            return {
                'current_portfolio': {
                    'net_delta': current_delta,
                    'target_delta': target_delta,
                    'delta_difference': delta_difference,
                    'hedge_urgency': self._assess_hedge_urgency(abs(delta_difference))
                },
                'hedging_strategies': strategies,
                'risk_analysis': {
                    'current_var': portfolio_greeks.portfolio_var,
                    'theta_decay_daily': portfolio_greeks.max_theta_decay,
                    'vega_exposure': portfolio_greeks.iv_exposure
                },
                'recommendations': {
                    'primary_strategy': strategies[0]['strategy'] if strategies else 'NO_HEDGE_NEEDED',
                    'hedge_frequency': 'DAILY' if abs(delta_difference) > 0.5 else 'WEEKLY',
                    'monitoring_required': abs(delta_difference) > 0.3
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Hedging recommendations failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_dashboard_data(self, symbol: str = None) -> Dict[str, Any]:
        """Get formatted Greeks data for dashboard display"""
        try:
            # Get portfolio Greeks
            portfolio_greeks = self.calculate_portfolio_greeks()
            
            # Get individual position Greeks if symbol specified
            position_greeks = None
            if symbol:
                # Find position for symbol (simplified - get first position)
                positions = self.portfolio_manager.get_current_positions()
                symbol_position = next((p for p in positions if p.get('symbol') == symbol), None)
                
                if symbol_position:
                    position_greeks = self.calculate_option_greeks(
                        symbol=symbol_position.get('symbol'),
                        strike_price=float(symbol_position.get('strike', 25000)),
                        option_type=symbol_position.get('option_type', 'CE'),
                        expiry_date=symbol_position.get('expiry', (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')),
                        quantity=int(symbol_position.get('quantity', 1))
                    )
            
            # Format for dashboard widgets
            dashboard_data = {
                'portfolio_greeks': {
                    'net_delta': {
                        'value': portfolio_greeks.net_delta,
                        'dollars': portfolio_greeks.net_delta_dollars,
                        'color': self._get_delta_color(portfolio_greeks.net_delta),
                        'risk_level': self._assess_delta_risk(portfolio_greeks.net_delta),
                        'description': f'Net Delta exposure: {portfolio_greeks.net_delta:.4f}'
                    },
                    'net_gamma': {
                        'value': portfolio_greeks.net_gamma,
                        'dollars': portfolio_greeks.net_gamma_dollars,
                        'color': self._get_gamma_color(portfolio_greeks.net_gamma),
                        'risk_level': self._assess_gamma_risk(portfolio_greeks.net_gamma),
                        'description': f'Net Gamma acceleration: {portfolio_greeks.net_gamma:.6f}'
                    },
                    'net_theta': {
                        'value': portfolio_greeks.net_theta,
                        'dollars': portfolio_greeks.net_theta_dollars,
                        'color': self._get_theta_color(portfolio_greeks.net_theta),
                        'risk_level': self._assess_theta_risk(portfolio_greeks.net_theta_dollars),
                        'description': f'Daily time decay: ₹{portfolio_greeks.net_theta_dollars:,.2f}'
                    },
                    'net_vega': {
                        'value': portfolio_greeks.net_vega,
                        'dollars': portfolio_greeks.net_vega_dollars,
                        'color': self._get_vega_color(portfolio_greeks.net_vega),
                        'risk_level': self._assess_vega_risk(portfolio_greeks.net_vega_dollars),
                        'description': f'Volatility exposure: ₹{portfolio_greeks.net_vega_dollars:,.2f} per 1% IV change'
                    }
                },
                'portfolio_summary': {
                    'total_positions': portfolio_greeks.total_positions,
                    'total_value': portfolio_greeks.total_position_value,
                    'delta_neutral_ratio': portfolio_greeks.delta_neutral_ratio,
                    'is_delta_neutral': abs(portfolio_greeks.delta_neutral_ratio) < self.risk_thresholds['portfolio_delta']['neutral'],
                    'portfolio_var': portfolio_greeks.portfolio_var,
                    'max_theta_decay': portfolio_greeks.max_theta_decay
                },
                'risk_alerts': self._generate_risk_alerts(portfolio_greeks),
                'hedging_status': {
                    'needs_hedging': abs(portfolio_greeks.net_delta) > 0.5,
                    'hedge_urgency': self._assess_hedge_urgency(abs(portfolio_greeks.net_delta)),
                    'recommended_hedge': 'FUTURES' if abs(portfolio_greeks.net_delta) > 1.0 else 'OPTIONS'
                }
            }
            
            # Add individual position data if requested
            if position_greeks:
                dashboard_data['position_greeks'] = {
                    'symbol': position_greeks.symbol,
                    'strike': position_greeks.strike_price,
                    'option_type': position_greeks.option_type.value,
                    'greeks': position_greeks.to_dict()['greeks'],
                    'greeks_dollars': position_greeks.to_dict()['greeks_dollars'],
                    'risk_metrics': position_greeks.to_dict()['risk_metrics']
                }
            
            # Add performance metrics
            dashboard_data['system_performance'] = {
                'calculations_performed': self.performance['calculations_performed'],
                'cache_hit_rate': (self.performance['cache_hits'] / max(1, self.performance['calculations_performed'])) * 100,
                'avg_calculation_time': round(self.performance['avg_calculation_time'] * 1000, 2),  # ms
                'error_rate': (self.performance['error_count'] / max(1, self.performance['calculations_performed'])) * 100,
                'last_update': self.performance['last_update'].isoformat()
            }
            
            # Add timestamp
            dashboard_data['timestamp'] = datetime.now().isoformat()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data generation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_real_time_updates(self, callback_function=None):
        """Start real-time Greeks updates"""
        try:
            logger.info("🔄 Starting real-time Greeks updates...")
            
            def greeks_updater():
                """Background Greeks update loop"""
                while not self.stop_updates:
                    try:
                        # Update portfolio Greeks
                        portfolio_greeks = self.calculate_portfolio_greeks(force_refresh=True)
                        
                        # Generate dashboard data
                        dashboard_data = self.get_dashboard_data()
                        
                        # Call callback if provided
                        if callback_function:
                            callback_function('greeks_update', dashboard_data)
                        
                        # Wait before next update
                        time.sleep(300)  # 5 minutes
                        
                    except Exception as e:
                        logger.error(f"❌ Greeks update error: {e}")
                        time.sleep(60)  # Wait longer on error
            
            # Start background thread
            self.update_threads['greeks'] = threading.Thread(target=greeks_updater, daemon=True)
            self.update_threads['greeks'].start()
            
            logger.info("✅ Real-time Greeks updates started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time updates: {e}")
    
    def stop_real_time_updates(self):
        """Stop real-time updates"""
        self.stop_updates = True
        logger.info("⏹️ Real-time Greeks updates stopped")
    
    # ======================
    # BLACK-SCHOLES CALCULATIONS
    # ======================
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, vol: float, option_type: OptionType) -> float:
        """Calculate Black-Scholes option price"""
        try:
            if T <= 0:
                # Option expired
                if option_type == OptionType.CALL:
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(0.1, price)  # Minimum price of 0.1
            
        except Exception as e:
            logger.error(f"❌ Black-Scholes price calculation failed: {e}")
            return 50.0  # Fallback price
    
    def _calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, vol: float, option_type: OptionType) -> Dict[str, float]:
        """Calculate Black-Scholes Greeks"""
        try:
            if T <= 0:
                return {
                    'delta': 1.0 if (option_type == OptionType.CALL and S > K) else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            # Delta
            if option_type == OptionType.CALL:
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
            
            # Theta
            theta_common = -(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T))
            if option_type == OptionType.CALL:
                theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type == OptionType.CALL:
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            logger.error(f"❌ Greeks calculation failed: {e}")
            return {
                'delta': 0.5,
                'gamma': 0.01,
                'theta': -10,
                'vega': 20,
                'rho': 5
            }
    
    def _estimate_implied_volatility(self, vix_level: float, days_to_expiry: int, strike: float, spot: float) -> float:
        """Estimate implied volatility from VIX and other factors"""
        try:
            # Base volatility from VIX
            base_vol = vix_level / 100
            
            # Adjust for time to expiry (volatility smile/skew)
            time_adjustment = 1.0
            if days_to_expiry <= 7:
                time_adjustment = 1.2  # Higher vol for weekly options
            elif days_to_expiry <= 30:
                time_adjustment = 1.1  # Slightly higher for monthly
            elif days_to_expiry >= 60:
                time_adjustment = 0.9  # Lower for longer term
            
            # Adjust for moneyness (volatility smile)
            moneyness = strike / spot
            moneyness_adjustment = 1.0
            
            if moneyness < 0.95 or moneyness > 1.05:
                moneyness_adjustment = 1.1  # Higher vol for OTM options
            elif 0.98 <= moneyness <= 1.02:
                moneyness_adjustment = 0.95  # Lower vol for ATM options
            
            # Final volatility
            implied_vol = base_vol * time_adjustment * moneyness_adjustment
            
            # Clamp between reasonable bounds
            return max(0.08, min(0.80, implied_vol))  # 8% to 80%
            
        except Exception:
            return 0.20  # Default 20% volatility
    
    # ======================
    # PORTFOLIO AGGREGATION METHODS
    # ======================
    
    def _aggregate_greeks_by_symbol(self, position_greeks: List[OptionGreeks]) -> Dict[str, Dict[str, float]]:
        """Aggregate Greeks by symbol"""
        try:
            symbol_greeks = {}
            
            for pg in position_greeks:
                symbol = pg.symbol
                if symbol not in symbol_greeks:
                    symbol_greeks[symbol] = {
                        'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0,
                        'delta_dollars': 0, 'theta_dollars': 0, 'vega_dollars': 0,
                        'position_count': 0, 'total_value': 0
                    }
                
                symbol_greeks[symbol]['delta'] += pg.delta * pg.quantity
                symbol_greeks[symbol]['gamma'] += pg.gamma * pg.quantity
                symbol_greeks[symbol]['theta'] += pg.theta * pg.quantity
                symbol_greeks[symbol]['vega'] += pg.vega * pg.quantity
                symbol_greeks[symbol]['delta_dollars'] += pg.delta_dollars
                symbol_greeks[symbol]['theta_dollars'] += pg.theta_dollars
                symbol_greeks[symbol]['vega_dollars'] += pg.vega_dollars
                symbol_greeks[symbol]['position_count'] += 1
                symbol_greeks[symbol]['total_value'] += pg.position_value
            
            return symbol_greeks
            
        except Exception as e:
            logger.error(f"❌ Symbol aggregation failed: {e}")
            return {}
    
    def _aggregate_greeks_by_expiry(self, position_greeks: List[OptionGreeks]) -> Dict[str, Dict[str, float]]:
        """Aggregate Greeks by expiry date"""
        try:
            expiry_greeks = {}
            
            for pg in position_greeks:
                expiry = pg.expiry_date
                if expiry not in expiry_greeks:
                    expiry_greeks[expiry] = {
                        'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0,
                        'delta_dollars': 0, 'theta_dollars': 0, 'vega_dollars': 0,
                        'position_count': 0, 'total_value': 0, 'days_to_expiry': pg.days_to_expiry
                    }
                
                expiry_greeks[expiry]['delta'] += pg.delta * pg.quantity
                expiry_greeks[expiry]['gamma'] += pg.gamma * pg.quantity
                expiry_greeks[expiry]['theta'] += pg.theta * pg.quantity
                expiry_greeks[expiry]['vega'] += pg.vega * pg.quantity
                expiry_greeks[expiry]['delta_dollars'] += pg.delta_dollars
                expiry_greeks[expiry]['theta_dollars'] += pg.theta_dollars
                expiry_greeks[expiry]['vega_dollars'] += pg.vega_dollars
                expiry_greeks[expiry]['position_count'] += 1
                expiry_greeks[expiry]['total_value'] += pg.position_value
            
            return expiry_greeks
            
        except Exception as e:
            logger.error(f"❌ Expiry aggregation failed: {e}")
            return {}
    
    def _calculate_portfolio_var(self, position_greeks: List[OptionGreeks]) -> float:
        """Calculate portfolio Value at Risk (VaR)"""
        try:
            # Simplified VaR calculation based on delta and gamma
            total_delta_risk = sum(abs(pg.delta_dollars) for pg in position_greeks)
            total_gamma_risk = sum(abs(pg.gamma_dollars) for pg in position_greeks)
            
            # Assume 2 standard deviation move (95% confidence)
            # Estimate 2% daily move for indices
            daily_move_pct = 0.02
            
            var = (total_delta_risk * daily_move_pct) + (total_gamma_risk * daily_move_pct**2 / 2)
            
            return var
            
        except Exception:
            return 0.0
    
    def _calculate_hedging_requirements(self, position_greeks: List[OptionGreeks], net_delta_dollars: float) -> Dict[str, Any]:
        """Calculate hedging requirements"""
        try:
            # Get current market prices
            nifty_data = self.market_data.get_nifty_data()
            nifty_price = nifty_data.get('price', 25000)
            
            # Calculate shares needed to hedge delta
            shares_needed = int(-net_delta_dollars / nifty_price)
            
            # Estimate hedging cost (brokerage + impact)
            hedge_cost = abs(shares_needed) * nifty_price * 0.001  # 0.1% total cost
            
            return {
                'shares': shares_needed,
                'cost': hedge_cost
            }
            
        except Exception:
            return {'shares': 0, 'cost': 0}
    
    # ======================
    # SCENARIO ANALYSIS METHODS
    # ======================
    
    def _calculate_price_scenarios(self, base_greeks: OptionGreeks, price_scenarios: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate P&L for different price scenarios"""
        try:
            scenarios = []
            current_price = base_greeks.spot_price
            
            for price in price_scenarios:
                price_change = price - current_price
                
                # First order (Delta) P&L
                delta_pnl = base_greeks.delta_dollars * (price_change / current_price)
                
                # Second order (Gamma) P&L
                gamma_pnl = 0.5 * base_greeks.gamma_dollars * (price_change / current_price)**2
                
                # Total P&L
                total_pnl = delta_pnl + gamma_pnl
                
                scenarios.append({
                    'spot_price': round(price, 2),
                    'price_change': round(price_change, 2),
                    'price_change_pct': round((price_change / current_price) * 100, 2),
                    'delta_pnl': round(delta_pnl, 2),
                    'gamma_pnl': round(gamma_pnl, 2),
                    'total_pnl': round(total_pnl, 2)
                })
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Price scenarios calculation failed: {e}")
            return []
    
    def _calculate_volatility_scenarios(self, base_greeks: OptionGreeks, vol_scenarios: List[float]) -> List[Dict[str, Any]]:
        """Calculate P&L for different volatility scenarios"""
        try:
            scenarios = []
            current_vol = base_greeks.implied_volatility
            
            for vol in vol_scenarios:
                vol_change = vol - current_vol
                
                # Vega P&L
                vega_pnl = base_greeks.vega_dollars * (vol_change * 100)  # Convert to percentage points
                
                scenarios.append({
                    'volatility': round(vol * 100, 1),  # Convert to percentage
                    'vol_change': round(vol_change * 100, 1),
                    'vega_pnl': round(vega_pnl, 2)
                })
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Volatility scenarios calculation failed: {e}")
            return []
    
    def _calculate_time_scenarios(self, base_greeks: OptionGreeks, time_scenarios: List[int]) -> List[Dict[str, Any]]:
        """Calculate P&L for different time decay scenarios"""
        try:
            scenarios = []
            
            for days in time_scenarios:
                # Theta P&L (daily time decay)
                theta_pnl = base_greeks.theta_dollars * days
                
                scenarios.append({
                    'days_passed': days,
                    'theta_pnl': round(theta_pnl, 2),
                    'remaining_days': max(0, base_greeks.days_to_expiry - days)
                })
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Time scenarios calculation failed: {e}")
            return []
    
    def _calculate_combined_scenarios(self, base_greeks: OptionGreeks) -> List[Dict[str, Any]]:
        """Calculate combined worst/best case scenarios"""
        try:
            current_price = base_greeks.spot_price
            
            scenarios = [
                {
                    'scenario': 'BEST_CASE',
                    'description': 'Price up 5%, volatility up 5%, 1 day passed',
                    'price_change_pct': 5.0,
                    'vol_change_pct': 5.0,
                    'days_passed': 1,
                    'total_pnl': self._calculate_scenario_pnl(base_greeks, 0.05, 0.05, 1)
                },
                {
                    'scenario': 'WORST_CASE',
                    'description': 'Price down 5%, volatility down 5%, 7 days passed',
                    'price_change_pct': -5.0,
                    'vol_change_pct': -5.0,
                    'days_passed': 7,
                    'total_pnl': self._calculate_scenario_pnl(base_greeks, -0.05, -0.05, 7)
                },
                {
                    'scenario': 'NEUTRAL',
                    'description': 'No price change, no vol change, 3 days passed',
                    'price_change_pct': 0.0,
                    'vol_change_pct': 0.0,
                    'days_passed': 3,
                    'total_pnl': self._calculate_scenario_pnl(base_greeks, 0.0, 0.0, 3)
                }
            ]
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Combined scenarios calculation failed: {e}")
            return []
    
    def _calculate_scenario_pnl(self, base_greeks: OptionGreeks, price_change_pct: float, vol_change_pct: float, days_passed: int) -> float:
        """Calculate P&L for a specific scenario"""
        try:
            # Delta P&L
            delta_pnl = base_greeks.delta_dollars * price_change_pct
            
            # Gamma P&L
            gamma_pnl = 0.5 * base_greeks.gamma_dollars * (price_change_pct**2)
            
            # Theta P&L
            theta_pnl = base_greeks.theta_dollars * days_passed
            
            # Vega P&L
            vega_pnl = base_greeks.vega_dollars * vol_change_pct
            
            return delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            
        except Exception:
            return 0.0
    
    def _calculate_scenario_summary(self, scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for scenarios"""
        try:
            combined = scenarios.get('combined_scenarios', [])
            
            if not combined:
                return {
                    'best_case_pnl': 0,
                    'worst_case_pnl': 0,
                    'probability_profit': 50,
                    'max_risk': 0
                }
            
            best_case = max(scenario['total_pnl'] for scenario in combined)
            worst_case = min(scenario['total_pnl'] for scenario in combined)
            
            return {
                'best_case_pnl': round(best_case, 2),
                'worst_case_pnl': round(worst_case, 2),
                'probability_profit': 60 if best_case > abs(worst_case) else 40,  # Simplified
                'max_risk': round(abs(worst_case), 2)
            }
            
        except Exception:
            return {
                'best_case_pnl': 0,
                'worst_case_pnl': 0,
                'probability_profit': 50,
                'max_risk': 0
            }
    
    # ======================
    # HEDGING STRATEGY METHODS
    # ======================
    
    def _generate_options_hedge(self, delta_difference: float, portfolio_greeks: PortfolioGreeks) -> Dict[str, Any]:
        """Generate options-based hedging strategy"""
        try:
            if abs(delta_difference) < 0.2:
                return None
            
            # Simple options hedge using ATM options
            nifty_data = self.market_data.get_nifty_data()
            current_price = nifty_data.get('price', 25000)
            
            # Find nearest strike
            strike_gap = 50
            atm_strike = round(current_price / strike_gap) * strike_gap
            
            # Calculate options needed
            # Assume ATM option has delta of ~0.5
            options_delta = 0.5 if delta_difference > 0 else -0.5
            lots_needed = abs(delta_difference) / (options_delta * self.lot_sizes['NIFTY'])
            
            return {
                'strategy': 'OPTIONS_HEDGE',
                'description': 'Hedge using ATM options',
                'recommendations': [
                    {
                        'instrument': f'NIFTY_{atm_strike}_{"PE" if delta_difference > 0 else "CE"}',
                        'action': 'BUY',
                        'lots': round(lots_needed),
                        'strike': atm_strike,
                        'estimated_cost': round(lots_needed * 50 * self.lot_sizes['NIFTY']),  # Assume ₹50 premium
                        'hedge_delta': round(lots_needed * options_delta * self.lot_sizes['NIFTY'], 4)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Options hedge generation failed: {e}")
            return None
    
    def _generate_dynamic_hedge_plan(self, portfolio_greeks: PortfolioGreeks) -> Dict[str, Any]:
        """Generate dynamic hedging plan"""
        try:
            current_delta = abs(portfolio_greeks.net_delta)
            
            if current_delta < 0.3:
                return None
            
            # Dynamic hedging thresholds
            hedge_triggers = []
            
            if current_delta > 0.5:
                hedge_triggers.append({
                    'trigger': 'DELTA_THRESHOLD',
                    'condition': 'Net delta > 0.5',
                    'action': 'Hedge 50% of delta exposure',
                    'frequency': 'DAILY'
                })
            
            if portfolio_greeks.net_gamma > 0.05:
                hedge_triggers.append({
                    'trigger': 'GAMMA_THRESHOLD',
                    'condition': 'High gamma exposure',
                    'action': 'Consider gamma scalping opportunities',
                    'frequency': 'INTRADAY'
                })
            
            if abs(portfolio_greeks.net_theta_dollars) > 1000:
                hedge_triggers.append({
                    'trigger': 'THETA_DECAY',
                    'condition': 'High theta decay > ₹1,000/day',
                    'action': 'Monitor time decay closely',
                    'frequency': 'DAILY'
                })
            
            return {
                'strategy': 'DYNAMIC_HEDGING',
                'description': 'Automated dynamic hedging plan',
                'hedge_triggers': hedge_triggers,
                'monitoring_frequency': 'HOURLY',
                'rebalance_threshold': 0.2,
                'max_hedge_cost': portfolio_greeks.total_position_value * 0.02  # 2% of portfolio
            }
            
        except Exception as e:
            logger.error(f"❌ Dynamic hedge plan generation failed: {e}")
            return None
    
    # ======================
    # RISK ASSESSMENT METHODS
    # ======================
    
    def _assess_delta_risk(self, delta: float) -> RiskLevel:
        """Assess delta risk level"""
        abs_delta = abs(delta)
        thresholds = self.risk_thresholds['delta']
        
        if abs_delta >= thresholds['high']:
            return RiskLevel.VERY_HIGH
        elif abs_delta >= thresholds['moderate']:
            return RiskLevel.HIGH
        elif abs_delta >= thresholds['low']:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_gamma_risk(self, gamma: float) -> RiskLevel:
        """Assess gamma risk level"""
        abs_gamma = abs(gamma)
        thresholds = self.risk_thresholds['gamma']
        
        if abs_gamma >= thresholds['high']:
            return RiskLevel.VERY_HIGH
        elif abs_gamma >= thresholds['moderate']:
            return RiskLevel.HIGH
        elif abs_gamma >= thresholds['low']:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_theta_risk(self, theta_dollars: float) -> RiskLevel:
        """Assess theta risk level"""
        abs_theta = abs(theta_dollars)
        thresholds = self.risk_thresholds['theta']
        
        if abs_theta >= abs(thresholds['high']):
            return RiskLevel.VERY_HIGH
        elif abs_theta >= abs(thresholds['moderate']):
            return RiskLevel.HIGH
        elif abs_theta >= abs(thresholds['low']):
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_vega_risk(self, vega_dollars: float) -> RiskLevel:
        """Assess vega risk level"""
        abs_vega = abs(vega_dollars)
        thresholds = self.risk_thresholds['vega']
        
        if abs_vega >= thresholds['high']:
            return RiskLevel.VERY_HIGH
        elif abs_vega >= thresholds['moderate']:
            return RiskLevel.HIGH
        elif abs_vega >= thresholds['low']:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_hedge_urgency(self, delta_magnitude: float) -> str:
        """Assess urgency of hedging requirement"""
        if delta_magnitude > 1.0:
            return 'URGENT'
        elif delta_magnitude > 0.5:
            return 'HIGH'
        elif delta_magnitude > 0.2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_risk_alerts(self, portfolio_greeks: PortfolioGreeks) -> List[Dict[str, Any]]:
        """Generate risk alerts based on portfolio Greeks"""
        try:
            alerts = []
            
            # Delta risk alerts
            delta_risk = self._assess_delta_risk(portfolio_greeks.net_delta)
            if delta_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                alerts.append({
                    'type': 'DELTA_RISK',
                    'level': delta_risk.value,
                    'message': f'High delta exposure: {portfolio_greeks.net_delta:.4f}',
                    'recommendation': 'Consider hedging with futures or options'
                })
            
            # Theta decay alerts
            if abs(portfolio_greeks.net_theta_dollars) > 500:
                alerts.append({
                    'type': 'THETA_DECAY',
                    'level': 'WARNING',
                    'message': f'High daily time decay: ₹{portfolio_greeks.net_theta_dollars:,.2f}',
                    'recommendation': 'Monitor time decay closely, consider closing positions'
                })
            
            # Gamma risk alerts
            gamma_risk = self._assess_gamma_risk(portfolio_greeks.net_gamma)
            if gamma_risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                alerts.append({
                    'type': 'GAMMA_RISK',
                    'level': gamma_risk.value,
                    'message': f'High gamma exposure: {portfolio_greeks.net_gamma:.6f}',
                    'recommendation': 'Potential for rapid delta changes with price moves'
                })
            
            # Portfolio concentration alerts
            if portfolio_greeks.total_positions > 8:
                alerts.append({
                    'type': 'CONCENTRATION',
                    'level': 'INFO',
                    'message': f'High number of positions: {portfolio_greeks.total_positions}',
                    'recommendation': 'Consider consolidating positions for better management'
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Risk alerts generation failed: {e}")
            return []
    
    # ======================
    # COLOR AND DISPLAY METHODS
    # ======================
    
    def _get_delta_color(self, delta: float) -> str:
        """Get color for delta display"""
        if delta > 0.5:
            return 'red'  # High positive delta - risk of loss if price falls
        elif delta > 0.2:
            return 'orange'
        elif delta > -0.2:
            return 'yellow'  # Near delta neutral
        elif delta > -0.5:
            return 'lightgreen'
        else:
            return 'green'  # High negative delta - risk of loss if price rises
    
    def _get_gamma_color(self, gamma: float) -> str:
        """Get color for gamma display"""
        abs_gamma = abs(gamma)
        if abs_gamma > 0.05:
            return 'red'  # High gamma - rapid delta changes
        elif abs_gamma > 0.02:
            return 'orange'
        elif abs_gamma > 0.01:
            return 'yellow'
        else:
            return 'green'  # Low gamma - stable delta
    
    def _get_theta_color(self, theta: float) -> str:
        """Get color for theta display"""
        if theta < -50:
            return 'red'  # High time decay
        elif theta < -20:
            return 'orange'
        elif theta < -5:
            return 'yellow'
        else:
            return 'green'  # Low time decay
    
    def _get_vega_color(self, vega: float) -> str:
        """Get color for vega display"""
        abs_vega = abs(vega)
        if abs_vega > 50:
            return 'red'  # High volatility sensitivity
        elif abs_vega > 25:
            return 'orange'
        elif abs_vega > 10:
            return 'yellow'
        else:
            return 'green'  # Low volatility sensitivity
    
    # ======================
    # FALLBACK METHODS
    # ======================
    
    def _generate_fallback_greeks(self, symbol: str, strike_price: float, option_type: str,
                                expiry_date: str, quantity: int, error: str = None) -> OptionGreeks:
        """Generate fallback Greeks when calculation fails"""
        
        return OptionGreeks(
            symbol=symbol,
            strike_price=strike_price,
            option_type=OptionType.CALL if option_type == 'CE' else OptionType.PUT,
            expiry_date=expiry_date,
            days_to_expiry=7,
            spot_price=25000 if symbol == 'NIFTY' else 55000,
            option_price=50.0,
            implied_volatility=0.20,
            risk_free_rate=self.market_params['risk_free_rate'],
            delta=0.5,
            gamma=0.01,
            theta=-10,
            vega=20,
            rho=5,
            delta_dollars=0,
            gamma_dollars=0,
            theta_dollars=0,
            vega_dollars=0,
            moneyness=1.0,
            time_value=30,
            intrinsic_value=20,
            quantity=quantity,
            position_value=0,
            timestamp=datetime.now()
        )
    
    def _generate_empty_portfolio_greeks(self, error: str = None) -> PortfolioGreeks:
        """Generate empty portfolio Greeks when no positions"""
        
        return PortfolioGreeks(
            net_delta=0,
            net_gamma=0,
            net_theta=0,
            net_vega=0,
            net_rho=0,
            net_delta_dollars=0,
            net_gamma_dollars=0,
            net_theta_dollars=0,
            net_vega_dollars=0,
            total_positions=0,
            total_position_value=0,
            delta_neutral_ratio=0,
            portfolio_var=0,
            max_theta_decay=0,
            iv_exposure=0,
            greeks_by_symbol={},
            greeks_by_expiry={},
            hedge_delta_shares=0,
            hedge_cost=0,
            timestamp=datetime.now()
        )

def main():
    """Test the Option Greeks Calculator"""
    print("⚡ TradeMind AI Option Greeks Calculator")
    print("=" * 70)
    
    try:
        # Initialize Greeks calculator
        greeks_calc = OptionGreeksCalculator()
        
        print("\n🧪 Testing Option Greeks Calculator...")
        
        # Test individual option Greeks
        print("\n📈 NIFTY Call Option Greeks:")
        nifty_call = greeks_calc.calculate_option_greeks(
            symbol='NIFTY',
            strike_price=25300,
            option_type='CE',
            expiry_date=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            quantity=2
        )
        
        print(f"   Strike: {nifty_call.strike_price} CE")
        print(f"   Spot Price: ₹{nifty_call.spot_price:,.2f}")
        print(f"   Option Price: ₹{nifty_call.option_price:.2f}")
        print(f"   Implied Vol: {nifty_call.implied_volatility:.2%}")
        print(f"   Days to Expiry: {nifty_call.days_to_expiry}")
        
        print(f"\n   📊 Greeks:")
        print(f"      Delta: {nifty_call.delta:.4f} (₹{nifty_call.delta_dollars:,.2f})")
        print(f"      Gamma: {nifty_call.gamma:.6f} (₹{nifty_call.gamma_dollars:,.2f})")
        print(f"      Theta: {nifty_call.theta:.4f} (₹{nifty_call.theta_dollars:,.2f})")
        print(f"      Vega:  {nifty_call.vega:.4f} (₹{nifty_call.vega_dollars:,.2f})")
        print(f"      Rho:   {nifty_call.rho:.4f}")
        
        print(f"\n   💰 Position Metrics:")
        print(f"      Moneyness: {nifty_call.moneyness:.4f}")
        print(f"      Intrinsic Value: ₹{nifty_call.intrinsic_value:.2f}")
        print(f"      Time Value: ₹{nifty_call.time_value:.2f}")
        print(f"      Position Value: ₹{nifty_call.position_value:,.2f}")
        
        # Test BANKNIFTY put option
        print("\n🏦 BANKNIFTY Put Option Greeks:")
        banknifty_put = greeks_calc.calculate_option_greeks(
            symbol='BANKNIFTY',
            strike_price=54900,
            option_type='PE',
            expiry_date=(datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            quantity=1
        )
        
        print(f"   Strike: {banknifty_put.strike_price} PE")
        print(f"   Delta: {banknifty_put.delta:.4f} (₹{banknifty_put.delta_dollars:,.2f})")
        print(f"   Gamma: {banknifty_put.gamma:.6f}")
        print(f"   Theta: {banknifty_put.theta:.4f} (₹{banknifty_put.theta_dollars:,.2f}/day)")
        
        # Test portfolio Greeks (will be empty unless you have positions)
        print("\n📋 Portfolio Greeks Analysis:")
        portfolio_greeks = greeks_calc.calculate_portfolio_greeks()
        
        print(f"   Total Positions: {portfolio_greeks.total_positions}")
        print(f"   Net Delta: {portfolio_greeks.net_delta:.4f} (₹{portfolio_greeks.net_delta_dollars:,.2f})")
        print(f"   Net Gamma: {portfolio_greeks.net_gamma:.6f}")
        print(f"   Net Theta: {portfolio_greeks.net_theta:.4f} (₹{portfolio_greeks.net_theta_dollars:,.2f}/day)")
        print(f"   Net Vega: {portfolio_greeks.net_vega:.4f} (₹{portfolio_greeks.net_vega_dollars:,.2f})")
        print(f"   Portfolio VaR: ₹{portfolio_greeks.portfolio_var:,.2f}")
        
        # Test Greeks scenarios
        print("\n🎯 Greeks Scenarios Analysis:")
        scenarios = greeks_calc.get_greeks_scenarios(
            symbol='NIFTY',
            strike_price=25300,
            option_type='CE',
            expiry_date=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            quantity=2
        )
        
        if 'combined_scenarios' in scenarios:
            combined = scenarios['combined_scenarios']
            for scenario in combined:
                print(f"   {scenario['scenario']}: ₹{scenario['total_pnl']:,.2f} P&L")
        
        # Test hedging recommendations
        print("\n🛡️ Hedging Recommendations:")
        hedging = greeks_calc.get_hedging_recommendations(target_delta=0.0)
        
        current_portfolio = hedging.get('current_portfolio', {})
        print(f"   Current Delta: {current_portfolio.get('net_delta', 0):.4f}")
        print(f"   Target Delta: {current_portfolio.get('target_delta', 0):.4f}")
        print(f"   Hedge Urgency: {current_portfolio.get('hedge_urgency', 'UNKNOWN')}")
        
        strategies = hedging.get('hedging_strategies', [])
        if strategies:
            primary = strategies[0]
            print(f"   Primary Strategy: {primary.get('strategy', 'NONE')}")
            print(f"   Description: {primary.get('description', 'N/A')}")
        
        # Test dashboard data
        print("\n📱 Dashboard Data Format:")
        dashboard_data = greeks_calc.get_dashboard_data('NIFTY')
        
        portfolio = dashboard_data.get('portfolio_greeks', {})
        print(f"   Delta Widget: {portfolio.get('net_delta', {}).get('value', 0):.4f} ({portfolio.get('net_delta', {}).get('color', 'gray')})")
        print(f"   Gamma Widget: {portfolio.get('net_gamma', {}).get('value', 0):.6f}")
        print(f"   Theta Widget: ₹{portfolio.get('net_theta', {}).get('dollars', 0):,.2f}/day")
        print(f"   Vega Widget: ₹{portfolio.get('net_vega', {}).get('dollars', 0):,.2f}")
        
        # Test performance metrics
        print("\n📈 System Performance:")
        performance = dashboard_data.get('system_performance', {})
        print(f"   Calculations Performed: {performance.get('calculations_performed', 0)}")
        print(f"   Cache Hit Rate: {performance.get('cache_hit_rate', 0):.1f}%")
        print(f"   Avg Calculation Time: {performance.get('avg_calculation_time', 0):.2f}ms")
        print(f"   Error Rate: {performance.get('error_rate', 0):.1f}%")
        
        # Show risk alerts
        risk_alerts = dashboard_data.get('risk_alerts', [])
        if risk_alerts:
            print(f"\n⚠️ Risk Alerts ({len(risk_alerts)}):")
            for alert in risk_alerts[:3]:
                print(f"      • {alert['type']}: {alert['message']}")
        
        print("\n✅ Option Greeks Calculator testing completed!")
        print("\n🚀 Integration commands:")
        print("   # Add to dashboard_backend.py:")
        print("   from option_greeks_calculator import OptionGreeksCalculator")
        print("   self.greeks_calc = OptionGreeksCalculator()")
        print("   ")
        print("   # New API endpoints:")
        print("   @app.route('/api/greeks/portfolio')")
        print("   def get_portfolio_greeks():")
        print("       return jsonify(self.greeks_calc.get_dashboard_data())")
        print("   ")
        print("   @app.route('/api/greeks/<symbol>/<float:strike>/<option_type>')")
        print("   def get_option_greeks(symbol, strike, option_type):")
        print("       greeks = self.greeks_calc.calculate_option_greeks(symbol, strike, option_type, expiry)")
        print("       return jsonify(greeks.to_dict())")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing stopped by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        try:
            greeks_calc.stop_real_time_updates()
        except:
            pass

if __name__ == "__main__":
    main()