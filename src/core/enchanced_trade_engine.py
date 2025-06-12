# enhanced_trading_engine.py
import os
import json
import asyncio
from datetime import datetime, time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union  # Fixed: Added Union
import requests
from dataclasses import dataclass
import yfinance as yf

@dataclass
class TradeSignal:
    """Complete trade signal with all parameters"""
    symbol: str
    strike: int
    option_type: str  # CE or PE
    entry_price: float
    current_price: float
    target1: float
    target2: float
    stoploss: float
    quantity: int
    lot_size: int
    confidence: float
    risk_reward: float
    max_loss: float
    max_profit: float
    signal_time: datetime
    expiry: str

@dataclass
class TradeExecution:
    """Track trade execution"""
    order_id: str
    signal: TradeSignal
    entry_time: datetime
    entry_price: float
    current_price: float
    pnl: float
    status: str  # PENDING, EXECUTED, TARGET1_HIT, TARGET2_HIT, SL_HIT, EXITED
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

class SmartStrikeSelector:
    """Intelligent strike selection based on multiple factors"""
    
    def __init__(self):
        # UPDATED LOT SIZES AS PER SEBI GUIDELINES (June 2025)
        self.nifty_lot_size = 75      # Changed from 25 on Dec 26, 2024
        self.banknifty_lot_size = 30  # Changed from 15 on Dec 24, 2024
        self.strike_gaps = {
            'NIFTY': 50,
            'BANKNIFTY': 100
        }
        
    def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Fetch real option chain data"""
        # In production, this would fetch from broker API
        # For now, simulating with realistic data
        current_price = self.get_current_price(symbol)
        strikes = self.generate_strikes(symbol, current_price)
        
        option_data = []
        for strike in strikes:
            # Simulate option data
            ce_premium = self.calculate_premium(current_price, strike, 'CE')
            pe_premium = self.calculate_premium(current_price, strike, 'PE')
            
            option_data.append({
                'strike': strike,
                'CE_LTP': ce_premium,
                'PE_LTP': pe_premium,
                'CE_OI': np.random.randint(1000, 50000),
                'PE_OI': np.random.randint(1000, 50000),
                'CE_volume': np.random.randint(100, 5000),
                'PE_volume': np.random.randint(100, 5000),
                'CE_IV': np.random.uniform(15, 25),
                'PE_IV': np.random.uniform(15, 25)
            })
            
        return pd.DataFrame(option_data)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price of underlying"""
        ticker = yf.Ticker(f"^{'NSEI' if symbol == 'NIFTY' else 'NSEBANK'}")
        data = ticker.history(period="1d", interval="1m")
        return data['Close'].iloc[-1] if not data.empty else (25000 if symbol == 'NIFTY' else 56000)
    
    def generate_strikes(self, symbol: str, current_price: float) -> List[int]:
        """Generate relevant strikes around current price"""
        gap = self.strike_gaps[symbol]
        atm_strike = round(current_price / gap) * gap
        
        strikes = []
        for i in range(-10, 11):  # 10 strikes above and below ATM
            strikes.append(atm_strike + (i * gap))
        
        return strikes
    
    def calculate_premium(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate option premium using simplified Black-Scholes"""
        # Simplified calculation for demonstration
        intrinsic = max(0, spot - strike) if option_type == 'CE' else max(0, strike - spot)
        time_value = abs(spot - strike) * 0.02  # 2% time value
        volatility_premium = np.random.uniform(10, 30)
        
        return round(intrinsic + time_value + volatility_premium, 2)
    
    def select_best_strike(self, symbol: str, direction: str, confidence: float) -> Dict:
        """Select the best strike based on multiple factors"""
        current_price = self.get_current_price(symbol)
        option_chain = self.get_option_chain(symbol)
        
        # Strike selection logic based on confidence
        if confidence >= 90:
            # High confidence = ATM or slightly OTM
            offset = 0 if confidence >= 95 else 1
        elif confidence >= 80:
            # Medium confidence = 1-2 strikes OTM
            offset = 2
        else:
            # Lower confidence = 2-3 strikes OTM for better risk-reward
            offset = 3
        
        gap = self.strike_gaps[symbol]
        atm_strike = round(current_price / gap) * gap
        
        if direction == 'BULLISH':
            selected_strike = atm_strike + (offset * gap)
            option_type = 'CE'
        else:
            selected_strike = atm_strike - (offset * gap)
            option_type = 'PE'
        
        # Get option details
        option_data = option_chain[option_chain['strike'] == selected_strike].iloc[0]
        premium = option_data[f'{option_type}_LTP']
        
        return {
            'strike': selected_strike,
            'option_type': option_type,
            'premium': premium,
            'current_spot': current_price,
            'moneyness': 'ATM' if offset == 0 else f'{offset}-OTM',
            'lot_size': self.nifty_lot_size if symbol == 'NIFTY' else self.banknifty_lot_size
        }

class AdvancedTradeManager:
    """Complete trade management with entry, targets, SL"""
    
    def __init__(self):
        self.strike_selector = SmartStrikeSelector()
        self.active_trades: List[TradeExecution] = []
        self.completed_trades: List[TradeExecution] = []
        
    def calculate_targets_and_sl(self, entry_price: float, confidence: float, 
                                volatility: float = 20) -> Dict:
        """Calculate targets and stop loss based on risk-reward"""
        # Dynamic risk-reward based on confidence
        if confidence >= 90:
            risk_reward_ratio = 1.5  # 1:1.5
            sl_percentage = 15
        elif confidence >= 80:
            risk_reward_ratio = 2.0  # 1:2
            sl_percentage = 20
        else:
            risk_reward_ratio = 3.0  # 1:3
            sl_percentage = 25
        
        # Calculate SL
        stoploss = entry_price * (1 - sl_percentage / 100)
        risk_amount = entry_price - stoploss
        
        # Calculate targets
        target1 = entry_price + (risk_amount * risk_reward_ratio)
        target2 = entry_price + (risk_amount * risk_reward_ratio * 1.5)
        
        return {
            'entry': round(entry_price, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'stoploss': round(stoploss, 2),
            'risk_reward': risk_reward_ratio,
            'risk_percentage': sl_percentage
        }
    
    def create_trade_signal(self, symbol: str, market_analysis: Dict) -> TradeSignal:
        """Create complete trade signal with all parameters"""
        # Get strike selection
        strike_info = self.strike_selector.select_best_strike(
            symbol,
            market_analysis['direction'],
            market_analysis['confidence']
        )
        
        # Calculate entry, targets, and SL
        trade_params = self.calculate_targets_and_sl(
            strike_info['premium'],
            market_analysis['confidence']
        )
        
        # Determine quantity based on risk management
        max_risk_per_trade = 10000  # Max Rs. 10,000 risk per trade
        risk_per_lot = (trade_params['entry'] - trade_params['stoploss']) * strike_info['lot_size']
        quantity = min(int(max_risk_per_trade / risk_per_lot), 10)  # Max 10 lots
        
        # Create trade signal
        signal = TradeSignal(
            symbol=f"{symbol} {strike_info['strike']} {strike_info['option_type']}",
            strike=strike_info['strike'],
            option_type=strike_info['option_type'],
            entry_price=trade_params['entry'],
            current_price=trade_params['entry'],
            target1=trade_params['target1'],
            target2=trade_params['target2'],
            stoploss=trade_params['stoploss'],
            quantity=quantity,
            lot_size=strike_info['lot_size'],
            confidence=market_analysis['confidence'],
            risk_reward=trade_params['risk_reward'],
            max_loss=quantity * strike_info['lot_size'] * (trade_params['entry'] - trade_params['stoploss']),
            max_profit=quantity * strike_info['lot_size'] * (trade_params['target2'] - trade_params['entry']),
            signal_time=datetime.now(),
            expiry=self.get_current_expiry(symbol)
        )
        
        return signal
    
    def get_current_expiry(self, symbol: str) -> str:
        """Get current week expiry date"""
        today = datetime.now()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.time() > time(15, 30):
            days_until_thursday = 7
        
        expiry = today + pd.Timedelta(days=days_until_thursday)
        return expiry.strftime("%d%b%Y").upper()
    
    def execute_trade(self, signal: TradeSignal, broker_api=None) -> TradeExecution:
        """Execute trade with broker (simulated)"""
        # In production, this would place actual order via broker API
        order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        execution = TradeExecution(
            order_id=order_id,
            signal=signal,
            entry_time=datetime.now(),
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            pnl=0,
            status="EXECUTED"
        )
        
        self.active_trades.append(execution)
        
        return execution
    
    def monitor_trades(self) -> List[Dict]:
        """Monitor all active trades and update status"""
        updates = []
        
        for trade in self.active_trades:
            # Simulate price movement (in production, fetch from broker)
            price_change = np.random.uniform(-5, 5)
            trade.current_price = round(trade.entry_price * (1 + price_change / 100), 2)
            
            # Calculate P&L
            trade.pnl = (trade.current_price - trade.entry_price) * \
                       trade.signal.quantity * trade.signal.lot_size
            
            # Check for target/SL hit
            if trade.current_price >= trade.signal.target2:
                trade.status = "TARGET2_HIT"
                trade.exit_price = trade.signal.target2
                trade.exit_time = datetime.now()
                self._close_trade(trade)
                updates.append(self._create_update_message(trade, "TARGET 2 ACHIEVED! 🎯"))
                
            elif trade.current_price >= trade.signal.target1 and trade.status != "TARGET1_HIT":
                trade.status = "TARGET1_HIT"
                # Trail SL to cost
                trade.signal.stoploss = trade.entry_price
                updates.append(self._create_update_message(trade, "TARGET 1 HIT! SL moved to cost 📈"))
                
            elif trade.current_price <= trade.signal.stoploss:
                trade.status = "SL_HIT"
                trade.exit_price = trade.signal.stoploss
                trade.exit_time = datetime.now()
                self._close_trade(trade)
                updates.append(self._create_update_message(trade, "STOPLOSS HIT ⛔"))
        
        return updates
    
    def _close_trade(self, trade: TradeExecution):
        """Close trade and move to completed"""
        self.active_trades.remove(trade)
        self.completed_trades.append(trade)
    
    def _create_update_message(self, trade: TradeExecution, status_msg: str) -> Dict:
        """Create formatted update message"""
        return {
            'time': datetime.now().strftime("%H:%M:%S"),
            'symbol': trade.signal.symbol,
            'status': status_msg,
            'entry': trade.entry_price,
            'current': trade.current_price,
            'pnl': f"₹{trade.pnl:,.2f}",
            'roi': f"{(trade.pnl / (trade.entry_price * trade.signal.quantity * trade.signal.lot_size)) * 100:.2f}%"
        }
    
    def generate_trade_report(self, trade: Union[TradeSignal, TradeExecution]) -> str:
        """Generate formatted trade report"""
        if isinstance(trade, TradeSignal):
            return f"""
📊 **NEW TRADE SIGNAL** 📊
{'='*40}
🎯 **{trade.symbol}**
📅 Expiry: {trade.expiry}

💰 **ENTRY**: ₹{trade.entry_price}
🎯 **TARGET 1**: ₹{trade.target1} (+{((trade.target1/trade.entry_price - 1) * 100):.1f}%)
🎯 **TARGET 2**: ₹{trade.target2} (+{((trade.target2/trade.entry_price - 1) * 100):.1f}%)
🛡️ **STOPLOSS**: ₹{trade.stoploss} (-{((1 - trade.stoploss/trade.entry_price) * 100):.1f}%)

📈 **QUANTITY**: {trade.quantity} lots ({trade.quantity * trade.lot_size} qty)
⚖️ **RISK:REWARD**: 1:{trade.risk_reward}
💯 **CONFIDENCE**: {trade.confidence}%

💸 **MAX RISK**: ₹{trade.max_loss:,.0f}
💰 **MAX PROFIT**: ₹{trade.max_profit:,.0f}
{'='*40}
"""
        else:
            # Trade execution report
            return f"""
📊 **TRADE UPDATE** 📊
{'='*40}
🎯 **{trade.signal.symbol}**
🆔 Order ID: {trade.order_id}

💰 **ENTRY**: ₹{trade.entry_price}
📍 **CURRENT**: ₹{trade.current_price}
📈 **P&L**: ₹{trade.pnl:,.2f}
🚦 **STATUS**: {trade.status}

🎯 **TARGET 1**: ₹{trade.signal.target1} {'✅' if trade.current_price >= trade.signal.target1 else '⏳'}
🎯 **TARGET 2**: ₹{trade.signal.target2} {'✅' if trade.current_price >= trade.signal.target2 else '⏳'}
🛡️ **STOPLOSS**: ₹{trade.signal.stoploss}
{'='*40}
"""

class EnhancedMasterTrader:
    """Complete trading system with full order management"""
    
    def __init__(self):
        self.trade_manager = AdvancedTradeManager()
        self.monitoring = False
        
    async def analyze_and_trade(self, symbol: str, market_analysis: Dict):
        """Complete trade flow from analysis to execution"""
        print(f"\n{'='*60}")
        print(f"🤖 ENHANCED AI TRADING SYSTEM - {symbol}")
        print(f"{'='*60}")
        
        # 1. Create trade signal
        signal = self.trade_manager.create_trade_signal(symbol, market_analysis)
        print(self.trade_manager.generate_trade_report(signal))
        
        # 2. Execute trade
        execution = self.trade_manager.execute_trade(signal)
        print(f"\n✅ ORDER PLACED SUCCESSFULLY!")
        print(f"Order ID: {execution.order_id}")
        
        # 3. Start monitoring
        await self.monitor_loop()
    
    async def monitor_loop(self):
        """Continuous monitoring of active trades"""
        self.monitoring = True
        
        while self.monitoring and self.trade_manager.active_trades:
            # Monitor trades
            updates = self.trade_manager.monitor_trades()
            
            # Print updates
            for update in updates:
                print(f"\n🔔 TRADE UPDATE - {update['time']}")
                print(f"{update['symbol']}: {update['status']}")
                print(f"P&L: {update['pnl']} ({update['roi']})")
            
            # Print active trades status
            if self.trade_manager.active_trades:
                print("\n📊 ACTIVE POSITIONS:")
                for trade in self.trade_manager.active_trades:
                    print(self.trade_manager.generate_trade_report(trade))
            
            # Wait before next update
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary"""
        total_pnl = sum(t.pnl for t in self.trade_manager.completed_trades)
        active_pnl = sum(t.pnl for t in self.trade_manager.active_trades)
        
        return {
            'active_trades': len(self.trade_manager.active_trades),
            'completed_trades': len(self.trade_manager.completed_trades),
            'total_realized_pnl': total_pnl,
            'total_unrealized_pnl': active_pnl,
            'win_rate': self._calculate_win_rate(),
            'average_profit': self._calculate_avg_profit(),
            'average_loss': self._calculate_avg_loss()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trade_manager.completed_trades:
            return 0
        
        wins = sum(1 for t in self.trade_manager.completed_trades if t.pnl > 0)
        return (wins / len(self.trade_manager.completed_trades)) * 100
    
    def _calculate_avg_profit(self) -> float:
        """Calculate average profit from winning trades"""
        profits = [t.pnl for t in self.trade_manager.completed_trades if t.pnl > 0]
        return np.mean(profits) if profits else 0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average loss from losing trades"""
        losses = [t.pnl for t in self.trade_manager.completed_trades if t.pnl < 0]
        return np.mean(losses) if losses else 0

# Example usage
if __name__ == "__main__":
    # Create enhanced trader
    trader = EnhancedMasterTrader()
    
    # Simulate market analysis output
    nifty_analysis = {
        'direction': 'BULLISH',
        'confidence': 90,
        'volatility': 18
    }
    
    banknifty_analysis = {
        'direction': 'BULLISH', 
        'confidence': 95,
        'volatility': 22
    }
    
    # Run analysis and trading
    async def main():
        print("📦 Using UPDATED lot sizes: NIFTY=75, BANKNIFTY=30")
        print("🚀 Starting Enhanced Trading System...")
        
        # Analyze and trade NIFTY
        await trader.analyze_and_trade('NIFTY', nifty_analysis)
        
        # Analyze and trade BANKNIFTY
        await trader.analyze_and_trade('BANKNIFTY', banknifty_analysis)
    
    # Run the trading system
    asyncio.run(main())