# TradeMind_AI: Portfolio Manager & Performance Tracker
# Enhanced with Shared Balance Utility - NO CODE DUPLICATION

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv
import requests
import threading
import time

# Import shared balance utility - ELIMINATES CODE DUPLICATION
from src.utils.balance_utils import (
    balance_manager,
    fetch_current_balance,
    get_detailed_balance_info,
    send_balance_alert,
    update_balance_after_trade
)

class PortfolioManager:
    """Advanced portfolio management and performance tracking"""
    
    def __init__(self):
        """Initialize portfolio manager"""
        print("📊 Initializing TradeMind_AI Portfolio Manager...")
        print("💰 Using shared balance utility - no code duplication!")
        
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
        
        # Portfolio tracking
        self.trades_file = "data/trades_database.json"
        self.performance_file = "data/performance_analytics.json"
        
        # Load existing data
        self.trades_database = self.load_trades_database()
        self.performance_data = self.load_performance_data()
        
        # Portfolio metrics
        self.total_capital = float(os.getenv('TOTAL_CAPITAL', 100000))
        
        # Use shared balance utility instead of duplicate code
        self.available_capital = fetch_current_balance()
        
        self.allocated_capital = 0
        self.total_pnl = 0
        self.daily_pnl = 0
        
        # Risk management
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk
        self.max_single_trade_risk = 0.005  # 0.5% max per trade
        self.max_daily_loss = self.total_capital * 0.03  # 3% daily stop loss
        
        # UPDATED LOT SIZES AS PER SEBI GUIDELINES (June 2025)
        self.lot_sizes = {
            'NIFTY': 75,      # Changed from 25 on Dec 26, 2024
            'BANKNIFTY': 30   # Changed from 15 on Dec 24, 2024
        }
        
        # Start automatic balance tracking using shared utility
        self.auto_fetch_balance_schedule()
        
        print("✅ Portfolio Manager initialized!")
        print(f"📦 Using updated lot sizes: NIFTY={self.lot_sizes['NIFTY']}, BANKNIFTY={self.lot_sizes['BANKNIFTY']}")
        print(f"💰 Current Balance: ₹{self.available_capital:,.2f}")
        
        # Use shared utility for alert
        send_balance_alert(f"📊 TradeMind_AI Portfolio Manager is ONLINE!\n💰 Balance: ₹{self.available_capital:,.2f}")

    def auto_fetch_balance_schedule(self) -> None:
        """Schedule automatic balance fetching using shared utility"""
        def fetch_balance_periodically():
            while True:
                try:
                    # Fetch balance every 30 minutes during market hours
                    current_time = datetime.now().time()
                    market_open = datetime.strptime("09:15", "%H:%M").time()
                    market_close = datetime.strptime("15:30", "%H:%M").time()
                    
                    if market_open <= current_time <= market_close:
                        old_balance = self.available_capital
                        
                        # Use shared balance utility
                        new_balance = fetch_current_balance()
                        
                        if abs(new_balance - old_balance) > 1:  # If balance changed
                            self.available_capital = new_balance
                            # Use shared utility for balance update
                            update_balance_after_trade("AUTO_SYNC", 0)
                    
                    time.sleep(1800)  # 30 minutes
                    
                except Exception as e:
                    print(f"❌ Auto balance fetch error: {e}")
                    time.sleep(300)  # Retry after 5 minutes
        
        # Start balance fetching thread
        balance_thread = threading.Thread(target=fetch_balance_periodically, daemon=True)
        balance_thread.start()
        print("✅ Automatic balance tracking enabled (using shared utility)")

    def load_trades_database(self) -> dict:
        """Load trades database from file"""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "trades": [],
                    "summary": {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0,
                        "best_trade": 0,
                        "worst_trade": 0
                    }
                }
        except Exception as e:
            print(f"❌ Error loading trades database: {e}")
            return {"trades": [], "summary": {}}

    def save_trades_database(self) -> None:
        """Save trades database to file"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades_database, f, indent=2, default=str)
            print("💾 Trades database saved")
        except Exception as e:
            print(f"❌ Error saving trades database: {e}")

    def load_performance_data(self) -> dict:
        """Load performance analytics"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "daily_performance": [],
                    "monthly_performance": [],
                    "strategy_performance": {},
                    "risk_metrics": {}
                }
        except Exception as e:
            print(f"❌ Error loading performance data: {e}")
            return {}

    def save_performance_data(self) -> None:
        """Save performance analytics"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
            print("📈 Performance data saved")
        except Exception as e:
            print(f"❌ Error saving performance data: {e}")

    def get_lot_size(self, symbol: str) -> int:
        """Get correct lot size for symbol"""
        if 'BANKNIFTY' in symbol.upper():
            return self.lot_sizes['BANKNIFTY']
        elif 'NIFTY' in symbol.upper():
            return self.lot_sizes['NIFTY']
        else:
            # Default to NIFTY lot size
            return self.lot_sizes['NIFTY']

    def add_trade(self, trade_data: dict) -> None:
        """Add new trade to portfolio with shared balance utility"""
        try:
            # Generate trade ID
            trade_id = f"TM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get correct lot size
            lot_size = trade_data.get("lot_size", self.get_lot_size(trade_data.get("symbol", "")))
            
            # Complete trade data
            complete_trade = {
                "trade_id": trade_id,
                "timestamp": datetime.now().isoformat(),
                "symbol": trade_data.get("symbol"),
                "strike": trade_data.get("strike"),
                "option_type": trade_data.get("option_type"),
                "action": trade_data.get("action", "BUY"),
                "quantity": trade_data.get("quantity", 1),
                "lot_size": lot_size,
                "entry_price": trade_data.get("entry_price"),
                "current_price": trade_data.get("current_price", trade_data.get("entry_price")),
                "target_price": trade_data.get("target_price"),
                "stop_loss": trade_data.get("stop_loss"),
                "status": "OPEN",
                "pnl": 0,
                "pnl_percentage": 0,
                "capital_allocated": trade_data.get("capital_allocated"),
                "ai_confidence": trade_data.get("ai_confidence"),
                "strategy": trade_data.get("strategy", "Smart_AI"),
                "expiry": trade_data.get("expiry"),
                "days_to_expiry": trade_data.get("days_to_expiry"),
                "entry_reason": trade_data.get("entry_reason", "AI Signal"),
                "movement_score": trade_data.get("movement_score", 0),
                "expected_move": trade_data.get("expected_move", 0)
            }
            
            # Add to database
            self.trades_database["trades"].append(complete_trade)
            
            # Update allocated capital
            self.allocated_capital += complete_trade["capital_allocated"]
            
            # Save database
            self.save_trades_database()
            
            # Use shared balance utility for balance update
            balance_update = update_balance_after_trade(
                f"TRADE_OPENED_{trade_data['symbol']}",
                -complete_trade["capital_allocated"]
            )
            
            # Update local balance from shared utility response
            if 'new_balance' in balance_update:
                self.available_capital = balance_update['new_balance']
            
            # Send trade confirmation
            self.send_trade_confirmation(complete_trade)
            
            print(f"✅ Trade added: {trade_id}")
            
        except Exception as e:
            print(f"❌ Error adding trade: {e}")

    def update_trade_price(self, trade_id: str, current_price: float) -> None:
        """Update current price and P&L for a trade"""
        try:
            for trade in self.trades_database["trades"]:
                if trade["trade_id"] == trade_id and trade["status"] == "OPEN":
                    # Update current price
                    trade["current_price"] = current_price
                    
                    # Get lot size
                    lot_size = trade.get("lot_size", self.get_lot_size(trade["symbol"]))
                    
                    # Calculate P&L
                    if trade["action"] == "BUY":
                        pnl = (current_price - trade["entry_price"]) * trade["quantity"] * lot_size
                    else:
                        pnl = (trade["entry_price"] - current_price) * trade["quantity"] * lot_size
                    
                    trade["pnl"] = pnl
                    trade["pnl_percentage"] = (pnl / trade["capital_allocated"]) * 100
                    
                    # Check if target or stop loss hit
                    if trade["action"] == "BUY":
                        if current_price >= trade["target_price"]:
                            self.close_trade(trade_id, current_price, "TARGET_HIT")
                        elif current_price <= trade["stop_loss"]:
                            self.close_trade(trade_id, current_price, "STOP_LOSS")
                    
                    break
            
            self.save_trades_database()
            
        except Exception as e:
            print(f"❌ Error updating trade price: {e}")

    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> None:
        """Close a trade and update portfolio with shared balance utility"""
        try:
            for trade in self.trades_database["trades"]:
                if trade["trade_id"] == trade_id and trade["status"] == "OPEN":
                    # Update trade details
                    trade["exit_price"] = exit_price
                    trade["exit_time"] = datetime.now().isoformat()
                    trade["exit_reason"] = exit_reason
                    trade["status"] = "CLOSED"
                    
                    # Get lot size
                    lot_size = trade.get("lot_size", self.get_lot_size(trade["symbol"]))
                    
                    # Calculate final P&L
                    if trade["action"] == "BUY":
                        final_pnl = (exit_price - trade["entry_price"]) * trade["quantity"] * lot_size
                    else:
                        final_pnl = (trade["entry_price"] - exit_price) * trade["quantity"] * lot_size
                    
                    trade["pnl"] = final_pnl
                    trade["pnl_percentage"] = (final_pnl / trade["capital_allocated"]) * 100
                    
                    # Update portfolio metrics
                    self.total_pnl += final_pnl
                    self.daily_pnl += final_pnl
                    self.allocated_capital -= trade["capital_allocated"]
                    
                    # Update summary
                    summary = self.trades_database["summary"]
                    summary["total_trades"] = summary.get("total_trades", 0) + 1
                    summary["total_pnl"] = summary.get("total_pnl", 0) + final_pnl
                    
                    if final_pnl > 0:
                        summary["winning_trades"] = summary.get("winning_trades", 0) + 1
                    else:
                        summary["losing_trades"] = summary.get("losing_trades", 0) + 1
                    
                    # Track best and worst trades
                    summary["best_trade"] = max(summary.get("best_trade", 0), final_pnl)
                    summary["worst_trade"] = min(summary.get("worst_trade", 0), final_pnl)
                    
                    # Use shared balance utility for balance update
                    balance_update = update_balance_after_trade(
                        f"TRADE_CLOSED_{trade['symbol']}",
                        trade["capital_allocated"] + final_pnl
                    )
                    
                    # Update local balance from shared utility response
                    if 'new_balance' in balance_update:
                        self.available_capital = balance_update['new_balance']
                    
                    # Send trade close notification
                    self.send_trade_close_notification(trade)
                    
                    print(f"✅ Trade closed: {trade_id} | P&L: ₹{final_pnl:.2f}")
                    break
            
            self.save_trades_database()
            
        except Exception as e:
            print(f"❌ Error closing trade: {e}")

    def send_trade_confirmation(self, trade: dict) -> None:
        """Send trade confirmation using shared balance utility"""
        lot_size = trade.get('lot_size', self.get_lot_size(trade['symbol']))
        total_quantity = trade['quantity'] * lot_size
        
        message = f"""
📊 <b>TRADE CONFIRMATION</b>

🆔 Trade ID: {trade['trade_id']}
📊 Position: {trade['symbol']} {trade['strike']:.0f} {trade['option_type']}
💰 Entry Price: ₹{trade['entry_price']:.2f}
📦 Quantity: {trade['quantity']} lots × {lot_size} = {total_quantity} units
💵 Capital: ₹{trade['capital_allocated']:.0f}

🎯 <b>TARGETS:</b>
🟢 Target: ₹{trade['target_price']:.2f}
🔴 Stop Loss: ₹{trade['stop_loss']:.2f}
📅 Expiry: {trade['expiry']}

🚀 Movement Score: {trade.get('movement_score', 0):.1f}/100
🧠 AI Confidence: {trade['ai_confidence']:.1f}%
📈 Strategy: {trade['strategy']}
💰 Current Balance: ₹{self.available_capital:,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """
        
        # Use shared balance utility for sending alert
        send_balance_alert(message)

    def send_trade_close_notification(self, trade: dict) -> None:
        """Send trade close notification using shared balance utility"""
        profit_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
        lot_size = trade.get('lot_size', self.get_lot_size(trade['symbol']))
        
        message = f"""
{profit_emoji} <b>TRADE CLOSED</b>

🆔 {trade['trade_id']}
📊 {trade['symbol']} {trade['strike']:.0f} {trade['option_type']}
💰 Entry: ₹{trade['entry_price']:.2f}
💰 Exit: ₹{trade['exit_price']:.2f}
📦 Quantity: {trade['quantity']} lots × {lot_size}

{profit_emoji} <b>P&L: ₹{trade['pnl']:.2f} ({trade['pnl_percentage']:.1f}%)</b>
🔄 Reason: {trade['exit_reason']}

📊 <b>Updated Portfolio:</b>
💵 Total P&L: ₹{self.total_pnl:.2f}
💰 Current Balance: ₹{self.available_capital:.0f}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """
        
        # Use shared balance utility for sending alert
        send_balance_alert(message)

    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary using shared balance utility"""
        try:
            # Update balance using shared utility
            self.available_capital = fetch_current_balance()
            
            open_trades = [t for t in self.trades_database["trades"] if t["status"] == "OPEN"]
            closed_trades = [t for t in self.trades_database["trades"] if t["status"] == "CLOSED"]
            
            # Calculate metrics
            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t["pnl"] > 0])
            losing_trades = len([t for t in closed_trades if t["pnl"] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Average metrics
            avg_win = np.mean([t["pnl"] for t in closed_trades if t["pnl"] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t["pnl"] for t in closed_trades if t["pnl"] < 0]) if losing_trades > 0 else 0
            
            # Risk metrics
            portfolio_value = self.available_capital + self.allocated_capital
            portfolio_return = (self.total_pnl / self.total_capital) * 100
            
            return {
                "portfolio_value": portfolio_value,
                "available_capital": self.available_capital,
                "allocated_capital": self.allocated_capital,
                "total_pnl": self.total_pnl,
                "daily_pnl": self.daily_pnl,
                "portfolio_return": portfolio_return,
                "open_positions": len(open_trades),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                "best_trade": max([t["pnl"] for t in closed_trades]) if closed_trades else 0,
                "worst_trade": min([t["pnl"] for t in closed_trades]) if closed_trades else 0
            }
            
        except Exception as e:
            print(f"❌ Error calculating portfolio summary: {e}")
            return {}

    def send_daily_portfolio_report(self) -> None:
        """Send comprehensive daily portfolio report using shared balance utility"""
        try:
            # Get detailed balance info using shared utility
            balance_info = get_detailed_balance_info()
            
            summary = self.get_portfolio_summary()
            
            message = f"""
📊 <b>DAILY PORTFOLIO REPORT</b>

📅 Date: {datetime.now().strftime('%Y-%m-%d')}

💰 <b>Portfolio Overview:</b>
💵 Portfolio Value: ₹{summary['portfolio_value']:.0f}
💰 Available Capital: ₹{summary['available_capital']:.0f}
🔒 Allocated Capital: ₹{summary['allocated_capital']:.0f}
📈 Total P&L: ₹{summary['total_pnl']:.2f}
📊 Portfolio Return: {summary['portfolio_return']:.2f}%

📊 <b>Trading Performance:</b>
🎯 Open Positions: {summary['open_positions']}
📈 Total Trades: {summary['total_trades']}
🟢 Winning Trades: {summary['winning_trades']}
🔴 Losing Trades: {summary['losing_trades']}
🏆 Win Rate: {summary['win_rate']:.1f}%

💰 <b>Trade Analytics:</b>
🟢 Avg Win: ₹{summary['avg_win']:.2f}
🔴 Avg Loss: ₹{summary['avg_loss']:.2f}
⚡ Profit Factor: {summary['profit_factor']:.2f}
🚀 Best Trade: ₹{summary['best_trade']:.2f}
💥 Worst Trade: ₹{summary['worst_trade']:.2f}

📦 <b>Lot Sizes:</b>
NIFTY: {self.lot_sizes['NIFTY']} units
BANKNIFTY: {self.lot_sizes['BANKNIFTY']} units

💰 <b>Live Balance Info:</b>
💵 Available: ₹{balance_info['available_balance']:,.2f}
🔒 Used Margin: ₹{balance_info['used_margin']:,.2f}
📊 Mode: {balance_info['mode']}

⏰ Generated: {datetime.now().strftime('%H:%M:%S')}
            """
            
            # Use shared balance utility for sending report
            send_balance_alert(message)
            
        except Exception as e:
            print(f"❌ Error sending portfolio report: {e}")

    def check_risk_limits(self) -> bool:
        """Check if risk limits are being followed"""
        try:
            # Check daily loss limit
            if abs(self.daily_pnl) >= self.max_daily_loss:
                send_balance_alert("🚨 DAILY LOSS LIMIT REACHED! Trading paused.")
                return False
            
            # Check portfolio risk
            portfolio_risk = (self.allocated_capital / self.total_capital)
            if portfolio_risk > self.max_portfolio_risk:
                send_balance_alert("⚠️ Portfolio risk limit exceeded!")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking risk limits: {e}")
            return True

    def simulate_paper_trade(self, recommendation: dict) -> None:
        """Simulate a paper trade based on recommendation using shared balance utility"""
        try:
            # Update balance using shared utility
            self.available_capital = fetch_current_balance()
            
            # Get correct lot size based on symbol
            if 'NIFTY' in recommendation['symbol']:
                if 'BANKNIFTY' in recommendation['symbol']:
                    lot_size = self.lot_sizes['BANKNIFTY']  # 30
                else:
                    lot_size = self.lot_sizes['NIFTY']  # 75
            else:
                lot_size = recommendation.get('lot_size', self.lot_sizes['NIFTY'])
            
            # Calculate position size based on risk
            risk_amount = self.total_capital * self.max_single_trade_risk
            entry_price = recommendation['current_price']
            stop_loss = recommendation['stop_loss']
            
            # Calculate capital per lot
            capital_per_lot = entry_price * lot_size
            
            # Check affordability
            max_affordable_lots = int(self.available_capital * 0.8 / capital_per_lot) if capital_per_lot > 0 else 0
            
            if max_affordable_lots < 1:
                print(f"❌ Cannot afford {recommendation['symbol']} - Need ₹{capital_per_lot:,.0f} per lot")
                send_balance_alert(f"❌ Insufficient balance for {recommendation['symbol']} - Need ₹{capital_per_lot:,.0f}")
                return
            
            # Calculate quantity based on risk
            risk_per_share = abs(entry_price - stop_loss)
            risk_per_lot = risk_per_share * lot_size
            quantity = int(risk_amount / risk_per_lot) if risk_per_lot > 0 else 1
            
            # Take minimum of risk-based and affordable lots
            quantity = min(quantity, max_affordable_lots, 10)  # Max 10 lots
            
            capital_allocated = entry_price * quantity * lot_size
            
            # Create trade data
            trade_data = {
                "symbol": recommendation['symbol'],
                "strike": recommendation['strike'],
                "option_type": recommendation['option_type'],
                "action": "BUY",
                "quantity": quantity,
                "lot_size": lot_size,
                "entry_price": entry_price,
                "target_price": recommendation['target_price'],
                "stop_loss": stop_loss,
                "capital_allocated": capital_allocated,
                "ai_confidence": recommendation.get('confidence', 80),
                "strategy": "Exact_Strike_AI",
                "expiry": recommendation['expiry'],
                "days_to_expiry": recommendation.get('days_to_expiry', 7),
                "entry_reason": f"AI Score: {recommendation.get('score', 80)}/100",
                "movement_score": recommendation.get('movement_score', 0),
                "expected_move": recommendation.get('expected_daily_move', 0)
            }
            
            # Add trade to portfolio (will use shared balance utility)
            self.add_trade(trade_data)
            
            print(f"📊 Paper trade simulated: {trade_data['symbol']} {trade_data['strike']} {trade_data['option_type']}")
            print(f"   Lot Size: {lot_size}, Quantity: {quantity} lots")
            print(f"   Total Units: {quantity * lot_size}")
            print(f"   Capital Allocated: ₹{capital_allocated:,.2f}")
            print(f"   Movement Score: {recommendation.get('movement_score', 0):.1f}/100")
            
        except Exception as e:
            print(f"❌ Error simulating paper trade: {e}")

def main():
    """Main function to run portfolio manager"""
    print("📊 TradeMind_AI Portfolio Manager - FIXED VERSION")
    print("💰 Using Shared Balance Utility - NO CODE DUPLICATION")
    print("📦 Using UPDATED lot sizes: NIFTY=75, BANKNIFTY=30")
    print("🚀 Movement Potential Analysis Integrated")
    
    try:
        # Create portfolio manager
        portfolio_manager = PortfolioManager()
        
        # Generate and send daily report
        portfolio_manager.send_daily_portfolio_report()
        
        # Example: Simulate some paper trades
        print("\n🧪 Running portfolio management demo...")
        
        # Example recommendations with movement scores
        example_recommendations = [
            {
                "symbol": "BANKNIFTY",
                "strike": 57000,
                "option_type": "CE",
                "current_price": 85.50,
                "target_price": 111.15,
                "stop_loss": 68.40,
                "confidence": 85,
                "score": 78,
                "movement_score": 82.5,
                "expected_daily_move": 12.30,
                "movement_percentage": 14.4,
                "expiry": "2025-06-26",
                "days_to_expiry": 16
            },
            {
                "symbol": "NIFTY",
                "strike": 25500,
                "option_type": "CE",
                "current_price": 45.25,
                "target_price": 58.83,
                "stop_loss": 36.20,
                "confidence": 75,
                "score": 65,
                "movement_score": 68.3,
                "expected_daily_move": 5.40,
                "movement_percentage": 11.9,
                "expiry": "2025-06-12",
                "days_to_expiry": 2
            }
        ]
        
        # Simulate paper trades
        for rec in example_recommendations:
            portfolio_manager.simulate_paper_trade(rec)
            
        print("\n✅ Portfolio Manager demo completed!")
        print("📊 Check your Telegram for portfolio reports!")
        print("💰 Balance tracking active - using shared utility!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()