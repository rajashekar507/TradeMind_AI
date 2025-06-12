# TradeMind_AI Day 5: Portfolio Manager & Performance Tracker
# Enhanced with Balance Tracking and Movement Analysis

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

# Import centralized constants - FIXED DUPLICATION  
try:
    from config.constants import LOT_SIZES, get_lot_size
except ImportError:
    # Alternative path for different execution contexts
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config.constants import LOT_SIZES, get_lot_size

class PortfolioManager:
    """Advanced portfolio management and performance tracking"""
    
    def __init__(self):
        """Initialize portfolio manager"""
        print("📊 Initializing TradeMind_AI Portfolio Manager...")
        print("💰 Advanced P&L tracking and analytics")
        
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
        self.trades_file = "trades_database.json"
        self.performance_file = "performance_analytics.json"
        self.balance_history_file = "balance_history.json"
        
        # Load existing data
        self.trades_database = self.load_trades_database()
        self.performance_data = self.load_performance_data()
        
        # Portfolio metrics
        self.total_capital = float(os.getenv('TOTAL_CAPITAL', 100000))
        self.available_capital = self.fetch_current_balance()  # Fetch live balance
        self.allocated_capital = 0
        self.total_pnl = 0
        self.daily_pnl = 0
        
        # Risk management
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk
        self.max_single_trade_risk = 0.005  # 0.5% max per trade
        self.max_daily_loss = self.total_capital * 0.03  # 3% daily stop loss
        
        # REMOVED DUPLICATE LOT_SIZES - Now using centralized version from config.constants
        
        # Start automatic balance tracking
        self.auto_fetch_balance_schedule()
        
        print("✅ Portfolio Manager initialized!")
        print(f"📦 Using centralized lot sizes: NIFTY={get_lot_size('NIFTY')}, BANKNIFTY={get_lot_size('BANKNIFTY')}")
        print(f"💰 Current Balance: ₹{self.available_capital:,.2f}")
        self.send_portfolio_alert(f"📊 TradeMind_AI Portfolio Manager is ONLINE!\n💰 Balance: ₹{self.available_capital:,.2f}")

    def send_portfolio_alert(self, message: str) -> bool:
        """Send portfolio alerts to Telegram"""
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

    def fetch_current_balance(self) -> float:
        """Fetch current balance from Dhan"""
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

    def update_balance_after_trade(self, trade_type: str, amount: float) -> None:
        """Update balance after trade execution or closure"""
        try:
            # Fetch latest balance from broker
            old_balance = self.available_capital
            current_balance = self.fetch_current_balance()
            
            balance_update = {
                'timestamp': datetime.now().isoformat(),
                'trade_type': trade_type,
                'amount': amount,
                'balance_before': old_balance,
                'balance_after': current_balance,
                'change': current_balance - old_balance
            }
            
            # Update internal balance
            self.available_capital = current_balance
            
            # Log balance update
            self.log_balance_update(balance_update)
            
            # Send balance alert
            self.send_balance_update_alert(balance_update)
            
        except Exception as e:
            print(f"❌ Error updating balance: {e}")

    def log_balance_update(self, update: dict) -> None:
        """Log balance updates to file"""
        try:
            # Load existing log
            if os.path.exists(self.balance_history_file):
                with open(self.balance_history_file, 'r') as f:
                    balance_history = json.load(f)
            else:
                balance_history = []
            
            # Add new update
            balance_history.append(update)
            
            # Save updated log
            with open(self.balance_history_file, 'w') as f:
                json.dump(balance_history, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error logging balance: {e}")

    def send_balance_update_alert(self, update: dict) -> None:
        """Send balance update alert"""
        message = f"""
💰 <b>BALANCE UPDATE</b>

🔄 Type: {update['trade_type']}
💵 Amount: ₹{abs(update['amount']):,.2f}
📊 Balance Before: ₹{update['balance_before']:,.2f}
💰 Balance After: ₹{update['balance_after']:,.2f}
📈 Change: ₹{update['change']:+,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """
        self.send_portfolio_alert(message)

    def auto_fetch_balance_schedule(self) -> None:
        """Schedule automatic balance fetching"""
        def fetch_balance_periodically():
            while True:
                try:
                    # Fetch balance every 30 minutes during market hours
                    current_time = datetime.now().time()
                    market_open = datetime.strptime("09:15", "%H:%M").time()
                    market_close = datetime.strptime("15:30", "%H:%M").time()
                    
                    if market_open <= current_time <= market_close:
                        old_balance = self.available_capital
                        new_balance = self.fetch_current_balance()
                        
                        if abs(new_balance - old_balance) > 1:  # If balance changed
                            self.update_balance_after_trade("AUTO_SYNC", 0)
                    
                    time.sleep(1800)  # 30 minutes
                    
                except Exception as e:
                    print(f"❌ Auto balance fetch error: {e}")
                    time.sleep(300)  # Retry after 5 minutes
        
        # Start balance fetching thread
        balance_thread = threading.Thread(target=fetch_balance_periodically, daemon=True)
        balance_thread.start()
        print("✅ Automatic balance tracking enabled")

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

    # REMOVED DUPLICATE get_lot_size METHOD - Now using centralized version from config.constants

    def add_trade(self, trade_data: dict) -> None:
        """Add new trade to portfolio with balance update"""
        try:
            # Generate trade ID
            trade_id = f"TM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get correct lot size using centralized function - FIXED
            lot_size = trade_data.get("lot_size", get_lot_size(trade_data.get("symbol", "")))
            
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
            
            # Update balance after trade
            self.update_balance_after_trade(
                f"TRADE_OPENED_{trade_data['symbol']}",
                -complete_trade["capital_allocated"]
            )
            
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
                    
                    # Get lot size using centralized function - FIXED
                    lot_size = trade.get("lot_size", get_lot_size(trade["symbol"]))
                    
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
        """Close a trade and update portfolio with balance update"""
        try:
            for trade in self.trades_database["trades"]:
                if trade["trade_id"] == trade_id and trade["status"] == "OPEN":
                    # Update trade details
                    trade["exit_price"] = exit_price
                    trade["exit_time"] = datetime.now().isoformat()
                    trade["exit_reason"] = exit_reason
                    trade["status"] = "CLOSED"
                    
                    # Get lot size using centralized function - FIXED
                    lot_size = trade.get("lot_size", get_lot_size(trade["symbol"]))
                    
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
                    
                    # Update balance after trade closure
                    self.update_balance_after_trade(
                        f"TRADE_CLOSED_{trade['symbol']}",
                        trade["capital_allocated"] + final_pnl
                    )
                    
                    # Send trade close notification
                    self.send_trade_close_notification(trade)
                    
                    print(f"✅ Trade closed: {trade_id} | P&L: ₹{final_pnl:.2f}")
                    break
            
            self.save_trades_database()
            
        except Exception as e:
            print(f"❌ Error closing trade: {e}")

    def send_trade_confirmation(self, trade: dict) -> None:
        """Send trade confirmation to Telegram"""
        lot_size = trade.get('lot_size', get_lot_size(trade['symbol']))  # FIXED
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
        
        self.send_portfolio_alert(message)

    def send_trade_close_notification(self, trade: dict) -> None:
        """Send trade close notification"""
        profit_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
        lot_size = trade.get('lot_size', get_lot_size(trade['symbol']))  # FIXED
        
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
        
        self.send_portfolio_alert(message)

    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary"""
        try:
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
        """Send comprehensive daily portfolio report"""
        try:
            # Update balance before report
            self.available_capital = self.fetch_current_balance()
            
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

📦 <b>Centralized Lot Sizes:</b>
NIFTY: {get_lot_size('NIFTY')} units
BANKNIFTY: {get_lot_size('BANKNIFTY')} units

💰 <b>Live Balance:</b> ₹{self.available_capital:,.2f}

⏰ Generated: {datetime.now().strftime('%H:%M:%S')}
            """
            
            self.send_portfolio_alert(message)
            
        except Exception as e:
            print(f"❌ Error sending portfolio report: {e}")

    def check_risk_limits(self) -> bool:
        """Check if risk limits are being followed"""
        try:
            # Check daily loss limit
            if abs(self.daily_pnl) >= self.max_daily_loss:
                self.send_portfolio_alert("🚨 DAILY LOSS LIMIT REACHED! Trading paused.")
                return False
            
            # Check portfolio risk
            portfolio_risk = (self.allocated_capital / self.total_capital)
            if portfolio_risk > self.max_portfolio_risk:
                self.send_portfolio_alert("⚠️ Portfolio risk limit exceeded!")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking risk limits: {e}")
            return True

    def simulate_paper_trade(self, recommendation: dict) -> None:
        """Simulate a paper trade based on recommendation"""
        try:
            # Update balance before trade
            self.available_capital = self.fetch_current_balance()
            
            # Get correct lot size based on symbol using centralized function - FIXED
            lot_size = get_lot_size(recommendation['symbol'])
            
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
                self.send_portfolio_alert(f"❌ Insufficient balance for {recommendation['symbol']} - Need ₹{capital_per_lot:,.0f}")
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
            
            # Add trade to portfolio
            self.add_trade(trade_data)
            
            print(f"📊 Paper trade simulated: {trade_data['symbol']} {trade_data['strike']} {trade_data['option_type']}")
            print(f"   Lot Size: {lot_size}, Quantity: {quantity} lots")
            print(f"   Total Units: {quantity * lot_size}")
            print(f"   Capital Allocated: ₹{capital_allocated:,.2f}")
            print(f"   Movement Score: {recommendation.get('movement_score', 0):.1f}/100")
            
        except Exception as e:
            print(f"❌ Error simulating paper trade: {e}")

    def get_open_positions(self) -> list:
        """Get list of open positions"""
        return [t for t in self.trades_database["trades"] if t["status"] == "OPEN"]

    def display_portfolio(self) -> None:
        """Display portfolio in readable format"""
        print("\n📊 PORTFOLIO OVERVIEW")
        print("="*60)
        
        summary = self.get_portfolio_summary()
        
        print(f"💰 Available Capital: ₹{summary['available_capital']:,.2f}")
        print(f"🔒 Allocated Capital: ₹{summary['allocated_capital']:,.2f}")
        print(f"📈 Total P&L: ₹{summary['total_pnl']:,.2f}")
        print(f"📊 Portfolio Return: {summary['portfolio_return']:.2f}%")
        print(f"🎯 Open Positions: {summary['open_positions']}")
        print(f"🏆 Win Rate: {summary['win_rate']:.1f}%")
        print(f"📦 Lot Sizes: NIFTY={get_lot_size('NIFTY')}, BANKNIFTY={get_lot_size('BANKNIFTY')}")
        
        # Show open positions
        open_positions = self.get_open_positions()
        if open_positions:
            print(f"\n📋 OPEN POSITIONS:")
            for i, pos in enumerate(open_positions, 1):
                print(f"  {i}. {pos['symbol']} {pos['strike']} {pos['option_type']} - ₹{pos['current_price']:.2f} (P&L: ₹{pos['pnl']:.2f})")
        else:
            print("\n📋 No open positions")
        
        print("="*60)

def main():
    """Main function to run portfolio manager"""
    print("📊 TradeMind_AI Portfolio Manager - Enhanced Version")
    print("💰 Advanced P&L Tracking with Balance Monitoring")
    print(f"📦 Using CENTRALIZED lot sizes: NIFTY={get_lot_size('NIFTY')}, BANKNIFTY={get_lot_size('BANKNIFTY')}")
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
        print("💰 Balance tracking active - updates every 30 minutes")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()