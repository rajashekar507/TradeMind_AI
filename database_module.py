"""
TradeMind_AI: Database Module
Handles trade history, performance tracking, and analytics
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path: str = "trademind_ai.db"):
        """Initialize Database Manager"""
        print("🗄️ Initializing Database Manager...")
        
        self.db_path = db_path
        self.init_database()
        
        print("✅ Database Manager ready!")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    option_type TEXT,
                    strike_price REAL,
                    expiry_date DATE,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    status TEXT NOT NULL,
                    pnl REAL,
                    pnl_percent REAL,
                    strategy_name TEXT,
                    confidence_score REAL,
                    stop_loss REAL,
                    target REAL,
                    fees REAL DEFAULT 0,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Positions table (current open positions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT UNIQUE NOT NULL,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_percent REAL,
                    stop_loss REAL,
                    target REAL,
                    status TEXT NOT NULL,
                    last_updated TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                )
            ''')
            
            # Daily performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    starting_capital REAL NOT NULL,
                    ending_capital REAL NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    gross_pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    win_rate REAL,
                    largest_win REAL,
                    largest_loss REAL,
                    max_drawdown REAL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    executed_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    avg_confidence REAL,
                    success_rate REAL,
                    UNIQUE(strategy_name, date)
                )
            ''')
            
            # Market conditions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    volatility REAL,
                    rsi REAL,
                    market_sentiment TEXT,
                    vix_level REAL,
                    global_market_bias TEXT,
                    notes TEXT
                )
            ''')
            
            # Audit log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    user_action TEXT,
                    details TEXT,
                    ip_address TEXT,
                    status TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON daily_performance(date)')
            
            conn.commit()
    
    # Trade Management Methods
    def insert_trade(self, trade_data: Dict) -> str:
        """Insert a new trade"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            trade_id = trade_data.get('trade_id', f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            cursor.execute('''
                INSERT INTO trades (
                    trade_id, symbol, trade_type, option_type, strike_price,
                    expiry_date, entry_time, entry_price, quantity, status,
                    strategy_name, confidence_score, stop_loss, target, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trade_data['symbol'],
                trade_data['trade_type'],
                trade_data.get('option_type'),
                trade_data.get('strike_price'),
                trade_data.get('expiry_date'),
                trade_data['entry_time'],
                trade_data['entry_price'],
                trade_data['quantity'],
                trade_data.get('status', 'OPEN'),
                trade_data.get('strategy_name'),
                trade_data.get('confidence_score'),
                trade_data.get('stop_loss'),
                trade_data.get('target'),
                trade_data.get('notes')
            ))
            
            # Also insert into positions table if trade is open
            if trade_data.get('status', 'OPEN') == 'OPEN':
                position_id = f"POS_{trade_id}"
                cursor.execute('''
                    INSERT INTO positions (
                        position_id, trade_id, symbol, quantity, entry_price,
                        current_price, stop_loss, target, status, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_id,
                    trade_id,
                    trade_data['symbol'],
                    trade_data['quantity'],
                    trade_data['entry_price'],
                    trade_data['entry_price'],  # Initially same as entry
                    trade_data.get('stop_loss'),
                    trade_data.get('target'),
                    'OPEN',
                    datetime.now()
                ))
            
            conn.commit()
            
            # Log the trade
            self.log_event('TRADE_CREATED', f"New trade: {trade_id}", 
                          json.dumps({'symbol': trade_data['symbol'], 
                                     'quantity': trade_data['quantity']}))
            
            return trade_id
    
    def update_trade_exit(self, trade_id: str, exit_price: float, 
                         exit_time: datetime = None, fees: float = 0) -> bool:
        """Update trade with exit details"""
        if exit_time is None:
            exit_time = datetime.now()
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get trade details
            cursor.execute('SELECT * FROM trades WHERE trade_id = ?', (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                return False
            
            # Calculate P&L
            pnl = (exit_price - trade['entry_price']) * trade['quantity'] - fees
            pnl_percent = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
            
            # Update trade
            cursor.execute('''
                UPDATE trades 
                SET exit_time = ?, exit_price = ?, status = ?, 
                    pnl = ?, pnl_percent = ?, fees = ?
                WHERE trade_id = ?
            ''', (exit_time, exit_price, 'CLOSED', pnl, pnl_percent, fees, trade_id))
            
            # Remove from positions table
            cursor.execute('DELETE FROM positions WHERE trade_id = ?', (trade_id,))
            
            conn.commit()
            
            # Log the exit
            self.log_event('TRADE_CLOSED', f"Trade closed: {trade_id}", 
                          json.dumps({'pnl': pnl, 'pnl_percent': pnl_percent}))
            
            return True
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.*, t.strategy_name, t.confidence_score
                FROM positions p
                JOIN trades t ON p.trade_id = t.trade_id
                WHERE p.status = 'OPEN'
                ORDER BY p.last_updated DESC
            ''')
            
            positions = []
            for row in cursor.fetchall():
                positions.append(dict(row))
            
            return positions
    
    def update_position_price(self, position_id: str, current_price: float):
        """Update current price and unrealized P&L for a position"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get position details
            cursor.execute('SELECT * FROM positions WHERE position_id = ?', (position_id,))
            position = cursor.fetchone()
            
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                unrealized_pnl_percent = ((current_price - position['entry_price']) / 
                                         position['entry_price']) * 100
                
                cursor.execute('''
                    UPDATE positions 
                    SET current_price = ?, unrealized_pnl = ?, 
                        unrealized_pnl_percent = ?, last_updated = ?
                    WHERE position_id = ?
                ''', (current_price, unrealized_pnl, unrealized_pnl_percent, 
                      datetime.now(), position_id))
                
                conn.commit()
    
    # Performance Analytics Methods
    def get_trade_history(self, symbol: str = None, start_date: datetime = None, 
                         end_date: datetime = None, limit: int = None) -> pd.DataFrame:
        """Get trade history with filters"""
        query = 'SELECT * FROM trades WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if start_date:
            query += ' AND entry_time >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND entry_time <= ?'
            params.append(end_date)
        
        query += ' ORDER BY entry_time DESC'
        
        if limit:
            query += f' LIMIT {limit}'
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        return df
    
    def calculate_daily_performance(self, date: datetime = None) -> Dict:
        """Calculate and store daily performance metrics"""
        if date is None:
            date = datetime.now().date()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all trades for the day
            cursor.execute('''
                SELECT * FROM trades 
                WHERE DATE(entry_time) = ? OR DATE(exit_time) = ?
            ''', (date, date))
            
            trades = cursor.fetchall()
            
            # Calculate metrics
            total_trades = len([t for t in trades if t['exit_time'] and 
                               datetime.fromisoformat(t['exit_time']).date() == date])
            winning_trades = len([t for t in trades if t['pnl'] and t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] and t['pnl'] < 0])
            
            gross_pnl = sum(t['pnl'] for t in trades if t['pnl']) or 0
            fees = sum(t['fees'] for t in trades if t['fees']) or 0
            net_pnl = gross_pnl - fees
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            largest_win = max((t['pnl'] for t in trades if t['pnl'] and t['pnl'] > 0), default=0)
            largest_loss = min((t['pnl'] for t in trades if t['pnl'] and t['pnl'] < 0), default=0)
            
            # Calculate starting capital (simplified - would need account integration)
            starting_capital = 1000000  # Default
            ending_capital = starting_capital + net_pnl
            
            # Store daily performance
            cursor.execute('''
                INSERT OR REPLACE INTO daily_performance (
                    date, starting_capital, ending_capital, total_trades,
                    winning_trades, losing_trades, gross_pnl, fees, net_pnl,
                    win_rate, largest_win, largest_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                date, starting_capital, ending_capital, total_trades,
                winning_trades, losing_trades, gross_pnl, fees, net_pnl,
                win_rate, largest_win, largest_loss
            ))
            
            conn.commit()
            
            return {
                'date': date,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'net_pnl': net_pnl,
                'win_rate': win_rate
            }
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for specified days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        with self.get_connection() as conn:
            # Get daily performance data
            df_daily = pd.read_sql_query('''
                SELECT * FROM daily_performance 
                WHERE date >= ? AND date <= ?
                ORDER BY date
            ''', conn, params=(start_date, end_date))
            
            # Get all trades in period
            df_trades = pd.read_sql_query('''
                SELECT * FROM trades 
                WHERE DATE(entry_time) >= ? AND DATE(entry_time) <= ?
                AND status = 'CLOSED'
            ''', conn, params=(start_date, end_date))
        
        if df_daily.empty:
            return {'error': 'No performance data available'}
        
        # Calculate summary metrics
        total_trading_days = len(df_daily)
        total_pnl = df_daily['net_pnl'].sum()
        avg_daily_pnl = df_daily['net_pnl'].mean()
        
        winning_days = len(df_daily[df_daily['net_pnl'] > 0])
        losing_days = len(df_daily[df_daily['net_pnl'] < 0])
        
        best_day = df_daily.loc[df_daily['net_pnl'].idxmax()] if not df_daily.empty else None
        worst_day = df_daily.loc[df_daily['net_pnl'].idxmin()] if not df_daily.empty else None
        
        # Calculate streak
        current_streak = self._calculate_streak(df_daily)
        
        # Strategy performance
        strategy_performance = self._get_strategy_performance(start_date, end_date)
        
        summary = {
            'period': f'{days} days',
            'start_date': start_date,
            'end_date': end_date,
            'total_trading_days': total_trading_days,
            'total_trades': len(df_trades),
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'winning_days': winning_days,
            'losing_days': losing_days,
            'win_rate_days': (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0,
            'best_day': {
                'date': best_day['date'] if best_day is not None else None,
                'pnl': best_day['net_pnl'] if best_day is not None else 0
            },
            'worst_day': {
                'date': worst_day['date'] if worst_day is not None else None,
                'pnl': worst_day['net_pnl'] if worst_day is not None else 0
            },
            'current_streak': current_streak,
            'strategy_performance': strategy_performance
        }
        
        return summary
    
    def _calculate_streak(self, df_daily: pd.DataFrame) -> Dict:
        """Calculate current winning/losing streak"""
        if df_daily.empty:
            return {'type': 'none', 'days': 0}
        
        # Sort by date descending
        df_sorted = df_daily.sort_values('date', ascending=False)
        
        streak_type = None
        streak_count = 0
        
        for _, row in df_sorted.iterrows():
            if row['net_pnl'] > 0:
                if streak_type == 'winning' or streak_type is None:
                    streak_type = 'winning'
                    streak_count += 1
                else:
                    break
            elif row['net_pnl'] < 0:
                if streak_type == 'losing' or streak_type is None:
                    streak_type = 'losing'
                    streak_count += 1
                else:
                    break
            else:
                # Break even day
                break
        
        return {'type': streak_type or 'none', 'days': streak_count}
    
    def _get_strategy_performance(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get performance by strategy"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    strategy_name,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(confidence_score) as avg_confidence
                FROM trades
                WHERE DATE(entry_time) >= ? AND DATE(entry_time) <= ?
                AND status = 'CLOSED' AND strategy_name IS NOT NULL
                GROUP BY strategy_name
                ORDER BY total_pnl DESC
            ''', (start_date, end_date))
            
            strategies = []
            for row in cursor.fetchall():
                win_rate = (row['winning_trades'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
                strategies.append({
                    'name': row['strategy_name'],
                    'total_trades': row['total_trades'],
                    'win_rate': win_rate,
                    'total_pnl': row['total_pnl'],
                    'avg_pnl': row['avg_pnl'],
                    'avg_confidence': row['avg_confidence']
                })
            
            return strategies
    
    # Market Data Methods
    def log_market_conditions(self, market_data: Dict):
        """Log current market conditions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_conditions (
                    timestamp, symbol, price, volume, volatility,
                    rsi, market_sentiment, vix_level, global_market_bias, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data.get('timestamp', datetime.now()),
                market_data['symbol'],
                market_data['price'],
                market_data.get('volume'),
                market_data.get('volatility'),
                market_data.get('rsi'),
                market_data.get('market_sentiment'),
                market_data.get('vix_level'),
                market_data.get('global_market_bias'),
                market_data.get('notes')
            ))
            
            conn.commit()
    
    # Audit Methods
    def log_event(self, event_type: str, user_action: str, details: str = None, 
                  status: str = 'SUCCESS'):
        """Log an audit event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_log (event_type, user_action, details, status)
                VALUES (?, ?, ?, ?)
            ''', (event_type, user_action, details, status))
            
            conn.commit()
    
    # Report Generation
    def generate_trade_report(self, start_date: datetime = None, 
                            end_date: datetime = None) -> pd.DataFrame:
        """Generate comprehensive trade report"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        with self.get_connection() as conn:
            # Get trades with additional calculations
            df = pd.read_sql_query('''
                SELECT 
                    t.*,
                    ROUND(julianday(COALESCE(exit_time, datetime('now'))) - 
                          julianday(entry_time), 2) as holding_days,
                    CASE 
                        WHEN pnl > 0 THEN 'WIN'
                        WHEN pnl < 0 THEN 'LOSS'
                        ELSE 'ACTIVE'
                    END as trade_result
                FROM trades t
                WHERE entry_time >= ? AND entry_time <= ?
                ORDER BY entry_time DESC
            ''', conn, params=(start_date, end_date))
        
        return df
    
    def export_to_excel(self, filename: str = None):
        """Export all data to Excel file"""
        if filename is None:
            filename = f"trademind_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Export trades
            df_trades = self.get_trade_history()
            df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Export daily performance
            with self.get_connection() as conn:
                df_daily = pd.read_sql_query('SELECT * FROM daily_performance ORDER BY date DESC', conn)
                df_daily.to_excel(writer, sheet_name='Daily Performance', index=False)
                
                # Export current positions
                df_positions = pd.read_sql_query('SELECT * FROM positions WHERE status = "OPEN"', conn)
                df_positions.to_excel(writer, sheet_name='Open Positions', index=False)
        
        print(f"✅ Data exported to {filename}")
        return filename
    
    def display_dashboard(self):
        """Display trading dashboard"""
        print("\n" + "="*80)
        print("📊 TRADEMIND AI - TRADING DASHBOARD")
        print("="*80)
        
        # Get summary
        summary = self.get_performance_summary(30)
        
        if 'error' not in summary:
            print(f"\n📅 Period: {summary['start_date']} to {summary['end_date']}")
            
            print(f"\n💰 P&L Summary:")
            print(f"   Total P&L: ₹{summary['total_pnl']:,.2f}")
            print(f"   Average Daily P&L: ₹{summary['avg_daily_pnl']:,.2f}")
            print(f"   Best Day: ₹{summary['best_day']['pnl']:,.2f} ({summary['best_day']['date']})")
            print(f"   Worst Day: ₹{summary['worst_day']['pnl']:,.2f} ({summary['worst_day']['date']})")
            
            print(f"\n📈 Trading Statistics:")
            print(f"   Total Trades: {summary['total_trades']}")
            print(f"   Win Rate (Days): {summary['win_rate_days']:.1f}%")
            print(f"   Current Streak: {summary['current_streak']['days']} days {summary['current_streak']['type']}")
            
            if summary['strategy_performance']:
                print(f"\n🎯 Strategy Performance:")
                for strategy in summary['strategy_performance'][:3]:
                    print(f"   {strategy['name']}:")
                    print(f"      Trades: {strategy['total_trades']} | Win Rate: {strategy['win_rate']:.1f}%")
                    print(f"      Total P&L: ₹{strategy['total_pnl']:,.2f}")
        
        # Get open positions
        positions = self.get_open_positions()
        if positions:
            print(f"\n📊 Open Positions ({len(positions)}):")
            for pos in positions[:5]:
                print(f"   {pos['symbol']} - Qty: {pos['quantity']} @ ₹{pos['entry_price']}")
                if pos.get('unrealized_pnl'):
                    print(f"      Unrealized P&L: ₹{pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_percent']:.1f}%)")
        
        print("\n" + "="*80)


# Test the module
if __name__ == "__main__":
    print("🧪 Testing Database Module...")
    
    # Initialize database
    db = DatabaseManager("test_trademind.db")
    
    # Test 1: Insert a trade
    print("\n1️⃣ Inserting test trade...")
    trade_data = {
        'symbol': 'NIFTY',
        'trade_type': 'OPTIONS',
        'option_type': 'CE',
        'strike_price': 25000,
        'expiry_date': '2024-02-29',
        'entry_time': datetime.now(),
        'entry_price': 150.50,
        'quantity': 50,
        'strategy_name': 'Momentum',
        'confidence_score': 85.5,
        'stop_loss': 145.0,
        'target': 160.0,
        'notes': 'Test trade'
    }
    
    trade_id = db.insert_trade(trade_data)
    print(f"   Trade inserted: {trade_id}")
    
    # Test 2: Update position
    print("\n2️⃣ Updating position price...")
    positions = db.get_open_positions()
    if positions:
        db.update_position_price(positions[0]['position_id'], 155.0)
        print("   Position updated")
    
    # Test 3: Close trade
    print("\n3️⃣ Closing trade...")
    db.update_trade_exit(trade_id, 158.0, fees=50)
    print("   Trade closed")
    
    # Test 4: Calculate daily performance
    print("\n4️⃣ Calculating daily performance...")
    perf = db.calculate_daily_performance()
    print(f"   Today's P&L: ₹{perf['net_pnl']:,.2f}")
    
    # Test 5: Display dashboard
    print("\n5️⃣ Displaying dashboard...")
    db.display_dashboard()
    
    # Test 6: Export to Excel
    print("\n6️⃣ Exporting to Excel...")
    filename = db.export_to_excel()
    
    print("\n✅ Database Module ready for integration!")