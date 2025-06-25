"""
Rejected signals logger with comprehensive performance tracking
Uses evidence-based modeling for realistic outcome analysis
"""

import json
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger('trading_system.rejected_signals')

class RejectedSignalsLogger:
    """Logger for rejected signals with enhanced performance analytics"""
    
    def __init__(self, log_file: str = "rejected_signals_analysis.json"):
        self.log_file = log_file
        self.rejected_signals = []
        self.db_path = Path("rejected_signals_performance.db")
        self.init_performance_db()
        logger.info("✅ RejectedSignalsLogger initialized")
    
    def init_performance_db(self):
        """Initialize SQLite database for performance tracking with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    strike INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    rejection_reason TEXT NOT NULL,
                    rejection_category TEXT NOT NULL,
                    entry_price REAL,
                    spot_price REAL,
                    iv REAL,
                    volume INTEGER,
                    oi INTEGER,
                    market_context TEXT,
                    hypothetical_outcome REAL DEFAULT NULL,
                    outcome_calculated_at TEXT DEFAULT NULL,
                    outcome_reasoning TEXT DEFAULT NULL,
                    historical_volatility REAL DEFAULT NULL,
                    theta_impact REAL DEFAULT NULL,
                    days_to_expiry INTEGER DEFAULT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accepted_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    strike INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL DEFAULT NULL,
                    actual_outcome REAL DEFAULT NULL,
                    outcome_calculated_at TEXT DEFAULT NULL,
                    hold_duration_minutes INTEGER DEFAULT NULL,
                    exit_reason TEXT DEFAULT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculation_date TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    avg_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    calmar_ratio REAL,
                    profit_factor REAL,
                    win_rate REAL,
                    total_trades INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Enhanced performance tracking database initialized")
            
        except Exception as e:
            logger.error(f"❌ Enhanced performance DB initialization failed: {e}")
    
    def log_rejection(self, instrument: str, strike: int, direction: str, reason: str, 
                     market_data: Dict[str, Any], signal_data: Dict[str, Any] = None):
        """Log a rejected signal with enhanced data"""
        try:
            rejection_entry = {
                'timestamp': datetime.now().isoformat(),
                'instrument': instrument,
                'strike': strike,
                'direction': direction,
                'rejection_reason': reason,
                'rejection_category': self._categorize_rejection(reason),
                'market_data': market_data,
                'signal_data': signal_data or {}
            }
            
            self.rejected_signals.append(rejection_entry)
            
            self._store_rejection_in_db(rejection_entry)
            
            self._save_to_file()
            
            logger.info(f"📝 Logged rejection: {instrument} {strike} {direction} - {reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log rejection: {e}")
    
    def _categorize_rejection(self, reason: str) -> str:
        """Categorize rejection reason for analysis"""
        reason_lower = reason.lower()
        
        if 'price' in reason_lower or 'ltp' in reason_lower:
            return 'price_validation'
        elif 'volume' in reason_lower or 'liquidity' in reason_lower:
            return 'liquidity'
        elif 'iv' in reason_lower or 'volatility' in reason_lower:
            return 'volatility'
        elif 'greeks' in reason_lower or 'delta' in reason_lower:
            return 'greeks'
        elif 'validation' in reason_lower or 'source' in reason_lower:
            return 'data_validation'
        elif 'spread' in reason_lower or 'bid' in reason_lower:
            return 'spread'
        else:
            return 'other'
    
    def _store_rejection_in_db(self, rejection_entry: Dict[str, Any]):
        """Store rejection in database for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            signal_data = rejection_entry.get('signal_data', {})
            market_data = rejection_entry.get('market_data', {})
            
            spot_price = None
            if 'spot_data' in market_data and 'prices' in market_data['spot_data']:
                spot_price = market_data['spot_data']['prices'].get(rejection_entry['instrument'])
            
            cursor.execute('''
                INSERT INTO rejected_signals 
                (timestamp, instrument, strike, direction, rejection_reason, rejection_category,
                 entry_price, spot_price, iv, volume, oi, market_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rejection_entry['timestamp'],
                rejection_entry['instrument'],
                rejection_entry['strike'],
                rejection_entry['direction'],
                rejection_entry['rejection_reason'],
                rejection_entry['rejection_category'],
                signal_data.get('ltp'),
                spot_price,
                signal_data.get('iv'),
                signal_data.get('volume'),
                signal_data.get('oi'),
                'normal'  # Default market context
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store rejection in DB: {e}")
    
    def log_accepted_signal(self, signal: Dict[str, Any]):
        """Log an accepted signal for performance comparison"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO accepted_signals 
                (timestamp, instrument, strike, direction, entry_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                signal.get('timestamp', datetime.now().isoformat()),
                signal.get('instrument'),
                signal.get('strike'),
                signal.get('direction'),
                signal.get('entry_price')
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"📝 Logged accepted signal: {signal.get('instrument')} {signal.get('strike')} {signal.get('direction')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log accepted signal: {e}")
    
    def _save_to_file(self):
        """Save rejected signals to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.rejected_signals, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save rejected signals to file: {e}")
    
    async def calculate_realistic_outcomes(self):
        """Calculate realistic outcomes using evidence-based modeling"""
        try:
            from src.analysis.realistic_outcome_calculator import RealisticOutcomeCalculator
            
            outcome_calculator = RealisticOutcomeCalculator(str(self.db_path))
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, instrument, strike, direction, entry_price, timestamp, iv
                FROM rejected_signals 
                WHERE hypothetical_outcome IS NULL
                AND entry_price > 0
                AND timestamp > datetime('now', '-7 days')
            ''')
            
            rejected_signals = cursor.fetchall()
            
            for signal_id, instrument, strike, direction, entry_price, timestamp, iv in rejected_signals:
                try:
                    realistic_outcome, reasoning = await outcome_calculator.calculate_realistic_outcome(
                        instrument, strike, direction, entry_price, iv or 20.0, timestamp
                    )
                    
                    cursor.execute('''
                        UPDATE rejected_signals 
                        SET hypothetical_outcome = ?, outcome_calculated_at = ?, outcome_reasoning = ?
                        WHERE id = ?
                    ''', (realistic_outcome, datetime.now().isoformat(), reasoning, signal_id))
                    
                    logger.debug(f"📊 Calculated realistic outcome for {instrument} {strike} {direction}: ₹{realistic_outcome:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to calculate outcome for signal {signal_id}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"📊 Calculated realistic outcomes for {len(rejected_signals)} rejected signals using evidence-based modeling")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate realistic outcomes: {e}")
    
    def calculate_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics including Sharpe ratio and drawdown"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    hypothetical_outcome,
                    entry_price,
                    timestamp,
                    instrument,
                    rejection_category
                FROM rejected_signals 
                WHERE hypothetical_outcome IS NOT NULL
                AND timestamp > datetime('now', '-30 days')
                ORDER BY timestamp
            ''')
            
            rejected_data = cursor.fetchall()
            
            cursor.execute('''
                SELECT 
                    actual_outcome,
                    entry_price,
                    timestamp,
                    instrument
                FROM accepted_signals 
                WHERE actual_outcome IS NOT NULL
                AND timestamp > datetime('now', '-30 days')
                ORDER BY timestamp
            ''')
            
            accepted_data = cursor.fetchall()
            
            conn.close()
            
            rejected_metrics = self._calculate_portfolio_metrics(rejected_data, 'rejected')
            
            accepted_metrics = self._calculate_portfolio_metrics(accepted_data, 'accepted')
            
            time_analysis = self._calculate_time_based_performance(rejected_data + accepted_data)
            
            return {
                'rejected_signals_metrics': rejected_metrics,
                'accepted_signals_metrics': accepted_metrics,
                'time_based_analysis': time_analysis,
                'comparison': {
                    'rejected_vs_accepted_return': rejected_metrics.get('avg_return', 0) - accepted_metrics.get('avg_return', 0),
                    'rejected_vs_accepted_sharpe': rejected_metrics.get('sharpe_ratio', 0) - accepted_metrics.get('sharpe_ratio', 0),
                    'filter_effectiveness_score': self._calculate_filter_effectiveness(rejected_metrics, accepted_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_portfolio_metrics(self, data: List[tuple], signal_type: str) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        try:
            if not data:
                return {}
            
            returns = []
            for row in data:
                outcome = row[0]  # hypothetical_outcome or actual_outcome
                entry_price = row[1]
                if outcome and entry_price and entry_price > 0:
                    return_pct = (outcome - entry_price) / entry_price
                    returns.append(return_pct)
            
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            win_rate = np.sum(returns_array > 0) / len(returns_array)
            
            risk_free_rate = 0.0002  # Daily risk-free rate
            sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            return {
                'avg_return': round(avg_return * 100, 2),  # Convert to percentage
                'win_rate': round(win_rate * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'total_trades': len(returns),
                'winning_trades': int(np.sum(returns_array > 0)),
                'losing_trades': int(np.sum(returns_array < 0))
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio metrics calculation failed: {e}")
            return {}
    
    def _calculate_time_based_performance(self, all_data: List[tuple]) -> Dict[str, Any]:
        """Calculate time-based performance patterns"""
        try:
            if not all_data:
                return {}
            
            hourly_performance = {}
            daily_performance = {}
            monthly_performance = {}
            
            for row in all_data:
                outcome = row[0]
                entry_price = row[1]
                timestamp_str = row[2]
                
                if not (outcome and entry_price and entry_price > 0):
                    continue
                
                return_pct = (outcome - entry_price) / entry_price
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    hour = timestamp.hour
                    if hour not in hourly_performance:
                        hourly_performance[hour] = []
                    hourly_performance[hour].append(return_pct)
                    
                    day = timestamp.strftime('%A')
                    if day not in daily_performance:
                        daily_performance[day] = []
                    daily_performance[day].append(return_pct)
                    
                    month = timestamp.strftime('%B')
                    if month not in monthly_performance:
                        monthly_performance[month] = []
                    monthly_performance[month].append(return_pct)
                    
                except Exception:
                    continue
            
            hourly_avg = {hour: np.mean(returns) * 100 for hour, returns in hourly_performance.items()}
            daily_avg = {day: np.mean(returns) * 100 for day, returns in daily_performance.items()}
            monthly_avg = {month: np.mean(returns) * 100 for month, returns in monthly_performance.items()}
            
            return {
                'hourly_performance': hourly_avg,
                'daily_performance': daily_avg,
                'monthly_performance': monthly_avg,
                'best_hour': max(hourly_avg.items(), key=lambda x: x[1]) if hourly_avg else None,
                'worst_hour': min(hourly_avg.items(), key=lambda x: x[1]) if hourly_avg else None,
                'best_day': max(daily_avg.items(), key=lambda x: x[1]) if daily_avg else None,
                'worst_day': min(daily_avg.items(), key=lambda x: x[1]) if daily_avg else None
            }
            
        except Exception as e:
            logger.error(f"❌ Time-based performance calculation failed: {e}")
            return {}
    
    def _calculate_filter_effectiveness(self, rejected_metrics: Dict, accepted_metrics: Dict) -> float:
        """Calculate overall filter effectiveness score (0-100)"""
        try:
            rejected_return = rejected_metrics.get('avg_return', 0)
            accepted_return = accepted_metrics.get('avg_return', 0)
            
            rejected_sharpe = rejected_metrics.get('sharpe_ratio', 0)
            accepted_sharpe = accepted_metrics.get('sharpe_ratio', 0)
            
            return_score = 50 + min(50, max(-50, (accepted_return - rejected_return) * 2))
            sharpe_score = 50 + min(50, max(-50, (accepted_sharpe - rejected_sharpe) * 25))
            
            overall_score = (return_score + sharpe_score) / 2
            return round(overall_score, 1)
            
        except Exception:
            return 50.0  # Neutral score
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly analysis report"""
        try:
            week_ago = datetime.now() - timedelta(days=7)
            
            rejected_signals = [
                signal for signal in self.rejected_signals 
                if datetime.fromisoformat(signal['timestamp']) >= week_ago
            ]
            
            rejection_categories = {}
            for signal in rejected_signals:
                category = signal.get('rejection_category', 'other')
                rejection_categories[category] = rejection_categories.get(category, 0) + 1
            
            rejection_reasons = {}
            for signal in rejected_signals:
                reason = signal['rejection_reason']
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            most_common_rejections = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            
            performance_comparison = self._calculate_performance_comparison()
            
            filter_effectiveness = self._calculate_filter_effectiveness_score()
            
            enhanced_metrics = self.calculate_enhanced_performance_metrics()
            
            return {
                'total_rejected': len(rejected_signals),
                'rejection_categories': rejection_categories,
                'most_common_rejections': most_common_rejections,
                'performance_comparison': performance_comparison,
                'filter_effectiveness': filter_effectiveness,
                'enhanced_metrics': enhanced_metrics,
                'weekly_summary': f"Rejected {len(rejected_signals)} signals using evidence-based analysis"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate weekly report: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_comparison(self) -> Dict[str, Any]:
        """Calculate performance comparison between rejected and accepted signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(hypothetical_outcome), AVG(entry_price), COUNT(*)
                FROM rejected_signals 
                WHERE hypothetical_outcome IS NOT NULL
                AND timestamp > datetime('now', '-7 days')
            ''')
            
            rejected_result = cursor.fetchone()
            
            cursor.execute('''
                SELECT AVG(actual_outcome), AVG(entry_price), COUNT(*)
                FROM accepted_signals 
                WHERE actual_outcome IS NOT NULL
                AND timestamp > datetime('now', '-7 days')
            ''')
            
            accepted_result = cursor.fetchone()
            
            conn.close()
            
            rejected_avg_outcome = rejected_result[0] if rejected_result[0] else 0
            rejected_avg_entry = rejected_result[1] if rejected_result[1] else 0
            rejected_count = rejected_result[2] if rejected_result[2] else 0
            
            accepted_avg_outcome = accepted_result[0] if accepted_result[0] else 0
            accepted_avg_entry = accepted_result[1] if accepted_result[1] else 0
            accepted_count = accepted_result[2] if accepted_result[2] else 0
            
            rejected_return = ((rejected_avg_outcome - rejected_avg_entry) / rejected_avg_entry * 100) if rejected_avg_entry > 0 else 0
            accepted_return = ((accepted_avg_outcome - accepted_avg_entry) / accepted_avg_entry * 100) if accepted_avg_entry > 0 else 0
            
            return {
                'rejected_signals': {
                    'count': rejected_count,
                    'avg_return_pct': round(rejected_return, 2)
                },
                'accepted_signals': {
                    'count': accepted_count,
                    'avg_return_pct': round(accepted_return, 2)
                },
                'filter_saved_pct': round(accepted_return - rejected_return, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Performance comparison calculation failed: {e}")
            return {}
    
    def _calculate_filter_effectiveness_score(self) -> float:
        """Calculate filter effectiveness score"""
        try:
            performance_comparison = self._calculate_performance_comparison()
            
            if not performance_comparison:
                return 50.0
            
            filter_saved_pct = performance_comparison.get('filter_saved_pct', 0)
            
            if filter_saved_pct > 10:
                return 90.0
            elif filter_saved_pct > 5:
                return 80.0
            elif filter_saved_pct > 0:
                return 70.0
            elif filter_saved_pct > -5:
                return 60.0
            else:
                return 40.0
                
        except Exception:
            return 50.0
