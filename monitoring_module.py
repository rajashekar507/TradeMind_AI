"""
TradeMind_AI: Monitoring and Alerting Module
Real-time monitoring, performance metrics, and alert system
"""

import os
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import logging.handlers
from collections import deque
import requests
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    action_required: bool = False

class MonitoringService:
    def __init__(self, check_interval: int = 60):
        """Initialize Monitoring Service"""
        print("📡 Initializing Monitoring Service...")
        
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics_history = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'api_latency': deque(maxlen=1000),
            'trade_success_rate': deque(maxlen=100),
            'pnl': deque(maxlen=1000),
            'position_count': deque(maxlen=1000),
            'error_count': deque(maxlen=1000)
        }
        
        # Alert configuration
        self.alert_thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'api_latency': {'warning': 1000, 'critical': 3000},  # milliseconds
            'trade_success_rate': {'warning': 70, 'critical': 50},  # percentage
            'daily_loss': {'warning': -20000, 'critical': -50000},  # rupees
            'consecutive_losses': {'warning': 3, 'critical': 5},
            'error_rate': {'warning': 5, 'critical': 10}  # errors per minute
        }
        
        # Alert handlers
        self.alert_handlers = []
        self.alert_history = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_metrics = {
            'uptime_start': datetime.now(),
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0,
            'api_calls': 0,
            'api_errors': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger('MonitoringService')
        self._setup_logging()
        
        print("✅ Monitoring Service initialized!")
    
    def _setup_logging(self):
        """Setup monitoring logs"""
        os.makedirs('logs/monitoring', exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            'logs/monitoring/monitoring.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def start(self):
        """Start monitoring service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Monitoring service started")
        print("🟢 Monitoring service started")
    
    def stop(self):
        """Stop monitoring service"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Monitoring service stopped")
        print("🔴 Monitoring service stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check thresholds
                self._check_thresholds()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_history['cpu_usage'].append({
            'timestamp': timestamp,
            'value': cpu_percent
        })
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_history['memory_usage'].append({
            'timestamp': timestamp,
            'value': memory.percent
        })
        
        # Process-specific metrics
        process = psutil.Process()
        process_info = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'threads': process.num_threads()
        }
        
        self.logger.debug(f"System metrics: CPU={cpu_percent}%, Memory={memory.percent}%")
    
    def record_api_call(self, api_name: str, latency_ms: float, success: bool):
        """Record API call metrics"""
        timestamp = datetime.now()
        
        self.metrics_history['api_latency'].append({
            'timestamp': timestamp,
            'api': api_name,
            'latency': latency_ms,
            'success': success
        })
        
        self.performance_metrics['api_calls'] += 1
        if not success:
            self.performance_metrics['api_errors'] += 1
        
        # Check API performance
        if latency_ms > self.alert_thresholds['api_latency']['critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                f"Critical API Latency - {api_name}",
                f"API response time: {latency_ms}ms",
                metric='api_latency',
                value=latency_ms,
                threshold=self.alert_thresholds['api_latency']['critical']
            )
    
    def record_trade(self, trade_id: str, success: bool, pnl: float = 0):
        """Record trade execution metrics"""
        timestamp = datetime.now()
        
        self.performance_metrics['total_trades'] += 1
        if success:
            self.performance_metrics['successful_trades'] += 1
        else:
            self.performance_metrics['failed_trades'] += 1
        
        self.performance_metrics['total_pnl'] += pnl
        
        # Update success rate
        if self.performance_metrics['total_trades'] > 0:
            success_rate = (self.performance_metrics['successful_trades'] / 
                          self.performance_metrics['total_trades']) * 100
            
            self.metrics_history['trade_success_rate'].append({
                'timestamp': timestamp,
                'value': success_rate
            })
            
            # Check success rate threshold
            if success_rate < self.alert_thresholds['trade_success_rate']['critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Critical Trade Success Rate",
                    f"Success rate dropped to {success_rate:.1f}%",
                    metric='trade_success_rate',
                    value=success_rate,
                    threshold=self.alert_thresholds['trade_success_rate']['critical'],
                    action_required=True
                )
    
    def record_error(self, error_type: str, error_message: str, 
                    component: str = None):
        """Record system errors"""
        timestamp = datetime.now()
        
        self.metrics_history['error_count'].append({
            'timestamp': timestamp,
            'type': error_type,
            'message': error_message,
            'component': component
        })
        
        # Calculate error rate (errors per minute)
        recent_errors = [
            e for e in self.metrics_history['error_count']
            if (timestamp - e['timestamp']).total_seconds() < 60
        ]
        
        error_rate = len(recent_errors)
        
        if error_rate > self.alert_thresholds['error_rate']['critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "High Error Rate Detected",
                f"{error_rate} errors in the last minute",
                metric='error_rate',
                value=error_rate,
                threshold=self.alert_thresholds['error_rate']['critical'],
                action_required=True
            )
    
    def _check_thresholds(self):
        """Check all metric thresholds"""
        # Check CPU usage
        if self.metrics_history['cpu_usage']:
            latest_cpu = self.metrics_history['cpu_usage'][-1]['value']
            if latest_cpu > self.alert_thresholds['cpu_usage']['critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Critical CPU Usage",
                    f"CPU usage at {latest_cpu}%",
                    metric='cpu_usage',
                    value=latest_cpu,
                    threshold=self.alert_thresholds['cpu_usage']['critical']
                )
            elif latest_cpu > self.alert_thresholds['cpu_usage']['warning']:
                self._create_alert(
                    AlertLevel.WARNING,
                    "High CPU Usage",
                    f"CPU usage at {latest_cpu}%",
                    metric='cpu_usage',
                    value=latest_cpu,
                    threshold=self.alert_thresholds['cpu_usage']['warning']
                )
        
        # Check memory usage
        if self.metrics_history['memory_usage']:
            latest_memory = self.metrics_history['memory_usage'][-1]['value']
            if latest_memory > self.alert_thresholds['memory_usage']['critical']:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Critical Memory Usage",
                    f"Memory usage at {latest_memory}%",
                    metric='memory_usage',
                    value=latest_memory,
                    threshold=self.alert_thresholds['memory_usage']['critical'],
                    action_required=True
                )
    
    def _create_alert(self, level: AlertLevel, title: str, message: str, **kwargs):
        """Create and dispatch alert"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            **kwargs
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        # Log alert
        log_message = f"[{alert.level.value}] {alert.title}: {alert.message}"
        if alert.level == AlertLevel.CRITICAL:
            self.logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            self.logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def get_system_health(self) -> Dict:
        """Get current system health status"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(),
            'uptime': str(datetime.now() - self.performance_metrics['uptime_start']),
            'metrics': {}
        }
        
        # Get latest metrics
        if self.metrics_history['cpu_usage']:
            health_status['metrics']['cpu_usage'] = self.metrics_history['cpu_usage'][-1]['value']
        
        if self.metrics_history['memory_usage']:
            health_status['metrics']['memory_usage'] = self.metrics_history['memory_usage'][-1]['value']
        
        # Calculate API success rate
        if self.performance_metrics['api_calls'] > 0:
            api_success_rate = ((self.performance_metrics['api_calls'] - 
                               self.performance_metrics['api_errors']) / 
                              self.performance_metrics['api_calls']) * 100
            health_status['metrics']['api_success_rate'] = api_success_rate
        
        # Trade metrics
        health_status['metrics']['total_trades'] = self.performance_metrics['total_trades']
        health_status['metrics']['total_pnl'] = self.performance_metrics['total_pnl']
        
        # Determine overall health
        recent_critical_alerts = [
            a for a in self.alert_history
            if a.level == AlertLevel.CRITICAL and
            (datetime.now() - a.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        if recent_critical_alerts:
            health_status['status'] = 'critical'
        elif any(a.level == AlertLevel.ERROR for a in list(self.alert_history)[-10:]):
            health_status['status'] = 'degraded'
        
        return health_status
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        report = {
            'timestamp': datetime.now(),
            'uptime': str(datetime.now() - self.performance_metrics['uptime_start']),
            'trading_metrics': {
                'total_trades': self.performance_metrics['total_trades'],
                'successful_trades': self.performance_metrics['successful_trades'],
                'failed_trades': self.performance_metrics['failed_trades'],
                'success_rate': 0,
                'total_pnl': self.performance_metrics['total_pnl']
            },
            'api_metrics': {
                'total_calls': self.performance_metrics['api_calls'],
                'errors': self.performance_metrics['api_errors'],
                'success_rate': 0,
                'avg_latency': 0
            },
            'system_metrics': {
                'avg_cpu_usage': 0,
                'max_cpu_usage': 0,
                'avg_memory_usage': 0,
                'max_memory_usage': 0
            },
            'alerts_summary': {
                'total': len(self.alert_history),
                'critical': 0,
                'error': 0,
                'warning': 0,
                'info': 0
            }
        }
        
        # Calculate success rates
        if report['trading_metrics']['total_trades'] > 0:
            report['trading_metrics']['success_rate'] = (
                report['trading_metrics']['successful_trades'] / 
                report['trading_metrics']['total_trades']
            ) * 100
        
        if report['api_metrics']['total_calls'] > 0:
            report['api_metrics']['success_rate'] = (
                (report['api_metrics']['total_calls'] - report['api_metrics']['errors']) /
                report['api_metrics']['total_calls']
            ) * 100
        
        # Calculate average latency
        if self.metrics_history['api_latency']:
            latencies = [m['latency'] for m in self.metrics_history['api_latency']]
            report['api_metrics']['avg_latency'] = sum(latencies) / len(latencies)
        
        # System metrics
        if self.metrics_history['cpu_usage']:
            cpu_values = [m['value'] for m in self.metrics_history['cpu_usage']]
            report['system_metrics']['avg_cpu_usage'] = sum(cpu_values) / len(cpu_values)
            report['system_metrics']['max_cpu_usage'] = max(cpu_values)
        
        if self.metrics_history['memory_usage']:
            memory_values = [m['value'] for m in self.metrics_history['memory_usage']]
            report['system_metrics']['avg_memory_usage'] = sum(memory_values) / len(memory_values)
            report['system_metrics']['max_memory_usage'] = max(memory_values)
        
        # Alert counts
        for alert in self.alert_history:
            report['alerts_summary'][alert.level.value.lower()] += 1
        
        return report
    
    def display_dashboard(self):
        """Display monitoring dashboard"""
        print("\n" + "="*70)
        print("📊 MONITORING DASHBOARD")
        print("="*70)
        
        health = self.get_system_health()
        report = self.get_performance_report()
        
        # System Status
        status_emoji = {
            'healthy': '🟢',
            'degraded': '🟡',
            'critical': '🔴'
        }
        
        print(f"\n{status_emoji.get(health['status'], '⚪')} System Status: {health['status'].upper()}")
        print(f"⏱️  Uptime: {report['uptime']}")
        
        # System Metrics
        print(f"\n💻 System Metrics:")
        if 'cpu_usage' in health['metrics']:
            print(f"   CPU Usage: {health['metrics']['cpu_usage']:.1f}%")
        if 'memory_usage' in health['metrics']:
            print(f"   Memory Usage: {health['metrics']['memory_usage']:.1f}%")
        
        # Trading Performance
        print(f"\n📈 Trading Performance:")
        tm = report['trading_metrics']
        print(f"   Total Trades: {tm['total_trades']}")
        print(f"   Success Rate: {tm['success_rate']:.1f}%")
        print(f"   Total P&L: ₹{tm['total_pnl']:,.2f}")
        
        # API Performance
        print(f"\n🔌 API Performance:")
        am = report['api_metrics']
        print(f"   Total Calls: {am['total_calls']}")
        print(f"   Success Rate: {am['success_rate']:.1f}%")
        print(f"   Avg Latency: {am['avg_latency']:.0f}ms")
        
        # Recent Alerts
        recent_alerts = list(self.alert_history)[-5:]
        if recent_alerts:
            print(f"\n⚠️  Recent Alerts:")
            for alert in reversed(recent_alerts):
                time_ago = datetime.now() - alert.timestamp
                minutes_ago = int(time_ago.total_seconds() / 60)
                print(f"   [{alert.level.value}] {alert.title} ({minutes_ago}m ago)")
        
        print("\n" + "="*70)


class TelegramAlertHandler:
    """Send alerts via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_alert(self, alert: Alert):
        """Send alert to Telegram"""
        # Format message
        emoji_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }
        
        message = f"{emoji_map.get(alert.level, '📢')} *{alert.title}*\n\n"
        message += f"{alert.message}\n"
        
        if alert.metric and alert.value:
            message += f"\nMetric: {alert.metric}\n"
            message += f"Value: {alert.value}"
            if alert.threshold:
                message += f" (Threshold: {alert.threshold})"
        
        if alert.action_required:
            message += "\n\n⚡ *Action Required!*"
        
        message += f"\n\n_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"
        
        # Send message
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Telegram alert: {e}")


# Example custom metrics
class CustomMetrics:
    """Define custom trading metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_dev == 0:
            return 0
        
        return (avg_return - risk_free_rate) / std_dev
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


# Test the module
if __name__ == "__main__":
    print("🧪 Testing Monitoring Module...")
    
    # Create monitoring service
    monitor = MonitoringService(check_interval=5)
    
    # Add console alert handler
    def console_alert_handler(alert: Alert):
        print(f"\n🔔 ALERT: [{alert.level.value}] {alert.title}")
        print(f"   {alert.message}")
        if alert.action_required:
            print("   ⚡ Action Required!")
    
    monitor.add_alert_handler(console_alert_handler)
    
    # Start monitoring
    monitor.start()
    
    # Simulate some activity
    print("\n📊 Simulating trading activity...")
    
    # Simulate API calls
    for i in range(5):
        latency = 100 + i * 50
        monitor.record_api_call("DHAN_API", latency, True)
        time.sleep(1)
    
    # Simulate trades
    monitor.record_trade("TRD001", True, 1500)
    monitor.record_trade("TRD002", True, 2000)
    monitor.record_trade("TRD003", False, -500)
    
    # Simulate an error
    monitor.record_error("APIError", "Connection timeout", "DhanAPI")
    
    # Wait a bit for metrics to accumulate
    time.sleep(10)
    
    # Display dashboard
    monitor.display_dashboard()
    
    # Get performance report
    report = monitor.get_performance_report()
    print(f"\n📋 Performance Report Generated")
    print(f"   API Success Rate: {report['api_metrics']['success_rate']:.1f}%")
    print(f"   Trading Success Rate: {report['trading_metrics']['success_rate']:.1f}%")
    
    # Stop monitoring
    monitor.stop()
    
    print("\n✅ Monitoring Module ready for integration!")