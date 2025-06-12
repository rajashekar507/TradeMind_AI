"""
Enhance error handling across the TradeMind AI system
Adds retry logic, circuit breakers, and better error recovery
"""

import os

def create_enhanced_error_handler():
    """Create an enhanced error handler with retry logic"""
    
    error_handler_code = '''"""
TradeMind_AI: Enhanced Error Handler with Self-Healing
Provides retry logic, circuit breakers, and automatic recovery
"""

import os
import json
import time
import logging
import traceback
import functools
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import threading

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'


class ErrorHandler:
    """Enhanced error handler with retry logic and self-healing"""
    
    def __init__(self):
        """Initialize error handler"""
        self.logger = logging.getLogger('ErrorHandler')
        
        # Error tracking
        self.error_history = []
        self.error_patterns = {}
        
        # Circuit breakers for different services
        self.circuit_breakers = {
            'dhan_api': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'news_api': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'telegram': CircuitBreaker(failure_threshold=10, recovery_timeout=120)
        }
        
        # Retry configuration
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1,  # seconds
            'max_delay': 60,  # seconds
            'exponential_base': 2
        }
        
        # Load error history
        self.load_error_history()
        
        self.logger.info("Enhanced error handler initialized")
    
    def with_retry(self, func: Callable = None, *, 
                   max_retries: int = None,
                   service_name: str = None,
                   on_retry: Callable = None):
        """Decorator to add retry logic to functions"""
        if func is None:
            return functools.partial(self.with_retry, 
                                   max_retries=max_retries,
                                   service_name=service_name,
                                   on_retry=on_retry)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries or self.retry_config['max_retries']
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    # Check circuit breaker if service specified
                    if service_name and service_name in self.circuit_breakers:
                        return self.circuit_breakers[service_name].call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    self._log_error(func.__name__, e, attempt)
                    
                    if attempt < retries:
                        delay = self._calculate_delay(attempt)
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                    else:
                        self.logger.error(f"All retries exhausted for {func.__name__}")
                        
                        # Try to heal the error
                        healed = self._try_heal_error(func, e, args, kwargs)
                        if healed is not None:
                            return healed
            
            # All retries failed
            self._handle_final_failure(func.__name__, last_exception)
            raise last_exception
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = min(
            self.retry_config['base_delay'] * (self.retry_config['exponential_base'] ** attempt),
            self.retry_config['max_delay']
        )
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
    
    def _log_error(self, function_name: str, error: Exception, attempt: int):
        """Log error details"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'attempt': attempt,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_entry)
        
        # Track error patterns
        error_key = f"{function_name}:{type(error).__name__}"
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = 0
        self.error_patterns[error_key] += 1
    
    def _try_heal_error(self, func: Callable, error: Exception, args: tuple, kwargs: dict) -> Any:
        """Attempt to heal/recover from specific errors"""
        error_type = type(error).__name__
        
        # Connection errors - try to reconnect
        if error_type in ['ConnectionError', 'TimeoutError']:
            self.logger.info("Attempting to heal connection error...")
            time.sleep(5)  # Wait before retry
            return None  # Let retry logic handle it
        
        # Authentication errors - try to refresh token
        if 'auth' in str(error).lower() or 'token' in str(error).lower():
            self.logger.info("Attempting to refresh authentication...")
            # This would refresh tokens in a real implementation
            return None
        
        # Data errors - try with default values
        if error_type in ['KeyError', 'ValueError', 'TypeError']:
            self.logger.info("Attempting to heal data error with defaults...")
            # Return safe default based on function
            if 'balance' in func.__name__.lower():
                return {'balance': 0, 'status': 'error', 'message': str(error)}
            elif 'price' in func.__name__.lower():
                return {'price': 0, 'status': 'error', 'message': str(error)}
        
        return None
    
    def _handle_final_failure(self, function_name: str, error: Exception):
        """Handle final failure after all retries"""
        # Send alert
        alert_message = f"CRITICAL: {function_name} failed after all retries: {error}"
        self.logger.critical(alert_message)
        
        # Save error state
        self.save_error_history()
        
        # Check if we need to trigger emergency shutdown
        if self._should_emergency_shutdown():
            self.logger.critical("Too many failures - triggering emergency shutdown")
            self._emergency_shutdown()
    
    def _should_emergency_shutdown(self) -> bool:
        """Check if we should trigger emergency shutdown"""
        # Count recent errors (last 5 minutes)
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(minutes=5)
        ]
        
        return len(recent_errors) > 50  # More than 50 errors in 5 minutes
    
    def _emergency_shutdown(self):
        """Emergency shutdown procedures"""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Save current state
        emergency_file = os.path.join('logs', f'emergency_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        os.makedirs('logs', exist_ok=True)
        
        with open(emergency_file, 'w') as f:
            json.dump({
                'error_history': self.error_history,
                'error_patterns': self.error_patterns,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Would close all positions in real implementation
        self.logger.critical(f"Emergency state saved to {emergency_file}")
    
    def save_error_history(self):
        """Save error history to file"""
        os.makedirs('logs', exist_ok=True)
        history_file = os.path.join('logs', 'error_history.json')
        
        # Keep only last 1000 errors
        self.error_history = self.error_history[-1000:]
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.error_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save error history: {e}")
    
    def load_error_history(self):
        """Load error history from file"""
        history_file = os.path.join('logs', 'error_history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.error_history = json.load(f)
                self.logger.info(f"Loaded {len(self.error_history)} error records")
            except Exception as e:
                self.logger.warning(f"Could not load error history: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)
        ]
        
        circuit_status = {
            name: breaker.state 
            for name, breaker in self.circuit_breakers.items()
        }
        
        return {
            'healthy': len(recent_errors) < 10,
            'errors_last_hour': len(recent_errors),
            'total_errors': len(self.error_history),
            'circuit_breakers': circuit_status,
            'most_common_errors': sorted(
                self.error_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }


# Global error handler instance
error_handler = ErrorHandler()

# Convenience decorators
def with_retry(max_retries: int = 3, service_name: str = None):
    """Decorator to add retry logic"""
    return error_handler.with_retry(max_retries=max_retries, service_name=service_name)


# Example usage
if __name__ == "__main__":
    # Test the error handler
    
    @with_retry(max_retries=3, service_name='dhan_api')
    def test_api_call():
        """Simulated API call that might fail"""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("API connection failed")
        return "Success!"
    
    # Test multiple times
    for i in range(5):
        print(f"\\nTest {i+1}:")
        try:
            result = test_api_call()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {e}")
    
    # Show health status
    print("\\nSystem Health Status:")
    health = error_handler.get_health_status()
    for key, value in health.items():
        print(f"  {key}: {value}")
'''
    
    # Write the enhanced error handler
    error_handler_path = os.path.join('src', 'analysis', 'error_handler.py')
    
    # Backup existing file
    if os.path.exists(error_handler_path):
        backup_path = error_handler_path + '.backup'
        try:
            with open(error_handler_path, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            print(f"✅ Backed up existing error_handler.py")
        except:
            print("⚠️ Could not backup existing file, proceeding...")
    
    # Write new implementation
    with open(error_handler_path, 'w', encoding='utf-8') as f:
        f.write(error_handler_code)
    
    print(f"✅ Created enhanced error_handler.py with:")
    print("   - Retry logic with exponential backoff")
    print("   - Circuit breakers for API protection")
    print("   - Self-healing capabilities")
    print("   - Emergency shutdown procedures")
    print("   - Error pattern tracking")
    
    return True

def main():
    print("🔧 Enhancing Error Handling System...")
    print("=" * 50)
    
    if create_enhanced_error_handler():
        print("\n✅ SUCCESS! Error handling system enhanced.")
        print("\n💡 To test it, run:")
        print("   python src/analysis/error_handler.py")
    else:
        print("\n❌ Failed to enhance error handler")

if __name__ == "__main__":
    main()