"""
TradeMind_AI: Advanced API Rate Limiter
Protects against API rate limit violations with intelligent throttling
"""

import time
import threading
from collections import deque, defaultdict
from typing import Dict, Callable, Any, Optional
from functools import wraps
import logging
from datetime import datetime, timedelta

from config.constants import API_RATE_LIMITS

class RateLimiter:
    """
    Thread-safe rate limiter with multiple algorithms:
    - Token Bucket
    - Sliding Window
    - Fixed Window
    """
    
    def __init__(self, api_name: str = 'DHAN_API'):
        self.api_name = api_name
        self.limits = API_RATE_LIMITS.get(api_name, {})
        self.logger = logging.getLogger(f'RateLimiter.{api_name}')
        
        # Token bucket implementation
        self.tokens = self.limits.get('calls_per_second', 10)
        self.max_tokens = self.tokens
        self.last_update = time.time()
        
        # Sliding window for per-minute and per-hour tracking
        self.call_timestamps = deque()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'blocked_calls': 0,
            'last_reset': datetime.now()
        }
        
        self.logger.info(f"Rate limiter initialized for {api_name}")
    
    def _update_tokens(self):
        """Update available tokens based on time passed"""
        current_time = time.time()
        time_passed = current_time - self.last_update
        
        # Replenish tokens based on rate
        tokens_to_add = time_passed * self.max_tokens
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_update = current_time
    
    def _clean_old_timestamps(self):
        """Remove timestamps older than 1 hour"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        while self.call_timestamps and self.call_timestamps[0] < cutoff_time:
            self.call_timestamps.popleft()
    
    def can_make_request(self) -> tuple[bool, Optional[float]]:
        """
        Check if request can be made
        Returns: (can_make_request, wait_time_if_not)
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_timestamps()
            self._update_tokens()
            
            # Check per-second limit (token bucket)
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.max_tokens
                return False, wait_time
            
            # Check per-minute limit
            minute_ago = current_time - 60
            recent_minute_calls = sum(1 for t in self.call_timestamps if t > minute_ago)
            
            minute_limit = self.limits.get('calls_per_minute', float('inf'))
            if recent_minute_calls >= minute_limit:
                oldest_in_minute = next((t for t in self.call_timestamps if t > minute_ago), current_time)
                wait_time = 60 - (current_time - oldest_in_minute) + 0.1
                return False, wait_time
            
            # Check per-hour limit
            hour_ago = current_time - 3600
            recent_hour_calls = sum(1 for t in self.call_timestamps if t > hour_ago)
            
            hour_limit = self.limits.get('calls_per_hour', float('inf'))
            if recent_hour_calls >= hour_limit:
                oldest_in_hour = next((t for t in self.call_timestamps if t > hour_ago), current_time)
                wait_time = 3600 - (current_time - oldest_in_hour) + 0.1
                return False, wait_time
            
            return True, None
    
    def record_request(self):
        """Record that a request was made"""
        with self.lock:
            current_time = time.time()
            
            # Consume a token
            self.tokens -= 1
            
            # Record timestamp
            self.call_timestamps.append(current_time)
            
            # Update statistics
            self.stats['total_calls'] += 1
    
    def wait_if_needed(self) -> bool:
        """
        Wait if rate limit would be exceeded
        Returns True if request can proceed, False if interrupted
        """
        max_wait_time = 300  # 5 minutes max wait
        total_waited = 0
        
        while total_waited < max_wait_time:
            can_proceed, wait_time = self.can_make_request()
            
            if can_proceed:
                self.record_request()
                return True
            
            if wait_time is None:
                wait_time = 1
            
            # Cap wait time
            wait_time = min(wait_time, max_wait_time - total_waited)
            
            self.logger.warning(
                f"Rate limit reached. Waiting {wait_time:.2f} seconds..."
            )
            self.stats['blocked_calls'] += 1
            
            time.sleep(wait_time)
            total_waited += wait_time
        
        self.logger.error("Max wait time exceeded for rate limit")
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            current_time = time.time()
            
            # Calculate current rates
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            calls_last_minute = sum(1 for t in self.call_timestamps if t > minute_ago)
            calls_last_hour = sum(1 for t in self.call_timestamps if t > hour_ago)
            
            return {
                'api': self.api_name,
                'total_calls': self.stats['total_calls'],
                'blocked_calls': self.stats['blocked_calls'],
                'calls_last_minute': calls_last_minute,
                'calls_last_hour': calls_last_hour,
                'available_tokens': self.tokens,
                'limits': self.limits,
                'uptime': str(datetime.now() - self.stats['last_reset'])
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.stats = {
                'total_calls': 0,
                'blocked_calls': 0,
                'last_reset': datetime.now()
            }


class MultiApiRateLimiter:
    """Manages rate limiters for multiple APIs"""
    
    def __init__(self):
        self.limiters = {}
        self.logger = logging.getLogger('MultiApiRateLimiter')
    
    def get_limiter(self, api_name: str) -> RateLimiter:
        """Get or create rate limiter for API"""
        if api_name not in self.limiters:
            self.limiters[api_name] = RateLimiter(api_name)
        return self.limiters[api_name]
    
    def wait_if_needed(self, api_name: str) -> bool:
        """Wait if needed for specific API"""
        limiter = self.get_limiter(api_name)
        return limiter.wait_if_needed()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all APIs"""
        return {
            api_name: limiter.get_stats()
            for api_name, limiter in self.limiters.items()
        }


# Global rate limiter instance
rate_limiter = MultiApiRateLimiter()


def rate_limit(api_name: str = 'DHAN_API'):
    """
    Decorator to apply rate limiting to functions
    
    Usage:
        @rate_limit('DHAN_API')
        def make_api_call():
            # API call code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Wait if rate limit would be exceeded
            if not rate_limiter.wait_if_needed(api_name):
                raise Exception(f"Rate limit timeout for {api_name}")
            
            # Execute the function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class AsyncRateLimiter:
    """Asynchronous rate limiter for non-blocking operations"""
    
    def __init__(self, api_name: str = 'DHAN_API'):
        self.limiter = RateLimiter(api_name)
        self.pending_queue = deque()
        self.worker_thread = None
        self.running = False
        self.logger = logging.getLogger(f'AsyncRateLimiter.{api_name}')
    
    def start(self):
        """Start the async processor"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            self.logger.info("Async rate limiter started")
    
    def stop(self):
        """Stop the async processor"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("Async rate limiter stopped")
    
    def _process_queue(self):
        """Process pending requests"""
        while self.running:
            if self.pending_queue:
                func, args, kwargs, callback, error_callback = self.pending_queue[0]
                
                try:
                    # Wait for rate limit
                    if self.limiter.wait_if_needed():
                        # Remove from queue
                        self.pending_queue.popleft()
                        
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Call callback if provided
                        if callback:
                            callback(result)
                    else:
                        # Rate limit timeout
                        self.pending_queue.popleft()
                        if error_callback:
                            error_callback(Exception("Rate limit timeout"))
                            
                except Exception as e:
                    self.logger.error(f"Error processing queued request: {e}")
                    self.pending_queue.popleft()
                    if error_callback:
                        error_callback(e)
            else:
                # No pending requests, sleep briefly
                time.sleep(0.1)
    
    def queue_request(self, func: Callable, *args, 
                     callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None,
                     **kwargs):
        """Queue a request for rate-limited execution"""
        self.pending_queue.append((func, args, kwargs, callback, error_callback))
        self.logger.debug(f"Queued request. Queue size: {len(self.pending_queue)}")


def create_rate_limited_session(api_name: str = 'DHAN_API'):
    """
    Create a requests session with rate limiting
    
    Usage:
        session = create_rate_limited_session('DHAN_API')
        response = session.get(url)  # Automatically rate limited
    """
    import requests
    
    class RateLimitedSession(requests.Session):
        def __init__(self, api_name: str):
            super().__init__()
            self.api_name = api_name
            self.logger = logging.getLogger(f'RateLimitedSession.{api_name}')
        
        def request(self, method, url, **kwargs):
            # Apply rate limiting
            if not rate_limiter.wait_if_needed(self.api_name):
                raise Exception(f"Rate limit timeout for {self.api_name}")
            
            # Make request
            self.logger.debug(f"{method} {url}")
            return super().request(method, url, **kwargs)
    
    return RateLimitedSession(api_name)


# Example usage and testing
if __name__ == "__main__":
    # Test basic rate limiting
    print("Testing Rate Limiter...")
    
    @rate_limit('DHAN_API')
    def make_test_call(i):
        print(f"Making call {i} at {datetime.now().strftime('%H:%M:%S.%f')}")
        return f"Result {i}"
    
    # Test synchronous rate limiting
    print("\n1. Testing synchronous rate limiting:")
    for i in range(15):
        try:
            result = make_test_call(i)
            print(f"   {result}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Show statistics
    print("\n2. Rate limiter statistics:")
    stats = rate_limiter.get_all_stats()
    for api, stat in stats.items():
        print(f"\n   {api}:")
        for key, value in stat.items():
            print(f"      {key}: {value}")
    
    # Test async rate limiting
    print("\n3. Testing async rate limiting:")
    async_limiter = AsyncRateLimiter('DHAN_API')
    async_limiter.start()
    
    def on_success(result):
        print(f"   Async success: {result}")
    
    def on_error(error):
        print(f"   Async error: {error}")
    
    # Queue multiple requests
    for i in range(10):
        async_limiter.queue_request(
            lambda x: f"Async result {x}",
            i,
            callback=on_success,
            error_callback=on_error
        )
    
    # Wait for processing
    time.sleep(5)
    async_limiter.stop()
    
    print("\nRate limiter testing complete!")