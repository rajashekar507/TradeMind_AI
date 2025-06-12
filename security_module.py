"""
TradeMind_AI: Security Enhancement Module
Provides API key encryption, request signing, and audit logging
"""

import os
import json
import hashlib
import hmac
import base64
import logging
import logging.handlers
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import threading
from typing import Dict, Any, Optional
from pathlib import Path

class SecurityModule:
    def __init__(self):
        """Initialize Security Module"""
        print("🔐 Initializing Security Module...")
        
        # Setup logging
        self.logger = logging.getLogger('SecurityAudit')
        self._setup_audit_logging()
        
        # Initialize encryption
        self.master_key = self._get_or_create_master_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # API request signing
        self.request_counter = 0
        self.request_lock = threading.Lock()
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
        
        # Rate limiting for security
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        
        print("✅ Security Module initialized!")
    
    def _setup_audit_logging(self):
        """Setup secure audit logging"""
        # Create logs directory
        os.makedirs('logs/audit', exist_ok=True)
        
        # Configure audit logger
        audit_handler = logging.handlers.RotatingFileHandler(
            'logs/audit/security_audit.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
        self.logger.setLevel(logging.INFO)
    
    def _get_or_create_master_key(self):
        """Get or create master encryption key"""
        key_file = 'config/.master_key'
        os.makedirs('config', exist_ok=True)
        
        if os.path.exists(key_file):
            # Load existing key
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set file permissions (works on Unix-like systems)
            try:
                os.chmod(key_file, 0o600)  # Read/write for owner only
            except:
                pass  # Windows doesn't support chmod
            
        return key
    
    def encrypt_api_key(self, api_key: str, key_name: str) -> str:
        """Encrypt an API key"""
        try:
            encrypted = self.cipher_suite.encrypt(api_key.encode())
            
            # Log the encryption (without the actual key)
            self.logger.info(f"API key encrypted: {key_name}")
            
            return encrypted.decode()
        except Exception as e:
            self.logger.error(f"Encryption failed for {key_name}: {str(e)}")
            raise
    
    def decrypt_api_key(self, encrypted_key: str, key_name: str) -> str:
        """Decrypt an API key"""
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_key.encode())
            
            # Log the decryption access
            self.logger.info(f"API key accessed: {key_name}")
            
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed for {key_name}: {str(e)}")
            raise
    
    def secure_env_file(self):
        """Encrypt all API keys in .env file"""
        print("\n🔐 Securing environment variables...")
        
        # Read current .env
        env_path = '.env'
        if not os.path.exists(env_path):
            print("❌ .env file not found!")
            return
        
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Create secure config
        secure_config = {}
        plain_config = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                plain_config.append(line)
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                
                # Identify sensitive keys
                sensitive_keys = ['API_KEY', 'TOKEN', 'SECRET', 'PASSWORD']
                is_sensitive = any(sensitive in key.upper() for sensitive in sensitive_keys)
                
                if is_sensitive and value:
                    # Encrypt sensitive values
                    encrypted_value = self.encrypt_api_key(value, key)
                    secure_config[key] = encrypted_value
                    plain_config.append(f"{key}=ENCRYPTED")
                else:
                    plain_config.append(line)
        
        # Save encrypted config
        with open('config/secure_config.json', 'w') as f:
            json.dump(secure_config, f, indent=2)
        
        # Set file permissions (works on Unix-like systems)
        try:
            os.chmod('config/secure_config.json', 0o600)
        except:
            pass  # Windows doesn't support chmod
        
        # Update .env to show encrypted status
        with open(env_path, 'w') as f:
            f.write('\n'.join(plain_config))
        
        print(f"✅ Encrypted {len(secure_config)} sensitive values")
        self.logger.info(f"Environment secured: {len(secure_config)} keys encrypted")
    
    def load_secure_config(self) -> Dict[str, str]:
        """Load and decrypt secure configuration"""
        config_path = 'config/secure_config.json'
        
        if not os.path.exists(config_path):
            self.logger.warning("Secure config not found")
            return {}
        
        with open(config_path, 'r') as f:
            encrypted_config = json.load(f)
        
        decrypted_config = {}
        for key, encrypted_value in encrypted_config.items():
            try:
                decrypted_config[key] = self.decrypt_api_key(encrypted_value, key)
            except Exception as e:
                self.logger.error(f"Failed to decrypt {key}: {str(e)}")
        
        return decrypted_config
    
    def sign_request(self, method: str, url: str, body: Optional[str] = None, 
                     api_secret: str = None) -> Dict[str, str]:
        """Sign API request for authentication"""
        with self.request_lock:
            self.request_counter += 1
            
        # Create request components
        timestamp = str(int(datetime.now().timestamp()))
        nonce = str(self.request_counter)
        
        # Build signature payload
        signature_payload = f"{method}\n{url}\n{nonce}\n{timestamp}"
        if body:
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            signature_payload += f"\n{body_hash}"
        
        # Create HMAC signature
        if api_secret:
            signature = hmac.new(
                api_secret.encode(),
                signature_payload.encode(),
                hashlib.sha256
            ).hexdigest()
        else:
            # Use a default signing method if no secret provided
            signature = hashlib.sha256(signature_payload.encode()).hexdigest()
        
        # Log the request signing
        self.logger.info(f"Request signed: {method} {url}")
        
        return {
            'X-Timestamp': timestamp,
            'X-Nonce': nonce,
            'X-Signature': signature
        }
    
    def create_session(self, user_id: str) -> str:
        """Create a secure session"""
        session_id = secrets.token_urlsafe(32)
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': None,  # Can be set from request
            'user_agent': None   # Can be set from request
        }
        
        self.logger.info(f"Session created for user: {user_id}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session"""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Invalid session attempted: {session_id[:8]}...")
            return None
        
        session = self.active_sessions[session_id]
        
        # Check timeout
        if (datetime.now() - session['last_activity']).total_seconds() > self.session_timeout:
            self.logger.info(f"Session expired: {session['user_id']}")
            del self.active_sessions[session_id]
            return None
        
        # Refresh activity
        session['last_activity'] = datetime.now()
        
        return session
    
    def log_api_call(self, api_name: str, endpoint: str, 
                     status_code: int, response_time: float,
                     error: Optional[str] = None):
        """Log API call for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'api': api_name,
            'endpoint': endpoint,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            'success': 200 <= status_code < 300,
            'error': error
        }
        
        # Log to audit file
        if log_entry['success']:
            self.logger.info(f"API Call: {json.dumps(log_entry)}")
        else:
            self.logger.error(f"API Call Failed: {json.dumps(log_entry)}")
        
        # Also save to structured log file
        api_log_file = f"logs/audit/api_calls_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(api_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_trade_action(self, action: str, symbol: str, 
                        quantity: int, price: float, 
                        order_id: Optional[str] = None,
                        status: str = 'pending'):
        """Log trading actions for audit"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'status': status,
            'value': quantity * price
        }
        
        self.logger.info(f"Trade Action: {json.dumps(trade_log)}")
        
        # Save to trade audit file
        trade_log_file = f"logs/audit/trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(trade_log_file, 'a') as f:
            f.write(json.dumps(trade_log) + '\n')
    
    def check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check if action is rate limited"""
        key = f"{identifier}:{action}"
        current_time = datetime.now()
        
        # Clean old entries
        self.failed_attempts = {
            k: v for k, v in self.failed_attempts.items()
            if (current_time - v['first_attempt']).total_seconds() < self.lockout_duration
        }
        
        # Check if locked out
        if key in self.failed_attempts:
            attempt_data = self.failed_attempts[key]
            if attempt_data['count'] >= self.max_failed_attempts:
                lockout_remaining = self.lockout_duration - (current_time - attempt_data['first_attempt']).total_seconds()
                if lockout_remaining > 0:
                    self.logger.warning(f"Rate limit exceeded for {key}. Lockout: {lockout_remaining:.0f}s")
                    return False
                else:
                    # Reset after lockout period
                    del self.failed_attempts[key]
        
        return True
    
    def record_failed_attempt(self, identifier: str, action: str):
        """Record a failed attempt for rate limiting"""
        key = f"{identifier}:{action}"
        current_time = datetime.now()
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {
                'count': 1,
                'first_attempt': current_time,
                'last_attempt': current_time
            }
        else:
            self.failed_attempts[key]['count'] += 1
            self.failed_attempts[key]['last_attempt'] = current_time
        
        self.logger.warning(f"Failed attempt recorded for {key}. Count: {self.failed_attempts[key]['count']}")
    
    def generate_api_health_report(self) -> Dict[str, Any]:
        """Generate API security health report"""
        # Read recent API logs
        today = datetime.now().strftime('%Y%m%d')
        api_log_file = f"logs/audit/api_calls_{today}.jsonl"
        
        api_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0,
            'errors_by_type': {},
            'calls_by_api': {}
        }
        
        if os.path.exists(api_log_file):
            response_times = []
            
            with open(api_log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        api_stats['total_calls'] += 1
                        
                        if log_entry['success']:
                            api_stats['successful_calls'] += 1
                        else:
                            api_stats['failed_calls'] += 1
                            
                            if log_entry.get('error'):
                                error_type = log_entry['error'].split(':')[0]
                                api_stats['errors_by_type'][error_type] = \
                                    api_stats['errors_by_type'].get(error_type, 0) + 1
                        
                        # Track by API
                        api_name = log_entry['api']
                        if api_name not in api_stats['calls_by_api']:
                            api_stats['calls_by_api'][api_name] = {
                                'total': 0, 'success': 0, 'failed': 0
                            }
                        
                        api_stats['calls_by_api'][api_name]['total'] += 1
                        if log_entry['success']:
                            api_stats['calls_by_api'][api_name]['success'] += 1
                        else:
                            api_stats['calls_by_api'][api_name]['failed'] += 1
                        
                        response_times.append(log_entry['response_time_ms'])
                        
                    except json.JSONDecodeError:
                        continue
            
            if response_times:
                api_stats['avg_response_time'] = round(sum(response_times) / len(response_times), 2)
        
        # Security metrics
        security_metrics = {
            'active_sessions': len(self.active_sessions),
            'rate_limited_users': len(self.failed_attempts),
            'encrypted_keys': len(self.load_secure_config()),
            'audit_logs_size_mb': self._get_audit_logs_size()
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'api_statistics': api_stats,
            'security_metrics': security_metrics
        }
    
    def _get_audit_logs_size(self) -> float:
        """Get total size of audit logs in MB"""
        audit_dir = Path('logs/audit')
        if not audit_dir.exists():
            return 0.0
        
        total_size = sum(f.stat().st_size for f in audit_dir.glob('*') if f.is_file())
        return round(total_size / (1024 * 1024), 2)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old audit logs"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        audit_dir = Path('logs/audit')
        
        if not audit_dir.exists():
            return
        
        cleaned_count = 0
        
        for log_file in audit_dir.glob('*.jsonl'):
            # Extract date from filename
            try:
                file_date_str = log_file.stem.split('_')[-1]
                file_date = datetime.strptime(file_date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    cleaned_count += 1
            except (ValueError, IndexError):
                continue
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old log files")
    
    def display_security_status(self):
        """Display current security status"""
        print("\n" + "="*70)
        print("🔐 SECURITY MODULE STATUS")
        print("="*70)
        
        # Get health report
        health_report = self.generate_api_health_report()
        
        print(f"\n📊 API Statistics (Today):")
        api_stats = health_report['api_statistics']
        print(f"   Total Calls: {api_stats['total_calls']}")
        print(f"   Successful: {api_stats['successful_calls']} ({api_stats['successful_calls']/max(1, api_stats['total_calls'])*100:.1f}%)")
        print(f"   Failed: {api_stats['failed_calls']}")
        print(f"   Avg Response Time: {api_stats['avg_response_time']}ms")
        
        if api_stats['errors_by_type']:
            print(f"\n   Error Types:")
            for error_type, count in api_stats['errors_by_type'].items():
                print(f"      • {error_type}: {count}")
        
        print(f"\n🔒 Security Metrics:")
        security = health_report['security_metrics']
        print(f"   Active Sessions: {security['active_sessions']}")
        print(f"   Rate Limited IPs: {security['rate_limited_users']}")
        print(f"   Encrypted API Keys: {security['encrypted_keys']}")
        print(f"   Audit Log Size: {security['audit_logs_size_mb']} MB")
        
        print("\n✅ Security module is active and monitoring")
        print("="*70)


# Test the module
if __name__ == "__main__":
    print("🧪 Testing Security Module...")
    
    # Initialize security module
    security = SecurityModule()
    
    # Test 1: Encrypt/Decrypt API keys
    print("\n1️⃣ Testing API key encryption...")
    test_key = "test_api_key_12345"
    encrypted = security.encrypt_api_key(test_key, "TEST_API")
    print(f"   Encrypted: {encrypted[:20]}...")
    
    decrypted = security.decrypt_api_key(encrypted, "TEST_API")
    print(f"   Decrypted: {decrypted}")
    print(f"   ✅ Encryption working: {test_key == decrypted}")
    
    # Test 2: Request signing
    print("\n2️⃣ Testing request signing...")
    headers = security.sign_request("GET", "/api/v1/orders", api_secret="secret123")
    print(f"   Signature headers generated:")
    for key, value in headers.items():
        print(f"      {key}: {value}")
    
    # Test 3: Session management
    print("\n3️⃣ Testing session management...")
    session_id = security.create_session("user123")
    print(f"   Session created: {session_id[:16]}...")
    
    valid = security.validate_session(session_id)
    print(f"   Session valid: {valid is not None}")
    
    # Test 4: Audit logging
    print("\n4️⃣ Testing audit logging...")
    security.log_api_call("DHAN_API", "/orders", 200, 0.125)
    security.log_api_call("DHAN_API", "/positions", 401, 0.089, "Unauthorized")
    security.log_trade_action("BUY", "NIFTY25000CE", 50, 150.50, "ORD123456", "executed")
    print("   ✅ Audit logs created")
    
    # Test 5: Display security status
    security.display_security_status()
    
    print("\n✅ Security Module ready for integration!")