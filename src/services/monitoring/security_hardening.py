import os
import time
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict, deque
import ipaddress
import re

from src.services.monitoring.enhanced_logging import log_security_event

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str
    source_ip: str
    user_agent: Optional[str]
    endpoint: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for the identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        request_times = self.requests[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if under limit
        if len(request_times) < self.max_requests:
            request_times.append(now)
            return True
        
        return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        request_times = self.requests[identifier]
        # Clean old requests
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        return max(0, self.max_requests - len(request_times))

class SecurityHardening:
    """Security hardening and monitoring system."""
    
    def __init__(self):
        self.rate_limiters: Dict[str, RateLimiter] = {
            "api": RateLimiter(max_requests=1000, window_seconds=3600),  # 1000/hour
            "auth": RateLimiter(max_requests=10, window_seconds=300),    # 10/5min
            "data": RateLimiter(max_requests=100, window_seconds=60)     # 100/min
        }
        
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Security configuration
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.suspicious_threshold = 3
    
    def _load_suspicious_patterns(self) -> List[re.Pattern]:
        """Load patterns for detecting suspicious requests."""
        patterns = [
            # SQL injection patterns
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
            r"(\b(or|and)\s+\d+\s*=\s*\d+)",
            r"(\b(or|and)\s+['\"].*['\"])",
            
            # XSS patterns
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            
            # Path traversal
            r"\.\./",
            r"\.\.\\",
            
            # Command injection
            r"[;&|`]",
            r"\$\(",
            
            # Common attack strings
            r"(eval|exec|system|shell_exec)\s*\(",
            r"(base64_decode|gzinflate|str_rot13)",
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def check_rate_limit(self, identifier: str, limit_type: str = "api") -> bool:
        """Check if request is within rate limits."""
        if limit_type not in self.rate_limiters:
            return True
        
        return self.rate_limiters[limit_type].is_allowed(identifier)
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, reason: str):
        """Block an IP address."""
        self.blocked_ips.add(ip_address)
        
        log_security_event(
            event_type="ip_blocked",
            details={
                "ip_address": ip_address,
                "reason": reason
            },
            severity="WARNING"
        )
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip_address)
        
        log_security_event(
            event_type="ip_unblocked",
            details={
                "ip_address": ip_address
            },
            severity="INFO"
        )
    
    def record_failed_attempt(self, identifier: str, ip_address: str):
        """Record a failed authentication attempt."""
        now = datetime.utcnow()
        
        # Clean old attempts
        cutoff_time = now - self.lockout_duration
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff_time
        ]
        
        # Add new attempt
        self.failed_attempts[identifier].append(now)
        
        # Check if should block
        if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
            self.block_ip(ip_address, f"Too many failed attempts for {identifier}")
            
            log_security_event(
                event_type="brute_force_detected",
                details={
                    "identifier": identifier,
                    "ip_address": ip_address,
                    "attempt_count": len(self.failed_attempts[identifier])
                },
                severity="ERROR"
            )
    
    def check_suspicious_content(self, content: str) -> List[str]:
        """Check content for suspicious patterns."""
        matches = []
        
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                matches.append(pattern.pattern)
        
        return matches
    
    def validate_input(self, data: Dict[str, Any], ip_address: str) -> bool:
        """Validate input data for security issues."""
        suspicious_findings = []
        
        def check_value(value, key_path=""):
            if isinstance(value, str):
                matches = self.check_suspicious_content(value)
                if matches:
                    suspicious_findings.extend([f"{key_path}: {match}" for match in matches])
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(v, f"{key_path}.{k}" if key_path else k)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    check_value(v, f"{key_path}[{i}]" if key_path else f"[{i}]")
        
        check_value(data)
        
        if suspicious_findings:
            log_security_event(
                event_type="suspicious_input_detected",
                details={
                    "ip_address": ip_address,
                    "findings": suspicious_findings,
                    "data_keys": list(data.keys()) if isinstance(data, dict) else []
                },
                severity="WARNING"
            )
            
            return False
        
        return True
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token."""
        secret = os.getenv("CSRF_SECRET", "default_secret_change_me")
        timestamp = str(int(time.time()))
        
        token_data = f"{session_id}:{timestamp}:{secret}"
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()
        
        return f"{timestamp}:{token_hash}"
    
    def validate_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token."""
        try:
            timestamp_str, token_hash = token.split(":", 1)
            timestamp = int(timestamp_str)
            
            # Check age
            if time.time() - timestamp > max_age:
                return False
            
            # Regenerate expected token
            secret = os.getenv("CSRF_SECRET", "default_secret_change_me")
            token_data = f"{session_id}:{timestamp_str}:{secret}"
            expected_hash = hashlib.sha256(token_data.encode()).hexdigest()
            
            return secrets.compare_digest(token_hash, expected_hash)
            
        except (ValueError, TypeError):
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    def check_file_upload(self, filename: str, content: bytes, 
                         allowed_extensions: Set[str] = None) -> Dict[str, Any]:
        """Check uploaded file for security issues."""
        if allowed_extensions is None:
            allowed_extensions = {'.txt', '.csv', '.json', '.pdf', '.png', '.jpg', '.jpeg'}
        
        issues = []
        
        # Check filename
        sanitized_name = self.sanitize_filename(filename)
        if sanitized_name != filename:
            issues.append("Filename contains dangerous characters")
        
        # Check extension
        _, ext = os.path.splitext(filename.lower())
        if ext not in allowed_extensions:
            issues.append(f"File extension {ext} not allowed")
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024
        if len(content) > max_size:
            issues.append(f"File too large: {len(content)} bytes")
        
        # Check for executable content
        if content.startswith(b'\x7fELF') or content.startswith(b'MZ'):
            issues.append("Executable file detected")
        
        # Check for script content in text files
        if ext in {'.txt', '.csv', '.json'}:
            try:
                text_content = content.decode('utf-8', errors='ignore')
                suspicious = self.check_suspicious_content(text_content)
                if suspicious:
                    issues.append(f"Suspicious content patterns: {suspicious}")
            except Exception:
                pass
        
        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "sanitized_filename": sanitized_name
        }

class AlertingSystem:
    """System for generating and managing alerts."""
    
    def __init__(self):
        self.alert_handlers: List[callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 5.0,  # 5 seconds
            "memory_usage": 0.85,  # 85% memory usage
            "disk_usage": 0.90,   # 90% disk usage
            "failed_logins": 10    # 10 failed logins in 5 minutes
        }
    
    def register_handler(self, handler: callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, alert_type: str, severity: str, message: str, 
                  details: Dict[str, Any] = None):
        """Send an alert through all registered handlers."""
        alert = {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
            "id": hashlib.md5(f"{alert_type}:{message}:{time.time()}".encode()).hexdigest()
        }
        
        # Store in history
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Send through handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        # Log the alert
        log_security_event(
            event_type="alert_generated",
            details=alert,
            severity=severity
        )
    
    def check_metrics_and_alert(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts."""
        
        # Error rate check
        if "error_rate" in metrics and metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            self.send_alert(
                "high_error_rate",
                "ERROR",
                f"Error rate {metrics['error_rate']:.2%} exceeds threshold {self.alert_thresholds['error_rate']:.2%}",
                {"current_rate": metrics["error_rate"], "threshold": self.alert_thresholds["error_rate"]}
            )
        
        # Response time check
        if "avg_response_time" in metrics and metrics["avg_response_time"] > self.alert_thresholds["response_time"]:
            self.send_alert(
                "slow_response_time",
                "WARNING",
                f"Average response time {metrics['avg_response_time']:.2f}s exceeds threshold {self.alert_thresholds['response_time']}s",
                {"current_time": metrics["avg_response_time"], "threshold": self.alert_thresholds["response_time"]}
            )
        
        # Memory usage check
        if "memory_usage_percent" in metrics and metrics["memory_usage_percent"] > self.alert_thresholds["memory_usage"]:
            self.send_alert(
                "high_memory_usage",
                "WARNING",
                f"Memory usage {metrics['memory_usage_percent']:.1%} exceeds threshold {self.alert_thresholds['memory_usage']:.1%}",
                {"current_usage": metrics["memory_usage_percent"], "threshold": self.alert_thresholds["memory_usage"]}
            )

# Default alert handlers
def console_alert_handler(alert: Dict[str, Any]):
    """Simple console alert handler."""
    print(f"ALERT [{alert['severity']}] {alert['type']}: {alert['message']}")

def log_alert_handler(alert: Dict[str, Any]):
    """Log alert handler."""
    logger.warning(f"Alert generated: {alert['type']} - {alert['message']}", extra={
        "extra_fields": alert
    })

# Global instances
security_hardening = SecurityHardening()
alerting_system = AlertingSystem()

# Register default alert handlers
alerting_system.register_handler(console_alert_handler)
alerting_system.register_handler(log_alert_handler)

# Security decorators
def require_rate_limit(limit_type: str = "api"):
    """Decorator to enforce rate limiting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract IP from Flask request context
            try:
                from flask import request
                ip_address = request.remote_addr
                
                if not security_hardening.check_rate_limit(ip_address, limit_type):
                    log_security_event(
                        event_type="rate_limit_exceeded",
                        details={
                            "ip_address": ip_address,
                            "limit_type": limit_type,
                            "endpoint": request.endpoint
                        },
                        severity="WARNING"
                    )
                    
                    from flask import jsonify
                    return jsonify({"error": "Rate limit exceeded"}), 429
                
            except (ImportError, RuntimeError):
                # Not in Flask context, skip rate limiting
                pass
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_ip_whitelist(allowed_ips: List[str] = None):
    """Decorator to enforce IP whitelisting."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if allowed_ips is None:
                return func(*args, **kwargs)
            
            try:
                from flask import request
                ip_address = request.remote_addr
                
                # Check if IP is in whitelist
                allowed = False
                for allowed_ip in allowed_ips:
                    try:
                        if ipaddress.ip_address(ip_address) in ipaddress.ip_network(allowed_ip, strict=False):
                            allowed = True
                            break
                    except ValueError:
                        if ip_address == allowed_ip:
                            allowed = True
                            break
                
                if not allowed:
                    log_security_event(
                        event_type="ip_not_whitelisted",
                        details={
                            "ip_address": ip_address,
                            "endpoint": request.endpoint
                        },
                        severity="WARNING"
                    )
                    
                    from flask import jsonify
                    return jsonify({"error": "Access denied"}), 403
                
            except (ImportError, RuntimeError):
                # Not in Flask context, skip check
                pass
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

