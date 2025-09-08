import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
from functools import wraps

# Create logs directory
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class ContextualLogger:
    """Logger with contextual information."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set contextual information."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear contextual information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context."""
        extra_fields = {**self.context, **kwargs}
        extra = {"extra_fields": extra_fields} if extra_fields else {}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)

def setup_logging(log_level: str = "INFO", enable_json: bool = True):
    """Setup comprehensive logging configuration."""
    
    # Convert log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers
    
    # General application log
    app_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "application.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    app_handler.setLevel(numeric_level)
    app_handler.setFormatter(JSONFormatter() if enable_json else console_formatter)
    root_logger.addHandler(app_handler)
    
    # Error log
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "errors.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter() if enable_json else console_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "performance.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(JSONFormatter() if enable_json else console_formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False
    
    # Security log
    security_logger = logging.getLogger("security")
    security_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "security.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(JSONFormatter() if enable_json else console_formatter)
    security_logger.addHandler(security_handler)
    security_logger.propagate = False
    
    # Audit log
    audit_logger = logging.getLogger("audit")
    audit_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "audit.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=20
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(JSONFormatter() if enable_json else console_formatter)
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False
    
    logging.info("Logging system initialized", extra={
        "extra_fields": {
            "log_level": log_level,
            "json_enabled": enable_json,
            "log_directory": str(LOG_DIR)
        }
    })

def log_performance(func):
    """Decorator to log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("performance")
        start_time = datetime.utcnow()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("Function executed successfully", extra={
                "extra_fields": {
                    "function": func.__name__,
                    "module": func.__module__,
                    "duration_seconds": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "success": True
                }
            })
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.error("Function execution failed", extra={
                "extra_fields": {
                    "function": func.__name__,
                    "module": func.__module__,
                    "duration_seconds": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            })
            
            raise
    
    return wrapper

def log_api_request(func):
    """Decorator to log API requests."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("audit")
        start_time = datetime.utcnow()
        
        # Extract request info (assuming Flask request context)
        try:
            from flask import request, g
            request_info = {
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get("User-Agent"),
                "content_length": request.content_length
            }
            
            # Add user info if available
            if hasattr(g, 'user_id'):
                request_info["user_id"] = g.user_id
                
        except (ImportError, RuntimeError):
            # Not in Flask context
            request_info = {}
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("API request completed", extra={
                "extra_fields": {
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "success": True,
                    **request_info
                }
            })
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.error("API request failed", extra={
                "extra_fields": {
                    "function": func.__name__,
                    "duration_seconds": duration,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    **request_info
                }
            })
            
            raise
    
    return wrapper

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO"):
    """Log security-related events."""
    logger = logging.getLogger("security")
    
    log_entry = {
        "event_type": event_type,
        "severity": severity,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if severity.upper() == "CRITICAL":
        logger.critical("Security event", extra={"extra_fields": log_entry})
    elif severity.upper() == "ERROR":
        logger.error("Security event", extra={"extra_fields": log_entry})
    elif severity.upper() == "WARNING":
        logger.warning("Security event", extra={"extra_fields": log_entry})
    else:
        logger.info("Security event", extra={"extra_fields": log_entry})

def log_audit_event(action: str, resource: str, user_id: Optional[str] = None, 
                   details: Optional[Dict[str, Any]] = None):
    """Log audit events for compliance."""
    logger = logging.getLogger("audit")
    
    audit_entry = {
        "action": action,
        "resource": resource,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    logger.info("Audit event", extra={"extra_fields": audit_entry})

# Global contextual loggers
app_logger = ContextualLogger("app")
data_logger = ContextualLogger("data")
model_logger = ContextualLogger("model")
api_logger = ContextualLogger("api")

# Initialize logging on import
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_json=os.getenv("LOG_FORMAT", "json").lower() == "json"
)

