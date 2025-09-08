import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import requests
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log, after_log
)
from src.services.cache.redis_client import get_json, set_json

logger = logging.getLogger(__name__)

# Configuration
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api")
FMP_RATE_LIMIT = int(os.getenv("FMP_RATE_LIMIT", "300"))  # requests per minute
FMP_TIMEOUT = int(os.getenv("FMP_TIMEOUT", "30"))

# Global session with connection pooling
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'EquityResearch/1.0',
    'Accept': 'application/json',
    'Connection': 'keep-alive'
})

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class DataValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality: DataQuality
    errors: List[str]
    warnings: List[str]
    completeness_score: float
    freshness_score: float
    metadata: Dict[str, Any]

@dataclass
class APICallMetrics:
    """Metrics for an API call."""
    endpoint: str
    symbol: str
    start_time: float
    end_time: float
    success: bool
    cached: bool
    response_size: int = 0
    error_message: str = ""
    rate_limited: bool = False

class FMPError(Exception):
    """Custom FMP API error."""
    pass

class RateLimitError(FMPError):
    """Rate limit exceeded error."""
    pass

class DataValidationError(FMPError):
    """Data validation error."""
    pass

class EnhancedFMPProvider:
    """Enhanced FMP provider with comprehensive error handling and validation."""
    
    def __init__(self):
        self.api_key = FMP_API_KEY
        self.base_url = FMP_BASE_URL
        self.session = SESSION
        self.call_metrics: List[APICallMetrics] = []
        self.max_metrics_history = 1000
        self.rate_limiter = self._init_rate_limiter()
        
        if not self.api_key:
            raise FMPError("FMP_API_KEY environment variable is required")

    def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory leaks."""
        if len(self.call_metrics) > self.max_metrics_history:
            self.call_metrics = self.call_metrics[-self.max_metrics_history:]
        
    def _init_rate_limiter(self) -> Dict[str, Any]:
        """Initialize rate limiter."""
        return {
            "calls": [],
            "max_calls_per_minute": FMP_RATE_LIMIT,
            "window_size": 60  # seconds
        }
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        now = time.time()
        window_start = now - self.rate_limiter["window_size"]
        
        # Remove old calls outside the window
        self.rate_limiter["calls"] = [
            call_time for call_time in self.rate_limiter["calls"] 
            if call_time > window_start
        ]
        
        # Check if we're at the limit
        if len(self.rate_limiter["calls"]) >= self.rate_limiter["max_calls_per_minute"]:
            sleep_time = self.rate_limiter["calls"][0] - window_start + 1
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            raise RateLimitError("Rate limit exceeded")
        
        # Record this call
        self.rate_limiter["calls"].append(now)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.5, max=10),
        retry=retry_if_exception_type((requests.RequestException, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    def _make_request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, APICallMetrics]:
        """Make API request with comprehensive error handling."""
        if params is None:
            params = {}
        
        # Add API key
        params = {**params, "apikey": self.api_key}
        url = f"{self.base_url}/v3/{path.lstrip('/')}"
        
        # Extract symbol for metrics (if present in path or params)
        symbol = params.get("symbol", path.split("/")[-1] if "/" in path else "unknown")
        
        start_time = time.time()
        metrics = APICallMetrics(
            endpoint=path,
            symbol=symbol,
            start_time=start_time,
            end_time=0,
            success=False,
            cached=False
        )
        
        try:
            # Check rate limits
            self._check_rate_limit()
            
            # Make request
            response = self.session.get(url, params=params, timeout=FMP_TIMEOUT)
            end_time = time.time()
            
            # Update metrics
            metrics.end_time = end_time
            metrics.response_size = len(response.content) if response.content else 0
            
            # Handle HTTP errors
            if response.status_code == 429:
                metrics.rate_limited = True
                raise RateLimitError("API rate limit exceeded")
            
            response.raise_for_status()
            
            # Parse JSON
            try:
                data = response.json()
            except ValueError as e:
                raise FMPError(f"Invalid JSON response: {e}")
            
            # Check for API-level errors
            if isinstance(data, dict) and "Error Message" in data:
                raise FMPError(f"API Error: {data['Error Message']}")
            
            metrics.success = True
            self.call_metrics.append(metrics)
            self._cleanup_old_metrics()
            
            logger.debug(f"API call successful: {path} ({end_time - start_time:.2f}s)")
            return data, metrics
            
        except requests.exceptions.Timeout:
            metrics.end_time = time.time()
            metrics.error_message = "Request timeout"
            self.call_metrics.append(metrics)
            self._cleanup_old_metrics()
            raise FMPError("Request timeout")
        except requests.exceptions.ConnectionError:
            metrics.end_time = time.time()
            metrics.error_message = "Connection error"
            self.call_metrics.append(metrics)
            self._cleanup_old_metrics()
            raise FMPError("Connection error")
        except requests.exceptions.HTTPError as e:
            metrics.end_time = time.time()
            metrics.error_message = str(e)
            self.call_metrics.append(metrics)
            self._cleanup_old_metrics()
            
            if response.status_code == 401:
                raise FMPError("Invalid API key")
            elif response.status_code == 403:
                raise FMPError("API access forbidden")
            elif response.status_code == 404:
                raise FMPError(f"Endpoint not found: {path}")
            else:
                raise FMPError(f"HTTP {response.status_code}: {e}")
        except Exception as e:
            metrics.end_time = time.time()
            metrics.error_message = str(e)
            self.call_metrics.append(metrics)
            self._cleanup_old_metrics()
            raise
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get provider health metrics."""
        if not self.call_metrics:
            return {"status": "no_data", "metrics": {}}
        
        recent_calls = [m for m in self.call_metrics if time.time() - m.end_time < 3600]  # Last hour
        
        if not recent_calls:
            return {"status": "no_recent_data", "metrics": {}}
        
        success_rate = sum(1 for m in recent_calls if m.success) / len(recent_calls)
        avg_response_time = sum(m.end_time - m.start_time for m in recent_calls) / len(recent_calls)
        cache_hit_rate = sum(1 for m in recent_calls if m.cached) / len(recent_calls)
        
        return {
            "status": "healthy" if success_rate > 0.95 else "degraded" if success_rate > 0.8 else "unhealthy",
            "metrics": {
                "total_calls_last_hour": len(recent_calls),
                "success_rate": round(success_rate, 3),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "rate_limit_hits": sum(1 for m in recent_calls if m.rate_limited),
                "error_rate": round(1 - success_rate, 3)
            }
        }

# Global instance
enhanced_fmp_provider = EnhancedFMPProvider()
