import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.services.providers.enhanced_fmp_provider import enhanced_fmp_provider, DataQuality
from src.services.cache.redis_client import get_json, set_json

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class DataSourceHealth:
    """Health metrics for a data source."""
    source_name: str
    status: PipelineStatus
    success_rate: float
    avg_response_time: float
    cache_hit_rate: float
    error_count: int
    last_successful_call: Optional[datetime]
    data_quality_score: float
    freshness_score: float
    issues: List[str]

@dataclass
class PipelineAlert:
    """Alert for pipeline issues."""
    alert_id: str
    severity: str  # "warning", "error", "critical"
    source: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class DataPipelineMonitor:
    """Monitor data pipeline health and quality."""
    
    def __init__(self):
        self.alert_history: List[PipelineAlert] = []
        self.health_history: List[Dict[str, Any]] = []
        self.thresholds = {
            "success_rate_warning": 0.95,
            "success_rate_error": 0.85,
            "response_time_warning": 5.0,  # seconds
            "response_time_error": 10.0,
            "cache_hit_rate_warning": 0.7,
            "data_quality_warning": 0.8,
            "data_quality_error": 0.6,
            "freshness_warning": 0.7,
            "freshness_error": 0.5
        }
    
    def check_fmp_health(self) -> DataSourceHealth:
        """Check FMP provider health."""
        try:
            health_metrics = enhanced_fmp_provider.get_health_metrics()
            
            if health_metrics["status"] == "no_data":
                return DataSourceHealth(
                    source_name="FMP",
                    status=PipelineStatus.CRITICAL,
                    success_rate=0.0,
                    avg_response_time=0.0,
                    cache_hit_rate=0.0,
                    error_count=0,
                    last_successful_call=None,
                    data_quality_score=0.0,
                    freshness_score=0.0,
                    issues=["No API calls recorded"]
                )
            
            metrics = health_metrics.get("metrics", {})
            success_rate = metrics.get("success_rate", 0.0)
            response_time = metrics.get("avg_response_time_ms", 0.0) / 1000.0
            cache_hit_rate = metrics.get("cache_hit_rate", 0.0)
            error_rate = metrics.get("error_rate", 0.0)
            
            # Determine status
            if success_rate >= self.thresholds["success_rate_warning"] and response_time <= self.thresholds["response_time_warning"]:
                status = PipelineStatus.HEALTHY
            elif success_rate >= self.thresholds["success_rate_error"] and response_time <= self.thresholds["response_time_error"]:
                status = PipelineStatus.DEGRADED
            elif success_rate > 0.5:
                status = PipelineStatus.UNHEALTHY
            else:
                status = PipelineStatus.CRITICAL
            
            # Identify issues
            issues = []
            if success_rate < self.thresholds["success_rate_warning"]:
                issues.append(f"Low success rate: {success_rate:.1%}")
            if response_time > self.thresholds["response_time_warning"]:
                issues.append(f"High response time: {response_time:.2f}s")
            if cache_hit_rate < self.thresholds["cache_hit_rate_warning"]:
                issues.append(f"Low cache hit rate: {cache_hit_rate:.1%}")
            if error_rate > 0.1:
                issues.append(f"High error rate: {error_rate:.1%}")
            
            return DataSourceHealth(
                source_name="FMP",
                status=status,
                success_rate=success_rate,
                avg_response_time=response_time,
                cache_hit_rate=cache_hit_rate,
                error_count=int(metrics.get("total_calls_last_hour", 0) * error_rate),
                last_successful_call=datetime.utcnow(),  # Approximate
                data_quality_score=0.9,  # Would need to track this separately
                freshness_score=0.9,     # Would need to track this separately
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error checking FMP health: {e}")
            return DataSourceHealth(
                source_name="FMP",
                status=PipelineStatus.CRITICAL,
                success_rate=0.0,
                avg_response_time=0.0,
                cache_hit_rate=0.0,
                error_count=1,
                last_successful_call=None,
                data_quality_score=0.0,
                freshness_score=0.0,
                issues=[f"Health check failed: {str(e)}"]
            )
    
    def check_data_quality(self, symbol: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Check data quality for a specific symbol."""
        if data_types is None:
            data_types = ["income_statement", "balance_sheet", "cash_flow"]
        
        quality_results = {}
        overall_score = 0.0
        total_weight = 0
        
        for data_type in data_types:
            try:
                if data_type == "income_statement":
                    result = enhanced_fmp_provider.income_statement(symbol, validate=True)
                elif data_type == "balance_sheet":
                    result = enhanced_fmp_provider.balance_sheet(symbol, validate=True)
                elif data_type == "cash_flow":
                    result = enhanced_fmp_provider.cash_flow(symbol, validate=True)
                else:
                    continue
                
                if "validation" in result:
                    validation = result["validation"]
                    quality_results[data_type] = {
                        "is_valid": validation.is_valid,
                        "quality": validation.quality.value,
                        "completeness_score": validation.completeness_score,
                        "freshness_score": validation.freshness_score,
                        "errors": validation.errors,
                        "warnings": validation.warnings
                    }
                    
                    # Weight income statement higher
                    weight = 2 if data_type == "income_statement" else 1
                    overall_score += validation.completeness_score * weight
                    total_weight += weight
                
            except Exception as e:
                logger.error(f"Error checking {data_type} quality for {symbol}: {e}")
                quality_results[data_type] = {
                    "is_valid": False,
                    "quality": "invalid",
                    "error": str(e)
                }
        
        overall_score = overall_score / total_weight if total_weight > 0 else 0.0
        
        return {
            "symbol": symbol,
            "overall_quality_score": overall_score,
            "data_types": quality_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": PipelineStatus.HEALTHY.value,
            "data_sources": {},
            "alerts": [],
            "summary": {}
        }
        
        # Check FMP health
        fmp_health = self.check_fmp_health()
        health_report["data_sources"]["FMP"] = asdict(fmp_health)
        
        # Determine overall status
        if fmp_health.status == PipelineStatus.CRITICAL:
            health_report["overall_status"] = PipelineStatus.CRITICAL.value
        elif fmp_health.status == PipelineStatus.UNHEALTHY:
            health_report["overall_status"] = PipelineStatus.UNHEALTHY.value
        elif fmp_health.status == PipelineStatus.DEGRADED:
            health_report["overall_status"] = PipelineStatus.DEGRADED.value
        
        # Generate alerts
        alerts = self._generate_alerts(fmp_health)
        health_report["alerts"] = [asdict(alert) for alert in alerts]
        
        # Create summary
        health_report["summary"] = {
            "total_data_sources": 1,
            "healthy_sources": 1 if fmp_health.status == PipelineStatus.HEALTHY else 0,
            "degraded_sources": 1 if fmp_health.status == PipelineStatus.DEGRADED else 0,
            "unhealthy_sources": 1 if fmp_health.status == PipelineStatus.UNHEALTHY else 0,
            "critical_sources": 1 if fmp_health.status == PipelineStatus.CRITICAL else 0,
            "active_alerts": len(alerts)
        }
        
        # Store health history
        self.health_history.append(health_report)
        
        # Keep only last 24 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.health_history = [
            h for h in self.health_history 
            if datetime.fromisoformat(h["timestamp"]) > cutoff_time
        ]
        
        # Cache the report
        set_json("pipeline:health:latest", health_report, ttl=300)  # 5 minutes
        
        return health_report
    
    def _generate_alerts(self, health: DataSourceHealth) -> List[PipelineAlert]:
        """Generate alerts based on health metrics."""
        alerts = []
        timestamp = datetime.utcnow()
        
        # Success rate alerts
        if health.success_rate < self.thresholds["success_rate_error"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_success_rate_{int(timestamp.timestamp())}",
                severity="error",
                source="FMP",
                message=f"Low success rate: {health.success_rate:.1%}",
                timestamp=timestamp
            ))
        elif health.success_rate < self.thresholds["success_rate_warning"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_success_rate_{int(timestamp.timestamp())}",
                severity="warning",
                source="FMP",
                message=f"Success rate below threshold: {health.success_rate:.1%}",
                timestamp=timestamp
            ))
        
        # Response time alerts
        if health.avg_response_time > self.thresholds["response_time_error"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_response_time_{int(timestamp.timestamp())}",
                severity="error",
                source="FMP",
                message=f"High response time: {health.avg_response_time:.2f}s",
                timestamp=timestamp
            ))
        elif health.avg_response_time > self.thresholds["response_time_warning"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_response_time_{int(timestamp.timestamp())}",
                severity="warning",
                source="FMP",
                message=f"Response time above threshold: {health.avg_response_time:.2f}s",
                timestamp=timestamp
            ))
        
        # Data quality alerts
        if health.data_quality_score < self.thresholds["data_quality_error"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_data_quality_{int(timestamp.timestamp())}",
                severity="error",
                source="FMP",
                message=f"Low data quality score: {health.data_quality_score:.1%}",
                timestamp=timestamp
            ))
        elif health.data_quality_score < self.thresholds["data_quality_warning"]:
            alerts.append(PipelineAlert(
                alert_id=f"fmp_data_quality_{int(timestamp.timestamp())}",
                severity="warning",
                source="FMP",
                message=f"Data quality below threshold: {health.data_quality_score:.1%}",
                timestamp=timestamp
            ))
        
        return alerts
    
    def get_historical_health(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical health data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            h for h in self.health_history 
            if datetime.fromisoformat(h["timestamp"]) > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[PipelineAlert]:
        """Get active (unresolved) alerts."""
        return [alert for alert in self.alert_history if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alert_history:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                return True
        return False
    
    def test_data_pipeline(self, test_symbols: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive data pipeline test."""
        if test_symbols is None:
            test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_symbols": test_symbols,
            "results": {},
            "overall_success": True,
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "avg_response_time": 0.0
            }
        }
        
        total_response_time = 0.0
        
        for symbol in test_symbols:
            symbol_results = {
                "symbol": symbol,
                "tests": {},
                "success": True
            }
            
            # Test each data type
            for data_type in ["income_statement", "balance_sheet", "cash_flow"]:
                start_time = time.time()
                
                try:
                    if data_type == "income_statement":
                        result = enhanced_fmp_provider.income_statement(symbol, limit=1, validate=True)
                    elif data_type == "balance_sheet":
                        result = enhanced_fmp_provider.balance_sheet(symbol, limit=1, validate=True)
                    elif data_type == "cash_flow":
                        result = enhanced_fmp_provider.cash_flow(symbol, limit=1, validate=True)
                    
                    response_time = time.time() - start_time
                    total_response_time += response_time
                    
                    success = "error" not in result
                    if success and "validation" in result:
                        success = result["validation"].is_valid
                    
                    symbol_results["tests"][data_type] = {
                        "success": success,
                        "response_time": response_time,
                        "cached": result.get("cached", False),
                        "validation": result.get("validation", {}) if success else None,
                        "error": result.get("error") if not success else None
                    }
                    
                    test_results["summary"]["total_tests"] += 1
                    if success:
                        test_results["summary"]["passed_tests"] += 1
                    else:
                        test_results["summary"]["failed_tests"] += 1
                        symbol_results["success"] = False
                        test_results["overall_success"] = False
                
                except Exception as e:
                    response_time = time.time() - start_time
                    total_response_time += response_time
                    
                    symbol_results["tests"][data_type] = {
                        "success": False,
                        "response_time": response_time,
                        "error": str(e)
                    }
                    
                    test_results["summary"]["total_tests"] += 1
                    test_results["summary"]["failed_tests"] += 1
                    symbol_results["success"] = False
                    test_results["overall_success"] = False
            
            test_results["results"][symbol] = symbol_results
        
        # Calculate average response time
        if test_results["summary"]["total_tests"] > 0:
            test_results["summary"]["avg_response_time"] = total_response_time / test_results["summary"]["total_tests"]
        
        return test_results

# Global instance
data_pipeline_monitor = DataPipelineMonitor()

