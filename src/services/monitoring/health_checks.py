import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp
import redis
from pathlib import Path

from src.services.providers.enhanced_fmp_provider import enhanced_fmp_provider
from src.services.data_pipeline_monitor import data_pipeline_monitor
from src.services.cache.redis_client import redis_client

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float

@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.register_default_checks()
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self.check_system_resources)
        self.register_check("redis_connection", self.check_redis_connection)
        self.register_check("fmp_api", self.check_fmp_api)
        self.register_check("data_pipeline", self.check_data_pipeline)
        self.register_check("disk_space", self.check_disk_space)
        self.register_check("memory_usage", self.check_memory_usage)
        self.register_check("cpu_usage", self.check_cpu_usage)
    
    async def run_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return HealthCheckResult(
                name=name,
                status=result.get("status", HealthStatus.HEALTHY),
                message=result.get("message", "Check passed"),
                details=result.get("details", {}),
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            logger.error(f"Health check {name} failed: {e}")
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        start_time = time.time()
        
        # Run all checks concurrently
        tasks = [
            self.run_check(name, check_func) 
            for name, check_func in self.checks.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions from gather
        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = list(self.checks.keys())[i]
                check_results.append(HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check execution failed: {str(result)}",
                    details={"error": str(result)},
                    timestamp=datetime.utcnow(),
                    response_time_ms=0
                ))
            else:
                check_results.append(result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(check_results)
        
        # Create summary
        summary = self._create_summary(check_results, time.time() - start_time)
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=check_results,
            summary=summary
        )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual check results."""
        if not results:
            return HealthStatus.CRITICAL
        
        statuses = [result.status for result in results]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _create_summary(self, results: List[HealthCheckResult], total_time: float) -> Dict[str, Any]:
        """Create summary of health check results."""
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(1 for r in results if r.status == status)
        
        return {
            "total_checks": len(results),
            "status_counts": status_counts,
            "total_time_ms": total_time * 1000,
            "avg_response_time_ms": sum(r.response_time_ms for r in results) / len(results) if results else 0,
            "slowest_check": max(results, key=lambda r: r.response_time_ms).name if results else None
        }
    
    # Individual health check implementations
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check overall system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.CRITICAL
                message = "Critical resource usage detected"
            elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
                status = HealthStatus.UNHEALTHY
                message = "High resource usage detected"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
                status = HealthStatus.DEGRADED
                message = "Elevated resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Failed to check system resources: {e}",
                "details": {"error": str(e)}
            }
    
    def check_redis_connection(self) -> Dict[str, Any]:
        """Check Redis connection and performance."""
        try:
            start_time = time.time()
            
            # Test basic operations
            test_key = "health_check_test"
            redis_client.set(test_key, "test_value", ex=60)
            retrieved_value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            if retrieved_value != "test_value":
                return {
                    "status": HealthStatus.CRITICAL,
                    "message": "Redis data integrity check failed",
                    "details": {"response_time_ms": response_time}
                }
            
            # Check response time
            if response_time > 1000:  # 1 second
                status = HealthStatus.UNHEALTHY
                message = "Redis response time is slow"
            elif response_time > 500:  # 500ms
                status = HealthStatus.DEGRADED
                message = "Redis response time is elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis connection healthy"
            
            # Get Redis info
            info = redis_client.info()
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "response_time_ms": response_time,
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_mb": info.get("used_memory", 0) / (1024**2),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Redis connection failed: {e}",
                "details": {"error": str(e)}
            }
    
    def check_fmp_api(self) -> Dict[str, Any]:
        """Check FMP API health."""
        try:
            health_metrics = enhanced_fmp_provider.get_health_metrics()
            
            if health_metrics["status"] == "no_data":
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": "No recent FMP API calls",
                    "details": health_metrics
                }
            
            metrics = health_metrics.get("metrics", {})
            success_rate = metrics.get("success_rate", 0)
            
            if success_rate >= 0.95:
                status = HealthStatus.HEALTHY
                message = "FMP API performing well"
            elif success_rate >= 0.85:
                status = HealthStatus.DEGRADED
                message = "FMP API performance degraded"
            elif success_rate >= 0.5:
                status = HealthStatus.UNHEALTHY
                message = "FMP API performance poor"
            else:
                status = HealthStatus.CRITICAL
                message = "FMP API critical issues"
            
            return {
                "status": status,
                "message": message,
                "details": health_metrics
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"FMP API health check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def check_data_pipeline(self) -> Dict[str, Any]:
        """Check data pipeline health."""
        try:
            pipeline_health = data_pipeline_monitor.run_health_check()
            
            overall_status = pipeline_health["overall_status"]
            
            if overall_status == "healthy":
                status = HealthStatus.HEALTHY
            elif overall_status == "degraded":
                status = HealthStatus.DEGRADED
            elif overall_status == "unhealthy":
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.CRITICAL
            
            return {
                "status": status,
                "message": f"Data pipeline status: {overall_status}",
                "details": pipeline_health
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Data pipeline health check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 5:
                status = HealthStatus.CRITICAL
                message = "Critical: Very low disk space"
            elif free_percent < 10:
                status = HealthStatus.UNHEALTHY
                message = "Warning: Low disk space"
            elif free_percent < 20:
                status = HealthStatus.DEGRADED
                message = "Caution: Disk space getting low"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space adequate"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "free_percent": free_percent
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Disk space check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = "Critical: Very high memory usage"
            elif memory.percent > 85:
                status = HealthStatus.UNHEALTHY
                message = "Warning: High memory usage"
            elif memory.percent > 75:
                status = HealthStatus.DEGRADED
                message = "Caution: Elevated memory usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent,
                    "cached_gb": memory.cached / (1024**3) if hasattr(memory, 'cached') else 0
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Memory check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            if cpu_percent > 95:
                status = HealthStatus.CRITICAL
                message = "Critical: Very high CPU usage"
            elif cpu_percent > 85:
                status = HealthStatus.UNHEALTHY
                message = "Warning: High CPU usage"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = "Caution: Elevated CPU usage"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_avg_1min": load_avg[0],
                    "load_avg_5min": load_avg[1],
                    "load_avg_15min": load_avg[2]
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"CPU check failed: {e}",
                "details": {"error": str(e)}
            }

# Global health checker instance
health_checker = HealthChecker()

# Convenience functions for Flask integration
def get_health_status() -> Dict[str, Any]:
    """Get current health status (synchronous)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        health = loop.run_until_complete(health_checker.run_all_checks())
        return asdict(health)
    finally:
        loop.close()

def get_readiness_status() -> Dict[str, Any]:
    """Get readiness status (essential services only)."""
    essential_checks = ["redis_connection", "fmp_api"]
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        tasks = [
            health_checker.run_check(name, health_checker.checks[name])
            for name in essential_checks
            if name in health_checker.checks
        ]
        
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        overall_status = health_checker._determine_overall_status(results)
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [asdict(result) for result in results]
        }
    finally:
        loop.close()

def get_liveness_status() -> Dict[str, Any]:
    """Get liveness status (basic application health)."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Application is running"
    }

