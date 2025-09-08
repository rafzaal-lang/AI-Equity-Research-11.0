import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram

REQ_COUNT = Counter("svc_requests_total", "Total requests", ["route", "status"])
REQ_LAT   = Histogram("svc_request_latency_seconds", "Request latency", ["route"])

class Timer:
    def __init__(self, route: str):
        self.route = route
        self.start = None
    def __enter__(self):
        self.start = time.perf_counter()
    def __exit__(self, exc_type, exc, tb):
        REQ_LAT.labels(self.route).observe(time.perf_counter() - self.start)

def record(route: str, status: str):
    REQ_COUNT.labels(route, status).inc()

def snapshot() -> Dict[str, Any]:
    return {"requests": "prometheus-counters"}

