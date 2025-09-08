import os
from typing import Any, Dict, List, Optional
import requests
from src.services.cache.redis_client import get_json, set_json

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def get_series(series_id: str, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
    if not FRED_API_KEY:
        raise RuntimeError("Missing FRED_API_KEY")
    params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json"}
    if start: params["observation_start"] = start
    if end: params["observation_end"] = end
    key = f"fred:{series_id}:{start}:{end}"
    cached = get_json(key)
    if cached is not None: return cached
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    set_json(key, data, ttl=3600)
    return data

