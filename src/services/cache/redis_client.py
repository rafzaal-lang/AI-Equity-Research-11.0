import redis
import json
import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

_r = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True
)

def get_json(key: str) -> Optional[Any]:
    """Get JSON value from Redis."""
    try:
        value = _r.get(key)
        if value is None:
            return None
        return json.loads(value)
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Error getting JSON from Redis key {key}: {e}")
        return None

def set_json(key: str, value: Any, ttl: int = 3600) -> bool:
    """Set JSON value in Redis with TTL."""
    try:
        json_value = json.dumps(value)
        return _r.setex(key, ttl, json_value)
    except (redis.RedisError, json.JSONEncodeError) as e:
        logger.error(f"Error setting JSON in Redis key {key}: {e}")
        return False

def delete_key(key: str) -> bool:
    """Delete key from Redis."""
    try:
        return bool(_r.delete(key))
    except redis.RedisError as e:
        logger.error(f"Error deleting Redis key {key}: {e}")
        return False

def exists(key: str) -> bool:
    """Check if key exists in Redis."""
    try:
        return bool(_r.exists(key))
    except redis.RedisError as e:
        logger.error(f"Error checking Redis key existence {key}: {e}")
        return False

def ping() -> bool:
    """Ping Redis to check connection."""
    try:
        return _r.ping()
    except Exception:
        return False

# Expose a raw client for modules that expect a client-like interface
redis_client = _r

# Optional convenience wrappers to mirror client usage in health checks
def set(key: str, value: str, ex: int | None = None): 
    return _r.set(key, value, ex=ex)

def get(key: str) -> str | None: 
    return _r.get(key)

def delete(key: str): 
    return _r.delete(key)

def info() -> dict: 
    return _r.info()

