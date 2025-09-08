# src/services/fmp_client.py
import os, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = os.getenv("FMP_BASE", "https://financialmodelingprep.com/api/v3")
APIKEY = os.getenv("FMP_API_KEY", "")

_session = requests.Session()
_session.headers.update({"User-Agent": "equity-ui/1.0"})
retry = Retry(
    total=3, backoff_factor=0.5,
    status_forcelist=[429,500,502,503,504],
    allowed_methods=frozenset(["GET","POST"])
)
_session.mount("https://", HTTPAdapter(max_retries=retry))
_session.mount("http://",  HTTPAdapter(max_retries=retry))

def get(path: str, params: dict | None = None, timeout: int = 30):
    """Safe GET: always adds apikey as a real query param; raises clear errors."""
    if not APIKEY:
        raise RuntimeError("FMP_API_KEY missing")
    q = {**(params or {}), "apikey": APIKEY}
    url = f"{BASE}/{path.lstrip('/')}"
    r = _session.get(url, params=q, timeout=timeout)
    if r.status_code in (401, 403):
        raise RuntimeError(f"FMP auth failed {r.status_code}: {r.text[:200]}")
    if r.status_code == 429:
        raise RuntimeError("FMP rate-limited (429)")
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise RuntimeError(f"FMP non-JSON at {url}: {r.text[:200]}")
