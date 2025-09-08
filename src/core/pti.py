# src/core/pti.py
from __future__ import annotations
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

# ---- helpers ---------------------------------------------------------------

def _parse_date(d: Any) -> Optional[datetime]:
    """Best-effort conversion of many date formats to datetime."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        return datetime(d.year, d.month, d.day)
    s = str(d).strip()
    # try fast ISO
    try:
        # handle "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
        return datetime.fromisoformat(s[:19])
    except Exception:
        pass
    # try dateutil if present
    try:
        from dateutil import parser  # type: ignore
        return parser.parse(s)
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _sorted_rows(
    rows: List[Dict[str, Any]],
    date_key: str,
    value_key: str,
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return rows with valid date+value sorted by date.
    Filters out rows with missing/invalid date or value.
    """
    cleaned: List[Tuple[datetime, float, Dict[str, Any]]] = []
    for r in rows or []:
        dt = _parse_date(r.get(date_key))
        val = _safe_float(r.get(value_key))
        if dt is None or val is None:
            continue
        cleaned.append((dt, val, r))
    cleaned.sort(key=lambda t: t[0], reverse=not ascending)
    return [r for _, _, r in cleaned]


# ---- public API ------------------------------------------------------------

def get_latest_value(
    rows: List[Dict[str, Any]],
    date_key: str = "date",
    value_key: str = "value",
) -> Tuple[Optional[float], Optional[str]]:
    """
    Return (latest_value, latest_date_string) for the most recent non-null row.
    If not available, returns (None, None).
    """
    s = _sorted_rows(rows, date_key=date_key, value_key=value_key, ascending=True)
    if not s:
        return None, None
    last = s[-1]
    val = _safe_float(last.get(value_key))
    d = last.get(date_key)
    return val, str(d) if d is not None else None


def calculate_cagr(
    rows: List[Dict[str, Any]],
    date_key: str = "date",
    value_key: str = "value",
) -> Optional[float]:
    """
    Compute CAGR between the earliest and latest valid points:

        CAGR = (V_end / V_start) ** (1 / years) - 1

    where years is computed from the date difference. Returns None if
    insufficient data or if values are non-positive.
    """
    s = _sorted_rows(rows, date_key=date_key, value_key=value_key, ascending=True)
    if len(s) < 2:
        return None

    v0 = _safe_float(s[0].get(value_key))
    v1 = _safe_float(s[-1].get(value_key))
    d0 = _parse_date(s[0].get(date_key))
    d1 = _parse_date(s[-1].get(date_key))

    if v0 is None or v1 is None or v0 <= 0 or v1 <= 0 or d0 is None or d1 is None:
        return None

    years = (d1 - d0).days / 365.25
    if years <= 0:
        return None

    try:
        return (v1 / v0) ** (1.0 / years) - 1.0
    except Exception:
        return None
