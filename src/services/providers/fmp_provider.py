# src/services/providers/fmp_provider.py
"""
Financial Modeling Prep (FMP) provider – thin, defensive wrappers around the FMP API.

This module centralizes all HTTP calls to FMP and normalizes the return shapes so
the rest of the app can rely on consistent structures.

Environment:
  FMP_API_KEY   – required for live calls

Notes on return shapes (to match existing callers in the repo):
  - enterprise_values() returns {"enterpriseValues": [ ... ]}  # wrapped list
  - ratios()            returns {"ratios": [ ... ]}            # wrapped list
  - key_metrics_ttm()   returns a single dict (the first/only TTM record)
  - historical_prices() returns a list of {date, close} dicts
  - latest_price()      returns a float or None
  - income_statement(), balance_sheet(), cash_flow() return lists (newest first)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/api/v3"
_API_KEY = os.getenv("FMP_API_KEY")
if not _API_KEY:
    log.warning("FMP_API_KEY is not set; live FMP calls will likely fail.")

# requests timeout can be a (connect, read) tuple
_DEFAULT_TIMEOUT = (5, 15)


class FMPError(Exception):
    pass


def _get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET helper with API key injection and safe fallbacks."""
    if params is None:
        params = {}
    if _API_KEY:
        # FMP expects 'apikey' param
        params.setdefault("apikey", _API_KEY)

    url = f"{_BASE.rstrip('/')}/{path.strip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Add response text if present (best-effort)
        body = ""
        try:
            body = f" body={resp.text[:300]!r}" if "resp" in locals() and hasattr(resp, "text") else ""
        except Exception:
            pass
        log.warning("FMP GET failed: url=%s params=%s err=%s%s", url, params, e, body)
        return None


def _first(lst: Any) -> Dict[str, Any]:
    return lst[0] if isinstance(lst, list) and lst else {}


# -----------------------------------------------------------------------------
# Basic company info
# -----------------------------------------------------------------------------
def profile(symbol: str) -> Dict[str, Any]:
    data = _get_json(f"profile/{symbol}") or []
    return _first(data)


def quote(symbol: str) -> Dict[str, Any]:
    data = _get_json(f"quote/{symbol}") or []
    return _first(data)


def latest_price(symbol: str) -> Optional[float]:
    # Try the short quote first (fast & small)
    data = _get_json(f"quote-short/{symbol}") or []
    price = _first(data).get("price")
    if price is not None:
        try:
            return float(price)
        except Exception:
            pass
    # Fallback to full quote
    q = quote(symbol)
    if q and q.get("price") is not None:
        try:
            return float(q["price"])
        except Exception:
            pass
    return None


def market_cap(symbol: str) -> Optional[float]:
    q = quote(symbol)
    mc = q.get("marketCap")
    if mc is not None:
        try:
            return float(mc)
        except Exception:
            pass
    p = profile(symbol)
    mc = p.get("mktCap")
    try:
        return float(mc) if mc is not None else None
    except Exception:
        return None


def shares_outstanding(symbol: str) -> Optional[float]:
    # Prefer profile field; naming varies in FMP payloads.
    p = profile(symbol)
    for key in ("sharesOutstanding", "shareOutstanding", "outstandingShares"):
        if p.get(key) is not None:
            try:
                return float(p[key])
            except Exception:
                pass
    # Fallback: shares float endpoint (if enabled on your plan)
    data = _get_json(f"shares_float/{symbol}") or {}
    # Some accounts get a list back
    if isinstance(data, list):
        data = _first(data)
    for key in ("outstandingShares", "outstanding"):
        if data.get(key) is not None:
            try:
                return float(data[key])
            except Exception:
                pass
    return None


# -----------------------------------------------------------------------------
# Historical prices (for quant signals)
# -----------------------------------------------------------------------------
def historical_prices(symbol: str, limit: int = 300) -> List[Dict[str, Any]]:
    """
    Returns list of {date, close}. Tries 'historical-price-full' first, then chart.
    """
    # Preferred – compact line series
    data = _get_json(
        f"historical-price-full/{symbol}",
        {"serietype": "line", "timeseries": int(limit)},
    ) or {}
    hist = data.get("historical")
    if isinstance(hist, list) and hist:
        # normalize keys -> {date, close}
        out: List[Dict[str, Any]] = []
        for r in hist:
            if not isinstance(r, dict):
                continue
            out.append(
                {
                    "date": r.get("date") or r.get("label") or r.get("timestamp"),
                    "close": r.get("close") or r.get("adjClose") or r.get("price"),
                }
            )
        return out

    # Fallback – chart endpoint (1day)
    data2 = _get_json(f"historical-chart/1day/{symbol}", {"limit": int(limit)}) or []
    out2: List[Dict[str, Any]] = []
    for r in data2 if isinstance(data2, list) else []:
        if not isinstance(r, dict):
            continue
        out2.append(
            {
                "date": r.get("date") or r.get("label") or r.get("timestamp"),
                "close": r.get("close") or r.get("adjClose") or r.get("price"),
            }
        )
    return out2


# -----------------------------------------------------------------------------
# Financial statements
# -----------------------------------------------------------------------------
def income_statement(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    """
    Returns a list (newest first) of income statement dicts.
    period: 'annual' or 'quarter'
    """
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"income-statement/{symbol}", {"period": per, "limit": int(limit)}) or []
    return data if isinstance(data, list) else []


def balance_sheet(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    """
    Returns a list (newest first) of balance sheet dicts.
    period: 'annual' or 'quarter'
    """
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"balance-sheet-statement/{symbol}", {"period": per, "limit": int(limit)}) or []
    return data if isinstance(data, list) else []


def cash_flow(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    """
    Returns a list (newest first) of cash flow statement dicts.
    period: 'annual' or 'quarter'
    """
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"cash-flow-statement/{symbol}", {"period": per, "limit": int(limit)}) or []
    return data if isinstance(data, list) else []


# -----------------------------------------------------------------------------
# Metrics & ratios
# -----------------------------------------------------------------------------
def key_metrics(symbol: str, period: str = "annual", limit: int = 4) -> List[Dict[str, Any]]:
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"key-metrics/{symbol}", {"period": per, "limit": int(limit)}) or []
    return data if isinstance(data, list) else []


def key_metrics_ttm(symbol: str) -> Dict[str, Any]:
    data = _get_json(f"key-metrics-ttm/{symbol}") or []
    return _first(data)


def ratios(symbol: str, period: str = "ttm", limit: int = 4) -> Dict[str, Any]:
    """
    Returns {"ratios": [...]} to match existing callers that do .get("ratios").
    - period='ttm' uses /ratios-ttm
    - otherwise uses /ratios?period={annual|quarter}
    """
    if period.lower() == "ttm":
        data = _get_json(f"ratios-ttm/{symbol}") or []
        return {"ratios": data if isinstance(data, list) else []}
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"ratios/{symbol}", {"period": per, "limit": int(limit)}) or []
    return {"ratios": data if isinstance(data, list) else []}


def enterprise_values(symbol: str, period: str = "annual", limit: int = 10) -> Dict[str, Any]:
    """
    Returns {"enterpriseValues": [...]} so callers can do .get("enterpriseValues").
    """
    per = "quarter" if period.lower().startswith("q") else "annual"
    data = _get_json(f"enterprise-values/{symbol}", {"period": per, "limit": int(limit)}) or []
    return {"enterpriseValues": data if isinstance(data, list) else []}


# -----------------------------------------------------------------------------
# SEC & outlook
# -----------------------------------------------------------------------------
def sec_filings(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    data = _get_json(f"sec-filings/{symbol}", {"limit": int(limit)}) or []
    return data if isinstance(data, list) else []


def company_outlook(symbol: str) -> Dict[str, Any]:
    data = _get_json("company-outlook", {"symbol": symbol}) or {}
    # some accounts get {"profile": {...}, "ratios": {...}, ...}
    return data if isinstance(data, dict) else {}


# -----------------------------------------------------------------------------
# Convenience helpers used in comps/valuation
# -----------------------------------------------------------------------------
def eps_ttm(symbol: str) -> Optional[float]:
    km = key_metrics_ttm(symbol)
    # FMP names can vary (epsTTM or trailingEps)
    for k in ("epsTTM", "trailingEps", "eps"):
        if km.get(k) is not None:
            try:
                return float(km[k])
            except Exception:
                pass
    # fallback from ratios-ttm if present
    r = ratios(symbol, period="ttm").get("ratios", [])
    if r:
        for k in ("priceEarningsRatioTTM", "peRatioTTM", "priceEarningsRatio"):
            val = _first(r).get(k)
            if val:
                p = latest_price(symbol)
                try:
                    return float(p) / float(val) if p and val else None
                except Exception:
                    return None
    return None


def ebitda_ttm(symbol: str) -> Optional[float]:
    # Try enterprise-values stream which sometimes includes EBITDA
    ev = enterprise_values(symbol, period="annual", limit=1).get("enterpriseValues", [])
    if ev:
        for k in ("ebitda", "EBITDA"):
            val = _first(ev).get(k)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
    # Fallback via cash flow + income statement (rough)
    inc = income_statement(symbol, period="annual", limit=1)
    cf = cash_flow(symbol, period="annual", limit=1)
    try:
        i = _first(inc)
        d_and_a = _first(cf).get("depreciationAndAmortization")
        if i and i.get("operatingIncome") is not None and d_and_a is not None:
            return float(i["operatingIncome"]) + float(d_and_a)
    except Exception:
        pass
    return None
