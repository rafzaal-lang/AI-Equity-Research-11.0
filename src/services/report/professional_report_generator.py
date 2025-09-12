# -*- coding: utf-8 -*- 
from __future__ import annotations

import os
import math
import json
import time
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------
# Config / Env
# ---------------------------
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SECTOR_SPDRS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
SPY = "SPY"

# ---------------------------
# Robust path resolution
# ---------------------------
def debug_yoy_calculation(current_val, previous_val, metric_name):
    """Debug helper to verify YoY calculations"""
    print(f"DEBUG {metric_name}:")
    print(f"  Current: {current_val}")
    print(f"  Previous: {previous_val}")
    
    if current_val is not None and previous_val is not None and previous_val != 0:
        yoy_pct = ((current_val - previous_val) / previous_val) * 100
        print(f"  YoY: {yoy_pct:.1f}%")
        return yoy_pct
    return None

def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / "apis").exists() and (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

def _resolve_paths() -> tuple[Path, Path, Path, str]:
    module_dir = Path(__file__).resolve().parent
    root = _find_project_root(module_dir)

    tpl_candidates = [root / "reports" / "templates", root / "templates"]
    static_candidates = [root / "static", root / "reports" / "static"]

    tpl_dir = next((p for p in tpl_candidates if p.exists()), tpl_candidates[0])
    static_dir = next((p for p in static_candidates if p.exists()), static_candidates[0])

    static_url = "/static" if static_dir == (root / "static") else "/reports-static"
    return root, tpl_dir, static_dir, static_url

ROOT_DIR, TEMPLATES_DIR, STATIC_DIR, STATIC_URL_PREFIX = _resolve_paths()

# ---------------------------
# Tiny markdown -> HTML helper (graceful if not installed)
# ---------------------------
try:
    import markdown as _md
    def _md_to_html(txt: Optional[str]) -> str:
        if not txt: return ""
        return _md.markdown(txt, extensions=["extra", "sane_lists", "tables"])
except Exception:
    _md = None
    def _md_to_html(txt: Optional[str]) -> str:
        if not txt: return ""
        s = txt.replace("**", "")
        s = s.replace("\r\n", "\n").strip()
        lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
        html = []
        for ln in lines:
            if ln.startswith("- "):
                html.append(f"<div>• {ln[2:]}</div>")
            else:
                html.append(f"<p>{ln}</p>")
        return "\n".join(html)

# ---------------------------
# Very small in-memory cache for FMP GETs
# ---------------------------
class _Cache:
    def __init__(self, ttl: int = 600):
        self.ttl = ttl
        self._store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        rec = self._store.get(key)
        if not rec:
            return None
        ts, val = rec
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        self._store[key] = (time.time(), val)

_cache = _Cache(ttl=600)

def _cache_key(path: str, params: Optional[Dict[str, Any]]) -> str:
    return json.dumps({"p": path, "q": params or {}}, sort_keys=True)

# ---------------------------
# Safe numeric coercion (prevents Markup issues)
# ---------------------------
def _safe_num(x: Any) -> Optional[float]:
    try:
        # If markupsafe.Markup is present, treat it like str
        from markupsafe import Markup  # type: ignore
        if isinstance(x, Markup):
            x = str(x)
    except Exception:
        pass

    if x is None:
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("$", "").replace("%", "").replace("x", "")
        try:
            f = float(s)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None
    try:
        f = float(x)  # last resort
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

# ---------------------------
# Custom Jinja2 functions to replace regex matching
# ---------------------------
def _is_numeric(s: str) -> bool:
    """Check if string represents a number using regex"""
    if not isinstance(s, str) or not s.strip():
        return False
    pattern = r'^-?\d*\.?\d+$'
    return bool(re.match(pattern, s.strip()))

# ---------------------------
# Jinja filters that always coerce via _safe_num
# ---------------------------
def _fmt_money0(v: Any) -> str:
    n = _safe_num(v)
    return "—" if n is None else "${:,.0f}".format(n)

def _fmt_money2(v: Any) -> str:
    n = _safe_num(v)
    return "—" if n is None else "${:,.2f}".format(n)

def _fmt_mult2(v: Any) -> str:
    n = _safe_num(v)
    return "—" if n is None else "{:.2f}x".format(n)

def _fmt_pct(v: Any, places: int = 2) -> str:
    n = _safe_num(v)
    return "—" if n is None else ("{0:." + str(places) + "f}%").format(n)

def _fmt_num1(v: Any) -> str:
    n = _safe_num(v)
    return "—" if n is None else "{:.1f}".format(n)

# ---------------------------
# HTTP helpers (FMP)
# ---------------------------
def _fmp_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if not FMP_API_KEY:
        raise RuntimeError("Missing FMP_API_KEY")

    url = f"https://financialmodelingprep.com{path}"
    q = {"apikey": FMP_API_KEY}
    if params:
        q.update(params)

    key = _cache_key(path, q)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        r = requests.get(url, params=q, timeout=20)
        r.raise_for_status()
        data = r.json()
        _cache.set(key, data)
        return data
    except Exception as e:
        logger.warning("FMP GET failed %s %s -> %s", path, params, e)
        return None

def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

# ---------------------------
# Technicals
# ---------------------------
def _rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if not prices or len(prices) <= period:
        return None
    s = pd.Series(prices, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean().replace(0.0, np.nan)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.dropna()
    return float(rsi.iloc[-1]) if not rsi.empty else None

def _sma(values: List[float], window: int) -> Optional[float]:
    if not values:
        return None
    s = pd.Series(values, dtype=float)
    if len(s) < window:
        return float(s.mean())
    return float(s.rolling(window).mean().dropna().iloc[-1])

# ---------------------------
# FMP data loaders
# ---------------------------
def fetch_profile(ticker: str) -> Dict[str, Optional[str]]:
    prof = _fmp_get(f"/api/v3/profile/{ticker}") or []
    p = prof[0] if isinstance(prof, list) and prof else {}
    return {
        "companyName": p.get("companyName"),
        "description": p.get("description") or p.get("companyName"),
        "sector": p.get("sector"),
        "industry": p.get("industry"),
    }

def fetch_quote_block(ticker: str) -> Dict[str, Optional[float]]:
    ql = _fmp_get(f"/api/v3/quote/{ticker}") or []
    q = ql[0] if ql else {}
    price = _as_float(q.get("price"))
    market_cap = _as_float(q.get("marketCap"))

    bsl = _fmp_get(f"/api/v3/balance-sheet-statement/{ticker}", {"period": "quarter", "limit": 1}) or []
    bs = bsl[0] if bsl else {}
    cash = _as_float(bs.get("cashAndShortTermInvestments") or bs.get("cashAndCashEquivalents"))
    debt = _as_float(bs.get("totalDebt") or (bs.get("shortTermDebt") or 0) + (bs.get("longTermDebt") or 0))
    ev = (market_cap or 0) + (debt or 0) - (cash or 0)
    ev = ev if market_cap is not None else None

    hist = _fmp_get(f"/api/v3/historical-price-full/{ticker}", {"serietype": "line", "timeseries": 120}) or {}
    series = hist.get("historical") or []
    closes = [_as_float(x.get("close")) for x in series if _as_float(x.get("close")) is not None]
    sma52 = _sma(closes, 52)

    return {"price": price, "market_cap": market_cap, "cash": cash, "debt": debt, "ev": ev, "sma52": sma52}

def _pick_number(d: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = _as_float(d.get(k))
        if v is not None:
            return v
    return None

def fetch_quarterlies(ticker: str) -> Dict[str, Any]:
    isl = _fmp_get(f"/api/v3/income-statement/{ticker}", {"period": "quarter", "limit": 8}) or []
    cfl = _fmp_get(f"/api/v3/cash-flow-statement/{ticker}", {"period": "quarter", "limit": 8}) or []
    return {"is": isl, "cf": cfl}

def fetch_annuals(ticker: str) -> Dict[str, Any]:
    isl = _fmp_get(f"/api/v3/income-statement/{ticker}", {"period": "annual", "limit": 5}) or []
    cfl = _fmp_get(f"/api/v3/cash-flow-statement/{ticker}", {"period": "annual", "limit": 5}) or []
    return {"is": isl, "cf": cfl}

def fetch_peers(ticker: str) -> List[str]:
    peers = _fmp_get(f"/api/v4/stock_peers", {"symbol": ticker}) or []
    if isinstance(peers, list) and peers:
        lst = peers[0].get("peersList") or []
        return [p for p in lst if p and p.upper() != ticker.upper()][:6]
    return []

def fetch_estimates_optional(ticker: str) -> Dict[str, Optional[float]]:
    data = _fmp_get(f"/api/v3/analyst-estimates/{ticker}", {"limit": 8}) or []
    rev_est = None
    ebitda_est = None
    for row in data:
        rev_est = rev_est or _as_float(row.get("estimatedRevenue") or row.get("revenueEstimate"))
        ebitda_est = ebitda_est or _as_float(row.get("estimatedEbitda") or row.get("ebitdaEstimate"))
        if rev_est and ebitda_est:
            break
    return {"revenue_est": rev_est, "ebitda_est": ebitda_est}

# ---------------------------
# NEW: Earnings context loader
# ---------------------------
def fetch_earnings_context(ticker: str) -> Dict[str, str]:
    """Fetch recent earnings press releases and call transcripts for context."""
    context = {"press_release": "", "transcript": "", "summary": ""}

    try:
        # Get recent earnings surprises for context
        surprises = _fmp_get(f"/api/v3/earnings-surprises/{ticker}", {"limit": 1}) or []
        if surprises:
            latest = surprises[0]
            period = latest.get("date", "")
            context["summary"] = f"Latest earnings ({period}): "

            eps_act = latest.get("actualEarningResult")
            eps_est = latest.get("estimatedEarning")
            if eps_act is not None and eps_est is not None:
                try:
                    eps_act_f = float(eps_act)
                    eps_est_f = float(eps_est)
                    if eps_est_f != 0:
                        surprise_pct = ((eps_act_f - eps_est_f) / abs(eps_est_f)) * 100
                        context["summary"] += f"EPS ${eps_act_f:.2f} vs est ${eps_est_f:.2f} ({surprise_pct:+.1f}% surprise). "
                except Exception:
                    pass

        # Get earnings call transcript excerpt
        transcripts = _fmp_get(f"/api/v3/earning_call_transcript/{ticker}", {"limit": 1}) or []
        if transcripts and isinstance(transcripts, list):
            transcript = transcripts[0].get("content", "")
            if transcript:
                # Extract management discussion (first 2000 chars after earnings discussion)
                lower = transcript.lower()
                mgmt_start = lower.find("management discussion")
                if mgmt_start == -1:
                    mgmt_start = lower.find("prepared remarks")
                if mgmt_start != -1:
                    context["transcript"] = transcript[mgmt_start:mgmt_start + 2000]
                else:
                    context["transcript"] = transcript[:2000]

    except Exception as e:
        logger.warning(f"Failed to fetch earnings context for {ticker}: {e}")

    return context

# ---------------------------
# Derived calcs
# ---------------------------
def _sum_last_n_quarters(rows: List[dict], key_candidates: List[str], n: int = 4) -> Optional[float]:
    if not rows:
        return None
    vals = []
    for r in rows[:n]:
        v = _pick_number(r, key_candidates)
        vals.append(v if v is not None else np.nan)
    s = np.nansum(vals)
    return float(s) if not np.isnan(s) else None

def _quarter_yoy_map(rows: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
    """Get current quarter and same quarter from previous year."""
    if not rows:
        return None, None
    
    q0 = rows[0]  # Current quarter (most recent)
    date0 = q0.get("date") or q0.get("calendarYear")
    if not date0:
        return q0, None
    
    # Extract year and quarter from date
    date_str = str(date0)
    if len(date_str) >= 10:  # YYYY-MM-DD format
        y0 = int(date_str[:4])
        month = int(date_str[5:7])
        # Determine quarter based on month
        current_quarter = (month - 1) // 3 + 1
    else:
        y0 = int(date_str[:4])
        current_quarter = 1  # Default fallback
    
    # Find same quarter from previous year
    q1 = None
    target_year = y0 - 1
    
    for r in rows[1:]:
        d = r.get("date") or r.get("calendarYear")
        if not d:
            continue
        
        date_str = str(d)
        if len(date_str) >= 10:
            y = int(date_str[:4])
            month = int(date_str[5:7])
            quarter = (month - 1) // 3 + 1
            
            # Look for same quarter in previous year
            if y == target_year and quarter == current_quarter:
                q1 = r
                break
        else:
            # Fallback for year-only dates
            y = int(date_str[:4])
            if y == target_year:
                q1 = r
                break
    
    return q0, q1

def build_financial_blocks(ticker: str) -> Dict[str, Any]:
    q = fetch_quarterlies(ticker)
    a = fetch_annuals(ticker)

    q_is = q["is"]; q_cf = q["cf"]
    a_is = a["is"]; a_cf = a["cf"]

    rev_ttm = _sum_last_n_quarters(q_is, ["revenue", "totalRevenue"], 4)
    ebitda_ttm = _sum_last_n_quarters(q_is, ["ebitda", "EBITDA"], 4)
    gp_ttm = _sum_last_n_quarters(q_is, ["grossProfit"], 4)
    ni_ttm = _sum_last_n_quarters(q_is, ["netIncome"], 4)
    ocf_ttm = _sum_last_n_quarters(q_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"], 4)
    capex_ttm = _sum_last_n_quarters(q_cf, ["capitalExpenditure", "capitalExpenditures"], 4)
    fcf_ttm = (ocf_ttm + capex_ttm) if (ocf_ttm is not None and capex_ttm is not None) else None

    q0_is, q1_is = _quarter_yoy_map(q_is)
    q0_cf, q1_cf = _quarter_yoy_map(q_cf)

    def _get(row: Optional[dict], keys: List[str]) -> Optional[float]:
        if not row:
            return None
        return _pick_number(row, keys)

    gm_pct = None
    rev_q = _get(q0_is, ["revenue", "totalRevenue"])
    gp_q = _get(q0_is, ["grossProfit"])
    if gp_q is not None and rev_q:
        gm_pct = float(gp_q / rev_q * 100.0)

    q_snapshot = {
        "period": (q0_is or {}).get("date"),
        "revenue": rev_q,
        "gross_margin_pct": gm_pct,
        "ebitda": _get(q0_is, ["ebitda", "EBITDA"]),
        "net_income": _get(q0_is, ["netIncome"]),
        "ocf": _get(q0_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"]),
        "fcf": None,
        "yoy": {
            "revenue": _get(q1_is, ["revenue", "totalRevenue"]),
            "ebitda": _get(q1_is, ["ebitda", "EBITDA"]),
            "net_income": _get(q1_is, ["netIncome"]),
            "ocf": _get(q1_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"]),
            "fcf": None,
        },
    }
    ocf_q = q_snapshot["ocf"]
    capex_q = _get(q0_cf, ["capitalExpenditure", "capitalExpenditures"])
    if ocf_q is not None and capex_q is not None:
        q_snapshot["fcf"] = float(ocf_q + capex_q)
    ocf_y = q_snapshot["yoy"]["ocf"]
    capex_y = _get(q1_cf, ["capitalExpenditure", "capitalExpenditures"])
    if ocf_y is not None and capex_y is not None:
        q_snapshot["yoy"]["fcf"] = float(ocf_y + capex_y)

    debug_yoy_calculation(
        q_snapshot.get("revenue"), 
        q_snapshot["yoy"].get("revenue"), 
        "Revenue"
    )
    debug_yoy_calculation(
        q_snapshot.get("ebitda"), 
        q_snapshot["yoy"].get("ebitda"), 
        "EBITDA"
    )
    debug_yoy_calculation(
        q_snapshot.get("net_income"), 
        q_snapshot["yoy"].get("net_income"), 
        "Net Income"
    )
    debug_yoy_calculation(
        q_snapshot.get("ocf"), 
        q_snapshot["yoy"].get("ocf"), 
        "Operating Cash Flow"
    )
    debug_yoy_calculation(
        q_snapshot.get("fcf"), 
        q_snapshot["yoy"].get("fcf"), 
        "Free Cash Flow"
    )
    
    this_year = datetime.utcnow().year
    def _sum_ytd(rows: List[dict], keys: List[str], year: int) -> Optional[float]:
        vals = []
        for r in rows:
            d = r.get("date")
            if not d:
                continue
            if int(str(d)[:4]) == year:
                v = _pick_number(r, keys)
                vals.append(v if v is not None else np.nan)
        if not vals:
            return None
        s = np.nansum(vals)
        return float(s) if not np.isnan(s) else None

    ytd_snapshot = {
        "year": this_year,
        "revenue": _sum_ytd(q_is, ["revenue", "totalRevenue"], this_year),
        "ebitda": _sum_ytd(q_is, ["ebitda", "EBITDA"], this_year),
        "net_income": _sum_ytd(q_is, ["netIncome"], this_year),
        "ocf": _sum_ytd(q_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"], this_year),
        "fcf": None,
    }
    if ytd_snapshot["ocf"] is not None:
        capex_ytd = _sum_ytd(q_cf, ["capitalExpenditure", "capitalExpenditures"], this_year) or 0.0
        ytd_snapshot["fcf"] = float(ytd_snapshot["ocf"] + capex_ytd)

    # previous calendar YTD snapshot
    ytd_prev_year = this_year - 1
    ytd_prev_snapshot = {
        "year": ytd_prev_year,
        "revenue": _sum_ytd(q_is, ["revenue", "totalRevenue"], ytd_prev_year),
        "ebitda": _sum_ytd(q_is, ["ebitda", "EBITDA"], ytd_prev_year),
        "net_income": _sum_ytd(q_is, ["netIncome"], ytd_prev_year),
        "ocf": _sum_ytd(q_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"], ytd_prev_year),
        "fcf": None,
    }
    if ytd_prev_snapshot["ocf"] is not None:
        capex_ytd_prev = _sum_ytd(q_cf, ["capitalExpenditure", "capitalExpenditures"], ytd_prev_year) or 0.0
        ytd_prev_snapshot["fcf"] = float(ytd_prev_snapshot["ocf"] + capex_ytd_prev)

    def _two_year_table() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        years: List[str] = []
        for row in a_is[:2]:
            years.append(str(row.get("date") or row.get("calendarYear")))
        years = [y for y in years if y][:2]

        def pick(row: dict, keys: List[str]) -> Optional[float]:
            return _pick_number(row, keys)

        def line(metric: str, ais_keys: List[str], acf_keys: Optional[List[str]] = None, compute_fcf: bool = False):
            r = {"metric": metric}
            for i, yrow in enumerate(a_is[:2]):
                y = years[i] if i < len(years) else None
                if not y:
                    continue
                if compute_fcf:
                    ocf = pick(a_cf[i], ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"]) if i < len(a_cf) else None
                    capex = pick(a_cf[i], ["capitalExpenditure", "capitalExpenditures"]) if i < len(a_cf) else None
                    r[y] = (ocf + capex) if (ocf is not None and capex is not None) else None
                elif acf_keys:
                    r[y] = pick(a_cf[i], acf_keys) if i < len(a_cf) else None
                else:
                    r[y] = pick(yrow, ais_keys)
            rows.append(r)

        line("Revenue", ["revenue", "totalRevenue"])
        line("EBITDA", ["ebitda", "EBITDA"])
        line("Net Income", ["netIncome"])
        line("Operating Cash Flow", [], ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"])
        line("Free Cash Flow", [], compute_fcf=True)
        return rows

    return {
        "ttm": {"revenue": rev_ttm, "gross_profit": gp_ttm, "ebitda": ebitda_ttm, "net_income": ni_ttm, "ocf": ocf_ttm, "fcf": fcf_ttm},
        "quarter": q_snapshot,
        "ytd": ytd_snapshot,
        "ytd_prev": ytd_prev_snapshot,
        "annual_two_years_table": _two_year_table(),
        "raw": {"q_is": q_is, "q_cf": q_cf, "a_is": a_is, "a_cf": a_cf},
    }

def compute_multiples(ev: Optional[float], ttm: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    rev = ttm.get("revenue")
    ebitda = ttm.get("ebitda")
    return {
        "ev_s": float(ev / rev) if ev and rev else None,
        "ev_ebitda": float(ev / ebitda) if ev and ebitda and ebitda != 0 else None,
    }

def build_peers_table(subject: str, peers: List[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    rows = []
    for t in [subject] + peers:
        q = fetch_quote_block(t)
        fin = build_financial_blocks(t)
        mult = compute_multiples(q.get("ev"), fin["ttm"])

        prof_quote = _fmp_get(f"/api/v3/quote/{t}") or []
        pq = prof_quote[0] if prof_quote else {}
        dividend_yield = _as_float(pq.get("yield"))  # percent per FMP
        mcap = q.get("market_cap")
        fcf = fin["ttm"].get("fcf")
        fcf_yield = float(fcf / mcap * 100.0) if (fcf is not None and mcap) else None

        rows.append({
            "Ticker": t,
            "EV/S (TTM)": mult["ev_s"],
            "EV/EBITDA (TTM)": mult["ev_ebitda"],
            "Dividend Yield %": dividend_yield,
            "FCF Yield %": fcf_yield,
        })

    df = pd.DataFrame(rows)

    five_avg = {"EV/S 5y Avg": None, "EV/EBITDA 5y Avg": None}
    try:
        annuals = fetch_annuals(subject)
        ev_latest = fetch_quote_block(subject).get("ev")
        if ev_latest:
            rev = [_pick_number(r, ["revenue", "totalRevenue"]) for r in annuals["is"][:5]]
            ebt = [_pick_number(r, ["ebitda", "EBITDA"]) for r in annuals["is"][:5]]
            ev_s_vals = [(ev_latest / r) if r else np.nan for r in rev]
            ev_eb_vals = [(ev_latest / e) if e else np.nan for e in ebt]
            if not np.isnan(ev_s_vals).all():
                five_avg["EV/S 5y Avg"] = float(np.nanmean(ev_s_vals))
            if not np.isnan(ev_eb_vals).all():
                five_avg["EV/EBITDA 5y Avg"] = float(np.nanmean(ev_eb_vals))
    except Exception as e:
        logger.warning("5y avg multiples failed: %s", e)

    return df, five_avg

def sector_momentum(sector_guess: Optional[str]) -> Dict[str, Any]:
    tickers = list(SECTOR_SPDRS.values()) + [SPY]
    ret = {}
    for t in tickers:
        h = _fmp_get(f"/api/v3/historical-price-full/{t}", {"serietype": "line", "timeseries": 70}) or {}
        hist = h.get("historical") or []
        closes = [_as_float(x.get("close")) for x in hist if _as_float(x.get("close")) is not None]
        if len(closes) < 22:
            continue
        last = closes[-1]
        r1m = (last / closes[-22] - 1.0) * 100.0 if len(closes) >= 22 else None
        r3m = (last / closes[-66] - 1.0) * 100.0 if len(closes) >= 66 else None
        ret[t] = {"1M": float(r1m) if r1m is not None else None, "3M": float(r3m) if r3m is not None else None}
    chosen = SECTOR_SPDRS.get(sector_guess or "", None)
    return {"sector_etf": chosen, "returns": ret, "spy": ret.get(SPY, {"1M": None, "3M": None})}

def ticker_technicals(ticker: str) -> Dict[str, Optional[float]]:
    h = _fmp_get(f"/api/v3/historical-price-full/{ticker}", {"serietype": "line", "timeseries": 120}) or {}
    hist = h.get("historical") or []
    closes = [_as_float(x.get("close")) for x in hist if _as_float(x.get("close")) is not None]
    return {"rsi_14": _rsi(closes, 14) if closes else None}

# ---------------------------
# LLM commentary (enhanced)
# ---------------------------
def llm_commentary(payload: Dict[str, Any], ticker: str) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        return {"financials": "", "industry": "", "sector": ""}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        def ask(prompt: str) -> str:
            try:
                r = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a senior equity research analyst. Write tight, neutral bullets focusing on business drivers and operational context."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                return ""

        fin = payload.get("fin")
        prof = payload.get("profile") or {}
        sect = payload.get("sector")
        tech = payload.get("tech")

        # Get earnings context for enhanced analysis (always fetch fresh here)
        earnings_context = fetch_earnings_context(ticker)

        # Enhanced financial analysis with earnings context
        context_info = ""
        if earnings_context.get("summary"):
            context_info = f"Recent earnings context: {earnings_context['summary']}"
        if earnings_context.get("transcript"):
            context_info += f"Management comments: {earnings_context['transcript'][:500]}..."

        p1 = (
            f"Analyze {ticker}'s last quarter and YTD performance with YoY context, focusing on business drivers and operational factors. "
            f"Financial data: {fin}. "
            f"{context_info} "
            f"Explain WHY key metrics changed (revenue, margins, cash flow) based on business fundamentals. "
            f"Focus on: Revenue drivers, margin dynamics, cash generation quality, operational efficiency. "
            f"Return 6-8 analytical bullets starting with '- '. Avoid just stating numbers - explain the business story."
        )

        p2 = (
            f"Industry analysis and competitive positioning for {ticker}. "
            f"Sector={prof.get('sector')}, Industry={prof.get('industry')}. "
            f"Analyze competitive dynamics, market positioning, and industry-specific factors affecting performance. "
            f"Return 6-8 bullets starting with '- '. Focus on strategic context and competitive advantages/challenges."
        )

        p3 = (
            f"Sector and technical analysis for {ticker}. "
            f"Sector momentum data: {sect}. Technical indicators: {tech}. "
            f"Assess sector rotation trends, relative performance vs peers, and technical setup. "
            f"Return 4-6 bullets starting with '- '. Connect technical patterns to fundamental thesis."
        )

        fin_md = ask(p1)
        ind_md = ask(p2)
        sec_md = ask(p3)

        return {
            "financials": _md_to_html(fin_md),
            "industry": _md_to_html(ind_md),
            "sector": _md_to_html(sec_md),
        }
    except Exception as e:
        logger.warning(f"LLM commentary failed: {e}")
        return {"financials": "", "industry": "", "sector": ""}

# ---------------------------
# Rendering (fallback template uses safe filters, not .format)
# ---------------------------
_FALLBACK_CSS = """
:root{--border:#e8e8e8;--muted:#666;--bg:#fff;--fg:#111}
*{box-sizing:border-box}
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0;padding:24px;line-height:1.55}
.container{max-width:1100px;margin:0 auto}
h1{font-size:28px;margin:0 0 8px}
h2{font-size:18px;margin:0 0 8px}
h3{font-size:15px;margin:0 0 8px}
.muted{color:var(--muted);font-size:13px}
.section{margin-top:28px}
.card{border:1px solid var(--border);border-radius:12px;padding:12px;background:#fff}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.k{color:var(--muted);font-size:12px}.v{font-weight:600}
table{width:100%;border-collapse:collapse}
th,td{padding:8px;border-bottom:1px solid #eee;text-align:right}
th:first-child,td:first-child{text-align:left}
.pill{display:inline-block;padding:4px 10px;border:1px solid var(--border);border-radius:999px;font-size:12px;color:#444;background:#fafafa;margin-right:8px}
.cols{display:grid;grid-template-columns:1.2fr 1fr;gap:16px}
@media (max-width:900px){.grid{grid-template-columns:1fr}.cols{grid-template-columns:1fr}}
"""

_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ ticker }} – Pro Report</title>
{% if css_href %}<link rel="stylesheet" href="{{ css_href }}">{% else %}<style>{{ css_inline }}</style>{% endif %}
</head>
<body>
<div class="container">
<header><h1>{{ ticker }} – Professional Report</h1><div class="muted">As of {{ as_of }}</div></header>

<section class="section">
<h2>Company Overview</h2>
<div class="card">
  {% if profile.description %}<p>{{ profile.description }}</p>{% else %}<p class="muted">No description available.</p>{% endif %}
  <div class="muted" style="margin-top:6px">
    <span class="pill">Sector: {{ profile.sector or "N/A" }}</span>
    <span class="pill">Industry: {{ profile.industry or "N/A" }}</span>
  </div>
</div>
</section>

<section class="section">
<h2>Snapshot</h2>
<div class="grid">
{% for row in header_table %}
  <div class="card">
    <div class="k">{{ row.label }}</div>
    <div class="v">
      {% if row.value is not none %}
        {% if row.label in ["Market Cap","Debt","Cash","Enterprise Value (EV)"] %}
          {{ row.value | money0 }}
        {% elif row.label in ["EV/S (TTM)","EV/EBITDA (TTM)"] %}
          {{ row.value | mult2 }}
        {% elif row.label in ["Share Price","52-day Avg"] %}
          {{ row.value | money2 }}
        {% else %}
          {{ row.value | money0 }}
        {% endif %}
      {% else %}—{% endif %}
    </div>
  </div>
{% endfor %}
</div>
</section>

<section class="section">
<h2>Financials – Last 2 Fiscal Years</h2>
<div class="card">
<table>
<thead><tr><th>Metric</th>
{% set years = [] %}
{% for r in two_year %}
  {% for k,v in r.items() if k != "metric" %}
    {% if k not in years %}{% set _ = years.append(k) %}{% endif %}
  {% endfor %}
{% endfor %}
{% for y in years %}<th>{{ y }}</th>{% endfor %}
</tr></thead>
<tbody>
{% for r in two_year %}
<tr>
  <td>{{ r.metric }}</td>
  {% for y in years %}
    {% set val = r.get(y) %}
    <td>{{ val | money0 }}</td>
  {% endfor %}
</tr>
{% endfor %}
{% if estimates and (estimates.revenue_est or estimates.ebitda_est) %}
<tr><td><em>Analyst Est. Revenue</em></td><td colspan="{{ years|length }}" style="text-align:right">{{ estimates.revenue_est | money0 }}</td></tr>
<tr><td><em>Analyst Est. EBITDA</em></td><td colspan="{{ years|length }}" style="text-align:right">{{ estimates.ebitda_est | money0 }}</td></tr>
{% endif %}
</tbody>
</table>
</div>
</section>

<section class="section">
<h2>Financial Snapshot</h2>
<div class="cols">
  <div class="card">
    <h3 style="margin-top:0">Last Quarter ({{ fin_snapshot.quarter.period or "N/A" }})</h3>
    <table>
      <tr><td>Revenue</td><td>{{ fin_snapshot.quarter.revenue | money0 }}</td></tr>
      <tr><td>Gross Margin %</td><td>{{ fin_snapshot.quarter.gross_margin_pct | pct1 }}</td></tr>
      <tr><td>EBITDA</td><td>{{ fin_snapshot.quarter.ebitda | money0 }}</td></tr>
      <tr><td>Net Income</td><td>{{ fin_snapshot.quarter.net_income | money0 }}</td></tr>
      <tr><td>Operating Cash Flow</td><td>{{ fin_snapshot.quarter.ocf | money0 }}</td></tr>
      <tr><td>Free Cash Flow</td><td>{{ fin_snapshot.quarter.fcf | money0 }}</td></tr>
    </table>
  </div>
  <div class="card">
    <h3 style="margin-top:0">Year-to-Date ({{ fin_snapshot.ytd.year }})</h3>
    <table>
      <tr><td>Revenue</td><td>{{ fin_snapshot.ytd.revenue | money0 }}</td></tr>
      <tr><td>EBITDA</td><td>{{ fin_snapshot.ytd.ebitda | money0 }}</td></tr>
      <tr><td>Net Income</td><td>{{ fin_snapshot.ytd.net_income | money0 }}</td></tr>
      <tr><td>Operating Cash Flow</td><td>{{ fin_snapshot.ytd.ocf | money0 }}</td></tr>
      <tr><td>Free Cash Flow</td><td>{{ fin_snapshot.ytd.fcf | money0 }}</td></tr>
    </table>
  </div>
</div>
{% if commentary.financials %}
<div class="card" style="margin-top:12px"><h3 style="margin-top:0">LLM Commentary – Financials</h3><div>{{ commentary.financials | safe }}</div></div>
{% endif %}
</section>

<section class="section">
<h2>Industry & Macro</h2>
<div class="card"><h3 style="margin-top:0">Industry Snapshot & Competitive Positioning</h3>
{% if commentary.industry %}<div>{{ commentary.industry | safe }}</div>{% else %}<p class="muted">No LLM commentary available.</p>{% endif %}
</div>
<div class="card" style="margin-top:12px"><h3 style="margin-top:0">Sector Momentum & Technicals</h3>
<p class="muted">Sector ETF: {{ sector_inputs.chosen_sector_etf or "N/A" }} • RSI(14): {{ technical.rsi_14 | num1 if technical.rsi_14 is not none else "N/A" }}</p>
{% if commentary.sector %}<div>{{ commentary.sector | safe }}</div>{% else %}<p class="muted">No LLM commentary available.</p>{% endif %}
</div>
</section>

<section class="section">
<h2>Peer Group (≤ 6)</h2>
<div class="card">
<table>
<thead><tr><th>Ticker</th><th>EV/S (TTM)</th><th>EV/EBITDA (TTM)</th><th>Dividend Yield %</th><th>FCF Yield %</th></tr></thead>
<tbody>
{% for r in peer_rows %}
<tr>
  <td>{{ r["Ticker"] }}</td>
  <td>{{ r["EV/S (TTM)"] | mult2 }}</td>
  <td>{{ r["EV/EBITDA (TTM)"] | mult2 }}</td>
  <td>{{ r["Dividend Yield %"] | pct2 }}</td>
  <td>{{ r["FCF Yield %"] | pct2 }}</td>
</tr>
{% endfor %}
</tbody>
</table>
<div class="muted" style="margin-top:8px">
Subject 5-yr avgs: EV/S {{ peer_five_year["EV/S 5y Avg"] | mult2 }},
EV/EBITDA {{ peer_five_year["EV/EBITDA 5y Avg"] | mult2 }}.
</div>
</div>
</section>

<footer class="section muted"><div>Data source: Financial Modeling Prep (FMP).</div></footer>
</div>
</body>
</html>"""

def _load_template(env: Environment) -> Any:
    try:
        return env.get_template("pro_report.html")
    except TemplateNotFound:
        logger.warning("pro_report.html not found in %s — using built-in fallback.", TEMPLATES_DIR)
        return env.from_string(_FALLBACK_HTML)

def fetch_profile_safe(ticker: str) -> Dict[str, Optional[str]]:
    try:
        return fetch_profile(ticker)
    except Exception as e:
        logger.warning("profile fetch failed: %s", e)
        return {"companyName": None, "description": None, "sector": None, "industry": None}

# ---------------------------
# Core renderers (string + file)
# ---------------------------
def _build_render_context(t: str) -> Dict[str, Any]:
    profile = fetch_profile_safe(t)
    quote = fetch_quote_block(t)
    fundamentals = build_financial_blocks(t)
    multiples = compute_multiples(quote.get("ev"), fundamentals["ttm"])
    estimates = fetch_estimates_optional(t)
    tech = ticker_technicals(t)
    sector = sector_momentum(profile.get("sector"))

    peers = fetch_peers(t)
    if not peers:
        static = {
            "XLK": ["AAPL","MSFT","NVDA","AVGO","CRM","ADBE"],
            "XLF": ["JPM","BAC","WFC","GS","MS","C"],
            "XLY": ["AMZN","TSLA","HD","MCD","NKE","SBUX"],
            "XLP": ["PG","KO","PEP","COST","WMT","MDLZ"],
            "XLE": ["XOM","CVX","COP","EOG","SLB","PSX"],
            "XLV": ["UNH","JNJ","LLY","MRK","ABBV","TMO"],
            "XLI": ["CAT","HON","RTX","GE","BA","DE"],
            "XLC": ["META","GOOGL","NFLX","CMCSA","TMUS","DIS"],
            "XLB": ["LIN","APD","SHW","FCX","ECL","NEM"],
            "XLRE": ["PLD","AMT","EQIX","O","PSA","SPG"],
            "XLU": ["NEE","DUK","SO","D","AEP","EXC"],
        }
        etf = SECTOR_SPDRS.get(profile.get("sector") or "", "SPY")
        peers = [p for p in static.get(etf, ["AAPL","MSFT","NVDA","AMZN","META","GOOGL"]) if p.upper() != t][:6]

    peer_df, five_year = build_peers_table(t, peers)

    header_table = [
        {"label": "Share Price", "value": _safe_num(quote.get("price"))},
        {"label": "52-day Avg", "value": _safe_num(quote.get("sma52"))},
        {"label": "Market Cap", "value": _safe_num(quote.get("market_cap"))},
        {"label": "Debt", "value": _safe_num(quote.get("debt"))},
        {"label": "Cash", "value": _safe_num(quote.get("cash"))},
        {"label": "Enterprise Value (EV)", "value": _safe_num(quote.get("ev"))},
        {"label": "EV/S (TTM)", "value": _safe_num(multiples.get("ev_s"))},
        {"label": "EV/EBITDA (TTM)", "value": _safe_num(multiples.get("ev_ebitda"))},
    ]
    header_map = {r["label"]: r["value"] for r in header_table}

    # include ytd_prev in the snapshot we pass onward
    fin_snapshot = {"quarter": fundamentals["quarter"], "ytd": fundamentals["ytd"], "ytd_prev": fundamentals["ytd_prev"], "ttm": fundamentals["ttm"]}

    sector_inputs = {
        "chosen_sector_etf": sector.get("sector_etf"),
        "returns": sector.get("returns"),
        "spy": sector.get("spy"),
        "peers": peers,
        "peer_table_preview": peer_df.head(6).to_dict(orient="records"),
    }

    # NEW: add earnings context, and pass it along to the commentary payload
    earnings_context = fetch_earnings_context(t)

    commentary = llm_commentary({
        "fin": fin_snapshot,
        "profile": profile,
        "sector": sector_inputs,
        "tech": tech,
        "earnings": earnings_context,  # included for completeness (though llm_commentary fetches fresh too)
    }, t)

    return {
        "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "ticker": t,
        "profile": profile,
        "header_table": header_table,
        "header_map": header_map,  # helpful for UI template variants
        "two_year": fundamentals["annual_two_years_table"],
        "estimates": estimates,
        "fin_snapshot": fin_snapshot,
        "peer_rows": peer_df.to_dict(orient="records"),
        "peer_five_year": five_year,
        "sector_inputs": sector_inputs,
        "technical": {"rsi_14": _safe_num(tech.get("rsi_14"))},
        "commentary": commentary,
    }

def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"])
    )
    # register robust filters
    env.filters["money0"] = _fmt_money0
    env.filters["money2"] = _fmt_money2
    env.filters["mult2"]  = _fmt_mult2
    env.filters["pct1"]   = lambda v: _fmt_pct(v, 1)
    env.filters["pct2"]   = lambda v: _fmt_pct(v, 2)
    env.filters["num1"]   = _fmt_num1
    
    # Add custom functions to globals for template use
    env.globals['is_numeric'] = _is_numeric
    env.globals['safe_num'] = _safe_num
    
    return env

def generate_html_report(payload: Dict[str, Any]) -> str:
    """
    UI expects this to return an HTML string.
    Accepts payload with company.symbol / ticker / symbol.
    """
    t = (
        ((payload.get("company") or {}).get("symbol")) or
        payload.get("ticker") or
        payload.get("symbol") or
        ""
    ).upper().strip()

    if not t:
        raise ValueError("ticker/symbol is required")

    if not FMP_API_KEY:
        return "<h3>Error</h3><p>Missing FMP_API_KEY in environment. Set it in Render → Environment.</p>"

    ctx = _build_render_context(t)

    env = _make_env()
    template = _load_template(env)

    css_file = (STATIC_DIR / "pro_report.css")
    css_href = f"{STATIC_URL_PREFIX}/pro_report.css" if css_file.exists() else None
    css_inline = _FALLBACK_CSS if not css_file.exists() else ""

    html = template.render(css_href=css_href, css_inline=css_inline, **ctx)
    return html

def render_pro_report_html(ticker: str) -> str:
    """
    Keeps your original behavior (write file and return path) — useful for API-only route.
    """
    if not FMP_API_KEY:
        err_html = f"<h3>Error</h3><p>Missing FMP_API_KEY in environment. Set it in Render → Environment.</p>"
        out_dir = ROOT_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pro_report_{ticker}.html"
        out_path.write_text(err_html, encoding="utf-8")
        return str(out_path)

    html = generate_html_report({"ticker": ticker})
    out_dir = ROOT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pro_report_{ticker.upper()}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)

# ---------- Charts/EML expected by UI ----------
def _create_charts(financial_summary: Dict[str, Any]) -> Dict[str, str]:
    return {}

def package_report_as_eml(html_content: str, charts: Dict[str, str], subject: str, to_email: str, from_email: str) -> str:
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.utils import formatdate

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Date"] = formatdate(localtime=True)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("This message contains an HTML report.", "plain"))
    alt.attach(MIMEText(html_content, "html"))
    msg.attach(alt)

    for cid, path in (charts or {}).items():
        try:
            with open(path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=Path(path).name)
                msg.attach(img)
        except Exception as e:
            logger.warning("Attach image failed for %s: %s", path, e)

    out_dir = ROOT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in subject if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    out_path = out_dir / f"{safe or 'report'}.eml"
    with open(out_path, "wb") as f:
        f.write(msg.as_bytes())
    return str(out_path)

# Export namespace expected by ui_minimal.py
class _ProGenNS:
    _create_charts = staticmethod(_create_charts)
    generate_html_report = staticmethod(generate_html_report)
    package_report_as_eml = staticmethod(package_report_as_eml)

# what the UI imports
professional_report_generator = progen = _ProGenNS()


