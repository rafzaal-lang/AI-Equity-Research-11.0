# UPDATED src/services/comps/engine.py file
# This file should ONLY contain business logic, no FastAPI routes!

from __future__ import annotations
from typing import Dict, Any, List, Optional
import statistics as stats
import logging
from src.services.providers import fmp_provider as fmp

logger = logging.getLogger(__name__)

def _num(x):
    try: return float(x) if x is not None else None
    except: return None

def safe_get(data, key, default=None):
    """Safely get value from data, handling both dict and non-dict cases"""
    if isinstance(data, dict):
        return data.get(key, default)
    else:
        logger.warning(f"Expected dict but got {type(data)} when accessing key '{key}'")
        return default

def safe_extract_dict(data, data_name, symbol=""):
    """Safely extract dictionary from various data structures returned by FMP"""
    if data:
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return data[0]
            else:
                logger.warning(f"{data_name} returned non-dict in list for {symbol}: {type(data[0])}")
        elif isinstance(data, dict):
            return data
        else:
            logger.warning(f"{data_name} returned unexpected type {type(data)} for {symbol}")
    return {}

def peer_universe(symbol: str, max_peers: int = 15) -> List[str]:
    try:
        prof = fmp.profile(symbol) or {}
        sector, industry = prof.get("sector"), prof.get("industry")
        if not sector or not industry: 
            logger.warning(f"No sector/industry found for {symbol}")
            return []
        
        peers = fmp.peers_by_screener(sector, industry, limit=60)
        peers = [p for p in peers if p.upper() != symbol.upper()]
        return peers[:max_peers]
    except Exception as e:
        logger.error(f"Error getting peer universe for {symbol}: {e}")
        return []

def latest_metrics(symbol: str, as_of: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Get latest financial metrics for a symbol with robust error handling."""
    try:
        # Get data with comprehensive error handling
        inc_data = None
        km_data = None
        evs_data = None
        
        try:
            inc_data = fmp.income_statement(symbol, period="annual", limit=1)
        except Exception as e:
            logger.warning(f"Failed to get income statement for {symbol}: {e}")
        
        try:
            km_data = fmp.key_metrics_ttm(symbol)
        except Exception as e:
            logger.warning(f"Failed to get key metrics for {symbol}: {e}")
        
        try:
            evs_data = fmp.enterprise_values(symbol, period="quarter", limit=1)
        except Exception as e:
            logger.warning(f"Failed to get enterprise values for {symbol}: {e}")
        
        # Enhanced error handling for data extraction
        inc = safe_extract_dict(inc_data, "income_statement", symbol)
        km = safe_extract_dict(km_data, "key_metrics_ttm", symbol)
        evs = safe_extract_dict(evs_data, "enterprise_values", symbol)
        
        # Get price with error handling
        price = None
        try:
            price = fmp.latest_price(symbol)
        except Exception as e:
            logger.warning(f"Failed to get latest price for {symbol}: {e}")
        
        # Extract values safely with enhanced type checking
        sales = _num(safe_get(inc, "revenue"))
        ebitda = _num(safe_get(inc, "ebitda")) or _num(safe_get(inc, "operatingIncome"))
        eps = _num(safe_get(km, "netIncomePerShareTTM"))
        market_cap = _num(safe_get(evs, "marketCapitalization"))
        ev = _num(safe_get(evs, "enterpriseValue"))
        shares = _num(safe_get(inc, "weightedAverageShsOut"))
        
        # Calculate derived metrics safely
        ps_ratio = None
        if market_cap and sales and sales != 0:
            ps_ratio = market_cap / sales
        
        pe_ratio = None
        if price and eps and eps != 0:
            pe_ratio = price / eps
        
        ev_ebitda_ratio = None
        if ev and ebitda and ebitda != 0:
            ev_ebitda_ratio = ev / ebitda
        
        return {
            "price": price,
            "sales": sales,
            "ebitda": ebitda,
            "eps_ttm": eps,
            "market_cap": market_cap,
            "enterprise_value": ev,
            "shares_out": shares,
            "ps": ps_ratio,
            "pe": pe_ratio,
            "ev_ebitda": ev_ebitda_ratio,
        }
        
    except Exception as e:
        # Enhanced error logging with more details
        logger.error(f"Error in latest_metrics for {symbol}: {type(e).__name__}: {e}")
        return {
            "price": None, "sales": None, "ebitda": None, "eps_ttm": None,
            "market_cap": None, "enterprise_value": None, "shares_out": None,
            "ps": None, "pe": None, "ev_ebitda": None,
        }

def comps_table(symbol: str, max_peers: int = 15, as_of: Optional[str] = None) -> Dict[str, Any]:
    """Generate a comprehensive comparables table with enhanced error handling."""
    try:
        tickers = [symbol.upper()] + peer_universe(symbol, max_peers=max_peers)
        
        if len(tickers) <= 1:
            logger.warning(f"No peers found for {symbol}")
            return {
                "symbol": symbol.upper(), 
                "peers": [{"ticker": symbol.upper(), **latest_metrics(symbol, as_of=as_of)}]
            }
        
        rows = []
        for t in tickers:
            try:
                metrics = latest_metrics(t, as_of=as_of)
                row = {"ticker": t, **metrics}
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed to get metrics for {t}: {e}")
                # Add empty row to maintain structure
                rows.append({
                    "ticker": t,
                    "price": None, "sales": None, "ebitda": None, "eps_ttm": None,
                    "market_cap": None, "enterprise_value": None, "shares_out": None,
                    "ps": None, "pe": None, "ev_ebitda": None,
                })
        
        # Calculate percentiles for comparison metrics
        fields = ["pe", "ps", "ev_ebitda", "market_cap"]
        for field in fields:
            try:
                vals = [r[field] for r in rows if r.get(field) is not None]
                if len(vals) >= 4:
                    q = stats.quantiles(vals, n=4)
                    p25, p50, p75 = q[0], q[1], q[2]
                else:
                    # Fallback for small samples
                    sorted_vals = sorted(vals)
                    n = len(sorted_vals)
                    if n >= 3:
                        p25 = sorted_vals[n//4]
                        p50 = sorted_vals[n//2] 
                        p75 = sorted_vals[3*n//4]
                    elif n >= 1:
                        p25 = p50 = p75 = sorted_vals[0]
                    else:
                        p25 = p50 = p75 = None
                
                # Add percentile data to each row
                for r in rows:
                    r[f"{field}_p25"] = p25
                    r[f"{field}_p50"] = p50
                    r[f"{field}_p75"] = p75
                    
            except Exception as e:
                logger.warning(f"Failed to calculate percentiles for {field}: {e}")
                # Add None values if calculation fails
                for r in rows:
                    r[f"{field}_p25"] = None
                    r[f"{field}_p50"] = None
                    r[f"{field}_p75"] = None
        
        return {"symbol": symbol.upper(), "peers": rows}
        
    except Exception as e:
        logger.error(f"Error in comps_table for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "peers": [{"ticker": symbol.upper(), **latest_metrics(symbol, as_of=as_of)}],
            "error": str(e)
        }
