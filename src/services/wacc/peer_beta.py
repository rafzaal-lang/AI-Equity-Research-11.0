from __future__ import annotations
from typing import List, Dict, Optional
import os
import statistics as stats
from src.services.providers import fmp_provider as fmp

TAX_RATE_DEFAULT = float(os.getenv("TAX_RATE_DEFAULT", "0.25"))
MARKET_SYMBOL = os.getenv("MARKET_SYMBOL", "SPY")

def _num(x):
    """Convert to float safely."""
    try: 
        return float(x) if x is not None else None
    except: 
        return None

def _daily_returns(hist: List[dict]) -> List[float]:
    """Calculate daily returns from price history."""
    closes = [float(h["close"]) for h in sorted(hist, key=lambda x: x["date"]) 
              if h.get("close") is not None]
    rets = []
    for i in range(1, len(closes)):
        if closes[i-1] != 0:
            rets.append((closes[i] / closes[i-1]) - 1.0)
    return rets

def beta(symbol: str, as_of: Optional[str] = None, lookback_days: int = 252) -> Optional[float]:
    """Calculate beta using regression against market returns."""
    try:
        s_hist = fmp.historical_prices(symbol, as_of=as_of, limit=lookback_days+2)
        m_hist = fmp.historical_prices(MARKET_SYMBOL, as_of=as_of, limit=lookback_days+2)
        
        rs = _daily_returns(s_hist)
        rm = _daily_returns(m_hist)
        
        n = min(len(rs), len(rm))
        if n < 60:  # Need minimum data points
            return None
            
        rs, rm = rs[-n:], rm[-n:]
        
        # Calculate beta using covariance/variance formula
        mean_r = sum(rm) / n
        mean_s = sum(rs) / n
        
        cov = sum((rm[i] - mean_r) * (rs[i] - mean_s) for i in range(n)) / (n - 1)
        var_m = sum((rm[i] - mean_r) ** 2 for i in range(n)) / (n - 1)
        
        if var_m == 0:
            return None
            
        return cov / var_m
        
    except Exception:
        return None

def peer_beta_wacc(symbol: str, peers: List[str], rf: float = 0.045, mrp: float = 0.055, 
                   kd: float = 0.05, target_d_e: Optional[float] = None, 
                   as_of: Optional[str] = None) -> Dict[str, Any]:
    """Calculate WACC using peer beta analysis."""
    
    # Calculate betas for symbol and peers
    betas = {}
    for ticker in [symbol] + peers:
        b = beta(ticker, as_of=as_of)
        if b is not None:
            betas[ticker] = b
    
    if len(betas) < 2:  # Need at least some data
        # Fallback to industry average
        beta_est = 1.2
        beta_source = "industry_default"
    else:
        # Use peer median if available, otherwise own beta
        peer_betas = [betas[p] for p in peers if p in betas]
        if peer_betas and len(peer_betas) >= 3:
            beta_est = stats.median(peer_betas)
            beta_source = f"peer_median_{len(peer_betas)}_peers"
        elif symbol in betas:
            beta_est = betas[symbol]
            beta_source = "own_beta"
        else:
            beta_est = stats.median(list(betas.values()))
            beta_source = "available_median"
    
    # Get debt/equity ratio
    if target_d_e is not None:
        debt_ratio = target_d_e / (1 + target_d_e)
        de_source = "target"
    else:
        # Try to get from balance sheet
        try:
            bs = fmp.balance_sheet(symbol, period="annual", limit=1, as_of=as_of)
            if bs:
                total_debt = _num(bs[0].get("totalDebt"))
                total_equity = _num(bs[0].get("totalStockholdersEquity"))
                if total_debt and total_equity and total_equity > 0:
                    debt_ratio = total_debt / (total_debt + total_equity)
                    de_source = "actual_bs"
                else:
                    debt_ratio = 0.3  # Default
                    de_source = "default"
            else:
                debt_ratio = 0.3
                de_source = "default"
        except Exception:
            debt_ratio = 0.3
            de_source = "default"
    
    # Calculate WACC
    ke = rf + beta_est * mrp  # Cost of equity
    kd_after_tax = kd * (1 - TAX_RATE_DEFAULT)  # After-tax cost of debt
    wacc = debt_ratio * kd_after_tax + (1 - debt_ratio) * ke
    
    return {
        "symbol": symbol.upper(),
        "wacc": round(wacc, 4),
        "beta": round(beta_est, 3),
        "beta_source": beta_source,
        "cost_of_equity": round(ke, 4),
        "cost_of_debt_after_tax": round(kd_after_tax, 4),
        "debt_ratio": round(debt_ratio, 3),
        "debt_equity_source": de_source,
        "assumptions": {
            "risk_free_rate": rf,
            "market_risk_premium": mrp,
            "cost_of_debt": kd,
            "tax_rate": TAX_RATE_DEFAULT
        },
        "peer_betas": {k: round(v, 3) for k, v in betas.items() if k != symbol}
    }

