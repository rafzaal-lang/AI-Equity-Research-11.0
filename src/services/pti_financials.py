from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from src.services.providers import fmp_provider as fmp
from src.core.pti import (
    filter_series_asof, validate_time_series, get_period_data, 
    calculate_growth_metrics, create_pti_snapshot
)

logger = logging.getLogger(__name__)

def _num(x):
    """Convert to float safely."""
    try: 
        return float(x) if x is not None else None
    except: 
        return None

class PTIFinancialService:
    """Point-in-Time aware financial data service."""
    
    def __init__(self):
        self.cache_ttl = {
            "statements": 21600,  # 6 hours for statements
            "prices": 3600,       # 1 hour for prices
            "metrics": 7200       # 2 hours for metrics
        }
    
    def get_statements_asof(self, symbol: str, as_of: Optional[str] = None, 
                           periods: int = 8) -> Dict[str, Any]:
        """Get financial statements as of a specific date with PTI validation."""
        try:
            # Fetch raw data
            income_data = fmp.income_statement(symbol, period="annual", limit=periods)
            balance_data = fmp.balance_sheet(symbol, period="annual", limit=periods)
            cashflow_data = fmp.cash_flow(symbol, period="annual", limit=periods)
            
            # Apply PTI filtering
            if as_of:
                income_data = filter_series_asof(income_data, as_of)
                balance_data = filter_series_asof(balance_data, as_of)
                cashflow_data = filter_series_asof(cashflow_data, as_of)
            
            # Validate data quality
            validation_results = {
                "income_statement": validate_time_series(
                    income_data, value_keys=["revenue", "netIncome", "operatingIncome"]
                ),
                "balance_sheet": validate_time_series(
                    balance_data, value_keys=["totalAssets", "totalStockholdersEquity", "totalDebt"]
                ),
                "cash_flow": validate_time_series(
                    cashflow_data, value_keys=["operatingCashFlow", "freeCashFlow"]
                )
            }
            
            return {
                "symbol": symbol.upper(),
                "as_of": as_of,
                "data": {
                    "income_statement": income_data,
                    "balance_sheet": balance_data,
                    "cash_flow": cashflow_data
                },
                "validation": validation_results,
                "data_quality_score": self._calculate_quality_score(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Error fetching statements for {symbol} as of {as_of}: {e}")
            return {"error": str(e), "symbol": symbol, "as_of": as_of}
    
    def get_financial_metrics_asof(self, symbol: str, as_of: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive financial metrics as of a specific date."""
        try:
            statements = self.get_statements_asof(symbol, as_of, periods=5)
            
            if "error" in statements:
                return statements
            
            income_data = statements["data"]["income_statement"]
            balance_data = statements["data"]["balance_sheet"]
            cashflow_data = statements["data"]["cash_flow"]
            
            if not income_data or not balance_data:
                return {"error": "Insufficient data", "symbol": symbol, "as_of": as_of}
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                income_data, balance_data, cashflow_data
            )
            
            # Add growth analysis
            growth_metrics = self._calculate_growth_analysis(income_data, balance_data, cashflow_data)
            
            return {
                "symbol": symbol.upper(),
                "as_of": as_of,
                "metrics": metrics,
                "growth_analysis": growth_metrics,
                "data_quality": statements["validation"],
                "quality_score": statements["data_quality_score"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol} as of {as_of}: {e}")
            return {"error": str(e), "symbol": symbol, "as_of": as_of}
    
    def get_peer_comparison_asof(self, symbol: str, peers: List[str], 
                                as_of: Optional[str] = None) -> Dict[str, Any]:
        """Get peer comparison analysis as of a specific date."""
        try:
            all_symbols = [symbol] + peers
            peer_data = {}
            
            for ticker in all_symbols:
                metrics = self.get_financial_metrics_asof(ticker, as_of)
                if "error" not in metrics:
                    peer_data[ticker] = metrics
            
            if not peer_data:
                return {"error": "No valid peer data", "symbol": symbol, "as_of": as_of}
            
            # Calculate peer rankings and percentiles
            comparison = self._calculate_peer_rankings(symbol, peer_data)
            
            return {
                "symbol": symbol.upper(),
                "as_of": as_of,
                "peer_count": len(peers),
                "valid_peers": len(peer_data) - 1,  # Exclude the target symbol
                "comparison": comparison,
                "peer_data": peer_data
            }
            
        except Exception as e:
            logger.error(f"Error in peer comparison for {symbol} as of {as_of}: {e}")
            return {"error": str(e), "symbol": symbol, "as_of": as_of}
    
    def _calculate_quality_score(self, validation_results: Dict[str, Dict]) -> float:
        """Calculate overall data quality score (0-1)."""
        total_score = 0
        count = 0
        
        for source, validation in validation_results.items():
            if validation.get("valid", False):
                score = 1.0
                # Reduce score for warnings
                warning_count = len(validation.get("warnings", []))
                score -= (warning_count * 0.1)
                score = max(score, 0.0)
            else:
                score = 0.0
            
            total_score += score
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_comprehensive_metrics(self, income_data: List[Dict], 
                                       balance_data: List[Dict], 
                                       cashflow_data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics from statements."""
        if not income_data or not balance_data:
            return {}
        
        # Get most recent data
        latest_income = income_data[0]
        latest_balance = balance_data[0]
        latest_cashflow = cashflow_data[0] if cashflow_data else {}
        
        # Extract key values
        revenue = _num(latest_income.get("revenue"))
        net_income = _num(latest_income.get("netIncome"))
        gross_profit = _num(latest_income.get("grossProfit"))
        operating_income = _num(latest_income.get("operatingIncome"))
        
        total_assets = _num(latest_balance.get("totalAssets"))
        total_equity = _num(latest_balance.get("totalStockholdersEquity"))
        total_debt = _num(latest_balance.get("totalDebt"))
        current_assets = _num(latest_balance.get("totalCurrentAssets"))
        current_liabilities = _num(latest_balance.get("totalCurrentLiabilities"))
        
        operating_cf = _num(latest_cashflow.get("operatingCashFlow"))
        free_cf = _num(latest_cashflow.get("freeCashFlow"))
        capex = _num(latest_cashflow.get("capitalExpenditure"))
        
        # Calculate ratios
        profitability = {
            "gross_margin": (gross_profit / revenue) if revenue and gross_profit else None,
            "operating_margin": (operating_income / revenue) if revenue and operating_income else None,
            "net_margin": (net_income / revenue) if revenue and net_income else None,
            "roa": (net_income / total_assets) if net_income and total_assets else None,
            "roe": (net_income / total_equity) if net_income and total_equity else None,
        }
        
        liquidity = {
            "current_ratio": (current_assets / current_liabilities) if current_assets and current_liabilities else None,
            "working_capital": (current_assets - current_liabilities) if current_assets and current_liabilities else None,
        }
        
        leverage = {
            "debt_to_equity": (total_debt / total_equity) if total_debt and total_equity else None,
            "debt_to_assets": (total_debt / total_assets) if total_debt and total_assets else None,
            "equity_ratio": (total_equity / total_assets) if total_equity and total_assets else None,
        }
        
        cash_flow = {
            "operating_cf_margin": (operating_cf / revenue) if operating_cf and revenue else None,
            "fcf_margin": (free_cf / revenue) if free_cf and revenue else None,
            "capex_intensity": (abs(capex) / revenue) if capex and revenue else None,
        }
        
        return {
            "profitability": profitability,
            "liquidity": liquidity,
            "leverage": leverage,
            "cash_flow": cash_flow,
            "absolute_values": {
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_equity": total_equity,
                "free_cash_flow": free_cf
            }
        }
    
    def _calculate_growth_analysis(self, income_data: List[Dict], 
                                 balance_data: List[Dict], 
                                 cashflow_data: List[Dict]) -> Dict[str, Any]:
        """Calculate growth metrics across multiple periods."""
        growth_analysis = {}
        
        # Revenue growth
        if income_data:
            growth_analysis["revenue"] = calculate_growth_metrics(income_data, "revenue")
            growth_analysis["net_income"] = calculate_growth_metrics(income_data, "netIncome")
            growth_analysis["operating_income"] = calculate_growth_metrics(income_data, "operatingIncome")
        
        # Asset growth
        if balance_data:
            growth_analysis["total_assets"] = calculate_growth_metrics(balance_data, "totalAssets")
            growth_analysis["shareholders_equity"] = calculate_growth_metrics(balance_data, "totalStockholdersEquity")
        
        # Cash flow growth
        if cashflow_data:
            growth_analysis["operating_cash_flow"] = calculate_growth_metrics(cashflow_data, "operatingCashFlow")
            growth_analysis["free_cash_flow"] = calculate_growth_metrics(cashflow_data, "freeCashFlow")
        
        return growth_analysis
    
    def _calculate_peer_rankings(self, target_symbol: str, peer_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate peer rankings and percentiles."""
        if target_symbol not in peer_data:
            return {"error": "Target symbol not in peer data"}
        
        target_metrics = peer_data[target_symbol]["metrics"]
        
        # Define metrics to compare
        comparison_metrics = [
            ("profitability", "gross_margin"),
            ("profitability", "operating_margin"),
            ("profitability", "net_margin"),
            ("profitability", "roa"),
            ("profitability", "roe"),
            ("liquidity", "current_ratio"),
            ("leverage", "debt_to_equity"),
            ("cash_flow", "operating_cf_margin"),
            ("cash_flow", "fcf_margin")
        ]
        
        rankings = {}
        
        for category, metric in comparison_metrics:
            values = []
            target_value = None
            
            for symbol, data in peer_data.items():
                value = data["metrics"].get(category, {}).get(metric)
                if value is not None:
                    values.append((symbol, value))
                    if symbol == target_symbol:
                        target_value = value
            
            if target_value is not None and len(values) > 1:
                # Sort values (higher is better for most metrics, except debt ratios)
                reverse_sort = metric not in ["debt_to_equity", "debt_to_assets"]
                sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse_sort)
                
                # Find rank
                rank = next(i for i, (sym, _) in enumerate(sorted_values, 1) if sym == target_symbol)
                percentile = (len(sorted_values) - rank + 1) / len(sorted_values) * 100
                
                rankings[f"{category}_{metric}"] = {
                    "value": target_value,
                    "rank": rank,
                    "total_peers": len(sorted_values),
                    "percentile": round(percentile, 1),
                    "peer_range": {
                        "min": min(v for _, v in sorted_values),
                        "max": max(v for _, v in sorted_values),
                        "median": sorted(v for _, v in sorted_values)[len(sorted_values)//2]
                    }
                }
        
        return rankings

# Global instance
pti_financial_service = PTIFinancialService()

