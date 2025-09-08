from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from src.services.providers import fmp_provider as fmp
from src.services.wacc.peer_beta import peer_beta_wacc
from src.core.pti import calculate_cagr, get_latest_value

logger = logging.getLogger(__name__)

def _num(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _safe_div(n, d):
    """Safe division: returns None on bad/zero denominators."""
    try:
        n = float(n)
        d = float(d)
        if d == 0.0:
            return None
        return n / d
    except Exception:
        return None

def _to_df(rows: List[dict]) -> pd.DataFrame:
    """Convert list of dicts to DataFrame."""
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# -----------------------------------------------------------------------------
# TTM SNAPSHOT
# -----------------------------------------------------------------------------
def ttm_snapshot(symbol: str) -> Dict[str, Any]:
    inc = (fmp.income_statement(symbol, period="annual", limit=1) or [{}])[0]
    bs  = (fmp.balance_sheet(symbol,  period="annual", limit=1) or [{}])[0]
    cf  = (fmp.cash_flow(symbol,      period="annual", limit=1) or [{}])[0]

    # key_metrics_ttm returns a dict (first/only TTM row)
    km_raw = fmp.key_metrics_ttm(symbol)
    km = km_raw[0] if isinstance(km_raw, list) and km_raw else (km_raw or {})

    revenue           = _num(inc.get("revenue"))
    gross_profit      = _num(inc.get("grossProfit"))
    operating_income  = _num(inc.get("operatingIncome"))
    net_income        = _num(inc.get("netIncome"))
    total_assets      = _num(bs.get("totalAssets"))
    total_debt        = _num(bs.get("totalDebt"))
    shareholders_eq   = _num(bs.get("totalStockholdersEquity"))
    operating_cf      = _num(cf.get("operatingCashFlow"))
    capex             = _num(cf.get("capitalExpenditure"))

    fcf = None
    if operating_cf is not None and capex is not None:
        # capex from statements is often negative; addition preserves sign
        fcf = operating_cf + capex

    return {
        "symbol": symbol.upper(),
        "reported": {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_debt": total_debt,
            "shareholders_equity": shareholders_eq,
            "operating_cash_flow": operating_cf,
            "capex": capex,
            "free_cash_flow": fcf,
        },
        "margins": {
            "gross_margin":     _safe_div(gross_profit, revenue)        if revenue is not None and gross_profit is not None else None,
            "operating_margin": _safe_div(operating_income, revenue)    if revenue is not None and operating_income is not None else None,
            "net_margin":       _safe_div(net_income, revenue)          if revenue is not None and net_income is not None else None,
        },
        "ratios": {
            "debt_to_equity": _safe_div(total_debt, shareholders_eq)    if shareholders_eq is not None and total_debt is not None else None,
            "roa":            _safe_div(net_income, total_assets)       if total_assets is not None and net_income is not None else None,
            "roe":            _safe_div(net_income, shareholders_eq)    if shareholders_eq is not None and net_income is not None else None,
        }
    }

# -----------------------------------------------------------------------------
# SIMPLE DCF
# -----------------------------------------------------------------------------
def simple_dcf(symbol: str, years: int = 5) -> Dict[str, Any]:
    """Simple DCF model with equity value calculation."""
    inc = fmp.income_statement(symbol, period="annual", limit=3)
    if not inc:
        return {"error": "No income statements"}

    # Revenue growth from last 2 years if available
    if len(inc) >= 2 and inc[0].get("revenue") and inc[1].get("revenue"):
        try:
            rev_growth = float(inc[0]["revenue"]) / float(inc[1]["revenue"]) - 1.0
        except Exception:
            rev_growth = 0.05
    else:
        rev_growth = 0.05

    # Cap growth
    rev_growth = max(min(rev_growth, 0.25), -0.10)

    # Base FCF
    base_fcf = _num(inc[0].get("freeCashFlow")) or 0.0
    if base_fcf <= 0.0:
        base_fcf = (_num(inc[0].get("operatingIncome")) or 0.0) * 0.7  # rough tax adj

    # Discounting assumptions + terminal guard
    discount_rate   = max(0.01, 0.10)
    terminal_growth = min(0.025, discount_rate - 0.005)  # ensure r > g

    # Projections
    projections: List[Dict[str, float]] = []
    for year in range(1, years + 1):
        fcf = base_fcf * ((1.0 + rev_growth) ** year)
        pv  = fcf / ((1.0 + discount_rate) ** year)
        projections.append({"year": year, "fcf": fcf, "pv": pv})

    pv_sum        = sum(p["pv"] for p in projections)
    terminal_fcf  = projections[-1]["fcf"] * (1.0 + terminal_growth)
    terminal_val  = terminal_fcf / (discount_rate - terminal_growth)
    terminal_pv   = terminal_val / ((1.0 + discount_rate) ** years)
    enterprise_value = pv_sum + terminal_pv

    # Equity bridge and per-share
    equity_value = enterprise_value
    fair_value_per_share = None
    try:
        # Net debt
        bs = fmp.balance_sheet(symbol, period="annual", limit=1)
        if bs:
            total_debt = _num(bs[0].get("totalDebt")) or 0.0
            cash       = _num(bs[0].get("cashAndCashEquivalents")) or 0.0
            net_debt   = total_debt - cash
            equity_value = enterprise_value - net_debt

        # Shares outstanding: prefer provider helper, then TTM fields as fallback
        shares_out = fmp.shares_outstanding(symbol)
        if shares_out is None:
            km = fmp.key_metrics_ttm(symbol) or {}
            if isinstance(km, list):
                km = km[0] if km else {}
            shares_out = (
                _num(km.get("sharesOutstanding"))
                or _num(km.get("weightedAverageShsOutDil"))
                or _num(km.get("weightedAverageShsOut"))
            )

        if shares_out and shares_out > 0:
            fair_value_per_share = equity_value / shares_out
    except Exception:
        # keep equity_value; just leave per-share None if anything fails
        pass

    return {
        "symbol": symbol.upper(),
        "assumptions": {
            "revenue_growth": rev_growth,
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "projection_years": years,
        },
        "projections": projections,
        "terminal_value": terminal_val,
        "terminal_value_pv": terminal_pv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "fair_value_per_share": fair_value_per_share,
        "terminal_value_pct": (terminal_pv / enterprise_value) if enterprise_value else None,
    }


# -----------------------------------------------------------------------------
# BUILD MODEL (basic)
# -----------------------------------------------------------------------------
def build_model(symbol: str, period: str = "annual", force_refresh: bool = False) -> Dict[str, Any]:
    try:
        snapshot = ttm_snapshot(symbol)
        dcf      = simple_dcf(symbol)
        return {
            "symbol": symbol.upper(),
            "model_type": "basic_dcf",
            "core_financials": snapshot,
            "ttm_snapshot": snapshot["reported"],
            "dcf_valuation": dcf,
        }
    except Exception as e:
        logging.getLogger(__name__).exception("build_model failed for %s", symbol)
        msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return {"error": msg}

# -----------------------------------------------------------------------------
# ENHANCED SNAPSHOT / CORE FINANCIALS
# -----------------------------------------------------------------------------
def enhanced_ttm_snapshot(symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Enhanced TTM snapshot with more comprehensive metrics."""
    km = fmp.key_metrics_ttm(symbol) or {}
    if isinstance(km, list):
        km = km[0] if km else {}

    evs_list = fmp.enterprise_values(symbol, period="quarter", limit=8).get("enterpriseValues") or []
    ev0 = evs_list[0] if evs_list else {}

    return {
        "symbol": symbol.upper(),
        "ttm": {
            "revenue_per_share_ttm":     _num(km.get("revenuePerShareTTM")),
            "net_income_per_share_ttm":  _num(km.get("netIncomePerShareTTM")),
            "free_cash_flow_ttm":        _num(km.get("freeCashFlowTTM")),
            "operating_cash_flow_ttm":   _num(km.get("operatingCashFlowTTM")),
            "roe_ttm":                   _num(km.get("roeTTM")),
            "roa_ttm":                   _num(km.get("roaTTM")),
            "gross_margin_ttm":          _num(km.get("grossProfitMarginTTM")),
            "op_margin_ttm":             _num(km.get("operatingProfitMarginTTM")),
            "net_margin_ttm":            _num(km.get("netProfitMarginTTM")),
            "market_cap_latest":         _num(ev0.get("marketCapitalization")),
            "enterprise_value_latest":   _num(ev0.get("enterpriseValue")),
        },
    }

def core_financials(symbol: str, period: str = "annual", limit: int = 8) -> Dict[str, Any]:
    """Get core financial statements and calculate key metrics with historical trends."""
    inc = _to_df(fmp.income_statement(symbol, period=period, limit=limit))
    bs  = _to_df(fmp.balance_sheet(symbol,  period=period, limit=limit))
    cf  = _to_df(fmp.cash_flow(symbol,      period=period, limit=limit))

    if inc.empty or bs.empty:
        return {"error": "no data"}

    i0 = inc.iloc[0]
    b0 = bs.iloc[0]
    c0 = cf.iloc[0] if not cf.empty else {}

    revenue     = _num(i0.get("revenue"))
    net_income  = _num(i0.get("netIncome"))
    gross       = _num(i0.get("grossProfit"))
    op_income   = _num(i0.get("operatingIncome"))
    fcf         = _num((c0 or {}).get("freeCashFlow"))
    op_cf       = _num((c0 or {}).get("operatingCashFlow"))
    total_eq    = _num(b0.get("totalStockholdersEquity"))
    total_assets= _num(b0.get("totalAssets"))
    total_debt  = _num(b0.get("totalDebt"))
    shares      = _num(i0.get("weightedAverageShsOut"))

    margins = {
        "gross_margin":     _safe_div(gross, revenue)       if revenue is not None and gross is not None else None,
        "operating_margin": _safe_div(op_income, revenue)   if revenue is not None and op_income is not None else None,
        "net_margin":       _safe_div(net_income, revenue)  if revenue is not None and net_income is not None else None,
    }

    ratios = {
        "roe":            _safe_div(net_income, total_eq)     if total_eq is not None and net_income is not None else None,
        "roa":            _safe_div(net_income, total_assets) if total_assets is not None and net_income is not None else None,
        "debt_to_equity": _safe_div(total_debt, total_eq)     if total_eq is not None and total_debt is not None else None,
    }

    rev_data = [{"date": row.get("date"), "value": _num(row.get("revenue"))} for _, row in inc.iterrows()]
    ni_data  = [{"date": row.get("date"), "value": _num(row.get("netIncome"))} for _, row in inc.iterrows()]

    growth = {
        "revenue_cagr_3y":     calculate_cagr(rev_data[:3], "value"),
        "net_income_cagr_3y":  calculate_cagr(ni_data[:3], "value"),
    }

    return {
        "reported": {
            "revenue": revenue,
            "net_income": net_income,
            "free_cash_flow": fcf,
            "operating_cash_flow": op_cf,
            "shares_out": shares,
        },
        "margins": margins,
        "ratios": ratios,
        "growth_rates": growth,
        "historical_data": {
            "revenue_series":    [_num(row.get("revenue"))    for _, row in inc.iterrows()],
            "net_income_series": [_num(row.get("netIncome"))  for _, row in inc.iterrows()],
            "fcf_series":        [_num(row.get("freeCashFlow")) for _, row in cf.iterrows()] if not cf.empty else [],
        },
    }

# -----------------------------------------------------------------------------
# MULTI-STAGE DCF & ADVANCED MODELS
# -----------------------------------------------------------------------------
def multi_stage_dcf(base_fcf: float, years_stage1: int = 5, g1: float = 0.06,
                    years_stage2: int = 5, g2: float = 0.04, g_terminal: float = 0.025,
                    wacc: float = 0.09) -> Dict[str, Any]:
    """Enhanced multi-stage DCF with sensitivity analysis."""
    # Guard terminal math
    wacc = max(0.01, float(wacc))
    g_terminal = min(float(g_terminal), wacc - 0.005)

    # Stage 1
    stage1_flows = []
    fcf = float(base_fcf or 0.0)
    for t in range(1, years_stage1 + 1):
        fcf *= (1 + g1)
        pv = fcf / ((1 + wacc) ** t)
        stage1_flows.append({"year": t, "fcf": fcf, "pv": pv})

    # Stage 2
    stage2_flows = []
    for t in range(years_stage1 + 1, years_stage1 + years_stage2 + 1):
        fcf *= (1 + g2)
        pv = fcf / ((1 + wacc) ** t)
        stage2_flows.append({"year": t, "fcf": fcf, "pv": pv})

    terminal_cf = fcf * (1 + g_terminal)
    terminal_val = terminal_cf / (wacc - g_terminal)
    terminal_pv = terminal_val / ((1 + wacc) ** (years_stage1 + years_stage2))

    stage1_pv = sum(f["pv"] for f in stage1_flows)
    stage2_pv = sum(f["pv"] for f in stage2_flows)
    ev = stage1_pv + stage2_pv + terminal_pv

    # Sensitivity
    wacc_range = [wacc - 0.01, wacc, wacc + 0.01]
    tg_range   = [g_terminal - 0.005, g_terminal, g_terminal + 0.005]

    sensitivity: Dict[str, float] = {}
    for w in wacc_range:
        for tg in tg_range:
            if w <= tg:
                continue
            tv_cf = fcf * (1 + tg)  # last-stage fcf roll-forward
            s1_pv = sum(f["fcf"] / ((1 + w) ** f["year"]) for f in stage1_flows)
            s2_pv = sum(f["fcf"] / ((1 + w) ** f["year"]) for f in stage2_flows)
            tv    = tv_cf / (w - tg)
            tv_pv = tv / ((1 + w) ** (years_stage1 + years_stage2))
            key = f"wacc_{round(w*100,1)}%_tg_{round(tg*100,1)}%"
            sensitivity[key] = float(s1_pv + s2_pv + tv_pv)

    return {
        "assumptions": {
            "wacc": wacc, "g1": g1, "g2": g2, "g_terminal": g_terminal,
            "years_stage1": years_stage1, "years_stage2": years_stage2
        },
        "stage1_pv": float(stage1_pv),
        "stage2_pv": float(stage2_pv),
        "terminal_pv": float(terminal_pv),
        "enterprise_value": float(ev),
        "sensitivity_analysis": sensitivity,
        "terminal_value_pct": float(terminal_pv / ev) if ev else None,
        "cash_flows": {
            "stage1": stage1_flows,
            "stage2": stage2_flows,
            "terminal": {"fcf": terminal_cf, "value": terminal_val, "pv": terminal_pv}
        }
    }

def enhanced_build_model(symbol: str, period: str = "annual", force_refresh: bool = False,
                         peers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build comprehensive financial model with peer-beta WACC."""
    core = core_financials(symbol, period=period, limit=8)
    if "error" in core:
        return {"symbol": symbol.upper(), "error": "no data"}

    snap = enhanced_ttm_snapshot(symbol, force_refresh=force_refresh)

    base_fcf = core["reported"].get("free_cash_flow") or snap["ttm"].get("free_cash_flow_ttm")
    if base_fcf is None:
        return {"symbol": symbol.upper(), "error": "no base FCF"}

    if peers is None:
        try:
            prof = fmp.profile(symbol) or {}
            sector, industry = prof.get("sector"), prof.get("industry")
            if sector and industry and hasattr(fmp, "peers_by_screener"):
                peers = fmp.peers_by_screener(sector, industry, limit=20)[:10]
            else:
                peers = []
        except Exception:
            peers = []

    wacc_analysis = peer_beta_wacc(symbol, peers or [])
    wacc = wacc_analysis.get("wacc", 0.09)

    growth_rate = core["growth_rates"].get("revenue_cagr_3y") or 0.06
    growth_rate = max(min(growth_rate, 0.15), -0.05)

    dcf = multi_stage_dcf(
        base_fcf=base_fcf,
        years_stage1=5,
        g1=growth_rate,
        years_stage2=5,
        g2=0.04,
        g_terminal=0.025,
        wacc=wacc
    )

    return {
        "symbol": symbol.upper(),
        "model_type": "comprehensive_enhanced",
        "core_financials": core,
        "ttm_snapshot": snap,
        "wacc_analysis": wacc_analysis,
        "dcf_valuation": dcf,
        "peer_list": peers or []
    }

# -----------------------------------------------------------------------------
# COMPREHENSIVE RATIOS / SCENARIOS / MONTE CARLO
# -----------------------------------------------------------------------------
def calculate_comprehensive_ratios(income_data: List[Dict], balance_data: List[Dict],
                                   cashflow_data: List[Dict], market_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Calculate comprehensive financial ratios across multiple categories."""
    if not income_data or not balance_data:
        return {"error": "Insufficient data for ratio calculation"}

    latest_income    = income_data[0]
    latest_balance   = balance_data[0]
    latest_cashflow  = cashflow_data[0] if cashflow_data else {}

    revenue          = _num(latest_income.get("revenue"))
    net_income       = _num(latest_income.get("netIncome"))
    gross_profit     = _num(latest_income.get("grossProfit"))
    operating_income = _num(latest_income.get("operatingIncome"))
    ebit             = _num(latest_income.get("ebitda")) or operating_income
    interest_expense = _num(latest_income.get("interestExpense"))

    total_assets       = _num(latest_balance.get("totalAssets"))
    current_assets     = _num(latest_balance.get("totalCurrentAssets"))
    cash               = _num(latest_balance.get("cashAndCashEquivalents"))
    inventory          = _num(latest_balance.get("inventory"))
    receivables        = _num(latest_balance.get("netReceivables"))
    total_liabilities  = _num(latest_balance.get("totalLiabilities"))
    current_liabilities= _num(latest_balance.get("totalCurrentLiabilities"))
    total_debt         = _num(latest_balance.get("totalDebt"))
    long_term_debt     = _num(latest_balance.get("longTermDebt"))
    shareholders_eq    = _num(latest_balance.get("totalStockholdersEquity"))

    operating_cf       = _num(latest_cashflow.get("operatingCashFlow"))
    free_cf            = _num(latest_cashflow.get("freeCashFlow"))
    capex              = _num(latest_cashflow.get("capitalExpenditure"))

    market_cap = _num(market_data.get("market_cap")) if market_data else None
    share_price = _num(market_data.get("share_price")) if market_data else None
    shares_outstanding = _num(latest_income.get("weightedAverageShsOut"))

    profitability = {
        "gross_margin":     _safe_div(gross_profit, revenue)       if revenue and gross_profit else None,
        "operating_margin": _safe_div(operating_income, revenue)   if revenue and operating_income else None,
        "net_margin":       _safe_div(net_income, revenue)         if revenue and net_income else None,
        "ebit_margin":      _safe_div(ebit, revenue)               if revenue and ebit else None,
        "roa":              _safe_div(net_income, total_assets)    if net_income and total_assets else None,
        "roe":              _safe_div(net_income, shareholders_eq) if net_income and shareholders_eq else None,
        "roic": None,
        "asset_turnover":   _safe_div(revenue, total_assets)       if revenue and total_assets else None,
        "equity_multiplier":_safe_div(total_assets, shareholders_eq) if total_assets and shareholders_eq else None,
    }
    if net_income and total_assets and current_liabilities:
        invested_capital = total_assets - current_liabilities
        if invested_capital and invested_capital > 0:
            profitability["roic"] = net_income / invested_capital

    liquidity = {
        "current_ratio":         _safe_div(current_assets, current_liabilities) if current_assets and current_liabilities else None,
        "quick_ratio":           _safe_div((current_assets - inventory), current_liabilities) if current_assets and inventory is not None and current_liabilities else None,
        "cash_ratio":            _safe_div(cash, current_liabilities) if cash and current_liabilities else None,
        "working_capital":       (current_assets - current_liabilities) if current_assets and current_liabilities else None,
        "working_capital_ratio": _safe_div((current_assets - current_liabilities), revenue) if current_assets and current_liabilities and revenue else None,
    }

    leverage = {
        "debt_to_equity":       _safe_div(total_debt, shareholders_eq) if total_debt and shareholders_eq else None,
        "debt_to_assets":       _safe_div(total_debt, total_assets)    if total_debt and total_assets else None,
        "equity_ratio":         _safe_div(shareholders_eq, total_assets) if shareholders_eq and total_assets else None,
        "debt_service_coverage":_safe_div(operating_cf, total_debt)    if operating_cf and total_debt else None,
        "interest_coverage":    _safe_div(ebit, interest_expense)      if ebit and interest_expense else None,
        "long_term_debt_ratio": _safe_div(long_term_debt, (long_term_debt + (shareholders_eq or 0))) if long_term_debt and shareholders_eq else None,
    }

    efficiency = {
        "inventory_turnover":      _safe_div(revenue, inventory)     if revenue and inventory else None,
        "receivables_turnover":    _safe_div(revenue, receivables)   if revenue and receivables else None,
        "days_sales_outstanding":  _safe_div((receivables * 365.0), revenue) if receivables and revenue else None,
        "days_inventory_outstanding": _safe_div((inventory * 365.0), revenue) if inventory and revenue else None,
        "asset_turnover":          _safe_div(revenue, total_assets)  if revenue and total_assets else None,
        "fixed_asset_turnover":    None,
    }

    valuation = {}
    if market_cap and shares_outstanding:
        eps = _safe_div(net_income, shares_outstanding) if net_income and shares_outstanding else None
        bvps = _safe_div(shareholders_eq, shares_outstanding) if shareholders_eq and shares_outstanding else None
        valuation = {
            "pe_ratio": (share_price / eps) if share_price and eps and eps != 0 else None,
            "pb_ratio": (share_price / bvps) if share_price and bvps and bvps != 0 else None,
            "ps_ratio": _safe_div(market_cap, revenue) if market_cap and revenue else None,
            "pcf_ratio": _safe_div(market_cap, operating_cf) if market_cap and operating_cf else None,
            "ev_revenue": None,
            "ev_ebitda": None,
            "dividend_yield": None,
        }

    return {
        "profitability": profitability,
        "liquidity": liquidity,
        "leverage": leverage,
        "efficiency": efficiency,
        "cash_flow": {
            "operating_cf_ratio": _safe_div(operating_cf, current_liabilities) if operating_cf and current_liabilities else None,
            "free_cf_yield": _safe_div(free_cf, market_cap) if free_cf and market_cap else None,
            "capex_intensity": _safe_div(abs(capex) if capex is not None else None, revenue) if revenue else None,
            "fcf_conversion": _safe_div(free_cf, net_income) if free_cf and net_income else None,
            "operating_cf_margin": _safe_div(operating_cf, revenue) if operating_cf and revenue else None,
        },
        "valuation": valuation,
        "calculation_date": latest_income.get("date"),
        "data_quality": {
            "has_income": bool(income_data),
            "has_balance": bool(balance_data),
            "has_cashflow": bool(cashflow_data),
            "has_market_data": bool(market_data),
        },
    }

# -----------------------------------------------------------------------------
# SCENARIOS & MONTE CARLO
# -----------------------------------------------------------------------------
def scenario_analysis(base_model: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform scenario analysis on DCF model with different assumptions."""
    if "dcf_valuation" not in base_model:
        return {"error": "Base model must contain DCF valuation"}

    base_dcf = base_model["dcf_valuation"]
    base_assumptions = base_dcf["assumptions"]

    scenario_results: Dict[str, Any] = {}
    for i, scenario in enumerate(scenarios):
        scenario_name = scenario.get("name", f"Scenario_{i+1}")
        scenario_assumptions = {**base_assumptions, **scenario.get("assumptions", {})}
        try:
            base_fcf = None
            if "core_financials" in base_model:
                base_fcf = base_model["core_financials"]["reported"].get("free_cash_flow")
            if base_fcf is None:
                continue

            scenario_dcf = multi_stage_dcf(
                base_fcf=base_fcf,
                years_stage1=scenario_assumptions.get("years_stage1", 5),
                g1=scenario_assumptions.get("g1", 0.06),
                years_stage2=scenario_assumptions.get("years_stage2", 5),
                g2=scenario_assumptions.get("g2", 0.04),
                g_terminal=scenario_assumptions.get("g_terminal", 0.025),
                wacc=scenario_assumptions.get("wacc", 0.09),
            )

            base_ev = base_dcf.get("enterprise_value", 0) or 0
            scenario_ev = scenario_dcf.get("enterprise_value", 0) or 0
            variance_pct = ((scenario_ev - base_ev) / base_ev * 100.0) if base_ev else 0.0

            scenario_results[scenario_name] = {
                "assumptions": scenario_assumptions,
                "enterprise_value": scenario_ev,
                "variance_from_base": variance_pct,
                "terminal_value_pct": scenario_dcf.get("terminal_value_pct"),
                "description": scenario.get("description", ""),
            }
        except Exception as e:
            logger.error("Error in scenario %s: %s", scenario_name, e)
            scenario_results[scenario_name] = {"error": str(e)}

    return {
        "base_case": {
            "enterprise_value": base_dcf.get("enterprise_value"),
            "assumptions": base_assumptions,
        },
        "scenarios": scenario_results,
        "scenario_count": len(scenarios),
    }

def monte_carlo_simulation(base_model: Dict[str, Any], num_simulations: int = 1000,
                           parameter_ranges: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    """Perform Monte Carlo simulation on DCF model."""
    import random, statistics

    if "dcf_valuation" not in base_model:
        return {"error": "Base model must contain DCF valuation"}

    if parameter_ranges is None:
        parameter_ranges = {
            "g1": {"min": 0.02, "max": 0.12, "distribution": "normal"},
            "g2": {"min": 0.02, "max": 0.06, "distribution": "normal"},
            "g_terminal": {"min": 0.015, "max": 0.035, "distribution": "normal"},
            "wacc": {"min": 0.06, "max": 0.12, "distribution": "normal"},
        }

    base_dcf = base_model["dcf_valuation"]
    base_assumptions = base_dcf["assumptions"]

    base_fcf = None
    if "core_financials" in base_model:
        base_fcf = base_model["core_financials"]["reported"].get("free_cash_flow")
    if base_fcf is None:
        return {"error": "Cannot determine base FCF for simulation"}

    results: List[float] = []
    for _ in range(num_simulations):
        sim = base_assumptions.copy()
        for param, rng in parameter_ranges.items():
            if param in sim:
                lo, hi = rng["min"], rng["max"]
                if rng.get("distribution") == "normal":
                    current = sim[param]
                    std = (hi - lo) / 6.0
                    val = random.normalvariate(current, std)
                    sim[param] = max(lo, min(hi, val))
                else:
                    sim[param] = random.uniform(lo, hi)

        try:
            out = multi_stage_dcf(
                base_fcf=base_fcf,
                years_stage1=sim.get("years_stage1", 5),
                g1=sim["g1"],
                years_stage2=sim.get("years_stage2", 5),
                g2=sim["g2"],
                g_terminal=sim["g_terminal"],
                wacc=sim["wacc"],
            )
            results.append(out["enterprise_value"])
        except Exception:
            continue

    if not results:
        return {"error": "All simulations failed"}

    results_sorted = sorted(results)
    mean_ev   = statistics.mean(results)
    median_ev = statistics.median(results)
    std_dev   = statistics.stdev(results) if len(results) > 1 else 0.0

    def pct(p):  # p in [0,1]
        idx = int(p * (len(results_sorted) - 1))
        return results_sorted[idx]

    percentiles = {
        "p5":  pct(0.05),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
    }

    return {
        "simulation_count": len(results),
        "base_case_ev": base_dcf.get("enterprise_value"),
        "statistics": {
            "mean": mean_ev, "median": median_ev, "std_dev": std_dev,
            "min": min(results), "max": max(results),
        },
        "percentiles": percentiles,
        "parameter_ranges": parameter_ranges,
        "confidence_intervals": {
            "80%": [percentiles["p10"], percentiles["p90"]],
            "90%": [percentiles["p5"],  percentiles["p95"]],
        },
    }

# -----------------------------------------------------------------------------
# COMPREHENSIVE MODEL (top-level)
# -----------------------------------------------------------------------------
def comprehensive_financial_model(symbol: str, period: str = "annual",
                                  force_refresh: bool = False,
                                  peers: Optional[List[str]] = None,
                                  include_scenarios: bool = True,
                                  include_monte_carlo: bool = False) -> Dict[str, Any]:
    """Build the most comprehensive financial model with all enhancements."""
    try:
        base_model = enhanced_build_model(symbol, period, force_refresh, peers)
        if "error" in base_model:
            return base_model

        income_data   = fmp.income_statement(symbol, period=period, limit=8)
        balance_data  = fmp.balance_sheet(symbol,  period=period, limit=8)
        cashflow_data = fmp.cash_flow(symbol,      period=period, limit=8)

        # Market data for valuation ratios
        try:
            market_data: Dict[str, Any] = {}
            evd = fmp.enterprise_values(symbol, period="quarter", limit=1).get("enterpriseValues") or []
            if evd:
                market_data["market_cap"] = evd[0].get("marketCapitalization")
            price = fmp.latest_price(symbol)
            if price:
                market_data["share_price"] = price
        except Exception:
            market_data = {}

        comprehensive_ratios = calculate_comprehensive_ratios(
            income_data, balance_data, cashflow_data, market_data
        )
        base_model["comprehensive_ratios"] = comprehensive_ratios

        if include_scenarios:
            default_scenarios = [
                {"name": "Bull Case", "description": "Optimistic growth assumptions",
                 "assumptions": {"g1": 0.10, "g2": 0.06, "wacc": 0.08}},
                {"name": "Bear Case", "description": "Conservative growth assumptions",
                 "assumptions": {"g1": 0.02, "g2": 0.02, "wacc": 0.11}},
                {"name": "High Interest Rate", "description": "Higher cost of capital environment",
                 "assumptions": {"wacc": 0.12}},
            ]
            base_model["scenario_analysis"] = scenario_analysis(base_model, default_scenarios)

        if include_monte_carlo:
            base_model["monte_carlo_simulation"] = monte_carlo_simulation(base_model, num_simulations=1000)

        base_model["model_type"] = "comprehensive_institutional"
        return base_model

    except Exception as e:
        logger.error("Error building comprehensive model for %s: %s", symbol, e)
        return {"symbol": symbol.upper(), "error": str(e)}

