# src/services/report/professional_report_generator.py
from __future__ import annotations

import os
import math
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.getenv("PROJECT_ROOT", os.getcwd())
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional

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
# HTTP helpers (FMP)
# ---------------------------
def _fmp_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if not FMP_API_KEY:
        raise RuntimeError("Missing FMP_API_KEY")

    url = f"https://financialmodelingprep.com{path}"
    q = {"apikey": FMP_API_KEY}
    if params:
        q.update(params)
    try:
        r = requests.get(url, params=q, timeout=20)
        r.raise_for_status()
        return r.json()
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
    prof = _fmp_get(f"/api/v3/profile/{ticker}")
    p = prof[0] if isinstance(prof, list) and prof else {}
    return {
        "description": p.get("description") or p.get("companyName"),
        "sector": p.get("sector"),
        "industry": p.get("industry"),
    }


def fetch_quote_block(ticker: str) -> Dict[str, Optional[float]]:
    ql = _fmp_get(f"/api/v3/quote/{ticker}") or []
    q = ql[0] if ql else {}
    price = _as_float(q.get("price"))
    market_cap = _as_float(q.get("marketCap"))

    # balance sheet for debt/cash (latest quarter)
    bsl = _fmp_get(f"/api/v3/balance-sheet-statement/{ticker}", {"period": "quarter", "limit": 1}) or []
    bs = bsl[0] if bsl else {}
    cash = _as_float(bs.get("cashAndShortTermInvestments") or bs.get("cashAndCashEquivalents"))
    debt = _as_float(bs.get("totalDebt") or (bs.get("shortTermDebt") or 0) + (bs.get("longTermDebt") or 0))
    ev = (market_cap or 0) + (debt or 0) - (cash or 0)

    # historical closes for 52-day SMA
    hist = _fmp_get(f"/api/v3/historical-price-full/{ticker}", {"serietype": "line", "timeseries": 120}) or {}
    series = hist.get("historical") or []
    closes = [ _as_float(x.get("close")) for x in series if _as_float(x.get("close")) is not None ]
    sma52 = _sma(closes, 52)

    return {"price": price, "market_cap": market_cap, "cash": cash, "debt": debt, "ev": ev if market_cap else None, "sma52": sma52}


def _pick_number(d: dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = _as_float(d.get(k))
        if v is not None:
            return v
    return None


def fetch_quarterlies(ticker: str) -> Dict[str, Any]:
    """Return last up to 8 quarters of IS + CF for snapshot/TTM/YTD."""
    isl = _fmp_get(f"/api/v3/income-statement/{ticker}", {"period": "quarter", "limit": 8}) or []
    cfl = _fmp_get(f"/api/v3/cash-flow-statement/{ticker}", {"period": "quarter", "limit": 8}) or []
    return {"is": isl, "cf": cfl}


def fetch_annuals(ticker: str) -> Dict[str, Any]:
    """Return last up to 5 annual IS + CF for 2-year table & 5y avgs."""
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
    """
    Best-effort analyst estimates (revenue, ebitda) if FMP provides.
    We try both v3 endpoints variants.
    """
    # Try /v3/analyst-estimates
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
# Derived calculations
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
    """Return (latest_quarter_row, same_quarter_prior_year_row) using date keys."""
    if not rows:
        return None, None
    q0 = rows[0]
    date0 = q0.get("date") or q0.get("calendarYear")
    if not date0:
        return q0, None
    y0 = int(str(date0)[:4])
    # Find row with year-1 and same quarter if possible
    q1 = None
    for r in rows[1:]:
        d = r.get("date") or r.get("calendarYear")
        if not d:
            continue
        y = int(str(d)[:4])
        if y == y0 - 1:
            q1 = r
            break
    return q0, q1


def build_financial_blocks(ticker: str) -> Dict[str, Any]:
    q = fetch_quarterlies(ticker)
    a = fetch_annuals(ticker)

    q_is = q["is"]; q_cf = q["cf"]
    a_is = a["is"]; a_cf = a["cf"]

    # TTM (from last 4 quarterlies)
    rev_ttm = _sum_last_n_quarters(q_is, ["revenue", "totalRevenue"], 4)
    ebitda_ttm = _sum_last_n_quarters(q_is, ["ebitda", "EBITDA"], 4)
    gp_ttm = _sum_last_n_quarters(q_is, ["grossProfit"], 4)
    ni_ttm = _sum_last_n_quarters(q_is, ["netIncome"], 4)
    ocf_ttm = _sum_last_n_quarters(q_cf, ["netCashProvidedByOperatingActivities", "netCashProvidedByUsedInOperatingActivities"], 4)
    capex_ttm = _sum_last_n_quarters(q_cf, ["capitalExpenditure", "capitalExpenditures"], 4)
    fcf_ttm = (ocf_ttm + capex_ttm) if (ocf_ttm is not None and capex_ttm is not None) else None  # capex is negative

    # Quarter snapshot & YoY
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

    # YTD (sum quarterlies of current calendar year)
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

    # Annual last two years (table)
    def _two_year_table() -> List[Dict[str, Any]]:
        def pick(row: dict, keys: List[str]) -> Optional[float]:
            return _pick_number(row, keys)
        rows: List[Dict[str, Any]] = []
        years = []
        for row in a_is[:2]:
            years.append(row.get("date") or row.get("calendarYear"))
        years = [str(y) for y in years if y][:2]

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
    all_tickers = [subject] + peers
    for t in all_tickers:
        q = fetch_quote_block(t)
        fin = build_financial_blocks(t)  # uses quarterlies cached by FMP
        mult = compute_multiples(q["ev"], fin["ttm"])
        # dividend & FCF yield (best-effort)
        prof_quote = _fmp_get(f"/api/v3/quote/{t}") or []
        pq = prof_quote[0] if prof_quote else {}
        dividend_yield = _as_float(pq.get("yield"))  # in percent already per FMP
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

    # 5-year “average” multiples for subject (approx: latest EV / historical annuals)
    five_avg = {"EV/S 5y Avg": None, "EV/EBITDA 5y Avg": None}
    try:
        annuals = fetch_annuals(subject)
        ev_latest = fetch_quote_block(subject).get("ev")
        if ev_latest:
            rev = [ _pick_number(r, ["revenue", "totalRevenue"]) for r in annuals["is"][:5] ]
            ebt = [ _pick_number(r, ["ebitda", "EBITDA"]) for r in annuals["is"][:5] ]
            ev_s_vals = [ (ev_latest / r) if r else np.nan for r in rev ]
            ev_eb_vals = [ (ev_latest / e) if e else np.nan for e in ebt ]
            if not np.isnan(ev_s_vals).all():
                five_avg["EV/S 5y Avg"] = float(np.nanmean(ev_s_vals))
            if not np.isnan(ev_eb_vals).all():
                five_avg["EV/EBITDA 5y Avg"] = float(np.nanmean(ev_eb_vals))
    except Exception as e:
        logger.warning("5y avg multiples failed: %s", e)

    return df, five_avg


def sector_momentum(sector_guess: Optional[str]) -> Dict[str, Any]:
    """1M/3M returns for sector ETFs vs SPY (all via FMP)."""
    tickers = list(SECTOR_SPDRS.values()) + [SPY]
    ret = {}
    for t in tickers:
        h = _fmp_get(f"/api/v3/historical-price-full/{t}", {"serietype": "line", "timeseries": 70}) or {}
        hist = h.get("historical") or []
        closes = [ _as_float(x.get("close")) for x in hist if _as_float(x.get("close")) is not None ]
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
    closes = [ _as_float(x.get("close")) for x in hist if _as_float(x.get("close")) is not None ]
    return {"rsi_14": _rsi(closes, 14) if closes else None}


# ---------------------------
# LLM commentary (optional)
# ---------------------------
def llm_commentary(payload: Dict[str, Any], ticker: str) -> Dict[str, str]:
    try:
        if not OPENAI_API_KEY:
            return {"financials": "", "industry": "", "sector": ""}
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        def ask(prompt: str) -> str:
            try:
                r = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are a senior equity research analyst. Be concise, factual, objective."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return r.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                return ""

        fin = payload.get("fin")
        prof = payload.get("profile")
        sect = payload.get("sector")
        tech = payload.get("tech")

        p1 = f"Summarize last quarter and YTD for {ticker} with YoY context. Data: {fin}. Focus: Revenue, GM%, EBITDA, NI, OCF, FCF. 5-8 bullets."
        p2 = f"Industry snapshot & competitive positioning for {ticker} in 5-8 bullets. Sector={prof.get('sector')}, Industry={prof.get('industry')}."
        p3 = f"Sector view using 1M/3M momentum vs SPY and RSI(14). Data: {sect}. Technicals: {tech}. 5 bullets max."

        return {"financials": ask(p1), "industry": ask(p2), "sector": ask(p3)}
    except Exception:
        return {"financials": "", "industry": "", "sector": ""}


# ---------------------------
# Render HTML
# ---------------------------
def render_pro_report_html(ticker: str) -> str:
    t = ticker.upper().strip()

    profile = fetch_profile(t)
    quote = fetch_quote_block(t)
    fundamentals = build_financial_blocks(t)
    multiples = compute_multiples(quote.get("ev"), fundamentals["ttm"])
    estimates = fetch_estimates_optional(t)
    tech = ticker_technicals(t)
    sector = sector_momentum(profile.get("sector"))

    # peers
    peers = fetch_peers(t)
    if not peers:
        # heuristic fallback: take top sector ETF holdings (static) but exclude self
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
        {"label": "Share Price", "value": quote.get("price")},
        {"label": "52-day Avg", "value": quote.get("sma52")},
        {"label": "Market Cap", "value": quote.get("market_cap")},
        {"label": "Debt", "value": quote.get("debt")},
        {"label": "Cash", "value": quote.get("cash")},
        {"label": "Enterprise Value (EV)", "value": quote.get("ev")},
        {"label": "EV/S (TTM)", "value": multiples.get("ev_s")},
        {"label": "EV/EBITDA (TTM)", "value": multiples.get("ev_ebitda")},
    ]

    fin_snapshot = {
        "quarter": fundamentals["quarter"],
        "ytd": fundamentals["ytd"],
        "ttm": fundamentals["ttm"],
    }

    sector_inputs = {
        "chosen_sector_etf": sector.get("sector_etf"),
        "returns": sector.get("returns"),
        "spy": sector.get("spy"),
        "peers": peers,
        "peer_table_preview": peer_df.head(6).to_dict(orient="records"),
    }

    commentary = llm_commentary(
        {"fin": fin_snapshot, "profile": profile, "sector": sector_inputs, "tech": tech},
        t,
    )

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html", "xml"]))
    tpl = env.get_template("pro_report.html")
    html = tpl.render(
        as_of=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        ticker=t,
        profile=profile,
        header_table=header_table,
        two_year=fundamentals["annual_two_years_table"],
        estimates=estimates,
        fin_snapshot=fin_snapshot,
        peer_rows=peer_df.to_dict(orient="records"),
        peer_five_year=five_year,
        sector_inputs=sector_inputs,
        technical=tech,
        commentary=commentary,
        static_path="/static/pro_report.css",
    )

    out_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pro_report_{t}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
