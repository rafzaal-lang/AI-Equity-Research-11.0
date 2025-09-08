# ui_minimal.py
from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Form, Query, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from jinja2 import Environment, BaseLoader, select_autoescape

app = FastAPI(title="Equity Research — Minimal UI")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return {"ok": True}

# Quiet Render "HEAD /" health probes
@app.head("/", response_class=PlainTextResponse)
def _head_root():
    return PlainTextResponse("", status_code=200)

BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USD")
DEFAULT_TICKER = (os.getenv("DEFAULT_TICKER", "AAPL") or "AAPL").strip() or "AAPL"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Imports
try:
    from src.services.financial_modeler import build_model
    from src.services.report.composer import compose as compose_report
    from src.services.macro.snapshot import macro_snapshot
    from src.services.quant.signals import momentum as signal_momentum
    try:
        from src.services.quant.signals import rsi as signal_rsi
    except Exception:
        signal_rsi = None
    from src.services.providers import fmp_provider as fmp
except ImportError as e:
    logger.error("Import error: %s", e)

# ---------- Formatting helpers ----------
_CURRENCY_SYMBOLS = {"USD": "$", "CAD": "C$", "EUR": "€", "GBP": "£", "JPY": "¥"}
def _sym(cur: str) -> str:
    return _CURRENCY_SYMBOLS.get((cur or "").upper(), f"{cur or 'USD'} ")

def _fmt_money(x: Any, cur: str = "USD", digits: int = 0) -> str:
    try: v = float(x)
    except Exception: return "—"
    return f"{_sym(cur)}{v:,.{digits}f}"

def _fmt_pct(x: Any, digits: int = 1) -> str:
    try: v = float(x) * 100.0
    except Exception: return "—"
    return f"{v:.{digits}f}%"

def _fmt_num(x: Any, digits: int = 0) -> str:
    try: v = float(x)
    except Exception: return "—"
    return f"{v:,.{digits}f}"

def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])

def _html_error(note: str, status: int = 502) -> HTMLResponse:
    from html import escape
    block = f'<pre class="muted" style="white-space:pre-wrap">{escape(note)}</pre>'
    return HTMLResponse(f"<html><body><h1>Error</h1>{block}</body></html>", status_code=status)

# ---------- Momentum helpers ----------
def _sma(values: List[float], window: int) -> Optional[float]:
    if not values or len(values) < window:
        return None
    return sum(values[-window:]) / float(window)

def _calc_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if signal_rsi:
        try:
            hist = [{"close": c} for c in closes]
            out = signal_rsi(hist, period=period)
            if isinstance(out, dict) and "rsi" in out:
                return float(out["rsi"])
            return float(out)
        except Exception:
            pass
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-i] - closes[-i-1]
        (gains if d > 0 else losses).append(abs(d))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ---------- LLM commentary ----------
def _llm_commentary(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an equity research analyst. Be concise, neutral, and factual."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=350,
        )
        content = resp.choices[0].message.content.strip()
        return content or None
    except Exception as e:
        logger.warning("LLM commentary failed: %s", e)
        return None

def _bulletize(text: str) -> str:
    """Force bullet formatting + ensure Markdown renders as a list."""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # If single paragraph, split into sentence-ish chunks.
    if len(lines) <= 1 and "." in text:
        parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
        lines = parts
    bullets = []
    for ln in lines:
        if ln.startswith(("-", "•")):
            bullets.append("- " + ln.lstrip("-• ").strip())
        else:
            bullets.append("- " + ln)
    return "\n".join(bullets)

def _rule_based_summary(f: Dict[str, Any], dcf: Dict[str, Any], price_note: str | None, trend_note: str | None) -> str:
    rep = (f or {}).get("reported", {})
    margins = (f or {}).get("margins", {})
    ratios = (f or {}).get("ratios", {})
    rev = rep.get("revenue")
    gm = margins.get("gross_margin"); om = margins.get("operating_margin"); nm = margins.get("net_margin")
    dte = ratios.get("debt_to_equity"); roe = ratios.get("roe")
    tv_pct = dcf.get("terminal_value_pct")
    bullets = []
    if price_note: bullets.append(price_note)
    if trend_note: bullets.append(trend_note)
    if rev: bullets.append(f"Scale: { _fmt_money(rev, BASE_CURRENCY) } TTM revenue.")
    if gm is not None and om is not None and nm is not None:
        bullets.append(f"Margins: { _fmt_pct(gm) } gross / { _fmt_pct(om) } operating / { _fmt_pct(nm) } net.")
    if roe is not None:
        bullets.append(f"ROE { _fmt_pct(roe) } ({'high' if roe>0.20 else 'modest' if roe>0.10 else 'low'} capital efficiency).")
    if dte is not None:
        bullets.append(f"Leverage { _fmt_num(dte,2) }× D/E.")
    if tv_pct is not None:
        bullets.append(f"Terminal value is { _fmt_pct(tv_pct) } of EV, consistent with a mature profile.")
    return "\n".join("- " + b for b in bullets) if bullets else "- Key operating and valuation metrics available below."

# ---------- Peer comps ----------
def _fetch_peers(symbol: str) -> List[str]:
    try:
        prof = fmp.profile(symbol) or {}
        sector, industry = prof.get("sector"), prof.get("industry")
        if sector and industry and hasattr(fmp, "peers_by_screener"):
            peers = fmp.peers_by_screener(sector, industry, limit=12)
            uniq: List[str] = []
            for s in peers or []:
                s2 = (s or "").upper()
                if s2 and s2 != symbol.upper() and s2 not in uniq:
                    uniq.append(s2)
            return uniq[:8]
    except Exception:
        pass
    return []

def _build_peer_rows(tickers: List[str]) -> List[List[str]]:
    rows = []
    for sym in tickers:
        try:
            q = fmp.quote(sym) or {}
            km = fmp.key_metrics_ttm(sym) or {}
            if isinstance(km, list): km = km[0] if km else {}
            ratios = (fmp.ratios(sym, period="ttm") or {}).get("ratios") or []
            r0 = ratios[0] if ratios else {}
            pe  = r0.get("priceEarningsRatioTTM") or r0.get("priceEarningsRatio")
            roe = km.get("roeTTM"); gm = km.get("grossProfitMarginTTM"); om = km.get("operatingProfitMarginTTM")
            mcap = q.get("marketCap")
            rows.append([
                sym,
                _fmt_num(pe, 1) if pe is not None else "—",
                _fmt_pct(roe) if roe is not None else "—",
                _fmt_pct(gm) if gm is not None else "—",
                _fmt_pct(om) if om is not None else "—",
                _fmt_money(mcap, BASE_CURRENCY, 0) if mcap is not None else "—",
            ])
        except Exception:
            continue
    return rows

# ---------- Composer sanitizer ----------
def _clean_composer_markdown(md_core: str, ticker: str) -> str:
    """
    Remove leading printed-dict artifact & ensure a clean title.
    Handles cases where the dict and the title appear on the same line.
    """
    if not isinstance(md_core, str):
        return ""
    s = md_core.lstrip()

    # If the very first non-space char is '{', drop the first "line" (or up to first newline).
    if s.startswith("{"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl+1:]
        else:
            # No newline — nuke the blob entirely (we'll add our own header)
            s = ""

    # Also strip if the first line still looks like a Python dict (safety net)
    lines = s.splitlines()
    if lines and lines[0].strip().startswith("{'symbol'"):
        s = "\n".join(lines[1:])

    # If what's left begins with an em-dash title fragment or anything not a markdown header, prepend our header.
    if not s.lstrip().startswith("#"):
        as_of = datetime.utcnow().date().isoformat()
        header = f"# {ticker.upper()} — Equity Research Note\n_As of {as_of}_\n\n"
        s = header + s.lstrip()

    return s

# ---------- Report builder ----------
def _build_report_markdown(ticker: str) -> str:
    """Build report markdown or raise HTTPException with useful details."""
    try:
        model = build_model(ticker, force_refresh=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model error: {type(e).__name__}: {e}")

    if isinstance(model, dict) and "error" in model:
        logger.error("Model error for %s: %r", ticker, model["error"])
        err = model["error"]
        if not isinstance(err, str): err = repr(err)
        raise HTTPException(status_code=400, detail=f"Model error: {err}")

    try:
        try:
            macro = macro_snapshot()
        except Exception:
            macro = {}

        # Price series
        try:
            hist = fmp.historical_prices(ticker, limit=300) or []
        except Exception:
            hist = []

        closes: List[float] = []
        if hist:
            hist_sorted = sorted(
                [h for h in hist if isinstance(h, dict) and h.get("close") is not None and h.get("date")],
                key=lambda r: r["date"]
            )
            closes = [float(h["close"]) for h in hist_sorted]

        try:
            q_mom = signal_momentum(hist) if hist else {}
        except Exception:
            q_mom = {}

        sma20 = _sma(closes, 20)
        sma50 = _sma(closes, 50)
        sma200 = _sma(closes, 200)
        last_px = fmp.latest_price(ticker)
        rsi14 = _calc_rsi(closes, 14) if closes else None

        trend_note = None
        if last_px and sma200:
            if last_px > sma200 and sma50 and sma50 > sma200:
                trend_note = "Price is above 200D and 50D>200D (uptrend bias)."
            elif last_px < sma200 and sma50 and sma50 < sma200:
                trend_note = "Price is below 200D and 50D<200D (downtrend bias)."
        price_note = f"Last price: {_fmt_money(last_px, BASE_CURRENCY, 2)}." if last_px else None

        # Compose baseline (preserve your existing output)
        md_core_raw = compose_report({
            "symbol": ticker.upper(),
            "as_of": "latest",
            "call": "Review",
            "conviction": 7.0,
            "target_low": "—",
            "target_high": "—",
            "base_currency": BASE_CURRENCY,
            "fundamentals": model.get("core_financials", {}),
            "dcf": model.get("dcf_valuation", {}),
            "valuation": {"wacc": None},
            "quant": {
                "last_price": last_px,
                "sma20": sma20, "sma50": sma50, "sma200": sma200,
                "rsi14": rsi14, "momentum": q_mom
            },
            "comps": {"peers": []},
            "citations": [],
            "quarter": {},
            "artifact_id": "ui-session",
        })
        md_core = _clean_composer_markdown(md_core_raw, ticker)

        # Executive Summary
        fundamentals = model.get("core_financials", {}) or model.get("fundamentals", {}) or {}
        dcf          = model.get("dcf_valuation", {})  or model.get("dcf", {})          or {}
        summary_prompt = f"""
Write a concise (4–6 bullets) executive summary for {ticker.upper()} using these metrics:

TTM revenue: {fundamentals.get('reported', {}).get('revenue')}
Margins: gross={fundamentals.get('margins', {}).get('gross_margin')}, operating={fundamentals.get('margins', {}).get('operating_margin')}, net={fundamentals.get('margins', {}).get('net_margin')}
ROE={fundamentals.get('ratios', {}).get('roe')}, ROA={fundamentals.get('ratios', {}).get('roa')}, D/E={fundamentals.get('ratios', {}).get('debt_to_equity')}
DCF EV={dcf.get('enterprise_value')}, Equity={dcf.get('equity_value')}, Per-share={dcf.get('fair_value_per_share')}, Terminal%={dcf.get('terminal_value_pct')}
Recent price: {last_px}, SMA20={sma20}, SMA50={sma50}, SMA200={sma200}, RSI14={rsi14}

Tone: neutral, factual, non-promotional. Interpret; avoid absolute buy/sell language.
"""
        ai_note = _llm_commentary(summary_prompt)
        if ai_note:
            summary = _bulletize(ai_note)
        else:
            summary = _rule_based_summary(fundamentals, dcf, price_note, trend_note)

        # Momentum table
        momentum_rows = [
            ["Last Price",     _fmt_money(last_px, BASE_CURRENCY, 2)],
            ["SMA(20)",        _fmt_money(sma20, BASE_CURRENCY, 2) if sma20 is not None else "—"],
            ["SMA(50)",        _fmt_money(sma50, BASE_CURRENCY, 2) if sma50 is not None else "—"],
            ["SMA(200)",       _fmt_money(sma200, BASE_CURRENCY, 2) if sma200 is not None else "—"],
            ["RSI(14)",        f"{rsi14:.1f}" if rsi14 is not None else "—"],
        ]
        if trend_note:
            momentum_rows.append(["Trend Note", trend_note])

        # Peer comps (best-effort)
        peer_syms = _fetch_peers(ticker)
        peer_rows = _build_peer_rows(peer_syms) if peer_syms else []
        comps_md = ""
        if peer_rows:
            comps_md = "\n".join([
                "## Peer Comps (Quick)",
                _md_table(["Symbol", "P/E (TTM)", "ROE (TTM)", "Gross Mgn", "Op Mgn", "Market Cap"], peer_rows),
                ""
            ])

        # Append our sections after composer
        as_of = datetime.utcnow().date().isoformat()
        extras = []
        extras.append(f"\n---\n\n## Executive Summary (as of {as_of})\n\n")  # extra blank line to ensure bullets render
        extras.append(summary + "\n")
        extras.append("## Price & Momentum\n\n")
        extras.append(_md_table(["Item", "Value"], momentum_rows) + "\n")
        if comps_md:
            extras.append(comps_md)

        return md_core + "\n" + "\n".join(extras)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {type(e).__name__}: {e}")

# ---- Minimal HTML shell ----
BASE_HTML = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Equity Research</title>
<style>
:root { --bg:#f7f7f5; --panel:#fff; --ink:#0a0a0a; --muted:#6b7280; --line:#e5e7eb; --radius:16px; }
*{box-sizing:border-box} html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Helvetica,Arial,"Segoe UI",Roboto,"Noto Sans",sans-serif}
a{color:#0f172a;text-decoration:none} a:hover{text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:24px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.brand{font-weight:700;letter-spacing:.3px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:12px;border:1px solid var(--ink);color:#fff;background:var(--ink);cursor:pointer}
.input{width:100%;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:#fff}
pre{white-space:pre-wrap;background:#f8f9fa;padding:12px;border-radius:8px;overflow-x:auto}
h1,h2,h3{margin-top:8px}
.panel h2{margin-top:0}
</style></head><body>
  <div class="container">
    <header><div class="brand">Equity Research</div></header>
    {{ content | safe }}
  </div>
</body></html>
"""

REPORT_FORM = """
<div class="panel">
  <form method="post" action="/report">
    <div style="margin-bottom:10px;">
      <label>Ticker:</label>
      <input class="input" type="text" name="ticker" placeholder="AAPL" required />
    </div>
    <button class="btn" type="submit">Generate Report</button>
  </form>
</div>
"""

env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
def render(page: str, **kw) -> str:
    tpl = env.from_string(BASE_HTML)
    content_tpl = env.from_string(page)
    return tpl.render(content=content_tpl.render(**kw))

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(render(REPORT_FORM))

@app.get("/report")
def get_report(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    sym = (ticker or symbol or "").strip() or DEFAULT_TICKER
    if not sym:
        raise HTTPException(status_code=400, detail="ticker or symbol is required")
    md = _build_report_markdown(sym)
    return {"symbol": sym.upper(), "markdown": md}

@app.post("/report", response_class=HTMLResponse)
def post_report(ticker: str = Form(...)):
    try:
        md = _build_report_markdown(ticker)
        import markdown as md_parser
        html_content = md_parser.markdown(md, extensions=["tables"])
        result_html = f"""
        <div class="panel">
          <h2>{ticker.upper()} Report</h2>
          <div style="margin-bottom:12px;">
            <a href="/report.md?ticker={ticker}" class="btn" style="text-decoration:none;">Download Markdown</a>
          </div>
          <div style="border:1px solid var(--line);padding:18px;border-radius:12px;">{html_content}</div>
        </div>
        """
        return HTMLResponse(render(result_html))
    except HTTPException as he:
        return _html_error(he.detail, status=he.status_code)
    except Exception as e:
        return _html_error(f"Error generating report: {type(e).__name__}: {e}")

@app.get("/report.md", response_class=PlainTextResponse)
def download_report_md(ticker: str = Query(...)):
    md = _build_report_markdown(ticker)
    return PlainTextResponse(md, headers={"Content-Disposition": f"attachment; filename={ticker}_report.md"})

@app.get("/debug/env", response_class=PlainTextResponse)
def debug_env():
    keys = [
        ("FMP_API_KEY", bool(os.getenv("FMP_API_KEY"))),
        ("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY"))),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("BASE_CURRENCY", os.getenv("BASE_CURRENCY", "")),
        ("DEFAULT_TICKER", DEFAULT_TICKER),
    ]
    lines = [f"{k}={'SET' if v else 'MISSING'}" if isinstance(v, bool) else f"{k}={v}" for k, v in keys]
    return PlainTextResponse("\n".join(lines))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")))

