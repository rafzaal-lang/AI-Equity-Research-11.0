import sys, os, re, json, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime, date
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Response, Request, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.services.report.composer import compose
from src.services.financial_modeler import build_model
from src.services.macro.snapshot import macro_snapshot
from src.services.quant.signals import momentum, rsi, sma_cross
from src.services.comps.engine import comps_table
from src.services.providers import fmp_provider as fmp

log = logging.getLogger(__name__)

app = FastAPI(title="Reports Service")

# Permissive CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "AAPL").strip() or "AAPL"

class ReportResponse(BaseModel):
    symbol: str
    markdown: str

class ReportRequest(BaseModel):
    ticker: Optional[str] = Field(default=None)
    symbol: Optional[str] = Field(default=None)

    def get_symbol(self) -> str:
        return (self.ticker or self.symbol or "").strip()

# Utilities
_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")

def _looks_like_ticker(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or len(s) > 10:
        return False
    return bool(_TICKER_RE.match(s))

async def _extract_symbol_from_request(request: Request) -> Optional[str]:
    """Extract symbol from various request formats."""
    # 1) Try headers
    for header in ("x-ticker", "x-symbol", "ticker", "symbol"):
        value = request.headers.get(header)
        if value and _looks_like_ticker(value):
            return value.strip()
    
    # 2) Try query params
    for param in ("ticker", "symbol", "t", "s", "q"):
        value = request.query_params.get(param)
        if value and _looks_like_ticker(value):
            return value.strip()
    
    # 3) Try JSON body
    try:
        data = await request.json()
        if isinstance(data, dict):
            for key in ("ticker", "symbol", "t", "s", "q"):
                value = data.get(key)
                if value and _looks_like_ticker(str(value)):
                    return str(value).strip()
        elif isinstance(data, str) and _looks_like_ticker(data):
            return data.strip()
    except Exception:
        pass
    
    # 4) Try form data
    try:
        form = await request.form()
        for key, value in form.multi_items():
            if _looks_like_ticker(str(value)):
                return str(value).strip()
    except Exception:
        pass
    
    # 5) Try raw body as plain text
    try:
        raw = (await request.body()).decode(errors="ignore").strip()
        if raw and _looks_like_ticker(raw):
            return raw
    except Exception:
        pass
    
    return None

# Routes
@app.get("/")
def root():
    return {
        "service": "AI Equity Research - Reports Service",
        "version": "9.3",
        "status": "healthy",
        "endpoints": {
            "health": "/v1/health",
            "report_get": "/v1/report/{ticker}",
            "report_post": "/report",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_simple():
    return {"status": "healthy"}

@app.get("/v1/health")
def health():
    return {"status": "healthy", "service": "Reports"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _build_report_markdown(ticker: str) -> str:
    """Build report markdown for a ticker."""
    model = build_model(ticker)
    if isinstance(model, dict) and "error" in model:
        raise HTTPException(status_code=404, detail=model["error"])

    macro = macro_snapshot()

    # Get price history for quant signals
    try:
        hist = fmp.historical_prices(ticker, limit=300) or []
    except Exception:
        hist = []

    try:
        q_mom = momentum(hist) if hist else {}
    except Exception:
        q_mom = {}
    try:
        q_rsi = rsi(hist) if hist else {}
    except Exception:
        q_rsi = {}
    try:
        q_sma = sma_cross(hist) if hist else {}
    except Exception:
        q_sma = {}

    # Build markdown
    md = compose(
        ticker.upper(),
        as_of=(macro or {}).get("as_of") or date.today().isoformat(),
        data={
            "call": "Review",
            "conviction": 6.5,
            "target_low": "—",
            "target_high": "—",
            "momentum": q_mom,
            "fundamentals": model.get("core_financials", {}),
            "dcf": model.get("dcf_valuation", {}),
            "valuation": model.get("valuation", {}),
            "comps": {
                "peers": (model.get("comps") or {}).get("peers")
                         or (comps_table(ticker) or {}).get("peers", [])
            },
            "quarter": model.get("quarter", {}),
            "citations": model.get("citations", []),
            "risks": model.get("risks", []),
        },
    )
    return md

# Report endpoints
@app.get("/v1/report/{ticker}", response_model=ReportResponse)
def get_report_v1(ticker: str):
    """Get report for a ticker via URL path."""
    try:
        md = _build_report_markdown(ticker)
        return ReportResponse(symbol=ticker.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report", response_model=ReportResponse)
def get_report_compat(ticker: Optional[str] = Query(default=None), symbol: Optional[str] = Query(default=None)):
    """Get report via query parameters."""
    sym = (ticker or symbol or "").strip() or DEFAULT_TICKER
    if not sym:
        raise HTTPException(status_code=400, detail="ticker or symbol is required")
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/report", response_model=ReportResponse)
def post_report_v1(req: ReportRequest):
    """Post report with JSON body."""
    sym = req.get_symbol() or DEFAULT_TICKER
    if not sym:
        raise HTTPException(status_code=400, detail="ticker or symbol is required")
    try:
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report", response_model=ReportResponse)
async def post_report_compat(request: Request):
    """POST /report - accepts multiple formats."""
    try:
        # Extract symbol from request
        sym = await _extract_symbol_from_request(request)
        
        if not sym:
            sym = DEFAULT_TICKER
            log.info("No ticker found in request, using default: %s", sym)
        
        if not sym:
            raise HTTPException(status_code=400, detail="ticker or symbol is required")
        
        md = _build_report_markdown(sym)
        return ReportResponse(symbol=sym.upper(), markdown=md)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error("Error in POST /report: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8086")))
