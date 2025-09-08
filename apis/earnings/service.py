import os
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/earnings", tags=["earnings"])
FMP_API_KEY = os.getenv("FMP_API_KEY")

@router.get("/flash/{ticker}")
def flash(ticker: str):
    if not FMP_API_KEY:
        raise HTTPException(500, "FMP_API_KEY not configured")
    t = ticker.upper()
    url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{t}?apikey={FMP_API_KEY}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"FMP error: {r.text}")
    rows = r.json() or []
    if not rows:
        raise HTTPException(404, "No earnings data")
    r0 = rows[0]
    eps_act = r0.get("actualEarningResult")
    eps_est = r0.get("estimatedEarning")
    rev_act = r0.get("revenue")
    rev_est = r0.get("estimatedRevenue")
    period = r0.get("date")
    eps_surp_pct = None; rev_surp_pct = None
    try:
        if eps_act is not None and eps_est not in (None, 0):
            eps_surp_pct = 100.0 * (eps_act - eps_est) / abs(eps_est)
        if rev_act is not None and rev_est not in (None, 0):
            rev_surp_pct = 100.0 * (rev_act - rev_est) / abs(rev_est)
    except Exception:
        pass
    tag = "neu"
    if (eps_surp_pct is not None and eps_surp_pct >= 3.0) or (rev_surp_pct is not None and rev_surp_pct >= 2.0):
        tag = "pos"
    if (eps_surp_pct is not None and eps_surp_pct <= -3.0):
        tag = "neg"
    bullets = []
    if eps_surp_pct is not None: bullets.append(f"EPS surprise: {eps_surp_pct:.1f}%")
    if rev_surp_pct is not None: bullets.append(f"Revenue surprise: {rev_surp_pct:.1f}%")
    return {
        "ticker": t, "period": period,
        "actuals": {"eps": eps_act, "revenue": rev_act},
        "consensus": {"eps": eps_est, "revenue": rev_est},
        "surprise": {"eps_pct": eps_surp_pct, "rev_pct": rev_surp_pct},
        "tag": tag, "bullets": bullets
    }
