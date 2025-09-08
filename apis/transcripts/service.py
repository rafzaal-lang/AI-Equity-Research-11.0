import os, re, requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/transcripts", tags=["transcripts"])
FMP_API_KEY = os.getenv("FMP_API_KEY")

QA_KEYS = ["question-and-answer", "q&a", "question and answer"]

@router.get("/qa/{ticker}")
def qa(ticker: str, limit: int = 3):
    if not FMP_API_KEY:
        raise HTTPException(500, "FMP_API_KEY not configured")
    t = ticker.upper()
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{t}?apikey={FMP_API_KEY}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(502, f"FMP error: {r.text}")
    data = r.json() or []
    out = []
    for row in data[:limit]:
        txt = row.get("content") or ""
        date = row.get("date") or row.get("quarter")
        out.append({"date": date, "qa_markdown": _extract_qa_markdown(txt)})
    return {"ticker": t, "items": out}

def _extract_qa_markdown(text: str) -> str:
    if not text: return "(no transcript content)"
    low = text.lower(); start = 0
    for k in QA_KEYS:
        i = low.find(k)
        if i != -1: start = i; break
    snippet = text[start:]
    snippet = re.sub(r"\n{3,}", "\n\n", snippet)
    snippet = re.sub(r"operator:", "**Operator:**", snippet, flags=re.I)
    snippet = re.sub(r"analyst:", "**Analyst:**", snippet, flags=re.I)
    return snippet.strip()[:12000]
