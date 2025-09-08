import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Query, HTTPException
from typing import Optional
from src.services.vector_service import embed_text, search
from src.services.ranking_service import final_score

app = FastAPI(title="Retriever Service")

def _embed(query: str):
    return embed_text(query)

@app.get("/v1/health")
def health():
    return {"status": "healthy"}

@app.get("/v1/retrieve")
def retrieve(query: str = Query(..., min_length=2),
             k: int = Query(5, ge=1, le=50),
             tickers: Optional[str] = Query(None, description="Comma-separated tickers e.g. AAPL,MSFT")):
    try:
        qvec = _embed(query)
        tickers_list = [t.strip().upper() for t in tickers.split(",")] if tickers else None
        raw_hits = search(qvec, k=k*3, tickers=tickers_list)
        rescored = []
        for h in raw_hits:
            p = h.get("payload", {}) or {}
            s = final_score(h["score"], p.get("filing_date"), p.get("source_type"), [p.get("ticker")] if p.get("ticker") else None, tickers_list)
            rescored.append({"id": h["id"], "score": s, "payload": p})
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "k": k, "hits": [h for h in rescored[:k]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)

