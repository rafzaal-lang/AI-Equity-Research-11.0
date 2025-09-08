import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from src.services.financial_modeler import build_model, ttm_snapshot
from src.services.monitoring.metrics import Timer, record
from src.services.cache.redis_client import ping as redis_ping

app = FastAPI(title="Financial Model Service")

class ModelOut(BaseModel):
    symbol: str
    model_type: str | None = None
    core_financials: dict | None = None
    ttm_snapshot: dict | None = None
    dcf_valuation: dict | None = None

@app.get("/v1/health")
def health():
    ok_redis = redis_ping()
    return {"status": "healthy" if ok_redis else "degraded", "services": {"redis": "ok" if ok_redis else "down"}}

@app.get("/v1/model/{ticker}", response_model=ModelOut)
def get_model(ticker: str, period: str = Query("annual", enum=["annual","quarter"]), force_refresh: bool=False):
    route="/v1/model"
    with Timer(route):
        try:
            m = build_model(ticker, period=period, force_refresh=force_refresh)
            if "error" in m:
                record(route, "404")
                raise HTTPException(status_code=404, detail=m.get("error"))
            record(route, "200")
            return m
        except HTTPException:
            raise
        except Exception as e:
            record(route, "500")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)

