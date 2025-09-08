import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from fastapi import APIRouter, HTTPException
from src.services.comps.engine import comps_table, peer_universe

router = APIRouter(prefix="/comps", tags=["comps"])

@router.get("/{ticker}/table")
def comps_table_endpoint(ticker: str):
    try:
        data = comps_table(ticker)
        return data
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/{ticker}/peers")
def comps_peers_endpoint(ticker: str, size: int = 15):
    try:
        peers = peer_universe(ticker, max_peers=size)
        return {"ticker": ticker.upper(), "peers": peers}
    except Exception as e:
        raise HTTPException(500, str(e))
