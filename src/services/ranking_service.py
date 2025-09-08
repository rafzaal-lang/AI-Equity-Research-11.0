import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../retriever/config.yaml")

def load_config() -> Dict[str, float]:
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            return config.get("ranking", {
                "alpha": 0.4,  # semantic similarity
                "beta": 0.3,   # recency
                "gamma": 0.2,  # source type priority
                "delta": 0.1   # ticker match
            })
    except Exception:
        return {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}

def recency_score(filing_date: Optional[str]) -> float:
    if not filing_date:
        return 0.0
    try:
        file_dt = datetime.strptime(filing_date, "%Y-%m-%d")
        days_ago = (datetime.now() - file_dt).days
        # Decay over 2 years
        return max(0.0, 1.0 - (days_ago / 730))
    except Exception:
        return 0.0

def source_type_score(source_type: Optional[str]) -> float:
    priorities = {"10-K": 1.0, "10-Q": 0.8, "8-K": 0.6, "DEF 14A": 0.4}
    return priorities.get(source_type, 0.2)

def ticker_match_score(doc_ticker: Optional[str], query_tickers: Optional[List[str]]) -> float:
    if not doc_ticker or not query_tickers:
        return 0.5
    return 1.0 if doc_ticker.upper() in [t.upper() for t in query_tickers] else 0.0

def final_score(semantic_score: float, filing_date: Optional[str], 
                source_type: Optional[str], doc_tickers: Optional[List[str]], 
                query_tickers: Optional[List[str]]) -> float:
    config = load_config()
    
    recency = recency_score(filing_date)
    source = source_type_score(source_type)
    ticker = ticker_match_score(doc_tickers[0] if doc_tickers else None, query_tickers)
    
    return (config["alpha"] * semantic_score + 
            config["beta"] * recency + 
            config["gamma"] * source + 
            config["delta"] * ticker)

