import os, datetime as dt, math
from typing import Dict, List, Optional
import statistics as stats

import yaml
from fastapi import APIRouter, HTTPException, Query
from src.services.providers import fmp_provider as fmp

router = APIRouter(prefix="/sectors", tags=["sectors"])

DEFAULT_THEMES = {
    "sector": {
        "CleanTech": ["ENPH","SEDG","FSLR","RUN","PLUG"],
        "Semiconductors": ["NVDA","AMD","AVGO","INTC","TSM"],
        "BigTech": ["AAPL","MSFT","GOOGL","AMZN","META"],
    },
    "subsector": {
        "Green_Hydrogen": ["PLUG","BE"],
        "Utility_Solar": ["NEE","DUK","D","ED"],
        "AI_Infra": ["NVDA","AVGO","ASML","MU"],
    },
}

def _load_themes() -> Dict[str, Dict[str, List[str]]]:
    path = os.path.join("config","themes.yaml")
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
                return {"sector": y.get("sector",{}), "subsector": y.get("subsector",{})}
        except Exception:
            pass
    return DEFAULT_THEMES

def _sma(values: List[float], n: int) -> Optional[float]:
    if len(values) < n: return None
    return sum(values[-n:]) / n

def _pct(a: float, b: float) -> Optional[float]:
    try:
        return (a/b - 1.0) if b else None
    except Exception:
        return None

def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period+1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d,0.0))
        losses.append(max(-d,0.0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _get_closes(symbol: str, days: int = 365) -> List[float]:
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    hist = fmp.historical_prices(symbol, start=start, end=end, limit=10000) or []
    closes = [float(h["close"]) for h in sorted(hist, key=lambda x: x.get("date")) if h.get("close") is not None]
    return closes

@router.get("/recommendations")
def recommendations(windows: str = Query("5d,3m,6m"), group: str = Query("subsector")):
    if not os.getenv("FMP_API_KEY"):
        raise HTTPException(500, "FMP_API_KEY not configured")
    theme_map = _load_themes().get(group) or {}
    if not theme_map:
        raise HTTPException(500, f"No groups found for '{group}'. Add to config/themes.yaml.")
    win_list = [w.strip().lower() for w in windows.split(",") if w.strip()]
    win_to_days = {}
    for w in win_list:
        if w.endswith("d"): win_to_days[w] = int(w[:-1])
        elif w.endswith("m"): win_to_days[w] = int(w[:-1]) * 21
        else: win_to_days[w] = 63

    items = []
    for gname, tickers in theme_map.items():
        series = []
        for t in set(tickers):
            closes = _get_closes(t)
            if closes: series.append(closes)
        if not series:
            continue
        m = min(len(s) for s in series)
        stack = [row[-m:] for row in series]
        basket = [sum(row)/len(row) for row in zip(*stack)]
        mom = {}
        for w, nd in win_to_days.items():
            if len(basket) > nd:
                mom[w] = _pct(basket[-1], basket[-nd-1])
            else:
                mom[w] = None
        above50 = []; above200 = []
        for s in series:
            s50 = _sma(s,50); s200 = _sma(s,200)
            last = s[-1]
            if s50 is not None: above50.append(1.0 if last> s50 else 0.0)
            if s200 is not None: above200.append(1.0 if last> s200 else 0.0)
        breadth = {
            "pct_above_50dma": round(100*sum(above50)/max(1,len(above50)),1) if above50 else None,
            "pct_above_200dma": round(100*sum(above200)/max(1,len(above200)),1) if above200 else None,
        }
        items.append({"name": gname, "tickers": list(dict.fromkeys(tickers)), "mom": mom, "ta": {"breadth": breadth}})
    def score(it):
        m = it["mom"]; sc = 0.0
        if m.get("5d") is not None: sc += 0.5*m["5d"]
        if m.get("3m") is not None: sc += 0.3*m["3m"]
        if m.get("6m") is not None: sc += 0.2*m["6m"]
        return sc
    items.sort(key=score, reverse=True)
    return {"group": group, "windows": win_list, "items": items}

@router.get("/{group_name}/drilldown")
def drilldown(group_name: str, window: str = "3m", show: str = "all", limit: int = 50):
    theme_map = _load_themes().get("subsector", {}) | _load_themes().get("sector", {})
    if group_name not in theme_map:
        raise HTTPException(404, f"Unknown group {group_name}")
    tickers = list(dict.fromkeys(theme_map[group_name]))
    rows = []
    for t in tickers:
        closes = _get_closes(t)
        if len(closes) < 210:
            continue
        def win_ret(w: str):
            if w.endswith("d"): nd = int(w[:-1])
            elif w.endswith("m"): nd = int(w[:-1]) * 21
            else: nd = 63
            return _pct(closes[-1], closes[-nd-1]) if len(closes) > nd else None
        rsi = _rsi(closes, 14)
        s50 = _sma(closes, 50); s200 = _sma(closes, 200)
        rows.append({
            "ticker": t,
            "mom": {"5d": win_ret("5d"), "3m": win_ret("3m"), "6m": win_ret("6m")},
            "ta": {"rsi": rsi, "dma_state": {"above_50": (closes[-1] > s50) if s50 else None, "above_200": (closes[-1] > s200) if s200 else None}}
        })
    rows = [r for r in rows if r["mom"]["3m"] is not None]
    rows.sort(key=lambda r: r["mom"]["3m"], reverse=True)
    n = len(rows); top_cut = max(1, int(n*0.1))
    leaders = set(x["ticker"] for x in rows[:top_cut])
    laggards = set(x["ticker"] for x in rows[-top_cut:])
    for r in rows:
        r["flag"] = "leader" if r["ticker"] in leaders else "laggard" if r["ticker"] in laggards else "neutral"
    if show == "leaders": rows = [r for r in rows if r["flag"]=="leader"]
    if show == "laggards": rows = [r for r in rows if r["flag"]=="laggard"]
    return {"group": group_name, "items": rows[:limit]}
