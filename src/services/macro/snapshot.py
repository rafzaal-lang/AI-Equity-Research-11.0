from __future__ import annotations
from typing import Dict, Any, Optional, List
from src.services.providers import fred_provider as fred

SERIES = {
    "cpi": "CPIAUCSL",
    "ten_year": "DGS10",
    "fed_funds": "DFF",
    "hy_oas": "BAMLH0A0HYM2",
}

def _last(series: List[dict]) -> Optional[float]:
    vals = [float(x["value"]) for x in series if x.get("value") not in (None, ".", "")]
    return vals[-1] if vals else None

def regime(cpi: List[dict], ten_year: List[dict]) -> str:
    def yoy(series: List[dict], months=12) -> Optional[float]:
        vals = [float(x["value"]) for x in series if x.get("value") not in (None, ".", "")]
        if len(vals) < months+1: return None
        try:
            return (vals[-1] / vals[-1-months]) - 1
        except ZeroDivisionError:
            return None
    cpi_yoy = yoy(cpi); yield_level = _last(ten_year)
    if cpi_yoy is None or yield_level is None: return "neutral"
    if cpi_yoy > 0.03 and yield_level > 0.035: return "inflationary"
    if cpi_yoy < 0.0 and yield_level < 0.025: return "disinflationary"
    return "neutral"

def macro_snapshot(start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
    try:
        cpi = fred.get_series(SERIES["cpi"], start, end)
        dgs10 = fred.get_series(SERIES["ten_year"], start, end)
        dff = fred.get_series(SERIES["fed_funds"], start, end)
        hy = fred.get_series(SERIES["hy_oas"], start, end)
        return {
            "as_of": dgs10[-1]["date"] if dgs10 else None,
            "cpi_last": _last(cpi),
            "ten_year_last": _last(dgs10),
            "fed_funds_last": _last(dff),
            "hy_oas_last": _last(hy),
            "regime": regime(cpi, dgs10),
        }
    except Exception as e:
        return {
            "as_of": None,
            "cpi_last": None,
            "ten_year_last": None,
            "fed_funds_last": None,
            "hy_oas_last": None,
            "regime": "neutral",
            "error": str(e)
        }

