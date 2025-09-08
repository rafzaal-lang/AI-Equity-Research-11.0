from __future__ import annotations
from typing import Dict, Any, List, Optional

def _to_series(hist: List[dict], key="close") -> List[float]:
    ordered = sorted(hist, key=lambda x: x.get("date"))
    return [float(x.get(key)) for x in ordered if x.get(key) is not None]

def momentum(hist: List[dict], lookbacks=(20, 60, 120, 250)) -> Dict[str, Optional[float]]:
    closes = _to_series(hist)
    out = {}
    if not closes: return {f"mom_{lb}": None for lb in lookbacks}
    last = closes[-1]
    for lb in lookbacks:
        if len(closes) > lb:
            past = closes[-lb-1]
            out[f"mom_{lb}"] = (last - past) / past if past else None
        else:
            out[f"mom_{lb}"] = None
    return out

def rsi(hist: List[dict], period: int = 14) -> Optional[float]:
    closes = _to_series(hist)
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0.0))
        losses.append(abs(min(diff, 0.0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sma_cross(hist: List[dict], short: int = 20, long: int = 50) -> Dict[str, Optional[float]]:
    closes = _to_series(hist)
    if len(closes) < max(short, long): return {"sma_short": None, "sma_long": None, "cross": None}
    sma_s = sum(closes[-short:]) / short
    sma_l = sum(closes[-long:]) / long
    cross = 1 if sma_s > sma_l else (-1 if sma_s < sma_l else 0)
    return {"sma_short": sma_s, "sma_long": sma_l, "cross": cross}

def atr(hist: List[dict], period: int = 14) -> Optional[float]:
    ordered = sorted(hist, key=lambda x: x.get("date"))
    if len(ordered) < period + 1: return None
    trs = []
    prev_close = ordered[0]["close"]
    for o in ordered[1:]:
        high, low, close = o.get("high"), o.get("low"), o.get("close")
        if None in (high, low, close): return None
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    return sum(trs[-period:]) / period if len(trs) >= period else None

