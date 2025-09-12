# src/services/peers/peer_classifier.py
# Production-Grade Peer Classification System
# Uses FMP + (optional) SEC metadata. No API shape changes to your app.

from __future__ import annotations

import asyncio
import aiohttp
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- Data model ----

@dataclass
class CompanyMetrics:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    revenue: float
    employees: int
    pe_ratio: float
    ps_ratio: float
    ev_ebitda: float
    roe: float
    debt_equity: float
    profit_margin: float
    revenue_growth: float
    business_segments: List[str]
    competitors_mentioned: List[str]
    geographic_segments: List[str]
    data_quality_score: float

# ---- Classifier ----

class ProductionPeerClassifier:
    """
    Production peer classification using reliable sources (FMP; SEC optional).
    No external DB required; fully stateless and cacheable by your existing HTTP cache.
    """

    def __init__(self, fmp_api_key: str, polygon_api_key: str | None = None, alpha_vantage_key: str | None = None):
        if not fmp_api_key:
            raise ValueError("FMP API key is required")
        self.fmp_key = fmp_api_key
        self.polygon_key = polygon_api_key  # present for future extension
        self.alpha_vantage_key = alpha_vantage_key
        self.industry_peer_cache: Dict[str, List[str]] = {}  # optional in-memory cache

    # --------- Public entrypoint ---------

    async def classify_peers_production(self, target_symbol: str, max_peers: int = 6) -> List[str]:
        try:
            company_metrics = await self.build_company_profile_fmp(target_symbol)
            if not company_metrics:
                return self._sector_fallback(target_symbol, max_peers)

            candidates = await self._generate_candidates(company_metrics)
            scored = await self._score_candidates(company_metrics, candidates)
            peers = self._validate_and_trim(scored, max_peers)
            return peers
        except Exception as e:
            logger.error("Peer classification error for %s: %s", target_symbol, e)
            return self._emergency_fallback(max_peers)

    # --------- Data collection ---------

    async def build_company_profile_fmp(self, symbol: str) -> Optional[CompanyMetrics]:
        """
        Assemble a comprehensive profile from FMP (+SEC optional).
        Returns None if we can't get basic profile + a couple of metrics.
        """
        symbol = symbol.upper().strip()
        timeout = aiohttp.ClientTimeout(total=15)
        headers = {"User-Agent": "pro-report/1.0 (contact: admin@example.com)"}  # for sec.gov politeness

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [
                self._fmp_company_profile(session, symbol),
                self._fmp_is_latest(session, symbol),
                self._fmp_key_metrics_ttm(session, symbol),
                self._sec_business_summary(session, symbol),     # optional; will fail soft
                self._fmp_direct_peers(session, symbol),         # used as seed
            ]
            p, is_row, km, sec_info, fmp_peers = await asyncio.gather(*tasks, return_exceptions=True)

        profile = p if isinstance(p, dict) else {}
        income = is_row if isinstance(is_row, dict) else {}
        keym   = km if isinstance(km, dict) else {}
        secd   = sec_info if isinstance(sec_info, dict) else {}
        peers  = fmp_peers if isinstance(fmp_peers, list) else []

        # Must have the basics
        if not profile.get("companyName") or not profile.get("sector"):
            return None

        return CompanyMetrics(
            symbol=symbol,
            name=profile.get("companyName", ""),
            sector=profile.get("sector", ""),
            industry=profile.get("industry", ""),
            market_cap=float(profile.get("mktCap") or profile.get("marketCap") or 0) or 0.0,
            revenue=float(income.get("revenue") or income.get("totalRevenue") or 0) or 0.0,
            employees=int(profile.get("fullTimeEmployees") or 0),
            pe_ratio=float(keym.get("peRatio") or 0) or 0.0,
            ps_ratio=float(keym.get("priceToSalesRatio") or 0) or 0.0,
            ev_ebitda=float(keym.get("enterpriseValueMultiple") or 0) or 0.0,
            roe=float(keym.get("roe") or 0) or 0.0,
            debt_equity=float(keym.get("debtToEquity") or 0) or 0.0,
            profit_margin=float(keym.get("netProfitMargin") or 0) or 0.0,
            revenue_growth=float(keym.get("revenueGrowth") or 0) or 0.0,
            business_segments=self._extract_business_segments(secd),
            competitors_mentioned=(self._extract_competitors(secd) + peers),
            geographic_segments=self._extract_geo_segments(secd),
            data_quality_score=self._quality_score(profile, income, keym),
        )

    async def _fmp_company_profile(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        params = {"apikey": self.fmp_key}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_is_latest(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
        params = {"apikey": self.fmp_key, "limit": 1}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_key_metrics_ttm(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
        params = {"apikey": self.fmp_key}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_direct_peers(self, session: aiohttp.ClientSession, symbol: str) -> List[str]:
        url = "https://financialmodelingprep.com/api/v4/stock_peers"
        params = {"apikey": self.fmp_key, "symbol": symbol}
        async with session.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                if isinstance(js, list) and js:
                    return js[0].get("peersList", []) or []
        return []

    async def _sec_business_summary(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """
        Best-effort pull of company metadata from SEC. We keep it optional
        and soft-fail if anything goes wrong (no hard dependency).
        """
        try:
            # get CIK mapping
            async with session.get("https://www.sec.gov/files/company_tickers.json") as r:
                if r.status != 200:
                    return {}
                mapping = await r.json()
            cik = None
            for v in mapping.values():
                if str(v.get("ticker", "")).upper() == symbol.upper():
                    cik = str(v.get("cik_str")).zfill(10)
                    break
            if not cik:
                return {}

            # fetch company submissions (recent filings)
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            async with session.get(url) as r2:
                if r2.status != 200:
                    return {}
                return await r2.json()
        except Exception as e:
            logger.debug("SEC optional fetch failed for %s: %s", symbol, e)
            return {}

    # --------- Candidate generation & scoring ---------

    async def _generate_candidates(self, m: CompanyMetrics) -> Set[str]:
        candidates: Set[str] = set()

        # 1) FMP direct peers and SEC-identified competitors
        candidates.update([p.upper() for p in (m.competitors_mentioned or []) if isinstance(p, str)])

        # 2) Industry peers (simple cache; expand as desired)
        key = f"{m.sector}|{m.industry}"
        if key in self.industry_peer_cache:
            candidates.update(self.industry_peer_cache[key])

        # 3) Market-cap tier peers (coarse but helpful)
        candidates.update(self._market_cap_tier(m.market_cap, m.sector))

        # Remove self
        candidates.discard(m.symbol)
        # Basic sanity: US tickers only (letters, dots, dashes)
        candidates = {c for c in candidates if c and isinstance(c, str) and len(c) <= 6}
        return candidates

    async def _score_candidates(self, target: CompanyMetrics, candidates: Set[str]) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {"User-Agent": "pro-report/1.0"}

        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            for c in candidates:
                try:
                    cm = await self.build_company_profile_fmp(c)
                    if not cm:
                        continue
                    s = self._similarity_score(target, cm)
                    scored.append((c, s))
                except Exception:
                    continue
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # --------- Similarity logic ---------

    def _similarity_score(self, a: CompanyMetrics, b: CompanyMetrics) -> float:
        """
        Weighted similarity: industry/sector, size, ratios, margin, growth, segments.
        """
        score = 0.0
        weights = {
            "industry": 0.25,
            "market_cap": 0.20,
            "financials": 0.20,
            "segments": 0.15,
            "profitability": 0.10,
            "growth": 0.10,
        }

        # industry / sector
        if a.industry and a.industry == b.industry:
            score += weights["industry"]
        elif a.sector and a.sector == b.sector:
            score += weights["industry"] * 0.6

        # market cap (ratio bounded [0..1])
        if a.market_cap > 0 and b.market_cap > 0:
            score += weights["market_cap"] * (min(a.market_cap, b.market_cap) / max(a.market_cap, b.market_cap))

        # selected ratios (pe, ps, ev/ebitda)
        ratio_pairs = [
            (a.pe_ratio, b.pe_ratio),
            (a.ps_ratio, b.ps_ratio),
            (a.ev_ebitda, b.ev_ebitda),
        ]
        r_sum, r_cnt = 0.0, 0
        for x, y in ratio_pairs:
            if x > 0 and y > 0:
                r_sum += (min(x, y) / max(x, y))
                r_cnt += 1
        if r_cnt:
            score += weights["financials"] * (r_sum / r_cnt)

        # segments overlap (when we have them)
        if a.business_segments and b.business_segments:
            A = set([s.lower() for s in a.business_segments])
            B = set([s.lower() for s in b.business_segments])
            denom = len(A | B)
            if denom:
                score += weights["segments"] * (len(A & B) / denom)

        # profitability
        if a.profit_margin > 0 and b.profit_margin > 0:
            score += weights["profitability"] * (min(a.profit_margin, b.profit_margin) / max(a.profit_margin, b.profit_margin))

        # growth
        if a.revenue_growth and b.revenue_growth:
            # both are percentages; closeness → higher score
            diff = abs(a.revenue_growth - b.revenue_growth)
            score += weights["growth"] * max(0.0, 1.0 - diff / 100.0)

        return float(score)

    def _validate_and_trim(self, scored: List[Tuple[str, float]], max_peers: int) -> List[str]:
        MIN_SCORE = 0.30
        filtered = [t for t, s in scored if s >= MIN_SCORE]
        return filtered[:max_peers] if filtered else [t for t, _ in scored[:max_peers]]

    # --------- Utilities & fallbacks ---------

    def _market_cap_tier(self, mc: float, sector: str) -> List[str]:
        # coarse but practical; adjust as needed
        if mc >= 1_000_000_000_000:   # $1T+
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        if mc >= 200_000_000_000:
            return ["META", "AVGO", "TSLA", "V", "MA", "UNH", "JNJ", "XOM", "CVX"]
        if mc >= 50_000_000_000:
            return ["ORCL", "CRM", "ADBE", "NFLX", "INTC", "AMD", "QCOM", "TMO", "MRK", "JPM"]
        if mc >= 10_000_000_000:
            return ["TXN", "MU", "AMAT", "NOW", "WDAY", "REGN", "VRTX", "GS", "MS"]
        if mc >= 2_000_000_000:
            return ["PLTR", "SNOW", "DDOG", "NET", "OKTA", "ZS", "CRWD", "VEEV"]
        return ["WDC", "NTAP", "HPE", "HPQ", "DELL"]

    def _extract_business_segments(self, sec_json: Dict) -> List[str]:
        # TODO: parse Item 1 / Segment info when available
        return []

    def _extract_competitors(self, sec_json: Dict) -> List[str]:
        # TODO: simple NER over business section (kept empty to avoid false positives)
        return []

    def _extract_geo_segments(self, sec_json: Dict) -> List[str]:
        return []

    def _quality_score(self, profile: Dict, income: Dict, keym: Dict) -> float:
        score = 0.0
        if profile.get("companyName"): score += 0.2
        if profile.get("sector") and profile.get("industry"): score += 0.2
        if float(income.get("revenue") or 0) > 0: score += 0.3
        if float(keym.get("peRatio") or 0) > 0: score += 0.3
        return score

    def _sector_fallback(self, symbol: str, max_peers: int) -> List[str]:
        # last-resort, safe defaults
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"][:max_peers]

    def _emergency_fallback(self, max_peers: int) -> List[str]:
        return ["SPY", "QQQ", "VTI", "IWM", "DIA", "XLK"][:max_peers]

# ---- Sync wrapper used by the report generator ----

def peer_universe(symbol: str, max_peers: int, fmp_api_key: str) -> List[str]:
    """
    Synchronous façade (safe in existing threads/WSGI). Creates its own loop if needed.
    """
    classifier = ProductionPeerClassifier(fmp_api_key)

    async def _run():
        return await classifier.classify_peers_production(symbol, max_peers)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In case we're already inside an event loop (e.g., uvicorn), use a task.
            return asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=20)
        return loop.run_until_complete(_run())
    except Exception:
        # Fresh loop fallback
        return asyncio.run(_run())
