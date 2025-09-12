# src/services/peers/peer_classifier.py
# Production-Grade Peer Classification with SEC competitor mining.
from __future__ import annotations

import asyncio
import aiohttp
import logging
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

SEC_HEADERS = {"User-Agent": "pro-report/1.0 (admin@example.com)"}

class ProductionPeerClassifier:
    """
    Production peer classification using FMP as primary + SEC competitor mentions.
    """

    def __init__(self, fmp_api_key: str, polygon_api_key: str | None = None, alpha_vantage_key: str | None = None):
        if not fmp_api_key:
            raise ValueError("FMP API key is required")
        self.fmp_key = fmp_api_key
        self.industry_peer_cache: Dict[str, List[str]] = {}  # optional, can be warmed later

    async def classify_peers_production(self, target_symbol: str, max_peers: int = 6) -> List[str]:
        try:
            m = await self.build_company_profile_fmp(target_symbol)
            if not m:
                return self._sector_fallback(max_peers)

            candidates = await self._generate_candidates(m)
            scored = await self._score_candidates(m, candidates)
            peers = self._validate_and_trim(scored, max_peers)
            return peers
        except Exception as e:
            logger.error("Peer classification error for %s: %s", target_symbol, e)
            return self._emergency_fallback(max_peers)

    # ---------- Data collection ----------

    async def build_company_profile_fmp(self, symbol: str) -> Optional[CompanyMetrics]:
        symbol = symbol.upper().strip()
        timeout = aiohttp.ClientTimeout(total=18)

        async with aiohttp.ClientSession(timeout=timeout, headers=SEC_HEADERS) as s:
            p, is_row, km, sec_json, fmp_peers = await asyncio.gather(
                self._fmp_company_profile(s, symbol),
                self._fmp_is_latest(s, symbol),
                self._fmp_key_metrics_ttm(s, symbol),
                self._sec_company_summary_plus_competitors(s, symbol),  # <— now implemented
                self._fmp_direct_peers(s, symbol),
                return_exceptions=True
            )

        profile = p if isinstance(p, dict) else {}
        income  = is_row if isinstance(is_row, dict) else {}
        keym    = km if isinstance(km, dict) else {}
        secd    = sec_json if isinstance(sec_json, dict) else {}
        peers   = fmp_peers if isinstance(fmp_peers, list) else []

        comp_mentions = secd.get("competitor_tickers", []) + peers

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
            business_segments=[],  # could be extended later
            competitors_mentioned=[t for t in comp_mentions if isinstance(t, str)],
            geographic_segments=[],
            data_quality_score=self._quality_score(profile, income, keym),
        )

    async def _fmp_company_profile(self, s: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
        params = {"apikey": self.fmp_key}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_is_latest(self, s: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
        params = {"apikey": self.fmp_key, "limit": 1}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_key_metrics_ttm(self, s: aiohttp.ClientSession, symbol: str) -> Dict:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
        params = {"apikey": self.fmp_key}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                return js[0] if isinstance(js, list) and js else {}
        return {}

    async def _fmp_direct_peers(self, s: aiohttp.ClientSession, symbol: str) -> List[str]:
        url = "https://financialmodelingprep.com/api/v4/stock_peers"
        params = {"apikey": self.fmp_key, "symbol": symbol}
        async with s.get(url, params=params) as r:
            if r.status == 200:
                js = await r.json()
                if isinstance(js, list) and js:
                    return js[0].get("peersList", []) or []
        return []

    # ---------- SEC competitor mining (implemented) ----------

    async def _sec_company_summary_plus_competitors(self, s: aiohttp.ClientSession, symbol: str) -> Dict:
        """
        1) map symbol -> CIK (company_tickers.json)
        2) get company submissions to find latest 10-K (or 20-F)
        3) fetch the primary HTML filing
        4) extract 'Competition' section, mine org names, map to tickers via FMP search
        """
        try:
            # 1) Ticker map
            async with s.get("https://www.sec.gov/files/company_tickers.json") as r:
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

            # 2) Submissions
            sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            async with s.get(sub_url) as r2:
                if r2.status != 200:
                    return {}
                sub = await r2.json()

            # choose latest annual filing
            filings = sub.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            accns = filings.get("accessionNumber", [])
            prims = filings.get("primaryDocument", [])
            # scan for 10-K or 20-F
            idx = None
            for i, f in enumerate(forms):
                if f in ("10-K", "20-F"):
                    idx = i
                    break
            if idx is None:
                return {}

            accession = accns[idx].replace("-", "")
            primary = prims[idx]
            # 3) pull filing HTML
            # path: /Archives/edgar/data/{cik}/{accession}/{primary}
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary}"
            async with s.get(filing_url) as r3:
                if r3.status != 200:
                    return {}
                html = await r3.text()

            # 4) extract competitors
            comp_text = self._extract_competition_section_text(html)
            names = self._extract_company_names_from_text(comp_text)
            # map to symbols via FMP search (best-effort, limited)
            tickers = await self._map_company_names_to_tickers(s, names, limit=15)
            return {"competitor_tickers": tickers}
        except Exception as e:
            logger.debug("SEC competitor mining failed for %s: %s", symbol, e)
            return {}

    def _extract_competition_section_text(self, html: str) -> str:
        # strip HTML tags quickly
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)

        # find "Competition" heading vicinity (simple but effective)
        m = re.search(r"(Item\s+1\.\s*Business.*?)(Item\s+1A\.|Item\s+2\.)", text, flags=re.IGNORECASE)
        if m:
            segment = m.group(1)
        else:
            segment = text

        # narrow around the word "Competition"
        c = re.search(r"Competition(.*?)(Risk Factors|Employees|Intellectual Property|Regulation|Seasonality|Item\s+1A)", segment, re.IGNORECASE)
        if c:
            return c.group(1)[:8000]  # keep it bounded
        return segment[:8000]

    def _extract_company_names_from_text(self, text: str) -> List[str]:
        # crude NER: capture sequences of Capitalized words (2–5 tokens)
        candidates = re.findall(r"\b([A-Z][A-Za-z&\.\-]+(?:\s+[A-Z][A-Za-z&\.\-]+){0,4})\b", text)
        # prune obvious stopwords and self-refs
        STOP = {"United States", "U.S.", "United", "States", "Company", "Inc", "LLC", "Ltd", "Limited",
                "Group", "Corporation", "Holdings", "Holding", "Systems", "International", "Technologies",
                "and", "or", "including", "the", "we", "our", "its", "their", "Sales", "Products"}
        out: List[str] = []
        for name in candidates:
            name = name.strip()
            # require at least 1 space (2+ words) to reduce false positives
            if " " not in name:
                continue
            # short blacklist
            if any(tok in STOP for tok in name.split()):
                pass  # don't drop just for containing stop word
            # toss clearly generic phrases
            if len(name) < 5 or len(name) > 60:
                continue
            if name.lower().startswith(("item ", "part ", "note ")):
                continue
            out.append(name)
        # keep unique order
        seen: Set[str] = set()
        uniq = []
        for n in out:
            if n not in seen:
                uniq.append(n)
                seen.add(n)
        return uniq[:50]

    async def _map_company_names_to_tickers(self, s: aiohttp.ClientSession, names: List[str], limit: int = 15) -> List[str]:
        """Resolve company names to tickers using FMP search. Returns tickers only."""
        tickers: List[str] = []
        for nm in names[:limit]:
            try:
                url = "https://financialmodelingprep.com/api/v3/search"
                params = {"apikey": self.fmp_key, "query": nm, "limit": 1}
                async with s.get(url, params=params) as r:
                    if r.status != 200:
                        continue
                    js = await r.json()
                    if isinstance(js, list) and js:
                        sym = js[0].get("symbol")
                        if sym and isinstance(sym, str):
                            sym = sym.upper()
                            # basic sanity: 1–6 chars, letters/dash/period
                            if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", sym):
                                if sym not in tickers:
                                    tickers.append(sym)
            except Exception:
                continue
        return tickers

    # ---------- Candidate generation, scoring, validation ----------

    async def _generate_candidates(self, m: CompanyMetrics) -> Set[str]:
        cands: Set[str] = set()
        # SEC + FMP competitor mentions (now populated)
        cands.update([p.upper() for p in (m.competitors_mentioned or []) if isinstance(p, str)])

        # soft industry cache
        key = f"{m.sector}|{m.industry}"
        if key in self.industry_peer_cache:
            cands.update(self.industry_peer_cache[key])

        # market-cap tier
        cands.update(self._market_cap_tier(m.market_cap))

        # remove self; normalize
        cands.discard(m.symbol)
        cands = {x for x in cands if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", x or "")}
        return cands

    async def _score_candidates(self, target: CompanyMetrics, cands: Set[str]) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout, headers=SEC_HEADERS) as s:
            for c in cands:
                try:
                    cm = await self.build_company_profile_fmp(c)
                    if not cm:
                        continue
                    s_score = self._similarity_score(target, cm)
                    scored.append((c, s_score))
                except Exception:
                    continue
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _similarity_score(self, a: CompanyMetrics, b: CompanyMetrics) -> float:
        score = 0.0
        w = {"industry": 0.25, "market_cap": 0.20, "financials": 0.20, "segments": 0.15, "profitability": 0.10, "growth": 0.10}

        # industry / sector
        if a.industry and a.industry == b.industry:
            score += w["industry"]
        elif a.sector and a.sector == b.sector:
            score += w["industry"] * 0.6

        # size proximity
        if a.market_cap > 0 and b.market_cap > 0:
            score += w["market_cap"] * (min(a.market_cap, b.market_cap) / max(a.market_cap, b.market_cap))

        # ratios
        pairs = [(a.pe_ratio, b.pe_ratio), (a.ps_ratio, b.ps_ratio), (a.ev_ebitda, b.ev_ebitda)]
        acc, cnt = 0.0, 0
        for x, y in pairs:
            if x > 0 and y > 0:
                acc += min(x, y) / max(x, y)
                cnt += 1
        if cnt:
            score += w["financials"] * (acc / cnt)

        # segments overlap (placeholder lists right now)
        # if a.business_segments and b.business_segments:
        #     A,B = set(map(str.lower, a.business_segments)), set(map(str.lower, b.business_segments))
        #     denom = len(A|B) or 1
        #     score += w["segments"] * (len(A&B)/denom)

        # profitability
        if a.profit_margin > 0 and b.profit_margin > 0:
            score += w["profitability"] * (min(a.profit_margin, b.profit_margin) / max(a.profit_margin, b.profit_margin))

        # growth
        if a.revenue_growth and b.revenue_growth:
            diff = abs(a.revenue_growth - b.revenue_growth)
            score += w["growth"] * max(0.0, 1.0 - diff / 100.0)

        return float(score)

    def _validate_and_trim(self, scored: List[Tuple[str, float]], max_peers: int) -> List[str]:
        MIN_SCORE = 0.30
        filtered = [t for t, s in scored if s >= MIN_SCORE]
        return filtered[:max_peers] if filtered else [t for t, _ in scored[:max_peers]]

    # ---------- Utilities & fallbacks ----------

    def _market_cap_tier(self, mc: float) -> List[str]:
        if mc >= 1_000_000_000_000:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
        if mc >= 200_000_000_000:
            return ["TSLA", "AVGO", "V", "MA", "UNH", "JNJ", "XOM", "CVX"]
        if mc >= 50_000_000_000:
            return ["ORCL", "CRM", "ADBE", "NFLX", "INTC", "AMD", "QCOM", "JPM", "MRK"]
        if mc >= 10_000_000_000:
            return ["TXN", "MU", "AMAT", "NOW", "WDAY", "REGN", "VRTX", "GS", "MS"]
        if mc >= 2_000_000_000:
            return ["PLTR", "SNOW", "DDOG", "NET", "OKTA", "ZS", "CRWD", "VEEV"]
        return ["WDC", "NTAP", "HPE", "HPQ", "DELL"]

    def _quality_score(self, profile: Dict, income: Dict, keym: Dict) -> float:
        score = 0.0
        if profile.get("companyName"): score += 0.2
        if profile.get("sector") and profile.get("industry"): score += 0.2
        if float(income.get("revenue") or 0) > 0: score += 0.3
        if float(keym.get("peRatio") or 0) > 0: score += 0.3
        return score

    def _sector_fallback(self, max_peers: int) -> List[str]:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"][:max_peers]

    def _emergency_fallback(self, max_peers: int) -> List[str]:
        return ["SPY", "QQQ", "VTI", "IWM", "DIA", "XLK"][:max_peers]

# ---- Sync façade ----

def peer_universe(symbol: str, max_peers: int, fmp_api_key: str) -> List[str]:
    classifier = ProductionPeerClassifier(fmp_api_key)

    async def _run():
        return await classifier.classify_peers_production(symbol, max_peers)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=25)
        return loop.run_until_complete(_run())
    except Exception:
        return asyncio.run(_run())
