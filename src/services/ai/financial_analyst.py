# src/services/ai/financial_analyst.py
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
from openai import OpenAI

from src.services.llm.json_client import chat_json
from src.services.data.fundamentals_agg import fetch_fundamentals, robust_peer_pe

logger = logging.getLogger(__name__)

@dataclass
class FinancialInsight:
    category: str
    insight: str
    supporting_data: Dict[str, Any] | str
    confidence: float

@dataclass
class CompanyAnalysis:
    symbol: str
    executive_summary: str
    key_insights: List[FinancialInsight]
    investment_thesis: Dict[str, str]
    analyst_rating: str
    price_target_range: Dict[str, float]

def rate_limit_openai(calls_per_minute=15):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait = min_interval - elapsed
            if wait > 0:
                logger.info("Rate limiting: waiting %.2f seconds", wait)
                time.sleep(wait)
            out = func(*args, **kwargs)
            last_called[0] = time.time()
            return out
        return wrapper
    return decorator

class AIFinancialAnalyst:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - AI analysis disabled")

    @rate_limit_openai(12)
    def _chat_text(self, messages, max_tokens=1000, temperature=0.3) -> Optional[str]:
        if not self.client:
            return None
        try:
            r = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("OpenAI text call failed: %s: %s", type(e).__name__, e)
            return None

    @rate_limit_openai(12)
    def _chat_json(self, messages, max_tokens=1200, temperature=0.2) -> Dict[str, Any]:
        try:
            return chat_json(messages, model="gpt-4o-mini", max_tokens=max_tokens, temperature=temperature, enforce_json_mode=True)
        except Exception as e:
            logger.error("OpenAI JSON call failed: %s: %s", type(e).__name__, e)
            return {}

    def analyze_company(self, symbol: str, financial_data: Dict[str, Any], filing_excerpts: List[str] = None) -> CompanyAnalysis:
        if not self.client:
            return self._fallback(symbol, financial_data)

        try:
            fundamentals = (financial_data.get("fundamentals") or {}).copy()
            dcf = financial_data.get("dcf_valuation") or financial_data.get("dcf") or {}
            comps = (financial_data.get("comps") or {}).copy()

            # Enrich fundamentals if thin
            core = ["revenue", "net_income", "net_margin"]
            if not any(isinstance(fundamentals.get(k), (int, float)) for k in core):
                logger.info("Enriching fundamentals from FMP for %s", symbol)
                fundamentals.update({k: v for k, v in fetch_fundamentals(symbol).items() if v is not None})

            # Robust peer P/E
            if isinstance(comps.get("peers"), list):
                rpe = robust_peer_pe(comps["peers"])
                if rpe is not None:
                    comps["peer_pe_robust"] = rpe
                    fundamentals.setdefault("pe", rpe)

            financial_data["fundamentals"] = fundamentals
            financial_data["comps"] = comps

            insights = self._insights(symbol, fundamentals, dcf, comps)
            summary = self._exec_summary(symbol, insights, fundamentals)
            thesis = self._thesis(symbol, insights, financial_data)
            rating, targets = self._rating_targets(symbol, dcf, insights)

            return CompanyAnalysis(
                symbol=symbol,
                executive_summary=summary,
                key_insights=insights,
                investment_thesis=thesis,
                analyst_rating=rating,
                price_target_range=targets,
            )
        except Exception as e:
            logger.error("AI analysis failed for %s: %s", symbol, e)
            return self._fallback(symbol, financial_data)

    def _fallback(self, symbol: str, financial_data: Dict[str, Any]) -> CompanyAnalysis:
        f = financial_data.get("fundamentals", {}) or {}
        dcf = financial_data.get("dcf_valuation", {}) or {}
        insights: List[FinancialInsight] = []

        rev = f.get("revenue")
        if isinstance(rev, (int, float)) and rev > 1e9:
            insights.append(FinancialInsight("strength", f"Large-scale operations with revenue of ${rev/1e9:.1f}B", {"revenue": rev}, 0.9))

        nm = f.get("net_margin")
        rating = "Hold"
        if isinstance(nm, (int, float)):
            if nm > 0.15:
                insights.append(FinancialInsight("strength", f"Strong profitability with {nm:.1%} net margin", {"net_margin": nm}, 0.8))
                rating = "Buy"
            elif nm > 0:
                rating = "Hold"

        dcf_value = dcf.get("fair_value_per_share")
        if isinstance(dcf_value, (int, float)):
            targets = {"low": float(dcf_value) * 0.85, "base": float(dcf_value), "high": float(dcf_value) * 1.15}
        else:
            targets = {"low": 0.0, "base": 0.0, "high": 0.0}

        return CompanyAnalysis(
            symbol=symbol,
            executive_summary=f"{symbol} analysis completed. Review fundamentals and valuation below.",
            key_insights=insights,
            investment_thesis={"bull_case": "Upside from scale and execution.", "bear_case": "Macro and competitive risks."},
            analyst_rating=rating,
            price_target_range=targets,
        )

    def _insights(self, symbol: str, fundamentals: Dict, dcf: Dict, comps: Dict) -> List[FinancialInsight]:
        fs = self._summary(fundamentals, dcf, comps)
        schema = (
            "{ \"insights\": [ { \"category\": \"strength|weakness|opportunity|threat\", "
            "\"insight\": \"text\", \"supporting_data\": {\"metric\": \"value\"}, \"confidence\": 0.0 } ] }"
        )
        messages = [
            {"role": "system", "content": "You are a precise equity research co-pilot. Always return strict JSON."},
            {"role": "user", "content": f"Analyze {symbol} and produce 4-6 insights.\nFinancial Data:\n{fs}\nReturn JSON like:\n{schema}"}
        ]
        result = self._chat_json(messages, max_tokens=1200, temperature=0.2) or {}
        items = result.get("insights")
        if not isinstance(items, list):
            return []
        out: List[FinancialInsight] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                out.append(FinancialInsight(
                    category=str(it.get("category", "neutral")),
                    insight=str(it.get("insight", "")).strip(),
                    supporting_data=it.get("supporting_data", {}),
                    confidence=float(it.get("confidence", 0.7)),
                ))
            except Exception:
                continue
        return out

    def _exec_summary(self, symbol: str, insights: List[FinancialInsight], f: Dict) -> str:
        if not insights:
            return f"{symbol} analysis completed. Review detailed financial metrics below."
        insight_text = "\n".join(f"- {i.insight}" for i in insights if i.insight)

        def pct(x): return f"{x:.1%}" if isinstance(x, (int, float)) else "N/A"
        def money(x): 
            if isinstance(x, (int, float)):
                return f\"${x:,.0f}\"
            return \"N/A\"

        prompt = (
            f"Write a 3-4 sentence executive summary for {symbol} using these insights:\n\n"
            f"{insight_text}\n\n"
            f"Key metrics: Revenue {money(f.get('revenue'))}, Net Margin {pct(f.get('net_margin'))}, ROE {pct(f.get('roe'))}."
            f" Tone: professional, concise, investment-focused. Lead with a clear investment view."
        )
        r = self._chat_text([{"role": "user", "content": prompt}], max_tokens=220)
        return r or f"{symbol} analysis completed. See detailed insights below."

    def _thesis(self, symbol: str, insights: List[FinancialInsight], data: Dict) -> Dict[str, str]:
        strengths = [i.insight for i in insights if i.category == "strength"]
        weaknesses = [i.insight for i in insights if i.category == "weakness"]
        opps = [i.insight for i in insights if i.category == "opportunity"]
        threats = [i.insight for i in insights if i.category == "threat"]

        dcf_ev = data.get("dcf_valuation", {}).get("enterprise_value", 0)
        if not isinstance(dcf_ev, (int, float)):
            dcf_ev = 0

        bull = (
            f"Create a bull case for {symbol} based on Strengths: {'; '.join(strengths)}; "
            f"Opportunities: {'; '.join(opps)}; DCF EV: ${dcf_ev:,.0f}. 2-3 sentences."
        )
        bear = (
            f"Create a bear case for {symbol} based on Weaknesses: {'; '.join(weaknesses)}; "
            f"Threats: {'; '.join(threats)}. 2-3 sentences."
        )
        bull_r = self._chat_text([{"role": "user", "content": bull}], max_tokens=150)
        bear_r = self._chat_text([{"role": "user", "content": bear}], max_tokens=150)
        return {"bull_case": bull_r or "Upside potential based on strengths.", "bear_case": bear_r or "Downside risks from challenges."}

    def _rating_targets(self, symbol: str, dcf: Dict, insights: List[FinancialInsight]) -> Tuple[str, Dict[str, float]]:
        pos = len([i for i in insights if i.category in ("strength", "opportunity")])
        neg = len([i for i in insights if i.category in ("weakness", "threat")])
        rating = "Buy" if (pos - neg) >= 2 else ("Hold" if (pos - neg) >= 0 else "Sell")

        base = dcf.get("enterprise_value", 1e8)
        if not isinstance(base, (int, float)):
            base = 1e8
        targets = {"low": float(base) * 0.85, "base": float(base), "high": float(base) * 1.15}
        return rating, targets

    def _summary(self, f: Dict, dcf: Dict, comps: Dict) -> str:
        def fmt(x, pct=False):
            if not isinstance(x, (int, float)): return "N/A"
            return f"{x:.1%}" if pct else (f"${x:,.0f}" if x >= 1000 else f"{x:.2f}")
        peer_pe = comps.get("peer_pe_robust")
        peer_pe_str = f"{peer_pe:.2f}" if isinstance(peer_pe, (int, float)) else "N/A"
        return f"""
Revenue: {fmt(f.get('revenue'))}
Net Income: {fmt(f.get('net_income'))}
Free Cash Flow: {fmt(f.get('fcf'))}

Margins:
- Gross: {fmt(f.get('gross_margin'), True)}
- Operating: {fmt(f.get('op_margin'), True)}
- Net: {fmt(f.get('net_margin'), True)}

Returns:
- ROE: {fmt(f.get('roe'), True)}
- ROIC: {fmt(f.get('roic'), True)}

Valuation:
- Enterprise Value: {fmt((dcf or {}).get('enterprise_value'))}
- Peer Average P/E (robust): {peer_pe_str}
""".strip()

# global instance
ai_financial_analyst = AIFinancialAnalyst()
