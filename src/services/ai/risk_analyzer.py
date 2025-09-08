import os
import re
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from functools import wraps
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class RiskFactor:
    category: str  # "operational", "financial", "regulatory", "competitive", "market"
    description: str
    severity: str  # "high", "medium", "low"
    likelihood: str  # "high", "medium", "low"
    potential_impact: str
    mitigation_factors: List[str]

def rate_limit_openai_risk(calls_per_minute=10):
    """Decorator to rate limit OpenAI API calls for risk analysis"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                logger.info(f"Risk analysis rate limiting: waiting {left_to_wait:.2f} seconds")
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

class AIRiskAnalyzer:
    """AI-powered analysis of business risks from SEC filings and financial data."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - risk analysis will be disabled")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
    
    @rate_limit_openai_risk(calls_per_minute=8)  # Conservative rate limiting for risk analysis
    def _make_openai_request(self, messages, max_tokens=1500, temperature=0.2):
        """Centralized OpenAI request with rate limiting and error handling"""
        if not self.client:
            logger.warning("OpenAI client not available - returning fallback response")
            return None
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Updated to use available model
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error in risk analysis: {type(e).__name__}: {e}")
            return None
    
    def analyze_filing_risks(self, filing_text: str, symbol: str) -> List[RiskFactor]:
        """Extract and analyze risk factors from SEC filings using AI."""
        
        if not self.client:
            logger.info(f"AI risk analysis disabled for {symbol} - returning basic risks")
            return self._create_fallback_risks(symbol)
        
        # Extract risk factors section
        risk_section = self._extract_risk_section(filing_text)
        
        if not risk_section:
            logger.warning(f"No risk section found in filing for {symbol}")
            return self._create_fallback_risks(symbol)
        
        # Use AI to analyze and categorize risks
        prompt = f"""
        Analyze the following risk factors from {symbol}'s SEC filing and identify the 5 most significant risks.
        
        Risk Factors Text:
        {risk_section[:4000]}  # Limit to avoid token limits
        
        For each risk, provide:
        1. Category (operational/financial/regulatory/competitive/market)
        2. Clear description (1-2 sentences)
        3. Severity assessment (high/medium/low)
        4. Likelihood assessment (high/medium/low)
        5. Potential business impact
        6. Any mitigation factors mentioned
        
        Return as JSON:
        {{
            "risks": [
                {{
                    "category": "operational",
                    "description": "Clear risk description",
                    "severity": "high",
                    "likelihood": "medium", 
                    "potential_impact": "Impact on business",
                    "mitigation_factors": ["factor1", "factor2"]
                }}
            ]
        }}
        """
        
        response = self._make_openai_request([{"role": "user", "content": prompt}], max_tokens=1500)
        
        if not response:
            return self._create_fallback_risks(symbol)
        
        try:
            import json
            result = json.loads(response)
            
            risks = []
            for risk_data in result.get("risks", []):
                risks.append(RiskFactor(
                    category=risk_data.get("category", "general"),
                    description=risk_data.get("description", ""),
                    severity=risk_data.get("severity", "medium"),
                    likelihood=risk_data.get("likelihood", "medium"),
                    potential_impact=risk_data.get("potential_impact", ""),
                    mitigation_factors=risk_data.get("mitigation_factors", [])
                ))
            
            return risks
            
        except Exception as e:
            logger.error(f"Error analyzing risks for {symbol}: {e}")
            return self._create_fallback_risks(symbol)
    
    def _create_fallback_risks(self, symbol: str) -> List[RiskFactor]:
        """Create basic risk factors when AI is not available"""
        return [
            RiskFactor(
                category="market",
                description="Market volatility and economic uncertainty may impact stock performance",
                severity="medium",
                likelihood="high",
                potential_impact="Stock price fluctuations and reduced investor confidence",
                mitigation_factors=["Diversified business model", "Strong financial position"]
            ),
            RiskFactor(
                category="competitive",
                description="Intense competition in the industry may pressure margins and market share",
                severity="medium",
                likelihood="medium",
                potential_impact="Reduced profitability and competitive positioning",
                mitigation_factors=["Brand strength", "Innovation capabilities"]
            ),
            RiskFactor(
                category="regulatory",
                description="Changes in regulations and compliance requirements may increase costs",
                severity="medium",
                likelihood="medium",
                potential_impact="Higher compliance costs and operational restrictions",
                mitigation_factors=["Established compliance framework", "Government relations"]
            )
        ]
    
    def _extract_risk_section(self, filing_text: str) -> str:
        """Extract risk factors section from SEC filing."""
        
        # Common patterns for risk factor sections
        risk_patterns = [
            r"item\s+1a\.?\s*risk\s+factors(.*?)item\s+1b",
            r"risk\s+factors(.*?)(?:item\s+\d|unresolved|legal)",
            r"principal\s+risks(.*?)(?:item\s+\d|forward.looking)"
        ]
        
        filing_lower = filing_text.lower()
        
        for pattern in risk_patterns:
            match = re.search(pattern, filing_lower, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)[:10000]  # Return first 10k chars
        
        # Fallback: look for any mention of risks
        risk_keywords = ["risk", "uncertainty", "challenge", "threat"]
        for keyword in risk_keywords:
            if keyword in filing_lower:
                start_idx = filing_lower.find(keyword)
                return filing_text[start_idx:start_idx + 5000]
        
        return ""
    
    def analyze_financial_risks(self, financial_data: Dict[str, Any]) -> List[RiskFactor]:
        """Identify financial risks from quantitative data."""
        
        risks = []
        fundamentals = financial_data.get("fundamentals", {})
        
        # High debt risk
        de_ratio = fundamentals.get("de_ratio")
        if de_ratio and de_ratio > 2.0:
            risks.append(RiskFactor(
                category="financial",
                description=f"High debt-to-equity ratio of {de_ratio:.1f}x indicates elevated financial leverage",
                severity="high" if de_ratio > 3.0 else "medium",
                likelihood="high",
                potential_impact="Increased interest expense and refinancing risk during market stress",
                mitigation_factors=["Strong cash generation", "Diversified funding sources"]
            ))
        
        # Low profitability risk
        net_margin = fundamentals.get("net_margin")
        if net_margin is not None:
            if net_margin < 0:
                risks.append(RiskFactor(
                    category="operational",
                    description=f"Negative net margin of {net_margin:.1%} indicates operating losses",
                    severity="high",
                    likelihood="high",
                    potential_impact="Continued losses may threaten business sustainability",
                    mitigation_factors=["Cost reduction initiatives", "Revenue growth strategies"]
                ))
            elif net_margin < 0.05:
                risks.append(RiskFactor(
                    category="operational",
                    description=f"Low net margin of {net_margin:.1%} suggests operational inefficiency",
                    severity="medium",
                    likelihood="high",
                    potential_impact="Vulnerability to cost inflation and competitive pressure",
                    mitigation_factors=["Cost reduction initiatives", "Pricing power"]
                ))
        
        # Cash flow risk
        fcf = fundamentals.get("fcf")
        if fcf is not None and fcf < 0:
            risks.append(RiskFactor(
                category="financial",
                description="Negative free cash flow indicates cash consumption",
                severity="high",
                likelihood="high", 
                potential_impact="Potential need for external financing or asset sales",
                mitigation_factors=["Access to credit facilities", "Asset monetization options"]
            ))
        
        # Liquidity risk based on debt levels
        if de_ratio and de_ratio > 1.5:
            risks.append(RiskFactor(
                category="financial",
                description="High leverage may limit financial flexibility",
                severity="medium",
                likelihood="medium",
                potential_impact="Reduced ability to invest in growth or weather downturns",
                mitigation_factors=["Debt refinancing options", "Asset sales potential"]
            ))
        
        return risks

# Global instance
ai_risk_analyzer = AIRiskAnalyzer()
