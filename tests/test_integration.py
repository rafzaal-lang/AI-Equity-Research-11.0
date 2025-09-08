import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.providers.enhanced_fmp_provider import enhanced_fmp_provider
from services.financial_modeler import enhanced_financial_modeler
from services.report.professional_report_generator import professional_report_generator
from services.backtesting.framework import BacktestingFramework, momentum_strategy
from services.monitoring.health_checks import health_checker

class TestIntegration(unittest.TestCase):
    """Comprehensive integration tests for the entire platform."""

    def test_full_workflow(self):
        """Test the full workflow from data fetching to report generation and backtesting."""
        # 1. Data Fetching
        income_statement = enhanced_fmp_provider.income_statement('AAPL')
        self.assertIsNotNone(income_statement)
        self.assertIn('data', income_statement)

        # 2. Financial Modeling
        dcf_valuation = enhanced_financial_modeler.run_dcf_analysis('AAPL')
        self.assertIsNotNone(dcf_valuation)
        self.assertGreater(dcf_valuation['fair_value'], 0)

        # 3. Report Generation
        report_data = {
            "company": {"name": "Apple Inc.", "symbol": "AAPL", "logo_url": ""},
            "executive_summary": "Test summary.",
            "investment_thesis": "Test thesis.",
            "key_risks": [],
            "financial_summary": {
                "periods": ["2022", "2023"],
                "profitability": {"gross_margin": [0.4, 0.42], "net_margin": [0.2, 0.22]},
                "liquidity": {"current_ratio": [1.5, 1.6], "quick_ratio": [1.0, 1.1]},
            },
            "dcf_valuation": {
                "fair_value_per_share": dcf_valuation['fair_value'],
                "assumptions": dcf_valuation['assumptions']
            },
            "comparable_analysis": {"peers": []},
            "detailed_risks": []
        }
        html_report = professional_report_generator.generate_html_report(report_data)
        self.assertIn("Equity Research Report", html_report)

        # 4. Backtesting
        backtester = BacktestingFramework()
        price_data = backtester.load_price_data(['AAPL', 'MSFT'], '2023-01-01', '2023-12-31')
        backtest_results = backtester.run_backtest(momentum_strategy, price_data, '2023-01-01', '2023-12-31')
        self.assertIsNotNone(backtest_results)
        self.assertIn('performance_metrics', backtest_results)

        # 5. Health Checks
        health_status = health_checker.check_system_resources()
        self.assertIn(health_status['status'].value, ['healthy', 'degraded', 'unhealthy', 'critical'])

if __name__ == '__main__':
    unittest.main()

