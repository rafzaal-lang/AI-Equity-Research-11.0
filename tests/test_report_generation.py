import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.report.professional_report_generator import ProfessionalReportGenerator

class TestProfessionalReportGenerator(unittest.TestCase):
    """Test professional report generation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.generator = ProfessionalReportGenerator()
        
        # Sample report data
        self.sample_report_data = {
            "company": {
                "name": "Apple Inc.",
                "symbol": "AAPL",
                "logo_url": "https://example.com/aapl_logo.png"
            },
            "executive_summary": "This is a test summary.",
            "investment_thesis": "This is a test thesis.",
            "key_risks": [
                {"title": "Competition", "description": "Intense competition in the tech sector."},
                {"title": "Supply Chain", "description": "Reliance on global supply chains."}
            ],
            "financial_summary": {
                "periods": ["2022", "2023"],
                "profitability": {
                    "gross_margin": [0.40, 0.42],
                    "net_margin": [0.20, 0.22]
                },
                "liquidity": {
                    "current_ratio": [1.5, 1.6],
                    "quick_ratio": [1.0, 1.1]
                }
            },
            "dcf_valuation": {
                "fair_value_per_share": 150.00,
                "assumptions": {
                    "wacc": 0.08,
                    "g_terminal": 0.025,
                    "years_stage1": 5,
                    "g1": 0.10,
                    "years_stage2": 5,
                    "g2": 0.05
                }
            },
            "comparable_analysis": {
                "peers": [
                    {"symbol": "MSFT", "pe": 30.0, "ps": 10.0, "ev_ebitda": 20.0},
                    {"symbol": "GOOGL", "pe": 25.0, "ps": 8.0, "ev_ebitda": 18.0}
                ]
            },
            "detailed_risks": [
                {
                    "title": "Competition",
                    "description": "Apple faces intense competition from other major tech companies.",
                    "mitigation": "Strong brand loyalty and ecosystem."
                }
            ]
        }
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        html = self.generator.generate_html_report(self.sample_report_data)
        
        # Check for key sections
        self.assertIn("<h1>Apple Inc. (AAPL)</h1>", html)
        self.assertIn("<h2>1. Executive Summary</h2>", html)
        self.assertIn("<h2>2. Financial Analysis</h2>", html)
        self.assertIn("<h2>3. Valuation</h2>", html)
        self.assertIn("<h2>4. Risk Analysis</h2>", html)
        self.assertIn("This is not investment advice.", html)
    
    @patch("matplotlib.pyplot.savefig")
    def test_create_charts(self, mock_savefig):
        """Test chart creation."""
        charts = self.generator._create_charts(self.sample_report_data["financial_summary"])
        
        # Check that charts were created
        self.assertIn("profitability_chart", charts)
        self.assertIn("liquidity_chart", charts)
        
        # Check that chart files exist
        self.assertTrue(charts["profitability_chart"].exists())
        self.assertTrue(charts["liquidity_chart"].exists())
        
        # Clean up chart files
        for chart_path in charts.values():
            chart_path.unlink()
    
    @patch("matplotlib.pyplot.savefig")
    def test_generate_report_with_charts(self, mock_savefig):
        """Test generating report with charts."""
        html, charts = self.generator.generate_report_with_charts(self.sample_report_data)
        
        # Check HTML content
        self.assertIn("<h1>Apple Inc. (AAPL)</h1>", html)
        
        # Check charts
        self.assertIn("profitability_chart", charts)
        self.assertIn("liquidity_chart", charts)
        
        # Clean up chart files
        for chart_path in charts.values():
            chart_path.unlink()
    
    @patch("matplotlib.pyplot.savefig")
    def test_generate_full_report_html(self, mock_savefig):
        """Test generating a full HTML report."""
        report_path = self.generator.generate_full_report(
            self.sample_report_data, output_format="html"
        )
        
        # Check that report file was created
        self.assertTrue(report_path.exists())
        
        # Read content and check
        with open(report_path, "r") as f:
            content = f.read()
            self.assertIn("<h1>Apple Inc. (AAPL)</h1>", content)
        
        # Clean up report file
        report_path.unlink()
    
    @patch("matplotlib.pyplot.savefig")
    def test_generate_full_report_eml(self, mock_savefig):
        """Test generating a full EML report."""
        report_path = self.generator.generate_full_report(
            self.sample_report_data, output_format="eml"
        )
        
        # Check that report file was created
        self.assertTrue(report_path.exists())
        
        # Read content and check
        with open(report_path, "r") as f:
            content = f.read()
            self.assertIn("Subject: Equity Research Report: Apple Inc.", content)
            self.assertIn("Content-Type: multipart/related", content)
            self.assertIn("Content-ID: <profitability_chart>", content)
        
        # Clean up report file
        report_path.unlink()

if __name__ == '__main__':
    unittest.main()

