import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.financial_modeler import (
    calculate_comprehensive_ratios, scenario_analysis, monte_carlo_simulation,
    multi_stage_dcf, comprehensive_financial_model
)

class TestFinancialModeling(unittest.TestCase):
    """Test enhanced financial modeling functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_income = [
            {
                "date": "2023-12-31",
                "revenue": 100000000,
                "netIncome": 10000000,
                "grossProfit": 40000000,
                "operatingIncome": 15000000,
                "ebitda": 18000000,
                "interestExpense": 2000000,
                "weightedAverageShsOut": 10000000
            }
        ]
        
        self.sample_balance = [
            {
                "date": "2023-12-31",
                "totalAssets": 500000000,
                "totalCurrentAssets": 150000000,
                "cashAndCashEquivalents": 50000000,
                "inventory": 30000000,
                "netReceivables": 40000000,
                "totalLiabilities": 300000000,
                "totalCurrentLiabilities": 80000000,
                "totalDebt": 100000000,
                "longTermDebt": 80000000,
                "totalStockholdersEquity": 200000000
            }
        ]
        
        self.sample_cashflow = [
            {
                "date": "2023-12-31",
                "operatingCashFlow": 12000000,
                "freeCashFlow": 8000000,
                "capitalExpenditure": -4000000
            }
        ]
        
        self.sample_market_data = {
            "market_cap": 150000000,
            "share_price": 15.0
        }
    
    def test_calculate_comprehensive_ratios(self):
        """Test comprehensive ratio calculations."""
        ratios = calculate_comprehensive_ratios(
            self.sample_income, 
            self.sample_balance, 
            self.sample_cashflow,
            self.sample_market_data
        )
        
        # Test profitability ratios
        prof = ratios["profitability"]
        self.assertAlmostEqual(prof["gross_margin"], 0.4, places=2)  # 40M/100M
        self.assertAlmostEqual(prof["operating_margin"], 0.15, places=2)  # 15M/100M
        self.assertAlmostEqual(prof["net_margin"], 0.1, places=2)  # 10M/100M
        self.assertAlmostEqual(prof["roa"], 0.02, places=2)  # 10M/500M
        self.assertAlmostEqual(prof["roe"], 0.05, places=2)  # 10M/200M
        
        # Test liquidity ratios
        liq = ratios["liquidity"]
        self.assertAlmostEqual(liq["current_ratio"], 1.875, places=2)  # 150M/80M
        self.assertAlmostEqual(liq["quick_ratio"], 1.5, places=2)  # (150M-30M)/80M
        self.assertAlmostEqual(liq["cash_ratio"], 0.625, places=2)  # 50M/80M
        
        # Test leverage ratios
        lev = ratios["leverage"]
        self.assertAlmostEqual(lev["debt_to_equity"], 0.5, places=2)  # 100M/200M
        self.assertAlmostEqual(lev["debt_to_assets"], 0.2, places=2)  # 100M/500M
        self.assertAlmostEqual(lev["equity_ratio"], 0.4, places=2)  # 200M/500M
        
        # Test cash flow ratios
        cf = ratios["cash_flow"]
        self.assertAlmostEqual(cf["operating_cf_ratio"], 0.15, places=2)  # 12M/80M
        self.assertAlmostEqual(cf["operating_cf_margin"], 0.12, places=2)  # 12M/100M
        self.assertAlmostEqual(cf["capex_intensity"], 0.04, places=2)  # 4M/100M
        
        # Test valuation ratios
        val = ratios["valuation"]
        eps = 10000000 / 10000000  # Net income / shares
        self.assertAlmostEqual(val["pe_ratio"], 15.0, places=1)  # 15.0 / 1.0
        self.assertAlmostEqual(val["ps_ratio"], 1.5, places=1)  # 150M/100M
    
    def test_multi_stage_dcf(self):
        """Test multi-stage DCF calculation."""
        dcf = multi_stage_dcf(
            base_fcf=10000000,  # $10M base FCF
            years_stage1=5,
            g1=0.08,  # 8% growth stage 1
            years_stage2=5,
            g2=0.04,  # 4% growth stage 2
            g_terminal=0.025,  # 2.5% terminal growth
            wacc=0.10  # 10% WACC
        )
        
        # Check that all components are present
        self.assertIn("assumptions", dcf)
        self.assertIn("stage1_pv", dcf)
        self.assertIn("stage2_pv", dcf)
        self.assertIn("terminal_pv", dcf)
        self.assertIn("enterprise_value", dcf)
        self.assertIn("sensitivity_analysis", dcf)
        
        # Check that enterprise value is positive
        self.assertGreater(dcf["enterprise_value"], 0)
        
        # Check that terminal value percentage is reasonable
        self.assertGreater(dcf["terminal_value_pct"], 0.3)  # Should be significant portion
        self.assertLess(dcf["terminal_value_pct"], 0.9)     # But not overwhelming
        
        # Check sensitivity analysis has multiple scenarios
        self.assertGreater(len(dcf["sensitivity_analysis"]), 5)
    
    def test_scenario_analysis(self):
        """Test scenario analysis functionality."""
        # Create a mock base model
        base_model = {
            "dcf_valuation": {
                "assumptions": {
                    "wacc": 0.10,
                    "g1": 0.06,
                    "g2": 0.04,
                    "g_terminal": 0.025,
                    "years_stage1": 5,
                    "years_stage2": 5
                },
                "enterprise_value": 100000000
            },
            "core_financials": {
                "reported": {
                    "free_cash_flow": 10000000
                }
            }
        }
        
        scenarios = [
            {
                "name": "Bull Case",
                "description": "Optimistic assumptions",
                "assumptions": {"g1": 0.10, "wacc": 0.08}
            },
            {
                "name": "Bear Case",
                "description": "Conservative assumptions", 
                "assumptions": {"g1": 0.02, "wacc": 0.12}
            }
        ]
        
        results = scenario_analysis(base_model, scenarios)
        
        # Check structure
        self.assertIn("base_case", results)
        self.assertIn("scenarios", results)
        self.assertEqual(results["scenario_count"], 2)
        
        # Check that scenarios were calculated
        self.assertIn("Bull Case", results["scenarios"])
        self.assertIn("Bear Case", results["scenarios"])
        
        # Bull case should have higher EV than bear case
        bull_ev = results["scenarios"]["Bull Case"]["enterprise_value"]
        bear_ev = results["scenarios"]["Bear Case"]["enterprise_value"]
        self.assertGreater(bull_ev, bear_ev)
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        # Create a mock base model
        base_model = {
            "dcf_valuation": {
                "assumptions": {
                    "wacc": 0.10,
                    "g1": 0.06,
                    "g2": 0.04,
                    "g_terminal": 0.025,
                    "years_stage1": 5,
                    "years_stage2": 5
                },
                "enterprise_value": 100000000
            },
            "core_financials": {
                "reported": {
                    "free_cash_flow": 10000000
                }
            }
        }
        
        # Run with small number of simulations for testing
        results = monte_carlo_simulation(base_model, num_simulations=100)
        
        # Check structure
        self.assertIn("simulation_count", results)
        self.assertIn("statistics", results)
        self.assertIn("percentiles", results)
        self.assertIn("confidence_intervals", results)
        
        # Check that simulations ran
        self.assertGreater(results["simulation_count"], 50)  # Should complete most simulations
        
        # Check statistics
        stats = results["statistics"]
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("std_dev", stats)
        
        # Check percentiles
        percentiles = results["percentiles"]
        self.assertLess(percentiles["p5"], percentiles["p95"])  # Sanity check
    
    @patch('src.services.providers.fmp_provider.income_statement')
    @patch('src.services.providers.fmp_provider.balance_sheet')
    @patch('src.services.providers.fmp_provider.cash_flow')
    @patch('src.services.providers.fmp_provider.enterprise_values')
    @patch('src.services.providers.fmp_provider.latest_price')
    @patch('src.services.providers.fmp_provider.profile')
    @patch('src.services.providers.fmp_provider.peers_by_screener')
    def test_comprehensive_financial_model(self, mock_peers, mock_profile, mock_price, 
                                         mock_ev, mock_cf, mock_bs, mock_inc):
        """Test comprehensive financial model building."""
        # Set up mocks
        mock_inc.return_value = self.sample_income
        mock_bs.return_value = self.sample_balance
        mock_cf.return_value = self.sample_cashflow
        mock_ev.return_value = [{"marketCapitalization": 150000000}]
        mock_price.return_value = 15.0
        mock_profile.return_value = {"sector": "Technology", "industry": "Software"}
        mock_peers.return_value = ["MSFT", "GOOGL", "META"]
        
        # Build comprehensive model
        model = comprehensive_financial_model(
            "AAPL", 
            include_scenarios=True, 
            include_monte_carlo=False  # Skip MC for faster testing
        )
        
        # Check structure
        self.assertEqual(model["symbol"], "AAPL")
        self.assertEqual(model["model_type"], "comprehensive_institutional")
        self.assertIn("core_financials", model)
        self.assertIn("dcf_valuation", model)
        self.assertIn("wacc_analysis", model)
        self.assertIn("comprehensive_ratios", model)
        self.assertIn("scenario_analysis", model)
        
        # Check that comprehensive ratios were calculated
        ratios = model["comprehensive_ratios"]
        self.assertIn("profitability", ratios)
        self.assertIn("liquidity", ratios)
        self.assertIn("leverage", ratios)
        self.assertIn("efficiency", ratios)
        self.assertIn("cash_flow", ratios)
        self.assertIn("valuation", ratios)

if __name__ == '__main__':
    unittest.main()

