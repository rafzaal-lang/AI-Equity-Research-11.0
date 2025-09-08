import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.core.pti import (
    filter_series_asof, validate_time_series, get_period_data,
    calculate_growth_metrics, create_pti_snapshot
)
from src.services.pti_financials import PTIFinancialService

class TestPTICore(unittest.TestCase):
    """Test core PTI functionality."""
    
    def setUp(self):
        self.sample_data = [
            {"date": "2023-12-31", "revenue": 100000, "net_income": 10000},
            {"date": "2022-12-31", "revenue": 90000, "net_income": 9000},
            {"date": "2021-12-31", "revenue": 80000, "net_income": 8000},
            {"date": "2020-12-31", "revenue": 70000, "net_income": 7000},
        ]
    
    def test_filter_series_asof(self):
        """Test filtering data as of a specific date."""
        # Test with as_of date
        filtered = filter_series_asof(self.sample_data, "2022-12-31")
        self.assertEqual(len(filtered), 3)  # Should exclude 2023 data
        
        # Test without as_of date
        filtered = filter_series_asof(self.sample_data, None)
        self.assertEqual(len(filtered), 4)  # Should return all data
        
        # Test with future date
        filtered = filter_series_asof(self.sample_data, "2025-12-31")
        self.assertEqual(len(filtered), 4)  # Should return all data
    
    def test_validate_time_series(self):
        """Test time series validation."""
        # Test valid data
        validation = validate_time_series(self.sample_data, value_keys=["revenue", "net_income"])
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["record_count"], 4)
        
        # Test data with missing values
        incomplete_data = [
            {"date": "2023-12-31", "revenue": 100000, "net_income": None},
            {"date": "2022-12-31", "revenue": None, "net_income": 9000},
        ]
        validation = validate_time_series(incomplete_data, value_keys=["revenue", "net_income"])
        self.assertFalse(validation["valid"])  # Should be invalid due to high missing percentage
        
        # Test empty data
        validation = validate_time_series([])
        self.assertFalse(validation["valid"])
        self.assertIn("Empty dataset", validation["errors"])
    
    def test_get_period_data(self):
        """Test getting period data."""
        # Test getting last 2 periods as of 2022
        periods = get_period_data(self.sample_data, "2022-12-31", lookback_periods=2)
        self.assertEqual(len(periods), 2)
        self.assertEqual(periods[0]["date"], "2022-12-31")  # Most recent first
        self.assertEqual(periods[1]["date"], "2021-12-31")
    
    def test_calculate_growth_metrics(self):
        """Test growth metrics calculation."""
        growth = calculate_growth_metrics(self.sample_data, "revenue")
        
        # Check YoY growth (2023 vs 2022)
        expected_yoy = (100000 / 90000) - 1
        self.assertAlmostEqual(growth["yoy_growth"], expected_yoy, places=4)
        
        # Check that CAGR values are calculated
        self.assertIsNotNone(growth["cagr_3y"])
        self.assertIsNotNone(growth["avg_growth"])
    
    def test_create_pti_snapshot(self):
        """Test PTI snapshot creation."""
        data_sources = {
            "income_statement": self.sample_data,
            "balance_sheet": [
                {"date": "2023-12-31", "total_assets": 500000},
                {"date": "2022-12-31", "total_assets": 450000},
            ]
        }
        
        snapshot = create_pti_snapshot("AAPL", "2022-12-31", data_sources)
        
        self.assertEqual(snapshot["symbol"], "AAPL")
        self.assertEqual(snapshot["as_of"], "2022-12-31")
        self.assertIn("data_quality", snapshot)
        self.assertIn("financials", snapshot)

class TestPTIFinancialService(unittest.TestCase):
    """Test PTI Financial Service."""
    
    def setUp(self):
        self.service = PTIFinancialService()
        
        # Mock data
        self.mock_income = [
            {
                "date": "2023-12-31",
                "revenue": 100000000,
                "netIncome": 10000000,
                "grossProfit": 40000000,
                "operatingIncome": 15000000
            },
            {
                "date": "2022-12-31", 
                "revenue": 90000000,
                "netIncome": 9000000,
                "grossProfit": 36000000,
                "operatingIncome": 13500000
            }
        ]
        
        self.mock_balance = [
            {
                "date": "2023-12-31",
                "totalAssets": 500000000,
                "totalStockholdersEquity": 200000000,
                "totalDebt": 100000000,
                "totalCurrentAssets": 150000000,
                "totalCurrentLiabilities": 80000000
            },
            {
                "date": "2022-12-31",
                "totalAssets": 450000000,
                "totalStockholdersEquity": 180000000,
                "totalDebt": 90000000,
                "totalCurrentAssets": 135000000,
                "totalCurrentLiabilities": 75000000
            }
        ]
        
        self.mock_cashflow = [
            {
                "date": "2023-12-31",
                "operatingCashFlow": 12000000,
                "freeCashFlow": 8000000,
                "capitalExpenditure": -4000000
            },
            {
                "date": "2022-12-31",
                "operatingCashFlow": 11000000,
                "freeCashFlow": 7500000,
                "capitalExpenditure": -3500000
            }
        ]
    
    @patch('src.services.providers.fmp_provider.income_statement')
    @patch('src.services.providers.fmp_provider.balance_sheet')
    @patch('src.services.providers.fmp_provider.cash_flow')
    def test_get_statements_asof(self, mock_cf, mock_bs, mock_inc):
        """Test getting statements as of a specific date."""
        mock_inc.return_value = self.mock_income
        mock_bs.return_value = self.mock_balance
        mock_cf.return_value = self.mock_cashflow
        
        result = self.service.get_statements_asof("AAPL", "2022-12-31")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["as_of"], "2022-12-31")
        self.assertIn("data", result)
        self.assertIn("validation", result)
        self.assertGreater(result["data_quality_score"], 0)
    
    @patch('src.services.providers.fmp_provider.income_statement')
    @patch('src.services.providers.fmp_provider.balance_sheet')
    @patch('src.services.providers.fmp_provider.cash_flow')
    def test_get_financial_metrics_asof(self, mock_cf, mock_bs, mock_inc):
        """Test getting financial metrics as of a specific date."""
        mock_inc.return_value = self.mock_income
        mock_bs.return_value = self.mock_balance
        mock_cf.return_value = self.mock_cashflow
        
        result = self.service.get_financial_metrics_asof("AAPL", "2023-12-31")
        
        self.assertEqual(result["symbol"], "AAPL")
        self.assertIn("metrics", result)
        self.assertIn("growth_analysis", result)
        
        # Check that key metrics are calculated
        metrics = result["metrics"]
        self.assertIn("profitability", metrics)
        self.assertIn("liquidity", metrics)
        self.assertIn("leverage", metrics)
        self.assertIn("cash_flow", metrics)
        
        # Check specific calculations
        profitability = metrics["profitability"]
        self.assertAlmostEqual(profitability["gross_margin"], 0.4, places=2)  # 40M/100M
        self.assertAlmostEqual(profitability["net_margin"], 0.1, places=2)    # 10M/100M
    
    def test_calculate_quality_score(self):
        """Test data quality score calculation."""
        validation_results = {
            "income_statement": {"valid": True, "warnings": []},
            "balance_sheet": {"valid": True, "warnings": ["Some warning"]},
            "cash_flow": {"valid": False, "errors": ["Error"]}
        }
        
        score = self.service._calculate_quality_score(validation_results)
        
        # Should be (1.0 + 0.9 + 0.0) / 3 = 0.633...
        self.assertAlmostEqual(score, 0.633, places=2)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.service._calculate_comprehensive_metrics(
            self.mock_income, self.mock_balance, self.mock_cashflow
        )
        
        # Test profitability ratios
        prof = metrics["profitability"]
        self.assertAlmostEqual(prof["gross_margin"], 0.4, places=2)
        self.assertAlmostEqual(prof["operating_margin"], 0.15, places=2)
        self.assertAlmostEqual(prof["net_margin"], 0.1, places=2)
        
        # Test liquidity ratios
        liq = metrics["liquidity"]
        self.assertAlmostEqual(liq["current_ratio"], 1.875, places=2)  # 150M/80M
        
        # Test leverage ratios
        lev = metrics["leverage"]
        self.assertAlmostEqual(lev["debt_to_equity"], 0.5, places=2)  # 100M/200M

if __name__ == '__main__':
    unittest.main()

