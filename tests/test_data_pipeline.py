import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.providers.enhanced_fmp_provider import (
    EnhancedFMPProvider, DataQuality, DataValidationResult, FMPError, RateLimitError
)
from services.data_pipeline_monitor import DataPipelineMonitor, PipelineStatus

class TestEnhancedFMPProvider(unittest.TestCase):
    """Test enhanced FMP provider functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        os.environ["FMP_API_KEY"] = "test_key"
        os.environ["FMP_BASE_URL"] = "https://test.api.com/api"
        
        self.provider = EnhancedFMPProvider()
        
        # Sample valid financial data
        self.sample_income_data = [
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
        
        self.sample_balance_data = [
            {
                "date": "2023-12-31",
                "totalAssets": 500000000,
                "totalStockholdersEquity": 200000000,
                "totalDebt": 100000000
            },
            {
                "date": "2022-12-31",
                "totalAssets": 450000000,
                "totalStockholdersEquity": 180000000,
                "totalDebt": 90000000
            }
        ]
    
    def test_rate_limiter(self):
        """Test rate limiting functionality."""
        # Reset rate limiter
        self.provider.rate_limiter = self.provider._init_rate_limiter()
        self.provider.rate_limiter["max_calls_per_minute"] = 2  # Low limit for testing
        
        # Should not raise error for first few calls
        try:
            self.provider._check_rate_limit()
            self.provider._check_rate_limit()
        except RateLimitError:
            self.fail("Rate limiter raised error too early")
        
        # Should raise error when limit exceeded
        with self.assertRaises(RateLimitError):
            self.provider._check_rate_limit()
    
    def test_validate_financial_data_valid(self):
        """Test validation with valid data."""
        validation = self.provider.validate_financial_data(
            self.sample_income_data, "income_statement", "AAPL"
        )
        
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.quality, DataQuality.HIGH)
        self.assertEqual(len(validation.errors), 0)
        self.assertGreater(validation.completeness_score, 0.9)
        self.assertGreater(validation.freshness_score, 0.8)
    
    def test_validate_financial_data_invalid(self):
        """Test validation with invalid data."""
        invalid_data = [
            {"revenue": 100000000},  # Missing date
            {"date": "invalid-date", "revenue": 90000000}  # Invalid date format
        ]
        
        validation = self.provider.validate_financial_data(
            invalid_data, "income_statement", "AAPL"
        )
        
        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.quality, DataQuality.INVALID)
        self.assertGreater(len(validation.errors), 0)
    
    def test_validate_financial_data_empty(self):
        """Test validation with empty data."""
        validation = self.provider.validate_financial_data([], "income_statement", "AAPL")
        
        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.quality, DataQuality.INVALID)
        self.assertIn("No data returned", validation.errors)
        self.assertEqual(validation.completeness_score, 0.0)
        self.assertEqual(validation.freshness_score, 0.0)
    
    def test_validate_financial_data_old_data(self):
        """Test validation with old data."""
        old_data = [
            {
                "date": "2020-12-31",  # Old date
                "revenue": 100000000,
                "netIncome": 10000000
            }
        ]
        
        validation = self.provider.validate_financial_data(old_data, "income_statement", "AAPL")
        
        self.assertTrue(validation.is_valid)  # Structure is valid
        self.assertLess(validation.freshness_score, 0.8)  # But data is old
        self.assertIn([DataQuality.LOW, DataQuality.MEDIUM], [validation.quality])
    
    @patch('src.services.providers.enhanced_fmp_provider.SESSION')
    def test_make_request_success(self, mock_session):
        """Test successful API request."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_income_data
        mock_response.content = b'{"data": "test"}'
        mock_session.get.return_value = mock_response
        
        data, metrics = self.provider._make_request("income-statement/AAPL")
        
        self.assertEqual(data, self.sample_income_data)
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.symbol, "AAPL")
        self.assertGreater(metrics.end_time, metrics.start_time)
    
    @patch('src.services.providers.enhanced_fmp_provider.SESSION')
    def test_make_request_rate_limit(self, mock_session):
        """Test rate limit handling."""
        # Mock rate limit response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_session.get.return_value = mock_response
        
        with self.assertRaises(RateLimitError):
            self.provider._make_request("income-statement/AAPL")
    
    @patch('src.services.providers.enhanced_fmp_provider.SESSION')
    def test_make_request_http_error(self, mock_session):
        """Test HTTP error handling."""
        # Mock 404 response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_session.get.return_value = mock_response
        
        with self.assertRaises(FMPError):
            self.provider._make_request("invalid-endpoint")
    
    def test_get_health_metrics_no_data(self):
        """Test health metrics with no call history."""
        # Clear call metrics
        self.provider.call_metrics = []
        
        health = self.provider.get_health_metrics()
        
        self.assertEqual(health["status"], "no_data")
        self.assertEqual(health["metrics"], {})
    
    def test_get_health_metrics_with_data(self):
        """Test health metrics with call history."""
        from services.providers.enhanced_fmp_provider import APICallMetrics
        import time
        
        # Add some mock metrics
        now = time.time()
        self.provider.call_metrics = [
            APICallMetrics(
                endpoint="income-statement/AAPL",
                symbol="AAPL",
                start_time=now - 1,
                end_time=now,
                success=True,
                cached=False
            ),
            APICallMetrics(
                endpoint="balance-sheet/AAPL",
                symbol="AAPL", 
                start_time=now - 2,
                end_time=now - 1,
                success=True,
                cached=True
            )
        ]
        
        health = self.provider.get_health_metrics()
        
        self.assertIn(health["status"], ["healthy", "degraded", "unhealthy"])
        self.assertIn("success_rate", health["metrics"])
        self.assertIn("avg_response_time_ms", health["metrics"])
        self.assertIn("cache_hit_rate", health["metrics"])

class TestDataPipelineMonitor(unittest.TestCase):
    """Test data pipeline monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = DataPipelineMonitor()
    
    def test_check_fmp_health_no_data(self):
        """Test FMP health check with no data."""
        with patch('src.services.providers.enhanced_fmp_provider.enhanced_fmp_provider') as mock_provider:
            mock_provider.get_health_metrics.return_value = {"status": "no_data"}
            
            health = self.monitor.check_fmp_health()
            
            self.assertEqual(health.source_name, "FMP")
            self.assertEqual(health.status, PipelineStatus.CRITICAL)
            self.assertEqual(health.success_rate, 0.0)
            self.assertIn("No API calls recorded", health.issues)
    
    def test_check_fmp_health_healthy(self):
        """Test FMP health check with healthy metrics."""
        with patch('src.services.providers.enhanced_fmp_provider.enhanced_fmp_provider') as mock_provider:
            mock_provider.get_health_metrics.return_value = {
                "status": "healthy",
                "metrics": {
                    "success_rate": 0.98,
                    "avg_response_time_ms": 1500,
                    "cache_hit_rate": 0.85,
                    "error_rate": 0.02,
                    "total_calls_last_hour": 100
                }
            }
            
            health = self.monitor.check_fmp_health()
            
            self.assertEqual(health.source_name, "FMP")
            self.assertEqual(health.status, PipelineStatus.HEALTHY)
            self.assertEqual(health.success_rate, 0.98)
            self.assertEqual(health.avg_response_time, 1.5)
            self.assertEqual(len(health.issues), 0)
    
    def test_check_fmp_health_degraded(self):
        """Test FMP health check with degraded metrics."""
        with patch('src.services.providers.enhanced_fmp_provider.enhanced_fmp_provider') as mock_provider:
            mock_provider.get_health_metrics.return_value = {
                "status": "degraded",
                "metrics": {
                    "success_rate": 0.90,  # Below warning threshold
                    "avg_response_time_ms": 6000,  # Above warning threshold
                    "cache_hit_rate": 0.60,  # Below warning threshold
                    "error_rate": 0.10,
                    "total_calls_last_hour": 50
                }
            }
            
            health = self.monitor.check_fmp_health()
            
            self.assertEqual(health.source_name, "FMP")
            self.assertEqual(health.status, PipelineStatus.DEGRADED)
            self.assertGreater(len(health.issues), 0)
            self.assertTrue(any("Low success rate" in issue for issue in health.issues))
            self.assertTrue(any("High response time" in issue for issue in health.issues))
    
    def test_generate_alerts(self):
        """Test alert generation."""
        from services.data_pipeline_monitor import DataSourceHealth
        
        # Create unhealthy data source
        unhealthy_health = DataSourceHealth(
            source_name="FMP",
            status=PipelineStatus.UNHEALTHY,
            success_rate=0.80,  # Below error threshold
            avg_response_time=12.0,  # Above error threshold
            cache_hit_rate=0.50,
            error_count=20,
            last_successful_call=datetime.utcnow(),
            data_quality_score=0.50,  # Below error threshold
            freshness_score=0.40,
            issues=[]
        )
        
        alerts = self.monitor._generate_alerts(unhealthy_health)
        
        self.assertGreater(len(alerts), 0)
        
        # Check for specific alert types
        alert_messages = [alert.message for alert in alerts]
        self.assertTrue(any("Low success rate" in msg for msg in alert_messages))
        self.assertTrue(any("High response time" in msg for msg in alert_messages))
        self.assertTrue(any("Low data quality score" in msg for msg in alert_messages))
    
    def test_run_health_check(self):
        """Test comprehensive health check."""
        with patch.object(self.monitor, 'check_fmp_health') as mock_check:
            from services.data_pipeline_monitor import DataSourceHealth
            
            mock_check.return_value = DataSourceHealth(
                source_name="FMP",
                status=PipelineStatus.HEALTHY,
                success_rate=0.98,
                avg_response_time=1.5,
                cache_hit_rate=0.85,
                error_count=2,
                last_successful_call=datetime.utcnow(),
                data_quality_score=0.95,
                freshness_score=0.90,
                issues=[]
            )
            
            report = self.monitor.run_health_check()
            
            self.assertIn("timestamp", report)
            self.assertEqual(report["overall_status"], PipelineStatus.HEALTHY.value)
            self.assertIn("FMP", report["data_sources"])
            self.assertIn("summary", report)
            self.assertEqual(report["summary"]["healthy_sources"], 1)
    
    @patch('src.services.providers.enhanced_fmp_provider.enhanced_fmp_provider')
    def test_check_data_quality(self, mock_provider):
        """Test data quality checking."""
        # Mock provider responses
        mock_provider.income_statement.return_value = {
            "symbol": "AAPL",
            "data": [{"date": "2023-12-31", "revenue": 100000000}],
            "validation": DataValidationResult(
                is_valid=True,
                quality=DataQuality.HIGH,
                errors=[],
                warnings=[],
                completeness_score=0.95,
                freshness_score=0.90,
                metadata={}
            )
        }
        
        mock_provider.balance_sheet.return_value = {
            "symbol": "AAPL",
            "data": [{"date": "2023-12-31", "totalAssets": 500000000}],
            "validation": DataValidationResult(
                is_valid=True,
                quality=DataQuality.HIGH,
                errors=[],
                warnings=[],
                completeness_score=0.90,
                freshness_score=0.85,
                metadata={}
            )
        }
        
        mock_provider.cash_flow.return_value = {
            "symbol": "AAPL",
            "data": [{"date": "2023-12-31", "operatingCashFlow": 12000000}],
            "validation": DataValidationResult(
                is_valid=True,
                quality=DataQuality.MEDIUM,
                errors=[],
                warnings=["Some missing data"],
                completeness_score=0.80,
                freshness_score=0.85,
                metadata={}
            )
        }
        
        quality_report = self.monitor.check_data_quality("AAPL")
        
        self.assertEqual(quality_report["symbol"], "AAPL")
        self.assertGreater(quality_report["overall_quality_score"], 0.8)
        self.assertIn("income_statement", quality_report["data_types"])
        self.assertIn("balance_sheet", quality_report["data_types"])
        self.assertIn("cash_flow", quality_report["data_types"])

if __name__ == '__main__':
    unittest.main()

