import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.backtesting.framework import (
    BacktestingFramework, Position, Transaction, PerformanceMetrics, 
    RebalanceFrequency, momentum_strategy, mean_reversion_strategy
)
from services.backtesting.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationObjective, OptimizationConstraints, PortfolioResult
)

class TestBacktestingFramework(unittest.TestCase):
    """Test backtesting framework functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.framework = BacktestingFramework(initial_capital=100000, commission=0.001)
        
        # Create sample price data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        self.sample_price_data = {
            'AAPL': pd.DataFrame({
                'close': 150 + np.cumsum(np.random.randn(len(dates)) * 0.02),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates),
            'MSFT': pd.DataFrame({
                'close': 300 + np.cumsum(np.random.randn(len(dates)) * 0.015),
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates),
            'GOOGL': pd.DataFrame({
                'close': 2500 + np.cumsum(np.random.randn(len(dates)) * 0.025),
                'volume': np.random.randint(500000, 5000000, len(dates))
            }, index=dates)
        }
    
    def test_execute_trade_buy(self):
        """Test buying shares."""
        date = datetime(2023, 1, 15)
        success = self.framework.execute_trade('AAPL', 'buy', 100, 150.0, date)
        
        self.assertTrue(success)
        self.assertIn('AAPL', self.framework.positions)
        self.assertEqual(self.framework.positions['AAPL'].shares, 100)
        self.assertEqual(self.framework.positions['AAPL'].price, 150.0)
        self.assertLess(self.framework.cash, 100000)  # Cash should decrease
        self.assertEqual(len(self.framework.transactions), 1)
    
    def test_execute_trade_sell(self):
        """Test selling shares."""
        date = datetime(2023, 1, 15)
        
        # First buy some shares
        self.framework.execute_trade('AAPL', 'buy', 100, 150.0, date)
        initial_cash = self.framework.cash
        
        # Then sell some
        success = self.framework.execute_trade('AAPL', 'sell', 50, 155.0, date)
        
        self.assertTrue(success)
        self.assertEqual(self.framework.positions['AAPL'].shares, 50)
        self.assertGreater(self.framework.cash, initial_cash)  # Cash should increase
        self.assertEqual(len(self.framework.transactions), 2)
    
    def test_execute_trade_insufficient_cash(self):
        """Test buying with insufficient cash."""
        date = datetime(2023, 1, 15)
        
        # Try to buy more than we can afford
        success = self.framework.execute_trade('AAPL', 'buy', 1000, 150.0, date)
        
        self.assertFalse(success)
        self.assertNotIn('AAPL', self.framework.positions)
        self.assertEqual(len(self.framework.transactions), 0)
    
    def test_execute_trade_insufficient_shares(self):
        """Test selling more shares than owned."""
        date = datetime(2023, 1, 15)
        
        # Buy some shares first
        self.framework.execute_trade('AAPL', 'buy', 50, 150.0, date)
        
        # Try to sell more than we own
        success = self.framework.execute_trade('AAPL', 'sell', 100, 155.0, date)
        
        self.assertFalse(success)
        self.assertEqual(self.framework.positions['AAPL'].shares, 50)  # Should remain unchanged
        self.assertEqual(len(self.framework.transactions), 1)  # Only the buy transaction
    
    def test_update_portfolio_value(self):
        """Test portfolio value updates."""
        date = datetime(2023, 1, 15)
        
        # Buy some shares
        self.framework.execute_trade('AAPL', 'buy', 100, 150.0, date)
        self.framework.execute_trade('MSFT', 'buy', 50, 300.0, date)
        
        # Update with new prices
        current_prices = {'AAPL': 160.0, 'MSFT': 310.0}
        self.framework.update_portfolio_value(current_prices, date)
        
        # Check portfolio value
        expected_value = self.framework.cash + (100 * 160.0) + (50 * 310.0)
        self.assertAlmostEqual(self.framework.portfolio_value, expected_value, places=2)
        
        # Check position weights
        total_position_value = (100 * 160.0) + (50 * 310.0)
        expected_aapl_weight = (100 * 160.0) / expected_value
        expected_msft_weight = (50 * 310.0) / expected_value
        
        self.assertAlmostEqual(self.framework.positions['AAPL'].weight, expected_aapl_weight, places=4)
        self.assertAlmostEqual(self.framework.positions['MSFT'].weight, expected_msft_weight, places=4)
    
    def test_rebalance_portfolio(self):
        """Test portfolio rebalancing."""
        date = datetime(2023, 1, 15)
        
        # Start with some positions
        self.framework.execute_trade('AAPL', 'buy', 100, 150.0, date)
        current_prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
        self.framework.update_portfolio_value(current_prices, date)
        
        # Rebalance to equal weights
        target_weights = {'AAPL': 0.33, 'MSFT': 0.33, 'GOOGL': 0.34}
        self.framework.rebalance_portfolio(target_weights, current_prices, date)
        
        # Check that we now have positions in all three stocks
        self.assertIn('AAPL', self.framework.positions)
        self.assertIn('MSFT', self.framework.positions)
        self.assertIn('GOOGL', self.framework.positions)
    
    def test_get_rebalance_dates_monthly(self):
        """Test monthly rebalancing date generation."""
        dates = [
            datetime(2023, 1, 15),
            datetime(2023, 1, 20),
            datetime(2023, 2, 5),
            datetime(2023, 2, 15),
            datetime(2023, 3, 1)
        ]
        
        rebalance_dates = self.framework._get_rebalance_dates(dates, RebalanceFrequency.MONTHLY)
        
        # Should have one date per month
        self.assertEqual(len(rebalance_dates), 3)
        self.assertEqual(rebalance_dates[0].month, 1)
        self.assertEqual(rebalance_dates[1].month, 2)
        self.assertEqual(rebalance_dates[2].month, 3)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some portfolio history
        dates = [datetime(2023, 1, 1), datetime(2023, 6, 1), datetime(2023, 12, 31)]
        values = [100000, 110000, 120000]
        
        for date, value in zip(dates, values):
            self.framework.portfolio_history.append({
                'date': date,
                'portfolio_value': value,
                'cash': value * 0.1,
                'positions_value': value * 0.9,
                'positions': {}
            })
        
        metrics = self.framework.calculate_performance_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.total_return, 0)  # Should be positive
        self.assertGreater(metrics.annualized_return, 0)
        self.assertGreater(metrics.volatility, 0)
    
    def test_momentum_strategy(self):
        """Test momentum strategy function."""
        current_date = datetime(2023, 6, 15)
        weights = momentum_strategy(self.sample_price_data, current_date)
        
        self.assertIsInstance(weights, dict)
        if weights:  # If strategy returns any weights
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
            for weight in weights.values():
                self.assertGreaterEqual(weight, 0)
                self.assertLessEqual(weight, 1)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy function."""
        current_date = datetime(2023, 6, 15)
        weights = mean_reversion_strategy(self.sample_price_data, current_date)
        
        self.assertIsInstance(weights, dict)
        if weights:  # If strategy returns any weights
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
            for weight in weights.values():
                self.assertGreaterEqual(weight, 0)
                self.assertLessEqual(weight, 1)

class TestPortfolioOptimizer(unittest.TestCase):
    """Test portfolio optimization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.sample_price_data = {
            'AAPL': pd.DataFrame({
                'close': 150 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            }, index=dates),
            'MSFT': pd.DataFrame({
                'close': 300 + np.cumsum(np.random.randn(len(dates)) * 0.015)
            }, index=dates),
            'GOOGL': pd.DataFrame({
                'close': 2500 + np.cumsum(np.random.randn(len(dates)) * 0.025)
            }, index=dates)
        }
        
        # Calculate expected returns and covariance
        self.expected_returns = self.optimizer.calculate_expected_returns(self.sample_price_data)
        self.cov_matrix = self.optimizer.calculate_covariance_matrix(self.sample_price_data)
    
    def test_calculate_expected_returns(self):
        """Test expected returns calculation."""
        returns = self.optimizer.calculate_expected_returns(self.sample_price_data)
        
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), 3)
        self.assertTrue(all(symbol in returns.index for symbol in ['AAPL', 'MSFT', 'GOOGL']))
    
    def test_calculate_covariance_matrix(self):
        """Test covariance matrix calculation."""
        cov_matrix = self.optimizer.calculate_covariance_matrix(self.sample_price_data)
        
        self.assertIsInstance(cov_matrix, pd.DataFrame)
        self.assertEqual(cov_matrix.shape, (3, 3))
        self.assertTrue(all(symbol in cov_matrix.index for symbol in ['AAPL', 'MSFT', 'GOOGL']))
        self.assertTrue(all(symbol in cov_matrix.columns for symbol in ['AAPL', 'MSFT', 'GOOGL']))
        
        # Check that matrix is symmetric and positive semi-definite
        np.testing.assert_array_almost_equal(cov_matrix.values, cov_matrix.values.T)
        eigenvals = np.linalg.eigvals(cov_matrix.values)
        self.assertTrue(all(eigenvals >= -1e-8))  # Allow for small numerical errors
    
    def test_optimize_portfolio_max_sharpe(self):
        """Test maximum Sharpe ratio optimization."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns, self.cov_matrix, OptimizationObjective.MAX_SHARPE
        )
        
        self.assertIsInstance(result, PortfolioResult)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=4)
        self.assertGreater(result.expected_return, 0)
        self.assertGreater(result.expected_risk, 0)
        self.assertIsInstance(result.sharpe_ratio, float)
    
    def test_optimize_portfolio_min_variance(self):
        """Test minimum variance optimization."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns, self.cov_matrix, OptimizationObjective.MIN_VARIANCE
        )
        
        self.assertIsInstance(result, PortfolioResult)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=4)
        self.assertGreater(result.expected_risk, 0)
    
    def test_optimize_portfolio_equal_weight(self):
        """Test equal weight portfolio."""
        result = self.optimizer.optimize_portfolio(
            self.expected_returns, self.cov_matrix, OptimizationObjective.EQUAL_WEIGHT
        )
        
        self.assertIsInstance(result, PortfolioResult)
        self.assertEqual(len(result.weights), 3)
        
        # All weights should be approximately equal
        for weight in result.weights.values():
            self.assertAlmostEqual(weight, 1.0/3.0, places=4)
    
    def test_optimize_portfolio_with_constraints(self):
        """Test optimization with constraints."""
        constraints = OptimizationConstraints(
            max_weight=0.4,
            min_weight=0.1
        )
        
        result = self.optimizer.optimize_portfolio(
            self.expected_returns, self.cov_matrix, OptimizationObjective.MAX_SHARPE, constraints
        )
        
        self.assertIsInstance(result, PortfolioResult)
        
        # Check weight constraints
        for weight in result.weights.values():
            self.assertGreaterEqual(weight, 0.1 - 1e-6)  # Allow for small numerical errors
            self.assertLessEqual(weight, 0.4 + 1e-6)
    
    def test_generate_efficient_frontier(self):
        """Test efficient frontier generation."""
        efficient_portfolios = self.optimizer.generate_efficient_frontier(
            self.expected_returns, self.cov_matrix, n_points=10
        )
        
        self.assertIsInstance(efficient_portfolios, list)
        self.assertGreater(len(efficient_portfolios), 0)
        
        # Check that portfolios are ordered by risk
        risks = [p.expected_risk for p in efficient_portfolios]
        self.assertEqual(risks, sorted(risks))
        
        # Check that all portfolios are valid
        for portfolio in efficient_portfolios:
            self.assertIsInstance(portfolio, PortfolioResult)
            self.assertAlmostEqual(sum(portfolio.weights.values()), 1.0, places=3)
    
    def test_black_litterman_optimization(self):
        """Test Black-Litterman optimization."""
        views = {'AAPL': 0.15, 'MSFT': 0.12}  # Expected returns
        view_confidence = {'AAPL': 0.8, 'MSFT': 0.6}  # Confidence levels
        
        result = self.optimizer.black_litterman_optimization(
            self.sample_price_data, views, view_confidence
        )
        
        self.assertIsInstance(result, PortfolioResult)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=4)
    
    def test_risk_budgeting_optimization(self):
        """Test risk budgeting optimization."""
        risk_budgets = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        
        result = self.optimizer.risk_budgeting_optimization(
            self.expected_returns, self.cov_matrix, risk_budgets
        )
        
        self.assertIsInstance(result, PortfolioResult)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=4)
        
        # Check that risk contributions are close to target budgets
        if result.risk_contributions:
            total_risk_contrib = sum(result.risk_contributions.values())
            normalized_risk_contrib = {
                k: v/total_risk_contrib for k, v in result.risk_contributions.items()
            }
            
            for symbol in risk_budgets:
                if symbol in normalized_risk_contrib:
                    self.assertAlmostEqual(
                        normalized_risk_contrib[symbol], 
                        risk_budgets[symbol], 
                        places=1  # Allow for some optimization error
                    )

if __name__ == '__main__':
    unittest.main()

