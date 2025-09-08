import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    max_weight: float = 0.3  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    max_assets: Optional[int] = None  # Maximum number of assets
    sector_limits: Optional[Dict[str, float]] = None  # Sector exposure limits
    target_return: Optional[float] = None  # Target return for efficient frontier
    target_risk: Optional[float] = None  # Target risk level

@dataclass
class PortfolioResult:
    """Portfolio optimization result."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_success: bool
    optimization_message: str
    risk_contributions: Optional[Dict[str, float]] = None

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple objectives and constraints."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_expected_returns(self, price_data: Dict[str, pd.DataFrame], 
                                 method: str = "historical") -> pd.Series:
        """Calculate expected returns for assets."""
        returns_data = {}
        
        for symbol, df in price_data.items():
            if len(df) > 1:
                returns = df['close'].pct_change().dropna()
                
                if method == "historical":
                    # Simple historical mean
                    expected_return = returns.mean() * 252  # Annualized
                elif method == "ewma":
                    # Exponentially weighted moving average
                    expected_return = returns.ewm(span=60).mean().iloc[-1] * 252
                else:
                    expected_return = returns.mean() * 252
                
                returns_data[symbol] = expected_return
        
        return pd.Series(returns_data)
    
    def calculate_covariance_matrix(self, price_data: Dict[str, pd.DataFrame], 
                                  method: str = "historical") -> pd.DataFrame:
        """Calculate covariance matrix for assets."""
        # Align all price series
        aligned_prices = pd.DataFrame()
        
        for symbol, df in price_data.items():
            aligned_prices[symbol] = df['close']
        
        # Calculate returns
        returns = aligned_prices.pct_change().dropna()
        
        if method == "historical":
            # Simple historical covariance
            cov_matrix = returns.cov() * 252  # Annualized
        elif method == "ewma":
            # Exponentially weighted covariance
            cov_matrix = returns.ewm(span=60).cov().iloc[-len(returns.columns):] * 252
        elif method == "shrinkage":
            # Ledoit-Wolf shrinkage estimator
            cov_matrix = self._ledoit_wolf_shrinkage(returns) * 252
        else:
            cov_matrix = returns.cov() * 252
        
        return cov_matrix
    
    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage estimator for covariance matrix."""
        n, p = returns.shape
        
        # Sample covariance matrix
        sample_cov = returns.cov()
        
        # Shrinkage target (identity matrix scaled by average variance)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)
        
        # Calculate shrinkage intensity
        X_centered = returns - returns.mean()
        
        # Frobenius norm of sample covariance minus target
        d_squared = np.sum((sample_cov.values - target) ** 2)
        
        # Calculate shrinkage intensity (simplified version)
        shrinkage = min(1.0, 0.1)  # Use fixed shrinkage for simplicity
        
        # Shrunk covariance matrix
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov.values
        
        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
    
    def optimize_portfolio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                          objective: OptimizationObjective, 
                          constraints: OptimizationConstraints = None) -> PortfolioResult:
        """Optimize portfolio based on objective and constraints."""
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        n_assets = len(expected_returns)
        symbols = expected_returns.index.tolist()
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Bounds for each asset
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Constraints
        constraint_list = []
        
        # Weights sum to 1
        constraint_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        })
        
        # Target return constraint (if specified)
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, expected_returns.values) - constraints.target_return
            })
        
        # Target risk constraint (if specified)
        if constraints.target_risk is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(cov_matrix.values, x))) - constraints.target_risk
            })
        
        # Define objective function
        if objective == OptimizationObjective.MAX_SHARPE:
            def objective_func(x):
                portfolio_return = np.dot(x, expected_returns.values)
                portfolio_risk = np.sqrt(np.dot(x, np.dot(cov_matrix.values, x)))
                return -(portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else -1e6
        
        elif objective == OptimizationObjective.MIN_VARIANCE:
            def objective_func(x):
                return np.dot(x, np.dot(cov_matrix.values, x))
        
        elif objective == OptimizationObjective.MAX_RETURN:
            def objective_func(x):
                return -np.dot(x, expected_returns.values)
        
        elif objective == OptimizationObjective.RISK_PARITY:
            def objective_func(x):
                portfolio_risk = np.sqrt(np.dot(x, np.dot(cov_matrix.values, x)))
                risk_contributions = (x * np.dot(cov_matrix.values, x)) / portfolio_risk
                target_risk_contrib = portfolio_risk / n_assets
                return np.sum((risk_contributions - target_risk_contrib) ** 2)
        
        elif objective == OptimizationObjective.EQUAL_WEIGHT:
            # No optimization needed
            weights = np.array([1.0 / n_assets] * n_assets)
            return self._create_portfolio_result(weights, symbols, expected_returns, cov_matrix, True, "Equal weight portfolio")
        
        # Run optimization
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = minimize(
                    objective_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraint_list,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
            
            if result.success:
                weights = result.x
                # Clean up small weights
                weights[weights < 1e-6] = 0
                weights = weights / np.sum(weights)  # Renormalize
                
                return self._create_portfolio_result(
                    weights, symbols, expected_returns, cov_matrix, True, "Optimization successful"
                )
            else:
                logger.warning(f"Optimization failed: {result.message}")
                # Fall back to equal weights
                weights = np.array([1.0 / n_assets] * n_assets)
                return self._create_portfolio_result(
                    weights, symbols, expected_returns, cov_matrix, False, f"Optimization failed: {result.message}"
                )
        
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            # Fall back to equal weights
            weights = np.array([1.0 / n_assets] * n_assets)
            return self._create_portfolio_result(
                weights, symbols, expected_returns, cov_matrix, False, f"Optimization error: {str(e)}"
            )
    
    def _create_portfolio_result(self, weights: np.ndarray, symbols: List[str], 
                               expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                               success: bool, message: str) -> PortfolioResult:
        """Create portfolio result object."""
        
        # Create weights dictionary
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, weights) if weight > 1e-6}
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns.values)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Calculate risk contributions
        risk_contributions = None
        if portfolio_risk > 0:
            marginal_risk = np.dot(cov_matrix.values, weights) / portfolio_risk
            risk_contributions = {
                symbol: weight * marginal_risk[i] 
                for i, (symbol, weight) in enumerate(zip(symbols, weights))
                if weight > 1e-6
            }
        
        return PortfolioResult(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            optimization_success=success,
            optimization_message=message,
            risk_contributions=risk_contributions
        )
    
    def generate_efficient_frontier(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                                  n_points: int = 50, constraints: OptimizationConstraints = None) -> List[PortfolioResult]:
        """Generate efficient frontier portfolios."""
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Find minimum and maximum return portfolios
        min_var_portfolio = self.optimize_portfolio(
            expected_returns, cov_matrix, OptimizationObjective.MIN_VARIANCE, constraints
        )
        
        max_ret_constraints = OptimizationConstraints(
            max_weight=constraints.max_weight,
            min_weight=constraints.min_weight
        )
        max_ret_portfolio = self.optimize_portfolio(
            expected_returns, cov_matrix, OptimizationObjective.MAX_RETURN, max_ret_constraints
        )
        
        # Generate target returns between min and max
        min_return = min_var_portfolio.expected_return
        max_return = max_ret_portfolio.expected_return
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            target_constraints = OptimizationConstraints(
                max_weight=constraints.max_weight,
                min_weight=constraints.min_weight,
                target_return=target_return
            )
            
            portfolio = self.optimize_portfolio(
                expected_returns, cov_matrix, OptimizationObjective.MIN_VARIANCE, target_constraints
            )
            
            if portfolio.optimization_success:
                efficient_portfolios.append(portfolio)
        
        return efficient_portfolios
    
    def black_litterman_optimization(self, price_data: Dict[str, pd.DataFrame],
                                   views: Dict[str, float], view_confidence: Dict[str, float],
                                   market_cap_weights: Optional[Dict[str, float]] = None) -> PortfolioResult:
        """Black-Litterman portfolio optimization with investor views."""
        
        # Calculate historical returns and covariance
        expected_returns = self.calculate_expected_returns(price_data, method="historical")
        cov_matrix = self.calculate_covariance_matrix(price_data, method="historical")
        
        # Market capitalization weights (if not provided, use equal weights)
        if market_cap_weights is None:
            market_cap_weights = {symbol: 1.0/len(expected_returns) for symbol in expected_returns.index}
        
        # Convert to arrays
        symbols = expected_returns.index.tolist()
        w_market = np.array([market_cap_weights.get(symbol, 0) for symbol in symbols])
        
        # Implied equilibrium returns (reverse optimization)
        risk_aversion = 3.0  # Typical risk aversion parameter
        pi = risk_aversion * np.dot(cov_matrix.values, w_market)
        
        # Views matrix and view returns
        view_symbols = list(views.keys())
        P = np.zeros((len(view_symbols), len(symbols)))
        Q = np.zeros(len(view_symbols))
        
        for i, symbol in enumerate(view_symbols):
            if symbol in symbols:
                symbol_idx = symbols.index(symbol)
                P[i, symbol_idx] = 1.0
                Q[i] = views[symbol]
        
        # View uncertainty (Omega matrix)
        omega = np.diag([1.0 / view_confidence.get(symbol, 1.0) for symbol in view_symbols])
        
        # Tau parameter (scales the uncertainty of the prior)
        tau = 1.0 / len(price_data[list(price_data.keys())[0]])
        
        # Black-Litterman formula
        try:
            # M1 = inv(tau * Sigma)
            M1 = np.linalg.inv(tau * cov_matrix.values)
            
            # M2 = P' * inv(Omega) * P
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            
            # M3 = inv(tau * Sigma) * pi + P' * inv(Omega) * Q
            M3 = np.dot(M1, pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            
            # New expected returns
            mu_bl = np.dot(np.linalg.inv(M1 + M2), M3)
            
            # New covariance matrix
            sigma_bl = np.linalg.inv(M1 + M2)
            
            # Convert back to pandas
            mu_bl_series = pd.Series(mu_bl, index=symbols)
            sigma_bl_df = pd.DataFrame(sigma_bl, index=symbols, columns=symbols)
            
            # Optimize portfolio with Black-Litterman inputs
            return self.optimize_portfolio(
                mu_bl_series, sigma_bl_df, OptimizationObjective.MAX_SHARPE
            )
        
        except np.linalg.LinAlgError as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            # Fall back to standard optimization
            return self.optimize_portfolio(
                expected_returns, cov_matrix, OptimizationObjective.MAX_SHARPE
            )
    
    def risk_budgeting_optimization(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                                  risk_budgets: Dict[str, float]) -> PortfolioResult:
        """Optimize portfolio based on risk budgeting approach."""
        
        symbols = expected_returns.index.tolist()
        n_assets = len(symbols)
        
        # Normalize risk budgets
        total_budget = sum(risk_budgets.values())
        if total_budget > 0:
            risk_budgets = {k: v/total_budget for k, v in risk_budgets.items()}
        else:
            # Equal risk budgets
            risk_budgets = {symbol: 1.0/n_assets for symbol in symbols}
        
        # Initial guess
        x0 = np.array([risk_budgets.get(symbol, 1.0/n_assets) for symbol in symbols])
        
        # Objective function: minimize difference between actual and target risk contributions
        def objective_func(x):
            portfolio_risk = np.sqrt(np.dot(x, np.dot(cov_matrix.values, x)))
            if portfolio_risk == 0:
                return 1e6
            
            # Actual risk contributions
            marginal_risk = np.dot(cov_matrix.values, x) / portfolio_risk
            risk_contributions = x * marginal_risk
            
            # Target risk contributions
            target_contributions = np.array([risk_budgets.get(symbol, 1.0/n_assets) * portfolio_risk 
                                           for symbol in symbols])
            
            return np.sum((risk_contributions - target_contributions) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.001, 1.0) for _ in range(n_assets)]
        
        try:
            result = minimize(
                objective_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                return self._create_portfolio_result(
                    weights, symbols, expected_returns, cov_matrix, True, "Risk budgeting optimization successful"
                )
            else:
                # Fall back to equal weights
                weights = np.array([1.0 / n_assets] * n_assets)
                return self._create_portfolio_result(
                    weights, symbols, expected_returns, cov_matrix, False, f"Risk budgeting failed: {result.message}"
                )
        
        except Exception as e:
            logger.error(f"Risk budgeting optimization error: {e}")
            weights = np.array([1.0 / n_assets] * n_assets)
            return self._create_portfolio_result(
                weights, symbols, expected_returns, cov_matrix, False, f"Risk budgeting error: {str(e)}"
            )

# Global instance
portfolio_optimizer = PortfolioOptimizer()

