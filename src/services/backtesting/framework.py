import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from src.services.providers import fmp_provider as fmp
from src.core.pti import filter_series_asof

logger = logging.getLogger(__name__)

class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"

@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    shares: float
    price: float
    value: float
    weight: float
    entry_date: datetime
    
@dataclass
class Transaction:
    """Portfolio transaction."""
    date: datetime
    symbol: str
    action: str  # "buy", "sell"
    shares: float
    price: float
    value: float
    commission: float = 0.0

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    beta: float
    alpha: float
    information_ratio: float

class BacktestingFramework:
    """Comprehensive backtesting framework for equity strategies."""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # Commission as percentage of trade value
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.transactions: List[Transaction] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.benchmark_data: Optional[pd.DataFrame] = None
        
    def load_price_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load historical price data for symbols."""
        price_data = {}
        
        for symbol in symbols:
            try:
                data = fmp.historical_prices(symbol, start=start_date, end=end_date, limit=5000)
                if data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df.set_index('date', inplace=True)
                    price_data[symbol] = df
                else:
                    logger.warning(f"No price data found for {symbol}")
            except Exception as e:
                logger.error(f"Error loading price data for {symbol}: {e}")
        
        return price_data
    
    def load_benchmark_data(self, benchmark_symbol: str = "SPY", start_date: str = None, end_date: str = None):
        """Load benchmark data for performance comparison."""
        try:
            data = fmp.historical_prices(benchmark_symbol, start=start_date, end=end_date, limit=5000)
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df.set_index('date', inplace=True)
                self.benchmark_data = df
                logger.info(f"Loaded benchmark data for {benchmark_symbol}")
            else:
                logger.warning(f"No benchmark data found for {benchmark_symbol}")
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
    
    def execute_trade(self, symbol: str, action: str, shares: float, price: float, date: datetime):
        """Execute a trade (buy or sell)."""
        trade_value = shares * price
        commission_cost = trade_value * self.commission
        
        if action == "buy":
            total_cost = trade_value + commission_cost
            if self.cash >= total_cost:
                self.cash -= total_cost
                
                if symbol in self.positions:
                    # Add to existing position
                    old_pos = self.positions[symbol]
                    new_shares = old_pos.shares + shares
                    new_value = old_pos.value + trade_value
                    new_price = new_value / new_shares
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=new_shares,
                        price=new_price,
                        value=new_value,
                        weight=0,  # Will be calculated later
                        entry_date=old_pos.entry_date
                    )
                else:
                    # New position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=shares,
                        price=price,
                        value=trade_value,
                        weight=0,  # Will be calculated later
                        entry_date=date
                    )
                
                # Record transaction
                self.transactions.append(Transaction(
                    date=date,
                    symbol=symbol,
                    action=action,
                    shares=shares,
                    price=price,
                    value=trade_value,
                    commission=commission_cost
                ))
                
                return True
            else:
                logger.warning(f"Insufficient cash for {action} {shares} shares of {symbol}")
                return False
        
        elif action == "sell":
            if symbol in self.positions and self.positions[symbol].shares >= shares:
                self.cash += trade_value - commission_cost
                
                old_pos = self.positions[symbol]
                remaining_shares = old_pos.shares - shares
                
                if remaining_shares > 0:
                    # Partial sell
                    remaining_value = remaining_shares * price
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=remaining_shares,
                        price=price,
                        value=remaining_value,
                        weight=0,  # Will be calculated later
                        entry_date=old_pos.entry_date
                    )
                else:
                    # Full sell
                    del self.positions[symbol]
                
                # Record transaction
                self.transactions.append(Transaction(
                    date=date,
                    symbol=symbol,
                    action=action,
                    shares=shares,
                    price=price,
                    value=trade_value,
                    commission=commission_cost
                ))
                
                return True
            else:
                logger.warning(f"Insufficient shares for {action} {shares} shares of {symbol}")
                return False
        
        return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float], date: datetime):
        """Update portfolio value based on current prices."""
        total_position_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position.shares * current_prices[symbol]
                position.value = current_value
                position.price = current_prices[symbol]
                total_position_value += current_value
        
        self.portfolio_value = self.cash + total_position_value
        
        # Update position weights
        for position in self.positions.values():
            position.weight = position.value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Record portfolio state
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': total_position_value,
            'positions': {symbol: {
                'shares': pos.shares,
                'price': pos.price,
                'value': pos.value,
                'weight': pos.weight
            } for symbol, pos in self.positions.items()}
        })
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], current_prices: Dict[str, float], 
                           date: datetime, min_trade_size: float = 100):
        """Rebalance portfolio to target weights."""
        if not target_weights:
            return
        
        # Normalize weights to sum to 1
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            target_weights = {k: v/total_weight for k, v in target_weights.items()}
        
        # Calculate target values
        target_values = {symbol: weight * self.portfolio_value 
                        for symbol, weight in target_weights.items()}
        
        # Calculate current values
        current_values = {symbol: pos.value for symbol, pos in self.positions.items()}
        
        # Add symbols not currently held
        for symbol in target_weights:
            if symbol not in current_values:
                current_values[symbol] = 0
        
        # Calculate trades needed
        trades = []
        for symbol in target_weights:
            if symbol not in current_prices:
                logger.warning(f"No price data for {symbol} on {date}")
                continue
            
            target_value = target_values[symbol]
            current_value = current_values.get(symbol, 0)
            trade_value = target_value - current_value
            
            if abs(trade_value) > min_trade_size:
                shares = abs(trade_value) / current_prices[symbol]
                action = "buy" if trade_value > 0 else "sell"
                trades.append((symbol, action, shares, current_prices[symbol]))
        
        # Execute trades
        for symbol, action, shares, price in trades:
            self.execute_trade(symbol, action, shares, price, date)
    
    def run_backtest(self, strategy_func: Callable, price_data: Dict[str, pd.DataFrame], 
                     start_date: str, end_date: str, rebalance_freq: RebalanceFrequency = RebalanceFrequency.MONTHLY,
                     benchmark_symbol: str = "SPY") -> Dict[str, Any]:
        """Run a complete backtest."""
        
        # Load benchmark data
        self.load_benchmark_data(benchmark_symbol, start_date, end_date)
        
        # Create date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get all available dates from price data
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        
        all_dates = sorted([d for d in all_dates if start_dt <= d <= end_dt])
        
        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates, rebalance_freq)
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Rebalancing {len(rebalance_dates)} times")
        
        # Run backtest
        for i, date in enumerate(all_dates):
            # Get current prices
            current_prices = {}
            for symbol, df in price_data.items():
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'close']
            
            # Update portfolio value
            self.update_portfolio_value(current_prices, date)
            
            # Check if rebalancing is needed
            if date in rebalance_dates:
                # Get strategy signals
                try:
                    # Create point-in-time data for strategy
                    pti_data = {}
                    for symbol, df in price_data.items():
                        pti_df = df[df.index <= date]
                        if not pti_df.empty:
                            pti_data[symbol] = pti_df
                    
                    target_weights = strategy_func(pti_data, date)
                    
                    if target_weights:
                        self.rebalance_portfolio(target_weights, current_prices, date)
                        logger.info(f"Rebalanced on {date}: {target_weights}")
                
                except Exception as e:
                    logger.error(f"Error running strategy on {date}: {e}")
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        return {
            'performance_metrics': performance,
            'portfolio_history': self.portfolio_history,
            'transactions': [
                {
                    'date': t.date.isoformat(),
                    'symbol': t.symbol,
                    'action': t.action,
                    'shares': t.shares,
                    'price': t.price,
                    'value': t.value,
                    'commission': t.commission
                } for t in self.transactions
            ],
            'final_portfolio_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'benchmark_comparison': self._compare_to_benchmark() if self.benchmark_data is not None else None
        }
    
    def _get_rebalance_dates(self, all_dates: List[datetime], freq: RebalanceFrequency) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        rebalance_dates = []
        
        if freq == RebalanceFrequency.MONTHLY:
            current_month = None
            for date in all_dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
        
        elif freq == RebalanceFrequency.QUARTERLY:
            current_quarter = None
            for date in all_dates:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
        
        elif freq == RebalanceFrequency.ANNUAL:
            current_year = None
            for date in all_dates:
                if current_year != date.year:
                    rebalance_dates.append(date)
                    current_year = date.year
        
        return rebalance_dates
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Create returns series
        values = [h['portfolio_value'] for h in self.portfolio_history]
        dates = [h['date'] for h in self.portfolio_history]
        
        returns = pd.Series(values, index=dates).pct_change().dropna()
        
        # Total return
        total_return = (values[-1] - values[0]) / values[0]
        
        # Annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Beta and alpha (vs benchmark)
        beta, alpha = self._calculate_beta_alpha(returns)
        
        # Information ratio
        if self.benchmark_data is not None:
            benchmark_returns = self._get_benchmark_returns()
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                excess_returns = returns - benchmark_returns
                information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            else:
                information_ratio = 0
        else:
            information_ratio = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )
    
    def _calculate_beta_alpha(self, returns: pd.Series) -> Tuple[float, float]:
        """Calculate beta and alpha vs benchmark."""
        if self.benchmark_data is None:
            return 0.0, 0.0
        
        benchmark_returns = self._get_benchmark_returns()
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0
        
        # Align returns
        aligned_returns = returns.reindex(benchmark_returns.index).dropna()
        aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
        
        if len(aligned_returns) < 2 or len(aligned_benchmark) < 2:
            return 0.0, 0.0
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (annualized)
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        risk_free_rate = 0.02
        alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        return beta, alpha
    
    def _get_benchmark_returns(self) -> Optional[pd.Series]:
        """Get benchmark returns series."""
        if self.benchmark_data is None:
            return None
        
        return self.benchmark_data['close'].pct_change().dropna()
    
    def _compare_to_benchmark(self) -> Dict[str, Any]:
        """Compare portfolio performance to benchmark."""
        if self.benchmark_data is None:
            return None
        
        # Get portfolio and benchmark values for comparison
        portfolio_dates = [h['date'] for h in self.portfolio_history]
        portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
        
        # Align benchmark data
        benchmark_aligned = []
        for date in portfolio_dates:
            if date in self.benchmark_data.index:
                benchmark_aligned.append(self.benchmark_data.loc[date, 'close'])
            else:
                # Find closest date
                closest_date = min(self.benchmark_data.index, key=lambda x: abs((x - date).days))
                benchmark_aligned.append(self.benchmark_data.loc[closest_date, 'close'])
        
        if not benchmark_aligned:
            return None
        
        # Normalize to same starting value
        benchmark_normalized = [v * (self.initial_capital / benchmark_aligned[0]) for v in benchmark_aligned]
        
        # Calculate returns
        portfolio_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        benchmark_return = (benchmark_normalized[-1] - benchmark_normalized[0]) / benchmark_normalized[0]
        
        return {
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'excess_return': portfolio_return - benchmark_return,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_normalized,
            'dates': [d.isoformat() for d in portfolio_dates]
        }

# Example strategy functions
def momentum_strategy(price_data: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
    """Simple momentum strategy - buy top performers over last 3 months."""
    lookback_days = 90
    cutoff_date = current_date - timedelta(days=lookback_days)
    
    momentum_scores = {}
    
    for symbol, df in price_data.items():
        recent_data = df[df.index >= cutoff_date]
        if len(recent_data) >= 20:  # Need sufficient data
            momentum = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
            momentum_scores[symbol] = momentum
    
    if not momentum_scores:
        return {}
    
    # Select top 5 performers
    sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [symbol for symbol, score in sorted_symbols[:5] if score > 0]
    
    if not top_symbols:
        return {}
    
    # Equal weight
    weight = 1.0 / len(top_symbols)
    return {symbol: weight for symbol in top_symbols}

def mean_reversion_strategy(price_data: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
    """Simple mean reversion strategy - buy underperformers."""
    lookback_days = 30
    cutoff_date = current_date - timedelta(days=lookback_days)
    
    reversion_scores = {}
    
    for symbol, df in price_data.items():
        recent_data = df[df.index >= cutoff_date]
        if len(recent_data) >= 10:
            # Calculate deviation from moving average
            ma = recent_data['close'].mean()
            current_price = recent_data['close'].iloc[-1]
            deviation = (current_price - ma) / ma
            reversion_scores[symbol] = -deviation  # Negative deviation = underperformer
    
    if not reversion_scores:
        return {}
    
    # Select top 5 underperformers (highest negative deviation)
    sorted_symbols = sorted(reversion_scores.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [symbol for symbol, score in sorted_symbols[:5] if score > 0]
    
    if not top_symbols:
        return {}
    
    # Equal weight
    weight = 1.0 / len(top_symbols)
    return {symbol: weight for symbol in top_symbols}

