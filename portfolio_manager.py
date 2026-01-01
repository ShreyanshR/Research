"""
Portfolio Manager - Production-grade portfolio analysis system.
Designed for hedge fund-level code quality.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date
import pandas as pd
import numpy as np
import asyncio
from portfolio_backtest import SimplePortfolio, PortfolioAllocation
from assets import AssetClass
from pickle_utils import ensure_all_tickers_data


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    years: float
    leverage: float


@dataclass
class PositionInfo:
    """Information about a single position."""
    ticker: str
    weight: float
    allocation_pct: float
    total_return: float
    annualized_return: float
    market_value: float
    unrealized_pnl: float


class DataLoader:
    """Handles data loading from IBKR or cache."""
    
    def __init__(self, auto_download: bool = True):
        """
        Initialize data loader.
        
        Args:
            auto_download: If True, automatically download missing data
        """
        self.auto_download = auto_download
    
    async def load_tickers(
        self, 
        tickers: List[str], 
        years: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            years: Years of historical data to fetch
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        try:
            data = await ensure_all_tickers_data(
                tickers=tickers,
                years=years,
                force_refresh=False,
                update_if_old=False,
                auto_download=self.auto_download
            )
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load ticker data: {e}")


class ReturnsCalculator:
    """Calculates returns and aligns data."""
    
    @staticmethod
    def calculate_returns(
        data: Dict[str, pd.DataFrame],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        show_debug: bool = False
    ) -> pd.DataFrame:
        """
        Calculate returns for all tickers and align to common dates.
        
        Args:
            data: Dictionary of ticker to DataFrame
            start_date: Optional start date filter
            end_date: Optional end date filter
            show_debug: If True, print debug info about date ranges
            
        Returns:
            DataFrame with aligned returns
        """
        returns_dict = {}
        
        if show_debug:
            print(f"\nCalculating returns with date filter: {start_date} to {end_date}")
        
        for ticker, df in data.items():
            if df.empty:
                if show_debug:
                    print(f"  {ticker}: DataFrame is empty")
                continue
            
            # Find price column
            price_col = ReturnsCalculator._find_price_column(df, ticker)
            if price_col is None:
                if show_debug:
                    print(f"  {ticker}: No price column found")
                continue
            
            # Normalize timezone - check if it's a DatetimeIndex first
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            elif not isinstance(df.index, pd.DatetimeIndex):
                # Try to find a date column and set it as index
                for col in ['date', 'Date', 'DATE', 'time', 'Time', 'TIME']:
                    if col in df.columns:
                        df.index = pd.to_datetime(df[col])
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert('UTC').tz_localize(None)
                        break
            
            # Show original date range before filtering
            if show_debug:
                print(f"  {ticker}: Original data range: {df.index.min().date()} to {df.index.max().date()} ({len(df)} rows)")
            
            # Filter to date range
            original_len = len(df)
            if start_date:
                start_dt = pd.Timestamp(start_date)
                df = df[df.index >= start_dt]
            if end_date:
                end_dt = pd.Timestamp(end_date)
                df = df[df.index <= end_dt]
            
            if show_debug and original_len != len(df):
                print(f"  {ticker}: After filtering: {df.index.min().date()} to {df.index.max().date()} ({len(df)} rows)")
            
            # Calculate returns
            returns = df[price_col].pct_change()
            returns_dict[ticker] = returns
        
        if not returns_dict:
            raise ValueError("No valid returns calculated")
        
        # Align to common dates
        returns_df = pd.DataFrame(returns_dict)
        
        if show_debug:
            print(f"\nBefore alignment: {len(returns_df)} rows")
            for ticker in returns_df.columns:
                ticker_data = returns_df[ticker].dropna()
                if len(ticker_data) > 0:
                    print(f"  {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()} ({len(ticker_data)} rows)")
        
        common_dates = returns_df.dropna().index
        
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates found across tickers")
        
        if show_debug:
            print(f"\nAfter alignment (common dates): {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} rows)")
        
        return returns_df.loc[common_dates]
    
    @staticmethod
    def _find_price_column(df: pd.DataFrame, ticker: str) -> Optional[str]:
        """Find the price column in a DataFrame."""
        for col in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
            if col in df.columns:
                return col
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        
        return None


class PortfolioAnalyzer:
    """Analyzes portfolio performance and calculates metrics."""
    
    def __init__(self, returns_df: pd.DataFrame, weights: Dict[str, float], leverage: float = 1.0):
        """
        Initialize portfolio analyzer.
        
        Args:
            returns_df: DataFrame of returns for each ticker
            weights: Dictionary of ticker to weight
            leverage: Leverage multiplier
        """
        self.returns_df = returns_df
        self.weights = weights
        self.leverage = leverage
        self._portfolio_returns: Optional[pd.Series] = None
        self._calculate_portfolio_returns()
    
    def _calculate_portfolio_returns(self) -> None:
        """Calculate weighted portfolio returns with leverage."""
        portfolio_returns = pd.Series(0.0, index=self.returns_df.index)
        
        for ticker, weight in self.weights.items():
            if ticker in self.returns_df.columns:
                portfolio_returns += self.returns_df[ticker] * weight
        
        self._portfolio_returns = portfolio_returns * self.leverage
    
    @property
    def portfolio_returns(self) -> pd.Series:
        """Get portfolio returns series."""
        return self._portfolio_returns
    
    def calculate_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.
        
        Returns:
            PortfolioMetrics object
        """
        returns = self._portfolio_returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No returns data available")
        
        # Total and annualized return
        total_return = (1 + returns).prod() - 1
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            years=years,
            leverage=self.leverage
        )
    
    def get_position_info(self, initial_capital: float = 1.0) -> List[PositionInfo]:
        """
        Get detailed information for each position.
        
        Args:
            initial_capital: Initial capital for market value calculation
            
        Returns:
            List of PositionInfo objects
        """
        positions = []
        total_weight = sum(self.weights.values())
        years = (self.returns_df.index[-1] - self.returns_df.index[0]).days / 365.25
        
        for ticker, weight in self.weights.items():
            if ticker not in self.returns_df.columns:
                continue
            
            ticker_returns = self.returns_df[ticker]
            total_return = (1 + ticker_returns).prod() - 1
            annualized_return = (1 + total_return) ** (1 / years) - 1
            allocation_pct = (weight / total_weight * 100) if total_weight > 0 else 0
            
            # Calculate market value based on initial capital and weight
            market_value = weight * initial_capital
            unrealized_pnl = 0  # Would need cost basis for real P&L
            
            positions.append(PositionInfo(
                ticker=ticker,
                weight=weight,
                allocation_pct=allocation_pct,
                total_return=total_return,
                annualized_return=annualized_return,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl
            ))
        
        return positions
    
    def get_cumulative_returns(self) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """
        Get cumulative returns for portfolio and individual tickers.
        
        Returns:
            Tuple of (portfolio cumulative returns, dict of ticker cumulative returns)
        """
        portfolio_cumulative = (1 + self._portfolio_returns).cumprod()
        
        individual_cumulative = {}
        for ticker in self.returns_df.columns:
            individual_cumulative[ticker] = (1 + self.returns_df[ticker]).cumprod()
        
        return portfolio_cumulative, individual_cumulative


class PortfolioManager:
    """Main portfolio management class - orchestrates all components."""
    
    def __init__(self, auto_download: bool = True):
        """
        Initialize portfolio manager.
        
        Args:
            auto_download: If True, automatically download missing data
        """
        self.data_loader = DataLoader(auto_download=auto_download)
        self.analyzer: Optional[PortfolioAnalyzer] = None
        self.data: Optional[Dict[str, pd.DataFrame]] = None
    
    async def load_portfolio(
        self,
        tickers: List[str],
        years: int = 5
    ) -> None:
        """
        Load portfolio data.
        
        Args:
            tickers: List of ticker symbols
            years: Years of historical data
        """
        self.data = await self.data_loader.load_tickers(tickers, years)
    
    def analyze_portfolio(
        self,
        weights: Optional[Dict[str, float]] = None,
        leverage: float = 1.0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        show_data_info: bool = True
    ) -> PortfolioAnalyzer:
        """
        Analyze portfolio with given weights and leverage.
        
        Args:
            weights: Dictionary of ticker to weight (None for equal weights)
            leverage: Leverage multiplier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            PortfolioAnalyzer instance
        """
        if self.data is None:
            raise ValueError("Portfolio data not loaded. Call load_portfolio() first.")
        
        # Calculate returns
        returns_df = ReturnsCalculator.calculate_returns(
            self.data,
            start_date=start_date,
            end_date=end_date,
            show_debug=show_data_info
        )
        
        # Default to equal weights if not provided
        if weights is None:
            weights = {ticker: 1.0 / len(returns_df.columns) for ticker in returns_df.columns}
        
        # Create analyzer
        self.analyzer = PortfolioAnalyzer(
            returns_df=returns_df,
            weights=weights,
            leverage=leverage
        )
        
        # Show data info if requested
        if show_data_info:
            print(f"\nData Timeline:")
            print(f"  Start Date: {returns_df.index[0].date()}")
            print(f"  End Date: {returns_df.index[-1].date()}")
            print(f"  Total Days: {len(returns_df)}")
            print(f"  Tickers: {', '.join(returns_df.columns)}")
            for ticker in returns_df.columns:
                ticker_data = returns_df[ticker].dropna()
                if len(ticker_data) > 0:
                    print(f"    {ticker}: {ticker_data.index[0].date()} to {ticker_data.index[-1].date()} ({len(ticker_data)} days)")
        
        return self.analyzer
    
    def get_metrics(self) -> PortfolioMetrics:
        """Get portfolio metrics."""
        if self.analyzer is None:
            raise ValueError("Portfolio not analyzed. Call analyze_portfolio() first.")
        return self.analyzer.calculate_metrics()
    
    def get_positions(self, initial_capital: float = 1.0) -> List[PositionInfo]:
        """
        Get position information.
        
        Args:
            initial_capital: Initial capital for market value calculation
        """
        if self.analyzer is None:
            raise ValueError("Portfolio not analyzed. Call analyze_portfolio() first.")
        return self.analyzer.get_position_info(initial_capital=initial_capital)
    
    def get_returns_data(self) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """Get cumulative returns data for visualization."""
        if self.analyzer is None:
            raise ValueError("Portfolio not analyzed. Call analyze_portfolio() first.")
        return self.analyzer.get_cumulative_returns()
