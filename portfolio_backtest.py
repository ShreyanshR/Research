"""
Simple Portfolio Backtest Script

This script implements the portfolio allocation strategy from ideas.ipynb:
- 25% Equity ETF (SPY)
- 35% AI & Tech (25% AI, 10% Value) 
- 15% TLT & Bonds
- 10% Crypto
- 10% Gold & Silver
- 10% Speculation

Target leverage: 2x-3x
Target volatility: 15-20%
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional
from datetime import date
from dataclasses import dataclass
from assets import AssetClass, Position


@dataclass
class PortfolioAllocation:
    """Simple portfolio allocation definition"""
    ticker: str
    weight: float
    asset_class: AssetClass
    sector: Optional[str] = None
    strategy: Optional[str] = None


class SimplePortfolio:
    """
    Simple portfolio backtest class.
    Loads data, applies weights, calculates returns.
    """
    
    def __init__(
        self,
        allocations: List[PortfolioAllocation],
        leverage: float = 1.0,
        target_vol: Optional[float] = 0.20,
        backtest_years: int = 5,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        auto_download: bool = True
    ):
        """
        Initialize portfolio with allocations.
        
        Args:
            allocations: List of PortfolioAllocation objects
            leverage: Portfolio leverage multiplier (default: 1.0 = no leverage)
            target_vol: Target annualized volatility (optional, for scaling)
            backtest_years: Number of years of historical data to use (default: 5)
            start_date: Optional start date for backtest (overrides backtest_years if provided)
            end_date: Optional end date for backtest (default: today)
            auto_download: If True, automatically download missing data from IBKR (default: True)
        """
        self.allocations = allocations
        self.leverage = leverage
        self.target_vol = target_vol
        self.backtest_years = backtest_years
        self.start_date = start_date
        self.end_date = end_date if end_date else date.today()
        self.auto_download = auto_download
        self.data: Dict[str, pd.DataFrame] = {}
        self.returns: Optional[pd.DataFrame] = None
        self.portfolio_returns: Optional[pd.Series] = None
        
    async def load_data(self, tickers: Optional[List[str]] = None, force_refresh: bool = False) -> None:
        """
        Load price data for all tickers in allocations.
        Automatically downloads missing data and updates old data if auto_download is True.
        
        Args:
            tickers: Optional list of specific tickers to load (if None, loads all from allocations)
            force_refresh: If True, force re-download of all data (default: False)
        """
        if tickers is None:
            tickers = [alloc.ticker for alloc in self.allocations]
        
        print(f"Loading data for tickers: {tickers}")
        
        # Calculate years needed
        if self.start_date:
            years_needed = (self.end_date - self.start_date).days / 365.25
            years_needed = max(int(years_needed) + 1, self.backtest_years)
        else:
            years_needed = self.backtest_years
        
        # Use simple function to ensure all tickers have data
        from pickle_utils import ensure_all_tickers_data
        print(f"Ensuring all tickers have data...")
        self.data = await ensure_all_tickers_data(
            tickers=tickers,
            years=years_needed,
            force_refresh=force_refresh,
            update_if_old=self.auto_download,  # Only update if auto_download is enabled
            auto_download=self.auto_download  # Only download if auto_download is enabled
        )
        
        # Filter data to backtest period
        self._filter_to_backtest_period()
        
        print(f"\nSuccessfully loaded data for {len(self.data)} tickers")
        for ticker, df in self.data.items():
            if not df.empty:
                date_range = f"{df.index.min().date()} to {df.index.max().date()}"
                print(f"  {ticker}: {len(df)} rows, {date_range}")
    
    def _filter_to_backtest_period(self) -> None:
        """Filter all data to the backtest period (start_date to end_date)."""
        if not self.data:
            return
        
        # Determine start date
        if self.start_date:
            start_dt = pd.Timestamp(self.start_date)
        else:
            # Use end_date - backtest_years
            end_dt = pd.Timestamp(self.end_date)
            start_dt = end_dt - pd.DateOffset(years=self.backtest_years)
        
        end_dt = pd.Timestamp(self.end_date)
        
        print(f"\nFiltering data to backtest period: {start_dt.date()} to {end_dt.date()}")
        
        # Filter each dataframe
        for ticker, df in self.data.items():
            if df.empty:
                continue
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to find a date column
                date_col = None
                for col in ['date', 'Date', 'DATE', 'time', 'Time', 'TIME']:
                    if col in df.columns:
                        date_col = col
                        df.index = pd.to_datetime(df[date_col])
                        break
                if date_col is None:
                    print(f"Warning: {ticker} has no datetime index or date column")
                    continue
            
            # Handle timezone-aware vs timezone-naive comparison
            # Make comparison timestamps match the index timezone
            if df.index.tz is not None:
                # Index is timezone-aware, make comparison timestamps timezone-aware too
                start_dt_compare = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt
                end_dt_compare = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt
                # Convert to same timezone as index
                start_dt_compare = start_dt_compare.tz_convert(df.index.tz)
                end_dt_compare = end_dt_compare.tz_convert(df.index.tz)
            else:
                # Both are timezone-naive
                start_dt_compare = start_dt
                end_dt_compare = end_dt
            
            # Filter to date range
            mask = (df.index >= start_dt_compare) & (df.index <= end_dt_compare)
            filtered_df = df[mask].copy()
            
            if filtered_df.empty:
                print(f"Warning: {ticker} has no data in backtest period")
            else:
                self.data[ticker] = filtered_df
                print(f"  {ticker}: {len(df)} -> {len(filtered_df)} rows after filtering")
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns for each asset.
        Aligns all returns to the same date index (intersection of all dates).
        
        Returns:
            DataFrame with daily returns for each ticker, aligned to common dates
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        returns_dict = {}
        for ticker, df in self.data.items():
            if df.empty:
                print(f"Warning: {ticker} has no data, skipping")
                continue
                
            # Assume the price column is named 'close' or 'Close' or similar
            # Try common column names
            price_col = None
            for col in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                # If no close column found, use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_col = numeric_cols[0]
                    print(f"Warning: Using '{price_col}' as price column for {ticker}")
                else:
                    raise ValueError(f"No numeric column found for {ticker}")
            
            # Calculate daily returns
            returns = df[price_col].pct_change()
            
            # Normalize index to timezone-naive for consistent joining
            if returns.index.tz is not None:
                returns.index = returns.index.tz_convert('UTC').tz_localize(None)
            
            returns_dict[ticker] = returns
        
        if not returns_dict:
            raise ValueError("No valid returns calculated. Check your data.")
        
        # Align all returns to common date index (intersection)
        # All indices are now timezone-naive, so they can be joined
        self.returns = pd.DataFrame(returns_dict)
        
        # Find common date range (where all tickers have data)
        common_dates = self.returns.dropna().index
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates found across all tickers. Check date ranges.")
        
        self.returns = self.returns.loc[common_dates]
        
        print(f"\nAligned returns to common date range: {common_dates[0].date()} to {common_dates[-1].date()}")
        print(f"Total trading days: {len(common_dates)}")
        
        return self.returns
    
    def backtest(self) -> pd.Series:
        """
        Run backtest: calculate weighted portfolio returns.
        
        Returns:
            Series of daily portfolio returns
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Create weight dictionary
        weights = {alloc.ticker: alloc.weight for alloc in self.allocations}
        
        # Align returns with weights (only use tickers we have weights for)
        available_tickers = [t for t in weights.keys() if t in self.returns.columns]
        if len(available_tickers) == 0:
            raise ValueError("No matching tickers between allocations and data")
        
        # Normalize weights to sum to 1.0 (in case some tickers are missing)
        total_weight = sum(weights[t] for t in available_tickers)
        normalized_weights = {t: weights[t] / total_weight for t in available_tickers}
        
        print(f"\nPortfolio Weights (normalized):")
        for ticker, weight in normalized_weights.items():
            print(f"  {ticker}: {weight:.1%}")
        
        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=self.returns.index)
        for ticker, weight in normalized_weights.items():
            portfolio_returns += self.returns[ticker] * weight
        
        # Apply leverage
        portfolio_returns = portfolio_returns * self.leverage
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.portfolio_returns is None:
            raise ValueError("Run backtest() first")
        
        returns = self.portfolio_returns.dropna()
        
        # Annualized return
        total_return = (1 + returns).prod() - 1
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'years': years,
            'total_trades': len(returns)
        }
        
        return metrics
    
    def print_summary(self) -> None:
        """Print a nice summary of the backtest results."""
        if self.portfolio_returns is None:
            print("No backtest results. Run backtest() first.")
            return
        
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("PORTFOLIO BACKTEST SUMMARY")
        print("="*60)
        print(f"Period: {self.portfolio_returns.index[0].date()} to {self.portfolio_returns.index[-1].date()}")
        print(f"Years: {metrics['years']:.2f}")
        print(f"\nPerformance:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"\nPortfolio Settings:")
        print(f"  Leverage: {self.leverage}x")
        if self.target_vol:
            print(f"  Target Volatility: {self.target_vol:.1%}")
        print("="*60)


def create_target_portfolio() -> List[PortfolioAllocation]:
    """
    Create the target portfolio allocation from ideas.ipynb.
    
    Note: This uses available tickers. You'll need to add more tickers
    for crypto, gold, silver, and speculation as you get the data.
    """
    allocations = [
        # Equity ETF - 25%
        PortfolioAllocation("SPY", 0.25, AssetClass.EQUITY, sector="ETF", strategy="Market"),
        
        # AI & Tech - 25% (using NVDA as proxy for now)
        PortfolioAllocation("NVDA", 0.15, AssetClass.AI, sector="AI/Tech", strategy="Growth"),
        PortfolioAllocation("TSM", 0.20, AssetClass.AI, sector="AI/Tech", strategy="Growth"),
        PortfolioAllocation("GOOGL", 0.20, AssetClass.AI, sector="AI/Tech", strategy="Growth"),
        
        # Value - 10% (using AAPL as proxy for now)
        PortfolioAllocation("AAPL", 0.10, AssetClass.EQUITY, sector="Tech", strategy="Value"),
        PortfolioAllocation("NFLX", 0.10, AssetClass.EQUITY, sector="Tech", strategy="Growth"),
        
        # Bonds - 15%
        PortfolioAllocation("TLT", 0.15, AssetClass.BOND, sector="Bonds", strategy="Duration"),
        
        # Crypto - 10% (placeholder - need to add ticker when you have data)
        # PortfolioAllocation("BTC", 0.10, AssetClass.CRYPTO),
        
        # Gold & Silver - 10% (placeholder - need to add ticker when you have data)
        # PortfolioAllocation("GLD", 0.10, AssetClass.METALS),
        
        # Speculation - 10% (placeholder - need to add ticker when you have data)
        # PortfolioAllocation("SPEC", 0.10, AssetClass.SPECULATION),
    ]
    
    # Normalize weights to sum to 1.0 (since we're missing some assets)
    total_weight = sum(alloc.weight for alloc in allocations)
    if total_weight < 1.0:
        print(f"Note: Total weight is {total_weight:.1%} (missing some assets)")
        # Scale up to use available assets
        for alloc in allocations:
            alloc.weight = alloc.weight / total_weight
    
    return allocations


async def main():
    """Main function for running backtest."""
    # Create your target portfolio
    print("Creating portfolio allocation...")
    allocations = create_target_portfolio()
    
    # Initialize portfolio with 2x leverage (as per your target)
    portfolio = SimplePortfolio(
        allocations=allocations,
        leverage=2.0,  # 2x leverage as per your target
        target_vol=0.18,  # 18% target volatility (middle of 15-20% range)
        backtest_years=5,  # 5 years of backtest data
        auto_download=True  # Automatically download missing tickers
    )
    
    # Load data (will auto-download missing tickers)
    await portfolio.load_data()
    
    # Run backtest
    print("\nRunning backtest...")
    portfolio.backtest()
    
    # Print results
    portfolio.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
