import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable, Any, Optional
import warnings
import time


class Data:
    """
    Data cleaning and preprocessing class for financial time series data.
    
    Handles:
    - Date index conversion and timezone normalization
    - Null value handling (forward fill, backward fill)
    - Returns calculation (percentage returns)
    - Log returns calculation
    - Additional calendar features (day of month, etc.)
    """
    
    def __init__(self, tickers: List[str], dfs: Dict[str, pd.DataFrame], years: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None, portfolio_vol: float = 0.20):
        """
        Initialize Data object.
        
        Args:
            tickers: List of ticker symbols
            dfs: Dictionary mapping ticker symbols to DataFrames
            years: Number of years back from today (if provided, automatically sets end=today, start=end-years)
            start: Start date for data range (required if years not provided)
            end: End date for data range (defaults to today if years provided, else required)
            portfolio_vol: Target portfolio volatility (default 0.20 = 20%)
        
        Examples:
            # Simple: just pass years
            data_processor = Data(tickers=["AAPL"], dfs=all_data, years=5)
            
            # Explicit dates
            data_processor = Data(tickers=["AAPL"], dfs=all_data, 
                                 start=datetime(2020,1,1), end=datetime(2024,12,31))
        """
        self.tickers = tickers
        self.dfs = dfs
        
        # If years provided, automatically set end=today and start=end-years
        if years is not None:
            self.end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            self.start = self.end - timedelta(days=int(years * 365.25))  # Account for leap years
        else:
            if start is None or end is None:
                raise ValueError("Must provide either 'years' parameter or both 'start' and 'end' dates")
            self.start = start
            self.end = end
        
        self.portfolio_vol = portfolio_vol

    def clean_data(self, trade_range: Optional[pd.DatetimeIndex] = None, fill_method: str = 'ffill', fill_limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Clean and preprocess data for all tickers.
        
        Args:
            trade_range: Optional DatetimeIndex to align all dataframes to
            fill_method: Method for filling nulls ('ffill', 'bfill', or 'both')
                        - 'ffill': Forward fill (carry last known value forward)
                        - 'bfill': Backward fill (carry next known value backward)
                        - 'both': Forward fill then backward fill (recommended for gaps)
            fill_limit: Maximum number of consecutive nulls to fill (None = no limit)
        
        Returns:
            Dictionary of cleaned DataFrames keyed by ticker symbol
        """
        cleaned_dfs = {}
        
        for ticker in self.tickers:
            if ticker not in self.dfs:
                warnings.warn(f"Ticker {ticker} not found in data dictionary, skipping...")
                continue
                
            df = self.dfs[ticker].copy()
            
            # Set date as index if it exists as a column
            if 'date' in df.columns:
                df = df.set_index('date')
            
            # Convert index to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Handle timezone - normalize to UTC
            if not hasattr(df.index, 'tz') or df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Align to trade_range if provided
            if trade_range is not None:
                df = df.reindex(trade_range)
            
            # Handle null values in price columns
            price_columns = ['open', 'high', 'low', 'close']
            existing_price_cols = [col for col in price_columns if col in df.columns]
            
            if fill_method == 'both':
                # Forward fill first, then backward fill remaining nulls
                df[existing_price_cols] = df[existing_price_cols].ffill(limit=fill_limit)
                df[existing_price_cols] = df[existing_price_cols].bfill(limit=fill_limit)
            elif fill_method == 'ffill':
                df[existing_price_cols] = df[existing_price_cols].ffill(limit=fill_limit)
            elif fill_method == 'bfill':
                df[existing_price_cols] = df[existing_price_cols].bfill(limit=fill_limit)
            else:
                raise ValueError(f"fill_method must be 'ffill', 'bfill', or 'both', got '{fill_method}'")
            
            # Calculate returns (percentage returns)
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
            
            # Calculate log returns
            if 'close' in df.columns:
                df['log_returns'] = np.log(df['close']).diff()
            
            # Add calendar features
            df['day_of_month'] = df.index.day
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['year'] = df.index.year
            
            # Store cleaned dataframe
            cleaned_dfs[ticker] = df
        
        return cleaned_dfs
