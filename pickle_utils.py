import lzma
import pickle
import os
from typing import Any, Dict, Optional, List
import pandas as pd

def load_pickle(file_path: str) -> Any:
    """
    Load a pickled object from an lzma-compressed file.
    Args:
        file_path (str): Path to the .xz (lzma) pickle file.
    Returns:
        Any: The loaded Python object.
    """
    with lzma.open(file_path, "rb") as fp:
        return pickle.load(fp)

def save_pickle(file_path: str, obj: Any) -> None:
    """
    Save a Python object to an lzma-compressed pickle file.
    Args:
        file_path (str): Path to save the .xz (lzma) pickle file.
        obj (Any): The Python object to pickle.
    """
    with lzma.open(file_path, "wb") as fp:
        pickle.dump(obj, fp)


def load_all_pickle_data(tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Simple function to load your existing pickle data.
    
    Args:
        tickers: Optional list of tickers to load (e.g., ["SPY", "AAPL"]), if None loads all
    
    Returns:
        Dictionary of {ticker: DataFrame}
    
    Example:
        # Load all tickers
        all_data = load_all_pickle_data()
        
        # Load single ticker
        spy_data = load_all_pickle_data(tickers=["SPY"])
        
        # Load multiple tickers
        data = load_all_pickle_data(tickers=["SPY", "AAPL", "NVDA"])
    """
    from config import PICKLE_DIR
    
    loaded_data = {}
    
    if tickers is None:
        # Load all pickle files in directory
        if not os.path.exists(PICKLE_DIR):
            return {}
        
        for filename in os.listdir(PICKLE_DIR):
            if filename.endswith('.xz'):
                ticker_name = filename[:-3]  # Remove .xz extension
                pickle_path = os.path.join(PICKLE_DIR, filename)
                loaded_data[ticker_name] = load_pickle(pickle_path)
    else:
        # Load specific tickers
        for ticker in tickers:
            pickle_path = os.path.join(PICKLE_DIR, f"{ticker}.xz")
            if os.path.exists(pickle_path):
                loaded_data[ticker] = load_pickle(pickle_path)
            else:
                raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
    
    return loaded_data


def save_pickle_data(ticker: str, df: pd.DataFrame) -> None:
    """
    Simple function to save DataFrame to pickle file.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY")
        df: DataFrame to save
    
    Example:
        save_pickle_data("SPY", cleaned_df)
    """
    from config import PICKLE_DIR
    
    pickle_path = os.path.join(PICKLE_DIR, f"{ticker}.xz")
    save_pickle(pickle_path, df)
    print(f"Saved {ticker} to {pickle_path}")