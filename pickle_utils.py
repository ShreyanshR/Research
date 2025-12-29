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


async def ensure_all_tickers_data(
    tickers: List[str],
    years: int = 5,
    force_refresh: bool = False,
    update_if_old: bool = True,
    auto_download: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Simple function: check if pickle files exist, download missing ones, update old ones.
    Uses the existing fetch_all_tickers_data from ibkr_data_async.
    
    Args:
        tickers: List of ticker symbols
        years: Years of historical data to fetch
        force_refresh: If True, always re-download
        update_if_old: If True, check if data is old and update (default: True)
        auto_download: If False, only load from cache, don't download missing files (default: True)
    
    Returns:
        Dictionary of {ticker: DataFrame}
    """
    from config import PICKLE_DIR, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
    from ibkr_data_async import DataFetcherAsync
    from datetime import datetime, timedelta
    
    # Check which tickers need downloading
    tickers_to_download = []
    tickers_to_load = []
    
    for ticker in tickers:
        pickle_path = os.path.join(PICKLE_DIR, f"{ticker}.xz")
        
        needs_download = False
        if force_refresh:
            needs_download = True
            print(f"  {ticker}: Force refresh requested")
        elif not os.path.exists(pickle_path):
            needs_download = True
            print(f"  {ticker}: File not found, will download...")
        elif update_if_old:
            # Check if file is older than 1 day (data might be stale)
            file_time = datetime.fromtimestamp(os.path.getmtime(pickle_path))
            if datetime.now() - file_time > timedelta(days=1):
                needs_download = True
                print(f"  {ticker}: File is old (last updated {file_time.strftime('%Y-%m-%d')}), will update...")
            else:
                print(f"  {ticker}: File exists and is recent, loading from cache")
                tickers_to_load.append(ticker)
        else:
            tickers_to_load.append(ticker)
        
        if needs_download:
            tickers_to_download.append(ticker)
    
    # Download missing/old tickers using existing function
    if tickers_to_download and auto_download:
        print(f"\nDownloading {len(tickers_to_download)} tickers from IBKR...")
        fetcher = DataFetcherAsync(
            host=IBKR_HOST,
            port=IBKR_PORT,
            client_id=IBKR_CLIENT_ID
        )
        try:
            await fetcher.connect()
            downloaded_data = await fetcher.fetch_all_tickers_data(
                tickers=tickers_to_download,
                years=years,
                force_refresh=force_refresh,
                max_concurrent=5
            )
            print(f"Downloaded {len(downloaded_data)} tickers")
        finally:
            if fetcher.ib.isConnected():
                fetcher.ib.disconnect()
    elif tickers_to_download and not auto_download:
        raise FileNotFoundError(
            f"Missing data for tickers: {tickers_to_download}. "
            f"Set auto_download=True to download automatically."
        )
    
    # Load all tickers (from cache or newly downloaded)
    results = {}
    for ticker in tickers:
        pickle_path = os.path.join(PICKLE_DIR, f"{ticker}.xz")
        if os.path.exists(pickle_path):
            try:
                results[ticker] = load_pickle(pickle_path)
            except Exception as e:
                print(f"  ✗ Failed to load {ticker}: {e}")
                raise
        else:
            raise FileNotFoundError(f"Pickle file not found after download: {pickle_path}")
    
    return results