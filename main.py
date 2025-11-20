"""
Main file to connect to IBKR API and fetch data (command-line interface).

For Jupyter notebook usage, import and use ibkr_data_async.py directly.

Make sure TWS (Trader Workstation) or IB Gateway is running before executing this script.
Default settings: localhost:7497 (paper trading)

Usage:
    python main.py AAPL MSFT GOOGL --years 5 --force-refresh
    python main.py SPY --years 10
"""

import asyncio
import logging
import argparse
from ibkr_data_async import DataFetcherAsync
from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, HISTORICAL_YEARS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def fetch_data(tickers: list, years: int = HISTORICAL_YEARS, force_refresh: bool = False):
    """
    Fetch data for given tickers.
    
    Args:
        tickers: List of ticker symbols to fetch
        years: Number of years of historical data to fetch
        force_refresh: If True, bypass cache and re-download data
    
    Returns:
        dict: Dictionary of ticker -> DataFrame (for multiple tickers)
        or DataFrame: Single DataFrame (for single ticker)
    """
    # Initialize the data fetcher
    fetcher = DataFetcherAsync(
        host=IBKR_HOST,
        port=IBKR_PORT,
        client_id=IBKR_CLIENT_ID
    )
    
    try:
        # Connect to IBKR
        logger.info(f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT}...")
        await fetcher.connect()
        logger.info("Successfully connected to IBKR!")
        
        # Fetch data for all tickers
        if len(tickers) == 1:
            # Single ticker
            ticker = tickers[0]
            logger.info(f"Fetching {years} years of data for {ticker}...")
            df = await fetcher.fetch_ticker_data(ticker, years=years, force_refresh=force_refresh)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {ticker} data!")
                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                logger.info(f"\nFirst few rows:\n{df.head()}")
                logger.info(f"\nLast few rows:\n{df.tail()}")
                return df
            else:
                logger.warning(f"No data retrieved for {ticker}")
                return None
        else:
            # Multiple tickers
            logger.info(f"Fetching {years} years of data for {len(tickers)} tickers: {tickers}...")
            all_data = await fetcher.fetch_all_tickers_data(
                tickers=tickers,
                force_refresh=force_refresh,
                max_concurrent=5
            )
            
            logger.info(f"\nFetched data for {len(all_data)} tickers:")
            for ticker, df in all_data.items():
                if df is not None:
                    logger.info(f"  {ticker}: {df.shape[0]} rows, from {df.index.min()} to {df.index.max()}")
            return all_data
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None
    finally:
        # Disconnect
        if fetcher.ib.isConnected():
            fetcher.ib.disconnect()
            logger.info("Disconnected from IBKR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical data from IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py AAPL MSFT GOOGL
  python main.py SPY --years 10
  python main.py AAPL --years 5 --force-refresh
        """
    )
    
    parser.add_argument(
        'tickers',
        nargs='+',
        help='Ticker symbol(s) to fetch (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=HISTORICAL_YEARS,
        help=f'Number of years of historical data to fetch (default: {HISTORICAL_YEARS})'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force refresh: bypass cache and re-download data'
    )
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(fetch_data(
        tickers=args.tickers,
        years=args.years,
        force_refresh=args.force_refresh
    ))

