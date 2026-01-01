import asyncio
from ib_insync import IB, Stock, Crypto, util
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import os
import logging

from config import PICKLE_DIR, HISTORICAL_YEARS, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, WHAT_TO_SHOW
from pickle_utils import load_pickle, save_pickle

class DataFetcherAsync:
    """
    Uses ib_insync's asyncio capabilities to fetch data concurrently (experimental).
    """
    def __init__(self, host: str = IBKR_HOST, port: int = IBKR_PORT, client_id: int = IBKR_CLIENT_ID, pickle_dir: str = PICKLE_DIR):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.pickle_dir = pickle_dir
        os.makedirs(self.pickle_dir, exist_ok=True)
        self.ib = IB()
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        self.logger.info("Connected to IBKR (async mode).")

    async def fetch_ticker_data(self, ticker: str, years: int = HISTORICAL_YEARS, force_refresh: bool = False, what_to_show: str = WHAT_TO_SHOW) -> Optional[pd.DataFrame]:
        # Normalize ticker name for pickle file (remove exchange specification)
        # Support ticker format: "TICKER:EXCHANGE" or "TICKER@EXCHANGE" for custom exchange
        ticker_upper = ticker.upper()
        base_ticker = ticker_upper
        exchange_override = None
        if ':' in ticker_upper or '@' in ticker_upper:
            separator = ':' if ':' in ticker_upper else '@'
            parts = ticker_upper.split(separator, 1)
            base_ticker = parts[0].strip()
            exchange_override = parts[1].strip()
        
        # Use base ticker for pickle file name (so "BTC:PAXOS" and "BTC" use same cache)
        pickle_path = os.path.join(self.pickle_dir, f"{base_ticker}.xz")
        if not force_refresh and os.path.exists(pickle_path):
            try:
                self.logger.info(f"{ticker} (base: {base_ticker}): Loaded from cache.")
                df = load_pickle(pickle_path)
                self.logger.info(f"{ticker}: Data shape {df.shape if df is not None else None}")
                return df
            except Exception as e:
                self.logger.warning(f"{ticker}: Cache load failed, re-downloading. Error: {e}", exc_info=True)
        
        if base_ticker == "BTC" or base_ticker.startswith("BTC"):
            # Bitcoin is a cryptocurrency, use Crypto contract type
            # Try PAXOS first (most common), then ZEROHASH as fallback
            if exchange_override:
                contracts = [Crypto(base_ticker, exchange_override, "USD")]
            else:
                contracts = [Crypto("BTC", "PAXOS", "USD"), Crypto("BTC", "ZEROHASH", "USD")]
        elif base_ticker == "SPX":
            contracts = [Stock("SPX", "CBOE", "USD"), Stock("SPX", "SMART", "USD"), Stock("SPY", "ARCA", "USD")]
        elif base_ticker == "TLT":
            # TLT is an ETF on ARCA - try ARCA first, then SMART as fallback
            contracts = [Stock("TLT", "ARCA", "USD"), Stock("TLT", "SMART", "USD")]
        elif exchange_override:
            # User specified custom exchange
            contracts = [Stock(base_ticker, exchange_override, "USD")]
        else:
            contracts = [Stock(base_ticker, 'SMART', 'USD')]
        
        # Try different data types if ADJUSTED_LAST doesn't return enough data
        data_types_to_try = [what_to_show]
        if what_to_show == "ADJUSTED_LAST":
            # If ADJUSTED_LAST fails or returns limited data, try TRADES as fallback
            data_types_to_try.append("TRADES")
        
        best_df = None
        best_date_range = None
        best_data_type = None
        
        for contract in contracts:
            for data_type in data_types_to_try:
                try:
                    duration = f"{years} Y"
                    bars = await self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",  # Use empty string for latest data (fixes Error 321)
                        durationStr=duration,
                        barSizeSetting="1 day",
                        whatToShow=data_type,
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False
                    )
                    if bars is None:
                        self.logger.warning(f"{ticker}: No bars returned from {contract.symbol} {contract.exchange} with {data_type}.")
                        continue
                    df = util.df(bars)
                    if df is not None and not df.empty:
                        # Check if this gives us a better date range
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            date_min = df['date'].min()
                            date_max = df['date'].max()
                        else:
                            date_min = df.index.min() if hasattr(df.index, 'min') else None
                            date_max = df.index.max() if hasattr(df.index, 'max') else None
                        
                        # Keep the DataFrame with the earliest start date (most historical data)
                        # Prefer ADJUSTED_LAST when date ranges are equal
                        should_update = False
                        if best_df is None:
                            should_update = True
                        elif date_min is not None:
                            if best_date_range is None:
                                should_update = True
                            elif date_min < best_date_range:
                                should_update = True
                            elif date_min == best_date_range and data_type == "ADJUSTED_LAST" and best_data_type != "ADJUSTED_LAST":
                                # Prefer ADJUSTED_LAST when date ranges are equal
                                should_update = True
                        
                        if should_update:
                            best_df = df
                            best_date_range = date_min
                            best_data_type = data_type
                            self.logger.info(f"{ticker}: Found data from {contract.symbol} {contract.exchange} with {data_type}. Date range: {date_min} to {date_max}, Shape: {df.shape}")
                except Exception as e:
                    self.logger.error(f"{ticker}: Failed to fetch from {contract.symbol} {contract.exchange} with {data_type}. Error: {e}", exc_info=True)
        
        if best_df is not None and not best_df.empty:
            save_pickle(pickle_path, best_df)
            self.logger.info(f"{ticker} (base: {base_ticker}): Downloaded and cached. Final data shape {best_df.shape}")
            return best_df
        
        self.logger.error(f"{ticker}: All contract attempts failed.")
        return None

    async def fetch_all_tickers_data(self, tickers: List[str], years: int = HISTORICAL_YEARS, force_refresh: bool = False, max_concurrent: int = 5, what_to_show: str = WHAT_TO_SHOW) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all tickers using asyncio.gather, with a concurrency limit.
        
        Args:
            tickers: List of ticker symbols to fetch
            years: Number of years of historical data to fetch
            force_refresh: If True, bypass cache and re-download data
            max_concurrent: Maximum number of concurrent requests
            what_to_show: Type of data to fetch (e.g., "ADJUSTED_LAST" for adjusted close)
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        sem = asyncio.Semaphore(max_concurrent)
        async def sem_task(ticker):
            async with sem:
                return ticker, await self.fetch_ticker_data(ticker, years=years, force_refresh=force_refresh, what_to_show=what_to_show)
        tasks = [sem_task(ticker) for ticker in tickers]
        for coro in asyncio.as_completed(tasks):
            ticker, df = await coro
            if df is not None:
                results[ticker] = df
        return results 