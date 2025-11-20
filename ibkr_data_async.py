import asyncio
from ib_insync import IB, Stock, util
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import os
import logging

from config import PICKLE_DIR, HISTORICAL_YEARS, IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
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

    async def fetch_ticker_data(self, ticker: str, years: int = HISTORICAL_YEARS, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        pickle_path = os.path.join(self.pickle_dir, f"{ticker}.xz")
        if not force_refresh and os.path.exists(pickle_path):
            try:
                self.logger.info(f"{ticker}: Loaded from cache.")
                df = load_pickle(pickle_path)
                self.logger.info(f"{ticker}: Data shape {df.shape if df is not None else None}")
                return df
            except Exception as e:
                self.logger.warning(f"{ticker}: Cache load failed, re-downloading. Error: {e}", exc_info=True)
        # Special handling for index: try both SPX and SPY
        if ticker.upper() == "SPX":
            contracts = [Stock("SPX", "CBOE", "USD"), Stock("SPX", "SMART", "USD"), Stock("SPY", "ARCA", "USD")]
        else:
            contracts = [Stock(ticker, 'SMART', 'USD')]
        for contract in contracts:
            try:
                duration = f"{years} Y"
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",  # Use empty string for latest data (fixes Error 321)
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False
                )
                if bars is None:
                    self.logger.warning(f"{ticker}: No bars returned from {contract.symbol} {contract.exchange}.")
                    continue
                df = util.df(bars)
                if df is not None and not df.empty:
                    save_pickle(pickle_path, df)
                    self.logger.info(f"{ticker}: Downloaded and cached from {contract.symbol} {contract.exchange}. Data shape {df.shape}")
                    return df
                else:
                    self.logger.warning(f"{ticker}: No data returned from {contract.symbol} {contract.exchange}.")
            except Exception as e:
                self.logger.error(f"{ticker}: Failed to fetch from {contract.symbol} {contract.exchange}. Error: {e}", exc_info=True)
        self.logger.error(f"{ticker}: All contract attempts failed.")
        return None

    async def fetch_all_tickers_data(self, tickers: List[str], years: int = HISTORICAL_YEARS, force_refresh: bool = False, max_concurrent: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all tickers using asyncio.gather, with a concurrency limit.
        
        Args:
            tickers: List of ticker symbols to fetch
            years: Number of years of historical data to fetch
            force_refresh: If True, bypass cache and re-download data
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        results = {}
        sem = asyncio.Semaphore(max_concurrent)
        async def sem_task(ticker):
            async with sem:
                return ticker, await self.fetch_ticker_data(ticker, years=years, force_refresh=force_refresh)
        tasks = [sem_task(ticker) for ticker in tickers]
        for coro in asyncio.as_completed(tasks):
            ticker, df = await coro
            if df is not None:
                results[ticker] = df
        return results 