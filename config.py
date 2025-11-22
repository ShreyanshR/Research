"""
Configuration file for IBKR connection and data settings.
"""

import os

# IBKR Connection Settings
IBKR_HOST = "127.0.0.1"  # Localhost - TWS or IB Gateway must be running
IBKR_PORT = 7496  # Paper trading port (use 7496 for live trading)
IBKR_CLIENT_ID = 1  # Unique client ID (each connection needs unique ID)

# Data Settings
PICKLE_DIR = os.path.join(os.path.dirname(__file__), "data", "pickle")  # Directory for cached data
EXCEL_DIR = os.path.join(os.path.dirname(__file__), "data", "excel")  # Directory for Excel files
HISTORICAL_YEARS = 5  # Default years of historical data to fetch

# Data Type Settings
# Options: "ADJUSTED_LAST" (adjusted close), "TRADES" (unadjusted), "MIDPOINT", "BID", "ASK"
WHAT_TO_SHOW = "ADJUSTED_LAST"  # Use "ADJUSTED_LAST" for adjusted close data

