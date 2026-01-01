"""
Live Portfolio Manager - Fetches and displays current IBKR positions.
Production-grade class for real-time portfolio tracking.
"""

import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from ib_insync import IB
import pandas as pd
from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID_LIVE


@dataclass
class LivePosition:
    """Live position information from IBKR."""
    ticker: str
    position: float
    market_price: float
    market_value: float
    avg_cost: float
    unrealized_pnl: float
    realized_pnl: float
    allocation_pct: float


@dataclass
class AccountSummary:
    """Account summary information."""
    account: str
    net_liquidation: float
    total_cash: float
    available_cash: float
    buying_power: float
    gross_position_value: float
    excess_liquidity: float


class LivePortfolioManager:
    """Manages live portfolio data from IBKR."""
    
    def __init__(self):
        """Initialize live portfolio manager."""
        self.ib: Optional[IB] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Disconnect if already connected
            if self.connected and self.ib and self.ib.isConnected():
                self.ib.disconnect()
            
            self.ib = IB()
            await self.ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID_LIVE)
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to IBKR: {e}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
            self.connected = False
            self.ib = None
        except Exception as e:
            print(f"Error disconnecting: {e}")
            self.connected = False
            self.ib = None
    
    async def get_account_summary(self, account: Optional[str] = None) -> Optional[AccountSummary]:
        """
        Get account summary for specified account or first available.
        
        Args:
            account: Account ID (None to use first available)
            
        Returns:
            AccountSummary object or None
        """
        if not self.connected:
            return None
        
        account_values = self.ib.accountValues()
        
        # Find account
        accounts = set(av.account for av in account_values)
        if not accounts:
            return None
        
        target_account = account if account else sorted(accounts)[0]
        
        # Extract values
        summary = {
            'account': target_account,
            'net_liquidation': None,
            'total_cash': None,
            'available_cash': None,
            'buying_power': None,
            'gross_position_value': None,
            'excess_liquidity': None
        }
        
        for av in account_values:
            if av.account == target_account:
                if av.tag == 'NetLiquidation':
                    summary['net_liquidation'] = float(av.value)
                elif av.tag == 'TotalCashValue':
                    summary['total_cash'] = float(av.value)
                elif av.tag == 'AvailableFunds':
                    summary['available_cash'] = float(av.value)
                elif av.tag == 'BuyingPower':
                    summary['buying_power'] = float(av.value)
                elif av.tag == 'GrossPositionValue':
                    summary['gross_position_value'] = float(av.value)
                elif av.tag == 'ExcessLiquidity':
                    summary['excess_liquidity'] = float(av.value)
        
        if summary['net_liquidation'] is None:
            return None
        
        return AccountSummary(**summary)
    
    async def get_positions(self, account: Optional[str] = None) -> List[LivePosition]:
        """
        Get live positions for specified account.
        Matches the logic from get_portfolio.py which works correctly.
        
        Args:
            account: Account ID (None to use first available)
            
        Returns:
            List of LivePosition objects
        """
        if not self.connected:
            return []
        
        # Determine target account
        if account:
            target_account = account
        else:
            # Get first available account
            accounts = await self.get_all_accounts()
            target_account = accounts[0] if accounts else None
        
        if target_account is None:
            return []
        
        # Try portfolio() method first (like get_portfolio.py)
        portfolio = self.ib.portfolio()
        
        # Try positions() method as alternative
        positions_raw = self.ib.positions()
        
        # Use portfolio if available, otherwise use positions (matching get_portfolio.py logic)
        if portfolio:
            # Filter by account
            account_items = [item for item in portfolio if item.account == target_account]
        elif positions_raw:
            # Filter by account and convert to portfolio-like format
            account_positions = [pos for pos in positions_raw if pos.account == target_account]
            
            # Convert positions to portfolio-like items
            account_items = []
            for pos in account_positions:
                # Create a simple wrapper (matching get_portfolio.py)
                class PortfolioItem:
                    def __init__(self, pos):
                        self.account = pos.account
                        self.contract = pos.contract
                        self.position = pos.position
                        self.marketValue = pos.position * pos.avgCost  # Approximate
                        self.marketPrice = pos.avgCost
                        self.averageCost = pos.avgCost
                        self.unrealizedPNL = 0  # Not available in positions()
                        self.realizedPNL = 0
                account_items.append(PortfolioItem(pos))
        else:
            return []
        
        if not account_items:
            return []
        
        # Get account summary for net liquidation
        account_summary = await self.get_account_summary(target_account)
        net_liquidation = account_summary.net_liquidation if account_summary else None
        
        if net_liquidation is None:
            # Calculate from positions if not available
            net_liquidation = sum(abs(item.marketValue) for item in account_items)
        
        positions = []
        for item in account_items:
            contract = item.contract
            market_value = abs(item.marketValue)
            allocation_pct = (market_value / net_liquidation * 100) if net_liquidation > 0 else 0
            
            positions.append(LivePosition(
                ticker=contract.symbol,
                position=item.position,
                market_price=item.marketPrice,
                market_value=market_value,
                avg_cost=item.averageCost,
                unrealized_pnl=item.unrealizedPNL,
                realized_pnl=item.realizedPNL,
                allocation_pct=allocation_pct
            ))
        
        # Sort by market value (descending)
        positions.sort(key=lambda x: x.market_value, reverse=True)
        
        return positions
    
    async def get_all_accounts(self) -> List[str]:
        """
        Get list of all available accounts.
        
        Returns:
            List of account IDs
        """
        if not self.connected:
            return []
        
        accounts = set()
        
        # Get accounts from accountValues
        account_values = self.ib.accountValues()
        for av in account_values:
            accounts.add(av.account)
        
        # Also get accounts from portfolio
        portfolio = self.ib.portfolio()
        if portfolio:
            for item in portfolio:
                accounts.add(item.account)
        else:
            # Try positions as fallback
            positions = self.ib.positions()
            if positions:
                for pos in positions:
                    accounts.add(pos.account)
        
        return sorted(accounts)
