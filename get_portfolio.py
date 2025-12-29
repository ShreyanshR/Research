"""
Simple script to get positions from IBKR.
Shows all accounts and their positions with portfolio percentages.
"""

import asyncio
from ib_insync import IB
import pandas as pd
from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID


async def get_portfolio_data(account: str = None):
    """Get positions from IBKR. If account is None, shows all accounts."""
    ib = IB()
    
    try:
        # Connect to IBKR
        print(f"Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT}...")
        await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
        print("Connected!\n")
        
        # Debug: Check what accounts are available
        print("Checking available accounts...")
        account_values = ib.accountValues()
        if account_values:
            accounts_from_values = set(av.account for av in account_values)
            print(f"Accounts from accountValues(): {', '.join(sorted(accounts_from_values))}")
        
        # Try portfolio() method
        print("\nTrying portfolio() method...")
        portfolio = ib.portfolio()
        print(f"portfolio() returned {len(portfolio)} items")
        
        # Try positions() method as alternative
        print("\nTrying positions() method...")
        positions = ib.positions()
        print(f"positions() returned {len(positions)} items")
        
        if not portfolio and not positions:
            print("\nNo positions found using either method.")
            print("This could mean:")
            print("  1. You have no open positions")
            print("  2. The account data hasn't loaded yet (try waiting a moment)")
            print("  3. You need to request account data explicitly")
            return
        
        # Use portfolio if available, otherwise use positions
        if portfolio:
            all_accounts = set(item.account for item in portfolio)
            print(f"\nFound {len(all_accounts)} account(s) with positions: {', '.join(sorted(all_accounts))}\n")
        elif positions:
            all_accounts = set(pos.account for pos in positions)
            print(f"\nFound {len(all_accounts)} account(s) with positions: {', '.join(sorted(all_accounts))}\n")
        else:
            return
        
        # Filter by account if specified
        if account:
            account_positions = [item for item in portfolio if item.account == account]
            if not account_positions:
                print(f"No positions found for account {account}")
                print(f"Available accounts: {', '.join(sorted(all_accounts))}")
                return
            accounts_to_process = [account]
        else:
            account_positions = portfolio
            accounts_to_process = sorted(all_accounts)
        
        # Process each account
        for acc in accounts_to_process:
            if account:
                if portfolio:
                    account_items = account_positions
                else:
                    account_items = [pos for pos in positions if pos.account == acc]
            else:
                if portfolio:
                    account_items = [item for item in portfolio if item.account == acc]
                else:
                    account_items = [pos for pos in positions if pos.account == acc]
            
            if not account_items:
                continue
            
            # Handle both portfolio items and positions
            if portfolio:
                items = account_items
            else:
                # Convert positions to portfolio-like format
                items = []
                for pos in account_items:
                    # Create a simple object-like structure
                    class SimpleItem:
                        def __init__(self, pos):
                            self.account = pos.account
                            self.contract = pos.contract
                            self.position = pos.position
                            self.marketValue = pos.position * pos.avgCost  # Approximate
                            self.marketPrice = pos.avgCost
                            self.averageCost = pos.avgCost
                            self.unrealizedPNL = 0  # Not available in positions()
                    items.append(SimpleItem(pos))
        
            # Get total account value
            account_values = ib.accountValues()
            net_liquidation = None
            for av in account_values:
                if av.account == acc and av.tag == 'NetLiquidation':
                    net_liquidation = float(av.value)
                    break
            
            if net_liquidation is None:
                # Calculate from positions if not available
                net_liquidation = sum(abs(item.marketValue) for item in items)
            
            # Build positions data
            positions_data = []
            for item in items:
                contract = item.contract
                market_value = abs(item.marketValue)  # Use absolute value
                pct_of_portfolio = (market_value / net_liquidation * 100) if net_liquidation > 0 else 0
                
                positions_data.append({
                    'Symbol': contract.symbol,
                    'Position': item.position,
                    'Market Price': f"${item.marketPrice:.2f}",
                    'Market Value': f"${market_value:,.2f}",
                    '% of Portfolio': f"{pct_of_portfolio:.2f}%",
                    'Avg Cost': f"${item.averageCost:.2f}",
                    'Unrealized P&L': f"${item.unrealizedPNL:,.2f}",
                })
            
            df_positions = pd.DataFrame(positions_data)
            
            # Sort by market value (descending) - extract numeric value for sorting
            df_positions['_sort_value'] = df_positions['Market Value'].str.replace('$', '').str.replace(',', '').astype(float)
            df_positions = df_positions.sort_values('_sort_value', ascending=False).drop('_sort_value', axis=1)
            
            print("=" * 80)
            print(f"POSITIONS FOR ACCOUNT {acc}")
            print("=" * 80)
            print(df_positions.to_string(index=False))
            
            total_market_value = sum(abs(item.marketValue) for item in items)
            total_unrealized_pnl = sum(item.unrealizedPNL for item in items)
            
            # Calculate leverage metrics
            # Get available cash
            available_cash = None
            total_cash = None
            for av in account_values:
                if av.account == acc:
                    if av.tag == 'AvailableFunds':
                        available_cash = float(av.value)
                    elif av.tag == 'TotalCashValue':
                        total_cash = float(av.value)
            
            # Calculate leverage
            # Leverage = Total Position Value / Net Liquidation
            leverage = (total_market_value / net_liquidation) if net_liquidation > 0 else 0
            
            # Net exposure (long - short)
            long_value = sum(item.marketValue for item in items if item.marketValue > 0)
            short_value = abs(sum(item.marketValue for item in items if item.marketValue < 0))
            net_exposure = long_value - short_value
            
            print(f"\n{'='*80}")
            print(f"SUMMARY FOR ACCOUNT {acc}")
            print(f"{'='*80}")
            print(f"Net Liquidation:        ${net_liquidation:>15,.2f}")
            if total_cash is not None:
                print(f"Total Cash:             ${total_cash:>15,.2f}")
            if available_cash is not None:
                print(f"Available Cash:          ${available_cash:>15,.2f}")
            print(f"Total Position Value:   ${total_market_value:>15,.2f}")
            print(f"  Long Positions:       ${long_value:>15,.2f}")
            print(f"  Short Positions:       ${short_value:>15,.2f}")
            print(f"Net Exposure:            ${net_exposure:>15,.2f}")
            print(f"Leverage:                {leverage:>15.2f}x")
            print(f"Total Unrealized P&L:   ${total_unrealized_pnl:>15,.2f}")
            print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IBKR")


if __name__ == "__main__":
    # Show all accounts by default, or specify account like: get_portfolio_data("U19591453")
    asyncio.run(get_portfolio_data())
