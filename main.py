"""
Main entry point for Portfolio Analysis System.
Production-grade CLI interface for hedge fund operations.
"""

import argparse
import asyncio
import sys
from typing import List, Optional
from datetime import date
from portfolio_manager import PortfolioManager
from portfolio_backtest import SimplePortfolio, PortfolioAllocation, create_target_portfolio
from assets import AssetClass


def run_gui():
    """Launch the Streamlit GUI."""
    import subprocess
    import os
    
    script_path = os.path.join(os.path.dirname(__file__), "portfolio_gui.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])


async def run_backtest(
    tickers: Optional[List[str]] = None,
    years: int = 5,
    leverage: float = 2.0,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    auto_download: bool = True
):
    """
    Run portfolio backtest.
    
    Args:
        tickers: List of tickers (None to use default portfolio)
        years: Years of historical data
        leverage: Leverage multiplier
        start_date: Optional start date
        end_date: Optional end date
        auto_download: Auto-download missing data
    """
    print("=" * 80)
    print("PORTFOLIO BACKTEST")
    print("=" * 80)
    
    if tickers is None:
        # Use default portfolio from ideas.ipynb
        print("Using default portfolio allocation...")
        allocations = create_target_portfolio()
        tickers = [alloc.ticker for alloc in allocations]
    else:
        # Create equal-weight allocations
        allocations = [
            PortfolioAllocation(ticker, 1.0 / len(tickers), AssetClass.EQUITY)
            for ticker in tickers
        ]
    
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Years: {years}")
    print(f"Leverage: {leverage}x")
    print()
    
    # Create portfolio
    portfolio = SimplePortfolio(
        allocations=allocations,
        leverage=leverage,
        backtest_years=years,
        start_date=start_date,
        end_date=end_date,
        auto_download=auto_download
    )
    
    # Load data and run backtest
    print("Loading data...")
    await portfolio.load_data()
    
    print("\nRunning backtest...")
    portfolio.backtest()
    
    # Print results
    portfolio.print_summary()


async def run_analysis(
    tickers: List[str],
    years: int = 5,
    leverage: float = 2.0,
    weights: Optional[dict] = None,
    auto_download: bool = True
):
    """
    Run portfolio analysis using PortfolioManager.
    
    Args:
        tickers: List of ticker symbols
        years: Years of historical data
        leverage: Leverage multiplier
        weights: Optional dictionary of ticker to weight
        auto_download: Auto-download missing data
    """
    print("=" * 80)
    print("PORTFOLIO ANALYSIS")
    print("=" * 80)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Years: {years}")
    print(f"Leverage: {leverage}x")
    print()
    
    # Create manager
    manager = PortfolioManager(auto_download=auto_download)
    
    # Load data
    print("Loading data...")
    await manager.load_portfolio(tickers, years)
    
    # Analyze
    print("Analyzing portfolio...")
    analyzer = manager.analyze_portfolio(
        weights=weights,
        leverage=leverage
    )
    
    # Get metrics
    metrics = manager.get_metrics()
    positions = manager.get_positions()
    
    # Print results
    print("\n" + "=" * 80)
    print("PORTFOLIO METRICS")
    print("=" * 80)
    print(f"Total Return:          {metrics.total_return:.2%}")
    print(f"Annualized Return:    {metrics.annualized_return:.2%}")
    print(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
    print(f"Sharpe Ratio:         {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:         {metrics.max_drawdown:.2%}")
    print(f"Win Rate:             {metrics.win_rate:.2%}")
    print(f"Leverage:             {metrics.leverage:.1f}x")
    print(f"Period:               {metrics.years:.2f} years")
    
    print("\n" + "=" * 80)
    print("POSITIONS")
    print("=" * 80)
    print(f"{'Ticker':<10} {'Weight':<10} {'Allocation %':<15} {'Total Return':<15} {'Ann. Return':<15}")
    print("-" * 80)
    for pos in positions:
        print(f"{pos.ticker:<10} {pos.weight:<10.4f} {pos.allocation_pct:<15.2f}% {pos.total_return:<15.2%} {pos.annualized_return:<15.2%}")
    
    print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Portfolio Analysis System - Production Hedge Fund Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  python main.py gui
  
  # Run backtest with default portfolio
  python main.py backtest
  
  # Run backtest with custom tickers
  python main.py backtest --tickers SPY NVDA AAPL --years 5 --leverage 2.0
  
  # Run analysis
  python main.py analyze --tickers SPY NVDA AAPL --years 3 --leverage 2.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive GUI')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run portfolio backtest')
    backtest_parser.add_argument(
        '--tickers',
        nargs='+',
        help='Ticker symbols (default: uses portfolio from ideas.ipynb)'
    )
    backtest_parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )
    backtest_parser.add_argument(
        '--leverage',
        type=float,
        default=2.0,
        help='Leverage multiplier (default: 2.0)'
    )
    backtest_parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    backtest_parser.add_argument(
        '--no-auto-download',
        action='store_true',
        help='Disable auto-download of missing data'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run portfolio analysis')
    analyze_parser.add_argument(
        '--tickers',
        nargs='+',
        required=True,
        help='Ticker symbols'
    )
    analyze_parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )
    analyze_parser.add_argument(
        '--leverage',
        type=float,
        default=2.0,
        help='Leverage multiplier (default: 2.0)'
    )
    analyze_parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='Position weights (must match number of tickers)'
    )
    analyze_parser.add_argument(
        '--no-auto-download',
        action='store_true',
        help='Disable auto-download of missing data'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == 'gui':
        run_gui()
    
    elif args.command == 'backtest':
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = date.fromisoformat(args.start_date)
        if args.end_date:
            end_date = date.fromisoformat(args.end_date)
        
        asyncio.run(run_backtest(
            tickers=args.tickers,
            years=args.years,
            leverage=args.leverage,
            start_date=start_date,
            end_date=end_date,
            auto_download=not args.no_auto_download
        ))
    
    elif args.command == 'analyze':
        weights = None
        if args.weights:
            if len(args.weights) != len(args.tickers):
                print("Error: Number of weights must match number of tickers")
                sys.exit(1)
            weights = dict(zip(args.tickers, args.weights))
        
        asyncio.run(run_analysis(
            tickers=args.tickers,
            years=args.years,
            leverage=args.leverage,
            weights=weights,
            auto_download=not args.no_auto_download
        ))
    
    else:
        print("Error: No command specified. Use 'gui', 'backtest', or 'analyze'")
        print("Run 'python main.py --help' for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
