"""
Interactive Portfolio Analysis GUI - Production-grade Streamlit interface.
Uses clean architecture with separated business logic and visualization.
"""

import streamlit as st
from typing import List, Dict, Optional
from datetime import date, timedelta
import asyncio
import pandas as pd
import nest_asyncio

# Allow nested event loops for Streamlit
nest_asyncio.apply()

from portfolio_manager import PortfolioManager
from portfolio_visualizer import ChartGenerator, MetricsDisplay
from portfolio_live import LivePortfolioManager


class PortfolioGUI:
    """Main GUI controller class."""
    
    def __init__(self):
        """Initialize GUI."""
        self.manager: Optional[PortfolioManager] = None
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'portfolio_loaded' not in st.session_state:
            st.session_state['portfolio_loaded'] = False
        if 'tickers' not in st.session_state:
            st.session_state['tickers'] = []
        if 'years' not in st.session_state:
            st.session_state['years'] = 5
    
    def render_sidebar(self) -> Dict:
        """
        Render sidebar controls and return configuration.
        
        Returns:
            Dictionary with configuration
        """
        st.sidebar.header("Portfolio Configuration")
        
        # Ticker input
        st.sidebar.subheader("Tickers")
        default_tickers = "SPY, NVDA, AAPL, TLT"
        ticker_input = st.sidebar.text_input(
            "Add tickers (comma-separated)",
            value=default_tickers,
            help="Enter ticker symbols separated by commas (e.g., SPY, NVDA, AAPL, TLT)"
        )
        
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        # Show parsed tickers
        if tickers:
            st.sidebar.write(f"**Tickers:** {', '.join(tickers)}")
        else:
            st.sidebar.warning("⚠️ Please add at least one ticker")
        
        # Date range
        st.sidebar.subheader("Date Range")
        years = st.sidebar.slider("Years of data", 1, 10, 5)
        end_date = date.today()
        start_date = end_date - timedelta(days=years * 365)
        st.sidebar.write(f"**Date Range:** {start_date} to {end_date}")
        
        # Initial Capital
        st.sidebar.subheader("Initial Capital")
        initial_capital = st.sidebar.number_input(
            "Starting capital ($)",
            min_value=0.0,
            value=100000.0,
            step=10000.0,
            format="%.0f",
            help="Initial capital amount for equity calculations"
        )
        
        # Leverage
        st.sidebar.subheader("Leverage")
        leverage = st.sidebar.slider("Leverage multiplier", 1.0, 5.0, 2.0, 0.1)
        
        # Portfolio weights
        st.sidebar.subheader("Portfolio Weights")
        use_equal_weights = st.sidebar.checkbox("Use equal weights", value=True)
        
        weights = None
        if not use_equal_weights and tickers:
            st.sidebar.write("**Set custom weights:**")
            weight_inputs = {}
            total_weight = 0.0
            
            for ticker in tickers:
                weight = st.sidebar.number_input(
                    f"{ticker} weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(tickers),
                    step=0.01,
                    format="%.2f"
                )
                weight_inputs[ticker] = weight
                total_weight += weight
            
            # Show total and normalize if needed
            st.sidebar.write(f"**Total weight: {total_weight:.2f}**")
            if abs(total_weight - 1.0) > 0.01:
                st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}, will be normalized to 1.0")
                # Normalize weights
                weights = {k: v / total_weight for k, v in weight_inputs.items()}
            else:
                weights = weight_inputs
        
        # Auto-download
        auto_download = st.sidebar.checkbox("Auto-download missing data", value=True)
        
        return {
            'tickers': tickers,
            'years': years,
            'start_date': start_date,
            'end_date': end_date,
            'leverage': leverage,
            'weights': weights,
            'initial_capital': initial_capital,
            'auto_download': auto_download
        }
    
    async def load_data(self, tickers: List[str], years: int, auto_download: bool) -> bool:
        """
        Load portfolio data.
        
        Args:
            tickers: List of ticker symbols
            years: Years of historical data
            auto_download: Whether to auto-download missing data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.manager = PortfolioManager(auto_download=auto_download)
            await self.manager.load_portfolio(tickers, years)
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def render_metrics(self, metrics) -> None:
        """
        Render portfolio metrics.
        
        Args:
            metrics: PortfolioMetrics object
        """
        formatted = MetricsDisplay.format_metrics(metrics)
        initial_capital = st.session_state.get('config', {}).get('initial_capital', 100000.0)
        
        # Calculate final capital
        final_capital = initial_capital * (1 + metrics.total_return)
        capital_change = final_capital - initial_capital
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        cols = [col1, col2, col3, col4, col5, col6]
        
        # Show initial and final capital first
        with cols[0]:
            st.metric("Initial Capital", f"${initial_capital:,.2f}")
        with cols[1]:
            st.metric("Final Capital", f"${final_capital:,.2f}", delta=f"${capital_change:,.2f}")
        
        # Then show other metrics (including Max Drawdown)
        metrics_to_show = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
        for i, label in enumerate(metrics_to_show):
            if label in formatted:
                with cols[i + 2]:
                    st.metric(label, formatted[label])
        
        # Second row for remaining metrics
        if len(formatted) > len(metrics_to_show) + 2:  # +2 for Initial/Final Capital
            st.write("")  # Spacing
            col1, col2, col3, col4 = st.columns(4)
            remaining_metrics = [k for k in formatted.keys() if k not in metrics_to_show]
            for i, label in enumerate(remaining_metrics[:4]):
                with [col1, col2, col3, col4][i]:
                    st.metric(label, formatted[label])
    
    def render_charts(self) -> None:
        """Render portfolio charts."""
        if self.manager is None or self.manager.analyzer is None:
            return
        
        portfolio_cumulative, individual_cumulative = self.manager.get_returns_data()
        initial_capital = st.session_state.get('config', {}).get('initial_capital', 100000.0)
        positions = self.manager.get_positions(initial_capital=initial_capital)
        metrics = self.manager.get_metrics()
        
        # Cumulative returns - full width for larger display
        st.subheader("Cumulative Returns")
        # Get initial capital from session state
        initial_capital = st.session_state.get('config', {}).get('initial_capital', 100000.0)
        fig_returns = ChartGenerator.create_returns_chart(
            portfolio_cumulative,
            individual_cumulative,
            metrics.leverage,
            initial_capital=initial_capital
        )
        st.plotly_chart(fig_returns, use_container_width=True, key="backtest_returns_chart")
        
        # Allocation and Equity Curve side by side
        col1, col2 = st.columns([1, 2])  # Allocation smaller, equity curve larger
        
        with col1:
            st.subheader("Portfolio Allocation")
            fig_allocation = ChartGenerator.create_allocation_chart(positions)
            st.plotly_chart(fig_allocation, use_container_width=True, key="backtest_allocation_chart")
        
        with col2:
            st.subheader("Portfolio Equity Curve")
            initial_capital = st.session_state.get('config', {}).get('initial_capital', 100000.0)
            fig_equity = ChartGenerator.create_equity_curve(
                self.manager.analyzer.portfolio_returns,
                initial_capital=initial_capital
            )
            st.plotly_chart(fig_equity, use_container_width=True, key="backtest_equity_chart")
        
        # Correlation matrix (basic - full period)
        st.subheader("Correlation Matrix (Full Period)")
        returns_df = self.manager.analyzer.returns_df
        fig_corr = ChartGenerator.create_correlation_matrix(returns_df)
        st.plotly_chart(fig_corr, use_container_width=True, key="backtest_correlation_chart")
        
        # Data timeline info
        st.subheader("Data Timeline")
        
        # Get requested date range from config
        config = st.session_state.get('config', {})
        requested_start = config.get('start_date', None)
        requested_end = config.get('end_date', None)
        requested_years = config.get('years', 5)
        
        actual_start = returns_df.index[0].date()
        actual_end = returns_df.index[-1].date()
        actual_days = (actual_end - actual_start).days
        actual_years = actual_days / 365.25
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Start Date", actual_start.strftime('%Y-%m-%d'))
        with col2:
            st.metric("End Date", actual_end.strftime('%Y-%m-%d'))
        with col3:
            st.metric("Total Days", f"{actual_days}")
        
        # Show warning if requested range doesn't match available data
        if requested_start and actual_start > requested_start:
            days_diff = (actual_start - requested_start).days
            st.warning(
                f"⚠️ **Data Availability Warning:**\n\n"
                f"Requested: {requested_start} to {requested_end} ({requested_years} years)\n\n"
                f"Available: {actual_start} to {actual_end} ({actual_years:.1f} years)\n\n"
                f"**Missing {days_diff} days** of data at the start. "
                f"This is likely because one or more tickers only have data from {actual_start}. "
                f"Check the individual ticker date ranges below to see which ticker is limiting the range."
            )
        
        # Show date range for each ticker (before alignment)
        with st.expander("📅 Individual Ticker Date Ranges (Before Alignment)"):
            timeline_data = []
            if self.manager and self.manager.data:
                for ticker, df in self.manager.data.items():
                    if df.empty:
                        continue
                    # Get the actual data range from raw data
                    if isinstance(df.index, pd.DatetimeIndex):
                        ticker_start = df.index.min().date()
                        ticker_end = df.index.max().date()
                        ticker_days = len(df)
                    else:
                        ticker_start = "N/A"
                        ticker_end = "N/A"
                        ticker_days = len(df)
                    
                    # Get the range after filtering (in returns_df)
                    if ticker in returns_df.columns:
                        ticker_returns = returns_df[ticker].dropna()
                        if len(ticker_returns) > 0:
                            filtered_start = ticker_returns.index[0].date()
                            filtered_end = ticker_returns.index[-1].date()
                            filtered_days = len(ticker_returns)
                        else:
                            filtered_start = "N/A"
                            filtered_end = "N/A"
                            filtered_days = 0
                    else:
                        filtered_start = "N/A"
                        filtered_end = "N/A"
                        filtered_days = 0
                    
                    timeline_data.append({
                        'Ticker': ticker,
                        'Raw Data Start': str(ticker_start),
                        'Raw Data End': str(ticker_end),
                        'Raw Data Days': ticker_days,
                        'After Filter Start': str(filtered_start),
                        'After Filter End': str(filtered_end),
                        'After Filter Days': filtered_days
                    })
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline, use_container_width=True, hide_index=True)
                
                # Highlight which ticker is limiting the range
                if len(timeline_data) > 1:
                    # Find the ticker with the latest start date (after filter)
                    latest_start_ticker = None
                    latest_start_date = None
                    for item in timeline_data:
                        if item['After Filter Start'] != 'N/A':
                            try:
                                start_date = pd.to_datetime(item['After Filter Start']).date()
                                if latest_start_date is None or start_date > latest_start_date:
                                    latest_start_date = start_date
                                    latest_start_ticker = item['Ticker']
                            except:
                                pass
                    
                    if latest_start_ticker and latest_start_date == actual_start:
                        st.info(f"🔍 **Limiting Ticker:** `{latest_start_ticker}` has the latest start date ({actual_start}), which limits the portfolio's date range.")
            else:
                st.info("No date range data available")
    
    def render_positions_table(self) -> None:
        """Render positions table."""
        if self.manager is None or self.manager.analyzer is None:
            return
        
        initial_capital = st.session_state.get('config', {}).get('initial_capital', 100000.0)
        positions = self.manager.get_positions(initial_capital=initial_capital)
        df_positions = MetricsDisplay.format_positions(positions)
        
        st.subheader("Position Details")
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
    
    def render_leverage_analysis(self) -> None:
        """Render leverage analysis section."""
        if self.manager is None or self.manager.analyzer is None:
            return
        
        metrics = self.manager.get_metrics()
        
        st.subheader("Leverage Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Leverage Multiplier", f"{metrics.leverage:.2f}x")
            st.caption("Applied leverage on portfolio")
        
        with col2:
            # Effective leverage based on volatility
            vol_ratio = metrics.annualized_volatility / 0.15  # vs 15% baseline
            st.metric("Volatility-Adjusted Leverage", f"{vol_ratio:.2f}x")
            st.caption("Relative to 15% baseline volatility")
        
        with col3:
            st.metric("Risk-Adjusted Return", f"{metrics.sharpe_ratio:.2f}")
            st.caption("Sharpe ratio (return/volatility)")
    
    def render_live_portfolio(self) -> None:
        """Render live portfolio positions from IBKR."""
        st.header("💰 Live Portfolio Positions")
        
        # Connect button
        if 'live_manager' not in st.session_state:
            st.session_state['live_manager'] = LivePortfolioManager()
        
        live_manager = st.session_state['live_manager']
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔄 Refresh Positions", type="primary"):
                st.session_state['live_refresh'] = True
        with col2:
            if live_manager.connected:
                if st.button("🔌 Disconnect"):
                    with st.spinner("Disconnecting..."):
                        asyncio.run(live_manager.disconnect())
                    st.success("Disconnected from IBKR")
                    st.rerun()
        
        # Account selection
        if not live_manager.connected:
            if st.button("Connect to IBKR", type="primary"):
                with st.spinner("Connecting to IBKR..."):
                    success = asyncio.run(live_manager.connect())
                    if success:
                        st.success("Connected to IBKR!")
                        st.rerun()
                    else:
                        st.error("Failed to connect. Make sure TWS/IB Gateway is running.")
        
        if not live_manager.connected:
            st.info("👈 Click 'Connect to IBKR' to view your live portfolio positions")
            return
        
        # Get account list
        with st.spinner("Loading accounts..."):
            accounts = asyncio.run(live_manager.get_all_accounts())
        
        if not accounts:
            st.warning("No accounts found. Make sure TWS/IB Gateway is running and you have positions.")
            return
        
        # Account selector
        selected_account = st.selectbox("Select Account", accounts, key="live_account")
        
        # Refresh if button clicked
        if st.session_state.get('live_refresh', False):
            st.session_state['live_refresh'] = False
        
        # Get account summary and positions
        with st.spinner("Loading account data..."):
            account_summary = asyncio.run(live_manager.get_account_summary(selected_account))
            positions = asyncio.run(live_manager.get_positions(selected_account))
        
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"**Selected Account:** {selected_account}")
            st.write(f"**Accounts Available:** {accounts}")
            st.write(f"**Account Summary:** {account_summary}")
            st.write(f"**Positions Found:** {len(positions)}")
            
            # Try to get raw portfolio data
            if live_manager.connected:
                try:
                    portfolio_raw = live_manager.ib.portfolio()
                    positions_raw = live_manager.ib.positions()
                    st.write(f"**Raw portfolio() items:** {len(portfolio_raw)}")
                    st.write(f"**Raw positions() items:** {len(positions_raw)}")
                    
                    if portfolio_raw:
                        st.write("**Portfolio accounts:**", set(item.account for item in portfolio_raw))
                    if positions_raw:
                        st.write("**Positions accounts:**", set(pos.account for pos in positions_raw))
                except Exception as e:
                    st.write(f"**Error getting raw data:** {e}")
        
        if account_summary is None:
            st.error("Could not load account summary")
            return
        
        # Display account summary
        st.subheader("Account Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Net Liquidation", f"${account_summary.net_liquidation:,.2f}")
        with col2:
            st.metric("Total Cash", f"${account_summary.total_cash:,.2f}" if account_summary.total_cash else "N/A")
        with col3:
            st.metric("Available Cash", f"${account_summary.available_cash:,.2f}" if account_summary.available_cash else "N/A")
        with col4:
            st.metric("Buying Power", f"${account_summary.buying_power:,.2f}" if account_summary.buying_power else "N/A")
        
        # Calculate leverage
        if account_summary.gross_position_value and account_summary.net_liquidation:
            leverage = account_summary.gross_position_value / account_summary.net_liquidation
            st.metric("Portfolio Leverage", f"{leverage:.2f}x")
        
        # Display positions
        if not positions:
            st.info("No positions found in this account")
            return
        
        st.subheader("Positions")
        
        # Create positions DataFrame
        positions_data = []
        total_market_value = 0
        total_unrealized_pnl = 0
        
        for pos in positions:
            positions_data.append({
                'Ticker': pos.ticker,
                'Position': pos.position,
                'Market Price': f"${pos.market_price:.2f}",
                'Market Value': f"${pos.market_value:,.2f}",
                '% of Portfolio': f"{pos.allocation_pct:.2f}%",
                'Avg Cost': f"${pos.avg_cost:.2f}",
                'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}",
            })
            total_market_value += pos.market_value
            total_unrealized_pnl += pos.unrealized_pnl
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True)
        
        # Summary
        st.subheader("Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Market Value", f"${total_market_value:,.2f}")
        with col2:
            st.metric("Total Unrealized P&L", f"${total_unrealized_pnl:,.2f}")
        with col3:
            pnl_color = "normal" if total_unrealized_pnl >= 0 else "inverse"
            st.metric("P&L %", f"{(total_unrealized_pnl / account_summary.net_liquidation * 100):.2f}%")
        
        # Allocation chart
        st.subheader("Portfolio Allocation")
        # Create PositionInfo-like objects for the chart
        from portfolio_manager import PositionInfo as BacktestPositionInfo
        chart_positions = [
            BacktestPositionInfo(
                ticker=pos.ticker,
                weight=pos.allocation_pct / 100.0,
                allocation_pct=pos.allocation_pct,
                total_return=0.0,  # Not available for live positions
                annualized_return=0.0,
                market_value=pos.market_value,
                unrealized_pnl=pos.unrealized_pnl
            )
            for pos in positions
        ]
        fig_allocation = ChartGenerator.create_allocation_chart(chart_positions)
        st.plotly_chart(fig_allocation, use_container_width=True, key="live_allocation_chart")
    
    def run(self) -> None:
        """Main GUI execution loop."""
        st.set_page_config(page_title="Portfolio Analysis", layout="wide")
        st.title("📊 Interactive Portfolio Analysis")
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Store config in session state for later use
        st.session_state['config'] = config
        
        # Load data button
        if st.sidebar.button("Load Data", type="primary"):
            with st.spinner(f"Loading data for {len(config['tickers'])} tickers..."):
                success = asyncio.run(
                    self.load_data(
                        config['tickers'],
                        config['years'],
                        config['auto_download']
                    )
                )
                
                if success:
                    # Analyze portfolio
                    self.manager.analyze_portfolio(
                        weights=config['weights'],  # Use weights from config (None = equal weights)
                        leverage=config['leverage'],
                        start_date=config['start_date'],
                        end_date=config['end_date']
                    )
                    st.session_state['portfolio_loaded'] = True
                    st.success(f"Loaded data for {len(config['tickers'])} tickers!")
                else:
                    st.session_state['portfolio_loaded'] = False
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Backtest Analysis", "🔗 Correlation Builder", "💰 Live Portfolio"])
        
        with tab1:
            # Main content - Backtest Analysis
            if not st.session_state.get('portfolio_loaded', False) or self.manager is None:
                st.info("👈 Configure your portfolio in the sidebar and click 'Load Data'")
            else:
                # Render all components
                metrics = self.manager.get_metrics()
                self.render_metrics(metrics)
                self.render_charts()
                self.render_positions_table()
                self.render_leverage_analysis()
        
        with tab2:
            # Correlation Builder Tab
            if not st.session_state.get('portfolio_loaded', False) or self.manager is None:
                st.info("👈 Configure your portfolio in the sidebar and click 'Load Data' first")
            else:
                self.render_correlation_builder()
        
        with tab3:
            # Live Portfolio Tab
            self.render_live_portfolio()
    
    def render_correlation_builder(self) -> None:
        """Render correlation builder with adjustable lookback period."""
        if self.manager is None or self.manager.analyzer is None:
            return
        
        st.header("🔗 Correlation Builder")
        st.write("Analyze correlation between portfolio positions with adjustable time periods.")
        
        returns_df = self.manager.analyzer.returns_df
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            max_days = len(returns_df)
            use_lookback = st.checkbox("Use lookback period", value=False)
        with col2:
            if use_lookback:
                lookback_days = st.slider(
                    "Lookback days",
                    min_value=30,
                    max_value=max_days,
                    value=min(252, max_days),  # Default to 1 year or max available
                    step=1,
                    help="Number of trading days to use for correlation calculation"
                )
            else:
                lookback_days = None
        with col3:
            if use_lookback:
                st.metric("Period", f"{lookback_days} days ({lookback_days/252:.1f} years)")
            else:
                total_days = len(returns_df)
                st.metric("Period", f"{total_days} days ({total_days/252:.1f} years)")
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig_corr = ChartGenerator.create_correlation_matrix(returns_df, lookback_days=lookback_days)
        # Use lookback_days in key to make it unique when slider changes
        key_suffix = f"_{lookback_days}" if lookback_days else "_full"
        st.plotly_chart(fig_corr, use_container_width=True, key=f"correlation_builder_chart{key_suffix}")
        
        # Correlation statistics
        st.subheader("Correlation Statistics")
        
        if lookback_days is not None:
            corr_data = returns_df.tail(lookback_days).corr()
        else:
            corr_data = returns_df.corr()
        
        # Get upper triangle (avoid duplicates)
        import numpy as np
        mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
        corr_values = corr_data.where(mask).stack()
        corr_values = corr_data.where(mask).stack()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Correlation", f"{corr_values.mean():.3f}")
        with col2:
            st.metric("Min Correlation", f"{corr_values.min():.3f}")
        with col3:
            st.metric("Max Correlation", f"{corr_values.max():.3f}")
        with col4:
            st.metric("Std Deviation", f"{corr_values.std():.3f}")
        
        # Detailed correlation table
        with st.expander("📋 Detailed Correlation Table"):
            st.dataframe(corr_data, use_container_width=True)
        
        # Pairwise correlations
        st.subheader("Pairwise Correlations")
        pairs_data = []
        for i, ticker1 in enumerate(corr_data.columns):
            for j, ticker2 in enumerate(corr_data.columns):
                if i < j:  # Only upper triangle
                    corr_value = corr_data.loc[ticker1, ticker2]
                    pairs_data.append({
                        'Ticker 1': ticker1,
                        'Ticker 2': ticker2,
                        'Correlation': corr_value
                    })
        
        if pairs_data:
            pairs_df = pd.DataFrame(pairs_data)
            pairs_df = pairs_df.sort_values('Correlation', ascending=False)
            # Format correlation for display
            pairs_df['Correlation'] = pairs_df['Correlation'].apply(lambda x: f"{x:.3f}")
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)


def main():
    """Entry point for Streamlit app."""
    gui = PortfolioGUI()
    gui.run()


if __name__ == "__main__":
    main()
