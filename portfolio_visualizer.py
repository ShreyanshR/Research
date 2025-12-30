"""
Portfolio Visualizer - Production-grade visualization components.
Separated from business logic for clean architecture.
"""

from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from portfolio_manager import PositionInfo, PortfolioMetrics


class ChartGenerator:
    """Generates various portfolio charts."""
    
    @staticmethod
    def create_returns_chart(
        portfolio_cumulative: pd.Series,
        individual_cumulative: Dict[str, pd.Series],
        leverage: float,
        initial_capital: float = 1.0
    ) -> go.Figure:
        """
        Create cumulative returns chart (portfolio only).
        
        Args:
            portfolio_cumulative: Portfolio cumulative returns series
            individual_cumulative: Dict of ticker to cumulative returns (not used, kept for API compatibility)
            leverage: Leverage multiplier for labeling
            initial_capital: Initial capital amount for equity calculation
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Calculate equity value from initial capital
        equity_value = portfolio_cumulative * initial_capital
        
        # Add portfolio line only (thick, solid, prominent)
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=(portfolio_cumulative - 1) * 100,
            mode='lines',
            name=f'Portfolio ({leverage}x leverage)',
            line=dict(width=3, color='#1f77b4'),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            hovertemplate='Portfolio<br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )
        
        fig.update_layout(
            title='Portfolio Cumulative Returns (%)',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            height=600,  # Larger height
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_allocation_chart(positions: List[PositionInfo]) -> go.Figure:
        """
        Create portfolio allocation pie chart.
        
        Args:
            positions: List of PositionInfo objects
            
        Returns:
            Plotly figure
        """
        if not positions:
            fig = go.Figure()
            fig.add_annotation(text="No positions to display", showarrow=False)
            return fig
        
        labels = [pos.ticker for pos in positions]
        values = [pos.allocation_pct for pos in positions]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<br>Weight: %{value:.2f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title='Portfolio Allocation',
            height=300  # Reduced size
        )
        
        return fig
    
    @staticmethod
    def create_equity_curve(portfolio_returns: pd.Series) -> go.Figure:
        """
        Create equity curve chart (% return over time).
        
        Args:
            portfolio_returns: Portfolio returns series
            
        Returns:
            Plotly figure
        """
        # Calculate cumulative returns (equity curve)
        cumulative = (1 + portfolio_returns).cumprod()
        equity_pct = (cumulative - 1) * 100  # Convert to percentage
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_pct.index,
            y=equity_pct,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#1f77b4', width=2),
            name='Portfolio Equity',
            hovertemplate='Date: %{x}<br>Equity: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="0%"
        )
        
        fig.update_layout(
            title='Portfolio Equity Curve (%)',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=400,
            hovermode='x unified',
            showlegend=False
        )
        
        return fig


class MetricsDisplay:
    """Formats and displays portfolio metrics."""
    
    @staticmethod
    def format_metrics(metrics: PortfolioMetrics) -> Dict[str, str]:
        """
        Format metrics for display.
        
        Args:
            metrics: PortfolioMetrics object
            
        Returns:
            Dictionary of formatted metric strings
        """
        return {
            'Total Return': f"{metrics.total_return:.2%}",
            'Annualized Return': f"{metrics.annualized_return:.2%}",
            'Annualized Volatility': f"{metrics.annualized_volatility:.2%}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
            'Max Drawdown': f"{metrics.max_drawdown:.2%}",
            'Win Rate': f"{metrics.win_rate:.2%}",
            'Leverage': f"{metrics.leverage:.1f}x",
            'Period (Years)': f"{metrics.years:.2f}"
        }
    
    @staticmethod
    def format_positions(positions: List[PositionInfo]) -> pd.DataFrame:
        """
        Format positions for table display.
        
        Args:
            positions: List of PositionInfo objects
            
        Returns:
            Formatted DataFrame
        """
        data = []
        for pos in positions:
            data.append({
                'Ticker': pos.ticker,
                'Weight': f"{pos.weight:.4f}",
                'Allocation %': f"{pos.allocation_pct:.2f}%",
                'Total Return': f"{pos.total_return:.2%}",
                'Annualized Return': f"{pos.annualized_return:.2%}",
                'Market Value': f"${pos.market_value:,.2f}",
                'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}"
            })
        
        return pd.DataFrame(data)
