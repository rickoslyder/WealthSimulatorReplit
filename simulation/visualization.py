import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib import colors

def plot_wealth_distribution(wealth_array, bins=30):
    """
    Create a histogram of wealth distribution.
    
    Args:
        wealth_array: Array of wealth values
        bins: Number of bins for the histogram
        
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        x=wealth_array,
        nbins=bins,
        labels={'x': 'Wealth ($)'},
        title='Wealth Distribution',
        opacity=0.8,
        color_discrete_sequence=['#3366CC']
    )
    
    # Add median and mean lines
    median_wealth = np.median(wealth_array)
    mean_wealth = np.mean(wealth_array)
    
    fig.add_vline(
        x=median_wealth,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: ${median_wealth:,.0f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=mean_wealth,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_wealth:,.0f}",
        annotation_position="top right"
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Wealth ($)",
        yaxis_title="Number of Agents",
        bargap=0.2,
        xaxis=dict(
            tickformat="$,.0f",
            showgrid=True
        )
    )
    
    return fig

def plot_gini_coefficient(gini_history):
    """
    Create a line chart of Gini coefficient over time.
    
    Args:
        gini_history: List or array of Gini coefficients
        
    Returns:
        Plotly figure object
    """
    years = list(range(len(gini_history)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=gini_history,
        mode='lines+markers',
        name='Gini Coefficient',
        line=dict(color='#FF6347', width=2),
        marker=dict(size=6, color='#FF6347')
    ))
    
    # Add reference lines for common Gini values
    fig.add_hline(
        y=0.4,
        line_dash="dot",
        line_color="gray",
        annotation_text="Moderate Inequality (0.4)",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=0.6,
        line_dash="dot",
        line_color="gray",
        annotation_text="High Inequality (0.6)",
        annotation_position="right"
    )
    
    # Add initial Gini at year 0
    fig.add_annotation(
        x=0,
        y=gini_history[0],
        text=f"Initial: {gini_history[0]:.3f}",
        showarrow=True,
        arrowhead=1
    )
    
    # Add final Gini at last year
    fig.add_annotation(
        x=years[-1],
        y=gini_history[-1],
        text=f"Final: {gini_history[-1]:.3f}",
        showarrow=True,
        arrowhead=1
    )
    
    # Update layout
    fig.update_layout(
        title='Wealth Inequality Over Time (Gini Coefficient)',
        xaxis_title='Year',
        yaxis_title='Gini Coefficient',
        yaxis=dict(
            range=[0, 1],
            tickformat=".2f"
        ),
        hovermode="x unified"
    )
    
    return fig

def plot_wealth_shares(top1_history, top10_history, bottom50_history):
    """
    Create a line chart of wealth shares over time.
    
    Args:
        top1_history: List or array of top 1% wealth shares
        top10_history: List or array of top 10% wealth shares
        bottom50_history: List or array of bottom 50% wealth shares
        
    Returns:
        Plotly figure object
    """
    years = list(range(len(top1_history)))
    
    fig = go.Figure()
    
    # Add traces for each wealth group
    fig.add_trace(go.Scatter(
        x=years,
        y=top1_history,
        mode='lines',
        name='Top 1%',
        line=dict(color='#FF6347', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=top10_history,
        mode='lines',
        name='Top 10%',
        line=dict(color='#FF9500', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=bottom50_history,
        mode='lines',
        name='Bottom 50%',
        line=dict(color='#32CD32', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Wealth Shares by Population Group',
        xaxis_title='Year',
        yaxis_title='Wealth Share (%)',
        yaxis=dict(
            tickformat=".1f",
            ticksuffix="%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig

def plot_lorenz_curve(wealth_array, year=None):
    """
    Create a Lorenz curve visualization.
    
    Args:
        wealth_array: Array of wealth values
        year: Optional year label for the title
        
    Returns:
        Plotly figure object
    """
    # Sort wealth values
    sorted_wealth = np.sort(wealth_array)
    n = len(sorted_wealth)
    
    # Calculate cumulative wealth
    cum_wealth = np.cumsum(sorted_wealth)
    
    # Normalize to percentages
    cum_pct_wealth = 100 * cum_wealth / cum_wealth[-1]
    cum_pct_population = 100 * np.arange(1, n + 1) / n
    
    # Add 0,0 point
    cum_pct_population = np.insert(cum_pct_population, 0, 0)
    cum_pct_wealth = np.insert(cum_pct_wealth, 0, 0)
    
    # Create figure
    fig = go.Figure()
    
    # Add perfect equality line
    fig.add_trace(go.Scatter(
        x=cum_pct_population,
        y=cum_pct_population,
        mode='lines',
        name='Perfect Equality',
        line=dict(color='black', width=1, dash='dash')
    ))
    
    # Add Lorenz curve
    fig.add_trace(go.Scatter(
        x=cum_pct_population,
        y=cum_pct_wealth,
        mode='lines',
        name='Lorenz Curve',
        line=dict(color='#FF6347', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 99, 71, 0.2)'
    ))
    
    # Calculate Gini coefficient
    gini = 1 - np.sum((cum_pct_wealth[1:] + cum_pct_wealth[:-1]) * np.diff(cum_pct_population)) / 10000
    
    # Update layout
    title = f'Lorenz Curve of Wealth Distribution - Gini: {gini:.3f}'
    if year is not None:
        title = f'Lorenz Curve of Wealth Distribution - Year {year} - Gini: {gini:.3f}'
        
    fig.update_layout(
        title=title,
        xaxis_title='Cumulative % of Population',
        yaxis_title='Cumulative % of Wealth',
        xaxis=dict(
            tickformat=".0f",
            ticksuffix="%"
        ),
        yaxis=dict(
            tickformat=".0f",
            ticksuffix="%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_wealth_mobility(transition_matrix):
    """
    Create a heatmap visualization of the wealth mobility transition matrix.
    
    Args:
        transition_matrix: 5x5 matrix showing movement between quintiles
        
    Returns:
        Plotly figure object
    """
    quintile_labels = ['Bottom 20%', 'Lower-Mid 20%', 'Middle 20%', 'Upper-Mid 20%', 'Top 20%']
    
    fig = px.imshow(
        transition_matrix,
        x=quintile_labels,
        y=quintile_labels,
        color_continuous_scale='Viridis',
        labels=dict(x="Final Quintile", y="Initial Quintile", color="Probability"),
        title="Wealth Mobility Transition Matrix"
    )
    
    # Add text annotations
    for i in range(5):
        for j in range(5):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{transition_matrix[i, j]:.2f}",
                showarrow=False,
                font=dict(color="white" if transition_matrix[i, j] > 0.3 else "black")
            )
    
    fig.update_layout(
        xaxis=dict(side="top"),
        coloraxis_colorbar=dict(
            title="Probability",
            titleside="right",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0", "0.25", "0.5", "0.75", "1"]
        )
    )
    
    return fig
