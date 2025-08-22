"""
Plotting functions for peak fitting visualization.

This module provides matplotlib and plotly plotting functions
for visualizing peak fits and spectra.
"""

import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def linear_func(x: np.ndarray, m: float, b: float):
    """Linear function: y = mx + b"""
    return m * x + b


def gaussian_func(x: np.ndarray, amp: float, center: float, sigma: float):
    """Gaussian function"""
    return amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def plot_matplotlib(res: dict, save_path: Optional[str] = None, 
                   figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Create a matplotlib plot of the peak fit.
    
    Parameters
    ----------
    res : dict
        Results dictionary from peak fitting
    save_path : str, optional
        Path to save the plot. If None, shows the plot
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot the spectrum
    plt.plot(res['x'], res['y'], label='Spectrum', color='black', drawstyle='steps-mid')
    
    # Plot the background fit
    linear_y = linear_func(res['x'], 
                          res['slope'], 
                          res['intercept'])
    plt.plot(res['x'], linear_y, label='Background Fit', color='red')
    
    # Plot the Gaussian fit
    gauss_y = gaussian_func(res['x'],
                           res['amplitude'], 
                           res['centroid'],
                           res['sigma'])
    plt.plot(res['x'], linear_y + gauss_y, label='Total Fit', color='blue')
    
    # Add centroid line
    plt.axvline(x=res['centroid'], color='green', linestyle='--', 
                label=f'Centroid: {res["centroid"]:.2f} keV')
    
    # Add area text
    area_text = f"Area: {res['area']:.2f} ± {res['area_err']:.2f}"
    plt.text(0.05, 0.95, area_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Labels and formatting
    filename = os.path.basename(res.get('filename', 'Unknown'))
    plt.title(f"Peak fit for: {filename}")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_plotly(res: dict, df: pd.DataFrame, 
               range_x: Optional[Tuple[float, float]] = None,
               range_y: Optional[Tuple[float, float]] = None,
               save_path: Optional[str] = None) -> None:
    """
    Create an interactive plotly plot of the peak fit.
    
    Parameters
    ----------
    res : dict
        Results dictionary from peak fitting
    df : pd.DataFrame
        Full spectrum dataframe
    range_x : tuple, optional
        X-axis range (min, max)
    range_y : tuple, optional
        Y-axis range (min, max) - will be converted to log scale
    save_path : str, optional
        Path to save HTML file. If None, shows the plot
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not available. Install with: pip install plotly")
    
    # Create the main figure
    fig = go.Figure()
    
    # Add spectrum data
    fig.add_trace(go.Scatter(
        x=df['energy'], 
        y=df['counts'],
        mode='lines',
        name='Spectrum',
        line=dict(color='black', shape='hv'),  # hv creates step-like appearance
        hovertemplate='Energy: %{x:.2f} keV<br>Counts: %{y}<extra></extra>'
    ))

    # Calculate fit components
    linear_y = linear_func(res['x'], 
                          res['slope'], 
                          res['intercept'])
    gauss_y = gaussian_func(res['x'],
                           res['amplitude'], 
                           res['centroid'],
                           res['sigma'])
    
    # Add background fit
    fig.add_trace(go.Scatter(
        x=res['x'], 
        y=linear_y,
        mode='lines',
        name='Background Fit',
        line=dict(color='red'),
        hovertemplate='Energy: %{x:.2f} keV<br>Background: %{y:.2f}<extra></extra>'
    ))
    
    # Add total fit
    fig.add_trace(go.Scatter(
        x=res['x'], 
        y=linear_y + gauss_y,
        mode='lines',
        name='Total Fit',
        line=dict(color='blue'),
        hovertemplate='Energy: %{x:.2f} keV<br>Total Fit: %{y:.2f}<extra></extra>'
    ))
    
    # Add centroid vertical line
    fig.add_vline(
        x=res['centroid'], 
        line_color='green',
        line_dash='dash',
        annotation_text=f"Centroid: {res['centroid']:.2f} keV",
        annotation_position="top"
    )

    # Update layout
    filename = os.path.basename(res.get('filename', 'Unknown'))
    fig.update_layout(
        title=f"Gaussian + Linear Background Fit, file: {filename}",
        xaxis_title="Energy (keV)",
        yaxis_title="Counts (log scale)",
        hovermode='x unified',
        showlegend=True,
        height=800,
        template='plotly_white'
    )
    
    # Add grid and set log scale for y-axis
    max_counts = max(df['counts'])
    log_max = np.log10(max_counts) + 0.5  # Add some padding
    fig.update_yaxes(
        type="log",
        range=[0, log_max],
    )

    if range_x:
        fig.update_xaxes(range=range_x)
    if range_y:
        fig.update_yaxes(range=np.log10(range_y))

    # Save or show
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
