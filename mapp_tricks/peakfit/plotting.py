"""
Plotting functions for peak fitting visualization.

This module provides matplotlib and plotly plotting functions
for visualizing peak fits and spectra.
"""

import os
from typing import Tuple
import matplotlib.pyplot as plt # type: ignore
import plotly.graph_objects as go # type: ignore

from .models import PeakFitResult, linear_func, gaussian_func

def plot_matplotlib(
        peakfit_result: PeakFitResult,
        save_path: str, 
        figsize: Tuple[float, float] = (10, 6)
        ):
    """
    Create a matplotlib plot of the peak fit.
    
    Parameters
    ----------
    peakfit_result : PeakFitResult
        Result object from peak fitting
    save_path : str, optional
        Path to save the plot. If None, shows the plot
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
    """
    fig = plt.figure(figsize=figsize)
    
    # Plot the spectrum
    x = peakfit_result.energy_bins
    y = peakfit_result.counts

    plt.plot(x, y, label='Spectrum', color='black', drawstyle='steps-mid')
    
    # background fit
    linear_y = linear_func(x, peakfit_result.m.n, peakfit_result.b.n)
    plt.plot(x, linear_y, label='Background Fit', color='red')
    
    # gaussian fit
    gauss_y = gaussian_func(x, peakfit_result.amp.n, peakfit_result.mu.n, peakfit_result.sigma.n)
    plt.plot(x, linear_y + gauss_y, label='Total Fit', color='blue')
    
    # centroid line
    plt.axvline(x=peakfit_result.mu.n, color='green', linestyle='--', 
                label=f'Centroid: {peakfit_result.mu:.uS} keV')
    
    # Add area text
    area_text = f"Area: {peakfit_result.area:.uS} counts"
    plt.text(0.05, 0.95, area_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Labels and formatting
    filename = os.path.basename(peakfit_result.file_name)
    plt.title(f"Peak fit for: {filename} at {peakfit_result.mu:.uS} keV")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # save no show
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

