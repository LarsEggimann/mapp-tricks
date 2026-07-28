import os
from typing import Tuple
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import plotly.graph_objects as go # type: ignore

from .models import PeakFitResult, linear_func, gaussian_func, SpectrometryData
from ..plotting import apply_my_plotly_style

def plot_matplotlib(
    peakfit_result: PeakFitResult,
    save_path: str,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Create and save a matplotlib plot of the peak fit.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = peakfit_result.energy_bins
    y = peakfit_result.counts

    ax.plot(
        x,
        y,
        label="Spectrum",
        color="black",
        drawstyle="steps-mid",
    )

    # Background fit
    linear_y = linear_func(
        x,
        peakfit_result.m.n,
        peakfit_result.b.n,
    )
    ax.plot(
        x,
        linear_y,
        label="Background Fit",
        color="red",
    )

    # Gaussian fit
    gauss_y = gaussian_func(
        x,
        peakfit_result.amp.n,
        peakfit_result.mu.n,
        peakfit_result.sigma.n,
    )
    ax.plot(
        x,
        linear_y + gauss_y,
        label="Total Fit",
        color="blue",
    )

    # Centroid line
    ax.axvline(
        x=peakfit_result.mu.n,
        color="green",
        linestyle="--",
        label=f"Centroid: {peakfit_result.mu:.uS} keV",
    )

    # Area text
    area_text = f"Area: {peakfit_result.area:.uS} counts"
    ax.text(
        0.05,
        0.95,
        area_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Formatting
    filename = os.path.basename(peakfit_result.file_name)
    ax.set_title(
        f"Peak fit for: {filename} at {peakfit_result.mu:.uS} keV"
    )
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_plotly(
        spectrometry_data: SpectrometryData,
        peakfit_results: list[PeakFitResult] | None = None,
        save_path: str | None = None,
) -> go.Figure:
    """
    Create a Plotly plot of the spectrometry data, add peak fit results if provided, and optionally save the plot to a file.
    """
    fig = go.Figure()

    # Add the spectrometry data
    fig.add_trace(
        go.Scatter(
            x=spectrometry_data.energy,
            y=spectrometry_data.channels_data,
            mode='lines',
            name='Spectrum',
            line=dict(color='black', shape='hvh', width=0.5),  # hvh creates step-like appearance, centered
        )
    )

    if peakfit_results is not None:
        for peakfit_result in peakfit_results:
            x = np.linspace(peakfit_result.energy_range[0], peakfit_result.energy_range[1], 1000)
            # Background fit
            linear_y = linear_func(
                x,
                peakfit_result.m.n,
                peakfit_result.b.n,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=linear_y,
                    mode='lines',
                    name='Background Fit',
                    line=dict(color='red'),
                )
            )

            # Gaussian fit
            gauss_y = gaussian_func(
                x,
                peakfit_result.amp.n,
                peakfit_result.mu.n,
                peakfit_result.sigma.n,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=linear_y + gauss_y,
                    mode='lines',
                    name='Total Fit',
                    line=dict(color='blue'),
                )
            )

            # Centroid line
            fig.add_trace(
                go.Scatter(
                    x=[peakfit_result.mu.n, peakfit_result.mu.n],
                    y=[0, max(spectrometry_data.channels_data)],
                    mode='lines',
                    name=f'Centroid: {peakfit_result.mu:.uS} keV',
                    line=dict(color='green', dash='dash'),
                )
            )

    # Update layout
    filename = os.path.basename(spectrometry_data.original_file_name)
    title_text = f"Peak fit for: {filename}"
    if peakfit_results is not None:
        title_text += f" and {len(peakfit_results)} peak(s)"
    fig.update_layout(
        title=title_text,
        xaxis_title="Energy [keV]",
        yaxis_title="Counts",
        yaxis_type="log",
    )
    fig = apply_my_plotly_style(fig)


    if save_path is not None:
        fig.write_image(save_path)

    return fig
