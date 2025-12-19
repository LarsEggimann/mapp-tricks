"""
Core fitting functionality for peak analysis.

This module provides the main PeakFitter class and related functions
for fitting Gaussian peaks with linear backgrounds.
"""

import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd # type: ignore
from uncertainties import ufloat # type: ignore
from tqdm.auto import tqdm # type: ignore
from scipy.optimize import curve_fit # type: ignore
import matplotlib.pyplot as plt # type: ignore

from .parser import parse_spectrum_file
from .models import PeakFitterResult

def linear_func(x, m, b):
    """Linear function: y = mx + b"""
    return m * x + b

def gaussian_func(x, amp, center, sigma):
    """Gaussian function"""
    return (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def linear_gaussian_model(x, m, b, amp, center, sigma):
    """Combined linear and Gaussian model."""
    return linear_func(x, m, b) + gaussian_func(x, amp, center, sigma)

class PeakFitter:
    """
    A class for fitting Gaussian peaks with linear backgrounds in spectroscopy data.
    
    This class provides methods to fit peaks in a specified energy range,
    extract peak parameters, and process multiple files in batch.
    """
    
    def __init__(self):
        pass
    
    def fit_peak(self, df: pd.DataFrame, energy_range: Tuple[float, float]) -> Dict:
        """
        Fit a Gaussian peak with linear background in the specified energy range.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['channel', 'energy', 'counts', 'rate']
        energy_range : tuple
            (min_energy, max_energy) for the fitting range
        background_params : dict, optional
            Initial parameters for background fit {'intercept': value, 'slope': value}
        gaussian_params : dict, optional
            Initial parameters for Gaussian fit {'center': value, 'sigma': value}
            
        Returns
        -------
        dict
            Dictionary containing fit results and parameters
        """
        # Filter data to energy range
        df_peak = df[(df['energy'] >= energy_range[0]) & (df['energy'] <= energy_range[1])]
        
        x = df_peak['energy'].values
        y = df_peak['counts'].values
        

        # x and y are your data arrays
        popt, pcov = curve_fit(linear_gaussian_model, x, y, p0=[0, 0, np.sum(y), x[np.argmax(y)], 0.9])
        perr = np.sqrt(np.diag(pcov))  # uncertainties for each parameter

        slope = ufloat(popt[0], perr[0])
        intercept = ufloat(popt[1], perr[1])
        amp = ufloat(popt[2], perr[2])
        center = ufloat(popt[3], perr[3])
        sigma = ufloat(popt[4], perr[4])

        # Calculate area with uncertainty
        area = ufloat(amp.n, amp.s) / (x[1] - x[0])

        return {
            "area": area.n,
            "area_err": area.s,
            "centroid": center.n,
            "centroid_err": center.s,
            "amplitude": amp.n,
            "amplitude_err": amp.s,
            "sigma": sigma.n,
            "sigma_err": sigma.s,
            "x": x,
            "y": y,
            "energy_range": energy_range,
            "slope": slope.n,
            "slope_err": slope.s,
            "intercept": intercept.n,
            "intercept_err": intercept.s,
        }
    
    def process_file(self, filepath: str, energy_range: Tuple[float, float], output_dir: Optional[str] = None,) -> PeakFitterResult:
        """
        Process a single spectrum file.

        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        energy_range : tuple
            (min_energy, max_energy) for the fitting range

        Returns
        -------
        dict
            Dictionary containing fit results and parameters
        """
        
        parent_folder = os.path.dirname(filepath)
        if not os.path.exists(parent_folder):
            raise FileNotFoundError(f"Parent folder does not exist: {parent_folder}")
        
        file_name = os.path.basename(filepath)

        res = self.process_folder(
            folder_path=parent_folder,
            energy_range=energy_range,
            output_dir=output_dir,
            save_plots=True,
            save_plotly=False,
            file_pattern=file_name
        )

        return res[0]
    
    def process_file_multiple_peaks(self, filepath: str, energy_ranges: List[Tuple[float, float]], output_dir: Optional[str] = None,) -> List[PeakFitterResult]:
        """
        Process a single spectrum file for multiple peaks.

        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        energy_ranges : list of tuple
            List of (min_energy, max_energy) for the fitting ranges

        Returns
        -------
        list of PeakFitterResult
            List containing fit results for each peak
        """

        # make sure file and parent folder exist
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")
        parent_folder = os.path.dirname(filepath)
        if not os.path.exists(parent_folder):
            raise FileNotFoundError(f"Parent folder does not exist: {parent_folder}")
        
        file_name = os.path.basename(filepath)

        if output_dir is None:
            output_dir = os.path.join(parent_folder, "results")

        all_results = []

        for energy_range in tqdm(energy_ranges, desc="peakfit - processing energy ranges"):
            res = self.process_folder(
                folder_path=parent_folder,
                energy_range=energy_range,
                output_dir=output_dir,
                save_plots=True,
                save_plotly=False,
                file_pattern=file_name,
                process_multiple_peaks=True
            )
            all_results.append(res[0])

        # store a csv summary of all peaks


        return all_results

    def process_folder(self, folder_path: str, energy_range: Tuple[float, float],
                      output_dir: Optional[str] = None,
                      save_plots: bool = True,
                      save_plotly: bool = False,
                      file_pattern: str = "*.txt",
                      process_multiple_peaks:bool = False) -> list[PeakFitterResult]:
        """
        Process all spectrum files in a folder.
        
        Parameters
        ----------
        folder_path : str
            Path to folder containing spectrum files
        energy_range : tuple
            (min_energy, max_energy) for the fitting range
        output_dir : str, optional
            Directory to save results. If None, uses folder_path/results
        save_plots : bool, default True
            Whether to save matplotlib plots
        save_plotly : bool, default False
            Whether to save interactive plotly plots
        file_pattern : str, default "*.txt"
            File pattern to match
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all fit results
        """
        from .plotting import plot_matplotlib, plot_plotly
        
        # Find files
        files = glob.glob(os.path.join(folder_path, file_pattern))
        # Try to sort files numerically if they follow numeric pattern, otherwise sort alphabetically
        def sort_key(x):
            basename = os.path.basename(x).split('.')[0]
            try:
                return int(basename)
            except ValueError:
                # If not a number, return the string for alphabetical sorting
                return basename
        files = sorted(files, key=sort_key)

        if not process_multiple_peaks:
            print(f"peakfit - found {len(files)} files to process.")
        
        if not files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {folder_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(folder_path, "results")
        
        if save_plots:
            plots_dir = os.path.join(output_dir, "peakfit_fits")
            os.makedirs(plots_dir, exist_ok=True)
        
        results = []

        return_results = []
        
        # Process files
        for file in tqdm(files, desc="peakfit - processing files", disable=process_multiple_peaks):
            try:
                # Parse file
                df, calib, start_time, real_time, live_time, total_gamma_count = parse_spectrum_file(file)

                # Fit peak
                res = self.fit_peak(df, energy_range)
                res["filename"] = file
                res["calibration"] = calib
                res["start_time"] = start_time
                res["real_time"] = real_time
                res["live_time"] = live_time

                results.append(res)

                
                # Save plots
                plots_base_filename = f"{os.path.basename(file)}_{int(res["centroid"])}keV"
                fig = None
                if save_plots:
                    fig = plot_matplotlib(res, save_path=os.path.join(plots_dir, f"{plots_base_filename}.pdf"))
                return_results.append(PeakFitterResult(
                    area=ufloat(res["area"], res["area_err"]),
                    centroid=ufloat(res["centroid"], res["centroid_err"]),
                    start_time=start_time,
                    real_time=real_time,
                    live_time=live_time,
                    amplitude=ufloat(res["amplitude"], res["amplitude_err"]),
                    sigma=ufloat(res["sigma"], res["sigma_err"]),
                    figure=fig
                ))
                
                if save_plotly:
                    plot_plotly(res, df, save_path=os.path.join(plots_dir, f"{plots_base_filename}.html"))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        
        # Remove complex objects for CSV export
        csv_results = results_df.drop(columns=['fit_params', 'x', 'y', 'fit_result', 'calibration'], 
                                     errors='ignore')
        
        os.makedirs(output_dir, exist_ok=True)
        if not process_multiple_peaks:
            csv_results.to_csv(os.path.join(output_dir, "peakfit_results.csv"), index=False)
            print(f"peakfit - processed {len(results)} files and saved results to {output_dir}/peakfit_results.csv")
        
        return return_results
