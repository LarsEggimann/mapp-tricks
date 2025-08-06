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
from lmfit.models import GaussianModel, LinearModel # type: ignore
from uncertainties import ufloat # type: ignore
from tqdm import tqdm # type: ignore

from .parser import parse_spectrum_file

class PeakFitterResult:
    """
    Class to hold the results of a peak fitting operation.
    
    Attributes
    ----------
    area : ufloat
        Area of the fitted peak in counts
    centroid : ufloat
        Centroid of the peak in keV
    amplitude : ufloat
        Amplitude of the peak in counts
    sigma : ufloat
        Standard deviation of the Gaussian fit in keV
    fwhm : ufloat
        Full width at half maximum of the peak in keV

    Methods
    -------
    __repr__():
        String representation of the result object.
    """
    def __init__(self, area: ufloat, centroid: ufloat,
                 start_time: datetime, real_time: float, live_time: float,
                 amplitude: ufloat, sigma: ufloat,
                 fwhm: ufloat):
        self.area = area
        self.centroid = centroid
        self.start_time = start_time
        self.real_time = real_time
        self.live_time = live_time
        self.amplitude = amplitude
        self.sigma = sigma
        self.fwhm = fwhm


    def __repr__(self):
        return (f"PeakFitterResult(area={self.area:.2f}, "
                f"centroid={self.centroid:.2f}, "
                f"amplitude={self.amplitude:.2f}, "
                f"sigma={self.sigma:.2f}, "
                f"fwhm={self.fwhm:.2f}, "
                f"start_time={self.start_time}, "
                f"real_time={self.real_time:.2f}, "
                f"live_time={self.live_time:.2f})")



def linear_func(x: np.ndarray, m: float, b: float):
    """Linear function: y = mx + b"""
    return m * x + b


def gaussian_func(x: np.ndarray, amp: float, center: float, sigma: float):
    """Gaussian function"""
    return amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


class PeakFitter:
    """
    A class for fitting Gaussian peaks with linear backgrounds in spectroscopy data.
    
    This class provides methods to fit peaks in a specified energy range,
    extract peak parameters, and process multiple files in batch.
    """
    
    def __init__(self):
        """Initialize the PeakFitter with Gaussian and Linear models."""
        self.gauss_model = GaussianModel(prefix='g_')
        self.linear_model = LinearModel(prefix='b_')
        self.model = self.gauss_model + self.linear_model
    
    def fit_peak(self, df: pd.DataFrame, energy_range: Tuple[float, float], 
                 background_params: Optional[Dict] = None,
                 gaussian_params: Optional[Dict] = None) -> Dict:
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
        
        # Set up parameters
        params = self.model.make_params()
        
        # Set background parameters
        bg_params = background_params or {'intercept': 666, 'slope': -0.3}
        params['b_intercept'].set(value=bg_params['intercept'])
        params['b_slope'].set(value=bg_params['slope'])
        
        # Set Gaussian parameters
        peak_pos = x[np.argmax(y)]
        gauss_params = gaussian_params or {'center': peak_pos, 'sigma': 0.9}
        params['g_center'].set(value=gauss_params['center'])
        params['g_amplitude'].set(value=np.sum(y), min=0)
        params['g_sigma'].set(value=gauss_params['sigma'], min=0.2, max=3)
        
        # Perform fit
        result = self.model.fit(y, params, x=x)
        
        # Extract parameters
        amp = result.params['g_amplitude']
        sigma = result.params['g_sigma']
        center = result.params['g_center']
        fwhm = result.params['g_fwhm']
        height = result.params['g_height']
        
        # Calculate area with uncertainty
        area = ufloat(amp.value, amp.stderr) / (x[1] - x[0])
        
        return {
            "area": area.n,
            "area_err": area.s,
            "centroid": center.value,
            "centroid_err": center.stderr,
            "amplitude": amp.value,
            "amplitude_err": amp.stderr,
            "sigma": sigma.value,
            "sigma_err": sigma.stderr,
            "fwhm": fwhm.value,
            "fwhm_err": fwhm.stderr,
            "height": height.value,
            "height_err": height.stderr,
            "fit_params": result.params,
            "x": x,
            "y": y,
            "energy_range": energy_range,
            "fit_result": result
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

        return res[0] if res else None

    def process_folder(self, folder_path: str, energy_range: Tuple[float, float],
                      output_dir: Optional[str] = None,
                      save_plots: bool = True,
                      save_plotly: bool = False,
                      file_pattern: str = "*.txt") -> list[PeakFitterResult]:
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

        print(f"peakfit found {len(files)} files to process.")
        
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
        for file in tqdm(files, desc="Processing files"):
            try:
                # Parse file
                df, calib, start_time, real_time, live_time = parse_spectrum_file(file)

                # Fit peak
                res = self.fit_peak(df, energy_range)
                res["filename"] = file
                res["calibration"] = calib
                res["start_time"] = start_time
                res["real_time"] = real_time
                res["live_time"] = live_time

                results.append(res)

                return_results.append(PeakFitterResult(
                    area=ufloat(res["area"], res["area_err"]),
                    centroid=ufloat(res["centroid"], res["centroid_err"]),
                    start_time=start_time,
                    real_time=real_time,
                    live_time=live_time,
                    amplitude=ufloat(res["amplitude"], res["amplitude_err"]),
                    sigma=ufloat(res["sigma"], res["sigma_err"]),
                    fwhm=ufloat(res["fwhm"], res["fwhm_err"])
                ))
                
                # Save plots
                if save_plots:
                    plot_matplotlib(res, save_path=os.path.join(plots_dir, 
                                                              f"{os.path.basename(file)}.pdf"))
                
                if save_plotly:
                    plot_plotly(res, df, save_path=os.path.join(plots_dir, 
                                                              f"{os.path.basename(file)}.html"))
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        
        # Remove complex objects for CSV export
        csv_results = results_df.drop(columns=['fit_params', 'x', 'y', 'fit_result', 'calibration'], 
                                     errors='ignore')
        
        os.makedirs(output_dir, exist_ok=True)
        csv_results.to_csv(os.path.join(output_dir, "peakfit_results.csv"), index=False)

        print(f"peakfit processed {len(results)} files and saved results to {output_dir}/peakfit_results.csv")
        
        return return_results
