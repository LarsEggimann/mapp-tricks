"""
Core fitting functionality for peak analysis.

This module provides the main PeakFitter class and related functions
for fitting Gaussian peaks with linear backgrounds.
"""

import os
import glob
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd # type: ignore
from uncertainties import unumpy as unp # type: ignore
import uncertainties as unc # type: ignore
from tqdm.auto import tqdm # type: ignore
from scipy.optimize import curve_fit # type: ignore
import matplotlib.pyplot as plt # type: ignore

from .plotting import plot_matplotlib
from .parser import parse_spectrum_file
from .models import PeakFitResult, SpectrometryData, linear_gaussian_model

def save_peak_plot(args: tuple[str, PeakFitResult, str]) -> None:
    file, pf_res, plots_dir = args

    plots_base_filename = f"{Path(file).name}_{int(pf_res.mu.n)}keV"

    plot_matplotlib(
        pf_res,
        save_path=os.path.join(
            plots_dir,
            f"{plots_base_filename}.pdf",
        ),
    )

class PeakFitter:
    """
    A class for fitting gaussian peaks with linear backgrounds in spectroscopy data.
    
    This class provides methods to fit peaks in a specified energy range,
    extract peak parameters, and process multiple files in batch.

    Parameters
    ----------
    timezone : ZoneInfo
        Timezone of the time strings in the spectrum files, default is "Europe/Zurich". This is used for converting strings to datetime objects.
    """
    def __init__(self, timezone: ZoneInfo = ZoneInfo("Europe/Zurich")):
        self.timezone: ZoneInfo = timezone
        
    def fit_peak(self, spectra_data: SpectrometryData, energy_range: Tuple[float, float]) -> PeakFitResult:
        """
        Fit a Gaussian peak with linear background in the specified energy range.
        
        Parameters
        ----------
        spectra_data : SpectrometryData
            Dataclass containing spectrum information
        energy_range : tuple[float, float]
            (min_energy, max_energy) for the fitting range
        
        Returns
        -------
        PeakFitResult
            Dataclass containing the fit results and parameters
        """
        # assert that energy_range is a tuple of two floats
        assert isinstance(energy_range, tuple) and len(energy_range) == 2, f"energy_range must be a tuple of two values, got {energy_range}"
        assert all(isinstance(e, (int, float)) for e in energy_range), f"energy_range values must be numbers, got {energy_range}, types: {[type(e) for e in energy_range]}"

        # filter data to energy range
        energy_arr = np.array(spectra_data.energy)
        counts_arr = np.array(spectra_data.channels_data)
        mask = (energy_arr >= energy_range[0]) & (energy_arr <= energy_range[1])
        x = energy_arr[mask]
        y = counts_arr[mask]

        # try decrease fit convergence time by providing somewhat reasonable initial guesses for the parameters
        m_0 = 0
        b_0 = np.mean(y)
        mu_0 = x[np.argmax(y)]
        sigma_0 = (energy_range[1] - energy_range[0]) / 6
        peak_height_0 = max(0, np.max(y) - b_0)
        amp_0 = peak_height_0 * sigma_0 * np.sqrt(2 * np.pi)

        popt, pcov = curve_fit(
            linear_gaussian_model,
            x,
            y,
            p0=[m_0, b_0, amp_0, mu_0, sigma_0],
            bounds=(
                [-np.inf, -np.inf, 0,      energy_range[0], 0     ], # amp >= 0, sigma >= 0,  mu within fitting range
                [ np.inf,  np.inf, np.inf, energy_range[1], np.inf],
            ),
        )

        # use uncertainties package to convert popt and pcov and preserve correlated uncertainties
        cv_u = unc.correlated_values(popt, pcov)

        m, b, amp, mu, sigma = cv_u

        # Calculate area with uncertainty
        area = amp / (x[1] - x[0])

        # check peak significance
        snr = area.n / area.s
        if snr < 3:
            warnings.warn(
                f"Poor peak significance for {spectra_data.original_file_name}: "
                f"area = {area:.uS}, significance = {snr:.1f}σ",
                RuntimeWarning,
            )

        return PeakFitResult(
            area=area,
            mu=mu,
            amp=amp,
            sigma=sigma,
            m=m,
            b=b,
            energy_range=energy_range,
            energy_bins=x,
            counts=y,
            file_name=spectra_data.original_file_name,
            start_time=spectra_data.start_time,
            real_time=spectra_data.real_time,
            live_time=spectra_data.live_time
        )
    
    def process_file(self,
                     filepath: str,
                     energy_range: Tuple[float, float],
                     save_plots: bool = True,
                     save_results: bool = True,
                     output_dir: str | None = None,
                     verbose: bool = True
                ) -> PeakFitResult:
        """
        Process a single spectrum file. This is a convenience method that wraps around process_folder to handle a single file by extracting the parent folder and file name.

        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        energy_range : tuple
            (min_energy, max_energy) for the fitting range
        output_dir : str, optional
            Directory to save output files

        Returns
        -------
        PeakFitResult
            Fit result for the processed file
        """
        
        parent_folder = os.path.dirname(filepath)
        if not os.path.exists(parent_folder):
            raise FileNotFoundError(f"Parent folder does not exist: {parent_folder}")
        
        file_name = os.path.basename(filepath)

        res = self.process_folder(
            folder_path=parent_folder,
            energy_range=energy_range,
            output_dir=output_dir,
            file_pattern=file_name,
            save_plots=save_plots,
            save_results=save_results,
            verbose=verbose,
        )

        # assert that we got exactly one result back wit assert
        assert len(res) == 1, f"Expected exactly one result from processing file {filepath}, but got {len(res)} results."

        return res[0]
    
    def process_file_multiple_peaks(
            self,
            filepath: str,
            energy_ranges: List[Tuple[float, float]],
            output_dir: str | None = None,
            save_plots: bool = True,
            ) -> list[Tuple[Tuple[float, float], PeakFitResult]]:
        """
        Process a single spectrum file for multiple peaks, useful for calibrations and isotopes with multiple peaks.
        Here I just wrap around process_folder to handle a single file by extracting the parent folder and file name.

        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        energy_ranges : list of tuple
            List of (min_energy, max_energy) for the fitting ranges for each peak
        output_dir : str, optional
            Directory to save output files

        Returns
        -------
        list of Tuple[Tuple[float, float], PeakFitResult]
            List containing the tuple of energy range (also a tuple) and fit result for each processed peak
        """

        # make sure file and parent folder exist
        filepath = os.path.abspath(filepath)
        parent_folder = os.path.dirname(filepath)
        if not os.path.exists(parent_folder):
            raise FileNotFoundError(f"Parent folder does not exist: {parent_folder}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")
        file_name = os.path.basename(filepath)

        if output_dir is None:
            output_dir = os.path.join(parent_folder, "results")

        results: list[Tuple[Tuple[float, float], PeakFitResult]] = []

        for energy_range in tqdm(energy_ranges, desc="peakfit - processing energy ranges"):
            pf_res = self.process_folder(
                folder_path=parent_folder,
                energy_range=energy_range,
                output_dir=output_dir,
                save_plots=save_plots,
                save_results=False, # we set this to False to not create a CSV for each peak
                verbose=False, # we don't want to show progress and prints for each peak since we already have a progress bar for the energy ranges
                file_pattern=file_name,
            )[0]  # we expect exactly one result for each energy range, so we take the first element

            results.append((energy_range, pf_res))
        return results

    def process_folder(self,
                        folder_path: str, energy_range: Tuple[float, float],
                        output_dir: str | None = None,
                        save_plots: bool = True,
                        save_results: bool = True,
                        verbose: bool = True,
                        file_pattern: str = "*.cnf",
                    ) -> list[PeakFitResult]:
        """
        Process all spectrum files in a folder.
        
        Parameters
        ----------
        folder_path : str
            Path to folder containing spectrum files
        energy_range : tuple[float, float]
            (min_energy, max_energy) for the fitting range
        output_dir : str, optional
            Directory to save results. If None, uses folder_path/results
        save_plots : bool, default True
            Whether to save matplotlib plots
        save_results : bool, default True
            Whether to save fit results to CSV
        file_pattern : str, default "*.cnf"
            File pattern to match

        Returns
        -------
        list[PeakFitResult]
            List containing all fit results
        """
        
        
        # Find files
        # if cnf or CNF look for both
        files = glob.glob(os.path.join(folder_path, file_pattern))
        if not files:
            files = glob.glob(os.path.join(folder_path, file_pattern.upper()))

        # try to check for txt files just because a lot of places i already use this and assume there are txt files
        # this is of course not a good idea since if there are both cnf and txt files in the folder it will process only the cnf files and ignore the txt files.
        if not files:
            files = glob.glob(os.path.join(folder_path, file_pattern.replace(".cnf", ".txt")))
        
        # Try to sort files numerically if they follow numeric pattern, otherwise sort alphabetically
        def sort_key(x):
            basename = os.path.basename(x).split('.')[0]
            try:
                # Return (0, integer) so numeric files are grouped first and sorted by value
                return (0, int(basename))
            except ValueError:
                # If not a number, return (1, string) so they are grouped together alphabetically
                return (1, basename)
            
        files = sorted(files, key=sort_key)
        
        if not files:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {folder_path}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(folder_path, "mapp_tricks_results")
        
        if save_plots:
            plots_dir = os.path.join(output_dir, "peakfit_plot_fits")
            os.makedirs(plots_dir, exist_ok=True)
        
        results: list[tuple[str, PeakFitResult]] = []

        
        # process files first
        for file in tqdm(files, desc="peakfit - processing files", disable=not verbose):
            try:
                sd = parse_spectrum_file(file, timezone=self.timezone)
                pf_res = self.fit_peak(sd, energy_range)

                results.append((file, pf_res))

            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        # helper to save plots in parallel
        if save_plots:
            tasks = [
                (file, pf_res, plots_dir)
                for file, pf_res in results
            ]
            max_workers = min(8, os.cpu_count() or 1)  # use up to 8 workers or the number of CPUs available

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                list(
                    tqdm(
                        executor.map(save_peak_plot, tasks),
                        total=len(tasks),
                        desc=f"peakfit - saving plots using {max_workers} workers",
                        disable=not verbose,
                    )
                )

        # get first mu value to use in the output filename
        peakfit_results = [res for _, res in results]
        first_mu = int(peakfit_results[0].mu.n) if peakfit_results else 'XX'
        file_name = f'{first_mu}keV_peakfit_results.csv'

        # convert results to DataFrame and save to CSV
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame([res.to_dict() for res in peakfit_results])
            df.to_csv(os.path.join(output_dir, file_name), index=False)

        if verbose:
            print(f"peakfit - processed {len(results)} files and saved results to {output_dir}/{file_name}")
        return peakfit_results


