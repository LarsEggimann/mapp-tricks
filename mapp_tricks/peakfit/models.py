"""
Hold classes for peak fitting results, including area, centroid, amplitude, and sigma,
with uncertainties.
"""
from datetime import datetime
from dataclasses import dataclass, field
from typing import List
import numpy as np
from uncertainties import UFloat, ufloat # type: ignore


def linear_func(x, m, b):
    """Linear function: y = mx + b"""
    return m * x + b

def gaussian_func(x, amp, center, sigma):
    """Gaussian function"""
    return (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def linear_gaussian_model(x, m, b, amp, center, sigma):
    """Combined linear and Gaussian model."""
    return linear_func(x, m, b) + gaussian_func(x, amp, center, sigma)

@dataclass
class PeakFitResult:
    """
    Class to hold the results of a peak fitting operation according to a linear + gaussian model.

    f(E) = (m * E + b) + (amp / (sigma * sqrt(2 * pi))) * exp(-((E - mu) ** 2) / (2 * sigma ** 2))
    
    Attributes
    ----------
    area : UFloat
        Area of the fitted peak [counts] (amplitude / energy bin width)
    mu : UFloat
        Centroid of the peak [energy]
    amp : UFloat
        Amplitude of the peak [counts * energy]
    sigma : UFloat
        Standard deviation of the Gaussian fit [energy]
    m : UFloat
        Slope of the linear background [counts / energy]
    b : UFloat
        Intercept of the linear background [counts]
    energy_range : tuple[float, float]
        Tuple of (min_energy, max_energy) defining the fitting range for the peak (useful for plotting)
    energy_bins : np.ndarray | None
        Optional array of energy bin edges corresponding to the spectrum data (used for plotting)
    counts : np.ndarray | None
        Optional array of count values corresponding to the spectrum data (used for plotting)
    start_time : datetime
        Start time of the measurement, metadata from the spectrum file
    real_time : float
        Real time of the measurement [seconds], metadata from the spectrum file
    live_time : float
        Live time of the measurement [seconds], metadata from the spectrum file
    figure : go.Figure | None
        Optional plotly figure of the peak fit, if generated.
    """

    area: UFloat
    mu: UFloat
    amp: UFloat
    sigma: UFloat
    m: UFloat
    b: UFloat
    energy_range: tuple[float, float]
    energy_bins: np.ndarray
    counts: np.ndarray

    file_name: str
    start_time: datetime
    real_time: float
    live_time: float

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary suitable for a pandas DataFrame. Here I drop the energy_bins and counts arrays to avoid serialization issues."""
        return {
            "area": self.area.n,
            "area_err": self.area.s,
            "mu": self.mu.n,
            "mu_err": self.mu.s,
            "amp": self.amp.n,
            "amp_err": self.amp.s,
            "sigma": self.sigma.n,
            "sigma_err": self.sigma.s,
            "m": self.m.n,
            "m_err": self.m.s,
            "b": self.b.n,
            "b_err": self.b.s,
            "energy_range_min": self.energy_range[0],
            "energy_range_max": self.energy_range[1],
            "file_name": self.file_name,
            "start_time": self.start_time.isoformat(),
            "real_time": self.real_time,
            "live_time": self.live_time,
        }

@dataclass
class SpectrometryData:
    """Stores spectrometry sample metadata, calibration data, and channel measurements.

    Attributes:
        sample_name (str): The common or given name of the sample.
        sample_id (str): Unique identifier for tracking the sample.
        sample_type (str): Category or matrix type of the sample (e.g.,
            "Environmental", "Soil", "Liquid").
        user_name (str): Name of the operator or analyst who ran the sample.
        sample_description (str): Detailed notes or description regarding the
            sample context.
        start_time (datetime): Timestamp indicating when the measurement started.
        real_time (float): Total elapsed duration of the measurement run in
            seconds.
        live_time (float): Effective acquisition duration adjusting for system
            dead time, in seconds.
        total_counts (int): Total number of detected events across all spectrum
            channels.
        left_marker (int): Lower channel index bound defining the Region of
            Interest (ROI).
        right_marker (int): Upper channel index bound defining the Region of
            Interest (ROI).
        counts_in_markers (int): Sum of counts recorded within the ROI
            `[left_marker, right_marker]`.
        energy_coefficients (List[float]): Polynomial calibration constants
            [A_0, A_1, ..., A_n] mapping channel n to energy E via
            E = sum A_i * n^i. Defaults to an empty list.
        shape_coefficients (List[float]): Parameters defining peak resolution and
            tailing functions (FWHM and Low Tail parameters). Defaults to an
            empty list.
        energy_unit (str): Unit of energy measurement (e.g., "keV", "MeV").
            Defaults to "keV".
        channels (List[int]): List of channel index numbers (n). Defaults to an
            empty list.
        energy (List[float]): Calibrated energy values corresponding to each
            channel. Defaults to an empty list.
        channels_data (List[int]): Recorded raw counts for each channel.
            Defaults to an empty list.
    """
    sample_name: str
    sample_id: str
    sample_type: str
    user_name: str
    sample_description: str

    start_time: datetime
    real_time: float
    live_time: float

    total_counts: int

    left_marker: int
    right_marker: int
    counts_in_markers: int

    energy_coefficients: List[float] = field(default_factory=list)
    shape_coefficients: List[float] = field(default_factory=list)
    energy_unit: str = "keV"

    channels: List[int] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    channels_data: List[int] = field(default_factory=list)

    original_file_name: str = ""

    def write_to_file(self, filename: str) -> None:
        """Write sample dataclass data to a text file in InterSpec format.

        Args:
            filename (str): Path or name of the output text file.
        """
        # InterSpec format relies on ISO 8601 for the start time
        start_time_str = self.start_time.isoformat()

        # energy coefficients into a space-separated string
        coeffs_str = " ".join(f"{c:g}" for c in self.energy_coefficients)

        sample_num = self.sample_id if self.sample_id else "1"
        detector = self.user_name if self.user_name else "Unknown"
        remark = self.sample_description if self.sample_description else "None"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'Original File Name: {filename}\n')
            f.write(f'TotalGammaLiveTime: {self.live_time} seconds\n')
            f.write(f'TotalRealTime: {self.real_time} seconds\n')
            f.write(f'TotalGammaCounts: {self.total_counts:g} seconds\n') # somehow interspec adds this seconds after numbers even though it is not a time
            f.write('TotalNeutron: 0 seconds\n') # somehow interspec adds this seconds after numbers even though it is not a time
            f.write(f'Remark: {remark}\n')
            f.write('\n\n')
            f.write(f'StartTime: {start_time_str}\n')
            f.write(f'LiveTime: {self.live_time} seconds\n')
            f.write(f'RealTime: {self.real_time} seconds\n')
            f.write(f'SampleNumber: {sample_num}\n')
            
            f.write(f'DetectorName: {detector}\n') 
            f.write(f'Title: {self.sample_name}\n')
            f.write('EquationType: Polynomial\n')
            f.write(f'Coefficients: {coeffs_str}\n')
            f.write('Channel Energy Counts\n')

            # Write the spectrum data
            for channel, energy, count in zip(self.channels, self.energy, self.channels_data):
                f.write(f'{channel} {energy:g} {count}\n')


    def write_to_file_cnfconv(self, filename: str) -> None:
        """Write sample dataclass data to a text file using the cnfconv format.

        Args:
            filename (str): Path or name of the output text file.
        """
        start_time_str = self.start_time.strftime('%Y-%m-%d %H:%M:%S')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('#\n')
            f.write(f'# Sample name: {self.sample_name}\n')
            f.write('\n')

            f.write(f'# Sample id: {self.sample_id}\n')
            f.write(f'# Sample type: {self.sample_type}\n')
            f.write(f'# User name: {self.user_name}\n')
            f.write(f'# Sample description: {self.sample_description}\n')
            f.write('#\n')

            f.write(f'# Start time: {start_time_str}\n')
            f.write(f'# Real time (s): {self.real_time:.3f}\n')
            f.write(f'# Live time (s): {self.live_time:.3f}\n')
            f.write('#\n')

            f.write(f'# Total counts: {self.total_counts}\n')
            f.write('#\n')

            f.write(f'# Left marker: {self.left_marker}\n')
            f.write(f'# Right marker: {self.right_marker}\n')
            f.write(f'# Counts: {self.counts_in_markers}\n')
            f.write('#\n')

            f.write('# Energy calibration coefficients (E = sum(Ai * n**i))\n')
            for j, co in enumerate(self.energy_coefficients):
                f.write(f'#    A{j} = {co:.6e}\n')
            f.write(f'# Energy unit: {self.energy_unit}\n')
            f.write('#\n')

            f.write('# Channel data\n')
            f.write(f'#     n     energy({self.energy_unit})     counts     rate(1/s)\n')
            f.write('#' + 50 * '-' + '\n')

            for i, j, k in zip(self.channels, self.energy, self.channels_data):
                rate = k / self.live_time if self.live_time > 0 else 0.0
                f.write(f'{i:4d}\t{j:.3e}\t{k}\t{rate:.3e}\n')


