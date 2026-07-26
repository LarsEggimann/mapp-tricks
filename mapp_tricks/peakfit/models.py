"""
Hold classes for peak fitting results, including area, centroid, amplitude, and sigma,
with uncertainties.
"""
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from uncertainties import ufloat # type: ignore
import matplotlib.pyplot as plt # type: ignore

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

    Methods
    -------
    __repr__():
        String representation of the result object.
    """
    def __init__(self, area: ufloat, centroid: ufloat,
                 start_time: datetime, real_time: float, live_time: float,
                 amplitude: ufloat, sigma: ufloat, figure: Optional[plt.Figure] = None):
        self.area = area
        self.centroid = centroid
        self.start_time = start_time
        self.real_time = real_time
        self.live_time = live_time
        self.amplitude = amplitude
        self.sigma = sigma
        self.figure = figure


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
            $[A_0, A_1, \dots, A_n]$ mapping channel $n$ to energy $E$ via
            $$E = \sum A_i \cdot n^i$$. Defaults to an empty list.
        shape_coefficients (List[float]): Parameters defining peak resolution and
            tailing functions (FWHM and Low Tail parameters). Defaults to an
            empty list.
        energy_unit (str): Unit of energy measurement (e.g., "keV", "MeV").
            Defaults to "keV".
        channels (List[int]): List of channel index numbers ($n$). Defaults to an
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

        with open(filename, 'w') as f:
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

        with open(filename, 'w') as f:
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


