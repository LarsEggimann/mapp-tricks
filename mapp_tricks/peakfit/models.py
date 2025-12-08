"""
Hold classes for peak fitting results, including area, centroid, amplitude, and sigma,
with uncertainties.
"""
from datetime import datetime
from typing import Optional
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