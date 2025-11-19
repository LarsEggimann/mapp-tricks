from uncertainties import ufloat
from mapp_tricks.spectrometer_calibrations import XRayCalibration

calib = XRayCalibration(
    level=1,
)

test = calib.evaluate_efficiency_at(
    ufloat(10, 0.1)
    )

fig = calib.get_plot()
