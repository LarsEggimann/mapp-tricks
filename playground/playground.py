from uncertainties import ufloat
from mapp_tricks.spectrometer_calibrations import XRayCalibration

calib = XRayCalibration(
    level=1,
)

test = calib.evaluate_efficiency_at(
    ufloat(10, 0.1)
    )

fig = calib.get_plot()


from mapp_tricks.spectrometer_calibrations import HPGeCalibration

calib = HPGeCalibration(level=1, with_aluminum_foil=True)

e1 = 670.32
e2 = 962.84

eff1 = calib.evaluate_efficiency_at_energy(e1)
eff2 = calib.evaluate_efficiency_at_energy(e2)

print(f"Efficiency at {e1} keV: {eff1:uS}")
print(f"Efficiency at {e2} keV: {eff2:uS}")

calib.print_summary()
calib.plot_fit()
