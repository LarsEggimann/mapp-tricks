# mapp-tricks package
Reusable code developed during my PhD in the Medical Applications of Particle Physics (MAPP) group at the University of Bern.


Use at your own risk!
- Lars Eggimann

## Usage of HPGeCalibration

Importing:
```python
from mapp_tricks.spectrometer_calibrations import HPGeCalibration
```

Basic usage for directly getting the activity at the end of the beam:
```python
calibration = HPGeCalibration(level=1, with_aluminum_foil=True)

A_EoB = calibration.get_activity_for_peak_at_end_of_beam(
    peak_area=...,
    peak_energy=...,
    life_time=...,
    real_time=...,
    cooling_time=...,
    branching_ratio=...,
    half_life=...,
)

print(f"Activity at end of beam: {A_EoB:.3f} Bq")
```

Or just the efficiency:
```python
efficiency = calibration.evaluate_efficiency_at_energy(
    energy=...
)

print(f"Efficiency: {efficiency:.3f}")
```

You can also ask for activity of peak at the start of spectra measurement:
```python
A_SoM = calibration.get_activity_for_peak_at_start_of_measurement(
    peak_area=...,
    peak_energy=...,
    life_time=...,
    real_time=...,
    branching_ratio=...,
    half_life=...,
)

print(f"Activity at start of measurement: {A_SoM:.3f} Bq")
```

You can also show the fit to visually verify the calibration:
```python
calibration.plot_fit()
```
