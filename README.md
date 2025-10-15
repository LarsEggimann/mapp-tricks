# mapp-tricks package
Reusable code developed during my PhD in the Medical Applications of Particle Physics (MAPP) group at the University of Bern. It has several modules, some of which are explained below. The other modules I figured are too specific and probably not useful for others, but feel free to explore them.


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

## Usage of X-ray Spectrometer Efficiency Calibration
This module was developed by Samuel Dominique Juillerat.



## Usage Orbitos Utils
ORBITOS is a custom software developed to control and acquire data from various beam shaping and monitoring devices.

### ElectrometerDataAnalyzer
The `orbitos_utils` module provides a convenient tool to plot and analyze data from the electrometer in its most simple use case. For more advanced analysis one can still use the util to easily extract the raw data and perform custom analysis.

```python
from mapp_tricks.orbitos_utils import ElectrometerDataAnalyzer

eda = ElectrometerDataAnalyzer(path_to_csv="data/electrometer_data.csv")
electrometer_data = eda.analyze_beam_data()
```

The `electrometer_data` is a python object that contains the following attributes:
```python
class BeamData:
    start_of_beam: datetime
    end_of_beam: datetime
    t_irradiation: float
    integrated_charge: ufloat
    plot: go.Figure
```

i.e. to get the integrated charge (which is a ufloat):
```python
integrated_charge = electrometer_data.integrated_charge
print(f"Integrated charge: {integrated_charge:.3f} C")
```

To get only the raw data:
```python
eda = ElectrometerDataAnalyzer(path_to_csv="data/electrometer_data.csv")

raw_data = eda.df
```
The raw data is a pandas dataframe which looks like this
```
      timestamp       current                   datetime
0  1.756286e+09  1.500000e-12 2025-08-27 11:05:49.924095
1  1.756286e+09 -1.000000e-12 2025-08-27 11:05:50.227460
2  1.756286e+09 -5.000000e-13 2025-08-27 11:05:50.530236
3  1.756286e+09 -8.000000e-13 2025-08-27 11:05:50.834038
4  1.756286e+09 -5.000000e-13 2025-08-27 11:05:51.140312
...
```

#### Integrated Correction Factor
It also calculates the integrated correction factor accounting for the decay of a isotope produced during irradiation. It accounts for irregular beam-shape and accurately integrates the needed correction factor to effectively determine i.e. the cross section of a reaction. It is implemented according to the following equation:
$$
f(t) = \frac{\int_0^t P(t')\,dt'}{e^{-\lambda t}\int_0^t e^{\lambda t'} P(t')\,dt'}
$$
, where $P(t)$ is the production rate (which is proportional to the beam current for all times $t'$) and $\lambda$ is the decay constant of the isotope. For constant $P(t) = P$ this yields:
$$
f(t) = \frac{\lambda t}{1 - e^{-\lambda t}}
$$
, which is the well known correction factor we use for i.e. the cross section calculation.

To get the integrated correction factor one can use the following example for a Tc101 peak:

```python
eda = ElectrometerDataAnalyzer(path_to_csv="data/messy_beam_electrometer_data.csv")

electrometer_data = eda.analyze_beam_data()
integrated_correction_factor = eda.get_integrated_correction_factor(
    half_life=14.12 * 60
)

print(f"Integrated correction factor: {integrated_correction_factor:.3f}")
```

## Usage of Film Analyzer
The `film_analyzer` module provides tools to analyze scanned images of gafchromic films. It can read the image, extract the RGB channels, and convert the pixel values to dose using a calibration curve defined in a bundled JSON file.
### Basic Usage
```python
from mapp_tricks.film_analyzer.film_analyzer import  FilmAnalyzer

fa = FilmAnalyzer(
    folder='./data/film_reader_test_data/',
    dpi=1200,
    calibration_key='EBT3_new_METAS_ImageJwRGB',
    plot_downsample=0.5
)
```
In order to analyze films we need information about the center positiona and ROI shape and size. This is done via a config file that can be generated as a template and then has to be edited manually, using the film analyzer is a iterative process.
To generate a template config file and save it:
```python
config = fa.generate_default_config(
    default_shape='circular',
    default_max_dose=10.0,
)
fa.save_config(config, fa.folder / 'config.json')
```

Now run the analysis:
```python
fa.load_config(fa.folder / 'config.json')
fa.process_all(config)
```
This will create a subfolder `results/` in the folder where the images are located. In this folder you will find dose maps and profiles for each film analyzed and also a summary file `summary.csv` that contains the mean dose and standard deviation and other information in the ROI for each film.

Now use the generated html plots to optimize the config file and re-run the analysis until satisfied. Make sure to not overwrite the config file with the generated one each time you run the analysis -> remove the saving and generating part after the first time.

