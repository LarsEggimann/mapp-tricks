import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from scipy.optimize import curve_fit # type: ignore
from uncertainties import ufloat, unumpy as unp, Variable # type: ignore
from uncertainties.umath import exp # type: ignore # pylint: disable=no-name-in-module
import matplotlib.pyplot as plt # type: ignore
from matplotlib.figure import Figure


def load_calibration_data(level: int, with_aluminum_foil: bool):
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, "calibration-data/HPGe_efficiency_data.csv")
    data = pd.read_csv(path)

    # filter by level 0-10 and aluminum foil flag
    data = data[(data['level'] == level) & (data['with_aluminum_foil'] == with_aluminum_foil)]

    # create new df with same columns but ufloat where we have errors
    data['reference_peak_activity'] = data.apply(
        lambda row: ufloat(row['reference_peak_activity'], row['reference_peak_activity_error']), axis=1)
    data['net_peak_areas'] = data.apply(
        lambda row: ufloat(row['net_peak_areas'], row['net_peak_areas_error']), axis=1)
    data = data.drop(columns=['reference_peak_activity_error', 'net_peak_areas_error'])
    # convert datetime columns to datetime objects
    data['reference_date'] = pd.to_datetime(data['reference_date'])
    data['time_measurement_start'] = pd.to_datetime(data['time_measurement_start'])
    return data

# efficiency model: sum of (log(E)/E)**n terms up to n=5
def efficiency_model(E, a0, a1, a2, a3, a4, a5):
    h = np.log(E)
    return (1/E) * (
            a0 * 1 +
            a1 * h +
            a2 * h**2 +
            a3 * h**3 +
            a4 * h**4 +
            a5 * h**5)

# error vector for the efficiency model, used to plot the error of the fit
def get_error_vector(x, cov_beta):
    """
    Build the error vector for the efficiency model.
    x: energy values [keV]
    cov_beta: covariance matrix of the fit parameters
    """
    sigmas = []
    for x_i in x:
        A = np.zeros((6, 1))
        A[0, 0] = 1 / x_i                 # derivative with respect to a_0
        A[1, 0] = np.log(x_i) / x_i       # derivative with respect to a_1
        A[2, 0] = (np.log(x_i)**2) / x_i  # derivative with respect to a_2
        A[3, 0] = (np.log(x_i)**3) / x_i  # derivative with respect to a_3
        A[4, 0] = (np.log(x_i)**4) / x_i  # derivative with respect to a_4
        A[5, 0] = (np.log(x_i)**5) / x_i  # derivative with respect to a_5
        sigma = np.sqrt(np.diag(A.T @ cov_beta @ A))
        sigmas.append(sigma[0])  # take the first element since J is 6x1
    return np.array(sigmas)

class FitData:
    def __init__(self, fit_func, fit_params, fit_errors, fit_covariance):
        self.fit_func = fit_func
        self.fit_params = fit_params
        self.fit_errors = fit_errors
        self.fit_covariance = fit_covariance

    def __repr__(self):
        return f"FitData(fit_func={self.fit_func}, fit_params={self.fit_params}, fit_errors={self.fit_errors}, fit_covariance={self.fit_covariance})"

class HPGeCalibration:
    def __init__(self, level=0, with_aluminum_foil=False):
        # dataframe with the following columns:
        # level,energy,element,reference_peak_activity,reference_date,half_life,time_measurement_start,measurement_time_active,with_aluminum_foil,net_peak_areas
        self.data = load_calibration_data(level, with_aluminum_foil)
        self.level = level
        self.with_aluminum_foil = with_aluminum_foil

        # calculate the efficiency
        self._calculate_efficiency()
        self.fit_data: FitData = self._fit_curve_fit()

    def _activity_from_peak_area(self, peak_area, d_r, d_m, t_1_2, t_m_a):
        """
        Calculate the activity.
        peak_area: net peak area [#counts]
        d_r: reference date (datetime)
        d_m: measurement start date (datetime)
        t_1_2: half life [s]
        t_m_a: measurement time in which the detector was active [s]
        """

        # make sure all the inputs are numpy arrays for vectorized operations
        peak_area = np.asarray(peak_area)
        t_1_2 = np.asarray(t_1_2)
        t_m_a = np.asarray(t_m_a)
        d_r = np.asarray(d_r)
        d_m = np.asarray(d_m)
        t_r = np.asarray(d_m - d_r, dtype='timedelta64[s]').astype(float) # time difference in seconds between measurement start and reference date

        lambda_ = np.log(2) / t_1_2  # decay constant [1/s]
        return peak_area * lambda_ * np.exp(lambda_ * t_r) / (1 - np.exp(-lambda_ * t_m_a))

    def _calculate_efficiency(self):
        """
        Calculate the efficiency.
        """
        A_m = self._activity_from_peak_area(
            self.data['net_peak_areas'],
            self.data['reference_date'],
            self.data['time_measurement_start'],
            self.data['half_life'],
            self.data['measurement_time_active']
        )
        A_0 = self.data['reference_peak_activity']
        efficiency = A_m / A_0
        # add efficiency and its error to the dataframe
        self.data['efficiency'] = efficiency

    def _fit_curve_fit(self):

        Ey = unp.nominal_values(self.data['energy'])
        eff = unp.nominal_values(self.data['efficiency'])
        err_eff = unp.std_devs(self.data['efficiency'])

        # fit the efficiency data using curve_fit
        popt, pcov = curve_fit(efficiency_model, Ey, eff, sigma=err_eff, absolute_sigma=True, p0=[1, 1, 1, 1, 1, 1])

        return FitData(
            fit_func=efficiency_model,
            fit_params=popt,
            fit_errors=np.diag(pcov),
            fit_covariance=pcov
        )

    def plot_fit(self, name='') -> Figure:
        x = unp.nominal_values(self.data['energy'])
        y = unp.nominal_values(self.data['efficiency'])
        y_err = unp.std_devs(self.data['efficiency'])

        beta = self.fit_data.fit_params
        cov_beta = self.fit_data.fit_covariance 

        x_fit = np.linspace(np.min(x), np.max(x), 500)
        y_fit = self.fit_data.fit_func(x_fit, *beta)
        sigmas_fit = get_error_vector(x_fit, cov_beta)

        figure = plt.figure(figsize=(8, 6))
        plt.errorbar(x, y, yerr=y_err, fmt='o', label='Calibration Data', capsize=3)
        plt.plot(x_fit, y_fit, 'r-', label='Fit')

        # plot the fit uncertainty
        plt.fill_between(x_fit, y_fit - sigmas_fit, y_fit + sigmas_fit, color='red', alpha=0.3, label='Fit uncertainty')

        plt.xlabel('Energy [keV]')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.title(f'HPGe Detector Efficiency Fit, level {self.level}, with aluminum foil: {self.with_aluminum_foil}')
        plt.grid(True)
        plt.savefig(f'HPGe_calibration_fit_{name}.pdf', bbox_inches='tight')
        return figure

    def evaluate_efficiency_at_energy(self, energy) -> ufloat:
        """
        Evaluate the efficiency at a given energy.
        energy: energy in keV
        """
        # if the energy is a ufloat, extract the nominal value
        if isinstance(energy, Variable):
            energy = energy.n

        beta = self.fit_data.fit_params
        efficiency = self.fit_data.fit_func(energy, *beta)
        error_vector = get_error_vector(np.array([energy]), self.fit_data.fit_covariance)
        return ufloat(efficiency, error_vector[0])
    
    def print_summary(self):
        """
        Print a summary of the calibration data.
        """
        print("Calibration Data:")
        print(self.data[['level', 'with_aluminum_foil', 'energy', 'efficiency']])
        print("Fit Parameters:")
        print(self.fit_data.fit_params)

    def get_activity_for_peak_at_start_of_measurement(self, peak_area: ufloat, peak_energy, life_time, real_time, branching_ratio, half_life) -> ufloat:
        """
        Calculate the activity for a given peak at the start of measurement.
        - peak_area: net peak area [#counts], can be a ufloat
        - peak_energy: energy of the peak [keV], used to calculate the detector efficiency
        - life_time: life time of the measurement [s]
        - real_time: real time of the measurement [s]
        - branching_ratio: branching ratio of the decay that contributes to the peak
        - half_life: half life of the isotope [s]
        """
        efficiency = self.evaluate_efficiency_at_energy(peak_energy)
        decay_constant = np.log(2) / half_life  # decay constant [1/s]

        return (peak_area / (life_time * efficiency * branching_ratio)) * (decay_constant * real_time) / (1 - exp(-decay_constant * real_time))

    def get_activity_for_peak_at_end_of_beam(self, peak_area: ufloat, peak_energy, life_time, real_time, cooling_time, branching_ratio, half_life) -> ufloat:
        """
        Calculate the activity for a given peak at the end of beam. all parameters can be ufloat
        - peak_area: net peak area [#counts]
        - peak_energy: energy of the peak [keV], used to calculate the detector efficiency
        - life_time: life time of the measurement [s]
        - real_time: real time of the measurement [s]
        - cooling_time: cooling time, time between end of beam and start of detector measurement [s]
        - branching_ratio: branching ratio of the decay that contributes to the peak
        - half_life: half life of the isotope [s]
        """
        decay_constant = np.log(2) / half_life  # decay constant [1/s]
        activity_at_start_of_spectra_measurement = self.get_activity_for_peak_at_start_of_measurement(peak_area, peak_energy, life_time, real_time, branching_ratio, half_life)
        return activity_at_start_of_spectra_measurement * exp(decay_constant * cooling_time)  # activity at end of beam
