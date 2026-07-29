import ast
from pathlib import Path
from typing_extensions import Literal
import pandas as pd # type: ignore
import numpy as np # type: ignore
import uncertainties # type: ignore
from uncertainties import ufloat, UFloat, Variable # type: ignore
from uncertainties.umath import exp # type: ignore # pylint: disable=no-name-in-module
import plotly.graph_objects as go # type: ignore

from ...plotting import apply_my_plotly_style

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


# create a structure to hold the calibration paths and their associated names
calibration_data = {
    "calibration_2025": {
        "description": "Partial calibration data for Akimov, only level 1 ,2, 9 and 10 were measured with alu foil, created end of 2025",
        "path": Path(__file__).parent / "calibrations/calibration_2025.csv",
    },
    "calibration_2018": {
        "description": "Full calibration for Akimov, including level 0-10 with and without aluminum foil, created in 2018",
        "path": Path(__file__).parent / "calibrations/calibration_2018.csv",
    }
}

class HPGeCalibration:
    def __init__(self,
                 detector_name: Literal['Akimov', 'Toptunov'],
                 level: int, calibration_name: Literal["calibration_2025", "calibration_2018"],
                 with_aluminum_foil: bool = False
                 ):
        self.detector_name = detector_name
        self.level = level
        self.with_aluminum_foil = with_aluminum_foil
        self.calibration_name = calibration_name
        self.calibration_path = calibration_data[calibration_name]["path"]
        self.df_row = self._load_calibration_data()

        self.parameters = self.df_row["parameters"]
        self.covariance_matrix = self.df_row["covariance_matrix"]

    def _load_calibration_data(self):

        df = pd.read_csv(self.calibration_path)
        df = df[(df['detector_name'] == self.detector_name) & (df['level'] == self.level) & (df['with_aluminum_foil'] == self.with_aluminum_foil)]
        # if the dataframe is empty, raise an error
        if df.empty:
            raise ValueError(f"No calibration data found for detector {self.detector_name}, level {self.level}, with_aluminum_foil={self.with_aluminum_foil} in calibration {self.calibration_name}")

        # there should only be one row for each detector, level and aluminum foil combination
        if len(df) > 1:
            raise ValueError(f"Multiple calibration data found for detector {self.detector_name}, level {self.level}, with_aluminum_foil={self.with_aluminum_foil} in calibration {self.calibration_name}")

        row = df.iloc[0].copy()

        row["parameters"] = np.array(ast.literal_eval(row["parameters"]))
        row["covariance_matrix"] = np.array(ast.literal_eval(row["covariance_matrix"]))

        row["measured_efficiency_energy"] = np.array(ast.literal_eval(row["measured_efficiency_energy"]))
        row["measured_efficiency_eff"] = np.array(ast.literal_eval(row["measured_efficiency_eff"]))
        row["measured_efficiency_eff_error"] = np.array(ast.literal_eval(row["measured_efficiency_eff_error"]))

        return row


    def get_plot(self) -> go.Figure:
        """
        Returns a plotly figure of the efficiency calibration fit with error bands.
        """
        x = np.linspace(
            min(self.df_row.measured_efficiency_energy),
            max(self.df_row.measured_efficiency_energy),
            1000)  # keV, energy range for plotting
        
        y = efficiency_model(x, *self.parameters)
        y_err = get_error_vector(x, self.covariance_matrix)


        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Efficiency fit',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y - y_err, (y + y_err)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=self.df_row.measured_efficiency_energy,
            y=self.df_row.measured_efficiency_eff,
            error_y=dict(
                type='data',
                array=self.df_row.measured_efficiency_eff_error,
                visible=True
            ),
            mode='markers',
            name='Measured efficiency',
            marker=dict(color='black', size=8)
        ))
        fig.update_layout(
            title=f"Efficiency Calibration using {self.calibration_name} for {self.detector_name}, level {self.level}, with aluminum foil: {self.with_aluminum_foil}",
            xaxis_title="Energy [keV]",
            yaxis_title="Efficiency",
        )
        fig = apply_my_plotly_style(fig)
        return fig

    def evaluate_efficiency_at_energy(self, energy) -> UFloat:
        """
        Evaluate the efficiency at a given energy.
        energy: energy in keV
        """
        # if the energy is a ufloat, extract the nominal value
        if isinstance(energy, Variable):
            energy = energy.n
        efficiency = efficiency_model(energy, *self.parameters)
        error_vector = get_error_vector(np.array([energy]), self.covariance_matrix)
        return ufloat(efficiency, error_vector[0])
    
    def print_summary(self):
        """
        Print a summary of the calibration data.
        """
        print("Calibration Data:")
        print(f"Detector: {self.detector_name}")
        print(f"Level: {self.level}")
        print(f"With Aluminum Foil: {self.with_aluminum_foil}")
        print(f"Calibration: {self.calibration_name}, Description: {calibration_data[self.calibration_name]['description']}")
        params_unc = uncertainties.correlated_values(self.parameters, self.covariance_matrix)
        print(f"Parameters: {params_unc}")

    def get_activity_for_peak_at_start_of_measurement(self, peak_area: UFloat, peak_energy, life_time, real_time, branching_ratio, half_life) -> UFloat:
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

    def get_activity_for_peak_at_end_of_beam(self, peak_area: UFloat, peak_energy, life_time, real_time, cooling_time, branching_ratio, half_life) -> UFloat:
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
