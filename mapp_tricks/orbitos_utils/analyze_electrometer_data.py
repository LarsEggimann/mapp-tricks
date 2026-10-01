import os
from attr import dataclass
from typing_extensions import Literal
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.graph_objects as go  # type: ignore
from datetime import datetime
from zoneinfo import ZoneInfo
from uncertainties import ufloat, UFloat # type: ignore
from uncertainties import unumpy as unp # type: ignore
from scipy.signal import medfilt # type: ignore

from ..plotting.plotly_styler import apply_my_plotly_style

@dataclass
class BeamData:
    start_of_beam: datetime
    end_of_beam: datetime
    t_irradiation: UFloat
    integrated_charge: UFloat
    average_current: UFloat
    plot: go.Figure

    def __post_init__(self):
        """Runs automatically right after the generated __init__ finishes."""
        pass

    def __repr__(self):
        return (f"BeamData(start_of_beam={self.start_of_beam}, end_of_beam={self.end_of_beam}, "
                f"t_irradiation={self.t_irradiation} seconds, "
                f"integrated_charge={self.integrated_charge}, average_current={self.average_current})")

# taken from www.keysight.com/us/en/assets/7018-04484/data-sheets/5991-4878.pdf (page 16)
ELECTROMETER_ACCURACY = {
    '2pA':   (0.01,   3e-15),   # 1%    + 3   fA
    '20pA':  (0.005,  3e-15),   # 0.5%  + 3   fA
    '200pA': (0.005,  5e-15),   # 0.5%  + 5   fA
    '2nA':   (0.002,  300e-15), # 0.2%  + 300 fA
    '20nA':  (0.002,  500e-15), # 0.2%  + 500 fA
    '200nA': (0.002,  5e-12),   # 0.2%  + 5   pA
    '2uA':   (0.001,  50e-12),  # 0.1%  + 50  pA
    '20uA':  (0.0005, 500e-12), # 0.05% + 500 pA
    '200uA': (0.0005, 5e-9),    # 0.05% + 5   nA
    '2mA':   (0.0005, 50e-9),   # 0.05% + 50  nA
    '20mA':  (0.0005, 500e-9),  # 0.05% + 500 nA
}

class ElectrometerDataAnalyzer:
    def __init__(self,
                 path_to_csv: str,
                 beam_threshold: float = 400e-12,
                 electrometer_range_mode: Literal['2pA', '20pA', '200pA', '2nA', '20nA', '200nA', '2uA', '20uA', '200uA', '2mA', '20mA'] = '2uA',
                 timezone: ZoneInfo = ZoneInfo("Europe/Zurich")
                 ):
        
        self.path_to_csv = path_to_csv
        self.beam_threshold = beam_threshold
        self.electrometer_range_mode = electrometer_range_mode
        self.plot = None
        self.beam_data: BeamData | None = None
        if not os.path.exists(self.path_to_csv):
            raise FileNotFoundError(f"File not found: {self.path_to_csv}")
        self.df = pd.read_csv(self.path_to_csv)
        if self.df is None or self.df.empty:
            raise ValueError(f"Failed to read data from {self.path_to_csv} or file is empty.")
        
        # convert timestamps to datetime, we create objects in the specified timezone, be careful when comparing with other datetimes
        self.df['datetime'] = [datetime.fromtimestamp(ts, tz=timezone) for ts in self.df['timestamp']]

        # debug variables
        self._beam_start_idx = None
        self._beam_end_idx = None

    def analyze_beam_data(self, save_plot=True) -> BeamData:

        # find beam start and end times (current above threshold)
        beam_mask = self.df['current'] > self.beam_threshold

        # # set all values to 0 for which there is no beam
        # self.df.loc[~beam_mask, 'current'] = 0

        beam_indices = self.df.index[beam_mask]
        if len(beam_indices) > 0:
            self._beam_start_idx = beam_indices[0] - 1
            self._beam_end_idx = beam_indices[-1]  + 1
            self.beam_start_time = self.df.loc[self._beam_start_idx, 'datetime']
            self.beam_end_time = self.df.loc[self._beam_end_idx, 'datetime']
        else:
            print(f"No beam detected (no current above {self.beam_threshold:.1e} A)")
            self.beam_start_time = None
            self.beam_end_time = None

        t_irradiation = (self.beam_end_time - self.beam_start_time).total_seconds()
        delta_t = self.df['datetime'].diff().dt.total_seconds().mean()
        t_irradiation = ufloat(t_irradiation, delta_t)  # assume uncertainty in irradiation time is the average delta t between measurements

        # calc integrated charge using trapezoidal integration
        if self.beam_start_time is not None and self.beam_end_time is not None:
            beam_on_mask = (self.df['datetime'] >= self.beam_start_time) & (self.df['datetime'] <= self.beam_end_time)
            beam_currents = self.df['current'][beam_on_mask]
            beam_timestamps = self.df['timestamp'][beam_on_mask]

            # noise calculation during beam-on period using a median filter to remove spikes and then calculating the standard deviation of the residuals
            filtered_currents = medfilt(beam_currents, kernel_size=15)
            # remove values that are more than 3 sigma away from the median filtered values to avoid spikes affecting the noise calculation
            std_raw = np.std(beam_currents - filtered_currents)
            outlier_mask = np.abs(beam_currents - filtered_currents) <= 3 * std_raw
            filtered_currents_outliers_removed = filtered_currents[outlier_mask]
            sigma_noise_beam = np.std(beam_currents[outlier_mask] - filtered_currents_outliers_removed)

            # accuracy of the electrometer, based on the range mode (from datasheet)
            relative_accuracy, offset_accuracy = ELECTROMETER_ACCURACY[self.electrometer_range_mode]
            accuracies = (relative_accuracy * np.abs(beam_currents) + offset_accuracy)

            # the total uncertainty in the current measurements is the quadrature sum of the noise and the accuracy
            sigma_currents = np.sqrt(sigma_noise_beam**2 + accuracies**2)

            # print mean sigmas and accuracy for debugging
            # print(f"Mean noise uncertainty (medfilt):  {sigma_noise_beam:.2e} A,\n"
            #       f"Mean accuracy uncertainty:         {np.mean(accuracies):.2e} A,\n"
            #       f"Mean total uncertainty:            {np.mean(sigma_currents):.2e} A"
            #     )

            timestamps = beam_timestamps.to_numpy()
            currents = beam_currents.to_numpy()

            total_charge = np.trapezoid(currents, timestamps)
            dt = np.diff(timestamps)

            # # weights are the amount of time for which each current data point contributes to the trapezoidal integral
            weights = np.empty(len(currents))
            weights[0]    = dt[0]  / 2 # first and last points only contribute half the time interval according to the trapezoidal rule
            weights[-1]   = dt[-1] / 2
            weights[1:-1] = (dt[:-1] + dt[1:]) / 2 # all other points contribute the full time interval between the two neighboring points

            # then the uncertainty in the integrated charge is the quadrature sum of the uncertainties in each current measurement, weighted by the time interval they contribute to the integral
            # scale by N-1 to make sure the uncertainty does not decrease with more measurements, since the uncertainty in the current is not statistical but systematic (instrumental noise and accuracy)
            charge_uncertainty =  np.sqrt((len(weights)-1) * np.sum((weights * sigma_currents)**2))

            integrated_charge = ufloat(total_charge, charge_uncertainty)
            
            # average current (uncertainty propagates automatically via I = Q / t)
            if t_irradiation.n > 0:
                average_current = integrated_charge / t_irradiation
            else:
                average_current = np.nan

        else:
            integrated_charge = np.nan
            average_current = np.nan

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df['datetime'],
            y=self.df['current'],
            mode='lines',
            name='Current',
            line=dict(color='blue', width=1)
        ))

        # highlight beam-on region if it exists
        if self.beam_start_time is not None:
            beam_data = self.df[beam_mask]
            fig.add_trace(go.Scatter(
                x=beam_data['datetime'],
                y=beam_data['current'],
                mode='lines',
                name=f'Beam On >{self.beam_threshold:.1e} A',
                line=dict(color='red', width=2)
            ))

            # plot the median filtered current in black
            fig.add_trace(go.Scatter(
                x=self.df['datetime'][beam_on_mask][outlier_mask],
                y=filtered_currents_outliers_removed,
                mode='lines',
                name='Median Filtered Current',
                line=dict(color='black', width=2, dash='dot')
            ))

        # horizontal line for beam threshold
        fig.add_hline(
            y=self.beam_threshold,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Beam Threshold ({self.beam_threshold:.1e} A)"
        )

        # layout
        fig.update_layout(
            title='Current vs Time',
            xaxis_title='Time [datetime]',
            yaxis_title='Current [A]',
            # yaxis_type='log',  # Log scale for current
            showlegend=True,
            width=1000,
            height=600
        )

        # grey grid lines
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')

        # add relevant metadata to the plot
        bst = self.beam_start_time.strftime("%Y-%m-%d %H:%M:%S") if self.beam_start_time is not None else "N/A"
        bet = self.beam_end_time.strftime("%Y-%m-%d %H:%M:%S") if self.beam_end_time is not None else "N/A"
        fig.add_annotation(
            text=f"Integrated Charge: {integrated_charge:.uS} C<br>"
                 f"Beam Start:        {bst}<br>"
                 f"Beam End:          {bet}",
            xref="paper", yref="paper",
            x=0.05, y=0.90,
            showarrow=False,
            font=dict(size=12, color='black'),
            align='left'
        )

        fig = apply_my_plotly_style(fig)

        self.plot = fig
        self.beam_data = BeamData(
            start_of_beam=self.beam_start_time,
            end_of_beam=self.beam_end_time,
            t_irradiation=t_irradiation,
            integrated_charge=integrated_charge,
            average_current=average_current,
            plot=fig
        )

        # Save the plot as HTML
        results_path = os.path.join(os.path.dirname(self.path_to_csv), 'mapp_tricks_results')
        os.makedirs(results_path, exist_ok=True)
        file_name = os.path.basename(self.path_to_csv)
        file_name = os.path.splitext(file_name)[0]

        if save_plot:
            fig.write_html(os.path.join(results_path, f'{file_name}_plot.html'))

        return self.beam_data
    
    # def get_correction_factor():

    def get_integrated_correction_factor(self, half_life, start_of_beam: datetime | None = None, end_of_beam: datetime | None = None, show_plot = False):
        """
        If half life of peak of interest is comparable to irradiation time, the fluctuations in the current can become relevant.
        This function returns the integrated correction factor to properly account for production and decay during irradiation, based on the beam data.

        - half_life: The half-life of the isotope of interest (in seconds).
        - start_of_beam: The start time of the beam (optional).
        - end_of_beam: The end time of the beam (optional).
        - show_plot: Whether to show a plot of the correction factor over time (optional).

        """
        if self.beam_data is None:
            raise ValueError("Beam data not analyzed yet. Call analyze_beam_data() first.")

        # since the time axis in the beam file is in timestamp we can integrate directly
        half_life = unp.nominal_values(half_life)
        lambda_ = np.log(2) / half_life

        if start_of_beam is None:
            start_of_beam = self.beam_data.start_of_beam
        if end_of_beam is None:
            end_of_beam = self.beam_data.end_of_beam
    
        data = self.df[(self.df['datetime'] >= start_of_beam) & (self.df['datetime'] <= end_of_beam)]

        time = data['timestamp'].to_numpy(dtype=float, copy=True)
        current = data['current'].to_numpy(copy=True)

        # start time at 0
        if time.size > 0:
            time = time - time[0]

        # compute the integrated correction factor
        def compute(t, c):
            if len(t) < 2:
                return 1, 0, 0
            production_at_t = np.trapezoid(c, t)
            decay_at_t = np.trapezoid(np.exp(lambda_ * t) * c, t) * np.exp(-lambda_ * t[-1])
            return production_at_t / decay_at_t if decay_at_t != 0 else 1, production_at_t, decay_at_t

        # if true do it for each time step and plot it
        if show_plot:

            production_at_t = []
            decay_at_t = []
            res = []

            for i in range(len(time)):
                r, p, d = compute(time[:i+1], current[:i+1])
                res.append(r)
                production_at_t.append(p)
                decay_at_t.append(d)

            # plot current and the three things together with plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=production_at_t, yaxis='y1', mode='lines', name='Production', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=time, y=decay_at_t, yaxis='y1', mode='lines', name='Decay', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=time, y=res, yaxis='y1', mode='lines', name='Correction Factor', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=time, y=current, yaxis='y2', mode='lines', name='Current', line=dict(color='black')))
            fig.update_layout(
                title='Current and Correction Factors Over Time',
                xaxis_title='Time (s)',
                yaxis=dict(
                    title='Production/Decay/Correction Factor',
                ),
                yaxis2=dict(
                    title='Current (A)',
                    overlaying='y',
                    side='right'
                ),
                legend_title='Legend',
            )
            fig = apply_my_plotly_style(fig)
            fig.show()

        res,_,_ = compute(time, current)
        return res