import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.graph_objects as go  # type: ignore
from datetime import datetime
from uncertainties import ufloat # type: ignore
import uncertainties

class BeamData:
    start_of_beam: datetime
    end_of_beam: datetime
    t_irradiation: float
    integrated_charge: ufloat
    plot: go.Figure

    def __init__(self, start_of_beam: datetime, end_of_beam: datetime, integrated_charge: ufloat, 
                 plot: go.Figure):
        self.start_of_beam: datetime = start_of_beam
        self.end_of_beam: datetime = end_of_beam
        self.t_irradiation = (end_of_beam - start_of_beam).total_seconds()
        self.integrated_charge = integrated_charge
        self.plot = plot

    def __repr__(self):
        return (f"BeamData(start_of_beam={self.start_of_beam}, end_of_beam={self.end_of_beam}, "
                f"t_irradiation={self.t_irradiation} seconds, "
                f"integrated_charge={self.integrated_charge})")


class ElectrometerDataAnalyzer:
    def __init__(self, path_to_csv: str, beam_threshold: float = 400e-12):
        self.path_to_csv = path_to_csv
        self.beam_threshold = beam_threshold
        self.plot = None
        self.beam_data: BeamData | None = None
        if not os.path.exists(self.path_to_csv):
            raise FileNotFoundError(f"File not found: {self.path_to_csv}")
        self.df = pd.read_csv(self.path_to_csv)
        if self.df is None or self.df.empty:
            raise ValueError(f"Failed to read data from {self.path_to_csv} or file is empty.")
        # convert timestamps to datetime
        self.df['datetime'] = [datetime.fromtimestamp(ts) for ts in self.df['timestamp']]

    def analyze_beam_data(self, save_plot=True) -> BeamData:

        # find beam start and end times (current above threshold)
        beam_mask = self.df['current'] > self.beam_threshold

        # set all values to 0 for which there is no beam
        self.df.loc[~beam_mask, 'current'] = 0

        beam_indices = self.df.index[beam_mask]
        if len(beam_indices) > 0:
            beam_start_idx = beam_indices[0]
            beam_end_idx = beam_indices[-1]
            self.beam_start_time = self.df.loc[beam_start_idx, 'datetime']
            self.beam_end_time = self.df.loc[beam_end_idx, 'datetime']
        else:
            print(f"No beam detected (no current above {self.beam_threshold:.1e} A)")
            self.beam_start_time = None
            self.beam_end_time = None

        # calc integrated charge using trapezoidal integration
        if self.beam_start_time is not None and self.beam_end_time is not None:
            beam_mask = (self.df['datetime'] >= self.beam_start_time) & (self.df['datetime'] <= self.beam_end_time)
            total_charge = np.trapezoid(self.df['current'][beam_mask], self.df['timestamp'][beam_mask])
        else:
            total_charge = 0

        self.integrated_charge = ufloat(total_charge, np.std(self.df['current'][beam_mask]))

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
                name='Beam On (>1e-10 A)',
                line=dict(color='red', width=2)
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
            yaxis_type='log',  # Log scale for current
            showlegend=True,
            width=1000,
            height=600
        )

        # apply transparent background
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black')
        )

        # grey grid lines
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')

        # add relevant metadata to the plot
        fig.add_annotation(
            text=f"Beam Start: {self.beam_start_time}\nBeam End: {self.beam_end_time}\n"
                 f"Integrated Charge: {self.integrated_charge:.2e} C",
            xref="paper", yref="paper",
            x=0.05, y=0.90,
            showarrow=False,
            font=dict(size=12, color='black'),
            align='left'
        )

        self.plot = fig
        self.beam_data = BeamData(
            start_of_beam=self.beam_start_time,
            end_of_beam=self.beam_end_time,
            integrated_charge=self.integrated_charge,
            plot=fig
        )

        # Save the plot as HTML
        results_path = os.path.join(os.path.dirname(self.path_to_csv), 'results')
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

        The Math:
        f(t) = \frac{\int_0^t P(t')\,dt'}{e^{-\lambda t}\int_0^t e^{\lambda t'} P(t')\,dt'}
        """
        if self.beam_data is None:
            raise ValueError("Beam data not analyzed yet. Call analyze_beam_data() first.")

        # since the time axis in the beam file is in timestamp we can integrate directly
        if isinstance(half_life, uncertainties.core.Variable) or isinstance(half_life, uncertainties.UFloat):
            half_life = half_life.n
        lambda_ = np.log(2) / half_life

        if start_of_beam is None:
            start_of_beam = self.beam_data.start_of_beam
        if end_of_beam is None:
            end_of_beam = self.beam_data.end_of_beam
    
        data = self.df[(self.df['datetime'] >= start_of_beam) & (self.df['datetime'] <= end_of_beam)]

        time = data['timestamp'].values
        current = data['current'].values

        # start time at 0
        time -= time[0]

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
                template='plotly_white'
            )
            fig.show()

        res,_,_ = compute(time, current)
        return res