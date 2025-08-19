import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import plotly.graph_objects as go  # type: ignore
from datetime import datetime
from uncertainties import ufloat # type: ignore

class BeamData:
    start_of_beam: datetime
    end_of_beam: datetime
    t_irradiation: float
    integrated_charge: ufloat
    plot: go.Figure

    def __init__(self, start_of_beam: datetime, end_of_beam: datetime, integrated_charge: ufloat, 
                 plot: go.Figure):
        self.start_of_beam = start_of_beam
        self.end_of_beam = end_of_beam
        self.t_irradiation = (end_of_beam - start_of_beam).total_seconds()
        self.integrated_charge = integrated_charge
        self.plot = plot

    def __repr__(self):
        return (f"BeamData(start_of_beam={self.start_of_beam}, end_of_beam={self.end_of_beam}, "
                f"t_irradiation={self.t_irradiation} seconds, "
                f"integrated_charge={self.integrated_charge})")

def analyze_electrometer_data(path_to_csv: str):
    """
    Analyze electrometer data from a CSV file to extract beam start and end times,
    calculate integrated charge, and plot current vs time.
    Parameters
    ----------
    path_to_csv : str
        Path to the CSV file containing electrometer data.
    Returns
    -------
    beam_data : BeamData
        An object containing start and end times of the beam, irradiation time, and integrated charge.
    """

    # check the file exists
    if not os.path.exists(path_to_csv):
        raise FileNotFoundError(f"File not found: {path_to_csv}")
    else:
        print(f"Processing file: {path_to_csv}")

    # Read the CSV file
    df = pd.read_csv(path_to_csv)

    # Convert timestamps to datetime
    df['datetime'] = [datetime.fromtimestamp(ts) for ts in df['timestamp']]

    # Find beam start and end times (current above 100e-12 A)
    beam_threshold = 400e-12
    beam_mask = df['current'] > beam_threshold

    # Find beam start and end indices
    beam_indices = df.index[beam_mask]
    if len(beam_indices) > 0:
        beam_start_idx = beam_indices[0]
        beam_end_idx = beam_indices[-1]
        
        beam_start_time = df.loc[beam_start_idx, 'datetime']
        beam_end_time = df.loc[beam_end_idx, 'datetime']
    else:
        print(f"No beam detected (no current above {beam_threshold:.1e} A)")
        beam_start_time = None
        beam_end_time = None

    # Calculate integrated charge using trapezoidal integration
    # Current is in Amperes, time difference gives us Coulombs
    total_charge = np.trapezoid(df['current'], df['timestamp'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['current'],
        mode='lines',
        name='Current',
        line=dict(color='blue', width=1)
    ))

    # highlight beam-on region if it exists
    if beam_start_time is not None:
        beam_data = df[beam_mask]
        fig.add_trace(go.Scatter(
            x=beam_data['datetime'],
            y=beam_data['current'],
            mode='lines',
            name='Beam On (>1e-10 A)',
            line=dict(color='red', width=2)
        ))

    # horizontal line for beam threshold
    fig.add_hline(
        y=beam_threshold,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Beam Threshold ({beam_threshold:.1e} A)"
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

    integrated_charge = ufloat(total_charge, np.std(df['current']))


    # add relevant metadata to the plot
    fig.add_annotation(
        text=f"Beam Start: {beam_start_time}\nBeam End: {beam_end_time}\n"
             f"Integrated Charge: {integrated_charge:.2e} C",
        xref="paper", yref="paper",
        x=0.05, y=0.90,
        showarrow=False,
        font=dict(size=12, color='black'),
        align='left'
    )

    beam_data = BeamData(
        start_of_beam=beam_start_time,
        end_of_beam=beam_end_time,
        integrated_charge=integrated_charge,
        plot=fig
    )

    # Save the plot as HTML
    results_path = os.path.join(os.path.dirname(path_to_csv), 'results')
    os.makedirs(results_path, exist_ok=True)
    file_name = os.path.basename(path_to_csv)
    # remove file extension for the plot name
    file_name = os.path.splitext(file_name)[0]

    fig.write_html(os.path.join(results_path, f'{file_name}_plot.html'))

    return beam_data
