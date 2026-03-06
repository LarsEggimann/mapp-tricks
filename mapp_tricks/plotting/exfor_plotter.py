import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uncertainties.unumpy as unp

def plot_exfor_cs_data(path_exfor_csv: str) -> tuple[go.Figure, pd.DataFrame]:
    exfor_data = pd.read_csv(path_exfor_csv)

    # plot cross section with plotly express, y=y in barn and x=x2(eV) in eV
    exfor_data['x2(MeV)'] = exfor_data['x2(eV)'] * 1e-6  # Convert eV to MeV
    # group by author1 for plotting, add year1 as info
    exfor_data['author_year'] = exfor_data['author1'] + ' (' + exfor_data['year1'].astype(str) + ')'

    # convert x errors to MeV
    exfor_data['dx2(MeV)'] = exfor_data['dx2(eV)'] * 1e-6  # Convert eV to MeV

    # add errors, for x: dx2(MeV), for y: dy, and make error bars less prominent
    fig = px.scatter(
        exfor_data,
        x='x2(MeV)',
        y='y',
        labels={'x2(MeV)': 'Energy [MeV]', 'y': 'Cross Section [barn]'},
        error_x='dx2(MeV)',
        error_y='dy',
        color='author_year',
        hover_name='author_year'
    )

    fig.update_layout(
        xaxis_title='Energy [MeV]',
        yaxis_title='Cross Section [barn]',
        legend_title='Authors',
    )

    return fig, exfor_data