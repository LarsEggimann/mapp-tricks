import plotly.graph_objects as go  # type: ignore
import plotly.io as pio # type: ignore
from copy import deepcopy

from ..utils import convert_color_hex_to_rgba

base = pio.templates["seaborn"]
my_plotly_theme = deepcopy(base)

grid_color = "#F5F5F5"
line_color = "#000000"
text_color = "#000000"
bg_color = "#FFFFFF"

error_bar_color = convert_color_hex_to_rgba("#3f3f3f90")

my_plotly_theme.layout.update(

    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,

    margin=dict(l=0, r=0, t=50, b=80), 
    width=850,
    height=450,
    font=dict(
        family="Computer Modern",
        size=20,
        color=text_color
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=line_color,

        ticks="inside",
        tickwidth=2,
        ticklen=5,

        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
    ),
    yaxis=dict(
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=line_color,

        ticks="inside",
        tickwidth=2,
        ticklen=5,

        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1,
    ),
)

pio.templates["my_plotly_theme"] = my_plotly_theme
pio.templates.default = "my_plotly_theme"
colors = pio.templates[pio.templates.default].layout.colorway

def _default_plotly_style(fig: go.Figure) -> go.Figure:

    # style errorbars
    for trace in fig.data:
        if hasattr(trace, "error_x") and trace.error_x:
            trace.error_x.thickness = 1
            trace.error_x.width = 4
            trace.error_x.width = 0 # set width to 0 to remove end caps
            trace.error_x.color = error_bar_color

        if hasattr(trace, "error_y") and trace.error_y:
            trace.error_y.thickness = 1
            trace.error_y.width = 4
            trace.error_y.width = 0 # set width to 0 to remove end caps
            trace.error_y.color = error_bar_color

    # increase marker size for scatter plots
    for trace in fig.data:
        if isinstance(trace, go.Scatter):
            trace.marker.size = 8

    return fig

def apply_my_plotly_style(fig: go.Figure) -> go.Figure:
    return _default_plotly_style(fig)

def apply_my_plotly_style_with_transparent_background(fig: go.Figure) -> go.Figure:
    fig = _default_plotly_style(fig)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def float_legend(fig: go.Figure, x: float = 0.95, y: float = 0.98) -> go.Figure:
    fig.update_layout(
        legend=dict(
            x=x,
            xanchor='right',
            y=y,
            yanchor='top',
            orientation='v',

            # bgcolor='rgba(255,255,255,0.5)',
            # bordercolor='rgba(0,0,0,0.5)',
            # borderwidth=1
        )
    )
    return fig
