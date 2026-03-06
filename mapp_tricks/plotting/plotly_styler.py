import plotly.graph_objects as go  # type: ignore
import plotly.io as pio # type: ignore
from copy import deepcopy

base = pio.templates["seaborn"]
my_plotly_theme = deepcopy(base)

grid_color = '#D0D0D0'
line_color = "#000000"
text_color = "#000000"
bg_color = "#FFFFFF"

my_plotly_theme.layout.update(

    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,

    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(
        family="STIX Two",
        size=18,
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

def _default_plotly_style(fig: go.Figure) -> go.Figure:

    # style errorbars
    for trace in fig.data:
        if hasattr(trace, "error_x") and trace.error_x:
            trace.error_x.thickness = 1
            trace.error_x.width = 4

        if hasattr(trace, "error_y") and trace.error_y:
            trace.error_y.thickness = 1
            trace.error_y.width = 4

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
