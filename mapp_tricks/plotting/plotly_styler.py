import plotly.graph_objects as go  # type: ignore

def _default_plotly_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        xaxis=dict(
            zeroline=False,
        ),
        yaxis=dict(
            zeroline=False,
        ),
        margin=dict(l=20, r=20, t=60, b=70),
        template="plotly_white",
        font=dict(
        family="Times New Roman",
            size=18,
            color="black"
        ),
    )

    # gridcolor = "gray"

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        # mirror=True,
        ticks="inside",
        tickwidth=1,
        ticklen=5,

        # gridcolor=gridcolor,
        # gridwidth=0.1,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        # mirror=True,
        ticks="inside",
        tickwidth=1,
        ticklen=5,

        # gridcolor=gridcolor,
        # gridwidth=0.1,
    )

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
