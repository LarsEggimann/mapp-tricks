import numpy as np
import pandas as pd
import plotly.graph_objects as go
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit
from mapp_tricks.plotting import apply_my_plotly_style

# FIXME: cleanup this file, improve interface and make it more general for other isotopes, not just Tc99m

half_life_99mTc = ufloat(6.0072, 0.0009) * 60 * 60  # type: ignore # 6.0072 hours in seconds

def exponential_decay_t12_fixed(t, A0):
    half_life = half_life_99mTc.n
    tau = half_life / np.log(2)
    return A0 * np.exp(-t / tau)

def exponential_decay(t, A0, half_life):
    tau = half_life / np.log(2)
    return A0 * np.exp(-t / tau)

def double_exponential_decay(t, A0_1, half_life_1, A0_2, half_life_2):
    tau1 = half_life_1 / np.log(2)
    tau2 = half_life_2 / np.log(2)
    return A0_1 * np.exp(-t / tau1) + A0_2 * np.exp(-t / tau2)

def plot_decay_curve_with_fit(spectra_df: pd.DataFrame, target_name: str = "unknown target", show_plot: bool = True):

    x_data = spectra_df['time'].apply(lambda x: x.timestamp()).values
    x_offset = x_data.min()
    x_data = x_data - x_offset  # normalize time to start at 0

    y_data = unp.nominal_values(spectra_df.count_rate)
    y_err = unp.std_devs(spectra_df.count_rate)

    # fit exponential decay to count rate data
    popt, pcov = curve_fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[y_data.max(), 1000],
        sigma=y_err,
        absolute_sigma=True
    )

    a0 = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    fitted_half_life = ufloat(popt[1], np.sqrt(pcov[1, 1]))
    print(f"Fitted parameters: count rate at t0 = {a0:.uS}, fitted half-life = {fitted_half_life/60:.uS} min")
    print(f"data acquisition time: {(x_data.max() - x_data.min())/3600:.2f} hours")
    # print(f"Expected half-life: {half_life_99mTc/3600:.uS} hours")

    # plot count rate over time and residuals in subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("", "")
    )

    data_color = "#7A92FF"
    fit_color = "#000000"
    error_bar_color = "#545454"

    # add decay curve data to first subplot
    fig.add_trace(go.Scatter(
        x=spectra_df.time,
        y=unp.nominal_values(spectra_df.count_rate),
        error_y=dict(
            type='data',
            array=unp.std_devs(spectra_df.count_rate),
            visible=True,
        ),
        mode='markers',
        name='Count Rate',
        marker=dict(color=data_color)
    ), row=1, col=1)

    # add fitted curve to first subplot
    t_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = exponential_decay(t_fit, *popt)

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(t_fit + x_offset, unit='s'),
        y=y_fit,
        name='Fitted Decay Curve',
        line=dict(color=fit_color)
    ), row=1, col=1)

    # plot the residuals in second subplot
    residuals = y_data - exponential_decay(x_data, *popt)
    residuals_err = y_err

    # calculate percentage residuals
    percentage_residuals = residuals / y_data * 100
    percentage_residuals_err = residuals_err / y_data * 100

    fig.add_trace(go.Scatter(
        x=spectra_df.time,
        y=percentage_residuals,
        error_y=dict(
            type='data',
            array=percentage_residuals_err,
            visible=True,
        ),
        mode='markers',
        name='Residuals',
        marker=dict(symbol='diamond', color=data_color),
    ), row=2, col=1)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(title_text='Residuals [%]', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Count Rate (counts/s)', row=1, col=1)

    fig.update_layout(
        title_text=f'Decay Analysis - {target_name}',
        height=800,
        width=1200,
    )

        # make error bars less prominent for all traces
    for trace in fig.data:
        if hasattr(trace, 'error_x') and trace.error_x is not None:
            trace.error_x.thickness = 0.5
            trace.error_x.width = 4
            trace.error_x.color = error_bar_color

        if hasattr(trace, 'error_y') and trace.error_y is not None:
            trace.error_y.thickness = 0.5
            trace.error_y.width = 4
            trace.error_y.color = error_bar_color

    fig = apply_my_plotly_style(fig)
    fig.show()

def plot_decay_curve_with_double_exp_fit(spectra_df: pd.DataFrame, target_name: str = "unknown target", show_plot: bool = True):

    x_data = spectra_df['time'].apply(lambda x: x.timestamp()).values
    x_offset = x_data.min()
    x_data = x_data - x_offset  # normalize time to start at 0

    y_data = unp.nominal_values(spectra_df.count_rate)
    y_err = unp.std_devs(spectra_df.count_rate)

    # fit exponential decay to count rate data
    popt, pcov = curve_fit(
        double_exponential_decay,
        x_data,
        y_data,
        p0=[y_data.max(), 1000, y_data.max(), 1000],
        sigma=y_err,
        absolute_sigma=True
    )

    a0_1 = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    half_life_1 = ufloat(popt[1], np.sqrt(pcov[1, 1]))
    a0_2 = ufloat(popt[2], np.sqrt(pcov[2, 2]))
    half_life_2 = ufloat(popt[3], np.sqrt(pcov[3, 3]))
    print(f"Fitted parameters:")
    print(f"a0_1 at t0 = {a0_1:.uS}, fitted t12_1 = {half_life_1/60:.uS} min")
    print(f"a0_2 at t0 = {a0_2:.uS}, fitted t12_2 = {half_life_2/60:.uS} min")
    print(f"data acquisition time: {(x_data.max() - x_data.min())/3600:.2f} hours")
    # print(f"Expected half-life: {half_life_99mTc/3600:.uS} hours")

    # plot count rate over time and residuals in subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("", "")
    )

    data_color = "#7A92FF"
    fit_color = "#000000"
    error_bar_color = "#545454"

    # add decay curve data to first subplot
    fig.add_trace(go.Scatter(
        x=spectra_df.time,
        y=unp.nominal_values(spectra_df.count_rate),
        error_y=dict(
            type='data',
            array=unp.std_devs(spectra_df.count_rate),
            visible=True,
        ),
        mode='markers',
        name='Count Rate',
        marker=dict(color=data_color)
    ), row=1, col=1)

    # add fitted curve to first subplot
    t_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = double_exponential_decay(t_fit, *popt)

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(t_fit + x_offset, unit='s'),
        y=y_fit,
        name='Fitted Decay Curve',
        line=dict(color=fit_color)
    ), row=1, col=1)

    # plot the residuals in second subplot
    residuals = y_data - double_exponential_decay(x_data, *popt)
    residuals_err = y_err

    # calculate percentage residuals
    percentage_residuals = residuals / y_data * 100
    percentage_residuals_err = residuals_err / y_data * 100

    fig.add_trace(go.Scatter(
        x=spectra_df.time,
        y=percentage_residuals,
        error_y=dict(
            type='data',
            array=percentage_residuals_err,
            visible=True,
        ),
        mode='markers',
        name='Residuals',
        marker=dict(symbol='diamond', color=data_color),
    ), row=2, col=1)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(title_text='Residuals [%]', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='Count Rate (counts/s)', row=1, col=1)

    fig.update_layout(
        title_text=f'Decay Analysis - {target_name}',
        height=800,
        width=1200,
    )

        # make error bars less prominent for all traces
    for trace in fig.data:
        if hasattr(trace, 'error_x') and trace.error_x is not None:
            trace.error_x.thickness = 0.5
            trace.error_x.width = 4
            trace.error_x.color = error_bar_color

        if hasattr(trace, 'error_y') and trace.error_y is not None:
            trace.error_y.thickness = 0.5
            trace.error_y.width = 4
            trace.error_y.color = error_bar_color

    fig = apply_my_plotly_style(fig)
    fig.show()

def plot_bateman_decay_curve(spectra_df: pd.DataFrame):
    print("Fitting Bateman decay curve to data...")
    lambda_1 = np.log(2) / (65.924 * 3600)  # 65.924 h, Mo99  -> Tc99m
    lambda_2 = np.log(2) / (6.0072 * 3600)  # 6.0072 h, Tc99m -> Tc99

    def batemann(t, A_1_0, A_2_0):
        return (lambda_2 * A_1_0)/(lambda_2 - lambda_1) * (np.exp(-lambda_1 * t) - np.exp(-lambda_2 * t)) + A_2_0 * np.exp(-lambda_2 * t)


    x_data = spectra_df['time'].apply(lambda x: x.timestamp()).values
    x_offset = x_data.min()
    x_data = x_data - x_offset  # normalize time to start at 0

    y_data = unp.nominal_values(spectra_df.count_rate)
    y_err = unp.std_devs(spectra_df.count_rate)

    # fit exponential decay to count rate data
    popt, pcov = curve_fit(
        batemann,
        x_data,
        y_data,
        p0=[0, y_data.max()],
        bounds=([0, 0], [np.inf, np.inf]),
        sigma=y_err,
        absolute_sigma=True
    )

    # a0 = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    # fitted_half_life = ufloat(popt[1], np.sqrt(pcov[1, 1]))
    # print(f"Fitted parameters: count rate at t0 = {a0:.2f}, fitted half-life = {fitted_half_life/3600:.4f} hours")


    A_Mo99_0 = ufloat(popt[0], np.sqrt(pcov[0, 0]))
    A_Tc99m_0 = ufloat(popt[1], np.sqrt(pcov[1, 1]))
    print(f"Fitted parameters: ")
    print(f"  Mo99 initial count rate = {A_Mo99_0:.uS} 1/s")
    print(f"  Tc99m initial count rate = {A_Tc99m_0:.uS} 1/s")


    # plot count rate over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spectra_df.time,
        y=unp.nominal_values(spectra_df.count_rate),
        error_y=dict(
            type='data',
            array=unp.std_devs(spectra_df.count_rate),
            visible=True,
        ),
        mode='markers',
        name='Count Rate',
    ))

    # add fitted curve to plot
    t_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = batemann(t_fit, *popt)

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(t_fit + x_offset, unit='s'),
        y=y_fit,
        name='Fitted Decay Curve',
    ))

    fig.update_layout(
        title='Count Rate over Time - Bateman Fit',
        xaxis_title='Time',
        yaxis_title='Count Rate (counts/s)',
        template='plotly_white',
        height=600,
        width=1200,
    )

    fig.update_yaxes(type='log')

    fig = apply_my_plotly_style(fig)
    fig.show()