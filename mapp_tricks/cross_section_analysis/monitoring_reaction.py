from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from uncertainties import ufloat, UFloat, nominal_value

from ..peakfit import PeakFitter
from ..spectrometer_calibrations import HPGeCalibration
from ..utils import parse_iaea_monitor_reaction

@dataclass
class MonitoringReaction:
    name: str
    data_path: str
    gamma_lines: list[float]  # keV, sorted from low to high
    intensities: list[UFloat] # as fraction
    half_life: UFloat  # seconds
    molar_mass: UFloat | None = None  # g/mol, not used for natural compositions
    isotopic_abundance: UFloat | None = None  # as fraction, not used for natural compositions

monitoring_reactions_dict = {

    'natCu(p,x)63Zn': MonitoringReaction(
        name='natCu(p,x)63Zn',
        data_path='data/natCu_p_x_63Zn.txt',
        gamma_lines=[669.64, 962.07],
        intensities=[ufloat(0.083, 0.005), ufloat(0.0653, 0.0032)],
        half_life=ufloat(38.49, 0.05) * 60,  # seconds
        molar_mass=ufloat(63.546, 0.003),  # g/mol
    ),
    'natCu(p,x)65Zn': MonitoringReaction(
        name='natCu(p,x)65Zn',
        data_path='data/natCu_p_x_65Zn.txt',
        gamma_lines=[1115.539],
        intensities=[ufloat(0.5012, 0.0007)],
        half_life=ufloat(243.93, 0.09) * 60 * 60,  # seconds
        molar_mass=ufloat(63.546, 0.003),  # g/mol
    ),

}

N_A = 6.02214076e23  # Avogadro's number
q = 1.602176634e-19  # elementary charge in Coulombs


def calculate_Q_per_cm2(
        monitoring_reaction: Literal['natCu(p,x)63Zn', 'natCu(p,x)65Zn'],
        particle_energy_MeV: float,
        spectra_file_path: str,
        hpge_calibration: HPGeCalibration,
        monitoring_foil_mass: UFloat,
        cooling_time: float | None = None,
        correction_factor_f_c: float | None = None,
        irradiation_time: float | None = None
):
    """
    Calculate the charge per cm^2 from a given spectrum file and a monitoring reaction.

    Parameters
    ----------
    monitoring_reaction : Literal['natCu(p,x)63Zn', 'natCu(p,x)65Zn']
        The monitoring reaction to use for the calculation.
    particle_energy_MeV : float
        The energy of the incident particle in MeV.
    spectra_file_path : str
        Path to the spectrum file.
    hpge_calibration : HPGeCalibration
        The HPGe calibration object to use for efficiency calculations.
    monitoring_foil_mass : UFloat
        The mass of the monitoring foil in grams. If beamspot does not cover the entire foil, this should be the mass of the irradiated portion (based on collimator area)
    cooling_time : float | None, optional
        Cooling time in seconds, used to correct for decay between end of beam and start of measurement. If None, no decay correction is applied.
    correction_factor_f_c : float | None, optional
        Correction factor for the calculation. If None, no correction is applied.
    irradiation_time : float | None, optional
        Irradiation time in seconds. If None, no decay correction is applied.

    Returns
    -------
    Q_per_cm2 : UFloat
        The calculated charge per cm^2 with uncertainty.
    """
    # Q = (A_EoB * q) / (N_t * sigma * lambda)
    # N_t = (monitoring_foil_mass / molar_mass) * N_A

    # load monitoring reaction data
    reaction = monitoring_reactions_dict.get(monitoring_reaction)
    if not reaction:
        raise ValueError("Invalid monitoring reaction, check the available reactions in monitoring_reactions_dict.")

    # perform peak fitting on the spectra data
    fitter = PeakFitter()
    pres = fitter.process_file_multiple_peaks(
        filepath=spectra_file_path,
        energy_ranges=[(energy - 10, energy + 10) for energy in reaction.gamma_lines],
    )

    A_EoB_list = []
    for i, pres_i in enumerate(pres):
        # get the peak area and energy for the current peak
        peak_area = pres_i[1].area  # net peak area [#counts]
        peak_energy = reaction.gamma_lines[i]      # energy of the peak [keV]
        branching_ratio = reaction.intensities[i]  # branching ratio of the decay that contributes to the peak

        # calculate activity at end of beam
        A_EoB = hpge_calibration.get_activity_for_peak_at_end_of_beam(
            peak_area=peak_area,
            peak_energy=peak_energy,
            life_time=pres_i[1].live_time,
            real_time=pres_i[1].real_time,
            cooling_time=cooling_time if cooling_time is not None else 0,
            branching_ratio=branching_ratio,
            half_life=reaction.half_life
        )
        A_EoB_list.append(A_EoB)

    # calculate the average activity at end of beam, use loop to make sure uncertainties are handled correctly, i did not test np.mean and sum stuff
    sum = 0
    for A_EoB in A_EoB_list:
        sum += A_EoB
    A_EoB_avg = sum / len(A_EoB_list)


    # calculate the number of target atoms
    if reaction.molar_mass is None:
        raise ValueError("Molar mass is not defined for the selected monitoring reaction.")

    N_t = (monitoring_foil_mass / reaction.molar_mass) * N_A


    # load the cross-section data for the selected monitoring reaction
    this_file_path = "/".join(__file__.split("/")[:-1])
    cross_section_data = parse_iaea_monitor_reaction(f"{this_file_path}/{reaction.data_path}")

    # interpolate the cross-section data to find the cross-section at the given energy
    cs_interp_func = interp1d(cross_section_data.energy_MeV, cross_section_data.pade_fit_mb, kind='linear', fill_value='extrapolate')
    # also interpolate the uncertainty in the cross-section
    cs_uncertainty_interp_func = interp1d(cross_section_data.energy_MeV, cross_section_data.pade_fit_mb + cross_section_data.pade_fit_uncertainty_mb, kind='linear', fill_value='extrapolate')

    # TODO: replace with integral over energy range through the monitor foil, also make energy distribution possible (ie. gaussian distribution of energies, or user defined distribution)
    pen = nominal_value(particle_energy_MeV)
    # sigma = ufloat(cs_interp_func(pen), cs_uncertainty_interp_func(pen) - cs_interp_func(pen))  # cross-section in mb
    sigma = cs_interp_func(pen) # TODO: add uncertainty to cross-section,
    # convert mb to b
    sigma *= 1e-3  # cross-section in b
    # convert b to cm^2
    sigma *= 1e-24  # cross-section in cm^2


    _lambda = np.log(2) / reaction.half_life  # decay constant in 1/s

    Q_per_cm2 = (A_EoB_avg * q) / (N_t * sigma * _lambda)

    if correction_factor_f_c is not None and irradiation_time is not None:
        raise ValueError("Both correction_factor_f_c and irradiation_time are provided. Please provide only one or none.")

    if correction_factor_f_c is not None:
        Q_per_cm2 *= correction_factor_f_c
    elif irradiation_time is not None:
        # apply decay correction for irradiation time
        decay_correction_factor = 1 - np.exp(-_lambda * irradiation_time)
        Q_per_cm2 /= decay_correction_factor

    return Q_per_cm2