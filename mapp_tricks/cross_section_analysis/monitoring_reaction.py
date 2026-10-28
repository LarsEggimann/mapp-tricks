from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from uncertainties import ufloat, UFloat, nominal_value
from datetime import datetime

from ..peakfit import PeakFitter
from ..spectrometer_calibrations import HPGeCalibration
from ..orbitos_utils import ElectrometerDataAnalyzer
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
        half_life=ufloat(243.93, 0.09) * 24 * 60 * 60,  # seconds
        molar_mass=ufloat(63.546, 0.003),  # g/mol
    ),

}

N_A = 6.02214076e23  # Avogadro's number
q = 1.602176634e-19  # elementary charge in Coulombs


def calculate_Q_per_cm2(
        monitoring_reaction: Literal['natCu(p,x)63Zn', 'natCu(p,x)65Zn'],
        spectra_file_path: str,
        hpge_calibration: HPGeCalibration,
        monitoring_foil_mass: UFloat,
        particle_energy_MeV: float | None = None,
        particle_energy_distributions: list[tuple[np.ndarray, np.ndarray]] | None = None,
        end_of_beam_time: datetime | None = None,
        electrometer_data_analyzer: ElectrometerDataAnalyzer | None = None,
        irradiation_time: float | None = None
):
    """
    Calculate the charge per cm^2 from a given spectrum file and a monitoring reaction.

    Parameters
    ----------
    monitoring_reaction : Literal['natCu(p,x)63Zn', 'natCu(p,x)65Zn']
        The monitoring reaction to use for the calculation.
    spectra_file_path : str
        Path to the spectrum file.
    hpge_calibration : HPGeCalibration
        The HPGe calibration object to use for efficiency calculations.
    monitoring_foil_mass : UFloat
        The mass of the monitoring foil in grams. If beamspot does not cover the entire foil, this should be the mass of the irradiated portion (based on collimator area)
    particle_energy_MeV : float | None, optional
        The energy of the incident particle in MeV. If None, the energy distribution must be provided.
    particle_energy_distributions : list[tuple[np.ndarray, np.ndarray]] | None, optional
        A list of tuples, each containing two numpy arrays: the first array is the energy values in MeV, and the second array is the corresponding probability density values. If None, the particle_energy_MeV must be provided.
    end_of_beam_time : datetime | None, optional
        The time at the end of the beam, used to calculate cooling time. If None, no decay correction is applied.
    electrometer_data_analyzer : ElectrometerDataAnalyzer | None, optional
        The electrometer data analyzer to used to get the integrated correction factor for the beam. If None, no decay correction is applied.
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

    # Check that either particle_energy_MeV or particle_energy_distributions is provided, but not both
    if (particle_energy_MeV is None and particle_energy_distributions is None) or (particle_energy_MeV is not None and particle_energy_distributions is not None):
        raise ValueError("Either particle_energy_MeV or particle_energy_distributions must be provided, but not both. particle_energy_MeV will assume a mono-energetic beam in a thin foil (no energy loss)")

    # perform peak fitting on the spectra data
    fitter = PeakFitter()
    pres = fitter.process_file_multiple_peaks(
        filepath=spectra_file_path,
        energy_ranges=[(energy - 8, energy + 8) for energy in reaction.gamma_lines],
    )

    # cooling time
    cooling_time = None
    if end_of_beam_time is not None:
        spectra_start_time = pres[0][1].start_time  # assuming all peaks are from the same spectrum, so they have the same start time
        eob_time_offset = end_of_beam_time.utcoffset()
        ss_time_offset = spectra_start_time.utcoffset()
        print(f"end_of_beam_time: {end_of_beam_time}, utcoffset: {eob_time_offset}")
        print(f"spectra_start_time: {spectra_start_time}, utcoffset: {ss_time_offset}")
        # assert eob_time_offset == ss_time_offset, (
        #     f"Timezone offset mismatch! end_of_beam_time ({eob_time_offset}) and spectra_start_time ({ss_time_offset}) do not match. "
        #      "Ensure both datetime objects are in the same timezone."
        # )
        cooling_time = (spectra_start_time - end_of_beam_time).total_seconds()
    else:
        print("Warning: end_of_beam_time is None, no cooling time will be used for decay correction.")




    A_EoB_list = []
    A_SoM_list = []
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

        # activity at start of measurement, for debugging purposes
        A_SoM = hpge_calibration.get_activity_for_peak_at_start_of_measurement(
            peak_area=peak_area,
            peak_energy=peak_energy,
            life_time=pres_i[1].live_time,
            real_time=pres_i[1].real_time,
            branching_ratio=branching_ratio,
            half_life=reaction.half_life
        )
        A_SoM_list.append(A_SoM)

    # calculate the average activity at end of beam, use loop to make sure uncertainties are handled correctly, i did not test np.mean and sum stuff
    sum = 0
    for A_EoB in A_EoB_list:
        sum += A_EoB
        print(f"Activity at end of beam for peak {A_EoB:.uS} Bq")
    A_EoB_avg = sum / len(A_EoB_list)

    sum = 0
    for A_SoM in A_SoM_list:
        sum += A_SoM
    A_SoM_avg = sum / len(A_SoM_list)

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


    if particle_energy_distributions is not None:
        # TODO: implement integral over energy distribution to get effective cross-section
        # raise NotImplementedError("Energy distribution handling is not yet implemented. Please provide a single particle_energy_MeV for now.")
        # build the matrix of flux_i_j where i is the energy bin and j is the depth step into the foil
        # at the moment we assume two depth steps j, E_in and E_out, we assume the cross section changes linearly between these two energies, so we can just take the average cross section between E_in and E_out
        
        # particle_energy_distributions:
        # list of (energy_MeV, phi), where phi is the fluence per energy bin
        # All distributions use the same energy grid and bin boundaries.
        energy_arr = particle_energy_distributions[0][0]  # assuming all distributions use the same energy grid!
        n_ebin = len(energy_arr)
        n_xbin = len(particle_energy_distributions)

        phi_i_j = np.zeros((n_ebin, n_xbin))  # i, j sized array

        for j, (_, phi_i) in enumerate(particle_energy_distributions):
            phi_i_j[:, j] = phi_i

        # interp the IAEA cross section onto the common energy grid for the depth steps j
        # E_iaea: IAEA energies in MeV
        # sigma_iaea: IAEA cross sections in barn
        sigma_i = np.interp(energy_arr, cross_section_data.energy_MeV, cross_section_data.pade_fit_mb)

        sigma_eff_j = np.sum(phi_i_j * sigma_i[:, np.newaxis], axis=0) / np.sum(phi_i_j, axis=0)  # effective cross section at each depth step j

        # effective cross section assuming euqually spaced depth steps, no flux loss and linear change of cross section between depth steps
        sigma = np.mean(sigma_eff_j)  # effective cross section averaged over depth


        print(f"Effective cross section: {sigma:.6g} barn")

    else:
        pen = nominal_value(particle_energy_MeV)
        sigma = ufloat(cs_interp_func(pen), cs_uncertainty_interp_func(pen) - cs_interp_func(pen))  # cross-section in mb
        # sigma = cs_interp_func(pen) # TODO: add uncertainty to cross-section
        # unc = cs_uncertainty_interp_func(pen) - sigma

    # convert mb to b
    sigma = sigma * 1e-3  # cross-section in b
    # convert b to cm^2
    sigma = sigma * 1e-24  # cross-section in cm^2


    _lambda = np.log(2) / reaction.half_life  # decay constant in 1/s

    Q_per_cm2 = (A_EoB_avg * q) / (N_t * sigma * _lambda)

    if electrometer_data_analyzer is not None and irradiation_time is not None:
        raise ValueError("Both electrometer_data_analyzer and irradiation_time are provided. Either you assume constant beam (irradiation_time) or a variable beam (electrometer_data_analyzer). Please provide only one or none.")

    if electrometer_data_analyzer is not None:
        Q_per_cm2 *= electrometer_data_analyzer.get_integrated_correction_factor(half_life=reaction.half_life)
    elif irradiation_time is not None:
        # apply decay correction for irradiation time
        decay_correction_factor = 1 - np.exp(-_lambda * irradiation_time)
        Q_per_cm2 /= decay_correction_factor

    print(f"\nMonitoring reaction: {monitoring_reaction}")
    print(f"Monitor activity: {A_EoB_avg}")
    print(f"Start-of-measurement activity: {A_SoM_avg}")
    # print(f"Monitor foil mass: {monitoring_foil_mass}")
    # print(f"Monitor molar mass: {reaction.molar_mass}")
    # print(f"Monitor atoms N_t: {N_t}")
    print(f"Monitor cross section: {sigma / 1e-24:.6g} barn")
    # print(f"Monitor decay constant: {_lambda:.6g} 1/s")
    # print(f"Calculated Q/cm²: {Q_per_cm2}")

    # print("Monitor gamma activities:", A_EoB_list)
    # print("Interpolated monitor cross section [mb]:", cs_interp_func(particle_energy_MeV.n))
    # print("Monitor foil mass [g]:", monitoring_foil_mass)
    # print("Monitor charge density [C/cm²]:", Q_per_cm2)
    # print("correction_factor_f_c:", correction_factor_f_c)

    return Q_per_cm2
