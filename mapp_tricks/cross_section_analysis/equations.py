import numpy as np # type: ignore
from uncertainties import ufloat # type: ignore
from uncertainties.umath import exp # pylint: disable=no-name-in-module

N_A = 6.022140857E+23 #avogadro number [1/mol]
q = 1.6021766208E-19 #elementary charge [C]

def _get_activity_at_end_of_beam_with_known_cross_section_assume_no_decay(cross_section: ufloat, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area) -> ufloat:
    """
    Calculate the activity at the end of the beam with a known cross section.
    - cross_section: cross section [barn], can be a ufloat
    - half_life: half life of the isotope [s]
    - integrated_charge: integrated charge during the measurement [C]
    - target_mass: mass of the target [g]
    - molar_mass: molar mass of the target element [g/mol]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]
    """

    decay_constant = np.log(2) / half_life  # decay constant [1/s]
    cross_section_cm_from_barn = cross_section * 1e-24  # conversion from barn to cm^2

    return ((cross_section_cm_from_barn * decay_constant * integrated_charge * target_mass * N_A * isotopic_abundance * n_sto) / (q * collimator_area * molar_mass))


def get_activity_at_end_of_beam_with_known_cross_section(cross_section: ufloat, half_life, t_irradiation, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area) -> ufloat:
    """
    Calculate the activity at the end of the beam with a known cross section.
    - cross_section: cross section [barn], can be a ufloat
    - half_life: half life of the isotope [s]
    - t_irradiation: irradiation time [s]
    - integrated_charge: integrated charge during the measurement [C]
    - target_mass: mass of the target [g]
    - molar_mass: molar mass of the target element [g/mol]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]
    """

    decay_constant = np.log(2) / half_life  # decay constant [1/s]
    A_EoB = _get_activity_at_end_of_beam_with_known_cross_section_assume_no_decay(cross_section, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area)
    f_c = decay_constant * t_irradiation / (1 - exp(-decay_constant * t_irradiation))  # adjust for the irradiation time, assume constant production rate (stable beam)
    return A_EoB / f_c

def get_activity_at_end_of_beam_with_known_cross_section_with_integrated_correction_factor(cross_section: ufloat, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area, integrated_correction_factor) -> ufloat:
    """
    Calculate the activity at the end of the beam with a known cross section.
    - cross_section: cross section [barn], can be a ufloat
    - half_life: half life of the isotope [s]
    - t_irradiation: irradiation time [s]
    - integrated_charge: integrated charge during the measurement [C]
    - target_mass: mass of the target [g]
    - molar_mass: molar mass of the target element [g/mol]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]
    - integrated_correction_factor: integrated correction factor, accounting for non uniform beam and short halflives, i.e. calculated by ElectrometerDataAnalyzer.get_integrated_correction_factor()
    """

    A_EoB = _get_activity_at_end_of_beam_with_known_cross_section_assume_no_decay(cross_section, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area)
    return A_EoB / integrated_correction_factor  # adjust for the irradiation time

def _get_cross_section_assume_no_decay(activity_at_end_of_beam: ufloat, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area):
    """
    Calculate the cross section for a given activity.
    - activity_at_end_of_beam: activity at end of beam [Bq], can be a ufloat
    - half_life: half life of the isotope [s]
    - integrated_charge: integrated charge during the measurement [C]
    - molar_mass: molar mass of the target element [g/mol]
    - target_mass: mass of the target [g]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]

    Returns the cross section [barn].
    """

    decay_constant = np.log(2) / half_life

    cross_section = (activity_at_end_of_beam * q * collimator_area / (decay_constant * integrated_charge)) * (molar_mass / (target_mass * N_A * isotopic_abundance * n_sto))
    return cross_section * 1e24  # conversion to barn

def get_cross_section(activity_at_end_of_beam: ufloat, half_life, t_irradiation, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area):
    """
    Calculate the cross section for a given activity.
    - activity_at_end_of_beam: activity at end of beam [Bq], can be a ufloat
    - half_life: half life of the isotope [s]
    - t_irradiation: irradiation time [s]
    - integrated_charge: integrated charge during the measurement [C]
    - molar_mass: molar mass of the target element [g/mol]
    - target_mass: mass of the target [g]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]

    Returns the cross section [barn].
    """

    decay_constant = np.log(2) / half_life

    cross_section = _get_cross_section_assume_no_decay(activity_at_end_of_beam, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area)
    f_c = decay_constant * t_irradiation / (1 - exp(-decay_constant * t_irradiation))  # adjust for the irradiation time
    return cross_section * f_c

def get_cross_section_with_integrated_correction_factor(activity_at_end_of_beam: ufloat, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area, integrated_correction_factor):
    """
    Calculate the cross section for a given activity.
    - activity_at_end_of_beam: activity at end of beam [Bq], can be a ufloat
    - half_life: half life of the isotope [s]
    - integrated_charge: integrated charge during the measurement [C]
    - molar_mass: molar mass of the target element [g/mol]
    - target_mass: mass of the target [g]
    - isotopic_abundance: isotopic abundance of the target element
    - n_sto: stoichiometric coefficient in the reaction
    - collimator_area: area of the collimator [cm^2]
    - integrated_correction_factor: integrated correction factor, accounting for non uniform beam and short halflives, i.e. calculated by ElectrometerDataAnalyzer.get_integrated_correction_factor()

    Returns the cross section [barn].
    """

    cross_section = _get_cross_section_assume_no_decay(activity_at_end_of_beam, half_life, integrated_charge, target_mass, molar_mass, isotopic_abundance, n_sto, collimator_area)
    return cross_section * integrated_correction_factor  # adjust for the irradiation time
