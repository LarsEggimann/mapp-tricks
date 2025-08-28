import os
import numpy as np # type: ignore
from datetime import datetime, timedelta
from uncertainties import ufloat # type: ignore
import pandas as pd # type: ignore
from .equations import get_cross_section, get_cross_section_with_integrated_correction_factor
from ..peakfit import PeakFitter
from ..orbitos_utils import ElectrometerDataAnalyzer, BeamData
from ..spectrometer_calibrations import HPGeCalibration

class CrossSectionAnalysisResults:
    def __init__(self, activity_end_of_beam: ufloat, cross_section: ufloat):
        self.activity_end_of_beam = activity_end_of_beam  # in Bq
        self.cross_section = cross_section  # in barn

def do_cross_section_analysis(
    df: pd.DataFrame,
    row_condition
):
    """
    Expects a DataFrame for cross section analysis and a row condition.
    """

    df = df.copy()
    df_one_row: pd.DataFrame = df.loc[row_condition]

    if len(df_one_row) != 1:
        raise ValueError("Expected a single row DataFrame for cross section analysis. Choose a target, and peak?")

    row_idx = df_one_row.index[0]   # keep the row index
    df_row = df_one_row.squeeze()      # for convenience in code below

    fitter = PeakFitter()

    spectra_data = fitter.process_file(
        filepath=os.path.join(df_row.data_source_folder, df_row.spectra_file),
        energy_range=df_row.peak_energy_range_keV,
    )

    ea = ElectrometerDataAnalyzer(os.path.join(df_row.data_source_folder, df_row.orbitos_file))
    electrometer_data = ea.analyze_beam_data()
    

    cooling_time = (spectra_data.start_time - electrometer_data.end_of_beam).total_seconds()

    calibration = HPGeCalibration(level=df_row.spectra_level, with_aluminum_foil=df_row.spectra_with_aluminum_foil)

    A_EoB = calibration.get_activity_for_peak_at_end_of_beam(
        peak_area=spectra_data.area,
        peak_energy=spectra_data.centroid,
        life_time=spectra_data.live_time,
        real_time=spectra_data.real_time,
        cooling_time=cooling_time,
        branching_ratio=df_row.branching_ratio,
        half_life=df_row.half_life_s,
    )

    A_start_of_spectra = calibration.get_activity_for_peak_at_start_of_measurement(
        peak_area=spectra_data.area,
        peak_energy=spectra_data.centroid,
        life_time=spectra_data.live_time,
        real_time=spectra_data.real_time,
        branching_ratio=df_row.branching_ratio,
        half_life=df_row.half_life_s,
    )
    
    # cs = get_cross_section(
    #     activity_at_end_of_beam=A_EoB,
    #     target_mass=df_row.target_material_mass_g,
    #     molar_mass=df_row.molar_mass,
    #     isotopic_abundance=df_row.isotopic_abundance,
    #     n_sto=df_row.n_sto,  # stoichiometric coefficient
    #     t_irradiation=electrometer_data.t_irradiation,
    #     collimator_area=df_row.collimator_area_cm2,
    #     half_life=df_row.half_life_s,
    #     integrated_charge=electrometer_data.integrated_charge # C
    # )

    integrated_correction_factor = ea.get_integrated_correction_factor(half_life=df_row.half_life_s.n)

    cs = get_cross_section_with_integrated_correction_factor(
        activity_at_end_of_beam=A_EoB,
        target_mass=df_row.target_material_mass_g,
        molar_mass=df_row.molar_mass,
        isotopic_abundance=df_row.isotopic_abundance,
        n_sto=df_row.n_sto,  # stoichiometric coefficient
        collimator_area=df_row.collimator_area_cm2,
        half_life=df_row.half_life_s,
        integrated_charge=electrometer_data.integrated_charge, # C
        integrated_correction_factor=integrated_correction_factor
    )


    # add the results to the DataFrame where the condition applies
    df.loc[row_idx, "cross_section_b"] = cs
    df.loc[row_idx, "activity_at_end_of_beam_Bq"] = A_EoB
    df.loc[row_idx, "activity_at_start_of_spectra_Bq"] = A_start_of_spectra
    df.loc[row_idx, "start_of_beam_time"] = electrometer_data.start_of_beam
    df.loc[row_idx, "end_of_beam_time"] = electrometer_data.end_of_beam
    df.loc[row_idx, "irradiation_time_s"] = electrometer_data.t_irradiation
    df.loc[row_idx, "integrated_charge_C"] = electrometer_data.integrated_charge
    df.loc[row_idx, "integrated_correction_factor"] = integrated_correction_factor
    df.loc[row_idx, "cooling_time_s"] = cooling_time
    df.loc[row_idx, "spectra_start_time"] = spectra_data.start_time
    df.loc[row_idx, "spectra_end_time"] = spectra_data.start_time + timedelta(seconds=spectra_data.real_time)
    df.loc[row_idx, "spectra_real_time_s"] = spectra_data.real_time
    df.loc[row_idx, "spectra_live_time_s"] = spectra_data.live_time

    return df
