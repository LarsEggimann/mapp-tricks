from datetime import datetime, timedelta
from uncertainties import ufloat # type: ignore
from ..peakfit import PeakFitter
from ..orbitos_utils import analyze_electrometer_data
from ..spectrometer_calibrations import HPGeCalibration
from ..cross_section_analysis import GammaPeakAndTargetInfo, DataSource

class CrossSectionAnalysisResults:
    def __init__(self, activity_end_of_beam: ufloat, cross_section: ufloat):
        self.activity_end_of_beam = activity_end_of_beam  # in Bq
        self.cross_section = cross_section  # in barn

def do_cross_section_analysis(
    data_source: DataSource,
    target: GammaPeakAndTargetInfo
):
    fitter = PeakFitter()

    spectra_data = fitter.process_file(
        filepath=data_source.spectra_path,
        energy_range=target.peak_energy_range,
    )

    electrometer_data = analyze_electrometer_data(data_source.orbitos_path)

    cooling_time = (spectra_data.start_time - electrometer_data.end_of_beam).total_seconds()

    calibration = HPGeCalibration(level=data_source.spectra_level, with_aluminum_foil=data_source.spectra_with_aluminum_foil)

    A_EoB = calibration.get_activity_for_peak_at_end_of_beam(
        peak_area=spectra_data.area,
        peak_energy=spectra_data.centroid,
        life_time=spectra_data.live_time,
        real_time=spectra_data.real_time,
        cooling_time=cooling_time,
        branching_ratio=target.branching_ratio,
        half_life=target.half_life,
    )

    A_start_of_spectra = calibration.get_activity_for_peak_at_start_of_measurement(
        peak_area=spectra_data.area,
        peak_energy=spectra_data.centroid,
        life_time=spectra_data.live_time,
        real_time=spectra_data.real_time,
        branching_ratio=target.branching_ratio,
        half_life=target.half_life,
    )
    spectra_end = spectra_data.start_time + timedelta(seconds=spectra_data.real_time)
    print(f'Spectra Start Time: {spectra_data.start_time}, Spectra End Time: {spectra_end}')
    print(f"Beam Start Time: {electrometer_data.start_of_beam}, Beam End Time: {electrometer_data.end_of_beam}")
    print(f"Activity at end of beam: {A_EoB:.10f} Bq, cooling time: {cooling_time:.10f} s")
    print(f"Activity at start of spectra recording: {A_start_of_spectra:.10f} Bq")

    cs = calibration.get_cross_section(
        activity_at_end_of_beam=A_EoB,
        target_mass=target.target_mass,
        molar_mass=target.molar_mass,
        isotopic_abundance=target.isotopic_abundance,
        n_sto=target.n_sto,  # stoichiometric coefficient
        t_irradiation=electrometer_data.t_irradiation,
        collimator_area=target.collimator_area,
        half_life=target.half_life,
        integrated_charge=electrometer_data.integrated_charge # C
    )

    print(f"Cross section: {cs*1e3:.10f} mb")

    return CrossSectionAnalysisResults(
        activity_end_of_beam=A_EoB,
        cross_section=cs
    )