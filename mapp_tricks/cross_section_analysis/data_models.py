import os

class GammaPeakAndTargetInfo:
    def __init__(self, name, peak_energy_range, half_life, branching_ratio, target_mass, molar_mass, isotopic_abundance, n_sto=1, collimator_area=None):
        """
        Initialize the isotope and target information.
        """
        self.name = name
        self.peak_energy_range = peak_energy_range
        self.half_life = half_life  # in seconds
        self.branching_ratio = branching_ratio
        self.target_mass = target_mass
        self.molar_mass = molar_mass
        self.isotopic_abundance = isotopic_abundance
        self.n_sto = n_sto # stoichiometric coefficient
        self.collimator_area = collimator_area

class DataSource:
    def __init__(self, data_folder, spectra_file, orbitos_file, spectra_level=None, spectra_with_aluminum_foil=False):
        self.data_folder = data_folder
        self.spectra_file = spectra_file
        self.spectra_level = spectra_level
        self.spectra_with_aluminum_foil = spectra_with_aluminum_foil
        self.spectra_path = os.path.join(data_folder, spectra_file)
        self.orbitos_file = orbitos_file
        self.orbitos_path = os.path.join(data_folder, orbitos_file)


import pandas as pd
from pandas import DataFrame, Series
import uncertainties
from uncertainties import ufloat_fromstr, ufloat
from datetime import datetime
from typing import Optional, Callable, Any, Dict


# parsing funcs
def parse_float(val) -> Optional[float]:
    return float(val) if val else None

def parse_int(val) -> Optional[int]:
    return int(val) if val else None

def parse_bool(val) -> Optional[bool]:
    if val:
        if type(val) is bool:
            return val
        if type(val) is str:
            lower = val.strip().lower()
            if lower in {"true"}:
                return True
            elif lower in {"false"}:
                return False
        if type(val) is int:
            return bool(val)
    return None

def parse_ufloat(val) -> Optional[ufloat]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        return ufloat_fromstr(val)
    return val   # already parsed object, or pass through

def parse_datetime(val: str) -> Optional[datetime]:
    return datetime.fromisoformat(val) if val.strip() else None

def parse_tuple(val: str) -> Optional[tuple[float, float]]:
    """Example: convert range (130, 150) to tuple (130, 150)"""
    if val.strip():
        try:
            return eval(val.strip())
        except Exception:
            return None
    return None

def parse_identity(val):
    return val


# schema: column_name: parser function
PARSERS: Dict[str, Callable[[str], Any]] = {
    "target": parse_identity,
    "reaction": parse_identity,
    "notes": parse_identity,
    "beam_energy_MeV": parse_ufloat,
    "peak_energy_range_keV": parse_tuple,
    "half_life_s": parse_ufloat,
    "branching_ratio": parse_ufloat,
    "target_material_mass_g": parse_ufloat,
    "molar_mass": parse_ufloat,
    "isotopic_abundance": parse_ufloat,
    "n_sto": parse_int,                         # stoichiometric coefficient
    "collimator_area_cm2": parse_ufloat,
    "spectra_level": parse_int,
    "spectra_with_aluminum_foil": parse_bool,
    "data_source_folder": parse_identity,
    "spectra_file": parse_identity,
    "orbitos_file": parse_identity,
    "output_folder": parse_identity,

    # Calculated fields
    "cross_section_b": parse_ufloat,
    "activity_at_end_of_beam_Bq": parse_ufloat,
    "activity_at_start_of_spectra_Bq": parse_ufloat,
    "start_of_beam_time": parse_datetime,
    "end_of_beam_time": parse_datetime,
    "irradiation_time_s": parse_float,
    "integrated_charge_C": parse_ufloat,
    "integrated_correction_factor": parse_float,
    "cooling_time_s": parse_float,
    "spectra_start_time": parse_datetime,
    "spectra_end_time": parse_datetime,
    "spectra_real_time_s": parse_float,
    "spectra_live_time_s": parse_float,
    "interpolated_Mo98_Tc99_cross_section_b": parse_ufloat,
    "expected_peak_activity_from_98Mo_at_EoB_Bq": parse_ufloat,
    "corrected_peak_activity_at_EoB_Bq": parse_ufloat,
    "corrected_100Mo_99Tc_cross_section_barns": parse_ufloat
}


def parse_csv(filename: str) -> DataFrame:
    """
    Read DataFrame from a standardized CSV file and apply parsing.
    The resulting DataFrame will have typed columns according to the PARSERS mapping.
    - target: str
    - reaction: str
    - notes: str
    - beam_energy_MeV: ufloat
    - peak_energy_range_keV: tuple[float, float]
    - half_life_s: ufloat
    - branching_ratio: ufloat
    - target_material_mass_g: ufloat
    - molar_mass: ufloat
    - isotopic_abundance: ufloat
    - collimator_area_cm2: ufloat
    - spectra_level: int
    - spectra_with_aluminum_foil: bool
    - data_source_folder: str
    - spectra_file: str
    - orbitos_file: str
    - output_folder: str

    The following parameters are only available after calculation


    Notes
    -----
    The file paths need to be relative to the data source folder!
    
    """
    df = pd.read_csv(filename)
    # column-wise parsing
    for col, parser in PARSERS.items():
        if col in df.columns:
            df[col] = df[col].map(parser)
    return df

def store_csv(df: DataFrame, filename: str) -> None:
    """
    Store DataFrame to a standardized CSV file while preserving the precision of the ufloat

    """
    for index, row in df.iterrows():
        for col in df.columns:
            if isinstance(row[col], uncertainties.core.Variable) or isinstance(row[col], uncertainties.UFloat):
                ufloat_val = row[col]
                df.at[index, col] = f"{ufloat_val.nominal_value}+/-{ufloat_val.std_dev}"
    df.to_csv(filename)
