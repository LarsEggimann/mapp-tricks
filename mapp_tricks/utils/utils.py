from collections.abc import Callable
from typing import Optional
from scipy.interpolate import interp1d
from uncertainties import ufloat_fromstr, UFloat, Variable # type: ignore
import pandas as pd # type: ignore

def _parse_tuple(val: str) -> Optional[tuple[float, float]]:
    """Example: convert range (130, 150) to tuple (130, 150)"""
    if val.strip():
        try:
            return eval(val.strip())
        except Exception:
            print(f"Warning: Failed to parse tuple from value: {val}")
            return None
    return None

def safe_ufloat_parse(val):
    if not isinstance(val, str):
        return val
    try:
        return ufloat_fromstr(val.strip())
    except ValueError:
        # If it fails to parse (e.g. "N/A" or mixed text), return original value
        print(f"Warning: Failed to parse ufloat from value: {val}, returning original value.")
        return val

def parse_csv(filename: str) -> pd.DataFrame:
    """
    Parse CSV file and apply some tests to try to automatically detect and parse columns as ufloat, datetime, or tuple when appropriate.
    The parsing is based on the presence of specific patterns in the column values, if any cell in the column contains a string with the pattern:
    - 'value +/- uncertainty', also detects the bracket notation ufloat, e.g. 1.23(45), it will be parsed as a ufloat.
    - '(number, number)', it will be parsed as a tuple of two floats.
    - 'YYYY-MM-DD HH:MM:SS', it will be parsed as a datetime.
    """
    df = pd.read_csv(filename)
    for col in df.columns:
        col_str = df[col].astype(str)
        # check if columns contain '+/-' or '±' in any cell, if so, parse as ufloat, also check for the bracket notation ufloat, e.g. 1.23(45)
        ufloat_bracket_pattern = r'\d+(?:\.\d+)?\(\d+(?:\.\d+)?\)'
        if col_str.str.contains(r'\+/-', regex=True).any() or col_str.str.contains(f'±', regex=True).any() or col_str.str.contains(ufloat_bracket_pattern, regex=True).any():
            df[col] = df[col].apply(safe_ufloat_parse)

        # when the columns contain a opening and closing bracket and two numbers, parse it as tuple
        elif col_str.str.contains(r'\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)', regex=True).any():
            df[col] = df[col].apply(lambda x: _parse_tuple(x) if isinstance(x, str) else x)

        # if the column contains an ISO datetime string, parse it as datetime
        elif col_str.str.contains(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', regex=True).any():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df

def store_csv(df: pd.DataFrame, filename: str, ufloat_format: str = 'precision') -> None:
    """
    Store DataFrame to a standardized CSV file while preserving the precision of the ufloat
    values. The ufloat_format parameter can be set to 'precision' to store the values with their precision, or 'bracket' to store the nice bracket notation.
    """
    for index, row in df.iterrows():
        for col in df.columns:
            if isinstance(row[col], Variable) or isinstance(row[col], UFloat):
                ufloat_val = row[col]
                if ufloat_format == 'precision':
                    df.at[index, col] = f"{ufloat_val.nominal_value}+/-{ufloat_val.std_dev}"
                elif ufloat_format == 'bracket':
                    df.at[index, col] = f"{ufloat_val:.2uS}"
    df.to_csv(filename, index=False)


def convert_color_hex_to_rgba(hex_color: str) -> str:
    # Remove the hash sign
    hex_str = hex_color.lstrip('#')
    
    # Extract RGB components
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    
    # Extract Alpha component if it exists (8-character hex)
    if len(hex_str) == 8:
        alpha_hex = hex_str[6:8]
        # Divide by 255 to get a float between 0.0 and 1.0
        a = round(int(alpha_hex, 16) / 255, 2)
    else:
        a = 1.0
        
    return f"rgba({r}, {g}, {b}, {a})"

def parse_srim_data_normalized(file_path):
    """Parse SRIM data file and normalize the values to standard units (MeV for energy and cm for length).

    Args:
        file_path str: The path to the SRIM data file.

    Returns:
        pd.DataFrame: A DataFrame containing the normalized SRIM data. Units of the stopping power depends on the units chosen in SRIM, normally I use MeV / (mg cm^2)
                      Columns include:
                        - Energy_MeV: Energy in MeV
                        - dE/dx_Elec: Electronic stopping power
                        - dE/dx_Nuclear: Nuclear stopping power
                        - dE/dx_Total: Total stopping power
                        - Projected_Range_cm: Projected range in cm
                        - Longitudinal_Straggling_cm: Longitudinal straggling in cm
                        - Lateral_Straggling_cm: Lateral straggling in cm
    """
    energy_to_mev = {
        "eV": 1e-6,
        "keV": 1e-3,
        "MeV": 1.0,
        "GeV": 1e3,
    }
    length_to_cm = {
        "A": 1e-8,  # Angstrom to cm
        "um": 1e-4,  # Micrometer to cm
        "mm": 1e-1,  # Millimeter to cm
        "cm": 1.0,  # Centimeter to cm
        "m": 100.0,  # Meter to cm
    }

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    table_lines = []
    in_table = False

    for line in lines:
        # table starts after the dashed line following the header labels
        if "--------------" in line and not in_table:
            in_table = True
            continue
        # table ends at the next long footer divider line
        if "-----------------------------------------------------------" in line:
            break
        if in_table:
            cleaned = line.strip()
            if cleaned:
                table_lines.append(cleaned)

    # parse and normalize rows
    parsed_rows = []
    for line in table_lines:
        parts = line.split()

        # extract values and units
        energy_val, energy_unit = float(parts[0]), parts[1]
        elec_loss = float(parts[2])
        nuc_loss = float(parts[3])
        range_val, range_unit = float(parts[4]), parts[5]
        long_strag_val, long_strag_unit = float(parts[6]), parts[7]
        lat_strag_val, lat_strag_unit = float(parts[8]), parts[9]

        # calculate Total dE/dx
        total_loss = elec_loss + nuc_loss

        # convert everything into standard normalized units (MeV and cm)
        energy_mev = energy_val * energy_to_mev.get(energy_unit, 1.0)
        range_cm = range_val * length_to_cm.get(range_unit, 1.0)
        long_strag_cm = long_strag_val * length_to_cm.get(long_strag_unit, 1.0)
        lat_strag_cm = lat_strag_val * length_to_cm.get(lat_strag_unit, 1.0)

        parsed_rows.append(
            [
                energy_mev,
                elec_loss,
                nuc_loss,
                total_loss,
                range_cm,
                long_strag_cm,
                lat_strag_cm,
            ]
        )

    columns = [
        "Energy_MeV",
        "dE/dx_Elec",
        "dE/dx_Nuclear",
        "dE/dx_Total",
        "Projected_Range_cm",
        "Longitudinal_Straggling_cm",
        "Lateral_Straggling_cm",
    ]

    return pd.DataFrame(parsed_rows, columns=columns)

def interpolate_srim_data(df: pd.DataFrame) -> Callable:
    """Interpolate SRIM data, return a function that can be used to get stopping power at any energy within the range of the data.
    Args:
        df (pd.DataFrame): DataFrame containing SRIM data with columns 'Energy_MeV' and 'dE/dx_Total'. According to `parse_srim_data_normalized`,
    Returns:
        Callable: A function that takes an energy value (or array of values) in MeV and returns the interpolated stopping power in MeV/(mg/cm²).
    """
    
    # create an interpolation function
    interp_func = interp1d(
        df["Energy_MeV"],
        df["dE/dx_Total"],
        kind="linear",
        fill_value="extrapolate",
    )

    return interp_func


def parse_iaea_monitor_reaction(file_path):
    """Parse an IAEA monitor reaction data file.

    Returns:
        pd.DataFrame: Parsed IAEA monitor reaction data.
    The expected columns in the DataFrame are:
        - energy_MeV: Energy in MeV
        - pade_fit_mb: Pade fit in millibarns
        - pade_fit_uncertainty_mb: Uncertainty of the Pade fit in millibarns
        - physical_energy_MeV: Physical energy in MeV, should be the same as energy_MeV
        - physical_yield_MBq_uAh: Physical yield in MBq/uAh
        - physical_yield_mCi_uAh: Physical yield in mCi/uAh
        - physical_yield_1h_MBq_uA: Physical yield after 1 hour in MBq/uA
        - physical_yield_saturation_MBq_uA: Physical yield at saturation in MBq/uA
    """
    columns = [
        "energy_MeV",
        "pade_fit_mb",
        "pade_fit_uncertainty_mb",
        "physical_energy_MeV",
        "physical_yield_MBq_uAh",
        "physical_yield_mCi_uAh",
        "physical_yield_1h_MBq_uA",
        "physical_yield_saturation_MBq_uA",
    ]

    df = pd.read_csv(file_path, sep="\\s+", comment="#", header=None, names=columns, skiprows=5)

    return df
