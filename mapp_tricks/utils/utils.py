from typing import Optional
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

def parse_csv(filename: str) -> pd.DataFrame:
    """
    Parse CSV file and apply some tests to try to automatically detect and parse columns as ufloat, datetime, or tuple when appropriate.
    The parsing is based on the presence of specific patterns in the column values, if any cell in the column contains a string with the pattern:
    - 'value +/- uncertainty', it will be parsed as a ufloat.
    - '(number, number)', it will be parsed as a tuple of two floats.
    - 'YYYY-MM-DD HH:MM:SS', it will be parsed as a datetime.
    """
    df = pd.read_csv(filename)
    for col in df.columns:
        # check if columns contain '+/-' in any cell, if so, parse as ufloat
        if df[col].astype(str).str.contains(r'\+/-').any():
            df[col] = df[col].apply(lambda x: ufloat_fromstr(x) if isinstance(x, str) else x)

        # when the columns contain a opening and closing bracket and two numbers, parse it as tuple
        elif df[col].astype(str).str.contains(r'\(\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\)').any():
            df[col] = df[col].apply(lambda x: _parse_tuple(x) if isinstance(x, str) else x)

        # if the column contains an ISO datetime string, parse it as datetime
        elif df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').any():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df

def store_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Store DataFrame to a standardized CSV file while preserving the precision of the ufloat
    """
    for index, row in df.iterrows():
        for col in df.columns:
            if isinstance(row[col], Variable) or isinstance(row[col], UFloat):
                ufloat_val = row[col]
                df.at[index, col] = f"{ufloat_val.nominal_value}+/-{ufloat_val.std_dev}"
    df.to_csv(filename, index=False)


def convert_color_hex_to_rgba(hex_color: str) -> str:
    hex_str = hex_color.replace('#', '')
    if len(hex_str) != 6:
        hex_str = hex_str + '0'
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    a = 1
    if len(hex_str) > 6:
        alpha = float(int(hex_str[6:10], 16) / 255 * 1)
        a = round(alpha, 2)
    return 'rgba({},{},{},{})'.format(r, g, b, a)




def parse_srim_data_normalized(file_path):
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
