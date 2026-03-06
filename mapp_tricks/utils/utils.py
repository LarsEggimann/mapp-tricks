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