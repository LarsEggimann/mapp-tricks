from uncertainties import ufloat_fromstr, UFloat, Variable # type: ignore
import pandas as pd # type: ignore

def parse_csv(filename: str) -> pd.DataFrame:
    """
    Parse CSV file and convert columns with '+/-' format to ufloat
    """
    df = pd.read_csv(filename)
    # check if columns contain '+/-' in any cell, if so, parse as ufloat
    for col in df.columns:
        if df[col].astype(str).str.contains(r'\+/-').any():
            df[col] = df[col].apply(lambda x: ufloat_fromstr(x) if isinstance(x, str) else x)
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