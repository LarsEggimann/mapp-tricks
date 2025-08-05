"""
Parser module for spectrum files.

This module provides functions to parse spectrum files and extract
energy calibration parameters and data.
"""

from datetime import datetime
import pandas as pd


def parse_spectrum_file(filepath):
    """
    Parse a spectrum file to extract energy calibration and data.
    
    Parameters
    ----------
    filepath : str
        Path to the spectrum file
        
    Returns
    -------
    tuple
        (DataFrame with columns ['channel', 'energy', 'counts', 'rate'], 
         tuple of calibration parameters (A0, A1, A2, A3),
         datetime start_time,
         float real_time in seconds,
         float live_time in seconds)
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Extract Start Time: # Start time:    2025-05-07, 14:07:49
    start_time = None
    for line in lines:
        # Start time:    2025-07-31, 14:24:40
        if line.startswith("# Start time:"):
            start_time = ':'.join(line.split(":")[1:]).strip()
            start_time = datetime.strptime(start_time, "%Y-%m-%d, %H:%M:%S")
            break

    # Extract real_time
    real_time = None
    for line in lines:
        if line.startswith("# Real time (s):"):
            real_time = float(line.split(":")[1].strip())
            break

    # Extract live_time
    live_time = None
    for line in lines:
        if line.startswith("# Live time (s):"):
            live_time = float(line.split(":")[1].strip())
            break
    
    # Extract energy calibration
    for i, line in enumerate(lines):
        if line.startswith("#     A0:"):
            A0 = float(line.split(":")[1])
            A1 = float(lines[i+1].split(":")[1])
            A2 = float(lines[i+2].split(":")[1])
            A3 = float(lines[i+3].split(":")[1])
            break
    
    # Find start of data
    for i, line in enumerate(lines):
        if line.startswith("#-----------------------------------------------------------------------"):
            data_start = i + 1
            break
    
    df = pd.read_csv(filepath, sep='\t', skiprows=data_start, 
                     names=["channel", "energy", "counts", "rate"])

    return df, (A0, A1, A2, A3), start_time, real_time, live_time
