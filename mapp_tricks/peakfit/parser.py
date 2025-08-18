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
        # StartTime: 2025-08-15T15:20:33.988032
        if line.startswith("StartTime:"):
            start_time = datetime.fromisoformat(':'.join(line.split(":")[1:]).strip())
            break

    # Extract real_time
    real_time = None
    for line in lines:
        if line.startswith("# Real time (s):") or line.startswith("RealTime: "):
            real_time = float(line.split(":")[1].split()[0].strip())
            break

    # Extract live_time
    live_time = None
    for line in lines:
        if line.startswith("# Live time (s):") or line.startswith("LiveTime: "):
            live_time = float(line.split(":")[1].split()[0].strip())
            break
    
    # Find format of data if first line is "#" then we have converted from cnf
    if lines[0].startswith("#"):
        for i, line in enumerate(lines):
            if line.startswith("#-----------------------------------------------------------------------"):
                data_start = i + 1
                break
        df = pd.read_csv(filepath, sep='\t', skiprows=data_start, 
                     names=["channel", "energy", "counts", "rate"])

    else:
        # InterSpect text output format
        for i, line in enumerate(lines):
            if line.startswith("Channel Energy Counts"):
                data_start = i + 1
                break
        df = pd.read_csv(filepath, sep=' ', skiprows=data_start, 
                     names=["channel", "energy", "counts"])

    return df, (0, 0, 0, 0), start_time, real_time, live_time
