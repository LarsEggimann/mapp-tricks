"""
Parser module for spectrum files.

This module provides functions to parse spectrum files and extract
energy calibration parameters and data.
"""

import os
import glob
import re
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

    # Extract total gamma count
    total_gamma_count = None
    for line in lines:
        if line.startswith("# Total counts:") or line.startswith("TotalGammaCounts: "):
            total_gamma_count = float(line.split(":")[1].split()[0].strip())
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

    return df, (0, 0, 0, 0), start_time, real_time, live_time, total_gamma_count

def sum_spectras(paths_to_txt: list[str], result_file_path_name: str):
    """
    Sum the spectra from multiple text files.

    Parameters
    ----------
    paths_to_txt : list[str]
        List of paths to the text files

    Returns
    -------
    DataFrame
        DataFrame containing the summed spectra
    """

    summed_df = pd.DataFrame()
    start_times = []
    live_times = []
    real_times = []
    total_gamma_counts = []

    for path in paths_to_txt:

        df, _, start_time, real_time, live_time, total_gamma_count = parse_spectrum_file(path)

        # # check the first df entry, if it starts with channel 0, delete it
        # if not df.empty and df.iloc[0]['channel'] == 0:
        #     df = df.iloc[1:]


        start_times.append(start_time)
        real_times.append(real_time)
        live_times.append(live_time)
        total_gamma_counts.append(total_gamma_count)

        # sum "energy" of spectra
        if summed_df.empty:
            summed_df = df
        else:
            summed_df['counts'] = summed_df['counts'] + df['counts']

    # find earliest start_time
    earliest_start_time = min(start_times)

    # sum live and real times
    summed_live_time = sum(live_times)
    summed_real_time = sum(real_times)
    summed_total_gamma_count = sum(total_gamma_counts)

    # text file example exported from InterSpect

    # Original File Name: /tmp/summed_0f8f-53d9-4ab3-49ad
    # TotalGammaLiveTime: 355880 seconds
    # TotalRealTime: 356400 seconds
    # TotalGammaCounts: 4.05058e+06 seconds
    # TotalNeutron: 0 seconds
    # Remark: N42 file created by: InterSpec
    # Remark: MCA Type: Lynx


    # StartTime: 2025-08-27T14:28:04.021681
    # LiveTime: 355880 seconds
    # RealTime: 356400 seconds
    # SampleNumber: 1
    # DetectorName: My ADC
    # Title: Combination-20250901 09:07:25
    # EquationType: Polynomial
    # Coefficients: -0.126274 0.243864
    # Channel Energy Counts
    # 0 -0.126274 0
    # 1 0.11759 0
    # 2 0.361455 0

    # create folder if not exists
    os.makedirs(os.path.dirname(result_file_path_name), exist_ok=True)

    with open(result_file_path_name, 'w') as f:
        f.write(f"Original File Name: {path}\n")
        f.write(f"TotalGammaLiveTime: {summed_live_time} seconds\n")
        f.write(f"TotalRealTime: {summed_real_time} seconds\n")
        f.write(f"TotalGammaCounts: {int(summed_total_gamma_count)} seconds\n")
        f.write(f"TotalNeutron: 0 seconds\n")
        f.write(f"Remark: N42 file created by: Custom Python Script made by Lars Eggimann\n")
        f.write(f"Remark: MCA Type: Lynx\n")
        f.write(f"\n")
        f.write(f"StartTime: {earliest_start_time}\n")
        f.write(f"LiveTime: {summed_live_time} seconds\n")
        f.write(f"RealTime: {summed_real_time} seconds\n")
        f.write(f"SampleNumber: 1\n")
        f.write(f"DetectorName: My ADC\n")
        f.write(f"Title: Summation-{datetime.now()}\n")
        f.write(f"EquationType: Polynomial\n")
        f.write(f"Coefficients: -0.126274 0.243864\n")
        f.write(f"Channel Energy Counts\n")
        for index, row in summed_df.iterrows():
            f.write(f"{row['channel']} {row['energy']} {row['counts']}\n")

    print(f"Summed spectra saved to {result_file_path_name}")

    return summed_df, earliest_start_time, summed_real_time, summed_live_time

def sum_spectras_matching_pattern_in_folder(folder_path: str, pattern: str, result_file_name: str):
    """
    Sum the spectra from multiple text files in a folder matching a specific pattern.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the text files
    pattern : str
        Pattern to match the text files. Can be a single pattern or a list of patterns.
        For multiple patterns, separate with '|' (e.g., '*009.txt|*010.txt')
    result_file_name : str
        Name of the result file to save the summed spectra

    Returns
    -------
    DataFrame
        DataFrame containing the summed spectra
    """
    
    # handle multiple patterns separated by '|'
    if '|' in pattern:
        patterns = pattern.split('|')
        paths_to_txt = []
        for p in patterns:
            paths_to_txt.extend(glob.glob(os.path.join(folder_path, p.strip())))
        # remove duplicates and sort
        paths_to_txt = sorted(list(set(paths_to_txt)))
    else:
        paths_to_txt = glob.glob(os.path.join(folder_path, pattern))
    
    print(f"Found {len(paths_to_txt)} files matching pattern '{pattern}' in folder '{folder_path}'")
    print(f"Files: {paths_to_txt}")
    result_file_path = os.path.join(folder_path, result_file_name)
    return sum_spectras(paths_to_txt, result_file_path)



def sum_spectra_in_folder(folder_path: str, group_size: int = 4, prefix: str = "sum"):
    """
    Groups spectra files in a folder and sums them up in numeric order.

    Args:
        folder_path (str): Path to the folder containing spectra files.
        group_size (int): Number of spectra to sum in one group. Default = 4.
        prefix (str): Subfolder prefix for result files. Default = "sum".
    """
    # find all spectra files (*.txt)
    files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not files:
        print(f"No spectra files found in {folder_path}")
        return

    # extract numeric part and sort numerically
    def extract_num(fname):
        match = re.search(r"(\d+)", os.path.basename(fname))
        return int(match.group(1)) if match else float("inf")

    files = sorted(files, key=extract_num)

    # results folder
    result_dir = os.path.join(folder_path, prefix)
    os.makedirs(result_dir, exist_ok=True)

    # loop through files in groups
    for i in range(0, len(files), group_size):
        group = files[i:i + group_size]
        if not group:
            continue

        # build pattern (OR-separated filenames)
        pattern = "|".join(os.path.basename(f) for f in group)

        # get start/end numbers
        start_num = extract_num(group[0])
        end_num = extract_num(group[-1])

        result_file_name = os.path.join(result_dir, f"summed_{start_num}-{end_num}.txt")

        sum_spectras_matching_pattern_in_folder(
            folder_path=folder_path,
            pattern=pattern,
            result_file_name=result_file_name
        )

        print(f"Summed {len(group)} spectra → {result_file_name}")
