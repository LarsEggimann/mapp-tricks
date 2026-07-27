"""
Parser module for spectrum files.

This module provides functions to parse spectrum files and extract
energy calibration parameters and data.
"""

import os
import glob
from datetime import datetime
import pandas as pd # type: ignore
import numpy as np # type: ignore
from pyparsing import Literal
from uncertainties import ufloat # type: ignore
from zoneinfo import ZoneInfo

from .models import PeakFitResult, SpectrometryData
from .read_cnf import read_cnf_file


def parse_spectrum_file(filepath, timezone: ZoneInfo = ZoneInfo("Europe/Zurich")) -> SpectrometryData:
    """
    Parse a spectrum file to extract energy calibration and data into a SpectrometryData object.
    
    Parameters
    ----------
    filepath : str
        Path to the spectrum file
    timezone : ZoneInfo, optional
        Timezone for the start time (default: "Europe/Zurich")

    Returns
    -------
    SpectrometryData
        Dataclass containing sample metadata, calibration data, and channel measurements.
    """
    path = os.path.abspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    tz_info = timezone

    # --- Initialize Defaults ---
    sample_name = os.path.basename(filepath)
    sample_id = ""
    sample_type = ""
    user_name = ""
    sample_description = ""
    
    start_time = datetime.now(tz_info)
    real_time = 0.0
    live_time = 0.0
    total_gamma_count = 0
    
    left_marker = 0
    right_marker = 0
    counts_in_markers = 0
    
    energy_coefficients = [0.0, 0.0, 0.0, 0.0]
    shape_coefficients  = [0.0, 0.0, 0.0, 0.0]
    energy_unit = "keV"
    
    channels = []
    energys = []
    counts = []

    if path.endswith(".cnf") or path.endswith(".CNF"):
        res = read_cnf_file(path)

        # update metadata if available in CNF
        sample_name = res.get("Sample name", sample_name)
        sample_id = str(res.get("Sample id", sample_id))
        sample_type = str(res.get("Sample type", sample_type))
        user_name = str(res.get("User name", user_name))
        sample_description = str(res.get("Sample description", sample_description))

        if "Start time" in res:
            start_time = datetime.strptime(res["Start time"], "%d-%m-%Y, %H:%M:%S")
            start_time = start_time.replace(tzinfo=tz_info)
            
        real_time = float(res.get("Real time", real_time))
        live_time = float(res.get("Live time", live_time))
        total_gamma_count = int(res.get("Total counts", total_gamma_count))
        
        left_marker = int(res.get("Left marker", left_marker))
        right_marker = int(res.get("Right marker", right_marker))
        counts_in_markers = int(res.get("Counts in markers", counts_in_markers))

        energy_coefficients = res.get("Energy coefficients", energy_coefficients)
        shape_coefficients = res.get("Shape coefficients", shape_coefficients)
        energy_unit = res.get("Energy unit", energy_unit)

        channels = res.get("Channels", [])
        energys = res.get("Energy", [])
        counts = res.get("Channels data", [])
            
    elif path.endswith(".txt"):
        with open(path) as f:
            lines = f.readlines()
            
        # Extract Start Time
        for line in lines:
            if line.startswith("# Start time:"):
                start_time_str = ':'.join(line.split(":")[1:]).strip()
                start_time = datetime.strptime(start_time_str, "%Y-%m-%d, %H:%M:%S")
                start_time = start_time.replace(tzinfo=tz_info)
                break
            if line.startswith("StartTime:"):
                start_time_str = ':'.join(line.split(":")[1:]).strip()
                start_time = datetime.fromisoformat(start_time_str)
                start_time = start_time.replace(tzinfo=tz_info)
                break

        # Extract real_time
        for line in lines:
            if line.startswith("# Real time (s):") or line.startswith("RealTime: "):
                real_time = float(line.split(":")[1].split()[0].strip())
                break

        # Extract live_time
        for line in lines:
            if line.startswith("# Live time (s):") or line.startswith("LiveTime: "):
                live_time = float(line.split(":")[1].split()[0].strip())
                break

        # Extract total gamma count
        total_parsed = None
        for line in lines:
            if line.startswith("# Total counts:") or line.startswith("TotalGammaCounts: "):
                total_parsed = int(float(line.split(":")[1].split()[0].strip()))
                break

        # Find format of data, if it starts with '#' it's file converted with cnfconv
        if lines[0].startswith("#"):
            # Converted from CNF format
            for i, line in enumerate(lines):
                if line.startswith("#-----------------------------------------------------------------------"):
                    data_start = i + 1
                    break
            df = pd.read_csv(path, sep='\t', skiprows=data_start, 
                             names=["channel", "energy", "counts", "rate"])

            # we know its a cnfconv converted file, so we can extract the some more metadata
            # Energy calibration coefficients ( E = sum(Ai * n**i) )
            #     A0: -0.126274
            #     A1: 0.243864
            #     A2: 0.000000
            #     A3: 0.000000
            # Energy unit: keV
            if "Energy calibration coefficients" in lines:
                coeffs_start = lines.index("Energy calibration coefficients ( E = sum(Ai * n**i) )\n") + 1
                energy_coefficients = []
                for j in range(4):
                    coeff_line = lines[coeffs_start + j]
                    coeff_value = float(coeff_line.split(":")[1].strip())
                    energy_coefficients.append(coeff_value)
                energy_unit_line = lines[coeffs_start + 4]
                energy_unit = energy_unit_line.split(":")[1].strip()

        else:
            # InterSpect text output format
            for i, line in enumerate(lines):
                if line.startswith("Channel Energy Counts"):
                    data_start = i + 1
                    break
            df = pd.read_csv(path, sep=r'\s+', skiprows=data_start, 
                             names=["channel", "energy", "counts"])
            # we know its an InterSpect exported file, so we can extract some more metadata
            # Coefficients: -0.126274 0.243864
            for line in lines:
                if line.startswith("Coefficients:"):
                    coeffs_str = line.split(":")[1].strip()
                    energy_coefficients = [float(x) for x in coeffs_str.split()]
                    break

        # Extract lists from dataframe
        channels = df["channel"].astype(int).tolist()
        energys = df["energy"].astype(float).tolist()
        counts = df["counts"].astype(int).tolist()

        if total_parsed is not None:
            total_gamma_count = total_parsed
        else:
            total_gamma_count = sum(counts)

    else:
        raise ValueError(f"Unsupported file format: {path}. Only .txt and .cnf files are supported.")


    return SpectrometryData(
        sample_name=sample_name,
        sample_id=sample_id,
        sample_type=sample_type,
        user_name=user_name,
        sample_description=sample_description,
        start_time=start_time,
        real_time=real_time,
        live_time=live_time,
        total_counts=total_gamma_count,
        left_marker=left_marker,
        right_marker=right_marker,
        counts_in_markers=counts_in_markers,
        energy_coefficients=energy_coefficients,
        shape_coefficients=shape_coefficients,
        energy_unit=energy_unit,
        channels=channels,
        energy=energys,
        channels_data=counts
    )

def _load_spectra_from_files(file_paths: list[str]) -> list[SpectrometryData]:
    """
    Load spectra from a list of file paths.

    Parameters
    ----------
    file_paths : list[str]
        List of file paths to load spectra from

    Returns
    -------
    list[SpectrometryData]
        List of SpectrometryData objects loaded from the files
    """
    loaded_spectra: list[SpectrometryData] = []
    for path in file_paths:
        try:
            sd = parse_spectrum_file(path)
            loaded_spectra.append(sd)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return loaded_spectra

def sum_spectra(loaded_spectra: list[SpectrometryData], result_file_path_name: str) -> SpectrometryData:
    """
    Sum the spectra from multiple text files.

    Parameters
    ----------
    loaded_spectra : list[SpectrometryData]
        List of loaded spectra data

    Returns
    -------
    SpectrometryData
        Dataclass containing the summed spectra
    """

    summed_array = np.array([])
    start_times = []
    live_times = []
    real_times = []
    total_gamma_counts = []

    for sd in loaded_spectra:

        start_times.append(sd.start_time)
        real_times.append(sd.real_time)
        live_times.append(sd.live_time)
        total_gamma_counts.append(sd.total_counts)

        # sum "counts" of spectra
        if summed_array.size == 0:
            summed_array = np.array(sd.channels_data)
        else:
            summed_array = summed_array + np.array(sd.channels_data)

    # find earliest start_time
    earliest_start_time = min(start_times)

    # sum live and real times
    summed_live_time = sum(live_times)
    summed_real_time = sum(real_times)
    summed_total_gamma_count = sum(total_gamma_counts)

    # take last spectrometry data object and update it to reflect the summed data
    sd.channels_data = summed_array.tolist()
    sd.start_time = earliest_start_time
    sd.live_time = summed_live_time
    sd.real_time = summed_real_time
    sd.total_counts = int(summed_total_gamma_count)

    # create folder if not exists
    os.makedirs(os.path.dirname(result_file_path_name), exist_ok=True)

    sd.write_to_file(result_file_path_name)
    print(f"Summed {len(loaded_spectra)} -> spectra saved to {result_file_path_name}")

    return sd

def sum_spectra_matching_pattern_in_folder(folder_path: str, pattern: str, result_file_name: str):
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
    SpectrometryData
        Dataclass containing the summed spectra
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

    spectra = _load_spectra_from_files(paths_to_txt)

    return sum_spectra(spectra, result_file_path)

def sum_spectra_in_folder(folder_path: str, file_type: Literal["*.txt", "*.cnf"] = '*.cnf', group_size: int = 4, prefix: str = "sum", skip_first_n: int = 0):
    """
    Groups spectra files in a folder and sums them up in numeric order.

    Args:
        folder_path (str): Path to the folder containing spectra files.
        file_type (Literal["*.txt", "*.cnf"]): Type of spectra files to consider. Default = '*.cnf'.
        group_size (int): Number of spectra to sum in one group. Default = 4.
        prefix (str): Subfolder prefix for result files. Default = "sum".
        skip_first_n (int): Number of files to skip from the beginning. Default = 0.
    """
    # resolve absolute folder path
    folder_path = os.path.abspath(folder_path)


    # find all spectra files
    file_type_str = str(file_type)
    files = glob.glob(os.path.join(folder_path, file_type_str))
    if not files:
        # try with uppercase extension
        files = glob.glob(os.path.join(folder_path, file_type_str.upper()))

    if not files:
        # try find txt files if file_type is not txt
        if file_type_str != "*.txt":
            files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not files:
        print(f"No spectra files found in {folder_path}")
        return

    print(f"Found {len(files)} spectra files in {folder_path}")

    # load all the files
    loaded_spectra = _load_spectra_from_files(files)

    # sort files according to start_time: datetime
    loaded_spectra.sort(key=lambda x: x.start_time)

    # assert that the first spectra in list has min start_time
    assert loaded_spectra[0].start_time == min(sd.start_time for sd in loaded_spectra), "First spectra does not have the earliest start_time, sorting failed"

    # skip first n spectra if requested
    loaded_spectra = loaded_spectra[skip_first_n:]

    # results folder
    result_dir = os.path.join(folder_path, prefix)
    os.makedirs(result_dir, exist_ok=True)

    # loop through files in groups
    for i in range(0, len(loaded_spectra), group_size):
        group = loaded_spectra[i:i + group_size]
        if not group:
            continue

        result_file_name = os.path.join(result_dir, f"summed_{i+1}-{i+group_size}.txt")

        sum_spectra(group, result_file_name)

    print(f"Finished summing spectra in groups of {group_size}. Results saved in {result_dir}")

def read_peakfit_results_csv(filepath: str) -> list[PeakFitResult]:
    """
    Read peak fitting results from a CSV file and return a PeakFitResult object.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing peak fitting results

    Returns
    -------
    list[PeakFitResult]
        List of peak fitting results
    """
    df = pd.read_csv(filepath)

    results:list[PeakFitResult] = []
    try:
        for index, row in df.iterrows():
            # header: area,area_err,centroid,centroid_err,amplitude,amplitude_err,sigma,sigma_err,energy_range,slope,slope_err,intercept,intercept_err,filename,start_time,real_time,live_time

            area = ufloat(row['area'], row['area_err'])
            centroid = ufloat(row['centroid'], row['centroid_err'])
            amplitude = ufloat(row['amplitude'], row['amplitude_err'])
            sigma = ufloat(row['sigma'], row['sigma_err'])
            start_time = datetime.fromisoformat(row['start_time'])
            real_time = float(row['real_time'])
            live_time = float(row['live_time'])

            result = PeakFitResult(
                area=area,
                centroid=centroid,
                start_time=start_time,
                real_time=real_time,
                live_time=live_time,
                amplitude=amplitude,
                sigma=sigma,
                figure=None
            )
            results.append(result)
    except Exception as e:
        print(f"Error reading peakfit results from {filepath}: {e}")

    return results