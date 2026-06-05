#!/usr/bin/env python3
"""
Read FLUKA tab.lis text outputs and parse.
"""

from __future__ import annotations

import pandas as pd # type: ignore
from uncertainties import ufloat, unumpy # type: ignore
import fileinput

from dataclasses import dataclass, field
from pathlib import Path

from pyparsing import line

DETECTOR  = "# Detector n:"
INTERVALS = "# N. of energy intervals"


@dataclass
class DetectorDistribution:
	"""
	Container for a single detector spectrum.
	
	Attributes:
		detector_id (int): ID of the detector as given in the FLUKA output
		name (str): name of the detector as given in the FLUKA output, or generated if not provided
		description (str | None): optional description of the detector as given in the FLUKA output
		integrated_data (pd.DataFrame): DataFrame containing the integrated fluence data per primary particle with columns: energy_low [MeV], energy_high [MeV], energy [MeV], phi [1/cm^2], dphi [1/cm^2/MeV]
	"""
	detector_id: int
	name: str
	description: str | None = None
	integrated_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["energy_low", "energy_high", "energy", "phi", "dphi"]))

def _parse_detector_header(line: str) -> tuple[int, str, str | None] | None:
	if not line.strip().startswith(DETECTOR):
		return None
	parts = line.strip().split()
	try:
		detector_id = int(parts[3])  # "Detector n:" -> n is the 4th part (index 3)
		name = parts[4] if len(parts) > 4 else f"Detector_{detector_id}"
		description = " ".join(parts[5:]) if len(parts) > 5 else None
	except (ValueError, IndexError):
		return None
	return detector_id, name, description


def _parse_intervals_header(line: str) -> int | None:
	if not line.strip().startswith(INTERVALS):
		return None
	parts = line.strip().split()
	try:
		return int(parts[5])  # "N. of energy intervals" -> the number is the 6th part (index 5)
	except (ValueError, IndexError):
		return None

def _parse_data_line_where_integrated_over_solid_angle(line: str) -> tuple[float, float, float, float] | None:
	parts = line.strip().split()
	if len(parts) != 4: # integrated over solid angle has 4 columns: energy_low, energy_high, value, error_percent
		return None
	try:
		return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
	except ValueError:
		return None


def read_usrbdx_tab_lis(file_path: Path | str, verbose: bool = False) -> dict[str, DetectorDistribution]:
	"""
	Read a FLUKA USRBDX tab.lis file and return distributions per detector.
	Args:
		file_path (Path | str): path to the tab.lis file
		verbose (bool): whether to print verbose output
	Returns:
		dict[str, DetectorDistribution]: dictionary mapping detector names to their distributions

	
	"""

	path = Path(file_path).resolve()
	if not path.is_file():
		raise FileNotFoundError(f"FLUKA tab.lis file not found: {path}")

	detectors: dict[str, DetectorDistribution] = {}
	current: DetectorDistribution | None = None
	expected_intervals: int | None = None


	# find all detectors in the file and parse their data as pd.DataFrame
	with fileinput.input([path]) as file:
		for line in file:
			if not line.strip():
				continue

			header = _parse_detector_header(line)
			if header:
				detector_id, name, description = header
				current = DetectorDistribution(
					detector_id=detector_id,
					name=name,
					description=description,
				)

				start_idx = file.lineno() + 2 # start of data lines is 2 lines after the header

				intervals_line = next(file, "") # next line should be the intervals header
				expected_intervals = _parse_intervals_header(intervals_line)
				if verbose:
					print(f"Found detector \"{current.name}\" with ID \"{current.detector_id}\" expecting \"{expected_intervals}\" bins.")
				
				# add the found detector to the list
				detectors[current.name] = current

				# find out start idx and end idx of the data lines
				data = pd.read_csv(path, skiprows=start_idx-1, nrows=expected_intervals, names=["energy_low", "energy_high", "dphi", "dhpi_percentage_error"], sep=r"\s+")

				# # print first and last 5 lines of the data for sanity check
				# print(f"Data for detector {current.name} (ID {current.detector_id}):")
				# print(data.head())
				# print(data.tail())

				dphi = data.dphi * 1e-3 # convert from 1/cm^2/GeV to 1/cm^2/MeV
				dphi_error = data.dhpi_percentage_error * data.dphi * 1e-3 / 100 # convert percentage error to absolute error in 1/cm^2/MeV
				dphi_ufloat = unumpy.uarray(dphi, dphi_error)

				dE = (data.energy_high - data.energy_low) * 1e3 # convert from GeV to MeV

				phi = dphi_ufloat * dE # convert from differential flux to flux in the energy interval

				# create new df
				df = pd.DataFrame({
					"energy_low": data.energy_low * 1e3, # convert GeV to MeV
					"energy_high": data.energy_high * 1e3, # convert GeV to MeV
					"energy": ((data.energy_low + data.energy_high) / 2) * 1e3, # convert GeV to MeV
					"phi": phi,
					"dphi": dphi_ufloat,
				})

				current.integrated_data = df


				continue


			# TODO: also parse the double differential distrib data

	return detectors


__all__ = ["DetectorDistribution", "read_usrbdx_tab_lis"]
