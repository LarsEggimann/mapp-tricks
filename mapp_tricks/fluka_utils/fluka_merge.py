#!/usr/bin/env python3
# A utility script to merge FLUKA binary output files from multiple runs (processes) and cycles.
# Inspired by fluka2root.py from https://github.com/kbat/mc-tools

import sys
import re
import os
import argparse
from pathlib import Path
import shutil
import subprocess

DEFAULT_ESTIMATORS = [
    ("USRBDX", "usxsuw"),
    ("USRBIN", "usbsuw"),
    ("USRCOLL", "ustsuw"),
    ("USRTRACK", "ustsuw"),
    ("DETECT", "detsuw"),
    ("USRYIELD", "usysuw"),
    ("RESNUCLE", "usrsuw"),
]

class Estimator:
    def __init__(self, name, merger):
        self.name = name      # estimator name as defined by FLUKA cards, e.g. "USRBIN", "USRTRACK", etc.
        self.merger = merger  # name of standart FLUKA merge script used to merge the results
        self.units = {}       # dictionary of units and corresponding files

    def add_unit(self, u):
        """ Adds a key with the given unit in the units dictionary
        """
        self.units[u] = []

    def add_file(self, u, f):
        """ Adds the given file name to the unit
        """
        self.units[u].append(f)

    def __str__(self):
        return self.name+" "+self.merger+" "+str(self.units)


def _resolve_input_file(input_file: str | Path) -> Path | None:
    if isinstance(input_file, str):
        if not input_file.endswith(".inp"):
            input_file += ".inp"
        input_path = Path(input_file).resolve()
    elif isinstance(input_file, Path):
        input_path = input_file.resolve()
    else:
        print(f"Error: Invalid type for input_file: {type(input_file)}. Expected str or Path.")
        return None

    if not input_path.is_file():
        print(f"Error: Input file '{input_path}' not found.")
        return None

    return input_path


def _parse_units_from_input(input_path: Path, verbose: bool = False) -> list[Estimator]:
    estimators = [Estimator(name, merger) for name, merger in DEFAULT_ESTIMATORS]

    with input_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            for estimator in estimators:
                if estimator.name == "DETECT":
                    if re.search(rf"\A{estimator.name}", line):
                        unit = 17
                        if unit not in estimator.units:
                            estimator.add_unit(unit)
                else:
                    if re.search(rf"\A{estimator.name[:8]}", line) and not re.search(r"\&", line[70:80]): # make sure not to use the continuation card wiht the & at the end of the line
                        unit_field = line[20:30] if estimator.name[:8] == "RESNUCLE" else line[30:40]     # use fixed formatting of inp files, note that for RESNUCLE the unit is in 3rd column, for all others its the 4th
                        unit = int(unit_field.strip())
                        if unit < 0:
                            if unit not in estimator.units:
                                estimator.add_unit(unit)
                        elif verbose:
                            print(
                                f"Warning: text output files are not supported for {estimator.name} "
                                f"(unit {unit})."
                            )

    if verbose:
        print("Discovered estimator units:")
        for estimator in estimators:
            if estimator.units:
                print(f"  {estimator.name}: {sorted(estimator.units.keys())}")

    return estimators


def _collect_files_for_units(
    estimators: list[Estimator],
    source_dir: Path,
    verbose: bool = False,
) -> int:
    for estimator in estimators:
        for unit in estimator.units:
            fort_unit = abs(unit)
            files = sorted(source_dir.glob(f"*_fort.{fort_unit}")) # look for files with pattern *_fort.<unit> in the source directory

            for file_path in files:
                estimator.add_file(unit, file_path) # add the file to the corresponding unit of the estimator

            if verbose:
                print(
                    f"{estimator.name} unit {unit}: found {len(estimator.units[unit])} files "
                    f"in {source_dir}"
                )

    merged_input_count = sum(len(files) for est in estimators for files in est.units.values())
    if merged_input_count == 0:
        print(f"Error: No FLUKA binary files found in '{source_dir}'.")
        return 1

    return 0


def _resolve_merger_command(merger: str) -> str | None:
    local = shutil.which(merger)
    if local:
        return local

    flupro = os.environ.get("FLUPRO")
    if not flupro:
        return None

    candidate = Path(flupro) / "flutil" / merger
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)

    return None


def _merge_estimator_unit(
    estimator: Estimator,
    unit: int,
    source_files: list[Path],
    output_file: Path,
    verbose: bool = False,
) -> int:
    merger_cmd = _resolve_merger_command(estimator.merger)
    if merger_cmd is None:
        print(
            f"Error: Could not find FLUKA merge utility '{estimator.merger}' in PATH "
            "or $FLUPRO/flutil."
        )
        return 1

    payload_lines = [str(path.name) for path in source_files]
    payload_lines.append("")
    payload_lines.append(str(output_file.name))
    payload = "\n".join(payload_lines) + "\n"

    if verbose:
        print(
            f"Merging {len(source_files)} file(s) for {estimator.name} unit {unit} -> {output_file.name}"
        )

    try:
        result = subprocess.run(
            [merger_cmd],
            input=payload,
            cwd=output_file.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        print(f"Error while executing {estimator.merger}: {error}")
        return 1

    if result.returncode != 0:
        print(
            f"Error: FLUKA merge command '{estimator.merger}' failed for {estimator.name} unit {unit}."
        )
        print(result.stdout)
        return 1

    return 0


def fluka_merge(
    input_file: Path | str,
    source_dir: Path | str | None = None,
    name: str | None = None,
    verbose: bool = False,
) -> int:
    """Merge FLUKA binary estimator outputs across threads/cycles.

    Args:
        input_file (Path | str): Path to a FLUKA input file (.inp) used to detect estimator cards.
        source_dir (Path | str | None, optional): Directory containing FLUKA output files.
            Defaults to the input file directory.
        name (str | None, optional): Prefix of merged output files. Defaults to input stem.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
    """

    input_path = _resolve_input_file(input_file)
    if input_path is None:
        return 1

    if source_dir is None:
        source_path = input_path.parent
    else:
        source_path = Path(source_dir).resolve()

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' not found.")
        return 1

    output_prefix = name if name else re.sub(r"_thread\d+_$", "", input_path.stem)

    if verbose:
        print(f"Input file:   {input_path}")
        print(f"Source dir:   {source_path}")
        print(f"Output name:  {output_prefix}")

    estimators = _parse_units_from_input(input_path, verbose=verbose)

    if _collect_files_for_units(estimators=estimators, source_dir=source_path, verbose=verbose) != 0:
        return 1

    merge_jobs: dict[tuple[str, int], tuple[Estimator, list[Path]]] = {}
    for estimator in estimators:
        for unit, files in estimator.units.items():
            if files:
                merge_jobs[(estimator.name, unit)] = (estimator, files)

    if not merge_jobs:
        print("Error: No mergeable FLUKA binary outputs found.")
        return 1

    for (estimator_name, unit), (estimator, files) in merge_jobs.items():
        output_file = source_path / f"{output_prefix}_{abs(unit)}_{estimator_name.lower()}"
        if _merge_estimator_unit(estimator, unit, files, output_file, verbose=verbose) != 0:
            return 1

    print(f"Merged FLUKA outputs created in '{source_path}' with prefix '{output_prefix}'.")
    return 0


def main() -> int:
    """Merge FLUKA binary output files from multiple runs/threads/cycles.
    """
    argparser = argparse.ArgumentParser(
        description=main.__doc__,
        epilog="Created by Lars Eggimann (2026)",
        )
    argparser.add_argument("input_file", type=str, help="Path to a FLUKA .inp file used to detect estimator cards.")
    argparser.add_argument("-s", "--source-dir", type=str, default=None, help="Directory containing FLUKA output files (default: same directory as input_file).")
    argparser.add_argument("-n", "--name", type=str, default=None, help="Prefix for merged output files (default: input stem).")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output for debugging.")

    args = argparser.parse_args()

    return fluka_merge(
        input_file=args.input_file,
        source_dir=args.source_dir,
        name=args.name,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
