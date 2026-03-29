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
from dataclasses import dataclass, field

DEFAULT_ESTIMATORS = [
    ("USRBDX", "usxsuw"),
    ("USRBIN", "usbsuw"),
    ("USRCOLL", "ustsuw"),
    ("USRTRACK", "ustsuw"),
    ("DETECT", "detsuw"),
    ("USRYIELD", "usysuw"),
    ("RESNUCLE", "usrsuw"),
]

@dataclass
class MergedFileInfo:
    """Metadata for one merged FLUKA output file.

    Attributes:
        estimator_name: FLUKA estimator card name (for example USRBIN, USRTRACK).
        unit: FLUKA unit number associated with the merged output.
        merger: Merge utility name expected by FLUKA (for example usbsuw).
        merger_command: Resolved executable path used for the merge utility.
        output_file: Path to the merged output file created by FLUKA tools.
        input_files: Input binary files merged into output_file.
    """

    estimator_name: str
    unit: int
    merger: str
    merger_command: str
    output_file: Path
    input_files: list[Path]

    def to_dict(self) -> dict[str, str | int | list[str]]:
        return {
            "estimator_name": self.estimator_name,
            "unit": self.unit,
            "merger": self.merger,
            "merger_command": self.merger_command,
            "output_file": str(self.output_file),
            "input_files": [str(path) for path in self.input_files],
        }

@dataclass
class MergeResult:
    """Result object returned by FLUKA merging operations.

    Attributes:
        input_file: Resolved input .inp file used to detect estimator units 
        source_dir: Resolved directory searched for FLUKA binary output files.
        output_prefix: Prefix used for generated merged output files.
        merged_files: Metadata entries for each successfully merged output file.
        error: Error message when initialization or merge failed; None on success.
    """

    input_file: Path | None = None
    source_dir: Path | None = None
    output_prefix: str | None = None
    merged_files: list[MergedFileInfo] = field(default_factory=list)
    error: str | None = None

class Estimator:
    """Represents one FLUKA estimator definition and its discovered unit files.

    Attributes:
        name: Estimator card name as it appears in FLUKA input.
        merger: FLUKA merge utility name for this estimator type.
        units: Mapping from unit number (i.e. -21 -> *_fort.21) to collected files for that unit.
    """

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

def _resolve_input_file(input_file: Path | str) -> Path | None:
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

    return sum(len(files) for est in estimators for files in est.units.values())

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

class FlukaMerger:
    """Initializes and executes merging of FLUKA estimator outputs based on a given input file and source directory.

    Attributes:
        input_path: Resolved FLUKA input file path used to detect estimator definitions and units.
        source_path: Resolved directory containing FLUKA binary output files from multiple runs/threads/cycles to be merged.
        output_prefix: Prefix applied to generated merged output files.
        estimators: Parsed estimator definitions with discovered units/files from the input file and source directory.
        merge_jobs: Merge plan keyed by (estimator_name, unit).
        initialization_error: Error set during initialization, if any.
        verbose: Enables verbose logging for discovery and merge steps.
    """

    def __init__(
        self,
        input_file: Path | str,
        source_dir: Path | str | None = None,
        name: str | None = None,
        verbose: bool = False,
    ):

        self.input_path: Path | None = None
        self.source_path: Path | None = None  
        self.output_prefix: str | None = None
        self.estimators: list[Estimator] = []
        self.merge_jobs: dict[tuple[str, int], tuple[Estimator, list[Path]]] = {}
        self.initialization_error: str | None = None
        self.verbose = verbose

        self.input_path = _resolve_input_file(input_file)
        if self.input_path is None:
            self.initialization_error = "Failed to resolve input file."
            return

        if source_dir is None:
            self.source_path = self.input_path.parent
        else:
            self.source_path = Path(source_dir).resolve()

        if not self.source_path.is_dir():
            self.initialization_error = f"Source directory '{self.source_path}' not found."
            return

        self.output_prefix = name if name else self.input_path.stem

        if self.verbose:
            print(f"Input file:   {self.input_path}")
            print(f"Source dir:   {self.source_path}")
            print(f"Output name:  {self.output_prefix}")

        self.estimators = _parse_units_from_input(self.input_path, verbose=self.verbose)

        merged_input_count = _collect_files_for_units(
            estimators=self.estimators,
            source_dir=self.source_path,
            verbose=self.verbose,
        )
        if merged_input_count == 0:
            self.initialization_error = f"No FLUKA binary files found in '{self.source_path}'."
            return

        for estimator in self.estimators:
            for unit, files in estimator.units.items():
                if files:
                    self.merge_jobs[(estimator.name, unit)] = (estimator, files)

        if not self.merge_jobs:
            self.initialization_error = "No mergeable FLUKA binary outputs found."

    def _merge_estimator_unit(
        self,
        estimator: Estimator,
        unit: int,
        source_files: list[Path],
        output_file: Path,
    ) -> str | None:
        
        merger_cmd = _resolve_merger_command(estimator.merger)
        if merger_cmd is None:
            print(
                f"Error: Could not find FLUKA merge utility '{estimator.merger}' in PATH "
                "or $FLUPRO/flutil."
            )
            return None

        payload_lines = [str(path.name) for path in source_files]
        payload_lines.append("")
        payload_lines.append(str(output_file.name))
        payload = "\n".join(payload_lines) + "\n"

        if self.verbose:
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
            print(f"Error while executing {estimator.merger} using command {merger_cmd}: {error}")
            return None

        if result.returncode != 0:
            print(
                f"Error: FLUKA merge command '{estimator.merger}' failed for {estimator.name} unit {unit}."
            )
            print(result.stdout)
            return None

        return merger_cmd        

    def merge(self) -> MergeResult:
        if self.initialization_error is not None:
            return MergeResult(
                input_file=self.input_path,
                source_dir=self.source_path,
                output_prefix=self.output_prefix,
                error=self.initialization_error,
            )

        if self.source_path is None or self.output_prefix is None:
            return MergeResult(
                input_file=self.input_path,
                source_dir=self.source_path,
                output_prefix=self.output_prefix,
                error="Merger initialization is incomplete.",
            )

        merged_files: list[MergedFileInfo] = []
        for (estimator_name, unit), (estimator, files) in self.merge_jobs.items():
            output_file = self.source_path / f"{self.output_prefix}_{abs(unit)}_{estimator_name.lower()}"
            merger_cmd = self._merge_estimator_unit(estimator, unit, files, output_file)
            if merger_cmd is None:
                return MergeResult(
                    input_file=self.input_path,
                    source_dir=self.source_path,
                    output_prefix=self.output_prefix,
                    merged_files=merged_files,
                    error=f"Failed while merging {estimator.name} unit {unit}.",
                )

            merged_files.append(
                MergedFileInfo(
                    estimator_name=estimator.name,
                    unit=unit,
                    merger=estimator.merger,
                    merger_command=merger_cmd if merger_cmd else "",
                    output_file=output_file,
                    input_files=files,
                )
            )

        print(f"Merged FLUKA outputs created in '{self.source_path}' with prefix '{self.output_prefix}'.")
        return MergeResult(
            input_file=self.input_path,
            source_dir=self.source_path,
            output_prefix=self.output_prefix,
            merged_files=merged_files,
        )


def fluka_merge(
    input_file: Path | str,
    source_dir: Path | str | None = None,
    name: str | None = None,
    verbose: bool = False,
) -> MergeResult:
    """Merge FLUKA binary estimator outputs across threads/cycles.

    Args:
        input_file (Path | str): Path to a FLUKA input file (.inp) used to detect estimator cards.
        source_dir (Path | str | None, optional): Directory containing FLUKA output files.
            Defaults to the input file directory.
        name (str | None, optional): Prefix of merged output files. Defaults to input stem.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
    """

    merger = FlukaMerger(
        input_file=input_file,
        source_dir=source_dir,
        name=name,
        verbose=verbose,
    )
    return merger.merge()


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

    result = fluka_merge(
        input_file=args.input_file,
        source_dir=args.source_dir,
        name=args.name,
        verbose=args.verbose,
    )
    if result.error:
        print(f"Error: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
