#!/usr/bin/env python3
# A utility script to run FLUKA with multiprocessing support by launching multiple instances of FLUKA with different random seeds.

import sys
import re
import os
import argparse
from pathlib import Path
import shutil
import subprocess

FLUKA_BIN = 'rfluka' # $FLUPRO/flutil/rfluka
FLUKA_EXE = ''        # currently not supported in my script, use default EXE

DEFAULT_RANDOM_SEED = 54217137

def _replace_nprimaries(inp_path: Path, nprim: int, verbose: bool = False) -> int:
    text = inp_path.read_text()

    # match a FLUKA START line and keep "START" in first 10 chars, value right-aligned in next 10
    pattern = re.compile(r"(?m)^(\s*START)\s+\S+.*$")
    replacement = f"{'START':<10}{float(nprim):>10}" # make sure to parse as float to add .0

    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        print("Error: No START with value line found in input file.")
        return 1

    if verbose:
        print(f"In file {inp_path.stem}, set number of primary particles to: {nprim}")

    inp_path.write_text(new_text)
    return 0

def _replace_random_seed(inp_path: Path, seed: int, verbose: bool = False) -> int:
    text = inp_path.read_text()

    # match a FLUKA RANDOMIZ line and keep "RANDOMIZ" in first 10 chars, value right-aligned in next 10
    pattern = re.compile(r"(?m)^(\s*RANDOMIZ)\s+\S+.*$")
    replacement = f"{'RANDOMIZ':<10}{'1.0':>10}{seed:>10}" # here we just assume that we use the default output unit for the random seed reading, I dont know why I should ever change this?

    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        print("Error: No RANDOMIZ with value line found in input file.")
        return 1

    if verbose:
        print(f"In file {inp_path.stem}, set random seed to: {seed}")

    inp_path.write_text(new_text)
    return 0

def _get_seed_from_input_file(inp_path: Path, verbose: bool = False) -> int:
    for line in inp_path.read_text().splitlines()[::-1]: # read lines in reverse order, as RANDOMIZ is usually towards the end of the file
        if line.strip().startswith("RANDOMIZ"):
            parts = line.split()
            if len(parts) > 2:
                seed = parts[2] # should be in third column, after "RANDOMIZ" and some whitespace
                if seed.isdigit():
                    if verbose:
                        print(f"Found random seed in input file {inp_path.stem}: {seed}")
                    return int(seed)
                break # if we find a RANDOMIZ line but it does not have a valid seed, we break and use default seed
    if verbose:
        print(f"No RANDOMIZ seed found in input file {inp_path.stem}, using default random seed {DEFAULT_RANDOM_SEED}")
    return DEFAULT_RANDOM_SEED    


# some seed magic as suggested by https://prng.di.unimi.it/splitmix64.c
MASK64 = 0xFFFFFFFFFFFFFFFF
PHI64 = 0x9E3779B97F4A7C15
def _splitmix64(x: int) -> int:
    z = (x + PHI64) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64

# more seed magic, I do not pretend to understand any of it, but it should give some good random seeds for our different threads
def _derive_seed(base_seed: int, stream_id: int, max_seed: int = int(9e8)) -> int:
    mixed = _splitmix64(base_seed ^ (stream_id * PHI64))
    return 1 + (mixed % max_seed)  # keep positive and bounded

def fluka_run(
    input_file: Path | str,
    run_name: str,
    number_primary_particles: int | None = None,
    number_cycles: int = 5,
    threads: int = 1,
    verbose: bool = False,
    ) -> int:
    """Run FLUKA with the specified input file and parameters.
    Args:
        input_file (Path | str): Path to the FLUKA input file (.inp) to run.
        run_name (str): Name for the run. This will be used as the folder name where the input file will be copied and FLUKA run will be executed.
        number_primary_particles (int | None, optional): Number of primary particles to simulate. If specified, this will modify the input file accordingly. Defaults to None.
        number_cycles (int, optional): Number of cycles per thread. This will be used as start argument for FLUKA. Defaults to 5.
        threads (int, optional): Number of threads for multiprocessing. Defaults to 1.
        verbose (bool, optional): Enable verbose output for debugging. Defaults to False.
    """

    ### check input ###

    # prep input file path and check if it exists
    if isinstance(input_file, str):
        if not input_file.endswith(".inp"):
            input_file += ".inp"
        input_file_path = Path(input_file).resolve()
    elif isinstance(input_file, Path):
        input_file_path = input_file.resolve()
    else:
        print(f"Error: Invalid type for input_file: {type(input_file)}. Expected str or Path.")
        return 1

    if not input_file_path.is_file():
        print(f"Error: Input file '{input_file_path}' not found.")
        return 1

    run_name = run_name if run_name else input_file_path.stem

    # make sure threads is a positive integer, and smaller than the number of CPU cores
    if threads < 1:
        print(f"Error: Number of threads must be a positive integer, got {threads}.")
        return 1
    
    if threads > 1:
        max_threads = os.cpu_count()
        if max_threads is None:
            max_threads = 1 # fallback to 1 if os.cpu_count() fails for some reason
        if threads > max_threads:
            print(f"Error: Number of threads ({threads}) exceeds the number of CPU cores ({max_threads}). Using {max_threads} threads instead.")
            threads = max_threads

    # verbose output
    if verbose:
        print(f"Input file:                  {input_file}")
        print(f"Run name:                    {run_name}")
        print(f"Threads:                     {threads}")
        print(f"Number of primary particles: {number_primary_particles}")
        print(f"Number of cycles per thread: {number_cycles}")



    ### setup ###

    # get the directory of the input file and create a new directory for the run
    input_dir = input_file_path.parent
    run_dir = input_dir / run_name
    run_dir.mkdir(exist_ok=True)

    # copy the input file to the run directory
    run_input_file = run_dir / input_file_path.name
    shutil.copy(input_file_path, run_input_file) # this overwrites if already exists

    # replace the number of primary particles in the input file if specified
    if number_primary_particles is not None:
        if _replace_nprimaries(run_input_file, number_primary_particles, verbose=verbose) != 0:
            print(f"Error: Failed to update the number of primary particles in the input file '{run_input_file.stem}'.")
            return 1
        
    # get seed from inputfile or use default
    seed = _get_seed_from_input_file(run_input_file, verbose=verbose)

    # multiprocessing in FLUKA just launches multiple instances of FLUKA with the same input file EXCEPT that the random seed is different for each instance
    # create as many copies of the input file as threads, with different random seeds, if threads > 1
    inp_files:list[Path] = []
    for i in range(threads):
        thread_input_file = run_dir / f"{run_input_file.stem}_thread{i}_.inp"
        shutil.copy(run_input_file, thread_input_file)

        if i > 0: # for the first thread, we can just use the original input file with the modified nprim
            if _replace_random_seed(thread_input_file, _derive_seed(seed, i), verbose=verbose) != 0:
                print(f"Error: Failed to update the random seed for thread {i} in the input file '{thread_input_file.stem}'.")
                return 1
        inp_files.append(thread_input_file)

    # remove run_input_file
    run_input_file.unlink()

    # calculate the total number of primary particles to simulate across all threads, and the number of cycles per thread
    if number_primary_particles is not None:
        total_nprim = number_primary_particles * threads * number_cycles
        if verbose:
            print(f"Total number of primary particles across all threads and cycles: {total_nprim}")



    ### launch FLUKA ###

    # for each thread, launch FLUKA with the corresponding input file
    cmd_base = [FLUKA_BIN]
    if FLUKA_EXE:
        cmd_base += [f"-e{FLUKA_EXE}"]

    # to not pollute the console to much create a logs directory
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    processes: list[tuple[int, subprocess.Popen[bytes], Path]] = []

    for i, inp_file in enumerate(inp_files):
        cmd = [*cmd_base, "-N0", f"-M{number_cycles}", inp_file.stem]
        log_file = logs_dir / f"{inp_file.stem}.log"
        try:
            with open(log_file, "w", encoding='utf-8') as log_f:
                p = subprocess.Popen(
                    cmd,
                    cwd=run_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
        except OSError as e:
            print(f"Error launching thread {i}: {e}")
            return 1

        processes.append((i, p, log_file))
        print(f"Launched thread {i} (pid={p.pid}); stdout|stderr -> {log_file.name}")

    # block until all launched FLUKA processes are finished.
    exit_status = 0
    try:
        for i, process, log_file in processes:
            return_code = process.wait()
            if return_code != 0:
                print(
                    f"Thread {i} failed with exit code {return_code}; check log '{log_file.name}'."
                )
                exit_status = 1
            elif verbose:
                print(f"Thread {i} finished successfully.")
    except KeyboardInterrupt:
        print(" -> Interrupted while waiting for FLUKA to finish. Terminating running threads ...")
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        return 130

    return exit_status


def main() -> int:
    """Launch FLUKA with the specified input file and parameters from command line arguments. The script will create a new directory for the run, copy the input file there, modify it if necessary, and launch multiple instances of FLUKA with different random seeds for multiprocessing support.
    """
    argparser = argparse.ArgumentParser(
        description=main.__doc__,
        epilog="Created by Lars Eggimann (2026)",
        )
    argparser.add_argument("input_file", type=str, help="Path to the FLUKA .inp file to run.")
    argparser.add_argument("-n", "--name", type=str, default=None, help="Name for the run (default: same as input file name without extension). This will be used as the folder name where the input file will be copied and FLUKA run will be executed.")
    argparser.add_argument("-np", "--number-primary-particles", type=int, default=None, help="Number of primary particles to simulate (default: None). This will be used to modify the input file if specified.")
    argparser.add_argument("-nc", "--number-cycles", type=int, default=5, help="Number of cycles per thread (default: 5). This will be used as start argument for FLUKA.")
    argparser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads for multiprocessing (default: 1)")
    argparser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output for debugging.")

    args = argparser.parse_args()

    return fluka_run(
        input_file=args.input_file,
        run_name=args.name,
        number_primary_particles=args.number_primary_particles,
        number_cycles=args.number_cycles,
        threads=args.threads,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
