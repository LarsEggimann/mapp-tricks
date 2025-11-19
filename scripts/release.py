#!/usr/bin/env python3
"""Automate version bump, build and upload to PyPI.

Usage examples:
  python scripts/release.py            # interactive: choose version or keep current, then build and upload
  python scripts/release.py --version 0.1.2  # set version non-interactively
  python scripts/release.py --version 0.1.2 --commit --tag  # also commit & tag the release

This script will:
 - detect the current version from mapp_tricks/__init__.py
 - update version strings in pyproject.toml, setup.py and mapp_tricks/__init__.py
 - run `python -m build` to create distributions under dist/
 - prompt you to paste the PyPI API token (your input will be hidden)
 - upload artifacts with twine using username `__token__` and the pasted token as password

It will NOT automatically install dependencies; if `build` or `twine` are missing the script will print helpful hints.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_INIT = ROOT / "mapp_tricks" / "__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"


def read_current_version() -> str | None:
    if not PKG_INIT.exists():
        return None
    txt = PKG_INIT.read_text()
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", txt)
    return m.group(1) if m else None


def replace_version_in_init(new_version: str) -> None:
    txt = PKG_INIT.read_text()
    # Use a callable replacement to avoid accidental backreference parsing
    # when the version contains digit sequences like '10' (e.g. '\10').
    def repl_init(m: re.Match) -> str:
        return m.group(1) + new_version + m.group(3)

    txt2 = re.sub(r"(__version__\s*=\s*['\"])([^'\"]+)(['\"])", repl_init, txt)
    PKG_INIT.write_text(txt2)


def replace_version_in_pyproject(new_version: str) -> None:
    if not PYPROJECT.exists():
        return
    txt = PYPROJECT.read_text()
    # replace first occurrence of version = "x.y.z"
    def repl_pyproject(m: re.Match) -> str:
        return m.group(1) + new_version + m.group(2)

    txt2, n = re.subn(r'(version\s*=\s*")[^"]+(\")', repl_pyproject, txt, count=1)
    if n:
        PYPROJECT.write_text(txt2)


def replace_version_in_setup(new_version: str) -> None:
    if not SETUP_PY.exists():
        return
    txt = SETUP_PY.read_text()
    def repl_setup(m: re.Match) -> str:
        return m.group(1) + new_version + m.group(3)

    txt2, n = re.subn(r"(version\s*=\s*['\"])([^'\"]+)(['\"])", repl_setup, txt, count=1)
    if n:
        SETUP_PY.write_text(txt2)


def run_build() -> None:
    print("Running build: python -m build")
    try:
        subprocess.run([sys.executable, "-m", "build"], check=True)
    except FileNotFoundError:
        print("`python -m build` not found. Install with: pip install build")
        raise
    except subprocess.CalledProcessError as e:
        print("Build failed")
        raise


def run_twine_upload(token: str) -> None:
    dist_paths = sorted([str(p) for p in (ROOT / "dist").glob("*")])
    if not dist_paths:
        raise SystemExit("No files found in dist/. Did the build step succeed?")

    cmd = ["twine", "upload"] + dist_paths + ["-u", "__token__", "-p", token]
    print("Uploading to PyPI with twine...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("`twine` not found. Install with: pip install twine")
        raise
    except subprocess.CalledProcessError:
        print("twine upload failed")
        raise

def delete_contents_of_dist() -> None:
    dist_dir = ROOT / "dist"
    if dist_dir.exists() and dist_dir.is_dir():
        for file_path in dist_dir.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and upload package to PyPI (prompts for token)")
    parser.add_argument("--version", help="Version to set (e.g. 0.1.2). If omitted, version is prompted.")
    parser.add_argument("--clean", action="store_true", help="Clean the dist/ directory before building")
    args = parser.parse_args()

    cur = read_current_version()
    if not cur:
        print("Could not detect current version in mapp_tricks/__init__.py")
    else:
        print(f"Current version: {cur}")

    new_version = args.version
    if not new_version:
        new_version = input(f"Enter new version (current '{cur}'): ").strip()

    if not new_version:
        print("No version specified, aborting.")
        raise SystemExit(1)

    print(f"Setting version to: {new_version}")
    replace_version_in_init(new_version)
    replace_version_in_pyproject(new_version)
    replace_version_in_setup(new_version)

    if args.clean:
        print("Cleaning dist/ directory...")
        delete_contents_of_dist()

    # Build
    run_build()

    # verify versions and build artifacts, manual -> prompt user to check
    print("\nPlease verify that the version numbers in the following files are correct:")
    print(f" - {PKG_INIT}")
    if PYPROJECT.exists():
        print(f" - {PYPROJECT}")
    if SETUP_PY.exists():
        print(f" - {SETUP_PY}")
    if not input("Press Enter to continue, or Ctrl+C to abort..."):
        pass

    # verify built artifacts
    print("\nPlease verify that the built artifacts in dist/ have the correct version and contain all necessary files:")
    dist_files = glob.glob(os.path.join(ROOT, "dist", "*"))
    for f in dist_files:
        print(f" - {f}")
    if not input("Press Enter to continue, or Ctrl+C to abort..."):
        pass

    # Ask for token and upload
    token = getpass.getpass("Paste PyPI API token (input hidden): ")
    if not token:
        print("No token provided - aborting upload.")
        raise SystemExit(1)

    run_twine_upload(token)

    print("Release process finished.")


if __name__ == "__main__":
    main()
