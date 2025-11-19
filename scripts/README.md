Release script
==============

What this does
---------------

`scripts/release.py` automates the flow described in `how_to_build.md`:

- Bumps the version in `mapp_tricks/__init__.py`, `pyproject.toml`, and `setup.py` (if present)
- Builds the distribution with `python -m build`
- Asks you to verify the version numbers in the modified files and built artifacts
- Prompts you to paste the PyPI API token (hidden input)
- Uploads the artifacts to PyPI with `twine` using username `__token__` and your token as password

Security note: the script does not log or store the token. It is passed directly to `twine`.

Quick usage
-----------

From the project root:

```bash
python scripts/release.py
```

To set the version non-interactively:

```bash
python scripts/release.py --version 0.1.2
```

Prerequisites
-------------

- Python (same interpreter you want to run the build with, activated virtualenv if any)
- `build` package: `pip install build`
- `twine` package: `pip install twine`

If `build` or `twine` are missing the script will print an instruction telling you how to install them.

Notes & next steps
------------------

- The script attempts to update the three common version locations but makes conservative edits; please verify the changes before pushing.
