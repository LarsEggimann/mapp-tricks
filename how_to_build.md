# Build in package directory
Make sure to bump version in `pyproject.toml`, `setup.py` and `mapp_tricks/__init__.py`.

Make sure you have build installed, `pip install build`

Then run in the directory where `pyproject.toml` is:
```
python -m build
```

Builds to `dist/` folder.

Upload to PyPI, have twine installed (`pip install twine`)
```
twine upload dist/*
```
Which will ask for API key.