from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from uncertainties import ufloat, Variable  # type: ignore

from .utils import functions_geometrical_factor as geom  # type: ignore
from .utils import functions_normalized_efficiency_detector as norms  # type: ignore

@dataclass
class _FitParams:
    a: float
    b: float
    cov: np.ndarray  # shape (2,2)

    @property
    def a_err(self) -> float:
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def b_err(self) -> float:
        return float(np.sqrt(self.cov[1, 1]))

    @property
    def a_ufloat(self) -> ufloat:
        return ufloat(self.a, self.a_err)

    @property
    def b_ufloat(self) -> ufloat:
        return ufloat(self.b, self.b_err)



class XRayCalibration:
    """
    Class-based X-ray spectrometer calibration using precomputed linear model
    parameters a, b against a normalized efficiency curve f(E).

    Contract:
    - Inputs (init):
        level: int in [0..14]
        source_radius: mm (float), default is 0 -> faster execution time
        setup_name: default -> 'Gain3-398_PeakingTime1us' 
    - Public methods:
        evaluate_efficiency_at(energy_in_keV: float | ufloat) -> ufloat
        plot(store_path: str | Path | None = None) -> Figure
    - Error handling:
        Raises FileNotFoundError for missing calibration assets.
        Raises ValueError for out-of-range energies when evaluating.
    """

    # ---------- initialization ----------
    def __init__(self, level: int, source_radius: float = 0, setup_name: str = 'Gain3-398_PeakingTime1us'):
        self.level = int(level)
        if not (0 <= self.level <= 14):
            raise ValueError("level must be an integer in [0..14]")
        self.source_radius = float(source_radius)
        self.setup_name = str(setup_name)

        # Base paths (relative to this file)
        self._file_dir = Path(__file__).resolve().parent
        self._xray_base = self._file_dir
        self._results_dir   = self._xray_base / "calibration-data" / "result files calibration" / self.setup_name / self._level_str()
        self._eff_files_dir = self._xray_base / "calibration-data" / "SDD Efficiency files"

        # Load calibration result (a, b, cov) and data points used for fit
        txt_path = self._latest_result_file(self._results_dir)
        (
            self._eff_curve_filename,
            self._fit_params,
            self._data_peaks,
        ) = self._parse_result_file(txt_path)

        # Load normalized efficiency curve used for the fit
        self._eff_curve = self._load_efficiency_curve(
            self._eff_files_dir / self._eff_curve_filename, level=self.level
        )  # shape (N, 2) -> columns: energy[keV], f(E)
        self._interp = interp1d(
            self._eff_curve[:, 0],
            self._eff_curve[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )



    # ---------- public API ----------
    def evaluate_efficiency_at(self, energy_in_keV: Union[float, ufloat]) -> ufloat:
        """
        Evaluate efficiency at a given energy in keV and return as ufloat (value ± fit error),
        including multiplicative corrections for finite source radius and geometry.
        """
        E = energy_in_keV.n if isinstance(energy_in_keV, Variable) else float(energy_in_keV)

        # Compute base efficiency from linear model: eta(E) = a + b * f(E)
        fE = float(self._interp(E))
        a, b, cov = self._fit_params.a, self._fit_params.b, self._fit_params.cov
        eta = a + b * fE
        # Propagate fit covariance: sigma^2 = [1, f(E)] cov [1, f(E)]^T
        T = np.array([1.0, fE])
        sigma = float(np.sqrt(T @ cov @ T))
        eta_u = ufloat(eta, sigma)

        # Apply corrections
        eta_u *= self._add_correction_geometrical_factor()
        eta_u *= self._add_correction_shape_efficiency_curve(E)

        return eta_u

    def get_plot(self, store_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot calibration data points, fitted curve and 1-sigma band.
        If store_path is provided, store the figure there. If a directory is given,
        a descriptive filename is generated. Returns the Matplotlib Figure.

        Note: the generated plot represents the data for a point source, if your radius is no-negligible the efficiency obtained form `evaluate_efficiency_at()` will not be on the curve.
        """
        x_data = self._data_peaks[:, 0].astype(float)
        y_data = self._data_peaks[:, 1].astype(float)
        y_err = self._data_peaks[:, 2].astype(float)

        x_min = 1
        x_max = float(np.nanmax(x_data) + 20.0) if x_data.size else float(np.nanmax(self._eff_curve[:, 0]))
        x_fit = np.linspace(x_min, x_max, 400)
        fE_fit = self._interp(x_fit)

        a, b, cov = self._fit_params.a, self._fit_params.b, self._fit_params.cov
        y_fit = a + b * fE_fit
        # Fit uncertainty band
        T = np.vstack([np.ones_like(x_fit), fE_fit]).T  # shape (N,2)
        # sigma_i^2 = T_i * cov * T_i^T
        sigmas = np.sqrt(np.einsum("ij,jk,ik->i", T, cov, T))

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        if x_data.size:
            ax.errorbar(x_data, y_data, yerr=y_err, fmt="o", color="tab:blue", ecolor="tab:blue", markersize=3, capsize=2, label="Calibration data")
        ax.plot(x_fit, y_fit, color="indianred", label=r"Fit: $\eta(E) = a + b\,f(E)$")
        ax.fill_between(x_fit, y_fit - sigmas, y_fit + sigmas, color="lightcoral", alpha=0.35, label="1σ fit error")

        ax.set_yscale("log")
        ax.set_xlabel("Energy E [keV]")
        ax.set_ylabel(r"Efficiency $\eta$")
        ax.set_title(f"Efficiency curve on {self._level_str()} ({self.setup_name})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        if store_path is not None:
            out_path = self._resolve_store_path(store_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")

        return fig

    # ---------- non-public helpers ----------
    def _level_str(self) -> str:
        return f"level {self.level:02d}"

    def _latest_result_file(self, folder: Path) -> Path:
        if not folder.exists():
            raise FileNotFoundError(f"Calibration folder not found: {folder}")
        txts = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".txt" and p.is_file()])
        if not txts:
            raise FileNotFoundError(f"No calibration .txt files found in: {folder}")
        return txts[-1]

    def _parse_result_file(self, file_path: Path) -> Tuple[str, _FitParams, np.ndarray]:
        """
        Parse the exported calibration result .txt file.
        Returns (eff_curve_filename, _FitParams, data_peaks array[n,4]).
        """
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"File is empty: {file_path}")

        # First line: Used file for the fit: <filename>
        first = lines[0].strip()
        key = "Used file for the fit:"
        if not first.startswith(key):
            raise ValueError(f"Unexpected header in {file_path}: {first}")
        eff_curve_filename = first[len(key) :].strip()

        # Find sections by headers
        def find_index(startswith: str) -> int:
            for i, ln in enumerate(lines):
                if ln.strip().lower().startswith(startswith.lower()):
                    return i
            return -1

        i_params = find_index("parameters")
        i_cov = find_index("Covariance matrix")
        i_data_hdr = find_index("used data points for fitting")

        if i_params == -1 or i_cov == -1 or i_data_hdr == -1:
            raise ValueError(f"Failed to find required sections in {file_path}")

        # Parameters: expect header line next, then values line
        # e.g.:
        # offset a\tslope b
        # -0.00042\t0.0233
        try:
            params_line = lines[i_params + 2]
        except IndexError as exc:
            raise ValueError(f"Malformed parameters block in {file_path}") from exc
        parts = [p for p in params_line.strip().split("\t") if p != ""]
        if len(parts) < 2:
            raise ValueError(f"Invalid parameters line in {file_path}: {params_line}")
        a_val = float(parts[0])
        b_val = float(parts[1])

        # Covariance: two lines of tab-separated floats
        try:
            cov_row1 = [float(x) for x in lines[i_cov + 1].strip().split("\t") if x != ""]
            cov_row2 = [float(x) for x in lines[i_cov + 2].strip().split("\t") if x != ""]
            cov = np.array([cov_row1, cov_row2], dtype=float)
        except Exception as exc:
            raise ValueError(f"Malformed covariance block in {file_path}") from exc

        # Data points header line after the label; next line is column headers; data starts afterwards
        i_data_start = i_data_hdr + 2
        data_rows: list[tuple[float, float, float, str]] = []
        for ln in lines[i_data_start:]:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 4:
                # stop if footer or malformed
                continue
            try:
                energy = float(parts[0])
                eff = float(parts[1])
                eff_err = float(parts[2])
                nuclide = parts[3].strip()
                data_rows.append((energy, eff, eff_err, nuclide))
            except ValueError:
                # probably a footer line; skip
                continue

        data_peaks = np.array(data_rows, dtype=object) if data_rows else np.zeros((0, 4), dtype=object)
        fit_params = _FitParams(a=a_val, b=b_val, cov=cov)
        return eff_curve_filename, fit_params, data_peaks

    def _load_efficiency_curve(self, file_path: Path, level: int) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"Efficiency curve file not found: {file_path}")
        if file_path.suffix.lower() == ".txt":
            # default normalized efficiency (txt): skip 10 header rows; take column 0 (E) and 3 (default curve)
            data = np.loadtxt(file_path, skiprows=10)
            curve = np.column_stack((data[:, 0], data[:, 3]))
        else:
            # CSV with multiple level columns: col 0 energy, col level+1 is the column for this level
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            col = min(level + 1, data.shape[1] - 1)  # safety clamp
            curve = np.column_stack((data[:, 0], data[:, col]))
        # Ensure sorted by energy
        order = np.argsort(curve[:, 0])
        return curve[order]

    def _resolve_store_path(self, store_path: Union[str, Path]) -> Path:
        p = Path(store_path)
        if p.is_dir() or not p.suffix:
            fname = (
                f"efficiency_Level{self.level:02d}_{self.setup_name}_Radius{self._float_to_name(self.source_radius)}mm.pdf"
            )
            return p / fname
        return p

    @staticmethod
    def _float_to_name(value: float) -> str:
        # Convert 25.6 -> '25-6', 1.0 -> '1'
        s = ("%g" % value)
        return s.replace(".", "-")

    def _return_distance_source_detector(self, d_err: float) -> ufloat:
        # [mm]; original: 11.4 - 3 + 10*level
        return ufloat(11.4 - 3 + 10 * self.level, d_err)

    # ----- corrections -----
    def _add_correction_geometrical_factor(self) -> ufloat:
        # For point-like sources, correction is 1
        if self.source_radius < 1e-3:
            return ufloat(1.0, 0.0)

        s_dD = 0.5  # [mm]
        dD = self._return_distance_source_detector(s_dD)
        fG_coin_val, fG_coin_err = geom.geometrical_factor_coin(self.source_radius, 0.5, dD.nominal_value, dD.std_dev)
        fG_point_val, fG_point_err = geom.geometrical_factor_point(dD.nominal_value, dD.std_dev)
        fG_coin = ufloat(fG_coin_val, fG_coin_err)
        fG_point = ufloat(fG_point_val, fG_point_err)
        return fG_coin / fG_point

    def _add_correction_shape_efficiency_curve(self, E: float) -> float:
        # For point-like sources, correction is 1
        if self.source_radius < 1e-3:
            return 1.0

        # Only applies for specific normalized efficiency baselines
        fname = self._eff_curve_filename
        d0 = self._return_distance_source_detector(0.5)  # [mm]

        # Guard against numerical issues by using a tiny but finite radius for the denominator
        tiny_R = 1e-3
        if fname == "SDD_normalized-Efficiency_point-source_all-levels.csv":
            num = float(norms.efficiency_normalized_accurate(E, d0.nominal_value, self.source_radius))
            den = float(norms.efficiency_normalized_accurate(E, d0.nominal_value, tiny_R))
            return num / den if den > 0 else 1.0
        elif fname == "SDD_normalized-Efficiency_point-source_with-air_all-levels.csv":
            num = float(norms.efficiency_normalized_accurate_with_air(E, d0.nominal_value, self.source_radius))
            den = float(norms.efficiency_normalized_accurate_with_air(E, d0.nominal_value, tiny_R))
            return num / den if den > 0 else 1.0
        else:
            # Default file already represents a specific detector-normalized curve; do not apply extra shape correction
            return 1.0
