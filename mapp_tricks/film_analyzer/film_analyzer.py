from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from skimage import io  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from uncertainties import ufloat  # type: ignore
import uncertainties.unumpy as unp  # type: ignore


Shape = Literal["circular", "rectangular"]


@dataclass
class FileROIConfig:
    """Configuration for a single film file.

    All sizes are in pixels. Center is (x, y) in pixels. max_dose is in Gy.
    Only one of `radius` (for circular) or (`width`, `height`) (for rectangular)
    is used depending on `shape`.
    """

    filename: str
    shape: Shape = "circular"
    center: Tuple[int, int] = (0, 0)
    radius: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    max_dose: float = 10.0


class FileROIConfigDict(TypedDict, total=False):
    filename: str
    shape: Shape
    center: Tuple[int, int]
    radius: int
    width: int
    height: int
    max_dose: float


@dataclass
class AnalyzerConfig:
    folder: Path
    files: List[FileROIConfig]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) image to grayscale pixel values using luminance weights.

    Returns a float64 array in the same dynamic range as the input dtype.
    """
    if image.ndim == 2:
        g = image.astype(np.float64)
    else:
        # Use first 3 channels
        rgb = image[..., :3].astype(np.float64)
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
        g = np.tensordot(rgb, weights, axes=([-1], [0]))

    # Map to 8-bit-like range [0, 255] to match calibration pixel values
    # Common cases: uint16 TIFFs. If max > 255, scale down accordingly.
    gmax = float(np.nanmax(g)) if np.size(g) > 0 else 0.0
    if gmax <= 0:
        return g
    if g.dtype == np.uint16 or gmax > 255:
        # If original was 16-bit, approximate to 8-bit by dividing by 257
        g = g / 257.0
    elif g.dtype.kind == 'f' and gmax <= 1.0:
        g = g * 255.0
    # Clip to [0,255]
    g = np.clip(g, 0.0, 255.0)
    return g

def _roi_mask(shape: Tuple[int, int], cfg: FileROIConfig) -> np.ndarray:
    h, w = shape[0], shape[1]
    yy, xx = np.ogrid[:h, :w]  # creates open mesh grid for indexing
    cx, cy = cfg.center
    if cfg.shape == "circular":
        r = int(cfg.radius or max(1, min(w, h) // 4))
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    else:
        half_w = int((cfg.width or max(1, w // 2)) // 2)
        half_h = int((cfg.height or max(1, h // 2)) // 2)
        x0, x1 = cx - half_w, cx + half_w
        y0, y1 = cy - half_h, cy + half_h
        return (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)


def _sorted_tif_files(folder: Path) -> List[Path]:
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])
    return files


class FilmAnalyzer:
    """Analyze a folder of film scans (TIF), per-file ROI config, and save Plotly outputs.
    """

    def __init__(
        self,
        folder: Path | str,
        dpi: int = 400,
        calibration_key: str = "EBT3_new_METAS_ImageJwRGB",
        plot_downsample: float = 1,
        progress: Literal["auto", "tqdm", "print", "none"] = "auto",
    ):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {self.folder}")
        self.files: List[Path] = _sorted_tif_files(self.folder)
        if not self.files:
            raise ValueError(f"No .tif/.tiff files found in folder: {self.folder}")
        self.dpi: int = int(dpi)
        self.plot_downsample: float = float(plot_downsample)
        self.progress_mode: Literal["auto", "tqdm", "print", "none"] = progress
        # Load calibration data
        self.calibration_key: str = calibration_key
        self.calibration: Dict = self._load_calibration(calibration_key)

    # -------- Config handling --------
    def generate_default_config(self, default_shape: Shape = "circular", default_max_dose: float = 10.0) -> AnalyzerConfig:
        file_cfgs: List[FileROIConfig] = []
        for f in self.files:
            # Read only header/shape by loading the image (skimage is lazy but loads into memory).
            img = io.imread(str(f))
            h, w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else (img.shape[0], 1)
            cx, cy = w // 2, h // 2

            if default_shape == "circular":
                radius = int(min(w, h) * 0.25)
                cfg = FileROIConfig(
                    filename=f.name,
                    shape="circular",
                    center=(cx, cy),
                    radius=radius,
                    max_dose=default_max_dose,
                )
            elif default_shape == "rectangular":
                width = int(w * 0.5)
                height = int(h * 0.5)
                cfg = FileROIConfig(
                    filename=f.name,
                    shape="rectangular",
                    center=(cx, cy),
                    width=width,
                    height=height,
                    max_dose=default_max_dose,
                )
            else:
                raise ValueError(f"Invalid default_shape: {default_shape}, must be 'circular' or 'rectangular'")
            file_cfgs.append(cfg)
        return AnalyzerConfig(folder=self.folder, files=file_cfgs)

    def print_config(self, config: AnalyzerConfig) -> None:
        """Pretty-print a config to the console."""
        print(f"Folder: {config.folder}")
        print("Files:")
        for c in config.files:
            d = asdict(c)
            print(f"  - {d}")

    def to_dict(self, config: AnalyzerConfig) -> Dict:
        return {
            "folder": str(config.folder),
            "files": [asdict(c) for c in config.files],
        }

    def from_dict(self, data: Dict) -> AnalyzerConfig:
        folder = Path(data.get("folder", self.folder))
        file_cfgs: List[FileROIConfig] = []
        for item in data.get("files", []):
            file_cfgs.append(
                FileROIConfig(
                    filename=item["filename"],
                    shape=item.get("shape", "circular"),
                    center=tuple(item.get("center", (0, 0))),
                    radius=item.get("radius"),
                    width=item.get("width"),
                    height=item.get("height"),
                    max_dose=float(item.get("max_dose", 10.0)),
                )
            )
        return AnalyzerConfig(folder=folder, files=file_cfgs)

    def save_config(self, config: AnalyzerConfig | Dict, path: Path | str) -> None:
        """Save config as JSON to a file for manual editing."""
        if isinstance(config, dict):
            data = config
        else:
            data = self.to_dict(config)
        path = Path(path)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_config(self, path: Path | str) -> AnalyzerConfig:
        """Load config JSON from disk."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.from_dict(data)

    def process_all(self, config: AnalyzerConfig | Dict) -> None:
        """Process all films according to the provided config (dataclass or dict).

        Saves Plotly HTML (and PDF if kaleido is available) files into `<folder>/results/<filename>.html`.
        """
        if isinstance(config, dict):
            config = self.from_dict(config)

        # Build a lookup for configs by filename
        by_name: Dict[str, FileROIConfig] = {c.filename: c for c in config.files}

        results_folder = self.folder / "results"
        results_folder.mkdir(parents=True, exist_ok=True)

        # Configure progress iterator
        total = len(self.files)
        iterable = self.files
        use_tqdm = False
        if self.progress_mode in ("tqdm", "auto"):
            try:
                from tqdm.auto import tqdm  # type: ignore[import-not-found]
                iterable = tqdm(self.files, total=total, desc="Processing films", unit="file")  # type: ignore[assignment]
                use_tqdm = True
            except ImportError:
                use_tqdm = False

        for idx, f in enumerate(iterable, start=1):
            if not use_tqdm and self.progress_mode in ("print", "auto"):
                print(f"[{idx}/{total}] Processing {f.name} …")
            cfg = by_name.get(f.name)
            if cfg is None:
                # if a file was added after config generation, fallback to a simple default
                tmp_conf = self.generate_default_config().files[0]
                cfg = FileROIConfig(
                    filename=f.name,
                    shape=tmp_conf.shape,
                    center=tmp_conf.center,
                    radius=tmp_conf.radius,
                    width=tmp_conf.width,
                    height=tmp_conf.height,
                    max_dose=tmp_conf.max_dose,
                )
                print(f"Warning: No config for file {f.name}; using default: {cfg}")

            img = io.imread(str(f))
            gray = _to_gray(img)

            # create ROI mask depending on shape and position
            mask = _roi_mask(gray.shape, cfg)

            # dose map using inverse Green-Saunders calibration
            dose_map = self._dose_map_from_calibration(gray, cfg.max_dose)

            # Compute ROI mean dose with calibration uncertainty propagation
            mean_dose = self._roi_mean_with_calib_uncertainty(gray, mask, cfg.max_dose)

            # Plot result
            fig = self._make_plot(img, dose_map, mask, cfg, mean_dose)
            out_html = results_folder / f"{f.stem}.html"
            fig.write_html(str(out_html))
            # also store the plot as a static PDF for quick viewing
            out_pdf = results_folder / f"{f.stem}.pdf"
            try:
                fig.write_image(str(out_pdf))
            except RuntimeError as e:
                # Gracefully handle missing Chrome/Kaleido engine or other export issues.
                # Keep the HTML output and inform the user how to enable static export.
                msg = (
                    f"Warning: Could not save PDF for {f.name}: {e}.\n"
                    "Static image export uses Plotly's Kaleido engine, which now requires Google Chrome/Chromium.\n"
                    "To enable PDF export on Linux: either run 'plotly_get_chrome' inside your virtualenv, or install\n"
                    "Chrome/Chromium via your package manager (e.g., 'sudo dnf install chromium'). Skipping PDF."
                )
                print(msg)

            if not use_tqdm and self.progress_mode in ("print", "auto"):
                print(f"Saved: {out_html.name} - ROI mean dose {mean_dose.n:.3f} ± {mean_dose.s:.3f} Gy")

    # --- Calibration & dose ---
    def _load_calibration(self, key: str) -> Dict:
        calib_path = Path(__file__).parent / "calibration_data.json"
        data = json.loads(calib_path.read_text(encoding="utf-8"))
        if key not in data:
            raise KeyError(f"Calibration '{key}' not found in {calib_path}")
        return data[key]
    
    def print_available_calibrations(self) -> None:
        calib_path = Path(__file__).parent / "calibration_data.json"
        data = json.loads(calib_path.read_text(encoding="utf-8"))
        print("Available calibrations:")
        for k in data.keys():
            pars = data[k].get('pars', {})
            pars_str = ", ".join(
                f"{p}: {v.get('value', 'N/A')} ± {v.get('error', 'N/A')}" for p, v in pars.items()
            )
            print(f"- {k}: {data[k].get('calib_str', 'No description')}\n    Parameters: {pars_str}")

    @staticmethod
    def _inv_green_saunders(pixel_value: np.ndarray, Do: float, PVmin: float, PVmax: float, beta: float) -> np.ndarray:
        # Avoid division by zero and out-of-range artifacts
        pv = pixel_value.astype(np.float64)
        # Clip pixel values slightly within [PVmin+eps, PVmax-eps]
        eps = 1e-9
        pv = np.clip(pv, PVmin + eps, PVmax - eps)
        val = Do * ((pv - PVmin) / (PVmax - pv)) ** (1.0 / beta)
        return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

    def _dose_map_from_calibration(self, gray: np.ndarray, max_dose: float) -> np.ndarray:
        # luminance-based grayscale as pixel value input to inverse Green-Saunders
        g = gray.astype(np.float64)
        pars = self.calibration["pars"]
        # Build ufloat parameters; for the map we use nominal values; for ROI stats we carry std separately
        Do = ufloat(pars["Do"]["value"], pars["Do"].get("error", 0.0))
        PVmin = ufloat(pars["PVmin"]["value"], pars["PVmin"].get("error", 0.0))
        PVmax = ufloat(pars["PVmax"]["value"], pars["PVmax"].get("error", 0.0))
        beta = ufloat(pars["beta"]["value"], pars["beta"].get("error", 0.0))

        dose_nominal = self._inv_green_saunders(g, Do.n, PVmin.n, PVmax.n, beta.n)
        # Limit by max_dose (clamp rather than zeroing to preserve signal)
        dose_nominal = np.minimum(dose_nominal, float(max_dose))
        return dose_nominal

    def _roi_mean_with_calib_uncertainty(self, gray: np.ndarray, mask: np.ndarray, max_dose: float) -> ufloat:
        # Use ufloats to propagate calibration parameter uncertainties automatically
        roi_indices = np.where(mask)
        if roi_indices[0].size == 0:
            return ufloat(0.0, 0.0)

        # # If ROI is very large, subsample to keep it manageable
        # max_samples = 100_000
        # if roi_indices[0].size > max_samples:
        #     # stride in both dims to roughly keep under max_samples
        #     stride = int(np.ceil(np.sqrt(roi_indices[0].size / max_samples)))
        #     sampled_mask = np.zeros_like(mask, dtype=bool)
        #     sampled_mask[::stride, ::stride] = mask[::stride, ::stride]
        #     roi_indices = np.where(sampled_mask)

        pv = gray.astype(np.float64)[roi_indices]
        pars = self.calibration["pars"]
        Do = ufloat(pars["Do"]["value"], pars["Do"].get("error", 0.0))
        PVmin = ufloat(pars["PVmin"]["value"], pars["PVmin"].get("error", 0.0))
        PVmax = ufloat(pars["PVmax"]["value"], pars["PVmax"].get("error", 0.0))
        beta = ufloat(pars["beta"]["value"], pars["beta"].get("error", 0.0))

        # Compute dose as UFloat array
        # Clip pv into (PVmin, PVmax) using nominal values for stability in the ratio
        pv_clip = np.clip(pv, PVmin.n + 1e-9, PVmax.n - 1e-9)
        ratio = (pv_clip - PVmin) / (PVmax - pv_clip)
        dose_u = Do * ratio ** (1.0 / beta)

        # Apply max_dose cutoff using nominal values (clamp to max_dose)
        dose_nom = unp.nominal_values(dose_u)
        above = dose_nom > max_dose
        if np.any(above):
            dose_u = dose_u.copy()
            dose_u[above] = ufloat(float(max_dose), 0.0)

        # Mean with proper uncertainty propagation
        mean_u = np.sum(dose_u) / len(dose_u)
        return mean_u

    def _make_plot(self, image_rgb_like: np.ndarray, dose_map: np.ndarray, mask: np.ndarray, cfg: FileROIConfig, mean_dose: ufloat) -> go.Figure:
        # Downsample for plotting to reduce HTML size
        ds = self.plot_downsample
        step = max(1, int(round(1.0 / ds))) if ds < 1.0 else 1
        def _downsample(arr: np.ndarray) -> np.ndarray:
            if ds == 1.0:
                return arr
            # use slicing for speed; nearest-neighbor like
            return arr[::step, ::step] if arr.ndim == 2 else arr[::step, ::step, ...]

        # Convert to 3-channel if needed, then downsample for view
        if image_rgb_like.ndim == 2:
            img_vis = np.stack([image_rgb_like] * 3, axis=-1)
        elif image_rgb_like.shape[-1] == 4:
            img_vis = image_rgb_like[..., :3]
        else:
            img_vis = image_rgb_like

        img_vis_ds = _downsample(img_vis)
        # Ensure the display image is uint8 (0..255) to avoid white outputs
        if img_vis_ds.dtype == np.uint16:
            img_disp = (img_vis_ds / 257.0).astype(np.uint8)
        elif img_vis_ds.dtype.kind == 'f':
            mx = float(np.max(img_vis_ds)) if img_vis_ds.size else 1.0
            if mx <= 1.0:
                img_disp = np.clip(img_vis_ds * 255.0, 0, 255).astype(np.uint8)
            else:
                img_disp = np.clip(img_vis_ds, 0, 255).astype(np.uint8)
        elif img_vis_ds.dtype != np.uint8:
            # generic integer types >8-bit
            scale = float(np.iinfo(img_vis_ds.dtype).max)
            img_disp = np.clip((img_vis_ds.astype(np.float64) / scale) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_disp = img_vis_ds
        dose_map_ds = _downsample(dose_map)
        mask_ds = _downsample(mask.astype(np.uint8)).astype(bool)

        # Coordinates in mm using DPI
        h, w = dose_map_ds.shape[:2]
        px_to_mm = 25.4 / float(self.dpi)
        x_mm = np.arange(w) * px_to_mm * step
        y_mm = np.arange(h) * px_to_mm * step

        # Masked dose for heatmap
        masked_dose = np.where(mask_ds, dose_map_ds, np.nan)

        # Create 3-panel figure: original with ROI, heatmap, profiles
        fig = make_subplots(rows=1, cols=3, column_widths=[0.33, 0.33, 0.34], subplot_titles=("Original", "Dose (Gy)", "Profiles"))

        # Panel 1: original image with ROI overlay (red)
        fig.add_trace(go.Image(z=img_disp), row=1, col=1)
        # draw ROI outline in red
        if cfg.shape == "circular":
            cx, cy = cfg.center
            r = int(cfg.radius or 1)
            # scale centers by downsample step
            cx_ds, cy_ds, r_ds = cx // step, cy // step, r // step
            theta = np.linspace(0, 2 * np.pi, 256)
            xs = cx_ds + r_ds * np.cos(theta)
            ys = cy_ds + r_ds * np.sin(theta)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="red", width=1), name="ROI", showlegend=False), row=1, col=1)
        else:
            cx, cy = cfg.center
            half_w = int((cfg.width or 2) // 2) // step
            half_h = int((cfg.height or 2) // 2) // step
            cx_ds, cy_ds = cx // step, cy // step
            x0, x1 = cx_ds - half_w, cx_ds + half_w
            y0, y1 = cy_ds - half_h, cy_ds + half_h
            fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines", line=dict(color="red", width=1), showlegend=False), row=1, col=1)

        fig.update_xaxes(title_text="x [px]", row=1, col=1)
        fig.update_yaxes(title_text="y [px]", row=1, col=1)

        # Panel 2: dose heatmap (mm axes), flip y to top-down mm if desired
        fig.add_trace(
            go.Heatmap(z=masked_dose[::-1, :], x=x_mm, y=y_mm[::-1], coloraxis="coloraxis",
                       hovertemplate="x=%{x:.2f} mm y=%{y:.2f} mm dose=%{z:.3f} Gy<extra></extra>"),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="x [mm]", row=1, col=2)
        fig.update_yaxes(title_text="y [mm]", row=1, col=2)

        # Panel 3: profiles (horizontal and vertical) from center lines
        cy_idx = h // 2
        cx_idx = w // 2
        horiz = masked_dose[cy_idx, :]
        vert = masked_dose[:, cx_idx]
        # remove NaNs
        horiz_v = horiz[~np.isnan(horiz)]
        horiz_x = x_mm[~np.isnan(horiz)]
        vert_v = vert[~np.isnan(vert)]
        vert_y = y_mm[~np.isnan(vert)]

        fig.add_trace(go.Scatter(x=horiz_x, y=horiz_v, mode='lines', name='horizontal', line=dict(width=1.2)), row=1, col=3)
        fig.add_trace(go.Scatter(x=vert_y, y=vert_v, mode='lines', name='vertical', line=dict(width=1.2)), row=1, col=3)
        fig.update_xaxes(title_text="x,y [mm]", row=1, col=3)
        fig.update_yaxes(title_text="Dose [Gy]", row=1, col=3)

        fig.update_layout(
            coloraxis=dict(colorscale="Viridis", colorbar=dict(title="Gy")),
            title=f"{cfg.filename} - ROI mean dose: {mean_dose.n:.3f} ± {mean_dose.s:.3f} Gy",
            margin=dict(l=40, r=40, t=60, b=40),
            height=300,
            width=1000,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            template="plotly_white",
        )
        return fig


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a folder of TIF film images.")
    parser.add_argument("folder", type=str, help="Folder containing .tif files")
    parser.add_argument("--make-default-config", dest="make_default", action="store_true", help="Generate default config JSON and exit")
    parser.add_argument("--config", type=str, help="Path to config JSON for processing")
    parser.add_argument("--shape", type=str, default="circular", choices=["circular", "rectangular"], help="Default shape for generated config")
    parser.add_argument("--max-dose", type=float, default=10.0, help="Default max dose for generated config")
    parser.add_argument("--dpi", type=int, default=400, help="Dots per inch for mm scaling")
    parser.add_argument("--calibration", type=str, default="EBT3_new_METAS_ImageJwRGB", help="Calibration key from calibration_data.json")
    parser.add_argument("--plot-downsample", type=float, default=0.5, help="Downsample factor for plots (0<ds<=1; e.g., 0.5 halves resolution)")
    parser.add_argument(
        "--progress",
        type=str,
        default="auto",
        choices=["auto", "tqdm", "print", "none"],
        help="Progress reporting: auto (try tqdm), tqdm (force), print (simple prints), none",
    )
    args = parser.parse_args()

    analyzer = FilmAnalyzer(
        args.folder,
        dpi=args.dpi,
        calibration_key=args.calibration,
        plot_downsample=args.plot_downsample,
        progress=args.progress,
    )
    if args.make_default:
        cfg = analyzer.generate_default_config(default_shape=args.shape, default_max_dose=args.max_dose)
        out = Path(args.folder) / "film_config.json"
        analyzer.save_config(cfg, out)
        print(f"Wrote default config to {out}")
        analyzer.print_config(cfg)
        return

    if not args.config:
        raise SystemExit("--config is required unless --make-default-config is used")

    cfg = analyzer.load_config(args.config)
    analyzer.process_all(cfg)
    print(f"Done. Results in {Path(args.folder) / 'results'}")


if __name__ == "__main__":
    _cli()

