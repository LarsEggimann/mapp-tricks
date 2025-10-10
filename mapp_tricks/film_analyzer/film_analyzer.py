

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from skimage import io  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
from plotly.subplots import make_subplots  # type: ignore[import-not-found]
from uncertainties import ufloat  # type: ignore[import-not-found]


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
        return image.astype(np.float64)
    # Use first 3 channels
    rgb = image[..., :3].astype(np.float64)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    return np.tensordot(rgb, weights, axes=([-1], [0]))


def _sorted_tif_files(folder: Path) -> List[Path]:
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])
    return files


class FilmAnalyzer:
    """Analyze a folder of film scans (TIF), per-file ROI config, and save Plotly outputs.

    Basic dose calculation is scaled relative dose (0..max_dose) derived from per-image
    grayscale intensities: dose = (Imax - I) / (Imax - Imin) * max_dose, where Imin/Imax
    are robust (1%, 99%) quantiles to reduce outlier effects. This is a simple placeholder
    until a proper calibration is wired.
    """

    def __init__(self, folder: Path | str):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {self.folder}")
        self.files: List[Path] = _sorted_tif_files(self.folder)
        if not self.files:
            raise ValueError(f"No .tif/.tiff files found in folder: {self.folder}")

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
            else:
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

    # -------- Processing --------
    def process_all(self, config: AnalyzerConfig | Dict) -> None:
        """Process all films according to the provided config (dataclass or dict).

        Saves Plotly HTML files into `<folder>/results/<filename>.html`.
        """
        if isinstance(config, dict):
            config = self.from_dict(config)

        # Build a lookup for configs by filename
        by_name: Dict[str, FileROIConfig] = {c.filename: c for c in config.files}

        results_folder = self.folder / "results"
        results_folder.mkdir(parents=True, exist_ok=True)

        for f in self.files:
            cfg = by_name.get(f.name)
            if cfg is None:
                # If a file was added after config generation, fallback to a simple default
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

            img = io.imread(str(f))
            gray = _to_gray(img)

            # Create ROI mask
            mask = self._roi_mask(gray.shape, cfg)

            # Basic dose map (relative scaling using robust min/max)
            dose_map = self._basic_dose_map(gray, cfg.max_dose)
            dose_roi = np.where(mask, dose_map, np.nan)

            # Compute mean and std over ROI (ignore NaNs)
            roi_vals = dose_roi[~np.isnan(dose_roi)]
            if roi_vals.size == 0:
                mean_dose = ufloat(0.0, 0.0)
            else:
                mean = float(np.mean(roi_vals))
                std = float(np.std(roi_vals, ddof=1)) if roi_vals.size > 1 else 0.0
                mean_dose = ufloat(mean, std)

            # Plot result
            fig = self._make_plot(img, dose_map, mask, cfg, mean_dose)
            out_html = results_folder / f"{f.stem}.html"
            fig.write_html(str(out_html))

    # -------- Helpers --------
    @staticmethod
    def _roi_mask(shape: Tuple[int, int], cfg: FileROIConfig) -> np.ndarray:
        h, w = shape[0], shape[1]
        yy, xx = np.ogrid[:h, :w]
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

    @staticmethod
    def _basic_dose_map(gray: np.ndarray, max_dose: float) -> np.ndarray:
        # Robust scaling with quantiles
        g = gray.astype(np.float64)
        q1, q99 = np.quantile(g, [0.01, 0.99])
        if q99 <= q1:
            q1, q99 = float(np.min(g)), float(np.max(g) + 1e-9)
        scaled = (q99 - g) / (q99 - q1)
        scaled = np.clip(scaled, 0.0, 1.0)
        return scaled * float(max_dose)

    @staticmethod
    def _make_plot(image_rgb_like: np.ndarray, dose_map: np.ndarray, mask: np.ndarray, cfg: FileROIConfig, mean_dose: ufloat) -> go.Figure:
        # Prepare views for plotting
        # Convert image to 3-channel if needed
        if image_rgb_like.ndim == 2:
            img_vis = np.stack([image_rgb_like] * 3, axis=-1)
        elif image_rgb_like.shape[-1] == 4:
            img_vis = image_rgb_like[..., :3]
        else:
            img_vis = image_rgb_like

        masked_dose = np.where(mask, dose_map, np.nan)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Original", "Dose (Gy)"))
        fig.add_trace(go.Image(z=img_vis), row=1, col=1)
        fig.update_xaxes(title_text="x [px]", row=1, col=1)
        fig.update_yaxes(title_text="y [px]", row=1, col=1)

        fig.add_trace(
            go.Heatmap(z=np.flipud(masked_dose), coloraxis="coloraxis", hovertemplate="x=%{x} y=%{y} dose=%{z:.3f} Gy<extra></extra>"),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="x [px]", row=1, col=2)
        fig.update_yaxes(title_text="y [px] (flipped)", row=1, col=2)
        fig.update_layout(
            coloraxis=dict(colorscale="Viridis", colorbar=dict(title="Gy")),
            title=f"{cfg.filename} — ROI mean dose: {mean_dose.n:.3f} ± {mean_dose.s:.3f} Gy",
            margin=dict(l=40, r=40, t=60, b=40),
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
    args = parser.parse_args()

    analyzer = FilmAnalyzer(args.folder)
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

