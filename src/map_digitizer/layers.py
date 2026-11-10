"""Real-geography layers drawn in scan pixels: a slope grid from elevation data, and rivers."""
import glob
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.merge import merge

from .config import Region
from .georeference import Georeference

METRES_PER_DEGREE = 111_320.0


def _slope_degrees(dem: np.ndarray, degrees_per_cell: float, latitude: float) -> np.ndarray:
    metres_y = degrees_per_cell * METRES_PER_DEGREE
    metres_x = degrees_per_cell * METRES_PER_DEGREE * np.cos(np.radians(latitude))
    rise_y, rise_x = np.gradient(dem, metres_y, metres_x)
    return np.degrees(np.arctan(np.hypot(rise_x, rise_y))).astype(np.float32)


def load_elevation(dem_dir: Path) -> tuple[np.ndarray, "rasterio.Affine"]:
    sources = [rasterio.open(file) for file in sorted(glob.glob(str(dem_dir / "*.tif")))]
    if not sources:
        raise FileNotFoundError(f"no elevation tiles in {dem_dir}")
    mosaic, transform = merge(sources)
    return mosaic[0].astype(np.float32), transform


def build_relief_grid(dem_dir: Path, georef: Georeference, region: Region,
                      step: int = 8, smoothing_metres: float = 700.0) -> dict:
    """Mean slope and elevation for every step x step block of scan pixels. NaN where the data does not reach."""
    dem, transform = load_elevation(dem_dir)
    window = int(round(smoothing_metres / (abs(transform.e) * METRES_PER_DEGREE))) | 1
    slope = cv2.blur(_slope_degrees(dem, abs(transform.e), latitude=transform.f + transform.e * dem.shape[0] / 2), (window, window))
    columns_x = np.arange(region.left, region.left + region.width, step) + step / 2
    rows_y = np.arange(region.top, region.top + region.height, step) + step / 2
    grid_x, grid_y = np.meshgrid(columns_x, rows_y)
    lon, lat = georef.real_to_lonlat(georef.to_real.apply(np.c_[grid_x.ravel(), grid_y.ravel()]))
    dem_columns = ((np.asarray(lon) - transform.c) / transform.a).astype(int)
    dem_rows = ((np.asarray(lat) - transform.f) / transform.e).astype(int)
    inside = (dem_rows >= 0) & (dem_rows < dem.shape[0]) & (dem_columns >= 0) & (dem_columns < dem.shape[1])
    slope_grid = np.full(len(dem_rows), np.nan, np.float32)
    elevation_grid = slope_grid.copy()
    slope_grid[inside] = slope[dem_rows[inside], dem_columns[inside]]
    elevation_grid[inside] = dem[dem_rows[inside], dem_columns[inside]]
    return {"slope": slope_grid.reshape(grid_x.shape), "elevation": elevation_grid.reshape(grid_x.shape),
            "left": region.left, "top": region.top, "step": step}


def project_rivers(rivers: list[tuple[str, np.ndarray]], georef: Georeference, region: Region,
                   simplify_pixels: float = 2.0) -> list[dict]:
    """Rivers that touch the region, in scan pixels, simplified."""
    projected = []
    for name, lonlat in rivers:
        scan_points = georef.to_scan.apply(georef.lonlat_to_real(lonlat[:, 0], lonlat[:, 1]))
        if (scan_points[:, 0].max() < region.left or scan_points[:, 0].min() > region.left + region.width
                or scan_points[:, 1].max() < region.top or scan_points[:, 1].min() > region.top + region.height):
            continue
        simplified = cv2.approxPolyDP(scan_points.astype(np.float32).reshape(-1, 1, 2), simplify_pixels, False).reshape(-1, 2)
        if len(simplified) >= 2:
            projected.append({"name": name, "path": np.round(simplified, 1).tolist()})
    return projected
