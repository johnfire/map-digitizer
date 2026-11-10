"""Fit a scanned map to real geography.

A thin-plate spline bends the scan smoothly so anchor towns land on their real positions. Old surveys are
distorted unevenly, which a single straight transform cannot follow. Real positions are UTM metres.
"""
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

KILOMETRE = 1000.0
UTM_NORTHERN_ITALY = 32632


def lonlat_to_metres(lon: np.ndarray, lat: np.ndarray, crs: int = UTM_NORTHERN_ITALY) -> np.ndarray:
    east, north = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    return np.c_[east, north]


def metres_to_lonlat(metres: np.ndarray, crs: int = UTM_NORTHERN_ITALY) -> tuple[np.ndarray, np.ndarray]:
    return Transformer.from_crs(crs, 4326, always_xy=True).transform(metres[:, 0], metres[:, 1])


def _kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(first[:, None] - second[None], axis=2) / KILOMETRE
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(distance > 0, distance * distance * np.log(distance), 0.0)


@dataclass
class ThinPlateSpline:
    anchors_from: np.ndarray
    coefficients: np.ndarray

    @classmethod
    def fit(cls, anchors_from: np.ndarray, anchors_to: np.ndarray, smoothing: float = 0.05) -> "ThinPlateSpline":
        """Smoothing above zero lets the fit miss slightly misplaced anchors instead of bending around them."""
        count = len(anchors_from)
        kernel = _kernel(anchors_from, anchors_from) + np.eye(count) * smoothing
        linear = np.c_[np.ones(count), anchors_from / KILOMETRE]
        system = np.r_[np.c_[kernel, linear], np.c_[linear.T, np.zeros((3, 3))]]
        return cls(anchors_from, np.linalg.solve(system, np.r_[anchors_to, np.zeros((3, 2))]))

    def apply(self, points: np.ndarray, batch_size: int = 4000) -> np.ndarray:
        results = []
        for start in range(0, len(points), batch_size):
            chunk = points[start:start + batch_size]
            features = np.c_[_kernel(chunk, self.anchors_from), np.ones(len(chunk)), chunk / KILOMETRE]
            results.append(features @ self.coefficients)
        return np.vstack(results) if results else np.zeros((0, 2))


@dataclass
class Georeference:
    """Two-way link between scan pixels and real metres (UTM zone 32N, which covers northern Italy)."""
    scan_anchors: np.ndarray
    real_anchors: np.ndarray
    smoothing: float = 0.05
    crs: int = UTM_NORTHERN_ITALY

    def __post_init__(self):
        self.to_real = ThinPlateSpline.fit(self.scan_anchors, self.real_anchors, self.smoothing)
        self.to_scan = ThinPlateSpline.fit(self.real_anchors, self.scan_anchors, self.smoothing)

    def lonlat_to_real(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return lonlat_to_metres(lon, lat, self.crs)

    def real_to_lonlat(self, real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return metres_to_lonlat(real, self.crs)


def leave_out_errors_km(scan_anchors: np.ndarray, real_anchors: np.ndarray, tested: np.ndarray,
                        exclusion_metres: float = 4000.0, smoothing: float = 0.05) -> np.ndarray:
    """For each tested anchor: refit without anything near it, then measure how far off it lands. Honest accuracy."""
    errors = []
    for index in tested:
        keep = np.hypot(*(real_anchors - real_anchors[index]).T) > exclusion_metres
        spline = ThinPlateSpline.fit(scan_anchors[keep], real_anchors[keep], smoothing)
        predicted = spline.apply(scan_anchors[[index]])[0]
        errors.append(np.hypot(*(predicted - real_anchors[index])) / KILOMETRE)
    return np.array(errors)


def match_to_real_towns(spline: ThinPlateSpline, scan_points: np.ndarray, real_towns: np.ndarray,
                        limit_metres: float) -> dict[int, int]:
    """Pairs each scan point with the nearest real town within the limit; each real town is used once."""
    predicted = spline.apply(scan_points)
    best_for_town: dict[int, tuple[int, float]] = {}
    for point_index, position in enumerate(predicted):
        distances = np.hypot(*(real_towns - position).T)
        town_index = int(distances.argmin())
        distance = float(distances[town_index])
        if distance <= limit_metres and (town_index not in best_for_town or distance < best_for_town[town_index][1]):
            best_for_town[town_index] = (point_index, distance)
    return {point_index: town_index for town_index, (point_index, _) in best_for_town.items()}


def refine_with_matches(named_scan: np.ndarray, named_real: np.ndarray, scan_points: np.ndarray,
                        real_towns: np.ndarray, limits=(3000, 2200, 1600, 1200)) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Start from named anchors, add ring-to-town matches, refit, and tighten the match radius each pass."""
    scan_anchors, real_anchors, matches = named_scan, named_real, {}
    for limit in limits:
        spline = ThinPlateSpline.fit(scan_anchors, real_anchors)
        matches = match_to_real_towns(spline, scan_points, real_towns, limit)
        matched = sorted(matches)
        scan_anchors = np.r_[named_scan, scan_points[matched]]
        real_anchors = np.r_[named_real, real_towns[[matches[index] for index in matched]]]
    return scan_anchors, real_anchors, matches
