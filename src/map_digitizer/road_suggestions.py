"""Suggest roads between neighbouring places by following the ink drawn between them."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError

from . import routing


@dataclass(frozen=True)
class RoadRules:
    """Measured on the 1796 sheet: real roads rarely wander far, and a route through a third town is two roads."""
    max_pixels_apart: float = 300.0
    neighbours_per_place: int = 4
    max_detour: float = 1.40
    min_clearance_from_other_places: float = 12.0


def _triangles(points: np.ndarray) -> np.ndarray:
    """Triangulation needs three places not in a straight line; with fewer, nearest neighbours do the job alone."""
    try:
        return Delaunay(points).simplices
    except (QhullError, ValueError):
        return np.zeros((0, 3), int)


def candidate_pairs(points: np.ndarray, rules: RoadRules) -> set[tuple[int, int]]:
    """Neighbouring places: triangulation edges plus each place's nearest few."""
    pairs = set()
    for triangle in _triangles(points):
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[0], triangle[2])):
            pairs.add((min(first, second), max(first, second)))
    for index in range(len(points)):
        distances = np.hypot(*(points - points[index]).T)
        for neighbour in np.argsort(distances)[1:rules.neighbours_per_place + 1]:
            pairs.add((min(index, neighbour), max(index, neighbour)))
    return {(int(first), int(second)) for first, second in pairs
            if np.hypot(*(points[first] - points[second])) <= rules.max_pixels_apart}


def _closest_other_place(path: np.ndarray, points: np.ndarray, ends: tuple[int, int]) -> float:
    inner = path[12:-12] if len(path) > 30 else path
    others = [index for index in range(len(points)) if index not in ends]
    if not others:
        return np.inf
    return float(min(np.hypot(*(points[index] - inner).T).min() for index in others))


def suggest_roads(scan: np.ndarray, places: list[dict], km_per_pixel: float, relief=None, rivers=None,
                  rules: RoadRules = RoadRules()) -> list[dict]:
    """Suggested connections in the Map Builder's format, all marked unconfirmed."""
    points = np.array([[place["x"], place["y"]] for place in places], float)
    suggestions = []
    for first, second in sorted(candidate_pairs(points, rules)):
        path, density = routing.trace_segment(scan, points[first].tolist(), points[second].tolist())
        length = float(np.hypot(*np.diff(path, axis=0).T).sum())
        detour = length / float(np.hypot(*(points[first] - points[second])))
        if detour > rules.max_detour or _closest_other_place(path, points, (first, second)) < rules.min_clearance_from_other_places:
            continue
        shares = (routing.relief_shares(path, relief) if relief is not None else None) or routing.terrain_shares(density)
        suggestions.append({
            "fromPlaceId": places[first]["id"], "toPlaceId": places[second]["id"],
            "path": routing.simplify(path), "lengthKm": round(length * km_per_pixel, 1),
            "terrainShares": shares, "terrainType": max(shares, key=shares.get), "routeType": "road",
            "riverCrossing": "unknown" if routing.count_river_crossings(path, rivers) else "none", "status": "suggested",
        })
    return suggestions
