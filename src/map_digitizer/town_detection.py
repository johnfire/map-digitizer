"""Find town symbols on an engraved map: single rings for villages, double rings for towns.

Tuned on the Bacler d'Albe sheet (1796). Other maps draw towns differently; expect to add detectors.
"""
from dataclasses import dataclass

import cv2
import numpy as np

UPSCALE = 3  # small ring interiors only have a measurable shape when enlarged
SAMPLE_ANGLES = np.linspace(0, 2 * np.pi, 72, endpoint=False)
MERGE_DISTANCE_PIXELS = 14


@dataclass(frozen=True)
class Thresholds:
    """Measured on a hand-labelled sample: real rings are round, thin and at least two pixels across."""
    max_roundness: float = 0.13
    min_radius: float = 2.0
    max_thick_stroke: float = 5.0


@dataclass(frozen=True)
class TownCandidate:
    x: float
    y: float
    radius: float
    is_double_ring: bool
    is_confident: bool


def ink_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 14) > 0


def _ring_coverage(ink: np.ndarray, center_x: float, center_y: float, radius: float, reach: np.ndarray) -> float:
    """Share of directions in which ink is found at the given distances from the centre."""
    height, width = ink.shape
    hits = 0
    for angle in SAMPLE_ANGLES:
        columns = np.clip((center_x + reach * np.cos(angle)).astype(int), 0, width - 1)
        rows = np.clip((center_y + reach * np.sin(angle)).astype(int), 0, height - 1)
        hits += bool(ink[rows, columns].any())
    return hits / len(SAMPLE_ANGLES)


def _is_round_blank(component: np.ndarray, area: int) -> bool:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = cv2.arcLength(contours[0], True)
    height, width = component.shape
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / perimeter ** 2
    fill = area / (np.pi * (width + height) ** 2 / 16)
    return circularity >= 0.72 and fill >= 0.8


def find_ring_interiors(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Blank, round patches of paper fully enclosed by ink: the inside of a ring symbol."""
    enlarged = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ink = ink_mask(enlarged)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    found = []
    for label in range(1, count):
        left, top, width, height, area = stats[label]
        if not (8 <= width <= 66 and 8 <= height <= 66) or max(width, height) / min(width, height) > 1.25:
            continue
        component = (labels[top:top + height, left:left + width] == label).astype(np.uint8)
        if not _is_round_blank(component, area):
            continue
        center_x, center_y = centroids[label]
        radius = (width + height) / 4
        if _ring_coverage(ink, center_x, center_y, radius, np.arange(radius + 1, radius + 9)) >= 0.9:
            found.append((center_x / UPSCALE, center_y / UPSCALE, radius / UPSCALE))
    return found


def ring_features(enlarged_ink: np.ndarray, center_x: float, center_y: float) -> dict[str, float]:
    """Shape measurements around a candidate: distance to the ring, how round it is, how thick its strokes are."""
    height, width = enlarged_ink.shape
    first_ink, stroke = [], []
    for angle in SAMPLE_ANGLES:
        steps = np.arange(75)
        columns = np.clip((center_x * UPSCALE + steps * np.cos(angle)).astype(int), 0, width - 1)
        rows = np.clip((center_y * UPSCALE + steps * np.sin(angle)).astype(int), 0, height - 1)
        along = enlarged_ink[rows, columns]
        start = int(np.argmax(along)) if along.any() else 74
        end = start
        while end < 74 and along[end]:
            end += 1
        first_ink.append(start)
        stroke.append(end - start)
    first_ink, stroke = np.array(first_ink, float), np.array(stroke, float)
    return {
        "radius": float(np.median(first_ink)) / UPSCALE,
        "roundness": float(first_ink.std() / max(first_ink.mean(), 1)),
        "thick_stroke": float(np.percentile(stroke, 80)) / UPSCALE,
    }


def passes_thresholds(features: dict[str, float], thresholds: Thresholds) -> bool:
    return (features["roundness"] < thresholds.max_roundness
            and features["radius"] >= thresholds.min_radius
            and features["thick_stroke"] < thresholds.max_thick_stroke)


def _double_ring_template(outer_radius: int, gap: int) -> np.ndarray:
    size = 2 * outer_radius + 11
    middle = size // 2
    template = np.full((size, size), 235, np.uint8)
    cv2.circle(template, (middle, middle), outer_radius, 40, 2, cv2.LINE_AA)
    cv2.circle(template, (middle, middle), max(outer_radius - gap, 2), 40, 2, cv2.LINE_AA)
    return template


def _blank_share(ink: np.ndarray, center_x: float, center_y: float, radius: float) -> float:
    height, width = ink.shape
    columns = np.clip((center_x + radius * np.cos(SAMPLE_ANGLES)).astype(int), 0, width - 1)
    rows = np.clip((center_y + radius * np.sin(SAMPLE_ANGLES)).astype(int), 0, height - 1)
    return float(np.mean(~ink[rows, columns]))


def _has_two_rings(ink: np.ndarray, center_x: float, center_y: float, outer: int, gap: int) -> bool:
    """Both rings present all round, blank between them and blank in the middle: hill shading fails this."""
    scaled_x, scaled_y = center_x * UPSCALE, center_y * UPSCALE
    reach = lambda radius: np.arange(radius * UPSCALE - 3, radius * UPSCALE + 3.5)
    return (_ring_coverage(ink, scaled_x, scaled_y, outer, reach(outer)) >= 0.8
            and _ring_coverage(ink, scaled_x, scaled_y, outer - gap, reach(outer - gap)) >= 0.8
            and _blank_share(ink, scaled_x, scaled_y, (outer - gap / 2) * UPSCALE) >= 0.6
            and _blank_share(ink, scaled_x, scaled_y, max(outer - gap - 3, 1) * UPSCALE) >= 0.75)


def find_double_rings(gray: np.ndarray, enlarged_ink: np.ndarray) -> list[tuple[float, float, float]]:
    smoothed = cv2.GaussianBlur(gray, (0, 0), 0.8)
    hits = []
    for outer in range(8, 17):
        for gap in (3, 4, 5):
            template = _double_ring_template(outer, gap)
            offset = template.shape[0] // 2
            scores = cv2.matchTemplate(smoothed, template, cv2.TM_CCOEFF_NORMED)
            for row, column in zip(*np.where(scores > 0.42)):
                hits.append((float(scores[row, column]), column + offset, row + offset, outer, gap))
    hits.sort(reverse=True)
    kept: list[tuple[float, float, float]] = []
    for _score, center_x, center_y, outer, gap in hits:
        if any(np.hypot(center_x - other_x, center_y - other_y) <= 18 for other_x, other_y, _ in kept):
            continue
        if _has_two_rings(enlarged_ink, center_x, center_y, outer, gap):
            kept.append((float(center_x), float(center_y), float(outer)))
    return kept


def detect_towns(gray: np.ndarray, thresholds: Thresholds = Thresholds()) -> list[TownCandidate]:
    """All town candidates in one image region, in that region's pixels. Doubtful ones are kept, marked."""
    enlarged_ink = ink_mask(cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC))
    found = [TownCandidate(center_x, center_y, radius, True, True) for center_x, center_y, radius in find_double_rings(gray, enlarged_ink)]
    for center_x, center_y, radius in sorted(find_ring_interiors(gray), key=lambda ring: -ring[2]):
        if radius < 2 or any(np.hypot(center_x - town.x, center_y - town.y) < MERGE_DISTANCE_PIXELS for town in found):
            continue
        confident = passes_thresholds(ring_features(enlarged_ink, center_x, center_y), thresholds)
        found.append(TownCandidate(center_x, center_y, radius, False, confident))
    return found
