"""Follow the roads drawn on a scan between points, and measure terrain and river crossings along the way.

Shared with the route-finder container in the game engine (game-engine/map-worker/worker.py), which serves
the same functions over HTTP. Keep the two in step until the container imports this package directly.
"""
import cv2
import numpy as np
from skimage.graph import route_through_array

MAX_POINTS = 32
MAX_SEGMENT_PIXELS = 3000
MIN_CROP_MARGIN = 120
SIMPLIFY_EPSILON_PIXELS = 1.5
DENSITY_WINDOW = 61
SKIP_TOWN_PIXELS = 40
OPEN_BELOW = 0.30
HILLS_BELOW = 0.36
OPEN_SLOPE_BELOW = 3.0
HILLS_SLOPE_BELOW = 14.0
RIVER_DRAW_WIDTH = 3
LONGEST_CROSSING_PIXELS = 25


class BadRequest(Exception):
    pass


def rasterize_rivers(rivers, shape):
    mask = np.zeros(shape, np.uint8)
    for river in rivers:
        line = np.array(river["path"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [line], False, 1, RIVER_DRAW_WIDTH)
    return mask


def ink_mask(gray):
    smoothed = cv2.GaussianBlur(gray, (0, 0), 0.7)
    return cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)


def travel_cost(ink):
    """Moving along ink is cheap and across bare paper expensive, so routes follow what is drawn."""
    near_ink = cv2.GaussianBlur(ink, (0, 0), 1.0) / 255.0
    return 1.0 + 40.0 * (1.0 - np.clip(near_ink * 2.2, 0, 1)) ** 2


def trace_segment(scan, start, end):
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance > MAX_SEGMENT_PIXELS:
        raise BadRequest("points too far apart")
    margin = int(max(MIN_CROP_MARGIN, distance * 0.35))
    height, width = scan.shape
    left, top = max(int(min(start[0], end[0])) - margin, 0), max(int(min(start[1], end[1])) - margin, 0)
    right, bottom = min(int(max(start[0], end[0])) + margin, width), min(int(max(start[1], end[1])) + margin, height)
    ink = ink_mask(scan[top:bottom, left:right])
    begin = (int(start[1]) - top, int(start[0]) - left)
    finish = (int(end[1]) - top, int(end[0]) - left)
    steps, _ = route_through_array(travel_cost(ink), begin, finish, fully_connected=True, geometric=True)
    density = cv2.blur((ink > 0).astype(np.float32), (DENSITY_WINDOW, DENSITY_WINDOW))
    pixels = np.array(steps)
    return np.column_stack((pixels[:, 1] + left, pixels[:, 0] + top)), density[pixels[:, 0], pixels[:, 1]]


def terrain_shares(density_samples):
    """Whole percentages of the route in each band. The ends are skipped: town symbols are dense ink."""
    inner = density_samples[SKIP_TOWN_PIXELS:-SKIP_TOWN_PIXELS]
    samples = inner if inner.size else density_samples
    open_share = int(round(100 * float(np.mean(samples < OPEN_BELOW))))
    mountain_share = int(round(100 * float(np.mean(samples >= HILLS_BELOW))))
    return {"open": open_share, "hills": max(100 - open_share - mountain_share, 0), "mountain": mountain_share}


def relief_shares(path, relief):
    """Terrain from real slope, or None when the route is mostly outside the elevation data."""
    columns = ((path[:, 0] - int(relief["left"])) / int(relief["step"])).astype(int)
    rows = ((path[:, 1] - int(relief["top"])) / int(relief["step"])).astype(int)
    grid = relief["slope"]
    inside = (rows >= 0) & (rows < grid.shape[0]) & (columns >= 0) & (columns < grid.shape[1])
    slopes = grid[rows[inside], columns[inside]]
    slopes = slopes[~np.isnan(slopes)]
    if slopes.size < len(path) / 2:
        return None
    open_share = int(round(100 * float(np.mean(slopes < OPEN_SLOPE_BELOW))))
    mountain_share = int(round(100 * float(np.mean(slopes >= HILLS_SLOPE_BELOW))))
    return {"open": open_share, "hills": max(100 - open_share - mountain_share, 0), "mountain": mountain_share}


def count_river_crossings(path, rivers):
    """A short run over a river is a crossing; a long run means the road just follows the bank."""
    if rivers is None:
        return 0
    on_river = rivers[path[:, 1].astype(int), path[:, 0].astype(int)] > 0
    edges = np.diff(np.concatenate(([0], on_river.astype(np.int8), [0])))
    runs = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    return int(np.sum(runs <= LONGEST_CROSSING_PIXELS))


def simplify(path):
    contour = path.astype(np.int32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(contour, SIMPLIFY_EPSILON_PIXELS, False).reshape(-1, 2).astype(float).tolist()


def find_route(scan, points, relief=None, rivers=None):
    """Route through the given points along drawn ink, with terrain and river crossings."""
    if not 2 <= len(points) <= MAX_POINTS:
        raise BadRequest("need 2 to 32 points")
    height, width = scan.shape
    for point in points:
        if len(point) != 2 or not (0 <= point[0] < width and 0 <= point[1] < height):
            raise BadRequest("point outside the scan")
    paths, densities = [], []
    for start, end in zip(points, points[1:]):
        path, density = trace_segment(scan, start, end)
        paths.append(path)
        densities.append(density)
    whole_path = np.vstack(paths)
    terrain = relief_shares(whole_path, relief) if relief is not None else None
    return {
        "path": simplify(whole_path),
        "terrain": terrain or terrain_shares(np.concatenate(densities)),
        "riverCrossings": count_river_crossings(whole_path, rivers),
    }
