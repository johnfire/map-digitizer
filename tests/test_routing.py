import cv2
import numpy as np
import pytest

from map_digitizer import routing as worker


def paper(width=800, height=400):
    return np.full((height, width), 225, np.uint8)


def test_route_follows_a_drawn_road_instead_of_the_straight_line():
    scan = paper()
    # An L-shaped road: the straight line between its ends runs across blank paper.
    cv2.line(scan, (50, 50), (700, 50), 40, 2)
    cv2.line(scan, (700, 50), (700, 350), 40, 2)
    result = worker.find_route(scan, [[50, 50], [700, 350]])
    points = np.array(result["path"])
    assert np.hypot(points[:, 0] - 700, points[:, 1] - 50).min() < 8, "route should turn at the road's corner"


def test_route_passes_through_a_bend_point():
    scan = paper()
    result = worker.find_route(scan, [[50, 200], [400, 40], [750, 200]])
    points = np.array(result["path"])
    assert np.hypot(points[:, 0] - 400, points[:, 1] - 40).min() < 3


def test_path_is_simplified():
    scan = paper()
    cv2.line(scan, (20, 200), (780, 200), 40, 2)
    assert len(worker.find_route(scan, [[20, 200], [780, 200]])["path"]) < 12


def test_blank_country_reads_as_open_and_dense_hatching_as_mountain():
    blank = worker.find_route(paper(), [[100, 200], [700, 200]])["terrain"]
    assert blank["open"] == 100

    hatched = paper()
    for x in range(0, 800, 8):  # about 37% ink, like the heaviest engraved hill shading
        cv2.line(hatched, (x, 0), (x, 399), 40, 2)
    assert worker.find_route(hatched, [[100, 200], [700, 200]])["terrain"]["mountain"] > 80


def test_terrain_shares_add_up_to_100():
    shares = worker.terrain_shares(np.array([0.1] * 200 + [0.33] * 200 + [0.5] * 200))
    assert sum(shares.values()) == 100


@pytest.mark.parametrize("points", [[[10, 10]], [[10, 10], [5000, 10]], [[10, 10], [-1, 10]], [[10, 10, 3], [20, 20]]])
def test_invalid_points_are_refused(points):
    with pytest.raises(worker.BadRequest):
        worker.find_route(paper(), points)


def test_points_too_far_apart_are_refused():
    with pytest.raises(worker.BadRequest):
        worker.find_route(paper(5000, 100), [[10, 10], [4900, 10]])


def relief_grid(slope):
    return {"slope": np.array(slope, np.float32), "left": 0, "top": 0, "step": 100}


def test_real_slope_decides_terrain_when_relief_is_available():
    scan = paper()
    relief = relief_grid([[1.0, 1.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0]] * 4)
    shares = worker.find_route(scan, [[10, 200], [790, 200]], relief=relief)["terrain"]
    assert shares["open"] > 15 and shares["hills"] > 15 and shares["mountain"] > 40
    assert sum(shares.values()) == 100


def test_routes_off_the_relief_grid_fall_back_to_shading():
    relief = relief_grid([[float("nan")] * 8] * 4)
    assert worker.find_route(paper(), [[100, 200], [700, 200]], relief=relief)["terrain"]["open"] == 100


def river_mask(lines):
    return worker.rasterize_rivers([{"path": line} for line in lines], (400, 800))


def test_a_road_across_a_river_counts_one_crossing():
    scan = paper()
    cv2.line(scan, (50, 200), (750, 200), 40, 2)
    result = worker.find_route(scan, [[50, 200], [750, 200]], rivers=river_mask([[[400, 0], [400, 399]]]))
    assert result["riverCrossings"] == 1


def test_a_road_running_beside_a_river_is_not_a_crossing():
    scan = paper()
    cv2.line(scan, (50, 200), (750, 200), 40, 2)
    result = worker.find_route(scan, [[50, 200], [750, 200]], rivers=river_mask([[[100, 200], [700, 200]]]))
    assert result["riverCrossings"] == 0


def test_no_river_layer_means_no_crossings():
    assert worker.find_route(paper(), [[100, 200], [700, 200]])["riverCrossings"] == 0
