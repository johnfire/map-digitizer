import numpy as np

from map_digitizer.georeference import ThinPlateSpline, leave_out_errors_km, match_to_real_towns, refine_with_matches


def bent(points):
    """A smooth, uneven distortion like an old survey's: scale, rotation and a gentle bend, in metres."""
    x_values, y_values = points[:, 0], points[:, 1]
    east = 400000 + 85.0 * x_values + 4.0 * y_values + 0.002 * x_values * y_values
    north = 5000000 - 86.0 * y_values + 3.0 * x_values
    return np.c_[east, north]


def grid(count=7, spacing=400):
    values = np.arange(count) * spacing + 100.0
    return np.array([[x_value, y_value] for x_value in values for y_value in values])


def test_spline_recovers_a_smooth_distortion_between_anchors():
    anchors = grid()
    spline = ThinPlateSpline.fit(anchors, bent(anchors), smoothing=0.0)
    probes = np.array([[350.0, 760.0], [1720.0, 1100.0], [990.0, 2210.0]])
    assert np.hypot(*(spline.apply(probes) - bent(probes)).T).max() < 50  # metres


def test_leave_out_error_is_small_for_consistent_anchors():
    anchors = grid()
    assert np.median(leave_out_errors_km(anchors, bent(anchors), np.arange(len(anchors)), exclusion_metres=1.0)) < 0.5


def test_one_misplaced_anchor_shows_up_in_the_leave_out_errors():
    anchors = grid()
    real = bent(anchors)
    real[24] += [6000, 0]
    errors = leave_out_errors_km(anchors, real, np.arange(len(anchors)), exclusion_metres=1.0)
    assert int(errors.argmax()) == 24 and errors[24] > 3


def test_matching_pairs_each_real_town_with_one_scan_point_only():
    anchors = grid()
    spline = ThinPlateSpline.fit(anchors, bent(anchors))
    scan_points = np.array([[500.0, 500.0], [505.0, 500.0]])
    matches = match_to_real_towns(spline, scan_points, bent(np.array([[500.0, 500.0]])), limit_metres=2000)
    assert list(matches.values()) == [0] and len(matches) == 1


def test_refining_with_matched_towns_adds_anchors():
    named = grid(count=4, spacing=900)
    towns_scan = grid(count=8, spacing=350) + 37.0
    _scan, _real, matches = refine_with_matches(named, bent(named), towns_scan, bent(towns_scan))
    assert len(matches) > 40
