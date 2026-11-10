import cv2
import numpy as np

from map_digitizer.road_suggestions import RoadRules, candidate_pairs, suggest_roads


def paper(width=800, height=400):
    return np.full((height, width), 225, np.uint8)


def places(*points):
    return [{"id": f"place-{index}", "x": float(x_value), "y": float(y_value)} for index, (x_value, y_value) in enumerate(points)]


def test_pairs_only_join_places_within_reach():
    points = np.array([[0.0, 0.0], [100.0, 0.0], [5000.0, 0.0]])
    assert candidate_pairs(points, RoadRules(max_pixels_apart=300)) == {(0, 1)}


def test_a_drawn_road_becomes_a_suggestion_with_kilometres():
    scan = paper()
    cv2.line(scan, (100, 200), (400, 200), 40, 2)
    roads = suggest_roads(scan, places((100, 200), (400, 200)), km_per_pixel=0.1)
    assert len(roads) == 1
    assert 29 <= roads[0]["lengthKm"] <= 32 and roads[0]["status"] == "suggested"


def test_a_route_through_a_third_town_is_left_to_the_two_shorter_roads():
    scan = paper()
    cv2.line(scan, (100, 200), (500, 200), 40, 2)
    roads = suggest_roads(scan, places((100, 200), (300, 200), (500, 200)), km_per_pixel=0.1)
    joined = {frozenset((road["fromPlaceId"], road["toPlaceId"])) for road in roads}
    assert frozenset(("place-0", "place-2")) not in joined
    assert frozenset(("place-0", "place-1")) in joined and frozenset(("place-1", "place-2")) in joined


def test_a_long_detour_is_not_suggested():
    scan = paper()
    # A U-shaped road, kept inside the search box the router looks in (about 120 px around the two places).
    # Blank paper between them is expensive to cross, so the route follows the U: twice the straight distance.
    cv2.line(scan, (100, 300), (100, 200), 40, 2)
    cv2.line(scan, (100, 200), (300, 200), 40, 2)
    cv2.line(scan, (300, 200), (300, 300), 40, 2)
    assert suggest_roads(scan, places((100, 300), (300, 300)), km_per_pixel=0.1) == []
