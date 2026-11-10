import cv2
import numpy as np

from map_digitizer.town_detection import detect_towns


def paper(width=400, height=300):
    return np.full((height, width), 225, np.uint8)


def draw_village(image, center):
    cv2.circle(image, center, 5, 40, 1, cv2.LINE_AA)


def draw_town(image, center):
    """Engraved double rings use fine lines with a clear gap between them."""
    cv2.circle(image, center, 12, 40, 1, cv2.LINE_AA)
    cv2.circle(image, center, 8, 40, 1, cv2.LINE_AA)


def nearest(found, center):
    return min(found, key=lambda town: np.hypot(town.x - center[0], town.y - center[1]))


def test_finds_villages_and_marks_towns_as_double_rings():
    image = paper()
    for center in ((60, 60), (200, 80), (320, 220)):
        draw_village(image, center)
    draw_town(image, (120, 200))
    found = detect_towns(image)
    for center in ((60, 60), (200, 80), (320, 220)):
        village = nearest(found, center)
        assert np.hypot(village.x - center[0], village.y - center[1]) < 3
        assert not village.is_double_ring
    town = nearest(found, (120, 200))
    assert town.is_double_ring and np.hypot(town.x - 120, town.y - 200) < 3


def test_a_town_is_reported_once_not_as_two_rings():
    image = paper()
    draw_town(image, (200, 150))
    assert sum(np.hypot(town.x - 200, town.y - 150) < 20 for town in detect_towns(image)) == 1


def test_blank_paper_has_no_towns():
    assert detect_towns(paper()) == []


def test_a_slanted_oval_like_a_letter_o_is_not_a_confident_town():
    image = paper()
    cv2.ellipse(image, (200, 150), (7, 4), 60, 0, 360, 40, 2, cv2.LINE_AA)
    assert not any(town.is_confident and np.hypot(town.x - 200, town.y - 150) < 8 for town in detect_towns(image))
