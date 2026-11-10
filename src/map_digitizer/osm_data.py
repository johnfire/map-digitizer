"""Read towns and rivers from OpenStreetMap extracts (.osm.pbf). Data: OpenStreetMap contributors, ODbL."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium

PLACE_KINDS = ("city", "town", "village", "hamlet", "suburb")
SETTLEMENT_ORDER = {"city": 0, "town": 1, "village": 2, "hamlet": 3, "suburb": 4}


@dataclass(frozen=True)
class BoundingBox:
    south: float
    west: float
    north: float
    east: float

    def contains(self, lon: float, lat: float) -> bool:
        return self.south < lat < self.north and self.west < lon < self.east

    def overlaps(self, lons: np.ndarray, lats: np.ndarray) -> bool:
        return not (lats.max() < self.south or lats.min() > self.north or lons.max() < self.west or lons.min() > self.east)


@dataclass(frozen=True)
class RealPlace:
    name: str
    lon: float
    lat: float
    kind: str


def read_places(files: list[Path], box: BoundingBox) -> list[RealPlace]:
    places = []
    for file in files:
        processor = osmium.FileProcessor(str(file)).with_filter(osmium.filter.EntityFilter(osmium.osm.NODE))
        for node in processor.with_filter(osmium.filter.KeyFilter("place")):
            kind, name = node.tags.get("place"), node.tags.get("name")
            if kind in PLACE_KINDS and name and box.contains(node.lon, node.lat):
                places.append(RealPlace(name, node.lon, node.lat, kind))
    return places


def find_by_name(places: list[RealPlace], name: str) -> RealPlace | None:
    """The most important place of that exact name: a city beats a hamlet that shares it."""
    candidates = [place for place in places if place.name == name]
    return min(candidates, key=lambda place: SETTLEMENT_ORDER[place.kind]) if candidates else None


def read_rivers(files: list[Path], box: BoundingBox) -> list[tuple[str, np.ndarray]]:
    """Each river way as (name, array of lon/lat). Smaller streams are not tagged as rivers and are left out."""
    seen, rivers = set(), []
    for file in files:
        processor = osmium.FileProcessor(str(file)).with_locations().with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        for way in processor.with_filter(osmium.filter.KeyFilter("waterway")):
            if way.tags.get("waterway") != "river" or way.id in seen:
                continue
            seen.add(way.id)
            points = np.array([(node.lon, node.lat) for node in way.nodes if node.location.valid()])
            if len(points) >= 2 and box.overlaps(points[:, 0], points[:, 1]):
                rivers.append((way.tags.get("name", ""), points))
    return rivers
