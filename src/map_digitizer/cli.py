"""map-digitizer: run the steps for one map. See README.md for the order and what to check after each step."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import batches, layers, osm_data, routing
from .config import MapConfig, load_config
from .georeference import Georeference, leave_out_errors_km, lonlat_to_metres, refine_with_matches
from .road_suggestions import suggest_roads
from .town_detection import detect_towns

MERGE_WITH_EXISTING_PIXELS = 14


def _load_project(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _region_gray(config: MapConfig) -> np.ndarray:
    scan = cv2.imread(str(config.scan_path), cv2.IMREAD_GRAYSCALE)
    region = config.region
    return scan[region.top:region.top + region.height, region.left:region.left + region.width]


def _real_places(config: MapConfig) -> list[osm_data.RealPlace]:
    """OpenStreetMap places, cached per map because reading the extracts takes a minute."""
    cache = config.work_dir / "osm-places.json"
    if cache.exists():
        return [osm_data.RealPlace(**place) for place in json.loads(cache.read_text(encoding="utf-8"))]
    places = osm_data.read_places(config.osm_files, osm_data.BoundingBox(*config.bounds))
    config.work_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([place.__dict__ for place in places], ensure_ascii=False), encoding="utf-8")
    return places


def command_detect(config: MapConfig, project: dict) -> None:
    """Suggest places not yet in the project; doubtful ones are marked low confidence, not dropped."""
    existing = np.array([[place["x"], place["y"]] for place in project["places"]] or [[-1e9, -1e9]])
    found = []
    for town in detect_towns(_region_gray(config)):
        x_on_scan, y_on_scan = town.x + config.region.left, town.y + config.region.top
        if np.hypot(*(existing - [x_on_scan, y_on_scan]).T).min() < MERGE_WITH_EXISTING_PIXELS:
            continue
        found.append({"x": round(x_on_scan, 1), "y": round(y_on_scan, 1), "kind": "town" if town.is_double_ring else "village",
                      "rank": 2 if town.is_double_ring else 4, "status": "suggested",
                      "confidence": "high" if town.is_confident else "low"})
    path = batches.next_batch_path(config.pack_dir, "detected-places")
    batches.write_batch(path, path.stem, config.map_id, places=batches.numbered_ids(path.stem, found))
    print(f"new places {len(found)} (low confidence {sum(place['confidence'] == 'low' for place in found)}) -> {path}")


def _trusted_name(place: dict) -> str:
    """Only names a person gave: a typed printed name, or a modern name on a place a person confirmed.

    Suggested modern names came from an earlier alignment; using them as anchors would let it confirm itself."""
    if place.get("printedName"):
        return place["printedName"]
    return place.get("modernName", "") if place.get("status") == "confirmed" else ""


def _named_anchors(config: MapConfig, project: dict, real: list) -> tuple[list[dict], np.ndarray, np.ndarray]:
    named, scan_points, real_points = [], [], []
    for place in project["places"]:
        printed = _trusted_name(place)
        match = printed and osm_data.find_by_name(real, config.town_aliases.get(printed, printed))
        if match:
            named.append(place)
            scan_points.append([place["x"], place["y"]])
            real_points.append([match.lon, match.lat])
    return named, np.array(scan_points, float), np.array(real_points, float)


def _without_outliers(named: list[dict], scan: np.ndarray, real: np.ndarray, limit_km: float = 8.0):
    """Many Italian place names repeat; a name matched to the wrong town lands far off and is set aside, loudly."""
    errors = leave_out_errors_km(scan, real, np.arange(len(named)))
    keep = errors <= limit_km
    for index in np.where(~keep)[0]:
        print(f"set aside anchor {_trusted_name(named[index])!r}: {errors[index]:.1f} km from where the others put it")
    return [place for place, kept in zip(named, keep) if kept], scan[keep], real[keep]


def command_align(config: MapConfig, project: dict) -> Georeference:
    """Fit the scan to real geography from named places, then tighten it with ring-to-town matches."""
    real = _real_places(config)
    named, named_scan, named_lonlat = _named_anchors(config, project, real)
    named_real = lonlat_to_metres(named_lonlat[:, 0], named_lonlat[:, 1], config.crs)
    named, named_scan, named_real = _without_outliers(named, named_scan, named_real)
    towns = [place for place in real if place.kind in ("city", "town", "village")]
    town_real = lonlat_to_metres(np.array([town.lon for town in towns]), np.array([town.lat for town in towns]), config.crs)
    rings = [place for place in project["places"] if place.get("confidence") != "low"]
    ring_scan = np.array([[place["x"], place["y"]] for place in rings], float)
    scan_anchors, real_anchors, matches = refine_with_matches(named_scan, named_real, ring_scan, town_real)
    errors = leave_out_errors_km(scan_anchors, real_anchors, np.arange(len(named)))
    print(f"named anchors {len(named)}, matched rings {len(matches)}; tested error median {np.median(errors):.2f} km, "
          f"90th {np.percentile(errors, 90):.2f} km, worst {errors.max():.2f} km ({_trusted_name(named[int(errors.argmax())])})")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / "georef.json").write_text(json.dumps({
        "crs": config.crs, "scan_anchors": scan_anchors.tolist(), "real_anchors": real_anchors.tolist(),
        "suggested_names": {rings[index]["id"]: towns[town].name for index, town in matches.items()},
    }, ensure_ascii=False), encoding="utf-8")
    return Georeference(scan_anchors, real_anchors, crs=config.crs)


def command_anchors(config: MapConfig, anchors_file: Path) -> Georeference:
    """
    Fit the scan to real geography from towns a person read off it: a list of {name, lat, lon, x, y}. For maps whose
    towns the detector cannot find, and a check on any map: every anchor is tested by leaving it out.
    """
    anchors = json.loads(anchors_file.read_text(encoding="utf-8"))
    scan_anchors = np.array([[anchor["x"], anchor["y"]] for anchor in anchors], float)
    real_anchors = lonlat_to_metres(np.array([anchor["lon"] for anchor in anchors]), np.array([anchor["lat"] for anchor in anchors]), config.crs)
    errors = leave_out_errors_km(scan_anchors, real_anchors, np.arange(len(anchors)), exclusion_metres=1.0)
    print(f"{len(anchors)} anchors; tested error median {np.median(errors):.2f} km, worst {errors.max():.2f} km "
          f"({anchors[int(errors.argmax())]['name']})")
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / "georef.json").write_text(json.dumps({
        "crs": config.crs, "scan_anchors": scan_anchors.tolist(), "real_anchors": real_anchors.tolist(),
        "anchors_from": str(anchors_file), "suggested_names": {},
    }, ensure_ascii=False), encoding="utf-8")
    return Georeference(scan_anchors, real_anchors, crs=config.crs)


def _saved_georeference(config: MapConfig) -> tuple[Georeference, dict]:
    saved = json.loads((config.work_dir / "georef.json").read_text(encoding="utf-8"))
    # Saves from before the zone was a setting were all made in the default zone.
    crs = saved.get("crs", config.crs)
    return Georeference(np.array(saved["scan_anchors"]), np.array(saved["real_anchors"]), crs=crs), saved


def command_layers(config: MapConfig) -> None:
    georef, _ = _saved_georeference(config)
    directory = config.pack_dir / "map-layers"
    directory.mkdir(parents=True, exist_ok=True)
    relief = layers.build_relief_grid(config.dem_dir, georef, config.region, step=config.relief_step)
    np.savez_compressed(directory / f"{config.map_id}-relief.npz", **relief)
    rivers = layers.project_rivers(osm_data.read_rivers(config.osm_files, osm_data.BoundingBox(*config.bounds)), georef, config.region)
    (directory / f"{config.map_id}-rivers.json").write_text(json.dumps(rivers, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"relief grid {relief['slope'].shape}, rivers {len(rivers)} -> {directory}")


def _pack_layers(config: MapConfig, scan_shape: tuple[int, int]):
    directory = config.pack_dir / "map-layers"
    relief_file, rivers_file = directory / f"{config.map_id}-relief.npz", directory / f"{config.map_id}-rivers.json"
    relief = dict(np.load(relief_file)) if relief_file.exists() else None
    rivers = routing.rasterize_rivers(json.loads(rivers_file.read_text(encoding="utf-8")), scan_shape) if rivers_file.exists() else None
    return relief, rivers


def command_roads(config: MapConfig, project: dict) -> None:
    scan = cv2.imread(str(config.scan_path), cv2.IMREAD_GRAYSCALE)
    relief, rivers = _pack_layers(config, scan.shape)
    places = [place for place in project["places"] if place.get("confidence") != "low"]
    roads = suggest_roads(scan, places, config.km_per_pixel, relief, rivers)
    path = batches.next_batch_path(config.pack_dir, "road-suggestions")
    batches.write_batch(path, path.stem, config.map_id, connections=batches.numbered_ids(path.stem, roads))
    print(f"suggested roads {len(roads)}, {round(sum(road['lengthKm'] for road in roads))} km -> {path}")


def command_names(config: MapConfig, project: dict) -> None:
    _, saved = _saved_georeference(config)
    updates = [batches.guarded_place_update(place, modernName=saved["suggested_names"][place["id"]])
               for place in project["places"]
               if place["id"] in saved["suggested_names"] and place["status"] == "suggested" and not place.get("modernName")]
    path = batches.next_batch_path(config.pack_dir, "modern-names")
    batches.write_batch(path, path.stem, config.map_id, updates=updates)
    print(f"modern name suggestions {len(updates)} -> {path}")


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="map-digitizer", description=__doc__)
    parser.add_argument("step", choices=["detect", "align", "anchors", "layers", "roads", "names"])
    parser.add_argument("--config", type=Path, required=True, help="the map's settings, e.g. maps/italy-1796.toml")
    parser.add_argument("--project", type=Path, help="current project JSON from the Map Builder (see README)")
    parser.add_argument("--anchors", type=Path, help="towns read off the scan: JSON list of {name, lat, lon, x, y}")
    options = parser.parse_args(arguments)
    config = load_config(options.config)
    if options.step == "layers":
        command_layers(config)
        return
    if options.step == "anchors":
        if options.anchors is None:
            parser.error("the anchors step needs --anchors")
        command_anchors(config, options.anchors)
        return
    if options.project is None:
        parser.error(f"the {options.step} step needs --project")
    steps = {"detect": command_detect, "align": command_align, "roads": command_roads, "names": command_names}
    steps[options.step](config, _load_project(options.project))
