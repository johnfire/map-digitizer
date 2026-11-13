import json

import numpy as np
import pytest

from map_digitizer.cli import main
from map_digitizer.config import load_config
from map_digitizer.georeference import lonlat_to_metres


def write_config(tmp_path, extra=""):
    scan = tmp_path / "scan.jpg"
    scan.write_bytes(b"not read by these steps")
    config = tmp_path / "map.toml"
    config.write_text(f'''
id = "test-map"
scan = "{scan}"
pack_dir = "{tmp_path / 'pack'}"
work_dir = "{tmp_path / 'work'}"
km_per_pixel = 0.025
{extra}
[region]
left = 0
top = 0
width = 1000
height = 800
''')
    return config


def test_a_map_without_projection_settings_keeps_the_italian_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config.crs == 32632
    assert config.bounds == (43.9, 7.4, 46.0, 11.2)


def test_a_map_can_name_its_utm_zone_and_its_area(tmp_path):
    config = load_config(write_config(tmp_path, 'crs = 32635\nbounds = [34.75, 23.45, 35.75, 26.4]'))
    assert config.crs == 32635
    assert config.bounds == (34.75, 23.45, 35.75, 26.4)


def test_the_elevation_grid_step_is_a_setting_with_the_old_default(tmp_path):
    assert load_config(write_config(tmp_path)).relief_step == 8
    assert load_config(write_config(tmp_path, "relief_step = 24")).relief_step == 24


def test_anchors_read_off_the_scan_become_the_georeference(tmp_path):
    config_path = write_config(tmp_path, 'crs = 32635\nbounds = [34.75, 23.45, 35.75, 26.4]')
    # Five towns on a straight 40 pixels-per-km sheet whose top left is at 35.6 N, 23.5 E.
    towns = [("A", 35.50, 23.70), ("B", 35.45, 24.10), ("C", 35.30, 24.60), ("D", 35.10, 25.20), ("E", 35.20, 25.90)]
    origin = lonlat_to_metres(np.array([23.5]), np.array([35.6]), 32635)[0]
    anchors = []
    for name, lat, lon in towns:
        east, north = lonlat_to_metres(np.array([lon]), np.array([lat]), 32635)[0]
        anchors.append({"name": name, "lat": lat, "lon": lon, "x": (east - origin[0]) / 25, "y": (origin[1] - north) / 25})
    anchors_file = tmp_path / "anchors.json"
    anchors_file.write_text(json.dumps(anchors))

    main(["anchors", "--config", str(config_path), "--anchors", str(anchors_file)])

    saved = json.loads((tmp_path / "work/georef.json").read_text())
    assert saved["crs"] == 32635
    assert np.allclose(saved["scan_anchors"][1], [anchors[1]["x"], anchors[1]["y"]])
    east, north = lonlat_to_metres(np.array([24.10]), np.array([35.45]), 32635)[0]
    assert np.allclose(saved["real_anchors"][1], [east, north], atol=0.01)


def test_the_anchors_step_needs_an_anchors_file(tmp_path):
    with pytest.raises(SystemExit):
        main(["anchors", "--config", str(write_config(tmp_path))])
