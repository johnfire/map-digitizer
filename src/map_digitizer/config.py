"""One map's settings, read from a TOML file in maps/. Paths are relative to the repository root."""
import glob
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Northern Italy, the first map's area: UTM zone 32N and a south, west, north, east box in degrees.
DEFAULT_CRS = 32632
DEFAULT_BOUNDS = (43.9, 7.4, 46.0, 11.2)


@dataclass(frozen=True)
class Region:
    """The part of the scan being digitized, in scan pixels."""
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class MapConfig:
    map_id: str
    scan_path: Path
    region: Region
    km_per_pixel: float
    pack_dir: Path
    work_dir: Path
    osm_files: list[Path]
    dem_dir: Path
    town_aliases: dict[str, str] = field(default_factory=dict)
    # The UTM zone real positions are measured in (EPSG code); far from Italy, use the map's own zone.
    crs: int = DEFAULT_CRS
    # Where to read OpenStreetMap places and rivers from: south, west, north, east in degrees.
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS
    # Elevation grid cell size in scan pixels; pick it so a cell is roughly half a kilometre on the map's scale.
    relief_step: int = 8


def _resolve(relative: str) -> Path:
    return (REPOSITORY_ROOT / relative).resolve()


def _single_match(pattern: str) -> Path:
    matches = glob.glob(str(REPOSITORY_ROOT / pattern))
    if len(matches) != 1:
        raise ValueError(f"scan pattern {pattern!r} matched {len(matches)} files, expected 1")
    return Path(matches[0]).resolve()


def load_config(path: Path) -> MapConfig:
    with open(path, "rb") as config_file:
        raw = tomllib.load(config_file)
    region = raw["region"]
    return MapConfig(
        map_id=raw["id"],
        scan_path=_single_match(raw["scan"]),
        region=Region(region["left"], region["top"], region["width"], region["height"]),
        km_per_pixel=float(raw["km_per_pixel"]),
        pack_dir=_resolve(raw["pack_dir"]),
        work_dir=_resolve(raw.get("work_dir", f"work/{raw['id']}")),
        osm_files=[_resolve(entry) for entry in raw.get("osm_files", [])],
        dem_dir=_resolve(raw.get("dem_dir", "data/copernicus-dem")),
        town_aliases=dict(raw.get("town_aliases", {})),
        crs=int(raw.get("crs", DEFAULT_CRS)),
        bounds=tuple(float(value) for value in raw.get("bounds", DEFAULT_BOUNDS)),
        relief_step=int(raw.get("relief_step", 8)),
    )
