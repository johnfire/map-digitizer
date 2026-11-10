"""Write batch files that the Map Builder merges into a live project.

A batch applies once and never overrides a person: updates carry where a place was when detection last saw it,
and the server skips any place that has since been moved or confirmed. Files are numbered so they apply in order.
"""
import json
import re
from pathlib import Path

ADDITIONS_DIRECTORY = "map-seed-additions"
NUMBERED_FILE = re.compile(r"^(\d+)-")


def next_batch_path(pack_dir: Path, slug: str) -> Path:
    directory = pack_dir / ADDITIONS_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    numbers = [int(match.group(1)) for file in directory.glob("*.json") if (match := NUMBERED_FILE.match(file.name))]
    return directory / f"{max(numbers, default=0) + 10:02d}-{slug}.json"


def guarded_place_update(place: dict, **changes) -> dict:
    """An update that applies only while the place is still unconfirmed and exactly where it was."""
    return {"id": place["id"], "expectX": place["x"], "expectY": place["y"], "x": changes.get("x", place["x"]),
            "y": changes.get("y", place["y"]), "confidence": changes.get("confidence", place.get("confidence", "high")),
            "modernName": changes.get("modernName", "")}


def write_batch(path: Path, batch_id: str, project_id: str, **sections) -> Path:
    """Sections: places, updates, connections, connectionUpdates, kmPerPixel. Empty sections are left out."""
    content = {"batchId": batch_id, "projectId": project_id}
    content.update({name: value for name, value in sections.items() if value not in (None, [], {})})
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path


def numbered_ids(prefix: str, items: list[dict]) -> list[dict]:
    return [{**item, "id": f"{prefix}-{index:04d}"} for index, item in enumerate(items)]
