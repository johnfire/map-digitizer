import json

from map_digitizer.batches import guarded_place_update, next_batch_path, numbered_ids, write_batch


def test_batch_files_are_numbered_after_the_existing_ones(tmp_path):
    additions = tmp_path / "map-seed-additions"
    additions.mkdir()
    (additions / "10-scale.json").write_text("{}")
    (additions / "40-roads.json").write_text("{}")
    assert next_batch_path(tmp_path, "names").name == "50-names.json"


def test_an_update_remembers_where_the_place_was(tmp_path):
    place = {"id": "p1", "x": 10.0, "y": 20.0, "confidence": "low"}
    update = guarded_place_update(place, modernName="Genova")
    assert (update["expectX"], update["expectY"], update["x"], update["y"]) == (10.0, 20.0, 10.0, 20.0)
    assert update["confidence"] == "low" and update["modernName"] == "Genova"


def test_empty_sections_are_left_out_of_the_file(tmp_path):
    path = write_batch(tmp_path / "b.json", "b", "map", places=[], updates=[{"id": "x"}])
    assert json.loads(path.read_text()) == {"batchId": "b", "projectId": "map", "updates": [{"id": "x"}]}


def test_ids_carry_the_batch_name_so_batches_never_collide():
    assert [item["id"] for item in numbered_ids("50-roads", [{}, {}])] == ["50-roads-0000", "50-roads-0001"]
