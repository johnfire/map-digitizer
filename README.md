# map-digitizer

Turn a scanned historical map into a graph a game can move across: **places** (towns, villages, forts) joined by
**connections** that carry length in kilometres, terrain and river crossings. Real geography sits underneath, so
distances and slopes are measured, not guessed. The old map decides what existed then: roads, towns, old river courses.

The results are loaded into the **Map Builder** (https://1796.christopherrehm.de/builder), where a person reviews them.
Nothing here is trusted without that review: everything arrives marked *suggested*.

## How it fits together

| Piece | Where |
|---|---|
| Processing steps (this repo) | `src/map_digitizer/`, run with `uv run map-digitizer <step>` |
| One map's settings | `maps/<map>.toml` |
| Downloaded source data | `data/` (git-ignored, see below) |
| Intermediate results | `work/<map>/` (git-ignored) |
| Output for the Map Builder | the scenario pack named in the map's settings: `map-layers/` and numbered `map-seed-additions/` batches |
| Editor, server API and route-finder container | `../games/game-engine` and `../games/umpire-client` |

`routing.py` is shared with the route-finder container (`game-engine/map-worker/worker.py`). Keep them in step until the
container imports this package.

## Setup

```bash
uv sync
uv run pytest -q
```

Source data (about 1.5 GB for northern Italy), into `data/`:

- Elevation: Copernicus GLO-30 tiles from `https://copernicus-dem-30m.s3.amazonaws.com/`
  (one 1°×1° tile per file, e.g. `Copernicus_DSM_COG_10_N44_00_E009_00_DEM/…_DEM.tif`) into `data/copernicus-dem/`.
- OpenStreetMap: regional extracts from `https://download.geofabrik.de/` (e.g. `europe/italy/nord-ovest-latest.osm.pbf`)
  into `data/openstreetmap/`.

## The steps, in order

Each step that needs the current state of the map reads it from the server first. The project lives inside the
game server's container at `/data/map-projects/<map-id>.json`:

```bash
ssh <user>@<your-server> 'docker exec umpire-umpire-1 cat /data/map-projects/italy-1796.json' > work/italy-1796/project.json
```

| Step | Command | Check before moving on |
|---|---|---|
| 1. Find towns | `map-digitizer detect --config maps/italy-1796.toml --project work/italy-1796/project.json` | Spot-check a sample by eye. Doubtful finds are marked low confidence, never dropped. |
| 2. Align to real geography | `map-digitizer align …` | Tested error (median, 90th, worst) and any anchors *set aside*. Needs some towns named in the Map Builder first. |
| 3. Build layers | `map-digitizer layers --config maps/italy-1796.toml` | Overlay the rivers on the scan: they should land on the engraved rivers. |
| 4. Suggest roads | `map-digitizer roads …` | Look at the network picture; mountain routes are the weakest. |
| 5. Suggest modern names | `map-digitizer names …` | Names come from the nearest real town and will sometimes be wrong. |

Then commit the scenario pack and redeploy the game (`game-engine/deploy/deploy.sh`). On start, the server merges each
new batch once. It never changes a place a person has moved or confirmed, and never brings back anything deleted.

## Lessons from the 1796 sheet

- **Measure before trusting.** Label a random sample by eye and keep part of it back for testing. My first impression
  of how many detections were false was wrong in both directions.
- **Mark, don't delete.** A missing town is invisible; a doubtful ring costs one click.
- **Anchors must come from people.** Names suggested by an earlier alignment would let the alignment confirm itself.
  Repeated Italian place names produce wild anchors; they are set aside and reported.
- **Check against something the fit never saw.** Modern rivers landing on engraved rivers is the real proof.
- **Number batches.** They apply in file order, which matters on a fresh install.

## Adding another map

Copy `maps/italy-1796.toml`, set the scan, region, scale and scenario pack, and download source data for its area.
Expect town detection to need new work: it is tuned to engraved ring symbols, and other maps draw towns differently.

## Licence

The code is MIT licensed (see `LICENSE`). The data it downloads has its own terms, below.

## Credits

Elevation: Copernicus GLO-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under
COPERNICUS by the European Union and ESA. Rivers and places: © OpenStreetMap contributors, ODbL. Any public use must
show these credits.
