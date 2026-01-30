# Intact-Pyslm Quick Start (Windows)

Follow these steps in Windows PowerShell to install into a virtual environment and run the examples.

## 1) Prerequisites

- Install Python 3.10+ (64-bit) from https://www.python.org/downloads/
- Install Git for Windows: https://git-scm.com/download/win

## 2) Clone and create a virtual environment

```powershell
# Choose a folder (e.g., Documents)
cd $HOME\Documents

# Get the code
git clone https://github.com/intact-solutions/pyslm.git
cd pyslm

# Create and activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install requirements and Intact PySLM

```powershell
# Install dependencies from the repo root
pip install -r requirements.txt

# Use editable install so local changes are picked up
pip install -e .
```

If you see errors building C/C++ packages, install "Visual Studio Build Tools" (C++ workload):
https://visualstudio.microsoft.com/visual-cpp-build-tools/

## 4) Verify your install

```powershell
python -c "import sys, pyslm; print('Python:', sys.version); print('pyslm:', pyslm.__file__)"
```
The output path should point inside this cloned `pyslm` folder.

## 5) Run the examples

Run from the repository root.

- Visualize multi-scale island hatching
```powershell
python examples_intact\example_island_multiscale_viz.py
```

- Export SCODE files
```powershell
python examples_intact\export_scode_examples.py
```
This writes two files into `examples_intact/`:
- `neighborhood_paths_L0_x{X}_y{Y}_r{R}.scode`
- `layer_islands_L0.scode`

- Export SCODE files (GE bracket + zones)
```powershell
python examples_intact\export_scode_gebracket_zone_aware.py
```
This writes two files into `examples_intact/`:
- `gebracket_neighborhood_paths_L0_x{X}_y{Y}_r{R}.scode`
- `gebracket_layer_islands_L0.scode`

Notes:
- The `.scode` formats (columns) are the same as the non-GE examples.
- The scan parameters used during export (power/speed via `bid`/BuildStyle) vary by zone.

- Verify SCODE visually (replace the filename with yours if different)
```powershell
python examples_intact\plot_scode_verify.py --paths examples_intact\neighborhood_paths_L0_x-29.154_y-86.947_r4.00.scode --islands examples_intact\layer_islands_L0.scode
```

- Verify SCODE visually (GE bracket + zones)
```powershell
python examples_intact\plot_scode_verify.py --paths examples_intact\gebracket_neighborhood_paths_L0_x-94.180_y-28.258_r4.00.scode --islands examples_intact\gebracket_layer_islands_L0.scode
```

Notes:
- If your path contains spaces and you need to quote the python executable path, use PowerShell's call operator:
```powershell
& ".venv\Scripts\python.exe" "examples_intact\plot_scode_verify.py" --paths "..." --islands "..."
```

- Export island info for all layers
```powershell
# Per-layer files with per-layer indexing (default)
python examples_intact\example_export_all_islands.py

# Per-layer files with global indexing across layers
python examples_intact\example_export_all_islands.py --global-island-indexing

# Single aggregated file containing all islands (global indexing)
python examples_intact\example_export_all_islands.py --single-file
```
Notes:
- Default layer thickness is `0.5` (change via `--layer-thickness`).
- Empty layers are skipped (no file is created).
- Output directory defaults to `examples_intact` (override with `--outdir`).
- Default model is `models\frameGuide.stl` (override with `--model-path`).
- Aggregated output is written to `examples_intact\all_layer_islands_global.scode`.

- Export island info for all layers (GE bracket + zones)
```powershell
# Per-layer files with per-layer indexing (default)
python examples_intact\export_all_island_gebracket_zone_aware.py

# Per-layer files with global indexing across layers
python examples_intact\export_all_island_gebracket_zone_aware.py --global-island-indexing

# Single aggregated file containing all islands (global indexing)
python examples_intact\export_all_island_gebracket_zone_aware.py --single-file
```
Notes:
- Default model is `geometry_intact\zone_aware_island_gebracket\ge_bracket_original.stl` (override with `--model-path`).
- Aggregated output is written to `examples_intact\gebracket_all_layer_islands_global.scode`.
- The `.scode` format is unchanged; zone-specific scan parameters are applied via `bid`/BuildStyle.

## 6) Zone query API (point -> zone)

The zone-aware workflow exposes a public API for querying which zone a point lies in.

Steps:
- Load zone geometries as `pyslm.Part`
- Slice them at a given `z` with `build_zone_polygons()`
- Query points with `find_zone_at_point()`

Example (GE bracket):
```powershell
python examples_intact\query_island_centers_gebracket_zone_aware.py --z 10
python examples_intact\query_island_centers_gebracket_zone_aware.py --z 10 --plot
```

Python usage:
```python
from pyslm.analysis import build_zone_polygons, find_zone_at_point

zone_polys = build_zone_polygons(zone_parts, z)
zone_name = find_zone_at_point((x, y), zone_polys, priority=zone_priority, default='base')
```

## 6a) Zone-aware path planning (zones -> scan parameters -> SCODE)

This section documents the full zone-aware workflow used by:
- `examples_intact\export_scode_gebracket_zone_aware.py`
- `examples_intact\export_all_island_gebracket_zone_aware.py`

The goal is to export `.scode` files where **each island (and contour) carries zone-specific scan parameters**.

### Conceptual model

The zone-aware pipeline connects four concepts:

1) **Zone geometry** (meshes per zone)
- Each zone is a separate mesh (e.g., `high_sensi_zone.ply`, `boundary_zone.ply`).
- At a given layer height `z`, each zone mesh is sliced into shapely polygons via `build_zone_polygons(zone_parts, z)`.

2) **Island geometry** (output of island hatching)
- `IslandHatcher.hatch(slice)` produces a `Layer`.
- `layer.geometry` contains many geometry items, including:
  - islands (`subType == 'island'`)
  - contours (`subType == 'inner'` / `subType == 'outer'`)

3) **Classification** (assign a zone + buildstyle ID)
- `classify_layer_geometry(layer, zone_polys, zone_bids, contour_bid=..., priority=...)` modifies `layer.geometry` in place.
- For each island, it:
  - finds the zone containing the island centroid (with optional overlap priority)
  - sets `geom.zoneName` and `geom.bid` (BuildStyle ID)
- For contours, it assigns `geom.bid = contour_bid`.

4) **BuildStyle mapping** (bid -> power/speed)
- SCODE export resolves scan parameters using `(mid, bid) -> BuildStyle`.
- This means every `geom` must have:
  - `geom.bid` (assigned by `classify_layer_geometry`)
  - `geom.mid` (you assign this; examples use `mid = 1`)
- Your `models` list must contain a `Model(mid=...)` with `BuildStyle(bid=...)` entries matching the zone bids.

### Run the GE bracket zone-aware export (single layer)

```powershell
python examples_intact\export_scode_gebracket_zone_aware.py
```

Outputs (written into `examples_intact\`):
- `gebracket_neighborhood_paths_L0_x{X}_y{Y}_r{R}.scode`
  - Query 1: hatch segments for a chosen “owner” island plus neighbors within radius `R`
  - Columns include `power` and `speed` resolved from the island’s assigned zone
- `gebracket_layer_islands_L0.scode`
  - Query 2: one row per island with entry/exit proxy points and timing summary
  - Columns include `power` resolved from the island’s assigned zone

### Run the GE bracket zone-aware export (all layers)

```powershell
# Per-layer files with per-layer indexing (default)
python examples_intact\export_all_island_gebracket_zone_aware.py

# Per-layer files with global indexing across layers
python examples_intact\export_all_island_gebracket_zone_aware.py --global-island-indexing

# Single aggregated file containing all islands (global indexing)
python examples_intact\export_all_island_gebracket_zone_aware.py --single-file
```

Notes:
- Global indexing makes `island-idx` unique across layers, which is usually easier for downstream analysis.
- Empty layers are skipped.

### Adapting to your own part

To use this workflow on a new model:

1) **Provide the main part geometry**
- Load your part as a `pyslm.Part` and slice it with `getVectorSlice(z)`.

2) **Provide one mesh per zone**
- Create meshes for each zone region (STL/PLY) in the same coordinate system as the main part.
- Load each as a `pyslm.Part` and pass them into `build_zone_polygons(zone_parts, z)`.

3) **Define zone priority and zone bids**
- Define `zone_priority` to resolve overlaps deterministically.
- Define `zone_bids` mapping `zone_name -> bid`.

4) **Create a `Model` with matching BuildStyles**
- For every `bid` that may appear on geometry, ensure there is a `BuildStyle(bid=...)`.
- Assign `geom.mid` for every geometry in the layer to match `model.mid`.

5) **Export**
- Use `write_neighborhood_paths_scode(...)` for neighborhood segment dumps.
- Use `write_layer_island_info_scode(...)` for per-island row exports.

## 7) General tips

- Activate env in a new PowerShell window:
```powershell
.\.venv\Scripts\Activate
```
- Deactivate when finished:
```powershell
deactivate
```
- Update to latest code:
```powershell
git pull
pip install -e . --upgrade
```
- Confirm which PySLM is importing:
```powershell
python -c "import pyslm; print(pyslm.__file__)"
```

## 8) Troubleshooting

- Build errors for `triangle`, `manifold3d`, or `shapely`:
  - Install Visual C++ Build Tools (C++ workload) and retry `pip install -r requirements.txt`.
- The examples import the wrong PySLM (e.g., global install):
  - Ensure the virtual environment is activated and re-run `pip install -e .`.

- `ModuleNotFoundError` (e.g., `No module named 'networkx'`):
  - Confirm you're using the venv Python:
```powershell
where python
python -c "import sys; print(sys.executable)"
python -m pip -V
```
  - Reinstall dependencies into the active venv:
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

