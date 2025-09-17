# Multi-scale Thermal Simulation Support Plan (PySLM)

## Objectives

- Provide geometry and timing for a “contour + island” plan to drive a multi-scale thermal simulation with three levels:
  - Level 1 (Path level): detailed path geometry for the island containing a query location and its immediate neighbors.
  - Level 2 (Island level): temporal sequence of islands in the target layer and one layer underneath.
  - Level 3 (Super-layer level): aggregate time over grouped layers below; no detailed path needed.

## Configuration Variables (defaults, all overridable)

- Island and hatching
  - islandWidth [mm]: 5.0
  - islandOverlap [mm]: 0.0 (negative shrinks islands, positive expands)
  - hatchDistance [mm]: 0.08
  - hatchAngle [deg]: 67.5
  - layerAngleIncrement [deg/layer]: 66.67
  - scanContourFirst [bool]: False
  - numOuterContours: 1
  - numInnerContours: 2
  - contourOffset [mm]: 0.08
  - spotCompensation [mm]: 0.06
  - volumeOffsetHatch [mm]: 0.08

- Neighborhood and layering
  - neighborRadiusR [mm]: 0.8 × islandWidth (immediate neighbors)
  - includeLayersBelow: 1 (Level 2)
  - superLayerGroupSize: 10 layers

- Build style timing (pulsed-mode defaults)
  - pointDistance [μm]: 80
  - pointExposureTime [μs]: 100
  - jumpSpeed [mm/s]: 5000
  - jumpDelay [μs]: 0
  - laserPower [W]: 200 (tracked but not used in timing)
  - Alternatively specify continuous laserSpeed [mm/s]

## Design Overview (non-breaking extensions)

- Per-island output grouping (Stage 1)
  - Add `groupIslands` flag to `pyslm/hatching/islandHatcher.py::IslandHatcher` (default False).
  - When enabled, `IslandHatcher.hatch(...)` appends a separate `HatchGeometry` per island (true scan order) instead of merging all islands into one.
  - Attach per-island metadata to each `HatchGeometry`:
    - `subType = "island"`, `islandId`, `posId`, `boundaryPoly` (Shapely Polygon), `bbox` (minx, miny, maxx, maxy).
  - Contours remain distinct `ContourGeometry` groups. Respect `scanContourFirst`.

- Spatial lookup utilities (Stage 2)
  - Map a query point (x, y, z) to island via point-in-polygon against `boundaryPoly`.
  - Build R-tree index over island `bbox` to query neighbors within `neighborRadiusR`; confirm via polygon distance/overlap.

- Timing and ordering (Stage 3)
  - Use `pyslm/analysis/utils.py` for per-island geometry times and per-layer totals.
  - Level 2: aggregate target layer and `includeLayersBelow` layers.

- Super-layer aggregation (Stage 4)
  - Group layers by `superLayerGroupSize` and sum times (no geometry).

- Visualization (Stage 5)
  - Level 1: highlight island containing query location + immediate neighbors; plot their hatches distinctly; dim others.
  - Level 2: Gantt-like timeline for island sequence in target layer + one layer below.
  - Level 3: bar chart of total time per super-layer.

## Stages and Acceptance Criteria

- Stage 0: Defaults & configuration helper
  - Provide a simple configuration dict/object, and helper to apply params to `IslandHatcher`/`BuildStyle`.
  - Acceptance: examples run with defaults and produce reproducible island patterns.

- Stage 1: Per-island grouping
  - Implement `groupIslands` flag and metadata attachment.
  - Acceptance: `Layer.geometry` contains N island `HatchGeometry` in correct order when enabled; legacy behavior preserved when disabled.

- Stage 2: Spatial lookup & neighbors
  - Implement point-to-island lookup and neighbor discovery (R-tree + Shapely verification).
  - Acceptance: given a point, return owning island and immediate neighbors deterministically.

- Stage 3: Timing & Level 2 aggregation
  - Compute per-island times; aggregate for target + below layer.
  - Acceptance: times sum to `getLayerTime(...)` within tolerance; order matches `Layer.geometry`.

- Stage 4: Super-layer aggregation
  - Group by `superLayerGroupSize` and sum times.
  - Acceptance: super-layer sums match sums of underlying layers.

- Stage 5: Visualizations
  - Add example script to render Level 1/2/3 visual checks.
  - Acceptance: visuals clearly show island selection, order, and relative times.

- Stage 6: Docs & tests
  - Document usage and parameters; add simple tests for neighbor counts, order, and time sums.

## Risks & Mitigations

- Island overlap sign convention: defaults to 0.0; visuals validate intended behavior.
- Performance with many islands: R-tree keeps neighbor queries fast; grouping increases `Layer.geometry` count only when enabled.
- Timing accuracy depends on `Model`/`BuildStyle`: all timing parameters exposed; use `analysis.utils` for calculations.

## Artifacts

- Library changes: `pyslm/hatching/islandHatcher.py` (opt-in `groupIslands`).
- Helpers: `pyslm/analysis/island_utils.py` (point-to-island, neighbors, config helper) [Stage 2].
- Examples: `examples/example_island_multiscale_viz.py` [Stage 5].

## Next Steps

- Implement Stage 1 (`groupIslands`) and add a quick visualization snippet to verify per-island grouping.
