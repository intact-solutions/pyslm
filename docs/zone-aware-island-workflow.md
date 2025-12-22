# Zone-Aware Island Workflow

## Overview

This document describes a planned workflow for assigning scan parameters to islands based on geometric zones (e.g., bulk, overhang, boundary regions).

**Status:** Planning  
**Date:** December 2024

---

## Problem Statement

In SLM/L-PBF, different regions of a part may require different laser parameters:
- **Bulk regions**: Standard high-speed parameters
- **Overhang regions**: Reduced power/speed to prevent overheating unsupported material
- **Boundary regions**: Fine contour parameters for surface quality

Currently (`multi_infill_strategy.py`), we handle this by:
1. Pre-segmenting the geometry into separate STL files per zone
2. Hatching each zone independently with different parameters
3. Visualizing/exporting combined results

**Limitation:** Islands are generated per-zone, not globally. This can cause:
- Inconsistent island boundaries at zone interfaces
- Duplicate/overlapping scan vectors at zone edges

---

## Proposed Workflow

### Stage 1: Global Island Generation

Generate islands on the **original (full) geometry**, not per-zone:

```
Input: original_geometry.stl
       ├── zone_bulk.stl
       ├── zone_overhang.stl
       └── zone_boundary.stl (or polygons)

Output: Layer with per-island HatchGeometry (using groupIslands=True)
        Each island has: islandId, boundaryPoly, bbox, coords
```

### Stage 2: Zone Assignment

For each island, determine which zone it belongs to:

```python
for island in layer.geometry:
    if island.subType == "island":
        zone = classify_island(island, zones)
        island.zoneId = zone.id
        island.zoneName = zone.name
```

**Classification logic:**
1. **Fully contained:** If `island.boundaryPoly` is completely inside a zone polygon → assign that zone
2. **Partial overlap:** If island spans multiple zones → use centroid rule:
   - Compute `centroid = island.boundaryPoly.centroid`
   - Assign zone that contains the centroid
3. **No match:** Fallback to default zone (bulk)

### Stage 3: Parameter Assignment

Map zone → BuildStyle (laser parameters):

```python
zone_to_buildstyle = {
    "bulk":     BuildStyle(bid=1, laserPower=200, laserSpeed=800),
    "overhang": BuildStyle(bid=2, laserPower=150, laserSpeed=600),
    "boundary": BuildStyle(bid=3, laserPower=180, laserSpeed=400),
}

for island in layer.geometry:
    if hasattr(island, 'zoneName'):
        island.bid = zone_to_buildstyle[island.zoneName].bid
```

### Stage 4: Visualization / Export

- Visualize with color-coded zones
- Export to SCODE with zone info column
- (Future) Export to machine build file with per-island BuildStyle

---

## Reusable Components from Recent Work

### From `pyslm/hatching/islandHatcher.py`

| Component | Reuse |
|-----------|-------|
| `IslandHatcher.groupIslands = True` | ✅ Generates per-island `HatchGeometry` with metadata |
| `HatchGeometry.boundaryPoly` | ✅ Shapely polygon for spatial queries |
| `HatchGeometry.bbox` | ✅ Fast bounding box filtering |
| `HatchGeometry.islandId` | ✅ Unique island identifier |

### From `pyslm/analysis/island_utils.py`

| Component | Reuse |
|-----------|-------|
| `IslandIndex` (STRtree) | ⚠️ Concept reusable, but indexes islands not zones |
| `find_island_at_point()` | ⚠️ Could adapt to `find_zone_at_point()` |
| `compute_layer_geometry_times()` | ✅ Per-island timing with zone-specific parameters |

### New Components Needed

| Component | Purpose |
|-----------|---------|
| `ZoneIndex` | Spatial index over zone polygons (similar to IslandIndex) |
| `classify_island(island, zones)` | Assign zone based on containment/centroid |
| `ZoneRegistry` | Map zone names → BuildStyle |

---

## Data Flow Diagram

```
┌─────────────────┐
│ original.stl    │
└────────┬────────┘
         │ slice at Z
         ▼
┌─────────────────┐
│ geomSlice       │  (shapely polygons)
└────────┬────────┘
         │ IslandHatcher.hatch(groupIslands=True)
         ▼
┌─────────────────┐
│ Layer           │
│  └── islands[]  │  each with boundaryPoly, islandId
└────────┬────────┘
         │
         │  ┌──────────────────┐
         │  │ Zone STLs/Polys  │
         │  │  - bulk          │
         │  │  - overhang      │
         │  │  - boundary      │
         │  └────────┬─────────┘
         │           │ build ZoneIndex
         ▼           ▼
┌─────────────────────────────────┐
│ classify_island() for each      │
│   → island.zoneId, island.bid   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │  color by zone
│ or Export       │  SCODE with zone column
└─────────────────┘
```

---

## Design Decisions

1. **Zone representation:** STL meshes, sliced per layer
   - Each zone is an STL file (e.g., `bulk_zone.stl`, `overhang_zone.stl`)
   - Slice at each Z to get 2D zone polygons

2. **Partial island handling:** Centroid rule
   - Compute `centroid = island.boundaryPoly.centroid`
   - Assign island to whichever zone contains the centroid
   - If centroid is outside all zones → fallback to default (bulk)

3. **Contour handling:** Contours get their own parameters
   - Contours are NOT zone-classified like islands
   - Contours use a dedicated BuildStyle (e.g., `bid=10` for all contours)
   - This keeps surface quality consistent regardless of interior zone

4. **Inter-zone transitions:** No special handling for now
   - Islands at boundaries simply follow centroid rule

---

## Implementation Phases

### Phase 1: Proof of Concept (Visualization Only)
- [ ] Load original geometry + zone geometries
- [ ] Generate islands with `groupIslands=True`
- [ ] Implement `ZoneIndex` for zone lookup
- [ ] Classify islands by centroid
- [ ] Visualize with zone-colored hatches

### Phase 2: Parameter Assignment
- [ ] Define `ZoneRegistry` with BuildStyle per zone
- [ ] Assign `bid` to each island based on zone
- [ ] Compute per-island timing with zone-specific parameters

### Phase 3: Export
- [ ] Extend SCODE export with zone column
- [ ] (Future) Export to machine build files with per-island parameters

---

## Example Usage (Target API)

```python
import pyslm
from pyslm import Part
from pyslm.hatching import IslandHatcher
from pyslm.geometry import BuildStyle, Model

# -----------------------------------------------------
# 1. Load geometries
# -----------------------------------------------------
part = Part('original')
part.setGeometry('original.stl')

zone_parts = {
    'bulk':     Part('bulk').setGeometry('bulk_zone.stl'),
    'overhang': Part('overhang').setGeometry('overhang_zone.stl'),
    'boundary': Part('boundary').setGeometry('boundary_zone.stl'),
}

# -----------------------------------------------------
# 2. Define BuildStyles per zone + contours
# -----------------------------------------------------
buildstyles = {
    'bulk':     BuildStyle(bid=1, laserPower=200, laserSpeed=800),
    'overhang': BuildStyle(bid=2, laserPower=150, laserSpeed=600),
    'boundary': BuildStyle(bid=3, laserPower=180, laserSpeed=700),
    'contour':  BuildStyle(bid=10, laserPower=180, laserSpeed=400),  # all contours
}

# -----------------------------------------------------
# 3. Hatch original geometry with per-island output
# -----------------------------------------------------
z = 2.0
hatcher = IslandHatcher()
hatcher.groupIslands = True
hatcher.numOuterContours = 1
hatcher.numInnerContours = 1

geom_slice = part.getVectorSlice(z)
layer = hatcher.hatch(geom_slice)

# -----------------------------------------------------
# 4. Slice zone STLs at same Z to get zone polygons
# -----------------------------------------------------
zone_polys = {}
for name, zpart in zone_parts.items():
    zslice = zpart.getVectorSlice(z)
    # Convert to shapely polygons for containment tests
    zone_polys[name] = paths_to_multipolygon(zslice)  # utility function

# -----------------------------------------------------
# 5. Classify islands by centroid
# -----------------------------------------------------
for geom in layer.geometry:
    if getattr(geom, 'subType', '') == 'island':
        centroid = geom.boundaryPoly.centroid
        assigned = 'bulk'  # default fallback
        for zone_name, zone_mpoly in zone_polys.items():
            if zone_mpoly.contains(centroid):
                assigned = zone_name
                break
        geom.zoneName = assigned
        geom.bid = buildstyles[assigned].bid
    elif getattr(geom, 'subType', '') in ('outer', 'inner'):
        # Contours get dedicated parameters
        geom.zoneName = 'contour'
        geom.bid = buildstyles['contour'].bid

# -----------------------------------------------------
# 6. Visualize with zone colors
# -----------------------------------------------------
zone_colors = {
    'bulk': 'gray',
    'overhang': 'red',
    'boundary': 'blue',
    'contour': 'green',
}
# Custom visualization by zoneName...
```

---

## References

- `examples_intact/multi_infill_strategy.py` — Current zone-separate approach
- `pyslm/hatching/islandHatcher.py` — Island generation with metadata
- `pyslm/analysis/island_utils.py` — Spatial indexing and timing
