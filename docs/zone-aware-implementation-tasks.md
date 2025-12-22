# Zone-Aware Island Workflow — Implementation Tasks

**Parent doc:** `zone-aware-island-workflow.md`  
**Created:** December 2024

This document breaks down the implementation into discrete tasks suitable for separate development sessions.

---

## Task Overview

| Task | Description | Dependencies | Est. Effort |
|------|-------------|--------------|-------------|
| T1   | Zone utilities module | None | Small |
| T2   | Zone classification function | T1 | Small |
| T3   | Example script (single layer viz) | T1, T2 | Medium |
| T4   | Multi-layer support | T3 | Small |
| T5   | SCODE export with zone | T3 | Medium |
| T6   | Tests | T1-T3 | Small |

---

## Task 1: Zone Utilities Module

**Goal:** Create `pyslm/analysis/zone_utils.py` with helper functions for zone handling.

### Files to Create
- `pyslm/analysis/zone_utils.py`

### Functions to Implement

```python
# zone_utils.py

def slice_zone_part(zone_part: Part, z: float) -> Optional[MultiPolygon]:
    """
    Slice a zone Part at height z and return a shapely MultiPolygon.
    Returns None if slice is empty at this Z.
    
    Args:
        zone_part: pyslm.Part loaded from zone STL
        z: slice height
    
    Returns:
        shapely.geometry.MultiPolygon or None
    """

def build_zone_polygons(zone_parts: Dict[str, Part], z: float) -> Dict[str, MultiPolygon]:
    """
    Slice all zone Parts at height z.
    
    Args:
        zone_parts: dict mapping zone_name -> Part
        z: slice height
    
    Returns:
        dict mapping zone_name -> MultiPolygon (excludes empty zones)
    """

def find_zone_for_point(x: float, y: float, zone_polys: Dict[str, MultiPolygon], 
                        default: str = 'bulk') -> str:
    """
    Find which zone contains the point (x, y).
    
    Args:
        x, y: query point
        zone_polys: dict from build_zone_polygons()
        default: fallback zone if point not in any zone
    
    Returns:
        zone name string
    """
```

### Implementation Notes
- Use `Part.getVectorSlice(z)` to get paths
- Convert paths to shapely polygons using existing `pathsToClosedPolygons()` from `hatching/utils.py`
- Wrap in `MultiPolygon` for uniform handling

### Acceptance Criteria
- [ ] `slice_zone_part()` returns valid MultiPolygon for non-empty slices
- [ ] `slice_zone_part()` returns None for empty slices (Z above/below zone)
- [ ] `find_zone_for_point()` correctly identifies containing zone
- [ ] `find_zone_for_point()` returns default when point outside all zones

---

## Task 2: Island Classification Function

**Goal:** Add function to classify islands by zone based on centroid.

### Files to Modify
- `pyslm/analysis/zone_utils.py` (add to module from T1)

### Functions to Implement

```python
def classify_island_zone(island_geom, zone_polys: Dict[str, MultiPolygon],
                         default: str = 'bulk') -> str:
    """
    Classify an island HatchGeometry by which zone contains its centroid.
    
    Args:
        island_geom: HatchGeometry with boundaryPoly attribute
        zone_polys: dict from build_zone_polygons()
        default: fallback zone
    
    Returns:
        zone name string
    """

def classify_layer_geometry(layer: Layer, zone_polys: Dict[str, MultiPolygon],
                            zone_buildstyles: Dict[str, int],
                            contour_bid: int = 10,
                            default_zone: str = 'bulk') -> None:
    """
    Classify all geometry in a layer and assign zoneName + bid attributes.
    Modifies layer.geometry in place.
    
    Args:
        layer: Layer with geometry (islands have subType='island', contours have 'outer'/'inner')
        zone_polys: dict from build_zone_polygons()
        zone_buildstyles: dict mapping zone_name -> bid
        contour_bid: bid to assign to all contours
        default_zone: fallback for islands outside all zones
    
    Side effects:
        Sets geom.zoneName and geom.bid for each geometry
    """
```

### Implementation Notes
- Islands: use `geom.boundaryPoly.centroid` for classification
- Contours: always assign `contour_bid` regardless of position
- Handle missing `boundaryPoly` gracefully (skip or use coords centroid)

### Acceptance Criteria
- [ ] Islands get correct `zoneName` based on centroid
- [ ] Islands get correct `bid` from `zone_buildstyles` mapping
- [ ] Contours (subType='outer'/'inner') get `zoneName='contour'` and `bid=contour_bid`
- [ ] Islands outside all zones get default zone

---

## Task 3: Example Script (Single Layer Visualization)

**Goal:** Create working example that demonstrates the full workflow for one layer.

### Files to Create
- `examples_intact/zone_aware_hatching.py`

### Script Structure

```python
"""
Zone-aware island hatching example.

Demonstrates:
1. Loading original geometry + zone STLs
2. Hatching with per-island output
3. Classifying islands by zone (centroid rule)
4. Visualizing with zone-colored hatches
"""

def main():
    # --- Configuration ---
    original_stl = 'path/to/original.stl'
    zone_stls = {
        'bulk': 'path/to/bulk_zone.stl',
        'overhang': 'path/to/overhang_zone.stl',
        'boundary': 'path/to/boundary_zone.stl',
    }
    z = 2.0  # layer height
    
    # Zone -> BuildStyle bid mapping
    zone_bids = {'bulk': 1, 'overhang': 2, 'boundary': 3}
    contour_bid = 10
    
    # Zone -> color for visualization
    zone_colors = {
        'bulk': 'gray',
        'overhang': 'red',
        'boundary': 'blue',
        'contour': 'green',
    }
    
    # --- 1. Load geometries ---
    
    # --- 2. Setup hatcher with groupIslands=True ---
    
    # --- 3. Hatch original geometry ---
    
    # --- 4. Build zone polygons at Z ---
    
    # --- 5. Classify layer geometry ---
    
    # --- 6. Visualize ---
    
    # --- 7. Print summary stats ---
```

### Visualization Requirements
- Plot each geometry colored by `zoneName`
- Legend showing zone colors
- Title with layer Z and island counts per zone

### Acceptance Criteria
- [ ] Script runs without error on test geometry
- [ ] Islands correctly colored by zone
- [ ] Contours shown in contour color
- [ ] Summary prints count of islands per zone

---

## Task 4: Multi-Layer Support

**Goal:** Extend example to process multiple layers.

### Files to Modify
- `examples_intact/zone_aware_hatching.py` (or create `zone_aware_hatching_multilayer.py`)

### Additions

```python
def process_layer(part, zone_parts, z, hatcher, zone_bids, contour_bid):
    """Process a single layer and return classified Layer."""
    # ... encapsulate single-layer logic
    
def main():
    # ... 
    zs = np.arange(z_start, z_end, layer_thickness)
    
    layers = []
    for z in zs:
        layer = process_layer(part, zone_parts, z, hatcher, zone_bids, contour_bid)
        if layer is not None:
            layers.append(layer)
    
    # Summary: islands per zone per layer
    # Optional: animation or multi-layer plot
```

### Acceptance Criteria
- [ ] Processes multiple layers without error
- [ ] Handles empty slices (skip or empty layer)
- [ ] Prints per-layer zone statistics

---

## Task 5: SCODE Export with Zone

**Goal:** Extend SCODE export to include zone information.

### Files to Modify
- `pyslm/analysis/export_scode.py`

### Changes

1. Add optional `zone` column to SCODE output format:
   ```
   # columns: x1 y1 x2 y2 z power eq_speed total_time island-idx zone
   ```

2. Modify export functions to read `zoneName` attribute from geometry

### New/Modified Functions

```python
def export_layer_islands_with_zone(layer, models, z, fh, base_island_idx=0):
    """
    Export island info with zone column.
    
    Output columns: x1 y1 x2 y2 z power eq_speed total_time island-idx zone
    """
```

### Acceptance Criteria
- [ ] SCODE files include zone column when geometry has `zoneName`
- [ ] Backward compatible (zone column omitted if no zoneName)
- [ ] Zone column contains zone name string

---

## Task 6: Tests

**Goal:** Add unit tests for zone utilities.

### Files to Create
- `tests/test_zone_utils.py`

### Test Cases

```python
class TestZoneUtils:
    def test_slice_zone_part_nonempty(self):
        """Zone slice returns MultiPolygon when geometry present."""
        
    def test_slice_zone_part_empty(self):
        """Zone slice returns None when Z outside geometry."""
        
    def test_find_zone_for_point_inside(self):
        """Point inside zone returns correct zone name."""
        
    def test_find_zone_for_point_outside(self):
        """Point outside all zones returns default."""
        
    def test_classify_island_zone(self):
        """Island classified by centroid location."""
        
    def test_classify_layer_geometry_islands(self):
        """All islands in layer get zoneName and bid."""
        
    def test_classify_layer_geometry_contours(self):
        """Contours get contour zoneName and bid."""
```

### Test Data
- Create simple box geometries programmatically (no STL files needed)
- Or use existing `models/frameGuide.stl` with synthetic zone boxes

---

## Implementation Order

```
Session 1: T1 (zone_utils.py basics)
    └── Create module, implement slice_zone_part, build_zone_polygons, find_zone_for_point

Session 2: T2 (classification) + T3 (example)
    └── Add classify functions, create example script, test end-to-end

Session 3: T4 (multi-layer) + T5 (SCODE export)
    └── Extend to multi-layer, add zone to SCODE output

Session 4: T6 (tests) + cleanup
    └── Add tests, docstrings, update howto doc
```

---

## File Summary

### New Files
| File | Task |
|------|------|
| `pyslm/analysis/zone_utils.py` | T1, T2 |
| `examples_intact/zone_aware_hatching.py` | T3, T4 |
| `tests/test_zone_utils.py` | T6 |

### Modified Files
| File | Task |
|------|------|
| `pyslm/analysis/__init__.py` | T1 (export new module) |
| `pyslm/analysis/export_scode.py` | T5 |
| `docs/intact-pyslm-howto.md` | T6 (add example) |

---

## Test Data Requirements

For running the example, need zone STLs. Options:

1. **Use existing** `process_zones` test data (referenced in `multi_infill_strategy.py`):
   ```
   C:\Users\kumar\source_local\process_zones\tests\block\rotated\results\
       bulk_zone.stl
       overhang_zone.stl
       boundary_zone.stl
   ```

2. **Create synthetic** test geometry:
   - Simple box as original
   - Smaller boxes as zones (bulk = center, overhang = one corner, etc.)

---

## Prompts for Each Session

### Session 1 Prompt
```
Implement zone utilities for the zone-aware island workflow.
See docs/zone-aware-implementation-tasks.md Task 1.
Create pyslm/analysis/zone_utils.py with:
- slice_zone_part()
- build_zone_polygons()
- find_zone_for_point()
```

### Session 2 Prompt
```
Continue zone-aware workflow implementation.
See docs/zone-aware-implementation-tasks.md Tasks 2 and 3.
1. Add classify_island_zone() and classify_layer_geometry() to zone_utils.py
2. Create examples_intact/zone_aware_hatching.py example script
```

### Session 3 Prompt
```
Continue zone-aware workflow implementation.
See docs/zone-aware-implementation-tasks.md Tasks 4 and 5.
1. Extend zone_aware_hatching.py for multi-layer processing
2. Add zone column to SCODE export in export_scode.py
```

### Session 4 Prompt
```
Finalize zone-aware workflow implementation.
See docs/zone-aware-implementation-tasks.md Task 6.
1. Create tests/test_zone_utils.py with unit tests
2. Update docs/intact-pyslm-howto.md with new example
3. Add docstrings and cleanup
```
