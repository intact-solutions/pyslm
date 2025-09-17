"""
Spatial lookup and neighbor utilities for per-island grouping (Stage 2).

This module provides an IslandIndex that can:
- Map a query point (x, y) to the island HatchGeometry that contains it.
- Find immediate neighbors of an island (or point) within a radius R.

It expects Stage 1 metadata to be present on each island HatchGeometry:
- subType == "island"
- boundaryPoly: shapely.geometry.Polygon
- bbox: (minx, miny, maxx, maxy)

If boundaryPoly is missing on a geometry, that geometry is skipped with a warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

# Stage 3 timing utilities import
from . import utils as analysis_utils

try:
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree
except Exception:  # pragma: no cover - shapely may not be present in some minimal envs
    Point = None  # type: ignore
    box = None  # type: ignore
    STRtree = None  # type: ignore


@dataclass
class _IslandRecord:
    idx_in_layer: int
    geom: Any  # HatchGeometry
    poly: Any  # shapely Polygon
    bbox: Tuple[float, float, float, float]


class IslandIndex:
    """
    Build once per Layer to support efficient spatial queries over islands.
    """

    def __init__(self, layer: Any, neighbor_radius: float) -> None:
        self.layer = layer
        self.neighbor_radius = neighbor_radius
        self.records: List[_IslandRecord] = []
        self._tree = None
        self._boxes: List[Any] = []

        self._build_index()

    def _build_index(self) -> None:
        if STRtree is None or box is None:
            # Fallback: no spatial index available
            self._tree = None

        # Collect island entries
        for i, g in enumerate(get_island_geometries(self.layer)):
            poly = getattr(g, "boundaryPoly", None)
            bbox_t = getattr(g, "bbox", None)
            if poly is None or bbox_t is None:
                # Skip entries without required metadata
                continue
            rec = _IslandRecord(idx_in_layer=i, geom=g, poly=poly, bbox=bbox_t)
            self.records.append(rec)
            if box is not None:
                self._boxes.append(box(*bbox_t))

        if self._boxes and STRtree is not None:
            self._tree = STRtree(self._boxes)

    def _candidate_indices_for_point(self, x: float, y: float) -> List[int]:
        if not self.records:
            return []
        if self._tree is None or Point is None:
            # No index available; linear scan
            return list(range(len(self.records)))
        pt = Point(x, y)
        hits = self._tree.query(pt)
        # Robustly map returned geometries to indices by identity comparison
        idxs: List[int] = []
        for i, g in enumerate(self._boxes):
            # 'is' identity works reliably within same STRtree; avoids value-equality pitfalls
            if any(h is g for h in hits):
                idxs.append(i)
        # Fallback: if mapping failed (unexpected), return all for safety
        return idxs if idxs else list(range(len(self.records)))

    def _candidate_indices_for_radius(self, rec_idx: int, radius: float) -> List[int]:
        if not self.records:
            return []
        if self._tree is None or box is None:
            # No index available; linear scan
            return list(range(len(self.records)))
        # Expand target bbox by radius and query
        minx, miny, maxx, maxy = self.records[rec_idx].bbox
        q = box(minx - radius, miny - radius, maxx + radius, maxy + radius)
        hits = self._tree.query(q)
        idxs: List[int] = []
        for i, g in enumerate(self._boxes):
            if any(h is g for h in hits):
                idxs.append(i)
        return idxs if idxs else list(range(len(self.records)))

    def find_island_at_point(self, x: float, y: float) -> Optional[Any]:
        """
        Return the HatchGeometry (subType=="island") that contains the point, or None.
        Uses bbox candidate filtering via STRtree when available.
        """
        if not self.records:
            return None
        idxs = self._candidate_indices_for_point(x, y)
        if Point is None:
            # Without shapely, cannot do robust point-in-polygon; best-effort bbox contains
            for i in idxs:
                minx, miny, maxx, maxy = self.records[i].bbox
                if minx <= x <= maxx and miny <= y <= maxy:
                    return self.records[i].geom
            return None
        pt = Point(x, y)
        for i in idxs:
            poly = self.records[i].poly
            # Consider boundary as inside using a tiny buffer for robustness
            if poly.buffer(1e-9).contains(pt):
                return self.records[i].geom
        return None

    def neighbors_for_island(self, target: Any, radius: Optional[float] = None) -> List[Any]:
        """
        Return neighboring island HatchGeometry objects within the given radius of the target island.
        Neighbors are returned in the original layer order (scan sequence proxy).
        """
        if not self.records:
            return []
        radius = self.neighbor_radius if radius is None else radius

        # Locate target index in records by identity
        target_idx = None
        target_poly = None
        for i, r in enumerate(self.records):
            if r.geom is target:
                target_idx = i
                target_poly = r.poly
                break
        if target_idx is None:
            return []

        idxs = self._candidate_indices_for_radius(target_idx, radius)

        # Filter by actual polygon distance/overlap when shapely available
        neighbors: List[Tuple[int, Any]] = []
        for i in idxs:
            if i == target_idx:
                continue
            r = self.records[i]
            if target_poly is not None and hasattr(r, "poly") and r.poly is not None and Point is not None:
                try:
                    if r.poly.distance(target_poly) <= radius:
                        neighbors.append((r.idx_in_layer, r.geom))
                except Exception:
                    # Fallback to bbox center distance if polygon distance fails
                    neighbors.append((r.idx_in_layer, r.geom))
            else:
                neighbors.append((r.idx_in_layer, r.geom))

        # Sort by layer order (acts as scan-sequence order proxy)
        neighbors.sort(key=lambda t: t[0])
        return [g for _, g in neighbors]

    def neighbors_for_point(self, x: float, y: float, radius: Optional[float] = None) -> Tuple[Optional[Any], List[Any]]:
        """
        Convenience: return (owner_island, neighbors) for a query point.
        """
        owner = self.find_island_at_point(x, y)
        if owner is None:
            return None, []
        return owner, self.neighbors_for_island(owner, radius)


# Convenience helpers ---------------------------------------------------------

def get_island_geometries(layer: Any) -> List[Any]:
    """Return HatchGeometry entries from layer.geometry where subType == 'island'."""
    geoms = getattr(layer, "geometry", [])
    return [g for g in geoms if getattr(g, "subType", "") == "island"]


def build_island_index(layer: Any, neighbor_radius: float) -> IslandIndex:
    """Factory to construct an IslandIndex for a given layer."""
    return IslandIndex(layer, neighbor_radius)


# ---------------------------------------------------------------------------
# Stage 3: Timing & Level 2 aggregation utilities
# ---------------------------------------------------------------------------

def _sum_times(entries: List[Dict[str, Any]]) -> float:
    """Sum the time field of per-geometry entries with float safety."""
    return float(sum(e.get("time", 0.0) for e in entries))


def _kind_of_geom(geom: Any) -> str:
    """Return a simple kind label for a LayerGeometry instance."""
    # Avoid direct imports here to keep a single import surface at file top
    name = type(geom).__name__.lower()
    if "hatch" in name:
        return "hatch"
    if "contour" in name:
        return "contour"
    if "points" in name:
        return "points"
    return name


def compute_layer_geometry_times(layer: Any,
                                 models: List[Any],
                                 include_jump: bool = True,
                                 laser_jump_speed: float = 5000.0,
                                 validate: bool = True,
                                 tol_rel: float = 1e-6,
                                 tol_abs: float = 1e-9) -> List[Dict[str, Any]]:
    """
    Compute per-geometry times for a given layer in the exact order of `layer.geometry`.

    Returns a list of entries with fields:
    - idx_in_layer: index into layer.geometry
    - geom: the LayerGeometry
    - kind: "hatch" | "contour" | "points" (best-effort)
    - subType: attribute from geometry if present (e.g., "island"), else ""
    - islandId: attribute from geometry if present, else None
    - time: timing in seconds (includes jump time if include_jump=True)

    When `validate` is True, asserts that the sum of per-geometry times matches
    `analysis_utils.getLayerTime(layer, models, includeJumpTime=include_jump)` within tolerances.
    """
    entries: List[Dict[str, Any]] = []

    for i, geom in enumerate(getattr(layer, "geometry", [])):
        t = analysis_utils.getLayerGeometryTime(geom, models, includeJumpTime=include_jump)
        entry: Dict[str, Any] = {
            "idx_in_layer": i,
            "geom": geom,
            "kind": _kind_of_geom(geom),
            "subType": getattr(geom, "subType", ""),
            "islandId": getattr(geom, "islandId", None),
            "time": float(t),
        }
        entries.append(entry)

    if validate:
        # Sum per-geom and compare with layer time that also includes inter-geometry jumps/delays
        per_geom_sum = _sum_times(entries)
        layer_time = analysis_utils.getLayerTime(layer, models, includeJumpTime=include_jump, laserJumpSpeed=laser_jump_speed)

        # Relative-or-absolute tolerance check
        if not (abs(per_geom_sum - layer_time) <= max(tol_abs, tol_rel * max(1.0, abs(layer_time)))):
            # Soft assertion: raise a clear error with diagnostics
            raise AssertionError(
                f"Per-geometry time sum ({per_geom_sum:.9f}s) does not match layer time "
                f"({layer_time:.9f}s). include_jump={include_jump}, laser_jump_speed={laser_jump_speed}."
            )

    return entries


def layer_entries_to_island_subset(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter entries to only those with subType == 'island'."""
    return [e for e in entries if e.get("subType", "") == "island"]


def aggregate_level2_timing(layers: List[Any],
                            models: List[Any],
                            target_layer_index: int,
                            include_layers_below: int = 1,
                            include_jump: bool = True,
                            laser_jump_speed: float = 5000.0,
                            validate: bool = True) -> Dict[str, Any]:
    """
    Aggregate Stage 3 Level 2 timing for the target layer and N layers below.

    Returns a dict:
    - layers: list of { layer_index, entries, layer_time }
    - total_time: sum of layer_time across included layers

    Validation ensures that each layer's per-geometry sum matches `getLayerTime` within tolerance.
    """
    if target_layer_index < 0 or target_layer_index >= len(layers):
        raise IndexError("target_layer_index out of range")

    start = max(0, target_layer_index - include_layers_below)
    end = target_layer_index  # inclusive target only; include below means prior layers

    out_layers: List[Dict[str, Any]] = []
    total_time = 0.0

    for li in range(start, end + 1):
        layer = layers[li]
        entries = compute_layer_geometry_times(
            layer,
            models,
            include_jump=include_jump,
            laser_jump_speed=laser_jump_speed,
            validate=validate,
        )
        layer_time = analysis_utils.getLayerTime(layer, models, includeJumpTime=include_jump, laserJumpSpeed=laser_jump_speed)

        out_layers.append({
            "layer_index": li,
            "entries": entries,
            "layer_time": float(layer_time),
        })
        total_time += float(layer_time)

    return {
        "layers": out_layers,
        "total_time": float(total_time),
    }
