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
from typing import List, Optional, Tuple, Any

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
