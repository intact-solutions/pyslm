"""
Zone utilities for the zone-aware island workflow.

This module provides helper functions to:
- Slice zone Part geometries at a given Z height
- Build a dictionary of zone polygons for spatial queries
- Find which zone contains a given point

These utilities enable assigning different laser parameters to islands
based on geometric zones (e.g., bulk, overhang, boundary regions).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from shapely.geometry import MultiPolygon, Point, Polygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    MultiPolygon = None  # type: ignore
    Point = None  # type: ignore
    Polygon = None  # type: ignore
    unary_union = None  # type: ignore


def slice_zone_part(zone_part: Any, z: float) -> Optional[MultiPolygon]:
    """
    Slice a zone Part at height z and return a shapely MultiPolygon.
    
    Uses Part.getVectorSlice() with returnCoordPaths=False to get shapely
    Polygons directly, then wraps them in a MultiPolygon for uniform handling.
    
    Args:
        zone_part: pyslm.Part loaded from zone STL
        z: slice height
    
    Returns:
        shapely.geometry.MultiPolygon containing all polygons at this Z,
        or None if slice is empty (Z above/below zone geometry)
    """
    if MultiPolygon is None:
        raise ImportError("shapely is required for zone utilities")
    
    # Get shapely polygons directly from the Part
    polygons = zone_part.getVectorSlice(z, returnCoordPaths=False)
    
    if not polygons:
        return None
    
    # Ensure we have a list of Polygon objects
    valid_polys = []
    for poly in polygons:
        if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
            valid_polys.append(poly)
        elif hasattr(poly, 'buffer'):
            # Try to fix invalid polygon with zero buffer
            fixed = poly.buffer(0)
            if fixed.is_valid and not fixed.is_empty:
                if isinstance(fixed, Polygon):
                    valid_polys.append(fixed)
                elif isinstance(fixed, MultiPolygon):
                    valid_polys.extend(list(fixed.geoms))
    
    if not valid_polys:
        return None
    
    # Wrap in MultiPolygon for uniform handling
    return MultiPolygon(valid_polys)


def build_zone_polygons(zone_parts: Dict[str, Any], z: float) -> Dict[str, MultiPolygon]:
    """
    Slice all zone Parts at height z.
    
    Args:
        zone_parts: dict mapping zone_name -> Part
        z: slice height
    
    Returns:
        dict mapping zone_name -> MultiPolygon (excludes zones with empty slices)
    
    Example:
        >>> zone_parts = {
        ...     'bulk': bulk_part,
        ...     'overhang': overhang_part,
        ...     'boundary': boundary_part,
        ... }
        >>> zone_polys = build_zone_polygons(zone_parts, z=2.0)
        >>> # zone_polys will only include zones that have geometry at z=2.0
    """
    zone_polys: Dict[str, MultiPolygon] = {}
    
    for zone_name, zone_part in zone_parts.items():
        mpoly = slice_zone_part(zone_part, z)
        if mpoly is not None:
            zone_polys[zone_name] = mpoly
    
    return zone_polys


def find_zone_for_point(x: float, y: float, 
                        zone_polys: Dict[str, MultiPolygon],
                        default: str = 'bulk') -> str:
    """
    Find which zone contains the point (x, y).
    
    Iterates through zone polygons and returns the first zone that contains
    the query point. If the point is not inside any zone, returns the default.
    
    Args:
        x, y: query point coordinates
        zone_polys: dict from build_zone_polygons()
        default: fallback zone name if point not in any zone
    
    Returns:
        zone name string
    
    Note:
        If zones overlap, the first matching zone (in dict iteration order) is
        returned. For Python 3.7+, this is insertion order. If deterministic
        priority is needed, use an OrderedDict or sort zone_polys.keys().
    """
    if Point is None:
        raise ImportError("shapely is required for zone utilities")
    
    pt = Point(x, y)
    
    for zone_name, mpoly in zone_polys.items():
        if mpoly.contains(pt):
            return zone_name
    
    return default


def find_zone_for_point_with_priority(x: float, y: float,
                                       zone_polys: Dict[str, MultiPolygon],
                                       priority: List[str],
                                       default: str = 'bulk') -> str:
    """
    Find which zone contains the point (x, y), checking zones in priority order.
    
    This variant allows specifying the order in which zones are checked,
    useful when zones may overlap and a specific zone should take precedence.
    
    Args:
        x, y: query point coordinates
        zone_polys: dict from build_zone_polygons()
        priority: list of zone names in order of priority (first match wins)
        default: fallback zone name if point not in any zone
    
    Returns:
        zone name string
    
    Example:
        >>> # Check overhang first, then boundary, then bulk
        >>> zone = find_zone_for_point_with_priority(
        ...     x, y, zone_polys,
        ...     priority=['overhang', 'boundary', 'bulk'],
        ...     default='bulk'
        ... )
    """
    if Point is None:
        raise ImportError("shapely is required for zone utilities")
    
    pt = Point(x, y)
    
    for zone_name in priority:
        if zone_name in zone_polys:
            if zone_polys[zone_name].contains(pt):
                return zone_name
    
    # Check any zones not in priority list
    for zone_name, mpoly in zone_polys.items():
        if zone_name not in priority:
            if mpoly.contains(pt):
                return zone_name
    
    return default
