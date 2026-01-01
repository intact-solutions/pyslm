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
        try:
            if mpoly.contains(pt):
                return zone_name
        except Exception:
            # TopologyException - try fixing with buffer(0)
            try:
                if mpoly.buffer(0).contains(pt):
                    return zone_name
            except Exception:
                pass
    
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
            try:
                if zone_polys[zone_name].contains(pt):
                    return zone_name
            except Exception:
                # TopologyException - try fixing with buffer(0)
                try:
                    if zone_polys[zone_name].buffer(0).contains(pt):
                        return zone_name
                except Exception:
                    pass
    
    # Check any zones not in priority list
    for zone_name, mpoly in zone_polys.items():
        if zone_name not in priority:
            try:
                if mpoly.contains(pt):
                    return zone_name
            except Exception:
                try:
                    if mpoly.buffer(0).contains(pt):
                        return zone_name
                except Exception:
                    pass
    
    return default


# ---------------------------------------------------------------------------
# Island/Layer Classification Functions (Task 2)
# ---------------------------------------------------------------------------

def classify_island_zone(island_geom: Any,
                         zone_polys: Dict[str, MultiPolygon],
                         priority: Optional[List[str]] = None,
                         default: str = 'bulk') -> str:
    """
    Classify an island HatchGeometry by which zone contains its centroid.
    
    Args:
        island_geom: HatchGeometry with boundaryPoly attribute
        zone_polys: dict from build_zone_polygons()
        priority: optional list of zone names to check in order (first match wins)
        default: fallback zone if centroid not in any zone
    
    Returns:
        zone name string
    
    Note:
        If island_geom lacks boundaryPoly, falls back to computing centroid
        from coords if available, otherwise returns default.
    """
    if Point is None:
        raise ImportError("shapely is required for zone utilities")
    
    # Get centroid from boundaryPoly if available
    boundary_poly = getattr(island_geom, 'boundaryPoly', None)
    
    if boundary_poly is not None and hasattr(boundary_poly, 'centroid'):
        centroid = boundary_poly.centroid
        x, y = centroid.x, centroid.y
    else:
        # Fallback: compute centroid from coords
        coords = getattr(island_geom, 'coords', None)
        if coords is not None and len(coords) > 0:
            x, y = float(coords[:, 0].mean()), float(coords[:, 1].mean())
        else:
            return default
    
    # Use priority-based lookup if specified
    if priority:
        return find_zone_for_point_with_priority(x, y, zone_polys, priority, default)
    else:
        return find_zone_for_point(x, y, zone_polys, default)


def classify_layer_geometry(layer: Any,
                            zone_polys: Dict[str, MultiPolygon],
                            zone_buildstyles: Dict[str, int],
                            contour_bid: int = 10,
                            default_zone: str = 'bulk',
                            priority: Optional[List[str]] = None) -> None:
    """
    Classify all geometry in a layer and assign zoneName + bid attributes.
    Modifies layer.geometry in place.
    
    Args:
        layer: Layer with geometry (islands have subType='island', 
               contours have 'outer'/'inner')
        zone_polys: dict from build_zone_polygons()
        zone_buildstyles: dict mapping zone_name -> bid
        contour_bid: bid to assign to all contours
        default_zone: fallback for islands outside all zones
        priority: optional list of zone names to check in order
    
    Side effects:
        Sets geom.zoneName and geom.bid for each geometry in layer.geometry
    
    Example:
        >>> zone_buildstyles = {'bulk': 1, 'overhang': 2, 'boundary': 3}
        >>> classify_layer_geometry(layer, zone_polys, zone_buildstyles,
        ...                         contour_bid=10, default_zone='bulk')
        >>> # Now each geom in layer.geometry has .zoneName and .bid
    """
    geometry_list = getattr(layer, 'geometry', [])
    
    for geom in geometry_list:
        sub_type = getattr(geom, 'subType', '')
        
        if sub_type == 'island':
            # Classify island by centroid
            zone_name = classify_island_zone(geom, zone_polys, priority, default_zone)
            geom.zoneName = zone_name
            geom.bid = zone_buildstyles.get(zone_name, zone_buildstyles.get(default_zone, 1))
            
        elif sub_type in ('outer', 'inner'):
            # Contours get dedicated parameters
            geom.zoneName = 'contour'
            geom.bid = contour_bid
            
        else:
            # Unknown geometry type - try to classify by coords centroid
            coords = getattr(geom, 'coords', None)
            if coords is not None and len(coords) > 0:
                x, y = float(coords[:, 0].mean()), float(coords[:, 1].mean())
                if priority:
                    zone_name = find_zone_for_point_with_priority(x, y, zone_polys, priority, default_zone)
                else:
                    zone_name = find_zone_for_point(x, y, zone_polys, default_zone)
                geom.zoneName = zone_name
                geom.bid = zone_buildstyles.get(zone_name, zone_buildstyles.get(default_zone, 1))
            else:
                # No coords, assign default
                geom.zoneName = default_zone
                geom.bid = zone_buildstyles.get(default_zone, 1)


def get_zone_statistics(layer: Any) -> Dict[str, Dict[str, int]]:
    """
    Compute statistics about zone assignments in a layer.
    
    Args:
        layer: Layer with classified geometry (each geom has .zoneName)
    
    Returns:
        dict with:
        - 'by_zone': {zone_name: count}
        - 'by_type': {subType: count}
        - 'total': total geometry count
    """
    by_zone: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    
    geometry_list = getattr(layer, 'geometry', [])
    
    for geom in geometry_list:
        zone_name = getattr(geom, 'zoneName', 'unknown')
        sub_type = getattr(geom, 'subType', 'unknown')
        
        by_zone[zone_name] = by_zone.get(zone_name, 0) + 1
        by_type[sub_type] = by_type.get(sub_type, 0) + 1
    
    return {
        'by_zone': by_zone,
        'by_type': by_type,
        'total': len(geometry_list),
    }