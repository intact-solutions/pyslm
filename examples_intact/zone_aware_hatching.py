"""
Zone-aware island hatching example.

Demonstrates the zone-aware workflow:
1. Loading original geometry + zone STLs
2. Hatching the ORIGINAL geometry with per-island output (groupIslands=True)
3. Classifying islands by zone (centroid rule) - assigns bid per zone
4. Creating BuildStyles per zone with different power/speed
5. Exporting to SCODE with zone-specific parameters
6. Visualizing with zone-colored hatches

This approach differs from multi_infill_strategy.py in that islands are
generated globally on the original geometry, then classified by zone,
rather than hatching each zone separately.

The key insight: classify_layer_geometry() sets geom.bid per zone, and
SCODE export resolves BuildStyle by bid, so zone-specific power/speed
automatically appears in the output without needing a separate zone column.
"""

import numpy as np
import logging
import os
from pathlib import Path
from matplotlib import pyplot as plt

import pyslm
import pyslm.visualise
import pyslm.analysis
from pyslm import hatching
from pyslm.geometry import BuildStyle, Model
from pyslm.analysis import (
    build_zone_polygons,
    classify_layer_geometry,
    get_zone_statistics,
)
from pyslm.analysis.export_scode import write_layer_island_info_scode


def main():
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Path to zone STL files (same as multi_infill_strategy.py)
    base_path = r'C:\Users\kumar\source_local\process_zones\tests\block\rotated\results'
    
    # We'll use the bulk zone as the "original" geometry for hatching
    # In a real workflow, you'd have a separate original.stl
    original_stl = os.path.join(base_path, 'bulk_zone.stl')
    
    zone_stl_paths = {
        'bulk': os.path.join(base_path, 'bulk_zone.stl'),
        'overhang': os.path.join(base_path, 'overhang_zone.stl'),
        'boundary': os.path.join(base_path, 'boundary_zone.stl'),
    }
    
    # Layer height
    z = 2.0
    
    # Zone -> BuildStyle bid mapping (used by classify_layer_geometry)
    zone_bids = {
        'bulk': 1,
        'overhang': 2,
        'boundary': 3,
    }
    contour_bid = 10
    
    # Zone-specific laser parameters (power in W, speed in mm/s)
    zone_params = {
        'bulk':     {'power': 200.0, 'speed': 800.0},   # Standard high-speed
        'overhang': {'power': 150.0, 'speed': 600.0},   # Reduced for unsupported
        'boundary': {'power': 180.0, 'speed': 700.0},   # Fine surface finish
        'contour':  {'power': 180.0, 'speed': 400.0},   # Slow contours
    }
    
    # Zone -> color for visualization
    zone_colors = {
        'bulk': 'gray',
        'overhang': 'red',
        'boundary': 'blue',
        'contour': 'green',
        'unknown': 'black',
    }
    
    # Priority order for zone classification (first match wins for overlapping zones)
    zone_priority = ['overhang', 'boundary', 'bulk']
    
    # =========================================================================
    # 1. Load geometries
    # =========================================================================
    
    # Check original geometry exists
    if not os.path.exists(original_stl):
        raise FileNotFoundError(f"Original STL not found: {original_stl}")
    
    original_part = pyslm.Part('original')
    original_part.setGeometry(original_stl)
    logging.info(f"Loaded original geometry: {original_stl}")
    
    # Load zone Parts
    zone_parts = {}
    for zone_name, stl_path in zone_stl_paths.items():
        if os.path.exists(stl_path):
            part = pyslm.Part(zone_name)
            part.setGeometry(stl_path)
            zone_parts[zone_name] = part
            logging.info(f"Loaded zone '{zone_name}': {stl_path}")
        else:
            logging.warning(f"Zone STL not found, skipping: {stl_path}")
    
    # =========================================================================
    # 2. Setup hatcher with groupIslands=True
    # =========================================================================
    
    hatcher = hatching.IslandHatcher()
    hatcher.groupIslands = True  # Key setting for per-island output
    hatcher.islandWidth = 0.5
    hatcher.islandOverlap = 0.01
    hatcher.hatchAngle = 10
    hatcher.volumeOffsetHatch = -0.05
    hatcher.spotCompensation = 0.0
    hatcher.numInnerContours = 1
    hatcher.numOuterContours = 1
    hatcher.hatchSortMethod = hatching.AlternateSort()
    
    # =========================================================================
    # 3. Hatch original geometry
    # =========================================================================
    
    geom_slice = original_part.getVectorSlice(z, simplificationFactor=0.1)
    
    if not geom_slice:
        logging.error(f"No geometry at z={z}")
        return
    
    layer = hatcher.hatch(geom_slice)
    logging.info(f"Hatched layer at z={z}: {len(layer.geometry)} geometry objects")
    
    # =========================================================================
    # 4. Build zone polygons at Z
    # =========================================================================
    
    zone_polys = build_zone_polygons(zone_parts, z)
    logging.info(f"Zone polygons at z={z}: {list(zone_polys.keys())}")
    
    # =========================================================================
    # 5. Classify layer geometry
    # =========================================================================
    
    classify_layer_geometry(
        layer,
        zone_polys,
        zone_bids,
        contour_bid=contour_bid,
        default_zone='bulk',
        priority=zone_priority,
    )
    
    # Assign mid to all geometry (required for BuildStyle resolution)
    for geom in layer.geometry:
        geom.mid = 1
    
    # Get statistics
    stats = get_zone_statistics(layer)
    logging.info(f"Zone statistics: {stats['by_zone']}")
    logging.info(f"Type statistics: {stats['by_type']}")
    
    # =========================================================================
    # 6. Create BuildStyles per zone (for SCODE export)
    # =========================================================================
    
    model = Model()
    model.mid = 1
    
    # Create a BuildStyle for each zone with zone-specific parameters
    for zone_name, bid in zone_bids.items():
        bs = BuildStyle()
        bs.bid = bid
        bs.laserPower = zone_params[zone_name]['power']
        bs.laserSpeed = zone_params[zone_name]['speed']
        bs.jumpSpeed = 5000.0
        model.buildStyles.append(bs)
    
    # Contour BuildStyle
    bs_contour = BuildStyle()
    bs_contour.bid = contour_bid
    bs_contour.laserPower = zone_params['contour']['power']
    bs_contour.laserSpeed = zone_params['contour']['speed']
    bs_contour.jumpSpeed = 5000.0
    model.buildStyles.append(bs_contour)
    
    models = [model]
    logging.info(f"Created {len(model.buildStyles)} BuildStyles")
    
    # =========================================================================
    # 7. Export to SCODE (power/speed will reflect zone-specific parameters)
    # =========================================================================
    
    outdir = Path(__file__).resolve().parent
    scode_path = outdir / f"zone_aware_layer_z{z:.1f}.scode"
    
    n_islands = write_layer_island_info_scode(layer, models, z, str(scode_path), island_index_base=0)
    logging.info(f"Wrote {n_islands} islands to {scode_path}")
    
    # =========================================================================
    # 8. Visualize with zone colors
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Track which zones we've seen for legend
    seen_zones = set()
    
    for geom in layer.geometry:
        coords = geom.coords
        if len(coords) == 0:
            continue
        
        zone_name = getattr(geom, 'zoneName', 'unknown')
        color = zone_colors.get(zone_name, 'black')
        seen_zones.add(zone_name)
        
        # Check if it's hatch geometry (pairs of points) or contour (path)
        if hasattr(geom, 'subType') and geom.subType == 'island':
            # Hatch vectors: reshape to line segments
            coords = coords.reshape(-1, 2, 2)
            for line in coords:
                ax.plot(line[:, 0], line[:, 1], '-', color=color, linewidth=0.5)
        else:
            # Contour: single path
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color, linewidth=0.8)
    
    # Add legend
    for zone_name in sorted(seen_zones):
        color = zone_colors.get(zone_name, 'black')
        ax.plot([], [], '-', color=color, label=zone_name.capitalize(), linewidth=2)
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_title(f'Zone-Aware Hatching at z={z}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # 7. Print summary stats
    # =========================================================================
    
    print('\n' + '='*60)
    print(f'Zone-Aware Hatching Summary (z={z})')
    print('='*60)
    
    print(f"\nTotal geometry objects: {stats['total']}")
    
    print("\nBy Zone:")
    for zone_name, count in sorted(stats['by_zone'].items()):
        print(f"  {zone_name:12s}: {count:4d}")
    
    print("\nBy Type:")
    for sub_type, count in sorted(stats['by_type'].items()):
        print(f"  {sub_type:12s}: {count:4d}")
    
    print("\nPath Analysis:")
    print(f"  Total path distance: {pyslm.analysis.getLayerPathLength(layer):.1f} mm")
    print(f"  Total jump distance: {pyslm.analysis.getLayerJumpLength(layer):.1f} mm")
    
    print("\nZone Parameters (bid -> power/speed):")
    for zone_name, bid in zone_bids.items():
        p = zone_params[zone_name]
        print(f"  {zone_name:12s} (bid={bid}): {p['power']:.0f}W @ {p['speed']:.0f} mm/s")
    p = zone_params['contour']
    print(f"  {'contour':12s} (bid={contour_bid}): {p['power']:.0f}W @ {p['speed']:.0f} mm/s")
    
    print(f"\nSCODE Export:")
    print(f"  File: {scode_path}")
    print(f"  Islands written: {n_islands}")
    print("  Columns: x1 y1 x2 y2 z power eq_speed total_time island-idx")
    print("  (power reflects zone-specific BuildStyle)")


if __name__ == '__main__':
    main()
