"""
GE Bracket Zone-Aware Island Hatching Example

Demonstrates zone-aware island classification for the GE bracket case study
with 6 distinct zones and optimized laser speed parameters.

Zone mapping from case study:
    Zone 1: high_sensi  (High-sensitivity)   -> 2.5 mm/s optimized
    Zone 2: med_sensi   (Medium-sensitivity) -> 1.75 mm/s optimized
    Zone 3: low_sensi   (Low-sensitivity)    -> 2.5 mm/s optimized
    Zone 4: base        (Base)               -> 2.5 mm/s optimized
    Zone 5: boundary    (Boundary)           -> 2.5 mm/s optimized
    Zone 6: interface   (Interface)          -> 2.5 mm/s optimized

Workflow:
    1. Load original GE bracket geometry
    2. Load 6 zone PLY files
    3. Hatch with groupIslands=True
    4. Classify islands by centroid containment
    5. Assign zone-specific BuildStyles
    6. Export SCODE and visualize
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

import pyslm
from pyslm import hatching
from pyslm.geometry import BuildStyle, Model
from pyslm.analysis.zone_utils import (
    build_zone_polygons,
    classify_layer_geometry,
    get_zone_statistics,
)
from pyslm.analysis.export_scode import write_layer_island_info_scode


def main():
    logging.getLogger().setLevel(logging.INFO)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Path to zone geometry files
    base_path = Path(__file__).resolve().parent.parent / 'geometry_intact' / 'zone_aware_island_gebracket'
    
    # Original geometry
    original_stl = base_path / 'ge_bracket_original.stl'
    
    # Zone PLY file paths (6 zones)
    zone_ply_paths = {
        'high_sensi': base_path / 'high_sensi_zone.ply',
        'med_sensi':  base_path / 'med_sensi_zone.ply',
        'low_sensi':  base_path / 'low_sensi_zone.ply',
        'base':       base_path / 'base_zone.ply',
        'boundary':   base_path / 'boundary_zone.ply',
        'interface':  base_path / 'interface_zone.ply',
    }
    
    # Layer height to slice at
    z = 10.0  # Adjust based on bracket geometry (range: 0 to ~62mm)
    
    # Zone -> BuildStyle bid mapping
    zone_bids = {
        'high_sensi': 1,
        'med_sensi':  2,
        'low_sensi':  3,
        'base':       4,
        'boundary':   5,
        'interface':  6,
    }
    contour_bid = 10
    
    # Zone-specific laser parameters (from optimization table)
    # Power in W, speed in mm/s
    # Note: Image shows speed in mm/s for optimized values
    zone_params = {
        'high_sensi': {'power': 200.0, 'speed': 2500.0},   # Zone 1: 2.5 m/s = 2500 mm/s
        'med_sensi':  {'power': 200.0, 'speed': 1750.0},   # Zone 2: 1.75 m/s = 1750 mm/s
        'low_sensi':  {'power': 200.0, 'speed': 2500.0},   # Zone 3: 2.5 m/s
        'base':       {'power': 200.0, 'speed': 2500.0},   # Zone 4: 2.5 m/s
        'boundary':   {'power': 200.0, 'speed': 2500.0},   # Zone 5: 2.5 m/s
        'interface':  {'power': 200.0, 'speed': 2500.0},   # Zone 6: 2.5 m/s
        'contour':    {'power': 180.0, 'speed': 400.0},    # Contour parameters
    }
    
    # Zone colors for visualization
    zone_colors = {
        'high_sensi': 'red',
        'med_sensi':  'orange',
        'low_sensi':  'yellow',
        'base':       'gray',
        'boundary':   'blue',
        'interface':  'purple',
        'contour':    'green',
        'unknown':    'black',
    }
    
    # Priority order for zone classification (first match wins for overlapping zones)
    # Interface and high-sensitivity regions take precedence
    zone_priority = ['interface', 'high_sensi', 'med_sensi', 'boundary', 'low_sensi', 'base']
    
    # =========================================================================
    # 1. Load geometries
    # =========================================================================
    
    if not original_stl.exists():
        raise FileNotFoundError(f"Original STL not found: {original_stl}")
    
    original_part = pyslm.Part('ge_bracket')
    original_part.setGeometry(str(original_stl))
    logging.info(f"Loaded original geometry: {original_stl}")
    
    # Load zone Parts from PLY files
    zone_parts = {}
    for zone_name, ply_path in zone_ply_paths.items():
        if ply_path.exists():
            part = pyslm.Part(zone_name)
            part.setGeometry(str(ply_path))
            zone_parts[zone_name] = part
            logging.info(f"Loaded zone '{zone_name}': {ply_path}")
        else:
            logging.warning(f"Zone PLY not found, skipping: {ply_path}")
    
    if not zone_parts:
        raise RuntimeError("No zone geometries loaded!")
    
    # =========================================================================
    # 2. Setup hatcher with groupIslands=True
    # =========================================================================
    
    hatcher = hatching.IslandHatcher()
    hatcher.groupIslands = True  # Key setting for per-island output
    hatcher.islandWidth = 5.0
    hatcher.islandOverlap = 0.1
    hatcher.hatchAngle = 67
    hatcher.volumeOffsetHatch = -0.08
    hatcher.spotCompensation = 0.06
    hatcher.numInnerContours = 2
    hatcher.numOuterContours = 1
    hatcher.hatchSortMethod = hatching.AlternateSort()
    
    # =========================================================================
    # 3. Hatch original geometry
    # =========================================================================
    
    geom_slice = original_part.getVectorSlice(z, simplificationFactor=0.1)
    
    if not geom_slice:
        logging.error(f"No geometry at z={z}. Try a different Z height.")
        # Print bounding box to help find valid Z range
        bounds = original_part.boundingBox
        logging.info(f"Part bounding box Z range: {bounds[0][2]:.2f} to {bounds[1][2]:.2f}")
        return
    
    logging.info(f"Slice has {len(geom_slice)} polygon(s)")

    try:
        layer = hatcher.hatch(geom_slice)
    except IndexError as e:
        logging.error(f"Hatching failed at z={z}: {e}")
        logging.info("Try a different Z height or check geometry.")
        bounds = original_part.boundingBox
        logging.info(f"Part bounding box Z range: {bounds[0][2]:.2f} to {bounds[1][2]:.2f}")
        return
    logging.info(f"Hatched layer at z={z}: {len(layer.geometry)} geometry objects")
    
    # =========================================================================
    # 4. Build zone polygons at Z
    # =========================================================================
    
    zone_polys = build_zone_polygons(zone_parts, z)
    logging.info(f"Zone polygons at z={z}: {list(zone_polys.keys())}")
    
    if not zone_polys:
        logging.warning("No zone polygons at this Z height. Islands will use default zone.")
    
    # =========================================================================
    # 5. Classify layer geometry
    # =========================================================================
    
    classify_layer_geometry(
        layer,
        zone_polys,
        zone_bids,
        contour_bid=contour_bid,
        default_zone='base',  # Base is the fallback zone
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
    # 7. Export to SCODE
    # =========================================================================
    
    outdir = Path(__file__).resolve().parent
    scode_path = outdir / f"ge_bracket_zone_z{z:.1f}.scode"
    
    n_islands = write_layer_island_info_scode(layer, models, z, str(scode_path), island_index_base=0)
    logging.info(f"Wrote {n_islands} islands to {scode_path}")
    
    # =========================================================================
    # 8. Visualize with zone colors
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
                ax.plot(line[:, 0], line[:, 1], '-', color=color, linewidth=0.3)
        else:
            # Contour: single path
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color, linewidth=0.8)
    
    # Add legend
    for zone_name in sorted(seen_zones):
        color = zone_colors.get(zone_name, 'black')
        ax.plot([], [], '-', color=color, label=zone_name.replace('_', ' ').title(), linewidth=2)
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_title(f'GE Bracket Zone-Aware Hatching at z={z}mm')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = outdir / f"ge_bracket_zone_z{z:.1f}.png"
    plt.savefig(fig_path, dpi=150)
    logging.info(f"Saved visualization to {fig_path}")
    
    plt.show()
    
    # =========================================================================
    # 9. Print summary stats
    # =========================================================================
    
    print('\n' + '='*70)
    print(f'GE Bracket Zone-Aware Hatching Summary (z={z}mm)')
    print('='*70)
    
    print(f"\nTotal geometry objects: {stats['total']}")
    
    print("\nIslands by Zone:")
    for zone_name in zone_priority:
        count = stats['by_zone'].get(zone_name, 0)
        if count > 0:
            speed = zone_params[zone_name]['speed']
            print(f"  {zone_name:12s}: {count:4d} islands @ {speed:.0f} mm/s")
    
    # Any zones not in priority
    for zone_name, count in sorted(stats['by_zone'].items()):
        if zone_name not in zone_priority and zone_name != 'contour':
            print(f"  {zone_name:12s}: {count:4d}")
    
    contour_count = stats['by_zone'].get('contour', 0)
    if contour_count > 0:
        print(f"  {'contour':12s}: {contour_count:4d}")
    
    print("\nBy Geometry Type:")
    for sub_type, count in sorted(stats['by_type'].items()):
        print(f"  {sub_type:12s}: {count:4d}")
    
    print("\nPath Analysis:")
    print(f"  Total path distance: {pyslm.analysis.getLayerPathLength(layer):.1f} mm")
    print(f"  Total jump distance: {pyslm.analysis.getLayerJumpLength(layer):.1f} mm")
    
    print("\nZone Parameters (bid -> power/speed):")
    for zone_name in zone_priority:
        bid = zone_bids[zone_name]
        p = zone_params[zone_name]
        print(f"  {zone_name:12s} (bid={bid}): {p['power']:.0f}W @ {p['speed']:.0f} mm/s")
    p = zone_params['contour']
    print(f"  {'contour':12s} (bid={contour_bid}): {p['power']:.0f}W @ {p['speed']:.0f} mm/s")
    
    print(f"\nSCODE Export:")
    print(f"  File: {scode_path}")
    print(f"  Islands written: {n_islands}")


if __name__ == '__main__':
    main()
