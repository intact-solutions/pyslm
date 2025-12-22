"""
Test zone-aware island workflow with block + cone geometry.

Geometry setup:
- block_original.stl: The original geometry to hatch
- cone_inside.stl: Zone 2 (cone region with different parameters)
- Zone 1: Default/bulk (original minus cone region)

Workflow:
1. Hatch the ORIGINAL (block) with groupIslands=True
2. Slice cone zone at same Z to get zone polygons
3. Classify islands: cone region -> zone2, rest -> zone1 (bulk)
4. Create BuildStyles with zone-specific power/speed
5. Export SCODE with zone-specific parameters
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import logging
import os
from pathlib import Path
from matplotlib import pyplot as plt

import pyslm
from pyslm import hatching
from pyslm.geometry import BuildStyle, Model
from pyslm.analysis import (
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
    
    geometry_dir = Path(__file__).resolve().parent.parent / 'geometry_intact' / 'zone_aware_islands_test'
    
    original_stl = geometry_dir / 'block_original.stl'
    zone_stls = {
        'cone': geometry_dir / 'cone_inside.stl',  # Zone 2
    }
    
    # Zone 1 (bulk) is implicit - it's the original minus other zones
    # Zone priority: check cone first, then default to bulk
    zone_priority = ['cone']  # Cone takes precedence over bulk
    
    # Zone -> bid mapping
    zone_bids = {
        'bulk': 1,   # Zone 1: default/original regions
        'cone': 2,   # Zone 2: cone region
    }
    contour_bid = 10
    
    # Zone-specific laser parameters
    zone_params = {
        'bulk': {'power': 200.0, 'speed': 800.0},   # Standard parameters
        'cone': {'power': 150.0, 'speed': 500.0},   # Reduced for cone region
        'contour': {'power': 180.0, 'speed': 400.0},
    }
    
    # Zone colors for visualization
    zone_colors = {
        'bulk': 'gray',
        'cone': 'red',
        'contour': 'green',
    }
    
    # =========================================================================
    # 1. Load geometries
    # =========================================================================
    
    print(f"Loading original: {original_stl}")
    if not original_stl.exists():
        raise FileNotFoundError(f"Original STL not found: {original_stl}")
    
    original_part = pyslm.Part('original')
    original_part.setGeometry(str(original_stl))
    original_part.dropToPlatform()
    
    bbox = original_part.boundingBox
    print(f"Original bounds: X[{bbox[0]:.2f}, {bbox[3]:.2f}] Y[{bbox[1]:.2f}, {bbox[4]:.2f}] Z[{bbox[2]:.2f}, {bbox[5]:.2f}]")
    
    # Load zone parts
    zone_parts = {}
    for zone_name, stl_path in zone_stls.items():
        if stl_path.exists():
            part = pyslm.Part(zone_name)
            part.setGeometry(str(stl_path))
            part.dropToPlatform()
            zone_parts[zone_name] = part
            zb = part.boundingBox
            print(f"Zone '{zone_name}' bounds: X[{zb[0]:.2f}, {zb[3]:.2f}] Y[{zb[1]:.2f}, {zb[4]:.2f}] Z[{zb[2]:.2f}, {zb[5]:.2f}]")
        else:
            print(f"Warning: Zone STL not found: {stl_path}")
    
    # =========================================================================
    # 2. Setup hatcher
    # =========================================================================
    
    hatcher = hatching.IslandHatcher()
    hatcher.groupIslands = True
    hatcher.islandWidth = 2.0
    hatcher.islandOverlap = 0.05
    hatcher.hatchAngle = 45
    hatcher.volumeOffsetHatch = 0.05
    hatcher.spotCompensation = 0.0
    hatcher.numInnerContours = 0
    hatcher.numOuterContours = 1
    hatcher.hatchSortMethod = hatching.AlternateSort()
    
    # =========================================================================
    # 3. Process at a test Z height
    # =========================================================================
    
    # Pick Z in middle of geometry
    z = 7
    print(f"\n--- Processing layer at z={z:.2f} ---")
    
    geom_slice = original_part.getVectorSlice(z)
    if not geom_slice:
        print(f"No geometry at z={z}")
        return
    
    layer = hatcher.hatch(geom_slice)
    print(f"Hatched: {len(layer.geometry)} geometry objects")
    
    # =========================================================================
    # 4. Build zone polygons and classify
    # =========================================================================
    
    zone_polys = build_zone_polygons(zone_parts, z)
    print(f"Zone polygons at z={z}: {list(zone_polys.keys())}")
    
    classify_layer_geometry(
        layer,
        zone_polys,
        zone_bids,
        contour_bid=contour_bid,
        default_zone='bulk',
        priority=zone_priority,
    )
    
    # Assign mid for BuildStyle resolution
    for geom in layer.geometry:
        geom.mid = 1
    
    stats = get_zone_statistics(layer)
    print(f"Zone statistics: {stats['by_zone']}")
    print(f"Type statistics: {stats['by_type']}")
    
    # =========================================================================
    # 5. Create BuildStyles
    # =========================================================================
    
    model = Model()
    model.mid = 1
    
    for zone_name, bid in zone_bids.items():
        bs = BuildStyle()
        bs.bid = bid
        bs.laserPower = zone_params[zone_name]['power']
        bs.laserSpeed = zone_params[zone_name]['speed']
        bs.jumpSpeed = 5000.0
        model.buildStyles.append(bs)
    
    bs_contour = BuildStyle()
    bs_contour.bid = contour_bid
    bs_contour.laserPower = zone_params['contour']['power']
    bs_contour.laserSpeed = zone_params['contour']['speed']
    bs_contour.jumpSpeed = 5000.0
    model.buildStyles.append(bs_contour)
    
    models = [model]
    
    # =========================================================================
    # 6. Export SCODE
    # =========================================================================
    
    outdir = Path(__file__).resolve().parent
    scode_path = outdir / f"test_zone_aware_z{z:.1f}.scode"
    
    n_islands = write_layer_island_info_scode(layer, models, z, str(scode_path), island_index_base=0)
    print(f"\nSCODE exported: {n_islands} islands to {scode_path}")
    
    # Verify power values
    powers_seen = {}
    with open(scode_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                power = float(parts[5])
                powers_seen[power] = powers_seen.get(power, 0) + 1
    
    print(f"Power values in SCODE: {dict(sorted(powers_seen.items()))}")
    
    # =========================================================================
    # 7. Visualize
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    seen_zones = set()
    for geom in layer.geometry:
        coords = geom.coords
        if len(coords) == 0:
            continue
        
        zone_name = getattr(geom, 'zoneName', 'bulk')
        color = zone_colors.get(zone_name, 'black')
        seen_zones.add(zone_name)
        
        if getattr(geom, 'subType', '') == 'island':
            coords = coords.reshape(-1, 2, 2)
            for line in coords:
                ax.plot(line[:, 0], line[:, 1], '-', color=color, linewidth=0.5)
        else:
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color, linewidth=0.8)
    
    # Plot zone boundaries
    for zone_name, mpoly in zone_polys.items():
        for poly in mpoly.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, '--', color=zone_colors.get(zone_name, 'black'), 
                    linewidth=2, label=f'{zone_name} boundary')
    
    # Legend
    for zone_name in sorted(seen_zones):
        color = zone_colors.get(zone_name, 'black')
        ax.plot([], [], '-', color=color, label=f'{zone_name} hatches', linewidth=2)
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_title(f'Zone-Aware Islands Test (z={z:.2f})')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print('\n' + '='*60)
    print(f'Zone-Aware Islands Test Summary')
    print('='*60)
    print(f"Layer Z: {z:.2f}")
    print(f"Total geometry: {stats['total']}")
    print("\nBy Zone:")
    for zn, count in sorted(stats['by_zone'].items()):
        params = zone_params.get(zn, {})
        power = params.get('power', '?')
        speed = params.get('speed', '?')
        print(f"  {zn:12s}: {count:4d} islands  ({power}W @ {speed} mm/s)")
    print(f"\nSCODE: {scode_path}")


if __name__ == '__main__':
    main()
