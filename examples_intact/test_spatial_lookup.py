import sys, pathlib
from pathlib import Path

# Ensure local repo import without needing external PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyslm as slm
import pyslm.visualise as slm_visualise
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import (
    IslandIndex,
    build_island_index,
    get_island_geometries,
)


def main():
    # Build the same simple part/slice used in other examples
    solidPart = slm.Part("inversePyramid")
    solidPart.setGeometry("models/frameGuide.stl")
    solidPart.dropToPlatform()

    # Basic transforms matching example_island_grouping
    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 2.0
    solidPart.rotation = [0, 0.0, np.pi]

    z = 14.99
    geomSlice = solidPart.getVectorSlice(z)

    # Configure IslandHatcher with per-island grouping
    myHatcher = hatching.IslandHatcher()
    myHatcher.islandWidth = 5.0
    myHatcher.islandOverlap = -0.1
    myHatcher.groupIslands = True
    myHatcher.hatchAngle = 10
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSortMethod = hatching.AlternateSort()

    print("Hatching slice …")
    layer = myHatcher.hatch(geomSlice)
    islands = get_island_geometries(layer)
    print(f"Islands emitted: {len(islands)}")
    if not islands:
        print("No islands found; check z within part bbox or island parameters.")
        return

    # Stage 2: Build index and test queries
    neighbor_radius = 0.8 * myHatcher.islandWidth
    index = build_island_index(layer, neighbor_radius)

    # Choose the 20th island (1-based: 20 -> 0-based index 19), or last if fewer
    owner = islands[22] if len(islands) >= 22 else islands[-1]
    poly = getattr(owner, "boundaryPoly", None)
    if poly is None:
        print("Selected island has no boundaryPoly; cannot run spatial tests.")
        return

    # Use a robust interior point for lookup (always inside the polygon)
    owner_point = poly.representative_point().coords[0]

    if owner_point is None:
        print("No island with boundaryPoly found; cannot run spatial tests.")
        return

    # Test point-to-island mapping
    ox, oy = owner_point
    resolved = index.find_island_at_point(ox, oy)
    same_obj = (resolved is owner)
    owner_pos = getattr(owner, "posId", None)
    owner_idx = islands.index(owner)
    print(f"Selected island index={owner_idx} posId={owner_pos}")
    print(f"Point→island lookup: interior_point=({ox:.3f},{oy:.3f}) matched owner: {same_obj}")

    # Test neighbor discovery
    neighbors = index.neighbors_for_island(owner)
    # Print a compact summary: count and up to first 5 posIds if available
    def posid(g):
        return getattr(g, "posId", None)

    neighbor_ids = [posid(g) for g in neighbors[:5]]
    print(f"Neighbors within R={neighbor_radius:.2f} mm: count={len(neighbors)} sample={neighbor_ids}")

    # Visualization (similar to test_island_grouping.py)
    fig, ax = plt.subplots()
    ax.axis('equal')

    # Plot slice boundary for context
    try:
        slm_visualise.plotPolygon(geomSlice, handle=(fig, ax), lineColor='k', lineWidth=0.6)
    except Exception:
        pass

    # Prepare sets for quick membership checks
    neighbor_set = set(neighbors)

    # Draw island outlines color-coded by sequence and annotate index (small grey font)
    cmap = mpl.colormaps.get_cmap('coolwarm')
    num_islands = len(islands)
    owner_color = '#66c2a5cc'   # light teal with alpha
    neigh_color = '#ffcc80cc'   # light orange with alpha

    for idx, gi in enumerate(islands, start=1):
        poly = getattr(gi, 'boundaryPoly', None)
        if poly is None:
            continue
        # Outline color by sequence
        t = 0.5 if num_islands <= 1 else (idx - 1) / (num_islands - 1)
        line_color = cmap(t)

        x, y = poly.exterior.xy
        # Fill owner and neighbors with light color
        if gi is owner:
            slm_visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), plotFilled=True, lineColor=line_color, fillColor=owner_color, lineWidth=1.2)
        elif gi in neighbor_set:
            slm_visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), plotFilled=True, lineColor=line_color, fillColor=neigh_color, lineWidth=1.0)
        else:
            slm_visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), lineColor=line_color, lineWidth=0.9)

        # Small sequence label at centroid
        cx, cy = poly.centroid.coords[0]
        ax.text(cx, cy, str(idx), color='#666666', fontsize=4, ha='center', va='center')

    # Mark the query point
    ox, oy = owner_point
    ax.plot([ox], [oy], marker='o', markersize=3, color='black')

    plt.title('Owner Island and Neighbors (no scan paths)')
    plt.show()


if __name__ == "__main__":
    main()