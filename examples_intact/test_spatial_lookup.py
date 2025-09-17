import sys, pathlib
from pathlib import Path

# Ensure local repo import without needing external PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import (
    IslandIndex,
    build_island_index,
    get_island_geometries,
)


def main():
    # Build the same simple part/slice used in other examples
    solidPart = pyslm.Part("inversePyramid")
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


if __name__ == "__main__":
    main()