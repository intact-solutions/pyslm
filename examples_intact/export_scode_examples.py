import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import get_island_geometries, IslandIndex
from pyslm.analysis.export_scode import (
    write_neighborhood_paths_scode,
    write_layer_island_info_scode,
)


# ----------------------------
# Config
# ----------------------------
Z_TARGET = 0.001
ISLAND_WIDTH = 0.002
NEIGHBOR_RADIUS_R = 0.8 * ISLAND_WIDTH
OUTDIR = Path(__file__).resolve().parent


def build_layer(z: float):
    solidPart = pyslm.Part('inversePyramid')
    solidPart.setGeometry('ge_bracket_large_1_1.STL')
    solidPart.dropToPlatform()
    solidPart.origin[0] = 0.0
    solidPart.origin[1] = 0.0
    solidPart.scaleFactor = 0.001
    solidPart.rotation = [0, 0.0, np.pi]
    print(solidPart.boundingBox)

    geomSlice = solidPart.getVectorSlice(z)

    myHatcher = hatching.IslandHatcher()
    myHatcher.islandWidth = ISLAND_WIDTH
    myHatcher.islandOverlap = 0
    myHatcher.hatchAngle = 0
    myHatcher.volumeOffsetHatch = 0
    myHatcher.spotCompensation = 0
    myHatcher.numInnerContours = 0
    myHatcher.numOuterContours = 0
    myHatcher.hatchDistance = 1e-4
    myHatcher.hatchSortMethod = hatching.AlternateSort()
    myHatcher.groupIslands = True

    layer = myHatcher.hatch(geomSlice)
    return geomSlice, layer


def assign_model(layer):
    # Assign model/buildstyle ids to each geometry
    for g in getattr(layer, 'geometry', []):
        g.mid = 1
        g.bid = 1

    # Minimal BuildStyle/Model for timing/exports
    bstyle = pyslm.geometry.BuildStyle()
    bstyle.bid = 1
    bstyle.laserSpeed = 200.0  # [mm/s]
    bstyle.laserPower = 200.0  # [W]
    bstyle.jumpSpeed = 5000.0  # [mm/s]

    model = pyslm.geometry.Model()
    model.mid = 1
    model.buildStyles.append(bstyle)

    return [model]


def main():
    geomSlice, layer = build_layer(Z_TARGET)
    models = assign_model(layer)

    # Pick owner based on sequence, and take its representative interior point
    islands = get_island_geometries(layer)
    if islands:
        owner_idx = min(22, len(islands) - 1)  # 0-based index 22 if exists
        owner = islands[owner_idx]
        poly = getattr(owner, 'boundaryPoly', None)
        if poly is not None:
            ox, oy = poly.representative_point().coords[0]
        else:
            # Fallback to centroid of hatch coords
            c = getattr(owner, 'coords', None)
            if c is None or len(c) == 0:
                ox, oy = 0.0, 0.0
            else:
                arr = np.asarray(c)
                ox, oy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    else:
        ox, oy = 0.0, 0.0
    ox = -0.0950315
    oy = 0.00659148
    # Query 1: write neighborhood paths .scode
    q1_path = OUTDIR / f"neighborhood_paths_L0_x{ox:.3f}_y{oy:.3f}_r{NEIGHBOR_RADIUS_R:.2f}.scode"
    n1 = write_neighborhood_paths_scode(layer, models, ox, oy, NEIGHBOR_RADIUS_R, Z_TARGET, str(q1_path), island_index_base=0)
    print(f"Query1: wrote {n1} segments to {q1_path}")

    # Query 2: write island info .scode for the layer
    q2_path = OUTDIR / "layer_islands_L0.scode"
    n2 = write_layer_island_info_scode(layer, models, Z_TARGET, str(q2_path), island_index_base=0)
    print(f"Query2: wrote {n2} island rows to {q2_path}")


if __name__ == '__main__':
    main()
