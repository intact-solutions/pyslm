"""
Super-layer timing test (Stage 4)

This script hatches multiple layers using IslandHatcher with per-island grouping,
assigns a basic Model/BuildStyle, and aggregates timing over super-layers.
"""
import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.utils import getLayerTime
from pyslm.analysis.island_utils import aggregate_superlayers


def build_layers(part: pyslm.Part, z_values, hparams=None):
    myHatcher = hatching.IslandHatcher()

    # Defaults similar to other examples
    myHatcher.islandWidth = 5.0
    myHatcher.islandOverlap = -0.1
    myHatcher.hatchAngle = 10
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSortMethod = hatching.AlternateSort()
    myHatcher.groupIslands = True

    if hparams:
        for k, v in hparams.items():
            setattr(myHatcher, k, v)

    layers = []
    for z in z_values:
        geomSlice = part.getVectorSlice(float(z))
        layer = myHatcher.hatch(geomSlice)
        layers.append(layer)
    return layers


def assign_models_to_layers(layers):
    # Assign model/buildstyle ids to each geometry
    for layer in layers:
        for g in getattr(layer, 'geometry', []):
            g.mid = 1
            g.bid = 1

    # Minimal BuildStyle/Model for timing
    bstyle = pyslm.geometry.BuildStyle()
    bstyle.bid = 1
    bstyle.laserSpeed = 200.0  # [mm/s] continuous mode
    bstyle.laserPower = 200.0  # [W]
    bstyle.jumpSpeed = 5000.0  # [mm/s]

    model = pyslm.geometry.Model()
    model.mid = 1
    model.buildStyles.append(bstyle)

    return [model]


def main():
    # Load a part and position similar to other examples
    solidPart = pyslm.Part('inversePyramid')
    solidPart.setGeometry('models/frameGuide.stl')
    solidPart.dropToPlatform()
    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 2.0
    solidPart.rotation = [0, 0.0, np.pi]

    # Choose a few layer heights around 15 mm
    z_values = [14.50, 14.75, 14.99, 15.24, 15.50]

    layers = build_layers(solidPart, z_values)
    models = assign_models_to_layers(layers)

    # Print per-layer times
    layer_times = [getLayerTime(layer, models, includeJumpTime=True) for layer in layers]
    print('Layer z-values:', [f"{z:.2f}" for z in z_values])
    print('Layer times (s):', [f"{t:.6f}" for t in layer_times])

    # Aggregate into super-layers of size 2
    result = aggregate_superlayers(layers, models, group_size=2, include_jump=True)
    print('\nSuper-layer aggregation (group_size=2)')
    for g in result['groups']:
        idxs = g['layer_indices']
        times = g['layer_times']
        print(f"Group {g['group_index']}: layers={idxs} times={[f'{t:.6f}' for t in times]} group_time={g['group_time']:.6f}s")
    print('Total time:', f"{result['total_time']:.6f}s")


if __name__ == '__main__':
    main()
