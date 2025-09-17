"""
A simple example showing how to use PySLM  with the IslandHatcher approach, which decomposes the layer into several
island regions, which are tested for intersection and then the hatches generated are more efficiently clipped.
"""

import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import time
import matplotlib.pyplot as plt

from shapely.geometry import MultiPolygon

import pyslm
import pyslm.visualise
from pyslm import hatching as hatching

import inspect
from pyslm.hatching.islandHatcher import IslandHatcher

print("pyslm module path:", pyslm.__file__)
print("IslandHatcher path:", inspect.getfile(IslandHatcher))
print("Has groupIslands:", hasattr(IslandHatcher, "groupIslands"))

# Imports the part and sets the geometry to  an STL file (frameGuide.stl)
solidPart = pyslm.Part('inversePyramid')
solidPart.setGeometry('models/frameGuide.stl')
solidPart.dropToPlatform()

solidPart.origin[0] = 5.0
solidPart.origin[1] = 2.5
solidPart.scaleFactor = 2.0
solidPart.rotation = [0, 0.0, np.pi]

# Set te slice layer position
z = 14.99

# Create a StripeHatcher object for performing any hatching operations
myHatcher = hatching.IslandHatcher()
myHatcher.islandWidth = 5.0
myHatcher.islandOverlap = -0.1
"""
Opt-in: emit one HatchGeometry per island with metadata for multi-scale workflows.
Leave this as False to preserve legacy (single merged HatchGeometry) behavior.
"""
myHatcher.groupIslands = True

# Set the base hatching parameters which are generated within Hatcher
myHatcher.hatchAngle = 10
myHatcher.volumeOffsetHatch = 0.08
myHatcher.spotCompensation = 0.06
myHatcher.numInnerContours = 2
myHatcher.numOuterContours = 1
myHatcher.hatchSortMethod = hatching.AlternateSort()

# The traditional approach is to get the path ring coordinates and pass this to Island Hatcher
geomSlice = solidPart.getVectorSlice(z)

"""
Diagnostic: Set to True to show the general process for how IslandHatcher works. 
Note: disabled by default to keep the example focused on island outlines + sequence only.
"""
if False:
    # Generates a set of square islands which is guaranteed to cover the entire area of the boundaries.
    # The global orientation of the angle is provided as the second argument
    islands = myHatcher.generateIslands(geomSlice, 30)

    # The user can extract the ids of all the ids that are clipped or not clipped of the islands
    # The boundary should be provided to be clipped against.
    a, b = myHatcher.intersectIslands(geomSlice, islands)

    overlapIslands = [islands[i] for i in a]
    intersectIslands = [islands[i] for i in b]

    # The above intersectIsland internal method can also be achieved using the following approach below.

    # Get the Shapely Polygons from slicing the part
    poly = solidPart.getVectorSlice(z, False)

    # Use shapely MultiPolygon collection to allow full testing and clipping across all boundary regions
    poly = MultiPolygon(poly)

    intersectIslands = []
    overlapIslands = []

    # Python sets are used to perform boolean operations on a set to identify unclipped islands
    intersectIslandsSet = set()
    overlapIslandsSet= set()

    # Iterate across all the islands
    for i in range(len(islands)):

        island = islands[i]
        s = island.boundary()

        if poly.overlaps(s):
            overlapIslandsSet.add(i) # id
            overlapIslands.append(island)

        if poly.intersects(s):
            intersectIslandsSet.add(i)  # id
            intersectIslands.append(island)

    unTouchedIslandSet = intersectIslandsSet-overlapIslandsSet
    unTouchedIslands = [islands[i] for i in unTouchedIslandSet]

    print('Finished Island Clipping')

    fig, ax = pyslm.visualise.plotPolygon(geomSlice)

    # Plot using visualise.plotPolygon the original islands generated before intersection
    for island in islands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x,y]).T], handle=(fig, ax))

    for island in intersectIslands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x,y]).T], handle=(fig, ax),  plotFilled=True, lineColor='g', fillColor = '#19aeffff')

    for island in overlapIslands:
        x, y = island.boundary().exterior.xy
        pyslm.visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), plotFilled=True, lineColor='b', fillColor = '#ff4141ff')


startTime = time.time()

print('Hatching Started')
layer = myHatcher.hatch(geomSlice)
print('Completed Hatching')

# Only show island outlines and their sequence (no scan paths)
fig, ax = plt.subplots()
ax.axis('equal')

# Plot slice boundary for context
try:
    pyslm.visualise.plotPolygon(geomSlice, handle=(fig, ax), lineColor='k', lineWidth=0.6)
except Exception:
    pass

island_geoms = [g for g in layer.geometry if getattr(g, 'subType', '') == 'island']
print(f"Per-island grouping enabled. Islands emitted: {len(island_geoms)}")

# Draw island boundaries color-coded by sequence (blue->red) and annotate sequence number (smaller font, no circle)
centroids = []
num_islands = len(island_geoms)
from matplotlib import cm
cmap = cm.get_cmap('coolwarm')  # blue -> red
for idx, gi in enumerate(island_geoms, start=1):
    poly = getattr(gi, 'boundaryPoly', None)
    if poly is None:
        continue
    # Normalize color across sequence
    t = 0.5 if num_islands <= 1 else (idx - 1) / (num_islands - 1)
    color = cmap(t)

    x, y = poly.exterior.xy
    pyslm.visualise.plotPolygon([np.vstack([x, y]).T], handle=(fig, ax), lineColor=color, lineWidth=1.2)
    cx, cy = poly.centroid.coords[0]
    centroids.append((cx, cy))
    # 50% smaller font than before (was 8 -> now 4), grey font, no circle bbox
    ax.text(cx, cy, str(idx), color='#666666', fontsize=4, ha='center', va='center')

plt.title('Islands and Scan Sequence (no paths)')
plt.show()

