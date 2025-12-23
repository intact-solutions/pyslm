"""
Figure 1: Level 1 only — sequence-colored islands with scan paths for owner+neighbors

This script creates a single-axes plot that:
- Builds a target layer with IslandHatcher (groupIslands=True)
- Selects a point of interest like test_spatial_lookup.py (interior point of a chosen island)
- Colors island outlines by sequence (coolwarm) like test_island_grouping.py
- Shows BOTH sequence index and per-island timing annotations inside islands
- Draws scan paths (hatches) ONLY for the owner island and its neighbors like example_island_hatcher.py
"""
import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pyslm
import pyslm.visualise
from pyslm import hatching as hatching

from pyslm.analysis.island_utils import (
    IslandIndex,
    get_island_geometries,
    compute_layer_geometry_times,
)


# ----------------------------
# Config
# ----------------------------
Z_TARGET = 14.99
SCAN_CONTOUR_FIRST = False  # available if needed by your IslandHatcher setup
ISLAND_WIDTH = 5.0
NEIGHBOR_RADIUS_R = 0.8 * ISLAND_WIDTH
OWNER_SEQUENCE_INDEX_1BASED = 23  # similar selection strategy to test_spatial_lookup (choose a specific island)

# Colors
COLOR_FILL_OWNER = '#66c2a5cc'    # light teal with alpha (owner fill)
COLOR_FILL_NEIGHBOR = '#ffcc80cc' # light orange with alpha (neighbor fill)
COLOR_SEQ_LABEL = '#666666'       # sequence label color

# scan path colors for owner/neighbor lines
COLOR_OWNER_LINE = '#d62728'
COLOR_NEIGHBOR_LINE = '#1f77b4'

FONT_ISLAND_TIME = 5


def build_layer(z: float):
    solidPart = pyslm.Part('inversePyramid')
    solidPart.setGeometry('models/ge_bracket_large_1_1.STL')
    solidPart.dropToPlatform()
    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 2.0
    solidPart.rotation = [0, 0.0, np.pi]

    geomSlice = solidPart.getVectorSlice(z)

    myHatcher = hatching.IslandHatcher()
    myHatcher.islandWidth = ISLAND_WIDTH
    myHatcher.islandOverlap = -0.1
    myHatcher.hatchAngle = 0
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSortMethod = hatching.AlternateSort()
    myHatcher.groupIslands = True

    layer = myHatcher.hatch(geomSlice)
    return geomSlice, layer


def assign_model(layer):
    # Assign model/buildstyle ids to each geometry
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


def pick_owner_and_point(island_geoms):
    """Pick an owner island by sequence index (1-based), then use its robust interior point.
    Fallback to last island if index exceeds length.
    Returns (owner_geom, (ox, oy)).
    """
    if not island_geoms:
        return None, (0.0, 0.0)
    idx0 = max(1, OWNER_SEQUENCE_INDEX_1BASED) - 1
    owner = island_geoms[idx0] if idx0 < len(island_geoms) else island_geoms[-1]
    poly = getattr(owner, 'boundaryPoly', None)
    if poly is None:
        return owner, (0.0, 0.0)
    # Use Shapely representative point (always inside polygon)
    ox, oy = poly.representative_point().coords[0]
    return owner, (ox, oy)

def draw_figure1(ax, geomSlice, layer, models, owner, neighbors, owner_point, time_by_geom):
    ax.set_title('Figure 1: Sequence-colored islands + scan paths (owner & neighbors)')
    ax.axis('equal')

    # Base: plot slice boundary
    try:
        pyslm.visualise.plotPolygon(geomSlice, handle=(plt.gcf(), ax), lineColor='k', lineWidth=0.5)
    except Exception:
        pass

    islands = get_island_geometries(layer)
    cmap = mpl.colormaps.get_cmap('coolwarm')
    num_islands = len(islands)
    neighbor_set = set(neighbors)

    # Draw island outlines colored by sequence; outlines only (no fill)
    for idx, gi in enumerate(islands, start=1):
        poly = getattr(gi, 'boundaryPoly', None)
        if poly is None:
            continue
        x, y = poly.exterior.xy
        # Outline color by normalized sequence
        t = 0.5 if num_islands <= 1 else (idx - 1) / (num_islands - 1)
        line_color = cmap(t)

        # Always outline; no fill for Level 1 figure
        lw = 1.2 if gi is owner else (1.0 if gi in neighbor_set else 0.9)
        pyslm.visualise.plotPolygon([np.vstack([x, y]).T], handle=(plt.gcf(), ax), lineColor=line_color, lineWidth=lw)

        # Sequence label (small grey) slightly above centroid
        cx, cy = poly.centroid.coords[0]
        ax.text(cx, cy + 0.25, str(idx), color=COLOR_SEQ_LABEL, fontsize=4, ha='center', va='center')

        # Timing annotation further below the centroid to avoid overlap; add light bbox for readability
        t_island = time_by_geom.get(gi, None)
        if t_island is not None:
            dy = 0.8  # increased offset in mm to avoid overlap with sequence index
            txt_color = COLOR_SEQ_LABEL
            if gi is owner:
                txt_color = COLOR_OWNER_LINE
            elif gi in neighbor_set:
                txt_color = COLOR_NEIGHBOR_LINE
            ax.text(
                cx,
                cy - dy,
                f"{t_island:.3f}s",
                fontsize=FONT_ISLAND_TIME,
                color=txt_color,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5),
            )

    # Draw scan paths for owner + neighbors only
    def draw_hatch_paths(geom, color):
        coords = getattr(geom, 'coords', None)
        if coords is None:
            return
        try:
            segs = coords.reshape(-1, 2, 2)
        except Exception:
            return
        for p in segs:
            ax.plot([p[0,0], p[1,0]], [p[0,1], p[1,1]], color=color, linewidth=0.8, alpha=0.9)

    if owner is not None:
        draw_hatch_paths(owner, COLOR_OWNER_LINE)
    for nb in neighbors:
        draw_hatch_paths(nb, COLOR_NEIGHBOR_LINE)

    # Mark point of interest
    if owner_point is not None:
        ox, oy = owner_point
        ax.plot([ox], [oy], marker='o', markersize=3, color='black')


def main():
    geomSlice, layer = build_layer(Z_TARGET)
    models = assign_model(layer)

    #islands = get_island_geometries(layer)
    #owner, owner_point = pick_owner_and_point(islands)

    # Neighbors via spatial index
    index = IslandIndex(layer, neighbor_radius=NEIGHBOR_RADIUS_R)
    ox = 0
    oy = -50
    owner = index.find_island_at_point(ox, oy)
    owner_point = (ox,oy)
    neighbors = index.neighbors_for_island(owner) if owner is not None else []
    print(owner.boundingBox(),dir(owner))

    # Timing per island for annotation
    entries = compute_layer_geometry_times(layer, models, include_jump=True, validate=False)
    time_by_geom = {e['geom']: float(e['time']) for e in entries if getattr(e['geom'], 'subType', '') == 'island'}

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_figure1(ax, geomSlice, layer, models, owner, neighbors, owner_point, time_by_geom)
    plt.tight_layout()
    plt.savefig('test.png')


if __name__ == '__main__':
    main()
