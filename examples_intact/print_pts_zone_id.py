"""Query which zone each island center lies in (GE bracket example).

This script demonstrates the public zone-query API:
- build_zone_polygons(zone_parts, z)   -> shapely MultiPolygons per zone at a layer Z
- find_zone_at_point((x, y), zone_polys, priority=..., default=...)

It hatches the GE bracket with groupIslands=True, then for each island computes a
representative point (centroid) and classifies which zone contains that point.

Output:
  # columns: z island_id cx cy zone

Optional:
  --plot renders colored circles for island centers (color = zone)
"""

import sys
from pathlib import Path
from collections import defaultdict

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse

import numpy as np

import matplotlib.pyplot as plt

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.zone_utils import build_zone_polygons, find_zone_at_point


def _base_path() -> Path:
    return _repo_root / "geometry_intact" / "zone_aware_island_gebracket"


def build_part_and_hatcher(model_path: str):
    # Load the original geometry and configure the IslandHatcher with grouped islands.
    solid_part = pyslm.Part("ge_bracket")
    solid_part.setGeometry(model_path)
    try:
        solid_part.dropToPlatform()
    except Exception:
        pass

    my_hatcher = hatching.IslandHatcher()
    my_hatcher.groupIslands = True
    my_hatcher.islandWidth = 5.0
    my_hatcher.islandOverlap = 0.1
    my_hatcher.hatchAngle = 67
    my_hatcher.volumeOffsetHatch = -0.08
    my_hatcher.spotCompensation = 0.06
    my_hatcher.numInnerContours = 2
    my_hatcher.numOuterContours = 1
    my_hatcher.hatchSortMethod = hatching.AlternateSort()

    return solid_part, my_hatcher


def build_zone_parts(base_path: Path):
    # Load zone geometries (PLY) as Parts.
    zone_ply_paths = {
        "high_sensi": base_path / "high_sensi_zone.ply",
        "med_sensi": base_path / "med_sensi_zone.ply",
        "low_sensi": base_path / "low_sensi_zone.ply",
        "base": base_path / "base_zone.ply",
        "boundary": base_path / "boundary_zone.ply",
        "interface": base_path / "interface_zone.ply",
    }

    zone_parts = {}
    for zone_name, ply_path in zone_ply_paths.items():
        if ply_path.exists():
            part = pyslm.Part(zone_name)
            part.setGeometry(str(ply_path))
            zone_parts[zone_name] = part

    return zone_parts


def island_center_xy(geom):
    # Prefer shapely centroid if available (best for irregular island boundaries).
    boundary_poly = getattr(geom, "boundaryPoly", None)
    if boundary_poly is not None and hasattr(boundary_poly, "centroid"):
        c = boundary_poly.centroid
        return float(c.x), float(c.y)

    # Fallback: centroid of coordinates if boundaryPoly is not present.
    coords = getattr(geom, "coords", None)
    if coords is not None and len(coords) > 0:
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    return None

SCALE = 0.001
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(_base_path() / "ge_bracket_original.stl"))
    p.add_argument("--layer-thickness", type=float, default=0.1)
    p.add_argument("--z", type=float, default=None)
    p.add_argument("--z-start", type=float, default=None)
    p.add_argument("--z-end", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    # Visualization controls.
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save-plot", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    solid_part, my_hatcher = build_part_and_hatcher(args.model_path)

    zmax = float(solid_part.boundingBox[5])

    if args.z is not None:
        zs = [float(args.z)]
    else:
        z_start = 0.0 if args.z_start is None else float(args.z_start)
        z_end = zmax if args.z_end is None else float(args.z_end)
        zs = list(np.arange(z_start, z_end, float(args.layer_thickness)))
    base_path = _base_path()
    zone_parts = build_zone_parts(base_path)

    # Zone priority matters if zones overlap (first match wins).
    zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]

    print("# columns: z island_id cx cy zone")


    points_by_z = defaultdict(list)

    with open("part_data.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            x, y, z = map(float, parts[:3])
            rest = " ".join(parts[3:])  # keep as string

            points_by_z[str(round(z,6))].append((x, y,z, rest))
    print(zs,points_by_z.keys())
    for str_z,items in points_by_z.items():
        z = float(str_z)/SCALE
        # Slice the original part at this Z, hatch it into islands/contours.
        geom_slice = solid_part.getVectorSlice(z+1e-5, simplificationFactor=0.1)
        if not geom_slice:
            print("!!!!!!!",z)
            continue
        

        # Slice each zone at the same Z and build shapely MultiPolygons.
        zone_polys = build_zone_polygons(zone_parts, z+1e-5)
        for item in items:
            cx, cy = item[0]/SCALE, item[1]/SCALE
            zone = find_zone_at_point((cx, cy), zone_polys, priority=zone_priority, default="base")
            print(item[0],item[1],item[2],item[3],zone)
       

if __name__ == "__main__":
    main()
