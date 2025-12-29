import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.export_scode import write_layer_island_info_scode, write_neighborhood_paths_scode
from pyslm.analysis.island_utils import get_island_geometries
from pyslm.analysis.zone_utils import build_zone_polygons, classify_layer_geometry


Z_TARGET = 10.0
ISLAND_WIDTH = 5.0
NEIGHBOR_RADIUS_R = 0.8 * ISLAND_WIDTH
OUTDIR = Path(__file__).resolve().parent


def _base_path() -> Path:
    return _repo_root / "geometry_intact" / "zone_aware_island_gebracket"


def build_layer(z: float):
    base_path = _base_path()

    original_stl = base_path / "ge_bracket_original.stl"
    solid_part = pyslm.Part("ge_bracket")
    solid_part.setGeometry(str(original_stl))
    try:
        solid_part.dropToPlatform()
    except Exception:
        pass

    zone_ply_paths = {
        "high_sensi": base_path / "high_sensi_zone.ply",
        "med_sensi": base_path / "med_sensi_zone.ply",
        "low_sensi": base_path / "low_sensi_zone.ply",
        "base": base_path / "base_zone.ply",
        "boundary": base_path / "boundary_zone.ply",
        "interface": base_path / "interface_zone.ply",
    }

    zone_parts = {}
    for name, p in zone_ply_paths.items():
        if p.exists():
            part = pyslm.Part(name)
            part.setGeometry(str(p))
            zone_parts[name] = part

    geom_slice = solid_part.getVectorSlice(float(z), simplificationFactor=0.1)

    my_hatcher = hatching.IslandHatcher()
    my_hatcher.groupIslands = True
    my_hatcher.islandWidth = ISLAND_WIDTH
    my_hatcher.islandOverlap = 0.1
    my_hatcher.hatchAngle = 67
    my_hatcher.volumeOffsetHatch = -0.08
    my_hatcher.spotCompensation = 0.06
    my_hatcher.numInnerContours = 2
    my_hatcher.numOuterContours = 1
    my_hatcher.hatchSortMethod = hatching.AlternateSort()

    layer = my_hatcher.hatch(geom_slice)

    zone_polys = build_zone_polygons(zone_parts, float(z))

    zone_bids = {
        "high_sensi": 1,
        "med_sensi": 2,
        "low_sensi": 3,
        "base": 4,
        "boundary": 5,
        "interface": 6,
    }
    contour_bid = 10
    zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]

    classify_layer_geometry(
        layer,
        zone_polys,
        zone_bids,
        contour_bid=contour_bid,
        default_zone="base",
        priority=zone_priority,
    )

    for g in getattr(layer, "geometry", []) or []:
        g.mid = 1

    return geom_slice, layer, (zone_bids, contour_bid)


def build_models(zone_bids, contour_bid: int):
    zone_params = {
        "high_sensi": {"power": 200.0, "speed": 2.5},
        "med_sensi": {"power": 200.0, "speed": 1.75},
        "low_sensi": {"power": 200.0, "speed": 2.5},
        "base": {"power": 200.0, "speed": 2.5},
        "boundary": {"power": 200.0, "speed": 2.5},
        "interface": {"power": 200.0, "speed": 2.5},
        "contour": {"power": 180.0, "speed": 0.4},
    }

    model = pyslm.geometry.Model()
    model.mid = 1

    for zone_name, bid in zone_bids.items():
        bs = pyslm.geometry.BuildStyle()
        bs.bid = int(bid)
        bs.laserPower = float(zone_params[zone_name]["power"])
        bs.laserSpeed = float(zone_params[zone_name]["speed"])
        bs.jumpSpeed = 5000.0
        model.buildStyles.append(bs)

    bs_contour = pyslm.geometry.BuildStyle()
    bs_contour.bid = int(contour_bid)
    bs_contour.laserPower = float(zone_params["contour"]["power"])
    bs_contour.laserSpeed = float(zone_params["contour"]["speed"])
    bs_contour.jumpSpeed = 5000.0
    model.buildStyles.append(bs_contour)

    return [model]


def main():
    _, layer, (zone_bids, contour_bid) = build_layer(Z_TARGET)
    models = build_models(zone_bids, contour_bid)

    islands = get_island_geometries(layer)
    if islands:
        owner_idx = min(69, len(islands) - 1)
        owner = islands[owner_idx]
        poly = getattr(owner, "boundaryPoly", None)
        if poly is not None:
            ox, oy = poly.representative_point().coords[0]
        else:
            c = getattr(owner, "coords", None)
            if c is None or len(c) == 0:
                ox, oy = 0.0, 0.0
            else:
                arr = np.asarray(c)
                ox, oy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    else:
        ox, oy = 0.0, 0.0

    q1_path = OUTDIR / f"gebracket_neighborhood_paths_L0_x{ox:.3f}_y{oy:.3f}_r{NEIGHBOR_RADIUS_R:.2f}.scode"
    n1 = write_neighborhood_paths_scode([layer], models, ox, oy, NEIGHBOR_RADIUS_R, [Z_TARGET], str(q1_path),[0])
    print(f"Query1: wrote {n1} segments to {q1_path}")

    q2_path = OUTDIR / "gebracket_layer_islands_L0.scode"
    n2 = write_layer_island_info_scode(layer, models, Z_TARGET, str(q2_path),0)
    print(f"Query2: wrote {n2} island rows to {q2_path}")


if __name__ == "__main__":
    main()
