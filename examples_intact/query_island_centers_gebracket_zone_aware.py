import sys
from pathlib import Path

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
    boundary_poly = getattr(geom, "boundaryPoly", None)
    if boundary_poly is not None and hasattr(boundary_poly, "centroid"):
        c = boundary_poly.centroid
        return float(c.x), float(c.y)

    coords = getattr(geom, "coords", None)
    if coords is not None and len(coords) > 0:
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(_base_path() / "ge_bracket_original.stl"))
    p.add_argument("--layer-thickness", type=float, default=0.5)
    p.add_argument("--z", type=float, default=None)
    p.add_argument("--z-start", type=float, default=None)
    p.add_argument("--z-end", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
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

    zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]

    zone_colors = {
        "high_sensi": "red",
        "med_sensi": "orange",
        "low_sensi": "yellow",
        "base": "gray",
        "boundary": "blue",
        "interface": "purple",
        "unknown": "black",
    }

    print("# columns: z island_id cx cy zone")

    rows = 0
    for z in zs:
        geom_slice = solid_part.getVectorSlice(float(z), simplificationFactor=0.1)
        if not geom_slice:
            continue

        layer = my_hatcher.hatch(geom_slice)

        zone_polys = build_zone_polygons(zone_parts, float(z))

        plot_x = []
        plot_y = []
        plot_c = []

        for geom in getattr(layer, "geometry", []) or []:
            if getattr(geom, "subType", "") != "island":
                continue

            center = island_center_xy(geom)
            if center is None:
                continue

            cx, cy = center
            zone = find_zone_at_point((cx, cy), zone_polys, priority=zone_priority, default="base")
            island_id = getattr(geom, "islandId", None)
            island_id_str = "" if island_id is None else str(island_id)
            print(f"{float(z):g} {island_id_str} {cx:g} {cy:g} {zone}")

            if args.plot:
                plot_x.append(cx)
                plot_y.append(cy)
                plot_c.append(zone_colors.get(zone, zone_colors["unknown"]))

            rows += 1

            if args.limit is not None and rows >= int(args.limit):
                return

        if args.plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            if plot_x:
                ax.scatter(plot_x, plot_y, c=plot_c, s=30, marker="o", edgecolors="k", linewidths=0.2)

            handles = []
            for zone_name in zone_priority:
                handles.append(
                    ax.scatter([], [], c=zone_colors.get(zone_name, zone_colors["unknown"]), s=50, marker="o", label=zone_name)
                )
            ax.legend(loc="upper right")
            ax.set_aspect("equal")
            ax.set_title(f"GE bracket island centers by zone (z={float(z):g})")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")

            if args.save_plot is not None:
                save_path = args.save_plot
                if len(zs) > 1:
                    save_path = str(Path(args.save_plot).with_name(f"{Path(args.save_plot).stem}_z{float(z):g}{Path(args.save_plot).suffix}"))
                fig.savefig(save_path, dpi=150)

            if not args.no_show:
                plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
