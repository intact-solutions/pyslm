import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse

import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.export_scode import write_layer_island_info_scode
from pyslm.analysis.zone_utils import build_zone_polygons, classify_layer_geometry


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
    my_hatcher.islandWidth = 2.0
    my_hatcher.islandOverlap = 0
    my_hatcher.hatchAngle = 0
    my_hatcher.volumeOffsetHatch = 0
    my_hatcher.spotCompensation = 0
    my_hatcher.numInnerContours = 0
    my_hatcher.numOuterContours = 0
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


def build_models():
    zone_bids = {
        "high_sensi": 1,
        "med_sensi": 2,
        "low_sensi": 3,
        "base": 4,
        "boundary": 5,
        "interface": 6,
    }
    contour_bid = 10

    zone_params = {
        "high_sensi": {"power": 200.0, "speed": 2500.0},
        "med_sensi": {"power": 200.0, "speed": 1750.0},
        "low_sensi": {"power": 200.0, "speed": 2500.0},
        "base": {"power": 200.0, "speed": 2500.0},
        "boundary": {"power": 200.0, "speed": 2500.0},
        "interface": {"power": 200.0, "speed": 2500.0},
        "contour": {"power": 180.0, "speed": 400.0},
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

    return [model], zone_bids, contour_bid


def assign_model(layer, models):
    for g in getattr(layer, "geometry", []) or []:
        g.mid = models[0].mid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(_base_path() / "ge_bracket_original.stl"))
    p.add_argument("--layer-thickness", type=float, default=0.5)
    p.add_argument("--z-start", type=float, default=None)
    p.add_argument("--z-end", type=float, default=None)
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--global-island-indexing", action="store_true")
    p.add_argument("--single-file", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    args.single_file = True
    args.layer_thickness = 0.1
    if args.single_file and not args.global_island_indexing:
        args.global_island_indexing = True

    solid_part, my_hatcher = build_part_and_hatcher(args.model_path)

    # boundingBox is (xmin, ymin, zmin, xmax, ymax, zmax)
    zmax = float(solid_part.boundingBox[5])
    z_start = 0.0 if args.z_start is None else float(args.z_start)
    z_end = zmax if args.z_end is None else float(args.z_end)

    base_path = _base_path()
    zone_parts = build_zone_parts(base_path)

    models, zone_bids, contour_bid = build_models()

    zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]

    zs = np.arange(z_start, z_end, float(args.layer_thickness))
    base = 0

    if args.single_file:
        agg_path = outdir / "gebracket_all_layer_islands_global.scode"
        with open(agg_path, "w", encoding="utf-8") as fh:
            fh.write("# .scode Query 2 – island info by layer (global aggregation)\n")
            fh.write("# columns: x1 y1 x2 y2 z power eq_speed total_time island-idx\n")
            fh.write(
                f"# params: global_indexing=1 z_start={z_start:g} z_end={z_end:g} layer_thickness={float(args.layer_thickness):g}\n"
            )
            fh.write("# units: mm(mm/s for eq_speed), W(power), s(time)\n")

            L = 0
            total_rows = 0
            for z in zs:
                geom_slice = solid_part.getVectorSlice(float(z), simplificationFactor=0.1)
                layer = my_hatcher.hatch(geom_slice)

                zone_polys = build_zone_polygons(zone_parts, float(z))
                classify_layer_geometry(
                    layer,
                    zone_polys,
                    zone_bids,
                    contour_bid=contour_bid,
                    default_zone="base",
                    priority=zone_priority,
                )
                assign_model(layer, models)

                tmp_path = outdir / "_tmp_layer.scode"
                n = write_layer_island_info_scode(layer, models, float(z), str(tmp_path), island_index_base=base)
                if n == 0:
                    tmp_path.unlink(missing_ok=True)
                    L += 1
                    continue

                with open(tmp_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith("#"):
                            continue
                        fh.write(line)
                tmp_path.unlink(missing_ok=True)

                base += n
                total_rows += n
                print(f"L{L}: wrote {n} islands (base={base})")
                L += 1

        print(f"Wrote {total_rows} rows to {agg_path}")
        return

    L = 0
    for z in zs:
        geom_slice = solid_part.getVectorSlice(float(z), simplificationFactor=0.1)
        layer = my_hatcher.hatch(geom_slice)

        zone_polys = build_zone_polygons(zone_parts, float(z))
        classify_layer_geometry(
            layer,
            zone_polys,
            zone_bids,
            contour_bid=contour_bid,
            default_zone="base",
            priority=zone_priority,
        )
        assign_model(layer, models)

        out_path = outdir / f"gebracket_layer_islands_L{L}.scode"
        if args.global_island_indexing:
            n = write_layer_island_info_scode(layer, models, float(z), str(out_path), island_index_base=base)
            if n == 0:
                out_path.unlink(missing_ok=True)
                print(f"L{L}: empty, skipped")
            else:
                print(f"L{L}: wrote {n} islands to {out_path}")
                base += n
        else:
            n = write_layer_island_info_scode(layer, models, float(z), str(out_path), island_index_base=0)
            if n == 0:
                out_path.unlink(missing_ok=True)
                print(f"L{L}: empty, skipped")
            else:
                print(f"L{L}: wrote {n} islands to {out_path}")
        L += 1


if __name__ == "__main__":
    main()
