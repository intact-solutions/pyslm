import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import get_island_geometries, compute_layer_geometry_times
from pyslm.analysis.export_scode import write_layer_island_info_scode


def build_part_and_hatcher(model_path: str):
    solidPart = pyslm.Part('inversePyramid')
    solidPart.setGeometry(model_path)
    solidPart.dropToPlatform()
    solidPart.origin[0] = 5.0
    solidPart.origin[1] = 2.5
    solidPart.scaleFactor = 2.0
    solidPart.rotation = [0, 0.0, np.pi]

    myHatcher = hatching.IslandHatcher()
    myHatcher.islandWidth = 5.0
    myHatcher.islandOverlap = -0.1
    myHatcher.hatchAngle = 10
    myHatcher.volumeOffsetHatch = 0.08
    myHatcher.spotCompensation = 0.06
    myHatcher.numInnerContours = 2
    myHatcher.numOuterContours = 1
    myHatcher.hatchSortMethod = hatching.AlternateSort()
    myHatcher.groupIslands = True

    return solidPart, myHatcher


def build_minimal_models():
    bstyle = pyslm.geometry.BuildStyle()
    bstyle.bid = 1
    bstyle.laserSpeed = 200.0
    bstyle.laserPower = 200.0
    bstyle.jumpSpeed = 5000.0

    model = pyslm.geometry.Model()
    model.mid = 1
    model.buildStyles.append(bstyle)

    return [model]


def assign_model(layer, models):
    for g in getattr(layer, 'geometry', []) or []:
        g.mid = models[0].mid
        g.bid = models[0].buildStyles[0].bid


def _format_float(v: float) -> str:
    return ("%g" % v)


def _resolve_buildstyle(geom, models):
    if not models:
        return None
    mid = getattr(geom, 'mid', None)
    bid = getattr(geom, 'bid', None)
    if mid is None or bid is None:
        return None
    model = next((m for m in models if getattr(m, 'mid', None) == mid), None)
    if model is None:
        return None
    for bs in getattr(model, 'buildStyles', []) or []:
        if getattr(bs, 'bid', None) == bid:
            return bs
    return None


def _path_length(coords) -> float:
    if coords is None or len(coords) == 0:
        return 0.0
    try:
        segs = np.asarray(coords).reshape(-1, 2, 2)
    except Exception:
        return 0.0
    d = np.diff(segs, axis=1).reshape(-1, 2)
    return float(np.linalg.norm(d, axis=1).sum())


def _centroid_of(geom):
    poly = getattr(geom, 'boundaryPoly', None)
    if poly is None:
        coords = getattr(geom, 'coords', None)
        if coords is None or len(coords) == 0:
            return (0.0, 0.0)
        c = np.asarray(coords, dtype=float)
        return (float(c[:, 0].mean()), float(c[:, 1].mean()))
    cx, cy = poly.centroid.coords[0]
    return float(cx), float(cy)


def _edge_midpoints(poly):
    if poly is None:
        return []
    xys = list(poly.exterior.coords)
    if len(xys) < 5:
        return []
    mids = []
    for k in range(4):
        x1, y1 = xys[k]
        x2, y2 = xys[k + 1]
        mids.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    return mids


def _choose_entry_exit(cur, prev):
    poly = getattr(cur, 'boundaryPoly', None)
    mids = _edge_midpoints(poly)
    if not mids:
        coords = getattr(cur, 'coords', None)
        if coords is not None and len(coords) >= 4:
            try:
                pts = np.asarray(coords).reshape(-1, 2)
                return (float(pts[0, 0]), float(pts[0, 1])), (float(pts[2, 0]), float(pts[2, 1]))
            except Exception:
                pass
        return (0.0, 0.0), (0.0, 0.0)

    if prev is None:
        xs = [m[0] for m in mids]
        k = int(np.argmin(xs))
        e_in = mids[k]
        e_out = mids[(k + 2) % 4]
        return (float(e_in[0]), float(e_in[1])), (float(e_out[0]), float(e_out[1]))

    pcx, pcy = _centroid_of(prev)
    dists = [np.hypot(m[0] - pcx, m[1] - pcy) for m in mids]
    k = int(np.argmin(dists))
    e_in = mids[k]
    e_out = mids[(k + 2) % 4]
    return (float(e_in[0]), float(e_in[1])), (float(e_out[0]), float(e_out[1]))


def build_rows_for_layer(layer, models, z, index_base: int):
    islands = get_island_geometries(layer)
    if not islands:
        return [], 0

    entries = compute_layer_geometry_times(layer, models, include_jump=True, validate=False)
    time_by_id = {id(e['geom']): float(e['time']) for e in entries}

    rows = []
    for i, cur in enumerate(islands):
        prev = islands[i - 1] if i > 0 else None
        e_in, e_out = _choose_entry_exit(cur, prev)

        bs = _resolve_buildstyle(cur, models)
        power = float(getattr(bs, 'laserPower', 0.0) if bs is not None else 0.0)

        coords = getattr(cur, 'coords', None)
        total_len = _path_length(coords)
        total_time = float(time_by_id.get(id(cur), 0.0))
        eq_speed = float(total_len / total_time) if total_time > 0.0 else 0.0

        idx = int(index_base + i)
        rows.append(
            f"{_format_float(e_in[0])} {_format_float(e_in[1])} {_format_float(e_out[0])} {_format_float(e_out[1])} "
            f"{_format_float(z)} {_format_float(power)} {_format_float(eq_speed)} {_format_float(total_time)} {idx}\n"
        )
    return rows, len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, default=str(_repo_root / 'models' / 'frameGuide.stl'))
    p.add_argument('--layer-thickness', type=float, default=0.5)
    p.add_argument('--z-start', type=float, default=None)
    p.add_argument('--z-end', type=float, default=None)
    p.add_argument('--outdir', type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument('--global-island-indexing', action='store_true')
    p.add_argument('--single-file', action='store_true')
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.single_file and not args.global_island_indexing:
        args.global_island_indexing = True

    solidPart, myHatcher = build_part_and_hatcher(args.model_path)
    zmax = float(solidPart.boundingBox[5])
    z_start = 0.0 if args.z_start is None else float(args.z_start)
    z_end = zmax if args.z_end is None else float(args.z_end)

    models = build_minimal_models()

    zs = np.arange(z_start, z_end, float(args.layer_thickness))
    base = 0

    if args.single_file:
        agg_path = outdir / 'all_layer_islands_global.scode'
        with open(agg_path, 'w', encoding='utf-8') as fh:
            fh.write('# .scode Query 2 – island info by layer (global aggregation)\n')
            fh.write('# columns: x1 y1 x2 y2 z power eq_speed total_time island-idx\n')
            fh.write(f"# params: global_indexing=1 z_start={_format_float(z_start)} z_end={_format_float(z_end)} layer_thickness={_format_float(args.layer_thickness)}\n")
            fh.write('# units: mm(mm/s for eq_speed), W(power), s(time)\n')

            L = 0
            total_rows = 0
            for z in zs:
                geomSlice = solidPart.getVectorSlice(float(z))
                layer = myHatcher.hatch(geomSlice)
                assign_model(layer, models)
                rows, n = build_rows_for_layer(layer, models, float(z), base)
                if n == 0:
                    L += 1
                    continue
                for line in rows:
                    fh.write(line)
                base += n
                total_rows += n
                print(f"L{L}: wrote {n} islands (base={base})")
                L += 1
        print(f"Wrote {total_rows} rows to {agg_path}")
        return

    L = 0
    for z in zs:
        geomSlice = solidPart.getVectorSlice(float(z))
        layer = myHatcher.hatch(geomSlice)
        assign_model(layer, models)

        if args.global_island_indexing:
            rows, n = build_rows_for_layer(layer, models, float(z), base)
            if n == 0:
                L += 1
                continue
            out_path = outdir / f"layer_islands_L{L}.scode"
            with open(out_path, 'w', encoding='utf-8') as fh:
                fh.write('# .scode Query 2 – island info by layer\n')
                fh.write('# columns: x1 y1 x2 y2 z power eq_speed total_time island-idx\n')
                fh.write(f"# params: z={_format_float(float(z))} index_base={base}\n")
                fh.write('# units: mm(mm/s for eq_speed), W(power), s(time)\n')
                for line in rows:
                    fh.write(line)
            print(f"L{L}: wrote {n} islands to {out_path}")
            base += n
        else:
            out_path = outdir / f"layer_islands_L{L}.scode"
            n = write_layer_island_info_scode(layer, models, float(z), str(out_path), island_index_base=0)
            if n == 0:
                out_path.unlink(missing_ok=True)
                print(f"L{L}: empty, skipped")
            else:
                print(f"L{L}: wrote {n} islands to {out_path}")
        L += 1


if __name__ == '__main__':
    main()
