from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import os

import numpy as np

from .island_utils import IslandIndex, get_island_geometries, compute_layer_geometry_times

SCALE = 0.001
def _iter_segments(coords: np.ndarray) -> Iterable[Tuple[float, float, float, float]]:
    if coords is None:
        return
    try:
        segs = coords.reshape(-1, 2, 2)
    except Exception:
        return
    for p in segs:
        yield float(p[0, 0]), float(p[0, 1]), float(p[1, 0]), float(p[1, 1])


def _path_length(coords: np.ndarray) -> float:
    if coords is None or len(coords) == 0:
        return 0.0
    try:
        segs = coords.reshape(-1, 2, 2)
    except Exception:
        return 0.0
    d = np.diff(segs, axis=1).reshape(-1, 2)
    return float(np.linalg.norm(d, axis=1).sum())


def _resolve_buildstyle(geom: Any, models: List[Any]) -> Optional[Any]:
    if not models:
        return None
    mid = getattr(geom, "mid", None)
    bid = getattr(geom, "bid", None)
    if mid is None or bid is None:
        return None
    model = next((m for m in models if getattr(m, "mid", None) == mid), None)
    if model is None:
        return None
    for bs in getattr(model, "buildStyles", []) or []:
        if getattr(bs, "bid", None) == bid:
            return bs
    return None


def _island_sequence_map(layer: Any, base: int = 0) -> Dict[Any, int]:
    seq = {}
    for i, g in enumerate(get_island_geometries(layer)):
        seq[g] = base + i
    return seq


def _format_float(v: float) -> str:
    # Default str keeps scientific where appropriate
    return ("%g" % v)


def _write_header(fh, lines: List[str]) -> None:
    for line in lines:
        fh.write(f"# {line}\n")

def write_neighborhood_paths_scode(
    layers: List[Any],
    models: List[Any],
    x: float,
    y: float,
    radius: float,
    zs: List[float],
    out_path: str,
    bids: List[int]
) -> int:
    written = 0
    index = None
    owner = None
    seq_map = {}
    def centroid_of(geom: Any) -> Tuple[float, float]:
        poly = getattr(geom, "boundaryPoly", None)
        if poly is None:
            coords = getattr(geom, "coords", None)
            if coords is None or len(coords) == 0:
                return (0.0, 0.0)
            c = np.asarray(coords, dtype=float)
            return (float(c[:, 0].mean()), float(c[:, 1].mean()))
        cx, cy = poly.centroid.coords[0]
        return float(cx), float(cy)
    def write_geom(geom: Any) -> int:
        if geom is None:
            return 0
        bs = _resolve_buildstyle(geom, models)
        power = float(getattr(bs, "laserPower", 0.0) if bs is not None else 0.0)
        speed = float(getattr(bs, "laserSpeed", 0.0) if bs is not None else 0.0)
        if lidx<len(layers)-1:
            speed = 100000
            power = 0.01
        idx = int(seq_map.get(geom, -1))
        count = 0
        for x1, y1, x2, y2 in _iter_segments(getattr(geom, "coords", None)) or []:
            fh.write(
                f"{_format_float(x1*SCALE)} {_format_float(y1*SCALE)} {_format_float(x2*SCALE)} {_format_float(y2*SCALE)} {_format_float(z*SCALE)} {_format_float(power)} {_format_float(speed)} {idx}\n"
            )
            count += 1
        return count
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for lidx, (layer,z, island_index_base) in enumerate(zip(layers,zs,bids)):
            index = IslandIndex(layer, neighbor_radius=radius)
            owner = index.find_island_at_point(x, y)
            neighbors: List[Any] = index.neighbors_for_island(owner) if owner is not None else []

            seq_map = _island_sequence_map(layer, base=island_index_base)

            '''
            _write_header(
                fh,
                [
                    ".scode Query 1 – neighborhood paths",
                    "columns: x1 y1 x2 y2 z power speed island-idx",
                    f"params: x={_format_float(x)} y={_format_float(y)} r={_format_float(radius)} z={_format_float(z)} index_base={island_index_base}",
                    "units: mm(mm/s for speed), W(power)",
                ],
            )
            '''
            neighbors.append(owner)
            neighbors = sorted(neighbors,key=lambda geom: int(seq_map.get(geom, -1)))
            for nb in neighbors:
                if lidx == len(layers)-1 and int(seq_map.get(nb, -1)>int(seq_map.get(owner, -1))):
                    continue
                written += write_geom(nb)

    return written, int(seq_map.get(owner, -1)), centroid_of(owner)



def write_layer_island_info_scode(
    layer: Any,
    models: List[Any],
    z: float,
    out_path: str,
    island_index_base: int = 0,
    re: bool = True
) -> int:
    islands: List[Any] = get_island_geometries(layer)

    entries = compute_layer_geometry_times(layer, models, include_jump=True, validate=False)
    time_by_id: Dict[int, float] = {id(e["geom"]): float(e["time"]) for e in entries}

    seq_map = _island_sequence_map(layer, base=island_index_base)

    def centroid_of(geom: Any) -> Tuple[float, float]:
        poly = getattr(geom, "boundaryPoly", None)
        if poly is None:
            coords = getattr(geom, "coords", None)
            if coords is None or len(coords) == 0:
                return (0.0, 0.0)
            c = np.asarray(coords, dtype=float)
            return (float(c[:, 0].mean()), float(c[:, 1].mean()))
        cx, cy = poly.centroid.coords[0]
        return float(cx), float(cy)

    def edge_midpoints(poly: Any) -> List[Tuple[float, float]]:
        if poly is None:
            return []
        xys = list(poly.exterior.coords)
        if len(xys) < 5:
            return []
        mids: List[Tuple[float, float]] = []
        for k in range(4):
            x1, y1 = xys[k]
            x2, y2 = xys[k + 1]
            mids.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
        return mids

    def choose_entry_exit(cur: Any, prev: Optional[Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        poly = getattr(cur, "boundaryPoly", None)
        mids = edge_midpoints(poly)
        if not mids:
            # Fallback: use first and third points of coords if available
            coords = getattr(cur, "coords", None)
            if coords is not None and len(coords) >= 4:
                try:
                    pts = coords.reshape(-1, 2)
                    return (float(pts[0, 0]), float(pts[0, 1])), (float(pts[2, 0]), float(pts[2, 1]))
                except Exception:
                    pass
            return (0.0, 0.0), (0.0, 0.0)

        if prev is None:
            # Pick westernmost edge midpoint as entry; opposite as exit
            xs = [m[0] for m in mids]
            k = int(np.argmin(xs))
            e_in = mids[k]
            e_out = mids[(k + 2) % 4]
            return (float(e_in[0]), float(e_in[1])), (float(e_out[0]), float(e_out[1]))

        pcx, pcy = centroid_of(prev)
        dists = [math.hypot(m[0] - pcx, m[1] - pcy) for m in mids]
        k = int(np.argmin(dists))
        e_in = mids[k]
        e_out = mids[(k + 2) % 4]
        return (float(e_in[0]), float(e_in[1])), (float(e_out[0]), float(e_out[1]))

    written = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w" if re else "a", encoding="utf-8") as fh:

        for i, cur in enumerate(islands):
            prev = islands[i - 1] if i > 0 else None
            e_in, e_out = choose_entry_exit(cur, prev)

            bs = _resolve_buildstyle(cur, models)
            power = float(getattr(bs, "laserPower", 0.0) if bs is not None else 0.0)

            coords = getattr(cur, "coords", None)
            total_len = _path_length(coords)
            total_time = float(time_by_id.get(id(cur), 0.0))
            eq_speed = float(total_len / total_time) if total_time > 0.0 else 0.0

            idx = int(seq_map.get(cur, -1))
            fh.write(
                f"{_format_float(e_in[0]*SCALE)} {_format_float(e_in[1]*SCALE)} {_format_float(e_out[0]*SCALE)} {_format_float(e_out[1]*SCALE)} "
                f"{_format_float(z*SCALE)} {_format_float(power)} {_format_float(round(eq_speed,2))} {_format_float(total_time)} {idx}\n"
            )
            written += 1

    return written
