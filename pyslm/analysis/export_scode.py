from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math
import os

import numpy as np

from .island_utils import IslandIndex, get_island_geometries, compute_layer_geometry_times

SCALE = 0.001
TWIST = False
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

def island_to_random_power_speed(input_int: int, power: float, speed: float) -> [int,int]:
    r1 = 2*((input_int * 1234567 + 987654321) % 1756121)/1756121-1
    r2 = 2*((input_int * 2183181 + 349293103) % 3438243)/3438243-1
    return [power*(1+r1*0.01),speed*(1+r2*0.01)]

def calculate_torsion_position(x, y, z, z1 = 12, z2 = 38.2, n_degrees = 30):
    """
    Calculates the new position of a point under torsional deformation around the Z-axis.
    The bottom surface (z1) rotates by n_degrees, and the top (z2) stays fixed.
    
    Parameters:
    x, y, z   : Original coordinates of the point.
    z1        : Z-coordinate of the rotating bottom surface.
    z2        : Z-coordinate of the fixed top surface.
    n_degrees : Angle of rotation at z1, in degrees.
    
    Returns:
    tuple: The new (x_new, y_new, z_new) coordinates.
    """
    if not TWIST:
        return x, y, z
    if z1 == z2:
        raise ValueError("z1 and z2 cannot be the same height.")
    if z > z2 or z < z1:
        return x, y, z
    # Convert the maximum rotation angle from degrees to radians
    n_rad = math.radians(n_degrees)

    # Linearly interpolate the rotation angle for the specific z-height
    # At z = z1, theta = n_rad. At z = z2, theta = 0.
    theta = n_rad * ((z - z1) / (z2 - z1))

    # Apply the 2D rotation matrix for the X and Y coordinates
    x_new = x * math.cos(theta) - y * math.sin(theta)
    y_new = x * math.sin(theta) + y * math.cos(theta)
    
    # Z remains unchanged during a Z-axis rotation
    z_new = z 

    return x_new, y_new, z_new

def read_scode_pv(pv: List[Any], idx: int) -> [float, float]:
    return [pv[0][idx], pv[1][idx]]
    
def pos_to_power_speed(x: float, y: float, z: float, power: float, speed: float, bbox:[float, float, float, float, float, float] = [-0.09567338957373658, -0.06541411785034108, -4.718713626061803e-09, 0.08536702478482584, 0.038807186772294854, 0.06212175750732427]):
    return [power,speed] 
    '''
    if (x*x+y*y)<60*60:
        return [int(power*1.05),speed]
    else:
        return [power,speed] 
    '''
    x_13 = bbox[0] + 1/3*(bbox[3] - bbox[0])
    x_23 = bbox[0] + 2/3*(bbox[3] - bbox[0])
    y_13 = bbox[1] + 1/3*(bbox[4] - bbox[1])
    y_23 = bbox[1] + 2/3*(bbox[4] - bbox[1])
    z_13 = bbox[2] + 1/3*(bbox[5] - bbox[2])
    z_23 = bbox[2] + 2/3*(bbox[5] - bbox[2])
    cube_size = 0.02
    cube_height = 0.01
    offset_p = 0.05
    offset_v = 0.2
    if abs(x - x_13) < cube_size and abs(y - y_13) < cube_size and abs(z - z_13) < cube_height:
        return [power*(1-offset_p),speed*(1-offset_v)]
    elif abs(x - x_23) < cube_size and abs(y - y_13) < cube_size and abs(z - z_13) < cube_height:
        return [power*(1+offset_p),speed*(1-offset_v)]
    elif abs(x - x_23) < cube_size and abs(y - y_23) < cube_size and abs(z - z_13) < cube_height:
        return [power*(1+offset_p),speed*(1+offset_v)]
    elif abs(x - x_13) < cube_size and abs(y - y_23) < cube_size and abs(z - z_13) < cube_height:
        return [power*(offset_p),speed]
    elif abs(x - x_13) < cube_size and abs(y - y_13) < cube_size and abs(z - z_23) < cube_height:
        return [power*(1-offset_p),speed*(1-offset_v)]
    elif abs(x - x_23) < cube_size and abs(y - y_13) < cube_size and abs(z - z_23) < cube_height:
        return [power,speed*(1-offset_v)]
    elif abs(x - x_23) < cube_size and abs(y - y_23) < cube_size and abs(z - z_23) < cube_height:
        return [power,speed]
    elif abs(x - x_13) < cube_size and abs(y - y_23) < cube_size and abs(z - z_23) < cube_height:
        return [power*(1-offset_p),speed]
    else:
        return [power,speed]

def write_neighborhood_paths_scode(
    layers: List[Any],
    models: List[Any],
    x: float,
    y: float,
    radius: float,
    zs: List[float],
    out_path: str,
    bids: List[int],
    island_pv: List[Any] = []
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
            
        idx = int(seq_map.get(geom, -1))
        if lidx<len(layers)-1 or idx<owner_idx-1:
            speed = 100000
            power = 160
        elif idx>owner_idx:
            speed = 100000
            power = 0
        else:
            #[power,speed] = island_to_random_power_speed(idx,power,speed)
            x,y = centroid_of(geom)
            if len(island_pv) and (speed <10 and power>0):
                [power,speed] = read_scode_pv(island_pv,idx)
                power = island_pv[0][idx]
                speed = island_pv[1][idx]*60
            [power,speed] = pos_to_power_speed(x,y,z,power,speed)
            #if (power != 320 and speed != 1):
            #    print(power,speed)
        count = 0
        for x1, y1, x2, y2 in _iter_segments(getattr(geom, "coords", None)) or []:
            x1, y1, z1 = calculate_torsion_position(x1, y1, z)
            x2, y2, z2 = calculate_torsion_position(x2, y2, z)
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
            owner_idx = int(seq_map.get(owner, -1))

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
                written += write_geom(nb)
    cx, cy = centroid_of(owner)
    rx, ry, rz = calculate_torsion_position(cx, cy, zs[-1])
    return written, int(seq_map.get(owner, -1)), (rx, ry)



def write_layer_island_info_scode(
    layer: Any,
    models: List[Any],
    z: float,
    out_path: str,
    island_index_base: int,
    re: bool,
    island_pv: List[Any] = []
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

    def choose_entry_exit(cur: Any, prev: Optional[Any], succ: Optional[Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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

        if prev is None and not (succ is None):
            pcx, pcy = centroid_of(succ)
            dists = [math.hypot(m[0] - pcx, m[1] - pcy) for m in mids]
            k = int(np.argmin(dists))
            e_out = mids[k]
            e_in = mids[(k + 2) % 4]
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
            succ = islands[i + 1] if i < len(islands)-1 else None
            e_in, e_out = choose_entry_exit(cur, prev, succ)
            
            x1, y1, z = calculate_torsion_position(e_in[0],e_in[1],z)
            x2, y2, z = calculate_torsion_position(e_out[0],e_out[1],z)

            bs = _resolve_buildstyle(cur, models)
            power = float(getattr(bs, "laserPower", 0.0) if bs is not None else 0.0)

            coords = getattr(cur, "coords", None)
            total_len = _path_length(coords)
            total_time = float(time_by_id.get(id(cur), 0.0))
            eq_speed = float(total_len / total_time) if total_time > 0.0 else 0.0

            idx = int(seq_map.get(cur, -1))
            #[power,eq_speed] = island_to_random_power_speed(idx,power,eq_speed)
            if len(island_pv):
                [power,eq_speed] = read_scode_pv(island_pv,idx)
            [power,eq_speed] = pos_to_power_speed(0.5*(e_in[0]+e_out[0]),0.5*(e_in[1]+e_out[1]),z,power,eq_speed)
            fh.write(
                f"{_format_float(x1*SCALE)} {_format_float(y1*SCALE)} {_format_float(x2*SCALE)} {_format_float(y2*SCALE)} "
                f"{_format_float(z*SCALE)} {_format_float(power)} {_format_float(round(eq_speed,10))} {_format_float(total_time)} {idx}\n"
            )
            written += 1

    return written
    
def get_island_zone_name(
    layer: Any,
    x: float,
    y: float,
    radius: float,
    bid: int
) -> int:
    index = None
    owner = None
    seq_map = {}
    index = IslandIndex(layer, neighbor_radius=radius)
    owner = index.find_island_at_point(x, y)
    if owner == None:
        return ""
    return owner.zoneName
