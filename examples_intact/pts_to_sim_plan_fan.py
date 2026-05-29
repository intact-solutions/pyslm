"""
Figure 1: Level 1 only — sequence-colored islands with scan paths for owner+neighbors

This script creates a single-axes plot that:
- Builds a target layer with IslandHatcher (groupIslands=True)
- Selects a point of interest
- Colors island outlines by sequence (coolwarm)
- Shows BOTH sequence index and per-island timing annotations inside islands
- Draws scan paths (hatches) ONLY for the owner island and its neighbors
"""
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Any
import os
import math
import random
from collections import namedtuple
from struct import unpack

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab as pl
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union

sns.set()

import pyslm
import pyslm.visualise
from pyslm import hatching as hatching
from pyslm.analysis.export_scode import ( 
    write_layer_island_info_scode, 
    calculate_torsion_position
)
from pyslm.analysis.zone_utils import build_zone_polygons, classify_layer_geometry
from pyslm.analysis.island_utils import (
    IslandIndex,
    get_island_geometries,
    compute_layer_geometry_times,
)

# ----------------------------
# Config
# ----------------------------
Z_TARGET = 14.99
SCALE = 1
SCAN_CONTOUR_FIRST = False
ISLAND_WIDTH = 2
NEIGHBOR_RADIUS_R = 0.8 * ISLAND_WIDTH
OWNER_SEQUENCE_INDEX_1BASED = 23

# Colors
COLOR_FILL_OWNER = '#66c2a5cc'
COLOR_FILL_NEIGHBOR = '#ffcc80cc'
COLOR_SEQ_LABEL = '#666666'
COLOR_OWNER_LINE = '#d62728'
COLOR_NEIGHBOR_LINE = '#1f77b4'
FONT_ISLAND_TIME = 5

Road = namedtuple("Road", "x1 y1 x2 y2 z laser_power laser_speed layer_num island_num")
FIRST_LAYER_IDX = 1
INIT_Z = 0.00
RES = 1e-7
STL_RES = 1e-7

def build_zone_parts(base_path: Path):
    zone_ply_paths = {
        "core": base_path / "core.ply",
        "blade1": base_path / "blade1.ply",
        "blade2": base_path / "blade2.ply",
        "blade3": base_path / "blade3.ply",
        "blade4": base_path / "blade4.ply",
        "blade5": base_path / "blade5.ply",
        "blade6": base_path / "blade6.ply",
        "blade7": base_path / "blade7.ply",
        "blade8": base_path / "blade8.ply",
        "blade9": base_path / "blade9.ply",
        "blade10": base_path / "blade10.ply",
        "blade11": base_path / "blade11.ply",
        "blade12": base_path / "blade12.ply"
    }

    zone_parts = {}
    for zone_name, ply_path in zone_ply_paths.items():
        if ply_path.exists():
            part = pyslm.Part(zone_name)
            part.setGeometry(str(ply_path))
            part.scaleFactor = SCALE
            zone_parts[zone_name] = part

    return zone_parts

def build_models():
    zone_bids = {
        "core": 1, "blade1": 2, "blade2": 3, "blade3": 4, "blade4": 5, "blade5": 6,
        "blade6": 7, "blade7": 8, "blade8": 9, "blade9": 10, "blade10": 11,
        "blade11": 12, "blade12": 13
    }
    contour_bid = 10
    zone_params = {z: {"power": 230, "speed": 2/20} for z in zone_bids.keys()}
        
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
    bs_contour.jumpSpeed = 5000.0
    model.buildStyles.append(bs_contour)

    return [model], zone_bids, contour_bid

def assign_model(layer, models):
    for g in getattr(layer, "geometry", []) or []:
        g.mid = models[0].mid

def _base_path() -> Path:
    return _repo_root / "geometry_intact" / "zone_fan"

def uniform_sampling_in_polygon(polygon, origin, spacing):
    minx, miny, maxx, maxy = polygon.bounds
    x0, y0 = origin
    xs = np.arange(x0, maxx, spacing)
    ys = np.arange(y0, maxy, spacing)

    pts = []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                pts.append((x, y))
    return np.array(pts)

def slice_mesh_to_polygons(stl_path, z_slice, ox, oy):
    mesh = trimesh.load_mesh(stl_path)
    mesh.apply_scale(SCALE)
    section = mesh.section(plane_origin=[ox, oy, z_slice], plane_normal=[0, 0, 1])
    if section is None:
        return []

    slice_2d, transform = section.to_2D()
    polygons_3d = []
    for path in slice_2d.polygons_full:
        coords_2d = np.array(path.exterior.coords)
        coords_h = np.hstack([coords_2d, np.zeros((len(coords_2d), 1)), np.ones((len(coords_2d), 1))])
        coords_3d = (transform @ coords_h.T).T[:, :3]

        interiors_3d = []
        for ring in path.interiors:
            ring_2d = np.array(ring.coords)
            ring_h = np.hstack([ring_2d, np.zeros((len(ring_2d), 1)), np.ones((len(ring_2d), 1))])
            ring_3d = (transform @ ring_h.T).T[:, :3]
            interiors_3d.append(ring_3d)

        polygons_3d.append(Polygon(coords_3d, [r[:, :3] for r in interiors_3d]))
    return polygons_3d

def roundN(num,n):
    return round(num/n)*n

def poly_boundingbox(poly):
    xmin, xmax = 999999, -999999
    ymin, ymax = 999999, -999999
    zmin, zmax = 999999, -999999
    for line in poly:
        for p in line:
            if p[0] > xmax: xmax = p[0]
            if p[0] < xmin: xmin = p[0]
            if p[1] > ymax: ymax = p[1]
            if p[1] < ymin: ymin = p[1]
            if p[2] > zmax: zmax = p[2]
            if p[2] < zmin: zmin = p[2]
    return xmin,xmax,ymin,ymax,zmin,zmax

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])) and (q[2] <= max(p[2], r[2]) and q[2] >= min(p[2], r[2]))

def do_intersect(seg1, seg2):
    segment1 = LineString(seg1)
    segment2 = LineString(seg2)
    intersection = segment1.intersection(segment2)
    if intersection.is_empty:
        return None
    else:
        return intersection.coords[0]

def IsInArray(item,array):
    for i in array:
        ret = True
        for c_i,c_item in zip(i,item):
            if not FloatEqual(c_i,c_item,5e-6):
                ret = False
                break
        if ret: return True
    return False

def GetPolyDiameter(poly):
    x, y = [], []
    for line in poly:
        p1, p2 = line[0], line[1]
        x.extend([p1[0], p2[0]])
        y.extend([p1[1], p2[1]])
    return np.linalg.norm(np.array([max(x)-min(x),max(y)-min(y)]))

def IsPointInPoly(pt,poly):
    poly_z = pt[2]
    pt_inf = [987654321,128765112,poly_z]
    n_intersect = 0
    for line in poly:
        if do_intersect(line,[pt,pt_inf]):
            n_intersect += 1
    return n_intersect % 2 == 1

def IsPointInPolys(pt,polys):
    for poly in polys:
        if IsPointInPoly(pt,poly): return True
    return False

def IsPolyInPoly(polyi,polyo):
    for line in polyi:
        for p in line:
            if not IsPointInPoly(p,polyo): return False
    return True

def BinarySTL(fname):
    fp = open(fname, 'rb')
    Header = fp.read(80)
    nn = fp.read(4)
    Numtri = unpack('i', nn)[0]
    record_dtype = np.dtype([
                   ('normals', np.float32,(3,)),  
                   ('Vertex1', np.float32,(3,)),
                   ('Vertex2', np.float32,(3,)),
                   ('Vertex3', np.float32,(3,)) ,              
                   ('atttr', '<i2',(1,) )
    ])
    data = np.fromfile(fp , dtype = record_dtype , count =Numtri)
    fp.close()
 
    Normals = data['normals']
    Vertex1= data['Vertex1']*SCALE
    Vertex2= data['Vertex2']*SCALE
    Vertex3= data['Vertex3']*SCALE
    
    p = np.append(Vertex1,Vertex2,axis=0)
    p = np.append(p,Vertex3,axis=0)
    Points =np.array(list(set(tuple(p1) for p1 in p)))
    return Header,Points,Normals,Vertex1,Vertex2,Vertex3
  
def BoundingBox(points):
    if len(points) == 0: return
    xmin = xmax = points[0][0]
    ymin = ymax = points[0][1]
    zmin = zmax = points[0][2]
    for p in points:
        x,y,z = p[0],p[1],p[2]
        if x<xmin: xmin = x
        if x>xmax: xmax = x
        if y<ymin: ymin = y
        if y>ymax: ymax = y
        if z<zmin: zmin = z
        if z>zmax: zmax = z
    return [xmin,xmax,ymin,ymax,zmin,zmax]

def VertexEqual(v1,v2,e=RES):
    if (len(v1) == 3):
        return FloatEqual(v1[0],v2[0],e) and FloatEqual(v1[1],v2[1],e) and FloatEqual(v1[2],v2[2],e)
    elif (len(v1) == 2):
        return FloatEqual(v1[0],v2[0],e) and FloatEqual(v1[1],v2[1],e) 
    else:
        assert(False)

def FloatEqual(a,b,e=RES):
    return abs(a-b)<e

def intersectZ(v1,v2,v3,z):
    z1, z2, z3 = v1[2], v2[2], v3[2]
    n_zup = n_on = 0 
    v_up, v_down, v_on = [], [], []
    
    for v_z, v in [(z1, v1), (z2, v2), (z3, v3)]:
        if FloatEqual(v_z, z):
            n_on += 1
            v_on.append(v)
        elif v_z > z:
            n_zup += 1
            v_up.append(v)
        else:
            v_down.append(v)

    if n_zup == 3 or (n_zup == 0) or (n_on == 1 and (n_zup == 2 or n_zup == 2)):
        return None
    if (n_on == 2):
        return [(v_on[0][0], v_on[0][1],z), (v_on[1][0], v_on[1][1],z)]
    if (n_on == 1):
        t = (z-v_down[0][2])/(v_up[0][2]-v_down[0][2])
        return [(v_on[0][0], v_on[0][1],z), ((1-t)*v_down[0][0]+t*v_up[0][0],(1-t)*v_down[0][1]+t*v_up[0][1],z)]
    if (n_zup == 2):
        t1 = (z-v_down[0][2])/(v_up[0][2]-v_down[0][2])
        t2 = (z-v_down[0][2])/(v_up[1][2]-v_down[0][2])
        return [((1-t1)*v_down[0][0]+(t1)*v_up[0][0],(1-t1)*v_down[0][1]+(t1)*v_up[0][1],z),((1-t2)*v_down[0][0]+(t2)*v_up[1][0],(1-t2)*v_down[0][1]+(t2)*v_up[1][1],z)] 
    if (n_zup == 1):
        t1 = (z-v_down[0][2])/(v_up[0][2]-v_down[0][2])
        t2 = (z-v_down[1][2])/(v_up[0][2]-v_down[1][2])
        return [((1-t1)*v_down[0][0]+(t1)*v_up[0][0],(1-t1)*v_down[0][1]+(t1)*v_up[0][1],z),((1-t2)*v_down[1][0]+(t2)*v_up[0][0],(1-t2)*v_down[1][1]+(t2)*v_up[0][1],z)] 
    assert(False)
    return None

def shuffle_with_indices(arr):
    shuffled_indices = np.arange(len(arr))
    np.random.shuffle(shuffled_indices)
    shuffled_array = arr[shuffled_indices]
    return shuffled_array, shuffled_indices

def get_neihbor_id(center):
    if center[0]==0 or center[0]==4 or center[1]==0 or center[1]==4: return None
    return [np.array(center)+i for i in [[0,1],[1,1],[-1,1],[0,-1],[1,-1],[-1,-1],[1,0],[-1,0]]]

def plot_roads_2D(roads):
    fig, ax = plt.subplots(figsize=(8, 8))
    for x1, y1, x2, y2, z, *_ in roads:
        ax.plot([x1, x2], [y1, y2], 'b-')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    return fig, ax

def write_roads_to_file(roads, path, ncols=7):
    assert(roads and len(roads[0]) >= ncols)
    with open(path, 'w') as out_file:
        out_file.write('# x1 y1 x2 y2 z power speed island\n')
        for road in roads:
            line = ' '.join(map(str, [round(i,6) for i in road[:ncols-1]]))+" "+' '.join(map(str, [round(road[ncols])]))
            out_file.write(line + '\n')
    print("write {:d} roads to {:s}".format(len(roads), path))

def generate_grid(xmin,xmax,ymin,ymax,zmin,zmax, grid_size, layer_thickness):
    grid_dict = {}
    z = zmin
    while z <= zmax:
        layer_points = []
        y = ymin
        while y <= ymax:
            x = xmin
            while x <= xmax:
                layer_points.append([x, y])
                x += grid_size
            y += grid_size
        grid_dict[round(z, 6)] = layer_points
        z += layer_thickness
    return grid_dict

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_island_idx(owner: Any, layer: Any, base: int = 0):
    iss = get_island_geometries(layer)
    for i, g in enumerate(iss):
        if (g.islandId == owner.islandId):
            return base + i
    return -1

# ---------------------------------------------------------------------
# Adaptive Sampling Helpers
# ---------------------------------------------------------------------
def position_to_coord(p, i_size, layer_thickness, o):
    r = [p[0]-o[0], p[1]-o[1], p[2]-o[2]]
    return [int(r[0]/i_size), round(r[1]/i_size), round(r[2]/layer_thickness)]

def build_distance_field(polygons, step=0.001):
    if not polygons: return None, None, None
    polys_2d = []
    for p in polygons:
        coords = np.array(p.exterior.coords)
        if coords.shape[1] == 3:
            ext2d = coords[:, :2]
            ints2d = [np.array(r.coords)[:, :2] for r in p.interiors]
            polys_2d.append(Polygon(ext2d, ints2d))
        else:
            polys_2d.append(p)

    union_poly = unary_union(polys_2d)
    minx, miny, maxx, maxy = union_poly.bounds

    nx = max(2, int((maxx - minx) / step) + 1)
    ny = max(2, int((maxy - miny) / step) + 1)
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    xx, yy = np.meshgrid(xs, ys)

    coords_grid = np.column_stack((xx.ravel(), yy.ravel()))
    mask = np.array([union_poly.contains(Point(x, y)) for x, y in coords_grid])
    mask_img = mask.reshape(ny, nx)
    dist_img = distance_transform_edt(mask_img) * step

    return dist_img, mask_img, (minx, miny, step)

def query_distances(points, dist_img, mask_img, grid_params):
    if dist_img is None: return np.full(len(points), -1.0)
    minx, miny, step = grid_params
    ny, nx = mask_img.shape
    distances = []

    for px, py in points:
        ix = int((px - minx) / step)
        iy = int((py - miny) / step)
        if 0 <= ix < nx and 0 <= iy < ny and mask_img[iy, ix]:
            distances.append(dist_img[iy, ix])
        else:
            distances.append(-1.0)
    return np.array(distances)


if __name__ == "__main__":
    
    OUTDIR = Path(__file__).resolve().parent
    
    plot_island_size = 3
    layer_thickness = 0.1*SCALE
    hatch_space = 0.1*SCALE
    fname = "Fan_untwisted"
    q2_path = OUTDIR / (fname+"_layer.scode")
    island_path = OUTDIR / (fname+".scode")
    
    base_path = _base_path()
    zone_parts = build_zone_parts(base_path)

    models, zone_bids, contour_bid = build_models()
    zone_priority = ["core","blade6","blade7","blade8","blade9","blade10","blade11","blade12","blade1","blade2","blade3","blade4","blade5"]

    # -------------------------------------------------------------
    # NEW PER-ZONE HATCHER SETUP
    # Assign specific rotation/hatch angles to each geometric zone
    # -------------------------------------------------------------
    zone_hatchers = {}
    angle_offset = 90 
    
    for zone in zone_priority:
        h = hatching.IslandHatcher()
        h.islandWidth = ISLAND_WIDTH
        h.islandOverlap = 0
        h.hatchAngle = angle_offset  # <-- Each zone gets a unique rotation
        h.volumeOffsetHatch = 0
        h.spotCompensation = 0
        h.numInnerContours = 0
        h.numOuterContours = 0
        h.layerAngleIncrement = 0
        h.hatchDistance = hatch_space
        h.hatchSortMethod = hatching.AlternateSort()
        h.groupIslands = True
        
        zone_hatchers[zone] = h
        angle_offset -= 30.0 # Example: Increment angle by 15 degrees per zone

    original_stl = base_path / "Fan_untwisted.STL"
    solidPart = pyslm.Part("Fan_untwisted")
    solidPart.setGeometry(str(original_stl))
    solidPart.scaleFactor = SCALE
    solidPart.dropToPlatform()
    
    [xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
    print(xmin,ymin,zmin,xmax,ymax,zmax)
    
    island_dict = {}
    n_island = 0
    all_islands = []
    
    # -------------------------------------------------------------
    # NEW PER-ZONE Z-LOOP
    # Slice independently and merge into a master layer
    # -------------------------------------------------------------
    for z in np.arange(zmin, zmax, layer_thickness):
        master_layer = pyslm.geometry.Layer()
        
        # 1. Slice and hatch each zone independently
        for zone_name in zone_priority:
            if zone_name in zone_parts:
                part = zone_parts[zone_name]
                geomSlice = part.getVectorSlice(z + 1e-5, simplificationFactor=0.1)
                if len(geomSlice):
                    zone_layer = zone_hatchers[zone_name].hatch(geomSlice)
                    master_layer.geometry.extend(zone_layer.geometry)
        
        # 2. Classify and apply bids
        zone_polys = build_zone_polygons(zone_parts, float(z))
        classify_layer_geometry(
            master_layer,
            zone_polys,
            zone_bids,
            contour_bid=contour_bid,
            default_zone="core",
            priority=zone_priority,
        )
        assign_model(master_layer, models)
        
        # 3. Island extraction and indexing using physical centroids
        islands = get_island_geometries(master_layer)
        layer_idx = round(z/layer_thickness)
        
        for islandId, island in enumerate(islands):
            if layer_idx not in island_dict:
                index = IslandIndex(master_layer, neighbor_radius=NEIGHBOR_RADIUS_R)
                x_center = island.boundaryPoly.centroid.x
                y_center = island.boundaryPoly.centroid.y
                island_dict[layer_idx] = {"layer":index,"bid":n_island,"pts":[{"coord":[x_center,y_center,z],"island":island,"id":islandId}]}
                all_islands.append(islandId+n_island)
            else:
                x_center = island.boundaryPoly.centroid.x
                y_center = island.boundaryPoly.centroid.y
                if (islandId not in [i["id"] for i in island_dict[layer_idx]["pts"]]):
                    island_dict[layer_idx]["pts"].append({"coord":[x_center,y_center,z],"island":island,"id":islandId}) 
                    all_islands.append(islandId+n_island)
            print(x_center,y_center,z)        
        n_island += write_layer_island_info_scode(master_layer, models, z, str(island_path), island_index_base=n_island, re=False)
        #print(f"Layer Z={z:.3f} | Total islands written: {n_island}")

    #print("Islands generated across layers:", list(island_dict.keys()))

    head,points,normals,vs1,vs2,vs3 = BinarySTL(fname+'.STL')
    points = np.vstack([vs1, vs2, vs3])
    faces = np.arange(len(points)).reshape(-1, 3)
    length = xmax - xmin
    width = ymax - ymin
    island_size = ISLAND_WIDTH
    origin = np.array([xmin,ymin,zmin])

    all_sim_layers = []
    all_sim_islands = []
    all_sim_paths = []
    N_temp = 2 # temporal neighborhood
    N_spatial = 0.9
    N_block = 1000000
    
    # Ensure point storage directory exists
    os.makedirs('./pts', exist_ok=True)
    
    point_of_interest = {}

    # -------------------------------------------------------------
    # UNIFORM SAMPLING
    # -------------------------------------------------------------
    init_grids1 = generate_grid(xmin, xmax, ymin, ymax, zmin, zmax, island_size, 5*layer_thickness)

    for z, coords in init_grids1.items():
            
        polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)

        for coord in coords:
            p = [coord[0], coord[1], z]
            rr = np.linalg.norm(coord)
            angle = np.arctan2(coord[1], coord[0])
            if rr < 40 or abs(angle+np.pi/6) > (np.pi / 12):
                continue
            if any(poly.contains(Point(p)) for poly in polygons):
                if z not in point_of_interest:
                    point_of_interest[z] = []
                if coord not in point_of_interest[z]:
                    point_of_interest[z].append(coord)
    
    for z, positions in point_of_interest.items():
        for pos in positions:
            print(pos[0]/1000,pos[1]/1000,z/1000)
    asdds
    # -------------------------------------------------------------
    # ADAPTIVE SAMPLING
    # -------------------------------------------------------------
    init_grids2 = generate_grid(xmin, xmax, ymin, ymax, zmin, zmax, island_size, 40*layer_thickness)
    
    for z, coords in init_grids2.items():
        polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)
        if not polygons: continue
        
        dist_img, mask_img, grid_params = build_distance_field(polygons, step=1*SCALE)
        if dist_img is None: continue

        dists = query_distances(coords, dist_img, mask_img, grid_params)
        for coord, dist in zip(coords, dists):
            p = [coord[0], coord[1], z]
            if dist < 0:
                continue
            
            n = round(dist / ISLAND_WIDTH)
            if not (n <= 1):
                continue
            
            if z not in point_of_interest:
                point_of_interest[z] = []
            if coord not in point_of_interest[z]:
                point_of_interest[z].append(coord)

    # Plot Point of Interest Maps
    for z, coords in point_of_interest.items():
        polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)
        plt.scatter([xyz[0] for xyz in coords], [xyz[1] for xyz in coords], s=4, c='blue', alpha=0.4)       
        for poly in polygons:
            if poly.is_empty: continue
            coords2d = np.array(poly.exterior.coords)[:, :2] if np.array(poly.exterior.coords).shape[1] == 3 else np.array(poly.exterior.coords)
            plt.plot(coords2d[:, 0], coords2d[:, 1], 'k-', linewidth=1)
            
            for interior in poly.interiors:
                hole = np.array(interior.coords)
                plt.plot(hole[:, 0], hole[:, 1], 'r--', linewidth=0.8)
                
        plt.axis('equal')
        plt.title("Sampled Points of Interest")
        plt.savefig(f'./pts/sampling_{round(z,6)}.png')
        plt.clf()
      
    for z, positions in point_of_interest.items():
        for pos in positions:
            print(pos[0]/1000,pos[1]/1000,z/1000)
    
    # 3D Plot of points of interest
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_x = [pos[0] for z, pos_list in point_of_interest.items() for pos in pos_list]
    plot_y = [pos[1] for z, pos_list in point_of_interest.items() for pos in pos_list]
    plot_z = [z for z, pos_list in point_of_interest.items() for pos in pos_list]
    
    lc = ax.scatter(plot_x, plot_y, plot_z, s=plot_island_size, c='lime', marker='s')
    set_axes_equal(ax)
    plt.savefig('3d.png')
    plt.close()

    # -------------------------------------------------------------
    # SPATIAL NEIGHBORHOOD VIA KD-TREE
    # -------------------------------------------------------------
    trees = {}
    for z, positions in point_of_interest.items():
        layer_idx = round(z/layer_thickness)
        if layer_idx not in island_dict:
            continue
            
        for pos in positions:
            p = [pos[0],pos[1],z]
            if z not in trees:
                trees[z] = cKDTree([item["coord"] for item in island_dict[layer_idx]["pts"]])
                
            dist, idx = trees[z].query(p)
            island_num_in_layer = island_dict[layer_idx]["pts"][idx]["id"]
            island_num = island_num_in_layer + island_dict[layer_idx]["bid"]
            
            if island_num not in all_sim_paths:
                all_sim_layers.append(z-layer_thickness)
                sim_path_list = [island_num]
                
                # Fetch adjacent islands directly using shapely geometry rather than integer grids
                neighbor_islands = island_dict[layer_idx]["layer"].neighbors_for_island(
                    island_dict[layer_idx]["pts"][idx]["island"], 
                    ISLAND_WIDTH * N_spatial
                )
                
                n_list = [i.islandId + island_dict[layer_idx]["bid"] for i in neighbor_islands]
                
                sim_path_list += [i for i in n_list if ((island_num-i) < N_temp and island_num >= i)]
                sim_island_list = [i for i in n_list if (i not in sim_path_list)]
                
                all_sim_paths += sim_path_list
                all_sim_islands += sim_island_list
                
                start_i = min(sim_path_list)
                end_i = max(sim_path_list)
                print(start_i, end_i, "true", n_list)

    all_sim_islands = list(set(all_sim_islands))
    all_sim_paths = list(set(all_sim_paths))
    all_sim_islands = [i for i in all_sim_islands if i not in all_sim_paths]

    for z, positions in point_of_interest.items():
        fig = plt.figure() 
        ax = fig.add_subplot()
        plot_x = [pos[0] for pos in positions]
        plot_y = [pos[1] for pos in positions]
        lc = ax.scatter(plot_x, plot_y, s=plot_island_size, c='lime', marker='s')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        plt.savefig(f'LOI_{round(z,6)}.png')
        plt.close()

    non_block_islands = sorted(all_sim_paths)
    block_idx = 0
    last_block_island = -1
    plot_x = []
    
    sorted_all_islands = sorted(all_islands)
    block_island_begin = -1
    block_island_current = -1
    
    for idx in range(len(sorted_all_islands)):
        island_idx = sorted_all_islands[idx]
        if island_idx not in non_block_islands:
            block_island_current = island_idx
            if block_island_begin < 0:
                block_island_begin = island_idx
            if len(plot_x) < N_block and idx < len(sorted_all_islands)-1:
                plot_x.append(island_idx)
            else:
                print( block_island_begin, island_idx, "false", [])
                block_island_begin = -1
                plot_x = [island_idx]
                block_idx += 1
        elif len(plot_x) != 0:
            if block_island_begin >= 0:
                print(block_island_begin, block_island_current, "false", [])
                block_island_begin = -1
            if island_idx in all_sim_islands:
                print(island_idx, island_idx, "false", [])
            plot_x = []
            block_idx += 1