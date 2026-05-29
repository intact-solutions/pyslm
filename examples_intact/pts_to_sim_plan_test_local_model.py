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
from scipy.spatial import cKDTree

from typing import Any
# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
	sys.path.insert(0, str(_repo_root))


import pyslm
from pyslm.analysis.export_scode import write_layer_island_info_scode

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
SCALE = 0.001
SCAN_CONTOUR_FIRST = False  # available if needed by your IslandHatcher setup
ISLAND_WIDTH = 2*SCALE
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


myHatcher = hatching.IslandHatcher()
myHatcher.islandWidth = ISLAND_WIDTH
myHatcher.islandOverlap = 0
myHatcher.hatchAngle = 0
myHatcher.volumeOffsetHatch = 0
myHatcher.spotCompensation = 0
myHatcher.numInnerContours = 0
myHatcher.numOuterContours = 0
myHatcher.hatchDistance = 1e-4
myHatcher.hatchSortMethod = hatching.AlternateSort()
myHatcher.groupIslands = True
def gen_island_slices(filename,layer_thickness):
	solidPart = pyslm.Part(filename)
	solidPart.setGeometry(filename+'.STL')
	
	solidPart.dropToPlatform()
	solidPart.origin[0] = 0.0
	solidPart.origin[1] = 0.0
	solidPart.scaleFactor = SCALE
	solidPart.rotation = [0, 0.0, np.pi]
	

	[xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
	#print(xmin,ymin,zmin,xmax,ymax,zmax)
	geomSlices = []
	layers = []
	zs = []
	for z in np.arange(zmin, zmax, layer_thickness):
		geomSlice = solidPart.getVectorSlice(z)
		#print(z)
		layer = myHatcher.hatch(geomSlice)
		geomSlices.append(geomSlice)
		layers.append(layer)
		zs.append(z)
	return geomSlices, layers, zs

def build_minimal_models():
	bstyle = pyslm.geometry.BuildStyle()
	bstyle.bid = 1
	bstyle.laserSpeed = 0.05
	bstyle.laserPower = 105
	bstyle.jumpSpeed = 5000.0

	model = pyslm.geometry.Model()
	model.mid = 1
	model.buildStyles.append(bstyle)

	return [model]
def assign_model(layer):
    for g in getattr(layer, 'geometry', []) or []:
        g.mid = models[0].mid
        g.bid = models[0].buildStyles[0].bid


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
# This module contains functions that
# can generate scode
##################################

##################################
# modified by Xin Liu
# 05/26/2021
# _generate_raster_path_helper1() is modified from _generate_raster_path_helper
# break the path into ngroup subpaths
# remove transverse scanning
##################################

# standard library
import sys
import copy
from collections import namedtuple
import math
import random
# 3rd party library
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from shapely.ops import unary_union

sns.set()

Road = namedtuple("Road", "x1 y1 x2 y2 z laser_power laser_speed layer_num island_num")
FIRST_LAYER_IDX = 1
INIT_Z = 0.00

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import pylab as pl

from struct import unpack
import trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString, Point


def uniform_sampling_in_polygon(polygon, origin, spacing):
	"""
	Uniformly sample points inside a polygon, starting from given origin.
	
	polygon: shapely Polygon (can include holes)
	origin: (x0, y0)
	spacing: float (grid spacing)
	"""
	minx, miny, maxx, maxy = polygon.bounds

	# Align grid to origin
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
	"""
	Slice a 3D mesh by a plane z=z_slice and return polygons (Shapely) 
	located at their correct 3D positions.
	"""
	# Load mesh and scale
	mesh = trimesh.load_mesh(stl_path)
	mesh.apply_scale(SCALE)

	# Slice the mesh
	section = mesh.section(plane_origin=[ox, oy, z_slice], plane_normal=[0, 0, 1])
	if section is None:
		return []

	# Convert to 2D (section.to_2D() returns transform matrix back to 3D)
	slice_2d, transform = section.to_2D()

	polygons_3d = []
	for path in slice_2d.polygons_full:
		coords_2d = np.array(path.exterior.coords)
		coords_h = np.hstack([coords_2d, np.zeros((len(coords_2d), 1)), np.ones((len(coords_2d), 1))])
		coords_3d = (transform @ coords_h.T).T[:, :3]

		# Build 3D polygon including holes
		interiors_3d = []
		for ring in path.interiors:
			ring_2d = np.array(ring.coords)
			ring_h = np.hstack([ring_2d, np.zeros((len(ring_2d), 1)), np.ones((len(ring_2d), 1))])
			ring_3d = (transform @ ring_h.T).T[:, :3]
			interiors_3d.append(ring_3d)

		polygons_3d.append(Polygon(coords_3d, [r[:, :3] for r in interiors_3d]))

	return polygons_3d

def point_distance_to_polygons(polygons, point, step=0.001):
	"""
	Compute distance from a 3D point [x, y, z] to the boundary
	of a set of 2D/3D shapely polygons.
	
	Returns:
		distance (float): distance to boundary if inside polygon,
						  -1 if outside all polygons.
	"""
	if not polygons:
		return -1.0

	# --- Project to XY plane if polygons are 3D ---
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
	
	if not any(poly.contains(Point(point)) for poly in polygons):
		return -1.0

	# --- Rasterize the polygon for distance transform ---
	minx, miny, maxx, maxy = union_poly.bounds
	nx = max(2, int((maxx - minx) / step) + 1)
	ny = max(2, int((maxy - miny) / step) + 1)

	xs = np.linspace(minx, maxx, nx)
	ys = np.linspace(miny, maxy, ny)
	xx, yy = np.meshgrid(xs, ys)
	coords_grid = np.column_stack((xx.ravel(), yy.ravel()))

	mask = np.array([union_poly.contains(Point(x, y)) for x, y in coords_grid])
	mask_img = mask.reshape(ny, nx)

	# --- Distance transform (distance to boundary for inside points) ---
	dist_img = distance_transform_edt(mask_img) * step

	# --- Sample distance at point location ---
	ix = np.clip(int((point[0] - minx) / step), 0, nx - 1)
	iy = np.clip(int((point[1] - miny) / step), 0, ny - 1)

	return float(dist_img[iy, ix])



def build_distance_field(polygons, step=0.001):
	"""
	Build a distance field (distance-to-boundary) inside the union of polygons.
	Returns:
		dist_img: 2D numpy array of distances
		mask_img: boolean mask (True inside polygon)
		(minx, miny, step): grid parameters for later querying
	"""
	if not polygons:
		return None, None, None

	# Project to XY plane if necessary
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

	# Grid dimensions
	nx = max(2, int((maxx - minx) / step) + 1)
	ny = max(2, int((maxy - miny) / step) + 1)

	xs = np.linspace(minx, maxx, nx)
	ys = np.linspace(miny, maxy, ny)
	xx, yy = np.meshgrid(xs, ys)

	# Fast rasterization
	coords_grid = np.column_stack((xx.ravel(), yy.ravel()))
	mask = np.array([union_poly.contains(Point(x, y)) for x, y in coords_grid])
	mask_img = mask.reshape(ny, nx)

	# Compute distance inside
	dist_img = distance_transform_edt(mask_img) * step

	return dist_img, mask_img, (minx, miny, step)

def query_distances(points, dist_img, mask_img, grid_params):
	"""
	Query distances for multiple points given a precomputed distance field.
	Returns an array of distances, -1 for outside points.
	"""
	if dist_img is None:
		return np.full(len(points), -1.0)

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

def fast_adaptive_sampling(polygon, step=0.005, n_samples=5000, decay=3.0):
	"""
	Fast adaptive sampling inside a polygon using a distance field.
	Density increases near the boundary.
	
	Parameters:
		polygon : shapely.geometry.Polygon
		step : float — grid spacing (smaller = more precise but slower)
		n_samples : int — number of points to sample
		decay : float — exponential decay factor for weighting
	Returns:
		np.ndarray (N, 2) sampled points
	"""
	# --- Step 1: Rasterize polygon onto a regular grid ---
	minx, miny, maxx, maxy = polygon.bounds
	nx = int((maxx - minx) / step)
	ny = int((maxy - miny) / step)

	xs = np.linspace(minx, maxx, nx)
	ys = np.linspace(miny, maxy, ny)
	xx, yy = np.meshgrid(xs, ys)
	coords = np.vstack((xx.ravel(), yy.ravel())).T

	# Binary mask: 1 inside, 0 outside
	mask = np.array([polygon.contains(Point(x, y)) for x, y in coords])
	mask_img = mask.reshape(ny, nx)

	# --- Step 2: Distance transform ---
	dist_img = distance_transform_edt(mask_img) * step  # convert to world distance

	# --- Step 3: Compute sampling weights ---
	weights = 1.0 / (dist_img + 1e-4)**2        # inverse-square falloff

	weights[mask_img == 0] = 0  # zero outside polygon
	weights /= weights.sum()

	# --- Step 4: Sample indices according to weights ---
	flat_idx = np.random.choice(weights.size, size=n_samples, p=weights.ravel())
	iy, ix = np.unravel_index(flat_idx, weights.shape)
	xs_samp, ys_samp = xs[ix], ys[iy]

	return np.column_stack((xs_samp, ys_samp))

def roundN(num,n):
	return round(num/n)*n
def poly_boundingbox(poly):
	xmin = 999999
	xmax = -999999
	ymin = 999999
	ymax = -999999
	zmin = 999999
	zmax = -999999
	for line in poly:
		p1 = line[0]
		p2 = line[1]
		if p1[0]>xmax:
			xmax = p1[0]
		if p1[0]<xmin:
			xmax = p1[0]
		if p1[1]>ymax:
			ymax = p1[1]
		if p1[1]<ymin:
			ymin = p1[1]
		if p1[2]>zmax:
			zmax = p1[2]
		if p1[2]<zmin:
			zmin = p1[2]
		if p2[0]>xmax:
			xmax = p2[0]
		if p2[0]<xmin:
			xmax = p2[0]
		if p2[1]>ymax:
			ymax = p2[1]
		if p2[1]<ymin:
			ymin = p2[1]
		if p2[2]>zmax:
			zmax = p2[2]
		if p2[2]<zmin:
			zmin = p2[2]
	return xmin,xmax,ymin,ymax,zmin,zmax
def orientation(p, q, r):
	# Function to determine the orientation of three points (p, q, r).
	# Returns 0 if they are collinear, 1 if clockwise, and 2 if counterclockwise.
	val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
	if val == 0:
		return 0  # Collinear
	return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def on_segment(p, q, r):
	# Function to check if point q lies on line segment pr.
	return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])) and (q[2] <= max(p[2], r[2]) and q[2] >= min(p[2], r[2]))

def do_intersect(seg1, seg2):
	segment1 = LineString(seg1)
	segment2 = LineString(seg2)

	intersection = segment1.intersection(segment2)
	if intersection.is_empty:
		return None
	else:
		return intersection.coords[0]
		

RES = 1e-7
STL_RES = 1e-7

def IsInArray(item,array):
	for i in array:
		ret = True
		for c_i,c_item in zip(i,item):
			if not FloatEqual(c_i,c_item,5e-6):
				ret = False
				break
		if ret:
			return True
	return False
def GetPolyDiameter(poly):
	x = []
	y = []
	for line in poly:
		p1 = line[0]
		p2 = line[1]
		x.append(p1[0])
		x.append(p2[0])
		y.append(p1[1])
		y.append(p2[1])
	return np.linalg.norm(np.array([max(x)-min(x),max(y)-min(y)]))
# can only tackle legal poly 
def IsPointInPoly(pt,poly):
	poly_z = pt[2]
	pt_inf = [987654321,128765112,poly_z]
	n_intersect = 0
	for line in poly:
		if do_intersect(line,[pt,pt_inf]):
			n_intersect += 1
	if n_intersect%2 == 1:
		return True
	else:
		return False
def IsPointInPolys(pt,polys):
	for poly in polys:
		if IsPointInPoly(pt,poly):
			return True
	return False
# can only tackle legal poly 
def IsPolyInPoly(polyi,polyo):
	for line in polyi:
		for p in line:
			if not IsPointInPoly(p,polyo):
				return False
	return True
	
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 06:37:35 2013
 
@author: Sukhbinder Singh
 
Reads a Binary file and 
Returns Header,Points,Normals,Vertex1,Vertex2,Vertex3
 
"""

def BinarySTL(fname):
	'''Reads a binary STL file '''
	fp = open(fname, 'rb')
	Header = fp.read(80)
	nn = fp.read(4)
	Numtri = unpack('i', nn)[0]
	#print 'Number of triangles in the STL file: ',nn
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
	p = np.append(p,Vertex3,axis=0) #list(v1)
	Points =np.array(list(set(tuple(p1) for p1 in p)))
	
	''' 
	# rotate orientation
	
	Normals = [[i[0],i[2],i[1]] for i in Normals ]
	Points = [[i[0],i[2],i[1]] for i in Points ]
	Vertex1 = [[i[0],i[2],i[1]] for i in Vertex1 ]
	Vertex2 = [[i[0],i[2],i[1]] for i in Vertex2 ]
	Vertex3 = [[i[0],i[2],i[1]] for i in Vertex3 ]
	'''
	return Header,Points,Normals,Vertex1,Vertex2,Vertex3
  
def GetFilePath(fileName):
	return os.path.join(os.path.dirname(os.path.realpath(__file__)), fileName)

def BoundingBox(points):
	if len(points) == 0:
		return

	xmin = points[0][0]
	xmax = points[0][0]
	ymin = points[0][1]
	ymax = points[0][1]
	zmin = points[0][2]
	zmax = points[0][2]
	
	for p in points:
		x,y,z = p[0],p[1],p[2]
		if x<xmin:
			xmin = x
		if x>xmax:
			xmax = x
		if y<ymin:
			ymin = y
		if y>ymax:
			ymax = y
		if z<zmin:
			zmin = z
		if z>zmax:
			zmax = z
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
	z1 = v1[2]
	z2 = v2[2]
	z3 = v3[2]
	n_zup = 0
	n_on = 0 
	v_up = []
	v_down = []
	v_on = []
	if FloatEqual(z1,z):
		n_on = n_on + 1
		v_on.append(v1)
	elif z1>z:
		n_zup = n_zup + 1
		v_up.append(v1)
	else:
		v_down.append(v1)

	if FloatEqual(z2,z):
		n_on = n_on + 1
		v_on.append(v2)
	elif z2>z:
		n_zup = n_zup + 1
		v_up.append(v2)
	else:
		v_down.append(v2)

	if FloatEqual(z3,z):
		n_on = n_on + 1
		v_on.append(v3) 
	elif z3>z:
		n_zup = n_zup + 1
		v_up.append(v3)
	else:
		v_down.append(v3)

	'''
	# ignore the triangles on z_plane
	if n_zup == 3 or (n_zup == 0 and n_on!=3) or (n_on == 1 and (n_zup == 2 or n_zup == 2)):
		return None
	if (n_on == 3):
		return [(v1[0],v1[1]),(v2[0],v2[1]),(v3[0],v3[1])]
	'''
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
		#print(v_down[0],v_up[0],v_up[1],z,[((1-t1)*v_down[0][0]+(t1)*v_up[0][0],(1-t1)*v_down[0][1]+(t1)*v_up[0][1]),((1-t2)*v_down[0][0]+(t2)*v_up[1][0],(1-t2)*v_down[0][1]+(t2)*v_up[1][1])])
		return [((1-t1)*v_down[0][0]+(t1)*v_up[0][0],(1-t1)*v_down[0][1]+(t1)*v_up[0][1],z),((1-t2)*v_down[0][0]+(t2)*v_up[1][0],(1-t2)*v_down[0][1]+(t2)*v_up[1][1],z)] 
	if (n_zup == 1):
		t1 = (z-v_down[0][2])/(v_up[0][2]-v_down[0][2])
		t2 = (z-v_down[1][2])/(v_up[0][2]-v_down[1][2])
		return [((1-t1)*v_down[0][0]+(t1)*v_up[0][0],(1-t1)*v_down[0][1]+(t1)*v_up[0][1],z),((1-t2)*v_down[1][0]+(t2)*v_up[0][0],(1-t2)*v_down[1][1]+(t2)*v_up[0][1],z)] 
	assert(False)
	return None
'''
@author: Xin Liu

'''
def stl2poly_p_and_v(fname,hatch_space,layer_thickness,plane_z,points,vs1,vs2,vs3):
	#[xmin,xmax,ymin,ymax,zmin,zmax] = BoundingBox(points)
	'''
	("# ",len(points), " points")
	print("# ",len(vs1), " triangles")
	
	print("xmin: ",xmin, " xmax: ",xmax)
	print("ymin: ",ymin, " ymax: ",ymax)
	print("zmin: ",zmin, " zmax: ",zmax)
	'''

	output_array = []
	thickness = layer_thickness
	hatch_space = hatch_space
	angle = 0.0
	d_angle = 0#*np.pi/2
	
	# all line intersect with plane_z
	intersect_lines = []
	for v1,v2,v3 in zip(vs1,vs2,vs3):	
		intersect_line = intersectZ(v1,v2,v3,plane_z)
		if intersect_line != None and len(intersect_line) == 2:
			intersect_lines.append(intersect_line)
		elif intersect_line != None and len(intersect_line) == 3:
			intersect_lines.append([intersect_line[0],intersect_line[1]])
			intersect_lines.append([intersect_line[1],intersect_line[2]])
			intersect_lines.append([intersect_line[0],intersect_line[2]])
	'''
	# plot 
	for line in intersect_lines:
		vtx0 = line[0]
		vtx1 = line[1]
		NL = 10
		plt.plot(np.linspace(vtx0[0],vtx1[0],NL),np.linspace(vtx0[1],vtx1[1],NL))

	plt.savefig('./slices/'+str(round(plane_z,6))+'.png')
	pl.close()
	'''
	# polygon list of the largest covering polygon (in plane_z)
	polys = []
	# line list of all line segs of and inside the covering polygon (in plane_z)
	lines = []
	_temp_polys = []
	# construct the polyons
	while len(intersect_lines):
		line = intersect_lines.pop()
		startp = line[0]
		endp = line[1]
		poly = [line]
		# find a poly (loop) connect to line seg (startp, endp)
		_nmax = 0
		while not VertexEqual(startp,endp,STL_RES):	
			if _nmax>9999:
				print('too many lines: stl file resolution too high?')
				assert(False)
			else:
				_nmax = _nmax+1
			find_next = False
			min_dist = 10000
			for l in intersect_lines:
				dist = np.linalg.norm(np.array(l[0])-np.array(startp))
				if dist<min_dist:
					min_dist = dist
				dist = np.linalg.norm(np.array(l[1])-np.array(startp))
				if dist<min_dist:
					min_dist = dist
				dist = np.linalg.norm(np.array(l[0])-np.array(endp))
				if dist<min_dist:
					min_dist = dist
				dist = np.linalg.norm(np.array(l[1])-np.array(endp))
				if dist<min_dist:
					min_dist = dist
					
				if VertexEqual(l[0],startp,STL_RES):
					startp = l[1]
					poly.insert(0,[l[1],l[0]])
					intersect_lines.remove(l)
					find_next = True
					break
				if VertexEqual(l[1],startp,STL_RES):
					startp = l[0]
					poly.insert(0,l)
					intersect_lines.remove(l)
					find_next = True
					break
				if VertexEqual(l[0],endp,STL_RES):
					endp = l[1]
					poly.append(l)
					intersect_lines.remove(l)
					find_next = True
					break
				if VertexEqual(l[1],endp,STL_RES):
					endp = l[0]
					poly.append([l[1],l[0]])
					intersect_lines.remove(l)
					find_next = True
					break
			
			if not find_next:
				
				for line in poly:
					vtx0 = line[0]
					vtx1 = line[1]
					NL = 10
					plt.plot(np.linspace(vtx0[0],vtx1[0],NL),np.linspace(vtx0[1],vtx1[1],NL))
				print('min dist:',min_dist,', gap: ',np.linalg.norm(np.array(startp)-np.array(endp)),endp,startp)
				plt.savefig('error_'+str(round(plane_z,6))+'.png',dpi=1200)
				pl.close()
				assert(False)	


		_temp_polys.append(poly)
	
	sorted_temp_polys = sorted(_temp_polys, key=lambda x: GetPolyDiameter(x),reverse=True)
	for poly in sorted_temp_polys:
		# check if the new poly is inside or outside or new to the polys
		IsNewPoly = True
		for _ip, polyo in enumerate(polys):
			#print(polyo)
			#print('-----------')
			if IsPolyInPoly(poly, polyo):
				for l in poly:
					#print(l,lines[_ip])
					lines[_ip].append(l)
				IsNewPoly = False
				#print('poly inside',len(lines[_ip]),GetPolyDiameter(poly))
				break
			elif IsPolyInPoly(polyo, poly):
				for l in polyo:
					#print(l,lines[_ip])
					lines[_ip].append(l)
				polys[_ip] = poly
				IsNewPoly = False
				#print('poly outside',len(lines[_ip]),GetPolyDiameter(poly))
				break
			
		if IsNewPoly:
			polys.append(list(poly))
			lines.append(list(poly))
			#print('new poly added',len(lines),GetPolyDiameter(poly))
	return lines,polys

def shuffle_with_indices(arr):
	# Generate shuffled indices
	shuffled_indices = np.arange(len(arr))
	np.random.shuffle(shuffled_indices)

	# Use the shuffled indices to create a shuffled version of the original array
	shuffled_array = arr[shuffled_indices]

	return shuffled_array, shuffled_indices

# center = [x,y]
def get_neihbor_id(center):
	if center[0]==0 or center[0]==4 or center[1]==0 or center[1]==4:
		return None
	return [np.array(center)+i for i in [[0,1],[1,1],[-1,1],[0,-1],[1,-1],[-1,-1],[1,0],[-1,0]]]

def generate_raster_path(layer_array, n_layers, hatch_space, z,
		laser_power, laser_speed):
	"""generate raster path."""
	roads = []

	layer_num = FIRST_LAYER_IDX

	for cb_array in layer_array:
		for item in cb_array:
			#print("one piece of checkerboard: ",item)
			x0,y0,length,width = item[0],item[1],item[2],item[3]
			orientation = item[4]
			island = item[5]
			
			roads.extend(_rastered_helper(x0,y0,length,width,hatch_space,orientation,
			z, layer_num, laser_power, laser_speed, island))
		
	return roads

def _rastered_helper(x0,y0,length,width,hatch_space,orientation,
		z=0, layer_num=1, laser_power=195, laser_speed=0.8, island = -1):
	roads = []
	switch = True
	if abs(orientation) == float('inf'):
		roads.append(Road(x0, y0, x0+hatch_space, y0, z, laser_power, laser_speed, layer_num, island))
		x1,y1 = x0+hatch_space,y0
		size_l = length-hatch_space
		size_w = width-hatch_space
	
		while size_l>=hatch_space and size_w>=hatch_space:
			x2,y2 = x1+size_l,y1
			roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))
			x1,y1 = x2,y2+hatch_space/2
			
			x2,y2 = x1,y1+size_w
			roads.append(Road(x1-hatch_space/2, y1, x2-hatch_space/2, y2, z, laser_power, laser_speed, layer_num, island))
			x1,y1 = x2-hatch_space/2,y2
			
			x2,y2 = x1-size_l,y1
			roads.append(Road(x1-hatch_space/2, y1-hatch_space/2, x2-hatch_space/2, y2-hatch_space/2, z, laser_power, laser_speed, layer_num, island))
			x1,y1 = x2,y2-hatch_space/2
			
			x2,y2 = x1,y1-size_w+hatch_space
			roads.append(Road(x1, y1-hatch_space/2, x2, y2-hatch_space/2, z, laser_power, laser_speed, layer_num, island))
			x1,y1 = x2+hatch_space/2,y2
			
			x2,y2 = x1+hatch_space,y1
			roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))
			x1,y1 = x2,y2
			
			
			size_l = size_l-2*hatch_space
			size_w = size_w-2*hatch_space
		x2,y2 = x1+hatch_space,y1
		roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))
		x1,y1 = x2,y2+hatch_space/2
		
		x2,y2 = x1,y1+hatch_space
		roads.append(Road(x1-hatch_space/2, y1, x2-hatch_space/2, y2, z, laser_power, laser_speed, layer_num, island))
		x1,y1 = x2-hatch_space/2,y2
		
		x2,y2 = x1-hatch_space,y1
		roads.append(Road(x1-hatch_space/2, y1-hatch_space/2, x2-hatch_space/2, y2-hatch_space/2, z, laser_power, laser_speed, layer_num, island))
		
	elif orientation>0 and orientation <90:
		dy = hatch_space/math.cos(math.pi*orientation/180)
		dx = hatch_space/math.sin(math.pi*orientation/180)
		if dx<length:
			x1,y1 = x0+dx,y0+width
		else:
			x1,y1 = x0+length,y0+width
		if dy<width:
			x2,y2 = x0,y0+width-dy
		else:
			x2,y2 = x0,y0
		while y1>=y0 and x2<=x0+length:
			if switch:
				roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))
				switch = not switch
			else:
				roads.append(Road(x2, y2, x1, y1, z, laser_power, laser_speed, layer_num, island))
				switch = not switch
			if x1+dx<=x0+length:
				x1 = x1+dx
			else:
				if y1 == y0+width:
					d = hatch_space - (length-x1+x0)*math.sin(math.pi*orientation/180)
					y1 = y0+width-d/math.cos(math.pi*orientation/180)
				else:
					y1 = y1-dy
				x1 = x0+length
			if y2-dy>=y0:
				y2 = y2-dy
			else:
				if x2 == x0:
					d = hatch_space - (y2-y0)*math.cos(math.pi*orientation/180)
					x2 = x0+d/math.sin(math.pi*orientation/180)
				else:
					x2 = x2+dx
				y2 = y0
				
	elif orientation>90 and orientation <180:
		dy = -hatch_space/math.cos(math.pi*orientation/180)
		dx = hatch_space/math.sin(math.pi*orientation/180)
		if dx<length:
			x1,y1 = x0+length-dx,width+y0
		else:
			x1,y1 = x0+length,width+y0
		if dy<width:
			x2,y2 = x0+length,width-dy+y0
		else:   
			x2,y2 = x0+length,y0
		while y1>=y0 and x2>=x0:
			if switch:
				roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))
				switch = not switch
			else:
				roads.append(Road(x2, y2, x1, y1, z, laser_power, laser_speed, layer_num, island))
				switch = not switch                        
			if x1-dx>=x0:
				x1 = x1-dx
			else:
				if y1 == y0+width:
					d = hatch_space - (x1-x0)*math.sin(math.pi*orientation/180)
					y1 = y0+width+d/math.cos(math.pi*orientation/180)
				else:
					y1 = y1-dy
				x1 = x0
			if y2-dy>=y0:
				y2 = y2-dy
			else:
				if x2 == x0+length:
					d = hatch_space + (y2-y0)*math.cos(math.pi*orientation/180)
					x2 = x0+length-d/math.sin(math.pi*orientation/180)
				else:
					x2 = x2-dx
				y2 = y0
				
	elif orientation == 0:
		ys = np.arange(y0, y0+width, hatch_space)
		x_mn, x_mx = x0, x0+length
		x1, x2 = x_mx, x_mn
		for y in ys:
			y1 = y2 = y
			x1, x2 = x2, x1 
			roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island)) 
	
	elif orientation == 90:
		xs = np.arange(x0+hatch_space/2, x0+length+hatch_space/2, hatch_space)
		y_mn, y_mx = y0-hatch_space/2, y0+width-hatch_space/2    
		y1, y2 = y_mx, y_mn
		for x in xs:
			x1 = x2 = x
			y1, y2 = y2, y1            
			roads.append(Road(x1, y1, x2, y2, z, laser_power, laser_speed, layer_num, island))

	return roads



def plot_roads_2D(roads):
	"""plot roads in 2D."""
	fig, ax = plt.subplots(figsize=(8, 8))
	for x1, y1, x2, y2, z, *_ in roads:
		ax.plot([x1, x2], [y1, y2], 'b-')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_aspect('equal')
	return fig, ax


def plot_roads(roads):
	"""plot roads in 3D."""
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')
	for x1, y1, x2, y2, z, *_ in roads:
		# ax.plot([x1, x2], [y1, y2], [z, z], 'b-')
		ax.plot([x1, x2], [y1, y2], [z, z])
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	return fig, ax


def write_roads_to_file(roads, path, ncols=7):
	"""write roads to file."""
	assert(roads and len(roads[0]) >= ncols)
	with open(path, 'w') as out_file:
		out_file.write('# x1 y1 x2 y2 z power speed island\n')
		for road in roads:
			line = ' '.join(map(str, [round(i,6) for i in road[:ncols-1]]))+" "+' '.join(map(str, [round(road[ncols])]))
			out_file.write(line + '\n')
	print("write {:d} roads to {:s}".format(len(roads), path))


def plot_roads_file(path):
	"""plot roads file."""
	M = np.loadtxt(path)
	return plot_roads(M)

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
		grid_dict[round(z, 6)] = layer_points  # rounding to avoid float precision issues
		z += layer_thickness

	return grid_dict

def island_to_coord(idx,nl,nw):
	layer_num = int(idx/(nl*nw))
	i = int((idx - layer_num*nl*nw)/nw)
	if i%2 == 0:
		j = int((idx - layer_num*nl*nw - i*nw))
	else:
		j = nw -1- int((idx - layer_num*nl*nw - i*nw))
	return [i,j,layer_num]

def island_to_position(idx,nl,nw,o,i_size,layer_thickness):
	[i,j,layer_num] = island_to_coord(idx,nl,nw)
	return [o[0]+i*i_size,o[1]+j*i_size,o[2]+layer_num*layer_thickness]

def position_to_coord(p,i_size,layer_thickness,o):
	r = [p[0]-o[0],p[1]-o[1],p[2]-o[2]]
	return [int(r[0]/i_size),round(r[1]/i_size),round(r[2]/layer_thickness)]

def coord_to_island(i_idx,j_idx,layer_num,nl,nw):
	if i_idx % 2 == 0:
		return layer_num*nl*nw+(nw)*i_idx+j_idx
	else:
		return layer_num*nl*nw+(nw)*i_idx+nw-1-j_idx
def neighbor_list(i_idx,j_idx,nw,nl,island_num,size):
	sim_list = []
	for di in range(-size, size + 1):
		for dj in range(-size, size + 1):
			if di == 0 and dj == 0:
				continue  # Skip the center cell itself

			ni, nj = i_idx + di, j_idx + dj
			if 0 <= ni < nl and 0 <= nj < nw:
				neighbor_num = island_num+di*nw+dj
				sim_list.append(neighbor_num)

	return sim_list
def set_axes_equal(ax):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	  ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_middle = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_middle = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_middle = np.mean(z_limits)

	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = 0.5*max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_island_idx(owner: Any, layer: Any, base: int = 0):
	seq_map = {}
	iss = get_island_geometries(layer)
	for i, g in enumerate(iss):
		if (g.islandId == owner.islandId):
			return base + i
	return -1
# input coords
# ouput 
if __name__ == "__main__":
	
	OUTDIR = Path(__file__).resolve().parent
	
	plot_island_size = 3
	layer_thickness = 1e-4
	hatch_space = 1e-4
	fname = "ge_bracket_large_1_1_block"
	q2_path = OUTDIR / (fname+"_layer.scode")
	island_path = OUTDIR / (fname+".scode")
	geomSlices, layers, zs = gen_island_slices(fname,layer_thickness)
	
	models = build_minimal_models()
	island_dict = {}
	n_island = 0
	all_islands = []
	for geoslice, layer, z in zip(geomSlices,layers,zs):
		#print("write island scode:",z)
		assign_model(layer)
		islands = get_island_geometries(layer)
		for islandId,island in enumerate(islands):
			if round(z/layer_thickness) not in island_dict:
				index = IslandIndex(layer, neighbor_radius=NEIGHBOR_RADIUS_R)
				x_center = island.boundaryPoly.centroid.x
				y_center = island.boundaryPoly.centroid.y
				island_dict[round(z/layer_thickness)] = {"layer":index,"bid":n_island,"pts":[{"coord":[x_center,y_center,z],"island":island,"id":islandId}]}
				all_islands.append(islandId+n_island)
			else:
				x_center = island.boundaryPoly.centroid.x
				y_center = island.boundaryPoly.centroid.y
				if (islandId not in [i["id"] for i in island_dict[round(z/layer_thickness)]["pts"]]):
					island_dict[round(z/layer_thickness)]["pts"].append({"coord":[x_center,y_center,z],"island":island,"id":islandId}) 
					all_islands.append(islandId+n_island)
		n_island += write_layer_island_info_scode(layer, models, z, str(island_path), island_index_base=n_island, re = False)
	
	#print("islands done")
	head,points,normals,vs1,vs2,vs3 = BinarySTL(fname+'.STL')
	#print(points)
	points = np.vstack([vs1, vs2, vs3])
	faces = np.arange(len(points)).reshape(-1, 3)
	[xmin,xmax,ymin,ymax,zmin,zmax] = BoundingBox(points)
	#print(xmin,xmax,ymin,ymax,zmin,zmax)
	length = xmax - xmin
	width = ymax - ymin
	island_size = ISLAND_WIDTH
	n_elem_island = 1
	nw = round(width/island_size)
	nl = round(length/island_size)
	origin = np.array([xmin,ymin,zmin])#0.035: np.array([0.05,0.02,plane_z])

	all_sim_layers = []
	# sim
	all_sim_islands = []
	all_sim_paths = []
	N_temp = 2 # temporal neighborhood
	N_spatial = 0.9
	N_block = 1000000
	'''
	point_of_interest = {
		0.1: [[1, 2], [3, 4]],
		0.5: [[5, 6], [7, 8]],
		1.0: [[9, 10], [11, 12]]
	}
	'''
	#point_of_interest = generate_grid(xmin,xmax,ymin,ymax,zmin,zmax, island_size*4, layer_thickness)
	
	
	#print('geometry grids:',grids)
	#print('inside_islands:',inside_islands)
	
	# randomly choose some points of interest
	init_grids1 = generate_grid(xmin,xmax,ymin,ymax,zmin,zmax, island_size, layer_thickness)
	init_grids2 = generate_grid(xmin,xmax,ymin,ymax,zmin,zmax, 2*island_size, 2*layer_thickness)
	
	point_of_interest = {}
	sampled_islands = []
	for z, coords in init_grids2.items():
		polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)
		for coord in coords:
			p = [coord[0],coord[1],z]
			if z<0.002:
				if (z not in point_of_interest):
					point_of_interest[z] = [coord]
				else:
					point_of_interest[z].append(coord)
			if z>0.003 and z<0.005:
				if any(poly.contains(Point(p)) for poly in polygons):
					if (z not in point_of_interest):
						point_of_interest[z] = [coord]
					else:
						point_of_interest[z].append(coord)
			if z<0.002:
				if any(poly.contains(Point(p)) for poly in polygons):
					if (z not in point_of_interest):
						point_of_interest[z] = [coord]
					else:
						point_of_interest[z].append(coord)
			if z>0.003 and z<0.005:
				if any(poly.contains(Point(p)) for poly in polygons):
					if (z not in point_of_interest):
						point_of_interest[z] = [coord]
					else:
						point_of_interest[z].append(coord)
			if z>0.032 and z<0.034:
				if any(poly.contains(Point(p)) for poly in polygons):
					if (z not in point_of_interest):
						point_of_interest[z] = [coord]
					else:
						point_of_interest[z].append(coord)
			if z>0.032 and z<0.034:
				if any(poly.contains(Point(p)) for poly in polygons):
					if (z not in point_of_interest):
						point_of_interest[z] = [coord]
					else:
						point_of_interest[z].append(coord)
		
		#else:
		#	print('loi:',z)
	
	
	for z, coords in init_grids1.items():
		polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)
		for coord in coords:
			p = [coord[0],coord[1],z]
			if z<0.002:
				if abs(coord[0]+0.05)<0.008 and abs(coord[1])<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cl1:",islandId)
							sampled_islands.append(islandId)
						#owner = island_dict[round(z/layer_thickness)]["layer"].find_island_at_point(coord[0],coord[1])
						#if owner != None:
						#	print("cl1:",get_island_idx(owner,island_dict[round(z/layer_thickness)]["layer"]))
			if z>0.003 and z<0.005:
				if abs(coord[0]+0.05)<0.008 and abs(coord[1])<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cl2:",islandId)
							sampled_islands.append(islandId)

						#owner = island_dict[round(z/layer_thickness)]["layer"].find_island_at_point(coord[0],coord[1])
							#if owner != None:
						#	print("cl2:",get_island_idx(owner,island_dict[round(z/layer_thickness)]["layer"]))
			if z<0.002:
				if abs(coord[0]-0.025)<0.008 and abs(coord[1])<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cr1:",islandId)
							sampled_islands.append(islandId)
						#owner = island_dict[round(z/layer_thickness)]["layer"].find_island_at_point(coord[0],coord[1])
						#if owner != None:
						#	print("cr1:",get_island_idx(owner,island_dict[round(z/layer_thickness)]["layer"]))
			if z>0.003 and z<0.005:
				if abs(coord[0]-0.025)<0.008 and abs(coord[1])<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cr2:",islandId)
							sampled_islands.append(islandId)
			if z>0.032 and z<0.034:
				if abs(coord[0]+0.02)<0.008 and abs(coord[1]+0.04)<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cu1:",islandId)
							sampled_islands.append(islandId)
			if z>0.032 and z<0.034:
				if abs(coord[0]-0.01)<0.008 and abs(coord[1]+0.04)<0.008:
					if any(poly.contains(Point(p)) for poly in polygons):
						dist = np.inf
						for data in island_dict[round(z/layer_thickness)]["pts"]:
							dist_data = np.linalg.norm(np.array(data["coord"]) - np.array(p))
							if dist_data<dist:
								dist = dist_data
								islandId = data["id"]
						islandId += island_dict[round(z/layer_thickness)]["bid"]
						if dist <= ISLAND_WIDTH and islandId not in sampled_islands:
							print("cu2:",islandId)
							sampled_islands.append(islandId)		
		#else:
		#	print('loi:',z)
	
	'''
	# uniform sampling + adaptive sampling
	for z, coords in init_grids1.items():
		if (z-zmin>5*layer_thickness and round((z-zmin)/layer_thickness)%40 != 0):
			continue
		#else:
		#	print('loi:',z)
			
		polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)

		for coord in coords:
			p = [coord[0],coord[1],z]
			if any(poly.contains(Point(p)) for poly in polygons):
				if (z not in point_of_interest):
					point_of_interest[z] = [coord]
				else:
					point_of_interest[z].append(coord)
	
	for z, coords in init_grids2.items():
		polygons = slice_mesh_to_polygons(fname+'.STL', z, xmin, ymin)
		# 1️⃣ Build the distance field ONCE for this slice
		dist_img, mask_img, grid_params = build_distance_field(polygons, step=0.001)

		# 3️⃣ Query distances all at once
		dists = query_distances(coords, dist_img, mask_img, grid_params)
		for coord, dist in zip(coords, dists):
			p = [coord[0],coord[1],z]
			if dist<0 or dist>0.008:
				continue
			[i,j,l] = position_to_coord(p,island_size,layer_thickness,origin)
			n = round(dist/0.002)
			if not (n == 0 or (i%n == 0 and j%n ==0)):
				continue
			if (z not in point_of_interest):
				point_of_interest[z] = [coord]
			else:
				if coord not in point_of_interest[z]:
					point_of_interest[z].append(coord)	
			
		if z in point_of_interest:
			plt.scatter([xyz[0] for xyz in point_of_interest[z]], [xyz[1] for xyz in point_of_interest[z]], s=4, c='blue', alpha=0.4)		
		for i, poly in enumerate(polygons):
			if poly.is_empty:
				continue
			# --- Project to 2D if needed ---
			coords = np.array(poly.exterior.coords)
			if coords.shape[1] == 3:
				coords2d = coords[:, :2]  # drop Z
			else:
				coords2d = coords

			# --- Plot polygon boundary ---
			plt.plot(coords2d[:, 0], coords2d[:, 1], 'k-', linewidth=1)

			# --- Plot holes (if any) ---
			for interior in poly.interiors:
				hole = np.array(interior.coords)
				plt.plot(hole[:, 0], hole[:, 1], 'r--', linewidth=0.8)
		plt.axis('equal')
		plt.title("Uniform Sampling")
		plt.savefig('./pts/uniform sampling_'+str(round(z,6))+'_.png')
		plt.clf()
	'''
	coords = []		
	for z, positions in point_of_interest.items():
		for pos in positions:
			print(pos[0],pos[1],z)
			coords.append([pos[0],pos[1],z])
	
	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plot_x = []
	plot_y = []
	plot_z = []
	for z, positions in point_of_interest.items():
		for pos in positions:
			plot_x.append(pos[0])
			plot_y.append(pos[1])
			plot_z.append(z)
	lc = ax.scatter(plot_x,plot_y,plot_z,s=plot_island_size,c='lime',marker='s')
	set_axes_equal(ax)
	plt.savefig('3d.png')#,bbox_inches='tight',pad_inches=0.0,dpi=1200
	plt.close()
	trees = {}
	for z, positions in point_of_interest.items():
		for pos in positions:
			p = [pos[0],pos[1],z]
			if z not in trees:
				trees[z] = cKDTree([item["coord"] for item in island_dict[round(z/layer_thickness)]["pts"]])
			dist, idx = trees[z].query(p)
			island_num_in_layer = island_dict[round(z/layer_thickness)]["pts"][idx]["id"]
			island_num = island_num_in_layer + island_dict[round(z/layer_thickness)]["bid"]
			if island_num not in all_sim_paths:
				all_sim_layers.append(z-layer_thickness)
				sim_path_list = [island_num]
				neighbor_islands = island_dict[round(z/layer_thickness)]["layer"].neighbors_for_island(island_dict[round(z/layer_thickness)]["pts"][idx]["island"],ISLAND_WIDTH*N_spatial)
				n_list = [i.islandId+island_dict[round(z/layer_thickness)]["bid"] for i in neighbor_islands]
				sim_path_list += [i for i in n_list if ((island_num-i)<N_temp and island_num>=i)]# get neighbor islands for path sim
				sim_island_list = [i for i in n_list if (i not in sim_path_list)]# get neighbor islands for island sim
				all_sim_paths += sim_path_list
				all_sim_islands += sim_island_list
				start_i = min(sim_path_list)
				end_i = max(sim_path_list)
				print(start_i,end_i,"true",n_list)
	# remove duplicated and out of shape ones
	#print("remove duplicated",len(all_sim_islands),len(all_sim_paths))
	all_sim_islands = list(set(all_sim_islands))
	all_sim_paths = list(set(all_sim_paths))
	all_sim_islands = [i for i in all_sim_islands if i not in all_sim_paths]
	#print("path:",len(all_sim_paths),"island:",len(all_sim_islands))
	#print("block",sum(len(v) for v in grids.values()))
	#print("layer",len(point_of_interest))
	#print(all_sim_paths)

	for z, positions in point_of_interest.items():
		#print(z) # every z (layer) has a plot
		fig = plt.figure() # LOI
		ax = fig.add_subplot()
		plot_x = []
		plot_y = []
		for pos in positions:
			plot_x.append(pos[0])
			plot_y.append(pos[1])
		lc = ax.scatter(plot_x,plot_y,s=plot_island_size,c='lime',marker='s')
		ax.set_xlim(xmin,xmax)
		ax.set_ylim(ymin,ymax)
		ax.set_aspect('equal')
		plt.savefig('LOI_'+str(round(z,6))+'.png')#,bbox_inches='tight',pad_inches=0.0,dpi=1200
		plt.close()

	non_block_islands = sorted(all_sim_paths)
	block_idx = 0
	last_block_island = -1
	plot_x = []
	sorted(all_islands)
	#print(all_islands)
	block_island_begin = -1
	block_island_current = -1
	for idx in range(len(all_islands)):
		island_idx = all_islands[idx]
		if island_idx not in non_block_islands:
			block_island_current = island_idx
			if block_island_begin<0:
				block_island_begin = island_idx
			if len(plot_x)<N_block and idx<len(all_islands)-1:
				plot_x.append(island_idx)
			else:
				print(block_island_begin,island_idx,"false",[])
				block_island_begin = -1
				plot_x = [island_idx]
				block_idx += 1
		elif len(plot_x) != 0:
			if block_island_begin>=0:
				print(block_island_begin,block_island_current,"false",[])
				block_island_begin = -1
			if island_idx in all_sim_islands:
				print(island_idx,island_idx,"false",[])
			plot_x = []
			block_idx += 1