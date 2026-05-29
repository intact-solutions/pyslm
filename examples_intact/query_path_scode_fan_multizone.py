import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
	sys.path.insert(0, str(_repo_root))

import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import get_island_geometries, IslandIndex
from pyslm.analysis.export_scode import (
	write_neighborhood_paths_scode,
	write_layer_island_info_scode,
	calculate_torsion_position
)


# ----------------------------
# Config
# ----------------------------

SCALE = 1
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

import sys
from pathlib import Path

# Ensure local repo import without needing PYTHONPATH set externally
_repo_root = Path(__file__).resolve().parents[1]  # points to repo root containing 'pyslm/'
if str(_repo_root) not in sys.path:
	sys.path.insert(0, str(_repo_root))

import numpy as np

import pyslm
from pyslm import hatching as hatching
from pyslm.analysis.island_utils import get_island_geometries, IslandIndex
from pyslm.analysis.export_scode import (
	write_neighborhood_paths_scode,
	write_layer_island_info_scode,
)
from pyslm.analysis.zone_utils import build_zone_polygons, classify_layer_geometry


# ----------------------------
# Config
# ----------------------------





def _base_path() -> Path:
	return _repo_root / "geometry_intact" / "zone_fan"

base_path = _base_path()
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
zone_priority = ["core","blade6","blade7","blade8","blade9","blade10","blade11","blade12","blade1","blade2","blade3","blade4","blade5"]

zone_bids = {
	"core": 1, "blade1": 2, "blade2": 3, "blade3": 4, "blade4": 5, "blade5": 6,
	"blade6": 7, "blade7": 8, "blade8": 9, "blade9": 10, "blade10": 11,
	"blade11": 12, "blade12": 13
}
zone_params = {z: {"power": 230, "speed": 2} for z in zone_bids.keys()}

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
	h.hatchDistance = 0.1*SCALE
	h.hatchSortMethod = hatching.AlternateSort()
	h.groupIslands = True
	
	zone_hatchers[zone] = h
	angle_offset -= 30.0 # Example: Increment angle by 15 degrees per zone

def build_zone_parts(base_path: Path):
	zone_parts = {}
	for zone_name, ply_path in zone_ply_paths.items():
		if ply_path.exists():
			part = pyslm.Part(zone_name)
			part.setGeometry(str(ply_path))
			zone_parts[zone_name] = part

	return zone_parts

def build_models():
	contour_bid = 10


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

def gen_island_slices(filename,layer_thickness):
	solidPart = pyslm.Part(filename)
	solidPart.setGeometry(filename+'.STL')
	
	solidPart.dropToPlatform()
	solidPart.origin[0] = 0.0
	solidPart.origin[1] = 0.0
	solidPart.scaleFactor = SCALE
	solidPart.rotation = [0, 0.0, np.pi]
	

	base_path = _base_path()

	zone_parts = {}
	for name, p in zone_ply_paths.items():
		if p.exists():
			part = pyslm.Part(name)
			part.setGeometry(str(p))
			zone_parts[name] = part

	[xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
	#print(xmin,ymin,zmin,xmax,ymax,zmax)
	geomSlices = []
	layers = []
	zs = []
	for z in np.arange(zmin, zmax, layer_thickness):
		geomSlice = solidPart.getVectorSlice(z+1e-5, simplificationFactor=0.1)
		geomSlices.append(geomSlice)
		#print(z)
		layer = zone_hatchers[zone_name].hatch(geomSlice)
		zone_polys = build_zone_polygons(zone_parts, float(z))
		contour_bid = 10
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
		layers.append(layer)
		zs.append(z)
	return geomSlices, layers, zs



def main():
	
	OUTDIR = Path(__file__).resolve().parent
	
	query_points = np.loadtxt("pts_blade_test.txt")
	'''
	for p in query_points:
		x,y,z = calculate_torsion_position(p[0],p[1],p[2],z1 = 0.012, z2 = 0.0382,)
		print(x,y,z)
	asddsa
	'''
	layer_thickness = 0.1*SCALE
	fname = "Fan_twisted"
	
	base_path = _base_path()
	zone_parts = build_zone_parts(base_path)

	models, zone_bids, contour_bid = build_models()


	original_stl = base_path / "Fan_untwisted.STL"
	solidPart = pyslm.Part("Fan_untwisted")
	solidPart.setGeometry(str(original_stl))
	solidPart.scaleFactor = SCALE
	solidPart.dropToPlatform()
	
	[xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
	print(xmin,ymin,zmin,xmax,ymax,zmax)
	
	island_dict = {}
	n_island = 0
	zs = np.arange(zmin, zmax, layer_thickness)
	for z in zs:
		
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
		
		if round(z/layer_thickness) not in island_dict:
			island_dict[round(z/layer_thickness)] = {"layer":master_layer,"bid":n_island}	
		n_island += len(islands)
		
	for p in query_points:
		idx = np.argmin(np.abs(zs - p[2]*1000*SCALE))
		if np.abs(zs[idx] - p[2]*1000*SCALE)>layer_thickness:
			continue
		Z_TARGET = zs[idx]
		n_z = round(Z_TARGET/layer_thickness)
		if n_z not in island_dict:
			continue
		q1_path = OUTDIR / "gcodes" / "Fan_dense_blade_OPT_5p" / str(fname+"_local_query_"+str(round(p[0],6))+"_"+str(round(p[1],6))+"_"+str(round(Z_TARGET/1000,6))+"_fine_laser_path.scode")
		layers = []
		param_zs = []
		bids = []
		for i in range(n_z-5,n_z+1):
			if i not in island_dict:
				continue
			layers.append(island_dict[i]["layer"])
			param_zs.append(Z_TARGET-(n_z-i)*layer_thickness)
			bids.append(island_dict[i]["bid"])
		for layer in layers:
			assign_model(layer, models)
		n1,iid,(px,py) = write_neighborhood_paths_scode(layers, models, p[0]*1000, p[1]*1000, NEIGHBOR_RADIUS_R, param_zs, str(q1_path), bids)
		if n1:
			print(f"{iid} -1 . {q1_path} {px/1000} {py/1000} {Z_TARGET/1000}")

	

if __name__ == '__main__':
	main()
