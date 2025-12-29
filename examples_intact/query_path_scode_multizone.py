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


# ----------------------------
# Config
# ----------------------------

SCALE = 0.001
SCAN_CONTOUR_FIRST = False  # available if needed by your IslandHatcher setup
ISLAND_WIDTH = 0.002
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
myHatcher.hatchAngle = 0
myHatcher.hatchSortMethod = hatching.AlternateSort()
myHatcher.groupIslands = True
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

SCALE = 0.001
SCAN_CONTOUR_FIRST = False  # available if needed by your IslandHatcher setup
ISLAND_WIDTH = 0.002
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
myHatcher.hatchAngle = 0
myHatcher.hatchSortMethod = hatching.AlternateSort()
myHatcher.groupIslands = True




def _base_path() -> Path:
	return _repo_root / "geometry_intact" / "zone_aware_island_gebracket"


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
		"high_sensi": {"power": 160.0, "speed": 2.5},
		"med_sensi": {"power": 200.0, "speed": 1.75},
		"low_sensi": {"power": 203.0, "speed": 2.5},
		"base": {"power": 320.0, "speed": 2.5},
		"boundary": {"power": 201.0, "speed": 2.5},
		"interface": {"power": 202.0, "speed": 2.5},
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

	[xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
	#print(xmin,ymin,zmin,xmax,ymax,zmax)
	geomSlices = []
	layers = []
	zs = []
	zone_bidss = []
	contour_bids = []
	for z in np.arange(zmin, zmax, layer_thickness):
		geomSlice = solidPart.getVectorSlice(z, simplificationFactor=0.1)
		geomSlices.append(geomSlice)
		#print(z)
		layer = myHatcher.hatch(geomSlice)
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
		layers.append(layer)
		zs.append(z)
		zone_bidss.append(zone_bids)
		contour_bids.append(contour_bid)
	return geomSlices, layers, zs, zone_bidss, contour_bids



def main():
	OUTDIR = Path(__file__).resolve().parent
	
	base_path = _base_path()
	zone_parts = build_zone_parts(base_path)
	layer_thickness = 0.1
	fname = "ge_bracket_large_1_1"
	geomSlices, layers, zs, zone_bidss, contour_bids = gen_island_slices(fname,layer_thickness)
	island_dict = {}
	n_island = 0
	all_islands = []
	for geoslice, layer, z, zbs, cbid in zip(geomSlices,layers,zs,zone_bidss, contour_bids):
		models = assign_model(zbs, cbid)
		islands = get_island_geometries(layer)
		if round(z/layer_thickness) not in island_dict:
			island_dict[round(z/layer_thickness)] = {"layer":layer,"bid":n_island}	
		n_island += len(islands)
		#print("generating slices:",z,round(z/layer_thickness),island_dict.keys(),n_island)
	print(island_dict.keys())
	query_points = np.loadtxt("pts.txt")
	for p in query_points:
		idx = np.argmin(np.abs(zs - p[2]))
		if np.abs(zs[idx] - p[2])>layer_thickness:
			continue
		Z_TARGET = zs[idx]
		n_z = round(Z_TARGET/layer_thickness)
		if n_z not in island_dict:
			continue
		q1_path = OUTDIR / "gcodes" / str(fname+"_local_query_"+str(round(p[0],6))+"_"+str(round(p[1],6))+"_"+str(round(Z_TARGET,6))+"_fine_laser_path.scode")
		layers = []
		models = []
		param_zs = []
		bids = []
		for i in range(n_z-5,n_z+1):
			if i not in island_dict:
				continue
			layers.append(island_dict[i]["layer"])
			models += assign_model(layers[-1])
			param_zs.append(Z_TARGET-(n_z-i)*layer_thickness)
			bids.append(island_dict[i]["bid"])
		n1,iid,(px,py) = write_neighborhood_paths_scode(layers, models, p[0], p[1], NEIGHBOR_RADIUS_R, param_zs, str(q1_path), bids)
		if n1:
			print(f"{iid} -1 . {q1_path} {px} {py} {Z_TARGET}")

	

if __name__ == '__main__':
	main()


def main():
	OUTDIR = Path(__file__).resolve().parent
	
	layer_thickness = 1e-4
	fname = "ge_bracket_large_1_1_block"
	geomSlices, layers, zs = gen_island_slices(fname,layer_thickness)
	island_dict = {}
	n_island = 0
	for geoslice, layer, z in zip(geomSlices,layers,zs):
		models = assign_model(layer)
		islands = get_island_geometries(layer)
		if round(z/layer_thickness) not in island_dict:
			island_dict[round(z/layer_thickness)] = {"layer":layer,"bid":n_island}	
		n_island += len(islands)
		#print("generating slices:",z,round(z/layer_thickness),island_dict.keys(),n_island)
	print(island_dict.keys())
	query_points = np.loadtxt("pts.txt")
	for p in query_points:
		idx = np.argmin(np.abs(zs - p[2]))
		if np.abs(zs[idx] - p[2])>layer_thickness:
			continue
		Z_TARGET = zs[idx]
		n_z = round(Z_TARGET/layer_thickness)
		if n_z not in island_dict:
			continue
		q1_path = OUTDIR / "gcodes" / str(fname+"_local_query_"+str(round(p[0],6))+"_"+str(round(p[1],6))+"_"+str(round(Z_TARGET,6))+"_fine_laser_path.scode")
		layers = []
		models = []
		param_zs = []
		bids = []
		for i in range(n_z-5,n_z+1):
			if i not in island_dict:
				continue
			layers.append(island_dict[i]["layer"])
			models += assign_model(layers[-1])
			param_zs.append(Z_TARGET-(n_z-i)*layer_thickness)
			bids.append(island_dict[i]["bid"])
		n1,iid,(px,py) = write_neighborhood_paths_scode(layers, models, p[0], p[1], NEIGHBOR_RADIUS_R, param_zs, str(q1_path), bids)
		if n1:
			print(f"{iid} -1 . {q1_path} {px} {py} {Z_TARGET}")

	

if __name__ == '__main__':
	main()
