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

myHatcher = hatching.IslandHatcher()
myHatcher.islandWidth = ISLAND_WIDTH
myHatcher.islandOverlap = 0
myHatcher.hatchAngle = 0
myHatcher.volumeOffsetHatch = 0
myHatcher.spotCompensation = 0
myHatcher.numInnerContours = 0
myHatcher.numOuterContours = 0
myHatcher.hatchDistance = 0.1*SCALE
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
		"high_sensi": {"power": 150.0, "speed": 0.1},
		"med_sensi": {"power": 160.0, "speed": 0.095},
		"low_sensi": {"power": 170.0, "speed": 0.09},
		"base": {"power": 180.0, "speed": 0.085},
		"boundary": {"power": 190.0, "speed": 0.08},
		"interface": {"power": 200.0, "speed": 0.075},
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
	for z in np.arange(zmin, zmax, layer_thickness):
		geomSlice = solidPart.getVectorSlice(z+1e-5, simplificationFactor=0.1)
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
	return geomSlices, layers, zs



def main():
	
	OUTDIR = Path(__file__).resolve().parent
	
	layer_thickness = 0.1*SCALE
	fname = "ge_bracket_large_1_1"
	
	base_path = _base_path()
	zone_parts = build_zone_parts(base_path)

	models, zone_bids, contour_bid = build_models()
	zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]


	original_stl = base_path / "ge_bracket_original.stl"
	solidPart = pyslm.Part("ge_bracket")
	solidPart.setGeometry(str(original_stl))
	solidPart.scaleFactor = SCALE
	solidPart.dropToPlatform()
	
	[xmin,ymin,zmin,xmax,ymax,zmax] = solidPart.boundingBox
	print(xmin,ymin,zmin,xmax,ymax,zmax)
	zone_priority = ["interface", "high_sensi", "med_sensi", "boundary", "low_sensi", "base"]
	
	island_dict = {}
	n_island = 0
	zs = np.arange(zmin, zmax, layer_thickness)
	for z in zs:
		geomSlice = solidPart.getVectorSlice(z+1e-5, simplificationFactor=0.1)
		layer = myHatcher.hatch(geomSlice)
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
		islands = get_island_geometries(layer)
		
		if round(z/layer_thickness) not in island_dict:
			island_dict[round(z/layer_thickness)] = {"layer":layer,"bid":n_island}	
		n_island += len(islands)
	query_points = np.loadtxt("pts.txt")
	for p in query_points:
		idx = np.argmin(np.abs(zs - p[2]*1000*SCALE))
		if np.abs(zs[idx] - p[2]*1000*SCALE)>layer_thickness:
			continue
		Z_TARGET = zs[idx]
		n_z = round(Z_TARGET/layer_thickness)
		if n_z not in island_dict:
			continue
		q1_path = OUTDIR / "gcodes" / str(fname+"_local_query_"+str(round(p[0],6))+"_"+str(round(p[1],6))+"_"+str(round(Z_TARGET/1000,6))+"_fine_laser_path.scode")
		layers = []
		param_zs = []
		bids = []
		for i in range(n_z-5,n_z+1):
			if i not in island_dict:
				continue
			layers.append(island_dict[i]["layer"])
			assign_model(layer, models)
			param_zs.append(Z_TARGET-(n_z-i)*layer_thickness)
			bids.append(island_dict[i]["bid"])
		n1,iid,(px,py) = write_neighborhood_paths_scode(layers, models, p[0]*1000, p[1]*1000, NEIGHBOR_RADIUS_R, param_zs, str(q1_path), bids)
		if n1:
			print(f"{iid} -1 . {q1_path} {px/1000} {py/1000} {Z_TARGET/1000}")

	

if __name__ == '__main__':
	main()
