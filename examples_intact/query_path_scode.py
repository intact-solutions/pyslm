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
myHatcher.hatchDistance = 1e-4
myHatcher.hatchAngle = 0
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



def assign_model(layer):
	# Assign model/buildstyle ids to each geometry
	for g in getattr(layer, 'geometry', []):
		g.mid = 1
		g.bid = 1

	# Minimal BuildStyle/Model for timing/exports
	bstyle = pyslm.geometry.BuildStyle()
	bstyle.bid = 1
	bstyle.laserSpeed = 1.2  # [mm/s] continuous mode
	bstyle.laserPower = 120.0  # [W]
	bstyle.jumpSpeed = 5000.0  # [mm/s]

	model = pyslm.geometry.Model()
	model.mid = 1
	model.buildStyles.append(bstyle)

	return [model]


def main():
	OUTDIR = Path(__file__).resolve().parent
	
	layer_thickness = 1e-4
	fname = "ge_bracket_large_1_1"
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
		q1_path = OUTDIR / "gcodes" / "120_12" / str(fname+"_local_query_"+str(round(p[0],6))+"_"+str(round(p[1],6))+"_"+str(round(Z_TARGET,6))+"_fine_laser_path.scode")
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
