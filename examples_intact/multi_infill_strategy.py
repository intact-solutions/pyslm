"""
Example showing how to implement different scanning strategies for overhang regions in SLM.
This example demonstrates:
1. Loading overhang and normal regions as separate geometries
2. Applying different scanning strategies to each
3. Visualizing the combined result
"""

import numpy as np
import logging
import os
from matplotlib import pyplot as plt

import pyslm
import pyslm.visualise
import pyslm.analysis
from pyslm.hatching import BasicIslandHatcher, AlternateSort
from pyslm import hatching as hatching#

def main():
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    
    # Process control flags
    process_bulk = True      # Control normal/bulk geometry processing
    process_overhang = True  # Control overhang geometry processing
    process_boundary = True  # Control boundary geometry processing
    
    # Define paths and check existence
    base_path = r'C:\Users\kumar\source_local\process_zones\tests\block\rotated\results'
    
    # Initialize variables
    normal_layer = None
    overhang_layer = None
    boundary_layer = None

    #layer height
    layer_height = 2.0
    
    # Process normal/bulk geometry
    if process_bulk:
        normal_stl = os.path.join(base_path, 'bulk_zone.stl')
        if not os.path.exists(normal_stl):
            raise FileNotFoundError(f"Normal region STL file not found: {normal_stl}")
        normal_part = pyslm.Part('normal_regions')
        normal_part.setGeometry(normal_stl)
        logging.info(f"Geometry information <normal_regions> - [{normal_stl}]")
        
        normal_hatcher = hatching.Hatcher()
        normal_hatcher.hatchAngle = 10
        normal_hatcher.volumeOffsetHatch = -0.1
        normal_hatcher.spotCompensation = 0.00
        normal_hatcher.numInnerContours = 0
        normal_hatcher.numOuterContours = 0
        normal_hatcher.hatchSortMethod = hatching.AlternateSort()
        
        normal_slice = normal_part.getVectorSlice(layer_height, simplificationFactor=0.1)
        normal_layer = normal_hatcher.hatch(normal_slice)
    
    # Process overhang geometry
    if process_overhang:
        overhang_stl = os.path.join(base_path, 'overhang_zone.stl')
        if not os.path.exists(overhang_stl):
            raise FileNotFoundError(f"Overhang region STL file not found: {overhang_stl}")
        overhang_part = pyslm.Part('overhang_regions')
        overhang_part.setGeometry(overhang_stl)
        logging.info(f"Geometry information <overhang_regions> - [{overhang_stl}]")
        
        overhang_hatcher = hatching.Hatcher()
        overhang_hatcher.hatchAngle = 45
        overhang_hatcher.volumeOffsetHatch = -0.1
        overhang_hatcher.spotCompensation = 0.00
        overhang_hatcher.numInnerContours = 0
        overhang_hatcher.numOuterContours = 0
        overhang_hatcher.hatchSortMethod = hatching.AlternateSort()
        
        overhang_slice = overhang_part.getVectorSlice(layer_height, simplificationFactor=0.1)
        overhang_layer = overhang_hatcher.hatch(overhang_slice)
    
    # Process boundary geometry
    if process_boundary:
        boundary_stl = os.path.join(base_path, 'boundary_zone.stl')
        if not os.path.exists(boundary_stl):
            raise FileNotFoundError(f"Boundary region STL file not found: {boundary_stl}")
        boundary_part = pyslm.Part('boundary_regions')
        boundary_part.setGeometry(boundary_stl)
        logging.info(f"Geometry information <boundary_regions> - [{boundary_stl}]")
        
        boundary_hatcher = hatching.Hatcher()
        boundary_hatcher.hatchAngle = 90
        boundary_hatcher.volumeOffsetHatch = 0.0
        boundary_hatcher.spotCompensation = 0.04
        boundary_hatcher.numInnerContours = 0
        boundary_hatcher.numOuterContours = 2
        boundary_hatcher.hatchSortMethod = hatching.AlternateSort()
        
        boundary_slice = boundary_part.getVectorSlice(layer_height, simplificationFactor=0.1)
        boundary_layer = boundary_hatcher.hatch(boundary_slice)
    
    # Create plot
    fig, ax = plt.subplots()
    
    # Plot normal (bulk) layer in gray
    if normal_layer:
        for layerGeom in normal_layer.geometry:
            coords = layerGeom.coords
            if len(coords) > 0:
                if isinstance(layerGeom, pyslm.geometry.HatchGeometry):
                    coords = coords.reshape(-1, 2, 2)
                    for line in coords:
                        ax.plot(line[:, 0], line[:, 1], '-', color='gray', linewidth=0.5)
                else:
                    ax.plot(coords[:, 0], coords[:, 1], '-', color='gray', linewidth=0.5)
                    
    # Plot overhang layer in red
    if overhang_layer:
        for layerGeom in overhang_layer.geometry:
            coords = layerGeom.coords
            if len(coords) > 0:
                if isinstance(layerGeom, pyslm.geometry.HatchGeometry):
                    coords = coords.reshape(-1, 2, 2)
                    for line in coords:
                        ax.plot(line[:, 0], line[:, 1], '-', color='red', linewidth=0.5)
                else:
                    ax.plot(coords[:, 0], coords[:, 1], '-', color='red', linewidth=0.5)
    
    # Plot boundary layer in blue
    if boundary_layer:
        for layerGeom in boundary_layer.geometry:
            coords = layerGeom.coords
            if len(coords) > 0:
                if isinstance(layerGeom, pyslm.geometry.HatchGeometry):
                    coords = coords.reshape(-1, 2, 2)
                    for line in coords:
                        ax.plot(line[:, 0], line[:, 1], '-', color='blue', linewidth=0.5)
                else:
                    ax.plot(coords[:, 0], coords[:, 1], '-', color='blue', linewidth=0.5)
    
    # Add legend only for processed geometries
    legend_elements = []
    if normal_layer:
        legend_elements.append(('Bulk', 'gray'))
    if boundary_layer:
        legend_elements.append(('Boundary', 'blue'))
    if overhang_layer:
        legend_elements.append(('Overhang', 'red'))
        
    for label, color in legend_elements:
        ax.plot([], [], '-', color=color, label=label, linewidth=0.5)
    
    if legend_elements:  # Only show legend if there are elements to show
        ax.legend()
    
    ax.set_aspect('equal')
    plt.show()
    
    # Print analysis information
    if normal_layer:
        print('\nNormal Region Analysis:')
        print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(normal_layer)))
        print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(normal_layer)))
    
    if overhang_layer:
        print('\nOverhang Region Analysis:')
        print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(overhang_layer)))
        print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(overhang_layer)))

    if boundary_layer:
        print('\nBoundary Region Analysis:')
        print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(boundary_layer)))
        print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(boundary_layer)))

if __name__ == '__main__':
    main()
