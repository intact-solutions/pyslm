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
from pyslm import hatching as hatching
from pyslm.geometry import ContourGeometry, HatchGeometry

def main():
    # Set up logging
    logging.getLogger().setLevel(logging.INFO)
    
    # Define paths and check existence
    base_path = r'C:\Users\kumar\source\Darpa Generative AM\process_zones\tests\block\rotated\results'
    normal_stl = os.path.join(base_path, 'bulk_zone.stl')
    overhang_stl = os.path.join(base_path, 'overhang_zone.stl')
    
    # Check if files exist
    if not os.path.exists(normal_stl):
        raise FileNotFoundError(f"Normal region STL file not found: {normal_stl}")
    if not os.path.exists(overhang_stl):
        raise FileNotFoundError(f"Overhang region STL file not found: {overhang_stl}")
        
    # Create parts - assume we have separate STLs for normal and overhang regions
    normal_part = pyslm.Part('normal_regions')
    normal_part.setGeometry(normal_stl)
    
    overhang_part = pyslm.Part('overhang_regions')
    overhang_part.setGeometry(overhang_stl)

    # Create hatchers
    normal_hatcher = hatching.Hatcher()
    overhang_hatcher = hatching.Hatcher()
    
    # Set the base hatching parameters which are generated within Hatcher
    normal_hatcher.hatchAngle = 10
    normal_hatcher.volumeOffsetHatch = 0.08
    normal_hatcher.spotCompensation = 0.06
    normal_hatcher.numInnerContours = 2
    normal_hatcher.numOuterContours = 1
    normal_hatcher.hatchSortMethod = hatching.AlternateSort()
    
    # Set the base hatching parameters which are generated within Hatcher
    overhang_hatcher.hatchAngle = 45
    overhang_hatcher.volumeOffsetHatch = 0.08
    overhang_hatcher.spotCompensation = 0.06
    overhang_hatcher.numInnerContours = 2
    overhang_hatcher.numOuterContours = 1
    overhang_hatcher.hatchSortMethod = hatching.AlternateSort()
    
    # Get slices at specific height
    z = 2.0  # mm
    normal_slice = normal_part.getVectorSlice(z, simplificationFactor=0.1)
    overhang_slice = overhang_part.getVectorSlice(z, simplificationFactor=0.1)
    
    # Generate hatches for each region
    normal_layer = normal_hatcher.hatch(normal_slice)
    overhang_layer = overhang_hatcher.hatch(overhang_slice)
    
    # Create a single figure for both plots
    plt.close('all')  # Close any existing plots
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Function to plot layer geometry with different styles for contours and hatches
    def plot_layer_geometry(layer, contour_color, hatch_color):
        if not layer:
            return
            
        for layerGeom in layer.geometry:
            coords = layerGeom.coords
            if len(coords) == 0:
                continue
                
            if isinstance(layerGeom, ContourGeometry):
                # Plot contours as continuous lines
                ax.plot(coords[:, 0], coords[:, 1], 
                       color=contour_color, linewidth=1.0, linestyle='-')
            
            elif isinstance(layerGeom, HatchGeometry):
                # Plot hatches as pairs of points
                coords = coords.reshape(-1, 2, 2)
                for line in coords:
                    ax.plot(line[:, 0], line[:, 1], 
                           color=hatch_color, linewidth=0.5, linestyle='-')
    
    # Plot normal layer (blue contours, light blue hatches)
    plot_layer_geometry(normal_layer, 'darkblue', 'blue')
    
    # Plot overhang layer (red contours, light red hatches)
    plot_layer_geometry(overhang_layer, 'darkred', 'red')
    
    # Add title and legend
    ax.set_title('Combined Scan Paths')
    ax.plot([], [], 'darkblue-', label='Normal Contours', linewidth=1.0)
    ax.plot([], [], 'blue-', label='Normal Hatches', linewidth=0.5)
    ax.plot([], [], 'darkred-', label='Overhang Contours', linewidth=1.0)
    ax.plot([], [], 'red-', label='Overhang Hatches', linewidth=0.5)
    ax.legend()
    ax.set_aspect('equal')
    
    # Adjust the plot to show all data
    ax.autoscale()
    plt.tight_layout()
    
    # Show the plot and block until the window is closed
    plt.show(block=True)
    
    # Print analysis information
    if normal_layer:
        print('\nNormal Region Analysis:')
        print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(normal_layer)))
        print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(normal_layer)))
    
    if overhang_layer:
        print('\nOverhang Region Analysis:')
        print('Total Path Distance: {:.1f} mm'.format(pyslm.analysis.getLayerPathLength(overhang_layer)))
        print('Total jump distance {:.1f} mm'.format(pyslm.analysis.getLayerJumpLength(overhang_layer)))

if __name__ == '__main__':
    main()
