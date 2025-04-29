import numpy as np
import pyslm.visualise
import pyslm.geometry as geom

# Create a layer
layer = geom.Layer()

# Create a contour geometry
contourGeom = geom.ContourGeometry(mid = 1, bid = 1)
contourGeom.coords = np.array([[0.,0.],
                               [0.,1],
                               [1.,1.],
                               [1.,0.],
                               [0.,0.]])

# Add the contour to the geometry
layer.geometry.append(contourGeom)


hatchGeom = geom.HatchGeometry()
hatchGeom.mid = 1
hatchGeom.bid = 2
hatchGeom.coords = np.array([[0.1, 0.1], [0.9, 0.1], # Hatch Vector 1
                             [0.1, 0.3], [0.9, 0.3], # Hatch Vector 2
                             [0.1, 0.5], [0.9, 0.5], # Hatch Vector 3
                             [0.1, 0.7], [0.9, 0.7], # Hatch Vector 4
                             [0.1, 0.9], [0.9, 0.9]  # Hatch Vector xw5
                            ])

# Append the layer geometry to the layer
layer.geometry.append(hatchGeom)

import matplotlib.pyplot as plt

# Plot the Layer
# handle = pyslm.visualise.plot(layer, plot3D=False, plotOrderLine=True, plotArrows=True)
# Plot using the plot sequential function
pyslm.visualise.plotSequential(layer, plotJumps=True, plotArrows=True)


#write out the scan paths in a file
for layerGeom in layer.geometry:
    coords = layerGeom.coords
    print("coords", coords)

    # if isinstance(layerGeom, ContourGeometry):
    #     delta = np.diff(coords, axis=0)
    #     lineDist = np.hypot(delta[:, 0], delta[:, 1])
    #     totalPathDist = np.sum(lineDist)

    # if isinstance(layerGeom, HatchGeometry):
    #     coords = coords.reshape(-1, 2, 2)
    #     delta = np.diff(coords, axis=1).reshape(-1, 2)
    #     lineDist = np.hypot(delta[:, 0], delta[:, 1])
    #     totalPathDist = np.sum(lineDist)

    # if isinstance(layerGeom, PointsGeometry):
    #     raise Exception('Cannot pass a PointsGeometry to calculate the total path length')
plt.show()