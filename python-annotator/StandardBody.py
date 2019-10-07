import pcl
import Rasterizer
import numpy as np
from scipy.spatial import cKDTree

pointCloud = pcl.load("body_model.pcd")
pointArray = np.asarray(pointCloud)
xbounds, ybounds, zbounds = Rasterizer.getPointArrayBounds(pointArray)
xmin, xmax = xbounds
ymin, ymax = ybounds
zmin, zmax = zbounds

xspread = xmax - xmin
yspread = ymax - ymin
zspread = zmax - zmin
xmid = (xmax + xmin) / 2.0
ymid = (ymax + ymin) / 2.0
zmid = (zmax + zmin) / 2.0

hxspread = xspread / 2.0
hyspread = yspread / 2.0
hzspread = zspread / 2.0

xydistsq = hxspread * hxspread + hyspread * hyspread

def xyzToRGBA(point):
    x, y, z = point

    x = x - xmid
    y = y - ymid
    z = z - zmid
    u = (x * hxspread + y * hyspread) / xydistsq
    v = (-y * hyspread + x * hxspread) / xydistsq
    b = z / hzspread
    r = ((u + 1.0) / 2.0) * 255.0
    g = ((v + 1.0) / 2.0) * 255.0
    b = ((b + 1.0) / 2.0) * 255.0
    return [r, g, b, 255.0]

standardColors = []
for i in range(pointArray.shape[0]):
    point = pointArray[i]
    standardColors.append(xyzToRGBA(point))
standardColors = np.array(standardColors)

standardKdTree = cKDTree(pointArray)

