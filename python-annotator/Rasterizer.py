import pcl
import math
import numpy as np

#Module of utilities for rasterizing RGBPointClouds
def getPointArrayBounds(pointArray):
    xmin = float('+inf')
    xmax = float('-inf')
    ymin = float('+inf')
    ymax = float('-inf')
    zmin = float('+inf')
    zmax = float('-inf')
    for point in pointArray:
        x, y, z = point
        if (x < xmin):
            xmin = x
        if (x > xmax):
            xmax = x
        if (y < ymin):
            ymin = y
        if (y > ymax):
            ymax = y
        if (z < zmin):
            zmin = z
        if (z > zmax):
            zmax = z
    return [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

import StandardBody

#Given a raster 
def colorizeRaster(oldRaster):
    height = len(oldRaster)
    width = len(oldRaster[0])
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            rasterValue = oldRaster[i][j]
            if (rasterValue is not None):
                b, g, r, a = StandardBody.xyzToRGBA(rasterValue)
                result[i][j] = [int(r), int(g), int(b)]
    return result

#Given a raster "image" of 3d coordinates generated using
#rasterizeOrtho, smooth out gaps in the data
def smoothRaster(raster, zvals, depthTol = 50.0):
    zvalsCopy = np.copy(np.array(zvals, dtype=np.float32))
    height = zvalsCopy.shape[0]
    width = zvalsCopy.shape[1]
    rasterCopy = []
    #First, go through and look at every pixel.
    #For each one, if it's occupied, 
    #scan in all directions until there are
    #pixels more than depthTol of its value. Fix up the value there.
    for i in range(height):
        rasterCopyLine = []
        for j in range(width):
            zval = zvals[i][j]
            rasterCopyLine.append(raster[i][j])
            if (zval == float('-inf')):
                continue
            greaterIndices = []
            for k in range(i, height):
                if (zvals[k][j] > zval + depthTol):
                    greaterIndices.append((k, j))
                    break
            for k in reversed(xrange(i)):
                if (zvals[k][j] > zval + depthTol):
                    greaterIndices.append((k, j))
                    break
            for k in range(j, width):
                if (zvals[i][k] > zval + depthTol):
                    greaterIndices.append((i, k))
                    break
            for k in reversed(xrange(j)):
                if (zvals[i][k] > zval + depthTol):
                    greaterIndices.append((i, k))
                    break
            #Only do something about it if we're surrounded on all four sides.
            #If that's the case, void the entry pre-emptively
            if (len(greaterIndices) == 4):
                zvalsCopy[i][j] = float('-inf')
                rasterCopyLine[j] = None
        rasterCopy.append(rasterCopyLine)

    #Now, find all of those indices whch have a border
    #to the left
    leftBordered = []
    for i in range(height):
        occupied = False
        for j in range(width):
            if (zvalsCopy[i][j] == float('-inf')):
                if (occupied):
                    #Must have transitioned from occupied to unoccupied
                    leftBordered.append((i, j))
            else:
                occupied = True

    rightBordered = []
    for i in range(height):
        occupied = False
        for j in reversed(xrange(width)):
            if (zvalsCopy[i][j] == float('-inf')):
                if (occupied):
                    rightBordered.append((i, j))
            else:
                occupied = True

    upBordered = []
    for j in range(width):
        occupied = False
        for i in range(height):
            if (zvalsCopy[i][j] == float('-inf')):
                if (occupied):
                    upBordered.append((i, j))
            else:
                occupied = True

    downBordered = []
    for j in range(width):
        occupied = False
        for i in reversed(xrange(height)):
            if (zvalsCopy[i][j] == float('-inf')):
                if (occupied):
                    downBordered.append((i, j))
            else:
                occupied = True
    bordered = set(leftBordered)
    bordered &= set(rightBordered)
    bordered &= set(upBordered)
    bordered &= set(downBordered)
    #Okay, great, now "bordered" contains all of the indices we need
    #to fill with something via linear interpolation
    for i, j in bordered:
        #Find bordering occupied values
        low_j = j
        for k in reversed(xrange(j)):
            if (zvalsCopy[i][k] != float('-inf')):
                low_j = k
                break
        high_j = j
        for k in range(j, width):
            if (zvalsCopy[i][k] != float('-inf')):
                high_j = k
                break
        low_i = i
        for k in reversed(xrange(i)):
            if (zvalsCopy[k][j] != float('-inf')):
                low_i = k
                break
        high_i = i
        for k in range(i, height):
            if (zvalsCopy[k][j] != float('-inf')):
                high_i = k
                break

        j_spread = float(high_j - low_j)
        i_spread = float(high_i - low_i)
        tot_spread = j_spread + i_spread
        low_j_weight = float(high_j - j) / j_spread
        low_i_weight = float(high_i - i) / i_spread

        horiz_interp = low_j_weight * rasterCopy[i][low_j] + (1.0 - low_j_weight) * rasterCopy[i][high_j]
        vert_interp = low_i_weight * rasterCopy[low_i][j] + (1.0 - low_i_weight) * rasterCopy[high_i][j]
        horiz_z_interp = low_j_weight * zvalsCopy[i][low_j] + (1.0 - low_j_weight) * zvalsCopy[i][high_j]
        vert_z_interp = low_i_weight * zvalsCopy[low_i][j] + (1.0 - low_i_weight) * zvalsCopy[high_i][j]
        vert_weight = j_spread / tot_spread
        horiz_weight = i_spread / tot_spread
        interp_point = vert_weight * vert_interp + horiz_weight * horiz_interp
        interp_z = vert_weight * vert_z_interp + horiz_weight * horiz_z_interp
        zvalsCopy[i][j] = interp_z 
        rasterCopy[i][j] = interp_point
    #Finally, fix 1-pixel-wide inlets
    #Horizontally
    for i in range(height):
        for j in range(1, width - 1):
            left_z = zvalsCopy[i][j - 1]
            right_z = zvalsCopy[i][j + 1]
            left_point = rasterCopy[i][j - 1]
            right_point = rasterCopy[i][j + 1]
            if (left_z != float('-inf') and
                zvalsCopy[i][j] == float('-inf') and
                right_z != float('-inf')):
                zvalsCopy[i][j] = (left_z + right_z) / 2.0
                rasterCopy[i][j] = (left_point + right_point) / 2.0
    for j in range(width):
        for i in range(1, height - 1):
            up_z = zvalsCopy[i - 1][j]
            down_z = zvalsCopy[i + 1][j]
            down_point = rasterCopy[i + 1][j]
            up_point = rasterCopy[i - 1][j]
            if (up_z != float('-inf') and
                zvalsCopy[i][j] == float('-inf') and
                down_z != float('-inf')):
                zvalsCopy[i][j] = (up_z + down_z) / 2.0
                rasterCopy[i][j] = (up_point + down_point) / 2.0
            

    return rasterCopy

#Given a point index (a 2d array containing 3d points or None), generate a point cloud
def pointIndexToCloud(pointIndex):
    pointList = []
    for row in pointIndex:
        for item in row:
            if (item is not None):
                pointList.append(item)
    pointList = np.array(pointList, dtype=np.float32)
    return pcl.PointCloud(pointList)

'''
#Given a raster "image" of 3d coordinates generated using
#rasterizeOrtho, smooth out gaps in the data
def smoothRaster(raster, zvals):
    zvalsCopy = np.copy(np.array(zvals, dtype=np.float32))
    rasterCopy = []
    #First, void all entries in the zvals which
    #have differences with adjacent pixels falling
    #outside of the depth tolerance

    height = len(raster)
    width = len(raster[0])
    for i in range(height):
        rasterCopyLine = []
        for j in range(width):
            rasterCopyLine.append(raster[i][j])
        rasterCopy.append(rasterCopyLine)
    
    for i in range(height):
        for j in range(width):
            zval = zvalsCopy[i][j]
            if (zval == float('-inf')):
                continue
            neighbors = []
            if (i > 0):
                neighbors.append((i - 1, j))
            if (i < height - 1):
                neighbors.append((i + 1, j))
            if (j > 0):
                neighbors.append((i, j - 1))
            if (j < width - 1):
                neighbors.append((i, j + 1))
            greaterValues = []
            greaterPositions = []
            for k, l in neighbors:
                if (zvalsCopy[k][l] > zval):
                    greaterValues.append(zvalsCopy[k][l])
                    greaterPositions.append(rasterCopy[k][l])
            if (len(greaterValues) > 2):
                zvalsCopy[i][j] = sum(greaterValues) / float(len(greaterValues))
                rasterCopy[i][j] = sum(greaterPositions) / float(len(greaterValues))

    #Do a second pass, and average with neighbors
    for i, j in bordered:
        neighbors = []
        if (i > 0):
            neighbors.append((i - 1, j))
        if (i < height - 1):
            neighbors.append((i + 1, j))
        if (j > 0):
            neighbors.append((i, j - 1))
        if (j < width - 1):
            neighbors.append((i, j + 1))
        values = []
        for k, l in neighbors:
            if (rasterCopy[k][l] is not None):
                values.append(rasterCopy[k][l])
        rasterCopy[i][j] = sum(values) / float(len(values))

    return rasterCopy
'''

class RasterizeParams():
    def __init__(self, projMatrix, xmin, ymin, mmPerPix):
        self.projMatrix = projMatrix
        self.xmin = xmin
        self.ymin = ymin
        self.mmPerPix = mmPerPix

    def project(self, point):
        projPoint = np.matmul(self.projMatrix, point)
        px, py, pz = projPoint
        ix = int(math.floor((px - self.xmin) / self.mmPerPix))
        iy = int(math.floor((py - self.ymin) / self.mmPerPix))
        return (ix, iy, pz)

#Rasterize from orthonormal viewing with a CCW-oriented viewing plane given
#by xvec and yvec (unit vectors perpendicular to each other)
def rasterizeOrtho(cloud, xvec, yvec, mmPerPix, pzThrow=0.0):
    origPoints = np.asarray(cloud)

    zvec = np.cross(xvec, yvec)
    projMatrix = np.transpose(np.array([xvec, yvec, zvec]))
    projectedPoints = []
    for point in cloud:
        projPoint = np.matmul(projMatrix, point)
        projectedPoints.append(projPoint)
    xbounds, ybounds, _ = getPointArrayBounds(projectedPoints)
    xmin, xmax = xbounds
    ymin, ymax = ybounds
    xspread = xmax - xmin
    yspread = ymax - ymin

    width = int(math.ceil(xspread / mmPerPix))
    height = int(math.ceil(yspread / mmPerPix))


    result = []
    zvals = []
    for _ in range(height):
        result.append([None] * width)
        zvals.append([float('-inf')] * width)

    rasterizeParams = RasterizeParams(projMatrix, xmin, ymin, mmPerPix)

    #TODO: Remove spaghet reference to projectedPoints
    for i in range(len(projectedPoints)):
        origpoint = origPoints[i]
        ix, iy, pz = rasterizeParams.project(origPoints[i])
        if (pz < pzThrow):
            continue
        oldzval = zvals[iy][ix]
        #Accept only points with higher z-values
        #belonging to the same bucket
        if (pz > oldzval):
            result[iy][ix] = origpoint
            zvals[iy][ix] = pz

    return (result, zvals, rasterizeParams)
