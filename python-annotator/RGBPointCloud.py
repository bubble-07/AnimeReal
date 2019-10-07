import numpy as np
import colorsys
import traceback
import BodyFilters
import pcl
import ColorFilters
import msgpack
from unionfind import *
import itertools
import KinectTwoCamera
from scipy.spatial import cKDTree
from scipy.stats import mode

class RGBPointCloud:
    #A point cloud is a PCL point cloud object
    #together with a parallel array of uint8 RGB color tuples
    #and a parallel array of original depth image index coordinates
    #and a timestamp
    #TODO: Also add camera identifier here
    def __init__(self, timestamp, points, colors, indices):
        self.timestamp = timestamp
        self.points = points
        self.colors = colors
        self.indices = indices
    def copy(self):
        points = np.copy(np.asarray(self.points))
        colors = np.copy(np.asarray(self.colors))
        indices = np.copy(np.asarray(self.indices))
        return RGBPointCloud(self.timestamp, points, colors, indices)
    def savePointCloud(self, fileName):
        self.points.to_file(fileName)
    def getPoints(self):
        return self.points
    def getIndices(self):
        return self.indices
    #Gets a binary mask of the depth image for points present
    #in this point cloud
    def getIndexMask(self):
        result = RGBPointCloud.emptyFrameWith(0)
        for i, j in self.indices:
            result[i][j] = 1
        return np.array(result, np.uint8)

    #Gets a uint8 RGB image in the same shape
    #as the depth image with the set background color
    #for omitted pixels
    def getRGBImage(self, omittedColor=[0, 255, 0]):
        result = RGBPointCloud.emptyFrameWith(omittedColor)
        ind = 0
        for i, j in self.indices:
            color = self.colors[ind]
            r, g, b = color[0:3]
            result[i][j] = [r, g, b]
            ind += 1
        return np.array(result, np.uint8)

        

    def getColors(self):
        return self.colors
    def getTimestamp(self):
        return self.timestamp
    def pointIterator(self):
        return itertools.izip(self.points, self.colors)
    def pointAndIndexIterator(self):
        return itertools.izip(self.points, self.colors, self.indices)

    #Given a 4x4 (affine) matrix, transform the points in this point cloud
    def transform4x4(self, transform):
        pointArray = np.asarray(self.points)
        if (pointArray.size == 0 or pointArray.shape == ()):
            return
        onesColumn = np.ones((pointArray.shape[0], 1), dtype=np.float32)
        homogPointArray = np.hstack((pointArray, onesColumn))
        transposeTransform = np.transpose(np.array(transform))
        homogResult = np.matmul(homogPointArray, transposeTransform)
        resultPoints = homogResult[:, 0:3].astype(np.float32)
        self.points = pcl.PointCloud(resultPoints)

    #Given a pcl point cloud which incorporates a subset of the
    #points in this cloud, reduce the points in this RGBPointCloud
    #to just those in the passed cloud
    def filterPointsTo(self, subset):
        if (len(self.indices) < 1):
            self.indices = [None] * len(self.colors)
        myDict = {}
        for i in range(len(self.colors)):
            myDict[self.points[i]] = (self.colors[i], self.indices[i])
        resultColors = []
        resultIndices = []
        for point in subset:
            color, index = myDict[point]
            resultColors.append(color)
            resultIndices.append(index)
        self.colors = resultColors
        self.indices = resultIndices
        self.points = subset

    #Filters out green screen background
    def applyBackgroundFilter(self):
        self.applyColorFilter(ColorFilters.maskGreenScreen)

    def applyBodyFilter(self, bodyFilter, negated=False):
        bodyMask = BodyFilters.pixelSpaceBodyMask(bodyFilter, np.asarray(self.points))
        self.applyPointMask(bodyMask, negated=negated)
        

    def applyPointMask(self, mask, negated=False):
        pointArray = np.copy(np.asarray(self.points))
        indexArray = np.copy(np.asarray(self.indices))
        colorArray = np.copy(np.asarray(self.colors))
        
        if (negated):
            mask = np.logical_not(mask)

        if (colorArray.shape[0] == 0):
            #Already empty! Masking does nothing!
            return
            
        self.points = pcl.PointCloud(pointArray[mask])
        self.indices = indexArray[mask]
        self.colors = colorArray[mask]

    def applyColorFilter(self, colorFilter, negated=False):
        colorArray = np.copy(np.asarray(self.colors))
        if (colorArray.size == 0):
            return
        colorMask = ColorFilters.pixelSpaceVectorMask(colorFilter, colorArray[:, 0:3])
        self.applyPointMask(colorMask, negated=negated)

    def voidIfMajorityGreen(self):
        colorArray = np.array(self.colors)
        if (colorArray.size == 0):
            self.colors = np.zeros((0, 4))
            self.points = pcl.PointCloud(np.zeros((0, 3), dtype=np.float32))
            self.indices = np.zeros((0, 2))
            return

        colorMask = ColorFilters.pixelSpaceVectorMask(ColorFilters.maskGreen, colorArray[:, 0:3])
        elems, votes = np.unique(colorMask, return_counts=True)
        if ((elems.shape[0] > 1 and votes[1] < votes[0]) or (elems.shape[0] == 1 and elems[0] == False)):
            self.points = pcl.PointCloud(np.zeros((0, 3), dtype=np.float32))
            self.colors = np.zeros((0, 4))
            self.indices = np.zeros((0, 2))


    #Filters out points based on centroid distance and stddev
    def applyCentroidFilter(self, throw=2.5):
        newPoints = []
        newColors = []
        newIndices = []
        centroidPosition = np.median(np.asarray(self.points), axis=0)
        stdDev = np.std(np.asarray(self.points), axis=0)
        stdDev = np.linalg.norm(stdDev)

        for i in range(len(self.colors)):
            point = self.points[i]
            standardPosition = (point - centroidPosition) / stdDev
            if (np.linalg.norm(standardPosition) < throw):
                newPoints.append(self.points[i])
                newColors.append(self.colors[i])
                newIndices.append(self.indices[i])
        self.colors = newColors
        self.indices = newIndices
        self.points = pcl.PointCloud(np.array(newPoints, dtype=np.float32))
        


    #Applies a statistical filter to the point cloud
    def applyStatisticalFilter(self, mean_k=50, stdev_thresh=1.0):
        resultCloud = self.points
        for i in range(2):
            fil = resultCloud.make_statistical_outlier_filter()
            fil.set_mean_k(mean_k)
            fil.set_std_dev_mul_thresh(stdev_thresh)
            resultCloud = fil.filter()

        self.filterPointsTo(resultCloud)

    @staticmethod
    def randomRows(A, num_rows):
        return A[np.random.choice(A.shape[0], num_rows, replace=False), :]
        
    @staticmethod
    def largestComponentMask(pointArray, max_edge_dist=350.0, max_samp_points=1000):
        #traceback.print_stack()
         
        if (pointArray.shape[0] <= max_samp_points):
            return RGBPointCloud.largestComponentMaskSlow(pointArray, max_edge_dist=max_edge_dist)
        else:
            sample = RGBPointCloud.randomRows(pointArray, max_samp_points)
            sample_component_inds = RGBPointCloud.largestComponentMaskSlow(sample, max_edge_dist=max_edge_dist)
            sample_component = sample[sample_component_inds]

            #Great. Now, find distances to the sample largest component, and keep each point
            #within max_edge_dist
            kdTree = cKDTree(sample_component)
            dists, inds = kdTree.query(pointArray)
            keep = dists < max_edge_dist
            return keep
        

    @staticmethod
    def largestComponentMaskSlow(pointArray, max_edge_dist=50.0):
        if (pointArray.shape[0] == 0):
            return np.array([], dtype=np.int32)

        print pointArray.shape[0]

        #This is connected components, but
        #we're trying to also reduce the number of edges
        #examined in the graph using a kd-tree
        kdTree = cKDTree(pointArray)
        indexPointPairs = kdTree.query_pairs(max_edge_dist)
        #TODO: Examine the source code of the unionfind
        #pip package -- on a half-second glance, it 
        #looks suspicious, and may
        #not be an actually efficient data structure.
        #For now, correctness is all that matters
        u = UnionFind(kdTree.n)
        for i, j in indexPointPairs:
            u.union(i, j)
        #Great, now find the largest component among all connected components
        all_component_parents = np.array([u.find(x) for x in range(kdTree.n)])
        most_common_parent = mode(all_component_parents)[0][0]
        component_inds = np.where(all_component_parents == most_common_parent)

        return component_inds

    #A point cloud filter which finds the largest connected
    #component, with the given maximum distance for an edge
    def applyLargestComponentFilter(self, max_edge_dist=350.0, max_samp_points=1000):

        pointArray = np.asarray(self.points)
        if (pointArray.shape == ()):
            return
        component_inds = RGBPointCloud.largestComponentMask(pointArray, max_edge_dist, max_samp_points)
       
        filteredPoints = pointArray[component_inds]
        self.filterPointsTo(pcl.PointCloud(filteredPoints))


    #Returns a 2-dimensional array where the
    #two dimensions are indexed by depth image index coordinates
    #and the values stored are (point position, point color) tuple pairs
    def get2DRep(self):
        result = RGBPointCloud.emptyFrameWith(None)
        for i in range(len(self.colors)):
            k, l = self.indices[i]
            result[k][l] = (self.points[i], self.colors[i])
        return result

    @staticmethod
    def emptyFrameWith(value):
        #TODO: Camera-specific!
        result = []
        for i in range(424):
            result.append([value] * 512)
        return result

    @staticmethod
    def fromFrame(frame):
        timestamp = frame.getTimestamp()
        rgb = frame.getRGB()
        depth = frame.getDepth()
        #TODO: Leverage camera kind and version info!
        #Don't just blindly assume Kinect v2!

        points = []
        colors = []
        indices = []

        #Convert depth buffer to world coords
        depth_world = KinectTwoCamera.depth_buffer_to_world(depth)
        #And to rgb coords!
        depth_rgb = KinectTwoCamera.depth_buffer_to_rgb(depth)
        #See which rgb coords are in bounds
        rgb_inbounds = KinectTwoCamera.depth_buffer_to_rgb_inbounds(depth)

        depthT = np.transpose(depth)
        depth_inbounds = depthT >= 1.0

        inbounds = np.logical_and(rgb_inbounds, depth_inbounds)

        #Great, now, loop through positions in the depth image
        rheight, rwidth, _ = rgb.shape
        dheight, dwidth = depth.shape

        for y_ind in range(dheight):
            for x_ind in range(dwidth):
                if (not inbounds[x_ind, y_ind]):
                    continue
                rgb_x, rgb_y = depth_rgb[x_ind, y_ind]
                rgb_values = rgb[rgb_y][rgb_x][:]
                b, g, r, a = rgb_values
                rgb_values = (r, g, b, a)
                vertex_pos = depth_world[x_ind][y_ind]
                #Great, now add this stuff to a point cloud
                points.append(vertex_pos)
                colors.append(rgb_values)
                indices.append((y_ind, x_ind))
        points = np.array(points, dtype=np.float32)
        pointcloud = None
        if (points.size > 0):
            pointcloud = pcl.PointCloud(points)
        else:
            pointcloud = pcl.PointCloud()
        return RGBPointCloud(timestamp, pointcloud, colors, indices)
