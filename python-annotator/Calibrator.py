#The calibrator is responsible for managing the state
#of camera calibration according to extrinsics.
#Ultimately, it associates camera names with their extrinsic calibration
#matrices. By convention, the projection for camera 0 represents the
#"true" world coordinates in millimeters
import os
import math
from FrameManager import *
from RGBPointCloud import *
import ColorFilters
from RootDirector import *
from Frame import *
from RGBPointCloud import *
import numpy as np
import msgpack
from scipy.spatial import cKDTree
import scipy as sp
import numpy as np
from Frame import *
from matplotlib.mlab import PCA
from sklearn import linear_model

def buildBackgroundVizCloud(leftPoints, rightPoints, floorPoints, backPoints):
    leftColor = [255, 0, 0, 255]
    rightColor = [0, 0, 255, 255]
    backColor = [255, 0, 255, 255]
    floorColor = [0, 255, 0, 255]
    timestamp = 0
    allPoints = []
    allColors = []
    for point in leftPoints:
        allPoints.append(point)
        allColors.append(leftColor)
    for point in rightPoints:
        allPoints.append(point)
        allColors.append(rightColor)
    for point in floorPoints:
        allPoints.append(point)
        allColors.append(floorColor)
    for point in backPoints:
        allPoints.append(point)
        allColors.append(backColor)
    allColors = np.array(allColors)
    allPoints = np.array(allPoints)
    return (allPoints, allColors)

        

def getDeterminant(v1, v2, v3):
    mat = np.vstack((v1, v2, v3))
    return np.linalg.det(mat)

#Apparently, all of the stuff about corners doesn't appear to work.
#So, what we'll do is we'll attempt to find the normals at
#every point in a point cloud, and based on a prior about the view,
#we'll classify points into categories of which wall they belong
#to based on the normal vectors and the prior.
#From there, we'll come up with implicit coordinates for each point
#which are entirely determined by the distances of each point to
#the closest point in each category.
#Finally, in the space defined by these implicit coordinates,
#we pair points which fall within a given distance of each other,
#and use this pairing to compute a best-fit rigid alignment

def computeNormalsFor(pointCloud, NEIGHBORS=20):
    pointArray = np.asarray(pointCloud)
    pointKdTree = cKDTree(pointArray)
    #Go through every point in the array and look at its neighbors.
    #Compute the PCA of these points to spit out a normal vector
    normals = []
    for i in range(pointArray.shape[0]):
        point = pointArray[i]
        _, neighborhoodInds = pointKdTree.query(point, k=NEIGHBORS)
        neighborhood = pointArray[neighborhoodInds]

        #Compute PCA of the neighborhood
        mu = np.mean(neighborhood, axis=0)
        neighborhood = neighborhood - mu

        eigenvectors, eigenvalues, V = np.linalg.svd(neighborhood.T, full_matrices=False)
        smallestEigIndex = np.argmin(eigenvalues)
        smallestEigenVector = eigenvectors[smallestEigIndex]
        normal = smallestEigenVector / np.linalg.norm(smallestEigenVector)

        #Great. Now, the LAST PC axis should be the normal vector, so spit that out
        normals.append(normal)
    normals = np.array(normals)
    return normals

def smoothNormals(pointCloud, normals, NEIGHBORS=500):
    pointArray = np.asarray(pointCloud)
    pointKdTree = cKDTree(pointArray)

    resultNormals = []
    
    for i in range(pointArray.shape[0]):
        point = pointArray[i]
        _, neighborhoodInds = pointKdTree.query(point, k=NEIGHBORS)
        normalNeighborhood = normals[neighborhoodInds]
        meanNormal = np.mean(normalNeighborhood, axis=0)
        resultNormals.append(meanNormal)
    resultNormals = np.array(resultNormals)
    return resultNormals


classifiedWalls = []

def sideImpute(side, back, floor):
    return 2750 - side + back * 0.16260229390038 + floor * 0.00594271220858 
    #return 2816.0142967519951424865 - side + back * 0.16260229390038 + floor * 0.00594271220858 
    #return 2829.040257356004075 + -0.99767 * side + back * 0.12425 + floor * 0.0199686

def computeEmbeddingsFor(pointCloud, rotMat, forwardUnitVec, rightUnitVec, downUnitVec):
    #First, compute the normal vector for every point in the point cloud 
    normals = computeNormalsFor(pointCloud)
    normals = smoothNormals(pointCloud, normals)

    pointArray = np.asarray(pointCloud)

    #Great, now that we have the normals, classify points in the point
    #cloud based on whether they belong to the left wall, floor, right wall,
    #or back wall
    leftWallPoints = []
    rightWallPoints = []
    backWallPoints = []
    floorPoints = []

    leftWallNormals = []
    rightWallNormals = []
    backWallNormals = []
    floorNormals = []

    for i in range(pointArray.shape[0]):
        point = pointArray[i]
        x, y, z = point
        normal = normals[i]
        nx, ny, nz = normal
        #First, correct the normal so that it's always pointing toward a positive z,
        #so that all normals are on the back-faces of objects relative to the camera
        if (nz < 0):
            normal *= -1
        unitNormal = normal / np.linalg.norm(normal)

        #rotUnitNormal = np.dot(np.transpose(rotMat), unitNormal)

        rotUnitNormal = unitNormal

        projForward = abs(np.dot(rotUnitNormal, forwardUnitVec))
        projRight = np.dot(rotUnitNormal, rightUnitVec)
        projDown = abs(np.dot(rotUnitNormal, downUnitVec))
        if (projDown > abs(projRight) and projDown > projForward):
            #Label with floor
            floorPoints.append(point)
            floorNormals.append(unitNormal)
            continue
        if (abs(projRight) > projDown and abs(projRight) > projForward):
            if (x > 0):
                rightWallPoints.append(point)
                rightWallNormals.append(unitNormal)
                continue
            else:
                leftWallPoints.append(point)
                leftWallNormals.append(unitNormal)
                continue
        if (projForward > abs(projRight) and projForward > projDown):
            backWallPoints.append(point)
            backWallNormals.append(unitNormal)
            continue
    leftWallPoints = np.array(leftWallPoints)
    rightWallPoints = np.array(rightWallPoints)
    backWallPoints = np.array(backWallPoints)
    floorPoints = np.array(floorPoints)

    global classifiedWalls
    classifiedWalls.append(buildBackgroundVizCloud(leftWallPoints, rightWallPoints, backWallPoints, floorPoints))

    leftWallNormals = np.array(leftWallNormals)
    rightWallNormals = np.array(rightWallNormals)
    backWallNormals = np.array(backWallNormals)
    floorNormals = np.array(floorNormals)

    max_edge_dist = 50.0


    #Okay, great. Now do a largest component mask on all of those thangs
    leftWallMask = RGBPointCloud.largestComponentMask(leftWallPoints, max_edge_dist=max_edge_dist)
    rightWallMask = RGBPointCloud.largestComponentMask(rightWallPoints, max_edge_dist=max_edge_dist)
    backWallMask = RGBPointCloud.largestComponentMask(backWallPoints, max_edge_dist=max_edge_dist)
    floorMask = RGBPointCloud.largestComponentMask(floorPoints, max_edge_dist=max_edge_dist)

    leftWallPoints = leftWallPoints[leftWallMask]
    rightWallPoints = rightWallPoints[rightWallMask]
    backWallPoints = backWallPoints[backWallMask]
    floorPoints = floorPoints[floorMask]

    leftWallNormals = leftWallNormals[leftWallMask]
    rightWallNormals = rightWallNormals[rightWallMask]
    floorNormals = floorNormals[floorMask]
    backWallNormals = backWallNormals[backWallMask]

    leftImpute = False
    rightImpute = False
    #Set sides to be imputed/not based on which of left/right side has
    #more observed points
    if (leftWallPoints.shape[0] > rightWallPoints.shape[0]):
        rightImpute = True
    else:
        leftImpute = True

    print "left wall", leftWallPoints
    print "right wall", rightWallPoints
    print "back wall", backWallPoints
    print "floor", floorPoints


    leftWallNormalMean = np.mean(np.array(leftWallNormals), axis=0)
    rightWallNormalMean = np.mean(np.array(rightWallNormals), axis=0)
    floorNormalMean = np.mean(np.array(floorNormals), axis=0)
    backWallNormalMean = np.mean(np.array(backWallNormals), axis=0)

    rotFloorNormalMean = np.dot(rotMat, floorNormalMean)
    rotWallNormalMean = np.dot(rotMat, backWallNormalMean)

    print "Floor normal mean, rotated", rotFloorNormalMean
    print "Wall normal mean, rotated", rotWallNormalMean

    empiricalLeftRight = np.cross(floorNormalMean, backWallNormalMean)
    empiricalLeftRight /= np.linalg.norm(empiricalLeftRight)

    print "left wall normal mean: ", leftWallNormalMean
    print "right wall normal mean: ", rightWallNormalMean
    print "floor normal mean: ", floorNormalMean
    print "back wall normal mean: ", backWallNormalMean

    #Okey, great. Now build kd trees for each segment
    leftKdTree = cKDTree(leftWallPoints)
    rightKdTree = cKDTree(rightWallPoints)
    backKdTree = cKDTree(backWallPoints)
    floorKdTree = cKDTree(floorPoints)

    backWallMean = np.mean(backWallPoints, axis=0)
    backWallMeanEmpiricalDot = np.dot(backWallMean, empiricalLeftRight)

    implicitPoints = []
    for i in range(pointArray.shape[0]):
        point = pointArray[i]
        backD, _ = backKdTree.query(point)
        floorD, _ = floorKdTree.query(point)
        leftD, _ = leftKdTree.query(point)
        rightD, _ = rightKdTree.query(point)
        if (leftImpute):
            leftD = sideImpute(rightD, backD, floorD)
        if (rightImpute):
            rightD = sideImpute(leftD, backD, floorD)

        dotEmpirical = np.dot(point, empiricalLeftRight) - backWallMeanEmpiricalDot
        implicitPoint = np.array([leftD, rightD, backD, floorD])
        #implicitPoint = np.array([backD, floorD, dotEmpirical])
        implicitPoints.append(implicitPoint)
    implicitPoints = np.array(implicitPoints)
    print "build Implicit point array"
    print implicitPoints
    print "Getting best fit for left/right wall distance"
    getImplicitPointRegressionData(implicitPoints)
    print "got best fit for left and right wall distance"
    return implicitPoints

def getImplicitPointRegressionData(implicitPoints):
    leftValues = []
    leftPredictVars = []
    for implicitPoint in implicitPoints:
        left, right, back, floor = implicitPoint
        leftValues.append(left)
        leftPredictVars.append(np.array([right, back, floor]))

    leftValues = np.array(leftValues)
    leftPredictVars = np.array(leftPredictVars)

    leftRegr = linear_model.LinearRegression()
    leftRegr.fit(leftPredictVars, leftValues)

    print "Linear coefficients for left [right, back, floor]", leftRegr.coef_
    print "Intercept for left", leftRegr.intercept_

    rightValues = []
    rightPredictVars = []
    for implicitPoint in implicitPoints:
        left, right, back, floor = implicitPoint
        rightValues.append(right)
        rightPredictVars.append(np.array([left, back, floor]))
    rightValues = np.array(rightValues)
    rightPredictVars = np.array(rightPredictVars)


    rightRegr = linear_model.LinearRegression()
    rightRegr.fit(rightPredictVars, rightValues)

    print "Linear coefficients for left [left, back, floor]", rightRegr.coef_
    print "Intercept for left", rightRegr.intercept_
   




def closestPointAlignImplicit(sourcePoints, targetPoints, MATCH_RADIUS=40):
    return alignImplicit(sourcePoints, sourcePoints, targetPoints, targetPoints, MATCH_RADIUS)

def alignImplicit(sourcePoints, sourceImplicit, targetPoints, targetImplicit, MATCH_RADIUS=5):

    sourcePoints = np.asarray(sourcePoints)
    targetPoints = np.asarray(targetPoints)
    if (sourcePoints.size == 0 or targetPoints.size == 0):
        return np.eye(4)

    #Build a kd-tree on sourceImplicit
    print "Building source tree"
    sourceKdTree = cKDTree(sourceImplicit)
    print "building target tree"
    targetKdTree = cKDTree(targetImplicit)


    print "Finding neighbors"
    neighbors = sourceKdTree.query_ball_tree(targetKdTree, MATCH_RADIUS)
    print "neighbors found"
    sourceMatched = []
    targetMatched = []
    for sourceInd in range(sourceImplicit.shape[0]):
        for targetInd in neighbors[sourceInd]:
            sourcePoint = sourcePoints[sourceInd]
            targetPoint = targetPoints[targetInd]
            sourceMatched.append(sourcePoint)
            targetMatched.append(targetPoint)
    print "Number of matched points: ", len(sourceMatched)
    return best4x4RigidFit(sourceMatched, targetMatched)

def fuzz(points, fuzzFactor=1.0):
    return points + fuzzFactor * np.random.standard_normal(points.shape)

def alignCanonicalThree(zerothCloud, firstCloud, secondCloud):
    priorAngle = .3504875
    #Prior for the floor normal at your usual viewing angles
    floorNormalPrior = np.array([0.0, math.cos(priorAngle), math.sin(priorAngle)])
    floorNormalPrior /= np.linalg.norm(floorNormalPrior)

    forwardNormalPrior = np.array([0.0, -math.sin(priorAngle), math.cos(priorAngle)])
    forwardNormalPrior /= np.linalg.norm(forwardNormalPrior)

    leftRightNormalPrior = np.array([1.0, 0.0, 0.0])

    #Great. Build a rotation matrix with those for rows so that we
    #can transform the other vectors
    rotMat = np.vstack((leftRightNormalPrior, floorNormalPrior, forwardNormalPrior))

    #Great. From there, find the 
    
    #First, compute embeddings for each cloud 
    forwardVec = np.array([0, 0, 1.0])
    rightVec = np.array([1.0, 0, 0])
    downVec = np.array([0.0, 1.0, 0.0])
    zerothEmbeddings = computeEmbeddingsFor(zerothCloud, rotMat, forwardVec, rightVec, downVec)
    zerothEmbeddings = fuzz(zerothEmbeddings)
    print "Found embeddings for zero"
    
    forwardVec = np.array([-.7, 0, .7])
    rightVec = np.array([.7, 0, 0.7])
    downVec = np.array([0.0, 1.0, 0.0])
    firstEmbeddings = computeEmbeddingsFor(firstCloud, rotMat, forwardVec, rightVec, downVec)
    firstEmbeddings = fuzz(firstEmbeddings)
    print "Found embeddings for one"

    forwardVec = np.array([.7, 0, .7])
    rightVec = np.array([-.7, 0, .7])
    downVec = np.array([0.0, 1.0, 0.0])
    secondEmbeddings = computeEmbeddingsFor(secondCloud, rotMat, forwardVec, rightVec, downVec)
    secondEmbeddings = fuzz(secondEmbeddings)
    print "Found embeddings for two"

    firstToZero = alignImplicit(firstCloud, firstEmbeddings, zerothCloud, zerothEmbeddings)
    secondToZero = alignImplicit(secondCloud, secondEmbeddings, zerothCloud, zerothEmbeddings)

    zeroToZero = np.eye(4, dtype=np.float32)
    return (zeroToZero, firstToZero, secondToZero)



#Given a list of points on a cube-ish shape,
#a point at one of the corners, and a radius for searches,
#find the unit vector that points in the opposite direction
#of the most corner-y direction
def mostUncorneryVector(pointList, cornerPoint, searchRadius=10.0):
    kdTree = cKDTree(pointList)
    close_indices = kdTree.query_ball_point(cornerPoint, searchRadius)
    print len(close_indices)
    close_points = pointList[close_indices]
    close_vectors = close_points - cornerPoint
    MAX_SAMP = 50
    if (len(close_indices) > MAX_SAMP):
        close_vectors = close_vectors[np.random.choice(close_vectors.shape[0], MAX_SAMP, replace=False), :]

    bigDet = float('-inf')
    bigDetVecs = []
    #Now, go through every triple in close_points and find the
    #triple with the largest determinant
    for v1 in close_vectors:
        for v2 in close_vectors:
            for v3 in close_vectors:
                det = getDeterminant(v1, v2, v3)
                if (det > bigDet):
                    bigDet = det
                    bigDetVecs = [v1, v2, v3]
    v1, v2, v3 = bigDetVecs
    print bigDetVecs
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    result = v1 + v2 + v3
    result = result / np.linalg.norm(result)
    return result

def mostUncorneryPoint(pointList, cornerPoint, searchRadius=50.0, vecRadius=10.0):
    vec = mostUncorneryVector(pointList, cornerPoint, searchRadius=searchRadius)
    return cornerPoint + vecRadius * vec

#Given a collection of vectors,
#find the vector which has the largest magnitude projection
#in a given direction
def biggestAlignment(pointList, v, k=5):
    transposev = np.reshape(v, (3, 1))
    dotProducts = np.dot(pointList, transposev).flatten()
    inds = np.argpartition(dotProducts, -k)[-k:]
    toAverage = pointList[inds]
    result = np.median(toAverage, axis=0)
    return result

def best4x4RigidFit(srcPoints, destPoints):
    print "finding best rigid fit"
    srcPoints = np.array(srcPoints)
    destPoints = np.array(destPoints)

    srcCentroid = np.mean(srcPoints, axis=0, keepdims=True)
    destCentroid = np.mean(destPoints, axis=0, keepdims=True)

    subtractSrcCentroidMat = np.eye(4, dtype=np.float32)
    subtractSrcCentroidMat[0:3, 3] = -srcCentroid

    addDestCentroidMat = np.eye(4, dtype=np.float32)
    addDestCentroidMat[0:3, 3] = destCentroid
    
    srcVecs = srcPoints - srcCentroid
    destVecs = destPoints - destCentroid
    rot_mat_component, _ = sp.linalg.orthogonal_procrustes(srcVecs, destVecs)
    rot_mat_component = np.transpose(rot_mat_component)
    rot_mat = np.eye(4, dtype=np.float32)
    rot_mat[0:3, 0:3] = rot_mat_component
    
    result_mat = np.matmul(addDestCentroidMat, np.matmul(rot_mat, subtractSrcCentroidMat))
    print "found best rigid fit"
    return result_mat

'''
#Given three lists of points, the first of which is a point cloud from default
#orientation, the second from 45 degree orientation, and the third from
#negative 45 degree orientation, return a list of respective matrix estimates to
#align each one to the first [zeroth]
def alignCanonicalThree(zerothCloud, firstCloud, secondCloud):
    #For the zeroth cloud, search in the greatest forward diagonal directions
    leftVec = np.array([-1.0, 0.2, 0.5])
    leftishVec = np.array([-1.0, 0.2, 1.0])
    straightVec = np.array([0.0, 0.2, 1.0])
    rightishVec = np.array([1.0, 0.2, 1.0])
    rightVec = np.array([1.0, 0.2, 0.5])

    zeroLeftCorner = biggestAlignment(zerothCloud, leftishVec)
    zeroRightCorner = biggestAlignment(zerothCloud, rightishVec)
    zeroLeftUncorner = mostUncorneryPoint(zerothCloud, zeroLeftCorner)
    zeroRightUncorner = mostUncorneryPoint(zerothCloud, zeroRightCorner)
    zeroPoints = [zeroLeftCorner, zeroRightCorner, zeroLeftUncorner, zeroRightUncorner]

    firstLeftCorner = biggestAlignment(firstCloud, leftVec)
    firstRightCorner = biggestAlignment(firstCloud, straightVec)
    firstLeftUncorner = mostUncorneryPoint(firstCloud, firstLeftCorner)
    firstRightUncorner = mostUncorneryPoint(firstCloud, firstRightCorner)
    firstPoints = [firstLeftCorner, firstRightCorner, firstLeftUncorner, firstRightUncorner]

    secondLeftCorner = biggestAlignment(secondCloud, straightVec)
    secondRightCorner = biggestAlignment(secondCloud, rightVec)
    secondLeftUncorner = mostUncorneryPoint(secondCloud, secondLeftCorner)
    secondRightUncorner = mostUncorneryPoint(secondCloud, secondRightCorner)
    secondPoints = [secondLeftCorner, secondRightCorner, secondLeftUncorner, secondRightUncorner]

    firstToZero = best4x4RigidFit(firstPoints, zeroPoints)
    secondToZero = best4x4RigidFit(secondPoints, zeroPoints)
    zeroToZero = np.eye(4, dtype=np.float32)
    return (zeroToZero, firstToZero, secondToZero)
'''

    

class Calibrator():
    def __init__(self, rootDirector):
        #TODO: Stateful stuff! Determine if there's
        #already an alignment.dat, and load from that
        #serialized file
        alignFilePath = rootDirector.getAlignmentFilePath()
        if (os.path.isfile(alignFilePath)):
            #The alignment file already exists, so just load that thing
            with open(alignFilePath, 'rb') as alignFile:
                self.alignment = msgpack.unpackb(alignFile.read(), raw=False)
        else:
            #The alignment file doesn't already exist, so create that thing
            calibFolderPath = rootDirector.getCalibFolderPath() 
            cameraLabels = rootDirector.getCameraLabels()

            defaultTransforms = {}
            defaultTransforms["0"] = np.eye(4, dtype=np.float32)
            #Default transforms for cams 1 and 2 are 45-deg
            #rotations about x-z followed by a translation about x
            sin = .70710678118
            cos = .70710678118
            tAmount = 1371.6
            
            defaultTransforms["2"] = np.linalg.inv(np.array([[cos, 0, -sin, tAmount], 
                                                [0, 1, 0, 0], 
                                                [sin, 0, cos, 0],
                                                [0, 0, 0, 1]], dtype=np.float32))

            defaultTransforms["1"] = np.linalg.inv(np.array([[cos, 0, sin, -tAmount], 
                                               [0, 1, 0, 0], 
                                                [-sin, 0, cos, 0],
                                                [0, 0, 0, 1]], dtype=np.float32))
            #For each camera, average all frames for the background
            #and all calibration frames
            clouds = {}
            calibClouds = {}
            defaultCloud = None
            defaultCalibCloud = None

            #Also load particular frames
            #COMMENT ME OUT!
            defaultParticularCloud = None
            particularClouds = {}
            for cameraLabel in cameraLabels:
                #Load particular clouds
                cameraCloudManager = rootDirector.getCloudManager("seq1", cameraLabel, cameraMatrix=np.eye(4))
                #Seek to a particular place
                cameraCloudManager.seekTo(1, 10)
                cameraCloud = cameraCloudManager.getCloud()
                particularClouds[cameraLabel] = cameraCloud
                if (cameraLabel == "0"):
                    defaultParticularCloud = cameraCloud.getPoints()

            for cameraLabel in cameraLabels:
                #First, load the background clouds
                averagedCameraFrame = rootDirector.getAverageBackgroundFrame(cameraLabel)
                averagedCameraFrame = Frame.gaussianConvolveDepth(averagedCameraFrame, sigma=15.0)
                averagedCloud = RGBPointCloud.fromFrame(averagedCameraFrame)

                #averagedCloud.applyStatisticalFilter()
                if (cameraLabel == "0"):
                    defaultCloud = averagedCloud.getPoints()
                clouds[cameraLabel] = averagedCloud

            for cameraLabel in cameraLabels:
                #Great, now load calibration clouds
                averagedCameraFrame = rootDirector.getAverageCalibFrame(cameraLabel)
                averagedCloud = RGBPointCloud.fromFrame(averagedCameraFrame)
                averagedCloud.applyColorFilter(ColorFilters.maskCalibCube, negated=True)
                if (cameraLabel == "0"):
                    defaultCalibCloud = averagedCloud.getPoints()
                calibClouds[cameraLabel] = averagedCloud

            print "Aligning canonical three"
            zerothCloud = np.asarray(clouds["0"].getPoints())
            firstCloud = np.asarray(clouds["1"].getPoints())
            secondCloud = np.asarray(clouds["2"].getPoints())
            canonicalAlign = alignCanonicalThree(clouds["0"].getPoints(), clouds["1"].getPoints(), clouds["2"].getPoints())
            align0, align1, align2 = canonicalAlign
            print canonicalAlign
            print "Canonical three aligned"

            defaultTransforms["0"] = align0
            defaultTransforms["1"] = align1
            defaultTransforms["2"] = align2

            defaultCloudArray = np.asarray(defaultCloud)

            #TODO: Repetition and spaghet -- could really abstract the copypasta away!

            #Great, now determine the best registrations to the default cloud
            #from the other cameras by using iterative closest point
            registrations = {}
            for cameraLabel in cameraLabels:
                #First, apply the default transform for the camera
                #to the camera's cloud
                currentTransform = defaultTransforms[cameraLabel]
                BACKGROUND_REFINE_STEPS=10
                #First series of refine steps is performed with the
                #background image, and then the next series is performed
                #with the calibration cube, at a finer resolution
                CALIB_CUBE_REFINE_STEPS=30
                CALIB_CUBE_REFINE_BIG_STEPS=10
                CALIB_CUBE_REFINE_MEDIUM_STEPS=20
                CALIB_CUBE_REFINE_MATCH_RADIUS=160
                CALIB_CUBE_REFINE_SMALL_MATCH_RADIUS=40
                CALIB_CUBE_REFINE_TINY_MATCH_RADIUS=20
                PARTICULAR_REFINE_STEPS=20
                PARTICULAR_REFINE_RADIUS=15
                print "Background refining"
                for _ in range(BACKGROUND_REFINE_STEPS):

                    otherCloudCopy = clouds[cameraLabel].copy()
                    otherCloudCopy.transform4x4(currentTransform)
                    otherCloudCurrentTransformed = otherCloudCopy.getPoints()

                    otherCloudCurrentTransformed = np.asarray(otherCloudCurrentTransformed)

                    #Do a closest point implicit align
                    transf = closestPointAlignImplicit(otherCloudCurrentTransformed, defaultCloud)
                    #Add to the current transform matrix
                    #Great. Now, the final transformation is the result of applying the default
                    #transform FIRST and then the one found by icp
                    currentTransform = np.matmul(transf, currentTransform)
                print "Calibration refining"
                for stepNum in range(CALIB_CUBE_REFINE_STEPS):
                    RADIUS = 0
                    if (stepNum > CALIB_CUBE_REFINE_MEDIUM_STEPS):
                        RADIUS = CALIB_CUBE_REFINE_TINY_MATCH_RADIUS
                    else:
                        if (stepNum > CALIB_CUBE_REFINE_BIG_STEPS):
                            RADIUS = CALIB_CUBE_REFINE_SMALL_MATCH_RADIUS
                        else:
                            RADIUS = CALIB_CUBE_REFINE_MATCH_RADIUS
                    otherCloudCopy = calibClouds[cameraLabel].copy()
                    otherCloudCopy.transform4x4(currentTransform)
                    otherCloudCurrentTransformed = otherCloudCopy.getPoints()

                    otherCloudCurrentTransformed = np.asarray(otherCloudCurrentTransformed)

                    #Do a closest point implicit align
                    transf = closestPointAlignImplicit(otherCloudCurrentTransformed, defaultCalibCloud, MATCH_RADIUS=RADIUS)
                    #Add to the current transform matrix
                    #Great. Now, the final transformation is the result of applying the default
                    #transform FIRST and then the one found by icp
                    currentTransform = np.matmul(transf, currentTransform)

                #Particular cloud alignment time! COMMENT ME OUT
                for stepNum in range(PARTICULAR_REFINE_STEPS):
                    RADIUS = PARTICULAR_REFINE_RADIUS

                    otherCloudCopy = particularClouds[cameraLabel].copy()
                    otherCloudCopy.transform4x4(currentTransform)
                    otherCloudCurrentTransformed = otherCloudCopy.getPoints()

                    otherCloudCurrentTransformed = np.asarray(otherCloudCurrentTransformed)

                    #Do a closest point implicit align
                    transf = closestPointAlignImplicit(otherCloudCurrentTransformed, defaultParticularCloud, MATCH_RADIUS=RADIUS)
                    #Add to the current transform matrix
                    #Great. Now, the final transformation is the result of applying the default
                    #transform FIRST and then the one found by icp
                    currentTransform = np.matmul(transf, currentTransform)



                '''
                #The registration is FROM other point clouds TO the default
                icp = otherCloudDefaultTransformed.make_IterativeClosestPoint()
                converged, transf, estimate, fitness = icp.icp(otherCloudDefaultTransformed, defaultCloud)
                '''

                registrations[cameraLabel] = currentTransform.tolist()
            #registrations["0"] = align0.tolist()
            #registrations["1"] = align1.tolist()
            #registrations["2"] = align2.tolist()
            self.alignment = registrations 
            
            with open(alignFilePath, 'wb') as alignFile:
                align_message = msgpack.packb(registrations, use_bin_type=True)
                alignFile.write(align_message)
    def getAlignmentMatrixForCamera(self, cameraName):
        alignMatrix = self.alignment[cameraName]
        return alignMatrix

        
