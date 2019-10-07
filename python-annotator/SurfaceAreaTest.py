#Simple test of how many points are required
#to form a cover of the standard body at various ball radii

import StandardBody
import numpy as np
from scipy.spatial import cKDTree
import ColorFilters
import pickle

coloredTemplateFile = "ColoredTemplate.pickle"

partMetricNames = ["GreenLeg", "YellowArm", "RedArm", "RedHand", "YellowHand",
                   "WhiteLeg", "Torso"]
templateColorFilterDict = {"GreenLeg" : ColorFilters.maskGreenLeg,
                   "YellowArm" : ColorFilters.maskYellowArm,
                   "RedArm" : ColorFilters.maskRedArm,
                   "RedHand" : ColorFilters.maskRedHand,
                   "YellowHand" : ColorFilters.maskYellowHand,
                   "WhiteLeg" : ColorFilters.maskWhiteLegInTemplate,
                   "Torso" : ColorFilters.maskTorso}

coloredBody = pickle.load(open(coloredTemplateFile, "rb"))
coloredBody.indices = np.zeros((np.asarray(coloredBody.points).shape[0], 2))


#Radii sizes to test
radiiToTest = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
counts = []
colorCounts = {"GreenLeg" : [],
               "YellowArm" : [],
               "RedArm" : [],
               "RedHand" : [],
               "YellowHand" : [],
               "WhiteLeg" : [],
               "Torso" : []}
print "Number of points on standard body: ", StandardBody.pointArray.shape[0]

for radius in radiiToTest:
    pointArray = np.copy(StandardBody.pointArray)

    np.random.shuffle(pointArray)

    kdTree = cKDTree(pointArray)

    unvisitedIndices = set(range(pointArray.shape[0]))

    coverCount = 0

    while (len(unvisitedIndices) > 0):
        #Pick an arbitrary unvisited index
        ind = unvisitedIndices.pop()
        
        point = pointArray[ind]

        otherInds = kdTree.query_ball_point(point, radius)

        #Remove each index retrieved from the unvisited set, since they're covered
        for covered_ind in otherInds:
            unvisitedIndices.discard(covered_ind)

        #Add to the count
        coverCount += 1

    counts.append(coverCount)

for metricName in partMetricNames:
    colorFilter = templateColorFilterDict[metricName]
    copyBody = coloredBody.copy()
    copyBody.applyColorFilter(colorFilter, negated=True)

    for radius in radiiToTest:
        pointArray = np.copy(np.asarray(copyBody.getPoints()))

        np.random.shuffle(pointArray)

        kdTree = cKDTree(pointArray)

        unvisitedIndices = set(range(pointArray.shape[0]))

        coverCount = 0

        while (len(unvisitedIndices) > 0):
            #Pick an arbitrary unvisited index
            ind = unvisitedIndices.pop()
            
            point = pointArray[ind]

            otherInds = kdTree.query_ball_point(point, radius)

            #Remove each index retrieved from the unvisited set, since they're covered
            for covered_ind in otherInds:
                unvisitedIndices.discard(covered_ind)

            #Add to the count
            coverCount += 1

        colorCounts[metricName].append(coverCount)


print "For whole body"
for radius, count in zip(radiiToTest, counts):
    print "At radius ", radius, " count ", count
totalCounts = 0
for metricName in partMetricNames:
    print "\n"
    print "For ", metricName
    counts = colorCounts[metricName]
    totalCounts += np.array(counts)
    for radius, count in zip(radiiToTest, counts):
        print "At radius ", radius, " count ", count

totalCounts = totalCounts.tolist()
print "\n"
print "For ALL PARTS"
for radius, count in zip(radiiToTest, totalCounts):
    print "At radius ", radius, " count ", count
