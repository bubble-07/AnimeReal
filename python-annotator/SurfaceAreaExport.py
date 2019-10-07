#Smoll module for exporting an equi-areal cover of the standard body
#with the given radius to the given csv file

import StandardBody
import sys
import numpy as np
from scipy.spatial import cKDTree
import csv

radius = float(sys.argv[1])
out_file = sys.argv[2]

#Radii sizes to test
print "Number of points on standard body: ", StandardBody.pointArray.shape[0]

cover_points = []

pointArray = np.copy(StandardBody.pointArray)

np.random.shuffle(pointArray)

kdTree = cKDTree(pointArray)

unvisitedIndices = set(range(pointArray.shape[0]))

coverCount = 0

while (len(unvisitedIndices) > 0):
    #Pick an arbitrary unvisited index
    ind = unvisitedIndices.pop()
    
    point = pointArray[ind]

    cover_points.append(point)

    otherInds = kdTree.query_ball_point(point, radius)

    #Remove each index retrieved from the unvisited set, since they're covered
    for covered_ind in otherInds:
        unvisitedIndices.discard(covered_ind)

    #Add to the count
    coverCount += 1

print "Number of points in cover: ", coverCount

#Great, now export the cover_points to csv
with open(out_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for point in cover_points:
        coordList = list(point)
        csvwriter.writerow(coordList)

