import numpy as np
import scipy.spatial

#A point cloud where every point is associated
#with a fixed number of neighbors

class NeighborlyPointCloud():
    def __init__(self, pointCloud, num_neighbors):
        N, k = pointCloud.shape
        self.num_neighbors = num_neighbors
        #First, using the existing pointCloud, build a
        #K-d tree so we can build this thing in O(num_neighbors * N * log(N))
        kdTree = scipy.spatial.KDTree(pointCloud)

        #Each row corresponds to a point, and the entries
        #in the row store the indices of the point's nearest neighbors
        neighborInds = np.zeros((N, num_neighbors), dtype=np.int32)
        neighborDists = np.zeros((N, num_neighbors), dtype=np.float32)

        #Next, using the kdTree we just constructed, go through each point
        #in the point cloud, and query 
        for i in range(N):
            point = pointCloud[i]
            dists, inds = kdTree.query([point], k=num_neighbors + 1)
            dists = dists[0]
            inds = inds[0]
            j = 0
            for dist, ind in zip(dists, inds):
                if (ind != i):
                    neighborInds[i][j] = ind
                    neighborDists[i][j] = dist
                    j += 1
        self.neighborInds = neighborInds
        self.neighborDists = neighborDists
        self.pointCloud = pointCloud
    def getPointCloud(self):
        return self.pointCloud
    def getNeighborInfo(self, i):
        return (self.neighborDists[i], self.neighborInds[i])

