import numpy as np
import math
import scipy.spatial
#Physics-simulation-based method to do landmark registration
#of the two point clouds. No idea if this will work, but whatever happens,
#it should be fun to watch in action!

class ElasticRegistration:
    #Requires a templateCloud (a NeighborlyPointCloud)
    #and an targetCloud (just a dumb Nx3 array of points)
    #the template cloud will be deformed to the target
    def __init__(self, templateCloud, targetCloud, landmarkDict):
        #Initialize all positions to template point cloud positions
        self.templateCloud = templateCloud
        self.origPositions = np.copy(templateCloud.getPointCloud())
        self.positions = np.copy(templateCloud.getPointCloud())
        self.N_template, self.dim = self.positions.shape
        #Initialize all velocities to zero
        self.velocities = np.zeros((self.N_template, self.dim), dtype=np.float32)
        #Construct a kd tree of the target point cloud
        self.targetCloud = targetCloud
        self.targetKdTree = scipy.spatial.KDTree(self.targetCloud)
        #Dictionary of landmarks
        self.landmarkDict = landmarkDict

        #Strength of forces applied due to distance-preserving constraints
        #self.shapeStrength = 2.4
        self.modShapeStrength = 0.025
        self.shapeStrength = 0.1

        self.angleStrength = 0.1
        #Strength of forces applied to achieve similarity
        self.similarityStrength = 64.0
        #Strength of forces applied to register landmarks
        self.landmarkStrength = 0.02
        #self.landmarkStrength = 0.0
        self.damping = 0.8

        #okay param set
        #self.shapeStrength = 2.0
        #self.similarityStrength = 100.0
        #self.landmarkStrength = 2.0
        #self.damping = 0.7
        
        self.num_target_neighbors = 5
    def getPositions(self):
        return self.positions

    def conformalForce(self, origCos, cos):
        return 1.0 * self.angleStrength * (origCos - cos)
    def landmarkForce(self, distance):
        #The force here is a spring force
        return self.landmarkStrength * distance * distance

    def fittingForce(self, distance):
        #The force here is inverse-square-law, with a cutoff
        min_dist = 10.0
        if (distance < min_dist):
            distance = min_dist
        return self.similarityStrength * (1.0 / (distance * distance))

    def springForce(self, origDistance, actualDistance):
        #The force here is used to punish proportional to percent change
        return self.shapeStrength * (math.log(actualDistance) - math.log(origDistance))

    #One step of the update loop
    def update(self):
        #self.applyConformalForces()
        self.applyModSpringForces()
        self.applySpringForces()
        self.applyLandmarkForces()
        self.applyFittingForces()
        self.applyVelocitySmoothing()
        self.applyDamping()
        self.updatePositions()

    def computeKinecticEnergy(self):
        energy = 0
        #TODO: A loop for this is dumb. Use a numpy-i-fied form
        for i in range(self.N_template):
            vel = self.velocities[i]
            v_squared = np.inner(vel, vel)
            energy += v_squared
        return energy * 0.5

    #Apply velocity smoothing
    def applyVelocitySmoothing(self):
        mix = 0.5
        velUpdate = np.zeros((self.N_template, self.dim))
        for i in range(self.N_template):
            pos = self.positions[i]
            origDists, inds, = self.templateCloud.getNeighborInfo(i)
            otherSum = np.zeros((self.dim))
            for origDist, ind in zip(origDists, inds):
                otherSum += self.velocities[ind]
            otherAverage = otherSum / len(inds)
            velUpdate[i] = (1.0 - mix) * self.velocities[i] + mix * otherAverage
        self.velocities = velUpdate
            


    #Apply landmark registration forces
    def applyLandmarkForces(self):
        for i in range(self.N_template):
            pos = self.positions[i]
            #Determine if there's a landmark for the given position.
            #if so, look it up, and apply landmark force for the point
            try:
                otherInd = self.landmarkDict[i]
                otherPos = self.targetCloud[otherInd]
                dist = np.linalg.norm(otherPos - pos)
                force = self.landmarkForce(dist)
                normal = (otherPos - pos) / dist
                self.velocities[i] += normal * force

            except KeyError:
                continue

    def modSpringForce(self, origDistOne, origDistTwo, distOne, distTwo):
        expected_fraction = origDistOne / origDistTwo
        actual_fraction = distOne / distTwo
        return -1.0 * self.modShapeStrength * (math.log(expected_fraction) - math.log(actual_fraction))

    #Apply spring forces between every neighboring pair of points
    #in the template cloud
    def applyModSpringForces(self):
        for i in range(self.N_template):
            pos = self.positions[i]
            origDists, inds = self.templateCloud.getNeighborInfo(i)
            #For every pair of neighbors
            for origDistOne, indOne in zip(origDists, inds):
                posOne = self.positions[indOne]
                distOne = np.linalg.norm(posOne - pos)
                normalOne = (posOne - pos) / distOne
                for origDistTwo, indTwo in zip(origDists, inds):
                    posTwo = self.positions[indTwo]
                    distTwo = np.linalg.norm(posTwo - pos)
                    normalTwo = (posTwo - pos) / distTwo
                    force = self.modSpringForce(origDistOne, origDistTwo, distOne, distTwo)
                    #Now, apply the force in the positive distOne direction, negative distTwo
                    self.velocities[i] += normalOne * force - normalTwo * force
                    self.velocities[indOne] -= normalOne * force
                    self.velocities[indTwo] += normalTwo * force

    #Applies a force which tries to make the transformation conformal
    def applyConformalForces(self):
        for i in range(self.N_template):
            origPos = self.origPositions[i]
            pos = self.positions[i]
            origDists, inds = self.templateCloud.getNeighborInfo(i)
            #For every pair of neighbors
            for origDistOne, indOne in zip(origDists, inds):
                origPosOne = self.origPositions[indOne]
                posOne = self.positions[indOne]
                distOne = np.linalg.norm(posOne - pos)
                normalOne = (posOne - pos) / distOne
                for origDistTwo, indTwo in zip(origDists, inds):
                    if (indTwo == indOne):
                        continue
                    origPosTwo = self.origPositions[indTwo]
                    posTwo = self.positions[indTwo]
                    distTwo = np.linalg.norm(posTwo - pos)
                    normalTwo = (posTwo - pos) / distTwo

                    origInner = np.inner(origPosOne - origPos, origPosTwo - origPos)
                    origCos = origInner / (origDistOne * origDistTwo)

                    inner = np.inner(posOne - pos, posTwo - pos)
                    cos = inner / (distOne * distTwo)

                    force = self.conformalForce(origCos, cos)

                    distBetween = np.linalg.norm(posTwo - posOne)
                    normalBetween = (posTwo - posOne) / distBetween

                    self.velocities[indOne] += force * normalBetween
                    self.velocities[indTwo] -= force * normalBetween






    #Apply spring forces between every neighboring pair of points
    #in the template cloud
    def applySpringForces(self):
        for i in range(self.N_template):
            pos = self.positions[i]
            origDists, inds = self.templateCloud.getNeighborInfo(i)
            for origDist, ind in zip(origDists, inds):
                otherPos = self.positions[ind]
                #Compute the distance between the two
                dist = np.linalg.norm(otherPos - pos)
                #Compute the magnitude of the force between the two
                force = self.springForce(origDist, dist)
                #Compute the normal vector between the two positions
                normal = (otherPos - pos) / dist
                #Apply the force by changing velocities of this and the other
                self.velocities[i] += normal * force
                self.velocities[ind] -= normal * force
    #Apply damping 
    def applyDamping(self):
        for i in range(self.N_template):
            self.velocities[i] *= self.damping
    #Update positions w.r.t. velocities
    def updatePositions(self):
        for i in range(self.N_template):
            self.positions[i] += self.velocities[i]

    #Apply fitting forces
    def applyFittingForces(self):
        for i in range(self.N_template):
            pos = self.positions[i]
            #Find nearest neighbors in the target
            dists, inds = self.targetKdTree.query([pos], k=self.num_target_neighbors)
            dists = dists[0]
            inds = inds[0]
            for dist, ind in zip(dists, inds):
                otherPos = self.targetCloud[ind]
                #For each of the nearest neighbors, apply a fiting force
                dist = np.linalg.norm(otherPos - pos)
                force = self.fittingForce(dist)
                normal = (otherPos - pos) / dist
                self.velocities[i] += normal * force

        
