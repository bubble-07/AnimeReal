#Script which takes all autolabels generated
#under a given sequence root, and builds TFrecord files
#(for depth image classifier training)
#This does no shuffling -- TODO: handle that with a separate utility
import sys
import pickle
import ColorFilters
import numpy as np
from RootDirector import *
from DepthTrainingTFWriter import *
import StandardBody
from scipy.spatial import cKDTree
from multiprocessing import Process, Pool
import traceback
import gc

NUM_PROCS = 5

def batcher(seqNameList):
    result = []
    mostRecent = []
    for seqName in seqNameList:
        mostRecent.append(seqName)
        if (len(mostRecent) >= NUM_PROCS):
            result.append(mostRecent)
            mostRecent = []
    result.append(mostRecent)
    return result

#For the alternative template thing here, get those indices on the
#template body which correspond to the torso

coloredTemplateFile = "ColoredTemplate.pickle"
coloredBody = pickle.load(open(coloredTemplateFile, "rb"))
colorArray = np.copy(np.asarray(coloredBody.colors))

torsoMask = ColorFilters.pixelSpaceVectorMask(ColorFilters.maskTorso, colorArray[:, 0:3])
torsoMask = np.logical_not(torsoMask)
torsoInds = np.where(torsoMask)


def alternativeGetTemplateIndexImage(pointCloud, labels):
    try:
        #Does all of the stuff you'd expect here, but first tries
        #to align the pointCloud and labels with a translation

        #First, find the points on the torso in the point cloud (if any)
        #If there aren't any, then we just defer to the default implementation
        torsoCloud = pointCloud.copy()
        torsoCloud.applyColorFilter(ColorFilters.maskTorso, negated=True)

        torsoCloudArray = np.asarray(torsoCloud.getPoints())
        torsoLabelArray = labels[torsoInds]

        if (torsoCloudArray.size < 6):
            return getTemplateIndexImage(pointCloud, labels)

        quantile_q = .1

        #Get the qth quantile of both the point cloud and labels z, and make them align
        #by adjusting the labels as needed
        labelZQuantile = np.quantile(torsoLabelArray[:, 2], quantile_q)

        pointCloudZQuantile = np.quantile(torsoCloudArray[:, 2], quantile_q)

        z_adjust = pointCloudZQuantile - labelZQuantile

        print z_adjust

        labels = np.copy(labels) + np.array([[0.0, 0.0, z_adjust]])

        return getTemplateIndexImage(pointCloud, labels)

    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
    




def advancedGetTemplateIndexImage(pointCloud, labels):

    try:
        #getTemplateIndexImage did things relatively simple, in
        #the sense that there, it was just a one-time assignment
        #of the closest distorted template point from points
        #in the point cloud

        #Here, we'll kinda-sorta do that, but iteratively
        #reassign points in the point cloud array to the average
        #of nearby neighbors (in template-space), weighted by probabilities given
        #by the original distance from the each point to the label cloud
        #We'll then snap those positions to positions on the template manifold,
        #take those as the new coordinates, and do it again.
        #The result is hopefully smoother and more coherent than the original,
        #and hopefully majority-takes-all for mislabelings

        labelKdTree = cKDTree(labels)

        cloudPointArray = np.asarray(pointCloud.getPoints(), dtype=np.float32)


        K = 8
        if (cloudPointArray.shape[0] < K):
            #Special case: not enough points to do the averaging!
            return getTemplateIndexImage(pointCloud, labels)

        #Get the indices of a K-neighborhood of every point in the input cloud
        cloudKdTree = cKDTree(cloudPointArray)
        _, cloudKNeighborhoods = cloudKdTree.query(cloudPointArray, K)

        
        labelDists, labelInds = labelKdTree.query(cloudPointArray)

        #Assign probability weights based on a decreasing exponential
        #of distance
        #Distance (mm) which constitutes a weight reduction to 1/e of what it was
        naturalDist = 50.0
        probWeights = np.exp(-labelDists / naturalDist)

        NUM_AVERAGING_STEPS = 10

        for _ in range(NUM_AVERAGING_STEPS):
            #Get corresponding points on the standard template
            templatePoints = StandardBody.pointArray[labelInds]
            #Space to store the results
            newTemplatePoints = np.copy(templatePoints)
            #TODO: Can this loop be sped up?
            for i in range(cloudPointArray.shape[0]):
                neighborInds = cloudKNeighborhoods[i]
                neighborTemplatePoints = templatePoints[neighborInds]
                neighborProbWeights = probWeights[neighborInds]
                #Okay, now compute a weighted average to determine the new template point pos
                newTemplatePoint = np.average(neighborTemplatePoints, axis=0, weights=neighborProbWeights)
                newTemplatePoints[i] = newTemplatePoint

            #Great, now using the new template points, find the closest labelInds by searching
            #the standard template kd Tree
            _, labelInds = StandardBody.standardKdTree.query(newTemplatePoints)

        cloudDepthIndexArray = np.asarray(pointCloud.getIndices(), dtype=np.int32)

        cloudDepthYindices = cloudDepthIndexArray[:, 0].flatten()
        cloudDepthXindices = cloudDepthIndexArray[:, 1].flatten()

        #Convert label indices to uint16 for output
        labelInds = labelInds.astype(np.uint16)

        result = np.full((424, 512), StandardBody.pointArray.shape[0], dtype=np.uint16)

        result[cloudDepthYindices, cloudDepthXindices] = labelInds

        #TODO: Void labels that aren't close to their constituent parts,
        #and void or correct whole frames which get the backside /frontside thing wrong

        return result
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


    



#Given a RGBPointCloud and corresponding standard body labels,
#return a 424x512 uint16 numpy array of the original depth frame's
#closest matched template points
def getTemplateIndexImage(pointCloud, labels):
    #First thing's first, initialize the result array to "not present"
    #values (taken as StandardBody.pointArray.shape[0])

    result = np.full((424, 512), StandardBody.pointArray.shape[0], dtype=np.uint16)

    #Build a cKdTree on the labels
    labelKdTree = cKDTree(labels)
    
    #Get the cloud point array
    cloudPointArray = np.asarray(pointCloud.getPoints(), dtype=np.float32)
    cloudDepthIndexArray = np.asarray(pointCloud.getIndices(), dtype=np.int32)

    cloudDepthYindices = cloudDepthIndexArray[:, 0].flatten()
    cloudDepthXindices = cloudDepthIndexArray[:, 1].flatten()

    #Find nearest neighbors of every point in the cloud point array
    _, labelInds = labelKdTree.query(cloudPointArray)
    labelInds = labelInds.astype(np.uint16)

    result[cloudDepthYindices, cloudDepthXindices] = labelInds

    return result

def handleSequence(argTuple):
    sequenceRoot, recordingName, destRoot = argTuple

    rootDirector = RootDirector(sequenceRoot)

    #Get the autolabelmanager for the sequence
    autoLabelManager = rootDirector.getAutoLabelFrameReadManager(recordingName)

    #Make destination directory if it doesn't exist
    destDir = os.path.join(destRoot, recordingName)
    os.makedirs(destDir)

    writer = DepthTrainingTFWriter(destDir)

    gc.collect()

    while True:
        #Extract the labeled frame, point cloud and labels
        label = autoLabelManager.getLabel() 
        cloud = autoLabelManager.getCloud()

        #If the cloud has no points, skip it
        if (cloud.getPoints().size != 0):
            origDepthImg = autoLabelManager.getOrigFrame().getDepth()

            templateIndexImg = alternativeGetTemplateIndexImage(cloud, label)

            writer.add(origDepthImg, templateIndexImg)

        advanced = autoLabelManager.advance()
        if (not advanced):
            #No more frames annotated!
            break
    writer.flush()

if __name__ == '__main__':
    sequenceRoot = sys.argv[1]
    destRoot = sys.argv[2]
    rootDirector = RootDirector(sequenceRoot)
    recordingNames = rootDirector.getPeopleRecordingNames()

    #Great, now create a process for each source directory
    argsLists = []
    for recordingName in recordingNames:
        argsLists.append((sequenceRoot, recordingName, destRoot))
    p = Pool(NUM_PROCS)
    p.map(handleSequence, argsLists)

