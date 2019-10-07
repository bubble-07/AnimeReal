#Labels a given sequence of frames
#with body position labels by fitting
#a neural network which takes space, time
#coordinates from the input and outputs
#spatial coordinates on the template body

from __future__ import print_function
from FrameManager import *
from CloudManager import *
from RootDirector import *
from AutoLabelManager import *
from MultiCloudManager import *
import colorsys
import BodyFilters
import pickle
import VizUtils
import PointTracking
import StandardBody
import time
import os
import numpy as np 
from scipy.spatial import cKDTree
import sys
import random
import itertools
import math
import cv2

import matplotlib
matplotlib.use('WX')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import scipy as sp
from scipy.stats import norm
import tensorflow as tf
from tensorflow.python.client import timeline

import ColorFilters

COVER_RADIUS = 3.0

SHUFFLE_BUFFER_SIZE=3000

#PATCH PARAMS
#Size of neighborhood for patches
NEIGHBORHOOD_SIZE=10

#Number of times to cover the body template manifold with randomized local patches
NUM_LOCAL_COVERINGS=0 #10

#Number of times to cover the body template manifold 
NUM_GLOBAL_COVERINGS=10
NUM_GLOBAL_COVERINGS=0

#Number of times to cover stitches
NUM_STITCH_COVERINGS=0 #10

#Number of times to cover rigid patches
NUM_RIGID_COVERINGS=30

#Number of times to cover semi-locally
NUM_SEMILOCAL_COVERINGS=10

#Maximum radius to explore for semilocal covering, in centimeters
semilocal_covering_radius = 5 #10
semilocal_covering_decline_rate = 0.8 #0.9


#Weights to afford to each
local_covering_weight = 10.0
global_covering_weight = 0.0 #Global covering seems bad for multiple reasons
stitch_covering_weight = 50.0
rigid_covering_weight = 400.0 #200.0
semilocal_covering_weight = 20.0

semilocal_weight_func = lambda sq_dist : semilocal_covering_weight * math.pow(semilocal_covering_decline_rate, math.sqrt(sq_dist))

#MARKER PARAMS
#Closeness required between a tracking point
#and a part before the correspondence is thrown out
#(in centimeters)
MARKER_PART_MAX_DIST = 2.0

MARKER_HUE_GIVEUP = 10.0

#RECONSTRUCT LOSS PARAMS
reconstruct_weight = 1.0
reconstruct_reproject_weight = 1.0


#LOSS FUNCTION WEIGHTING
isometricity_weight = 50.0 #50.0 #200.0
manifold_weight = 0 #5.0
part_weight = 30.0
reproject_weight = 250.0 #1000.0
rev_reproject_weight = reproject_weight
marker_weight = 100.0

VIEW_LOSS_AFTER = 100

VIEW_SNAPSHOT_AFTER = 1000

#EXPERIMENTAL SECTION:
''''
time_network_layers = 3
num_rigid_slots = 20
time_network_layer_width_multiplier = 8
time_network_out_net_width = 500
time_network_out_net_layers = 4
'''

#TRAINING PARAMS
init_training_rate = 1e-5
reconstruct_training_rate = 1e-4
decay_steps = 6000
decay_rate = 0.5

init_training_rate = 2e-5
decay_steps = 4000
decay_rate = 0.5
reconstruct_training_rate = 1e-5

reconstruct_training_rate = 1e-4
reconstruct_training_iters = 1500 #4000 #8000

momentum_multiplier = 0.25
#Number of training iterations per "megabatch", essentially
#when we say it's good enough for processing our chunk of frames
#to process in one training run
training_iters = 6000 #12000

#Number of frames per megabatch
MEGABATCH_SIZE=300

#Batching Params
RECONSTRUCT_BATCH_SIZE=1000
BATCH_SIZE = 1000
REV_BATCH_SIZE = BATCH_SIZE
PER_COLOR_BATCH_SIZE = 3000
DIST_BATCH_SIZE = 1000
MARKER_BATCH_SIZE = 500

#NETWORK PARAMS
fluid_network_layers = 10
fluid_network_width = 500
fluid_network_segments = 10 #4
ACTIV=tf.nn.elu

partMetricNames = ["GreenLeg", "YellowArm", "RedArm", "RedHand", "YellowHand",
                   "WhiteLeg", "Torso"]
templateColorFilterDict = {"GreenLeg" : ColorFilters.maskGreenLeg,
                   "YellowArm" : ColorFilters.maskYellowArm,
                   "RedArm" : ColorFilters.maskRedArm,
                   "RedHand" : ColorFilters.maskRedHand,
                   "YellowHand" : ColorFilters.maskYellowHand,
                   "WhiteLeg" : ColorFilters.maskWhiteLegInTemplate,
                   "Torso" : ColorFilters.maskTorso}

rgbColorFilterDict = {"GreenLeg" : ColorFilters.maskGreenLeg,
                   "YellowArm" : ColorFilters.maskYellowArm,
                   "RedArm" : ColorFilters.maskRedArm,
                   "RedHand" : ColorFilters.maskRedHand,
                   "YellowHand" : ColorFilters.maskYellowHand,
                   "WhiteLeg" : ColorFilters.maskWhiteLegInTemplate,
                   "Torso" : ColorFilters.maskTorso}

stitchBoundaries = [("GreenLeg", "Torso"), ("YellowArm", "Torso"), ("YellowHand", "YellowArm"),
                    ("RedArm", "Torso"), ("RedHand", "RedArm"), ("WhiteLeg", "Torso")]

rigidRegionMasks = [BodyFilters.maskLeftInnerArm, BodyFilters.maskRightInnerArm,
                    BodyFilters.maskLeftOuterArm, BodyFilters.maskRightOuterArm,
                    BodyFilters.maskLeftUpperLeg, BodyFilters.maskRightUpperLeg,
                    BodyFilters.maskLeftLowerLeg, BodyFilters.maskRightLowerLeg,
                    BodyFilters.maskTorso, BodyFilters.maskHead]

stitch_width = 3.0 #5.0

#Define a dictionary from part names to numpy arrays containing all points in the
#template body which they correspond to

templatePartPointsDict = {}
templatePartPointsDictCentered = {}
templatePartCoverDict = {}

coloredTemplateFile = "ColoredTemplate.pickle"

coloredBody = pickle.load(open(coloredTemplateFile, "rb"))
coloredBody.indices = np.zeros((np.asarray(coloredBody.points).shape[0], 2))
bodyPointArray = np.copy(np.asarray(coloredBody.points))
#Convert body point array to metric coordinates in meters
bodyPointArray = bodyPointArray / 10.0
bodyCentroid = np.mean(bodyPointArray, axis=0, keepdims=True)

bodyKdTree = cKDTree(bodyPointArray)

#Before doing anything else, first, build a cover of the entire body

unvisitedIndices = set(range(bodyPointArray.shape[0]))

bodyCover = []

while (len(unvisitedIndices) > 0):
    #Pick an arbitrary unvisited index
    ind = unvisitedIndices.pop()
    
    point = bodyPointArray[ind]
    bodyCover.append(point)

    otherInds = bodyKdTree.query_ball_point(point, COVER_RADIUS)

    #Remove each index retrieved from the unvisited set, since they're covered
    for covered_ind in otherInds:
        unvisitedIndices.discard(covered_ind)

bodyCover = np.array(bodyCover, dtype=np.float32)

for metricName in partMetricNames:
    colorFilter = templateColorFilterDict[metricName]
    copyBody = coloredBody.copy()

    #From the colored body, apply the color filter and a statistical filter
    copyBody.applyColorFilter(colorFilter, negated=True)
    copyBody.applyLargestComponentFilter(max_edge_dist=50.0)

    pointSubset = np.copy(np.asarray(copyBody.getPoints()))

    #Convert the point subset coordinates to metric in meters
    pointSubset = pointSubset / 10.0
    

    templatePartPointsDict[metricName] = pointSubset
    templatePartPointsDictCentered[metricName] = pointSubset - bodyCentroid

    #Great, now for each part, build a cover
    pointSubset = np.copy(pointSubset)

    np.random.shuffle(pointSubset)

    kdTree = cKDTree(pointSubset)

    unvisitedIndices = set(range(pointSubset.shape[0]))

    cover = []

    while (len(unvisitedIndices) > 0):
        #Pick an arbitrary unvisited index
        ind = unvisitedIndices.pop()
        
        point = pointSubset[ind]
        cover.append(point)

        otherInds = kdTree.query_ball_point(point, COVER_RADIUS)

        #Remove each index retrieved from the unvisited set, since they're covered
        for covered_ind in otherInds:
            unvisitedIndices.discard(covered_ind)

    cover = np.array(cover, dtype=np.float32)
    templatePartCoverDict[metricName] = cover

#Given a dictionary from part names to points in a given frame,
#and a collection of all points in a given frame, return a list of points
#representing the best rigid alignment of all points in the frame to the template
def getRigidAlignment(pointsDict, allPoints):
    #Step 1: Standardize so centroid of the frame is at zero
    frameCentroid = np.mean(allPoints, axis=0, keepdims=True)
    standardPoints = allPoints - frameCentroid
    #Step 2: Set up orthogonal procrustes to find the best rotation.
    #For this, we randomly pair points belonging to the same part
    #between the frame and the (centroid-subtracted) template.
    PAIRS_PER_PART = 20
    framePoints = np.zeros((0, 3))
    templatePoints = np.zeros((0, 3))
    for metricName in partMetricNames:
        if (pointsDict[metricName].shape[0] == 0):
            continue
        templatePartPoints = templatePartPointsDictCentered[metricName]
        framePartPoints = pointsDict[metricName] - frameCentroid
        templatePartChosen = randomRows(templatePartPoints, PAIRS_PER_PART)
        framePartChosen = randomRows(framePartPoints, PAIRS_PER_PART)

        templatePoints = np.vstack((templatePoints, templatePartChosen))
        framePoints = np.vstack((framePoints, framePartChosen))

    #Solve orthogonal procrustes
    ortho_mat, _ = sp.linalg.orthogonal_procrustes(framePoints, templatePoints)

    rot_mat = ortho_mat

    #Not done yet! Now, we have an orthogonal matrix. However, the determinant
    #of the matrix could be negative. In that case, what we should do is we should
    #find the template z-axis and flip everything about that
    if (np.linalg.det(ortho_mat) < 0):
        z_flip_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        rot_mat = np.matmul(ortho_mat, z_flip_mat)

    #Rotate all frame points
    rotatedStandardPoints = np.dot(standardPoints, rot_mat)

    #Align zero with body centroid 
    result = rotatedStandardPoints + bodyCentroid


    #Not done yet! Since a frame doesn't necessarily yield a full capture of the
    #body, iteratively find the closest points to every transformed frame
    #point on the template and translate so as to minimize the sum-of-distance-squared

    #But first, a heuristic: Move things out toward the screen by half a foot
    std_z_vec = np.array([[0.0, 0.0, -1.0]])
    z_vec = np.dot(std_z_vec, rot_mat)
    half_foot = 6.0 * 2.54

    z_vec *= half_foot
    result += z_vec


    NUM_REFINE_STEPS=2
    for _ in range(NUM_REFINE_STEPS):
        _, closestBodyPointInds = bodyKdTree.query(result)
        closestBodyPoints = bodyPointArray[closestBodyPointInds]
        diff = closestBodyPoints - result
        mean_diff = np.mean(diff, axis=0, keepdims=True)
        result = result + mean_diff


    return result


def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=True)]

def fcLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=ACTIV,
            reuse=reuse, scope=scope)

def fcLinLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=None, reuse=reuse, scope=scope)

def normRelaxer(x):
    eps = 0.01
    return tf.minimum(x, 3.14159 + eps * x)

#From a batch of axis-angle vectors and a batch of vectors to transform, apply Rodrigues' rotation formula
def rodrigues(rotationVecs, v):
    eps = 0.01
    norms = tf.maximum(tf.norm(rotationVecs, axis=-1, keepdims=True), eps)
    angles = normRelaxer(norms)
    #Following https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
    k = rotationVecs / norms
    cosine = tf.cos(angles)
    sine = tf.sin(angles)
    k_cross_v = tf.cross(k, v)
    k_dot_v = tf.reduce_sum(k * v, axis=-1, keepdims=True)
    k_k_dot_v = k * k_dot_v
    return v * cosine + k_cross_v * sine + k_k_dot_v * (1.0 - cosine)

#Given spatial positions as a (batch_size, 3) vector, and peeled vectors
#and matrices as given by a rigid time params network, return the (batch_size, num_rigid_slots, 3)
#tensor of transformed spatial positions
def applyRigidTimeTransforms(spatial_positions, peeled_rotation, peeled_translation):
    #TODO: Does the broadcast work this way? If not, you may need to swizzle some things
    broadened_positions = tf.ones([tf.shape(spatial_positions)[0], num_rigid_slots, 3]) * tf.expand_dims(spatial_positions, axis=1)
    
    translated_positions = broadened_positions + peeled_translation
    rotated_positions = rodrigues(peeled_rotation, translated_positions)
    #TODO: Experiment without this dumb as fuck line
    #rotated_positions = translated_positions
    return tf.squeeze(rotated_positions)


#A network where input vectors are used to determine rigid transformations,
#which are then applied to the spatial component of the input vector
def fluidTimeNetwork(x, reuse, namePrefix):
    scale_fac = 10.0

    x = x / scale_fac

    with tf.variable_scope(namePrefix):
        spatial_positions = x[:, 0:3]
        #First thing's first: Figure out how we parameterize our rigid transform!
        #Initial expansion layer
        with tf.variable_scope(namePrefix + "ExpandInitial") as s:
            out = fcLayer(x, fluid_network_width, reuse, s)
        #Middle layers
        for i in range(fluid_network_layers):
            with tf.variable_scope(namePrefix + "FC" + str(i)) as s:
                with tf.name_scope(namePrefix + "FC" + str(i)):
                    out = out + fcLayer(out, fluid_network_width, reuse, s)
        #For the final output, take (1 + fluid_network_segments * 2) * 3
        #linearly-transformed network outputs and interpret them
        #as pre-translation followed by a repeating sequence of rotations
        #and translations
        seg_slots = 1 + 2 * fluid_network_segments
        seg_params = seg_slots * 3
        with tf.variable_scope(namePrefix + "FinalLinear") as s:
            rigid_params = fcLinLayer(out, seg_params, reuse, s)
        reshaped_params = tf.reshape(rigid_params, [-1, seg_slots, 3])
        pre_translate = reshaped_params[:, 0, :]

        out = spatial_positions + pre_translate

        for i in range(1, seg_slots, 2):
            rotate = reshaped_params[:, i, :]
            post_translate = reshaped_params[:, i + 1, :]
            out = rodrigues(rotate, out)
            out = out + post_translate
        return out * scale_fac
    
def approxNetwork(x, reuse, namePrefix):
    return fluidTimeNetwork(x, reuse, namePrefix)

#Given an m1xdim and a m2xdim tensor,
#compute the m1xm2 tensor of pairwise squared euclidean distances
def sqDistanceMatrix(A, B):
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    return na - 2*tf.matmul(A, B, False, True) + nb

#Given a _ x NEIGHBORHOOD_SIZE x 3 tensor, interpret it as a collection of collections of points
#of dimension dim, and for each collection, compute the square distance matrix
def sqDistanceMatrices(PointCollections):
    #Use a tensorflow while loop for this thing

    #First variable: index into the tensor
    #second: result which is being built up
    loop_vars = (tf.constant(0), tf.zeros([0, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE]))
    #Loop condition: Go through every concievable point collection index
    c = lambda i, result : i < tf.shape(PointCollections)[0]
    b = lambda i, result : (i + 1, 
            tf.concat([result, tf.expand_dims(sqDistanceMatrix(PointCollections[i], PointCollections[i]), axis=0)], 0))
    _, final_result = tf.while_loop(c, b, loop_vars, 
                        shape_invariants=(tf.constant(0).get_shape(), tf.TensorShape([None, NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE])))
    #TODO: Tune the number of parallel operations here
    return final_result


#First thing's first -- let's build a collection of small surface patches
#which completely cover the colored template body
dist_a_points = []
dist_b_points = []
dist_sq_dists = []
dist_weights = []

def addNeighborhoodWithWeight(neighborhood, weight):
    func = lambda _ : weight
    addNeighborhoodWithFuncWeight(neighborhood, func)

def addNeighborhoodWithFuncWeight(neighborhood, func):
    global dist_a_points
    global dist_b_points
    global dist_sq_dists
    global dist_weights
    for i in range(neighborhood.shape[0]):
        point_a = neighborhood[i]
        for j in range(i):
            point_b = neighborhood[j]
            diffs = point_a - point_b
            sq_dist = np.sum(diffs * diffs)

            if (sq_dist <= 0.0000001):
                continue
            
            dist_a_points.append(point_a)
            dist_b_points.append(point_b)
            dist_sq_dists.append(sq_dist)
            weight = func(sq_dist)
            dist_weights.append(weight)

#Cover the standard body manifold NUM_LOCAL_COVERINGS times with (possibly-overlapping)
#local patches of NEIGHBORHOOD_SIZE
num_covers = 0
for i in range(NUM_LOCAL_COVERINGS):

    pointArray = np.copy(bodyPointArray)
    #Randomize the order of pointArray rows
    np.random.shuffle(pointArray)

    #Kdtree for fast nearest-neighbor lookup
    kdTree = cKDTree(pointArray)


    unvisitedIndices = set(range(pointArray.shape[0]))

    while (len(unvisitedIndices) > 0):
        #Pick an arbitrary unvisited index
        ind = unvisitedIndices.pop()
        
        point = pointArray[ind]

        #Find NEIGHBORHOOD_SIZE indices of nearest points
        _, otherInds = kdTree.query(point, k=NEIGHBORHOOD_SIZE)

        #Remove each index retrieved from the unvisited set, since they're covered
        for covered_ind in otherInds:
            unvisitedIndices.discard(covered_ind)
        
        #Great, retrieve that collection of points
        neighborhood = pointArray[otherInds]

        num_covers += 1
        addNeighborhoodWithWeight(neighborhood, local_covering_weight)
print("Number of local cover neighborhoods: ", num_covers)
num_covers = 0

#Now do the same for global coverings (these are randomly chosen, not necessarily
#from the same neighborhood)
for i in range(NUM_GLOBAL_COVERINGS):
    pointArray = np.copy(bodyPointArray)
    np.random.shuffle(pointArray)

    unvisitedIndices = set(range(pointArray.shape[0]))

    while (len(unvisitedIndices) > NEIGHBORHOOD_SIZE):
        #Pull out NEIGHBORHOOD_SIZE arbitrary unvisited indices
        inds = []
        for i in range(NEIGHBORHOOD_SIZE):
            ind = unvisitedIndices.pop()
            inds.append(ind)

        inds = np.array(inds)

        neighborhood = pointArray[inds]

        num_covers += 1
        addNeighborhoodWithWeight(neighborhood, global_covering_weight)

print("Number of global cover neighborhoods: ", num_covers)
num_covers = 0

for i in range(NUM_SEMILOCAL_COVERINGS):
    pointArray = np.copy(bodyPointArray)
    np.random.shuffle(pointArray)
    kdTree = cKDTree(pointArray)
    unvisitedIndices = set(range(pointArray.shape[0]))
    while (len(unvisitedIndices) > NEIGHBORHOOD_SIZE):
        #Pull out a point from the pointArray
        ind = unvisitedIndices.pop()

        point = pointArray[ind]
        otherInds = kdTree.query_ball_point(point, semilocal_covering_radius)
        otherInds = np.array(otherInds, np.int32)

        #Great, now select NEIGHBORHOOD_SIZE points among otherInds to make up the neighborhood
        neighborhoodInds = randomRows(otherInds, NEIGHBORHOOD_SIZE)

        for covered_ind in neighborhoodInds:
            unvisitedIndices.discard(covered_ind)

        neighborhood = pointArray[neighborhoodInds]

        num_covers += 1

        addNeighborhoodWithFuncWeight(neighborhood, semilocal_weight_func)

print("Number of semilocal cover neighborhoods: ", num_covers)
num_covers = 0

#Great, now compose those patches which straddle stitching boundaries between different body parts
for i in range(NUM_STITCH_COVERINGS):
    for stitchBoundary in stitchBoundaries:
        one, two = stitchBoundary
        onePoints = templatePartPointsDict[one]
        twoPoints = templatePartPointsDict[two]
        #Great, now with those, filter one and two based on the stitch width
        origOnekdTree = cKDTree(onePoints)
        origTwokdTree = cKDTree(twoPoints)

        oneWithinInds = origOnekdTree.query_ball_point(twoPoints, stitch_width).tolist()
        twoWithinInds = origTwokdTree.query_ball_point(onePoints, stitch_width).tolist()

        oneWithinInds = list(itertools.chain.from_iterable(oneWithinInds))
        twoWithinInds = list(itertools.chain.from_iterable(twoWithinInds))


        oneStitchPoints = np.array(onePoints[np.array(oneWithinInds)])
        twoStitchPoints = np.array(twoPoints[np.array(twoWithinInds)])

        #Okay, great. Now merge the collection of points
        mergedPoints = np.copy(np.vstack((oneStitchPoints, twoStitchPoints)))
        np.random.shuffle(mergedPoints)

        #Now, take random NEIGHBORHOOD_SIZE patches from those

        #TODO: This is duplicated a lot. Spaghet, if I've ever seen it

        unvisitedIndices = set(range(mergedPoints.shape[0]))

        while (len(unvisitedIndices) > NEIGHBORHOOD_SIZE):
            #Pull out NEIGHBORHOOD_SIZE arbitrary unvisited indices
            inds = []
            for i in range(NEIGHBORHOOD_SIZE):
                ind = unvisitedIndices.pop()
                inds.append(ind)

            inds = np.array(inds)

            neighborhood = mergedPoints[inds]

            num_covers += 1

            addNeighborhoodWithWeight(neighborhood, stitch_covering_weight)

print("Number of seam cover neighborhoods: ", num_covers)
num_covers = 0

for i in range(NUM_RIGID_COVERINGS):
    for rigidRegionMask in rigidRegionMasks:
        regionPoints = np.copy(StandardBody.pointArray)
        regionInds = np.logical_not(BodyFilters.pixelSpaceBodyMask(rigidRegionMask, regionPoints))
        regionPoints = regionPoints[regionInds]

        pointArray = np.copy(regionPoints) / 10.0
        np.random.shuffle(pointArray)

        unvisitedIndices = set(range(pointArray.shape[0]))

        while (len(unvisitedIndices) > NEIGHBORHOOD_SIZE):
            #Pull out NEIGHBORHOOD_SIZE arbitrary unvisited indices
            inds = []
            for i in range(NEIGHBORHOOD_SIZE):
                ind = unvisitedIndices.pop()
                inds.append(ind)

            inds = np.array(inds)

            neighborhood = pointArray[inds]
            num_covers += 1
            addNeighborhoodWithWeight(neighborhood, rigid_covering_weight)

print("Number of rigid cover neighborhoods: ", num_covers)
num_covers = 0
    

    

dist_a_points = np.array(dist_a_points)
dist_b_points = np.array(dist_b_points)
dist_sq_dists = np.array(dist_sq_dists)
dist_weights = np.array(dist_weights)
   
def minSurrogate(X):
    min_squares = tf.reduce_min(X, axis=1)
    eps = 0.00001
    min_squares = tf.maximum(min_squares, eps)
    return tf.sqrt(min_squares)

#Given a list of numpy tensors with the same first dimension,
#Returns a triple of a Dataset, an iterator, and a lambda (pass in a Session)
#to run to initialize the iterator
def buildRandomOrderBatchedDataset(tensorList, batchSize):
    #Doing this so that we don't have to over-rely on the shuffle buffer
    N_elems = tensorList[0].shape[0]
    p = np.random.permutation(N_elems)

    feed_dict = {}
    placeholders = []
    for tensor in tensorList:
        placeholder = tf.placeholder(tf.float32, tensor.shape)
        placeholders.append(placeholder)
        permedTensor = tensor[p]
        feed_dict[placeholder] = permedTensor

    dataset = tf.data.Dataset.from_tensor_slices(tuple(placeholders))
    dataset = dataset.repeat()
    shuffled_dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    batched_dataset = shuffled_dataset.batch(batchSize, drop_remainder=True)
    #TODO: Should we prefetch to device here?
    result_dataset = batched_dataset.prefetch(2)
    iterator = result_dataset.make_initializable_iterator()
    result_lambda = lambda sess : sess.run(iterator.initializer, feed_dict=feed_dict)
    return (result_dataset, iterator, result_lambda)



#At the end of the day, this thing is responsible for writing some auto-generated labels
#to the AutoLabelWriter passed as a first argument here
#autoLabelWriter: what you think it is
#origtiemstampsList: 1d Long list of ACTUAL world timestamps (what's recorded in the file)
#timestamps: 1d float array of ficticious world timestamps (normalized for learning)
#points: list (indexed by logical timestamp) of lists (indexed by point index)
#of world spatial coordinates
#colors: list (indexed by logical timestamp) of lists (indexed by point index)
#of RGB colors
#partPoints: dictionary indexed by metric file name where each value is a [None, 4] array
#of space-time coordinates for each part
#initialRigidPointsList:
#markerPosList: a list of 4-vectors containing positions of markers
def temporalRegister(autoLabelWriter, origTimestampsList, timestampsList, pointsList, colorsList, partPointsDict, initialRigidPointsList,
                     markerPosList):
    #Before doing anything, build two big, long parallel arrays of 4d points and 3d colors
    fourDeePoints = []
    rigidPoints = []
    threeDeeColors = []
    for i in range(len(timestampsList)):
        timestamp = timestampsList[i]
        #TODO: use numpy hstack?
        for j in range(len(pointsList[i])):
            rigidPoint = initialRigidPointsList[i][j]
            point = pointsList[i][j]
            color = colorsList[i][j]
            rigidPoints.append(rigidPoint)
            fourDeePoints.append(np.array([point[0], point[1], point[2], timestamp], dtype=np.float32))
            threeDeeColors.append(np.array([color[0], color[1], color[2]], dtype=np.float32))
    fourDeePoints = np.array(fourDeePoints)
    threeDeeColors = np.array(threeDeeColors)
    rigidPoints = np.array(rigidPoints)

    timestampsArray = np.array(timestampsList)

    #Reset the tensorflow graph, in case it has something
    tf.reset_default_graph()

    #Define the data sources for everything below, of which there are many 

    #Data source for paired template points with desired distances
    _, dist_iterator, dist_iterator_init = buildRandomOrderBatchedDataset([dist_a_points, dist_b_points, dist_sq_dists, dist_weights], DIST_BATCH_SIZE)
    #Data source for marker correspondences (source points, targets)
    _, marker_iterator, marker_iterator_init = buildRandomOrderBatchedDataset([markerPosList], MARKER_BATCH_SIZE)
    #Data source for forward points
    _, point_iterator, point_iterator_init = buildRandomOrderBatchedDataset([fourDeePoints], BATCH_SIZE)

    #Data sources for each parts's input point samples
    part_iterators = {}
    part_iterator_inits = {}
    for partMetricName in partMetricNames:
        _, part_iterator, part_iterator_init = buildRandomOrderBatchedDataset([partPointsDict[partMetricName]], PER_COLOR_BATCH_SIZE)
        part_iterators[partMetricName] = part_iterator
        part_iterator_inits[partMetricName] = part_iterator_init

        
    #Great, now finally, build a dataset for reverse body samples
    #and reverse timestamps (to be hstacked together)
    _, reverse_body_iterator, reverse_body_iterator_init = buildRandomOrderBatchedDataset([bodyPointArray], REV_BATCH_SIZE)
    _, reverse_timestamp_iterator, reverse_timestamp_iterator_init = buildRandomOrderBatchedDataset([np.reshape(timestampsArray, (-1, 1))], REV_BATCH_SIZE)

    _, reconstruct_iterator, reconstruct_init = buildRandomOrderBatchedDataset([fourDeePoints, rigidPoints], RECONSTRUCT_BATCH_SIZE)
            

    #Define the basic structure of the network -- fundamentally, we have two components,
    #a forward component (which takes world space-time coordinates and transforms
    #them into template spatial coordinates)
    #and a reverse component (which takes template space-time coordinates
    #and transforms them into world coordinates)
    #Both components are identical in structure, but with different weights

    forward_eval_in = point_iterator.get_next()[0]
    reconstruct_forward_in, reconstruct_template_in = reconstruct_iterator.get_next()

    reverse_body_eval_in = reverse_body_iterator.get_next()[0]
    reverse_timestamp_eval_in = reverse_timestamp_iterator.get_next()[0]
    reverse_eval_in = tf.concat([reverse_body_eval_in, reverse_timestamp_eval_in], axis=1)

    forward_name_prefix = "Forward"
    reverse_name_prefix = "Reverse"

    forward_eval_out = approxNetwork(forward_eval_in, False, forward_name_prefix)
    reverse_eval_out = approxNetwork(reverse_eval_in, False, reverse_name_prefix)

    test_forward_eval_in = tf.placeholder(tf.float32, shape=[None, 4])
    test_forward_eval_out = approxNetwork(test_forward_eval_in, True, forward_name_prefix)

    reconstruct_forward_out = approxNetwork(reconstruct_forward_in, True, forward_name_prefix)

    test_reverse_eval_in = tf.placeholder(tf.float32, shape=[None, 4])
    test_reverse_eval_out = approxNetwork(test_reverse_eval_in, True, reverse_name_prefix)

    #Define the marker position loss
    marker_eval_in = marker_iterator.get_next()[0]

    marker_eval_out = approxNetwork(marker_eval_in, True, forward_name_prefix)

    template_marker_positions = tf.constant(PointTracking.getAllPointPositions() / 10.0)

    marker_dist_mat = sqDistanceMatrix(marker_eval_out, template_marker_positions)
    min_marker_dists = minSurrogate(marker_dist_mat)

    with tf.name_scope('marker_loss'): 
        marker_loss = tf.reduce_mean(min_marker_dists)

    #Define the body metric loss using distances between the forward-mapped
    #point batch and the randomized body points
    #This template_body_in will also be used for the initial rigid transform reconstruction
    #loss, an initialization step where we're just trying to make the network behave
    #like the best rigid transform to the template at each time step
    #TODO: BETTER ORGANIZATION TO SEPARATE RECONSTRUCT INITIAL STEP FROM
    #DEFORMABLE REGISTRATION

    template_body_in = tf.constant(bodyCover)

    with tf.name_scope('reconstruction_loss'):
        reconstruct_loss = tf.losses.absolute_difference(reconstruct_forward_out, reconstruct_template_in)

    body_dist_mat = sqDistanceMatrix(forward_eval_out, template_body_in)

    min_body_dists = minSurrogate(body_dist_mat)
    
    with tf.name_scope('manifold_loss'):
        manifold_loss = tf.reduce_mean(min_body_dists) 


    #Okay, great. Now, we need to define the "reprojection error", which
    #is the difference between applying the composite forward -> reverse
    #and the identity (for points it is evaluated at!)
    timestamps = forward_eval_in[:, 3]
    inflated_timestamps = tf.reshape(timestamps, [-1, 1])
    reproject_in = tf.concat([forward_eval_out, inflated_timestamps], 1)

    reproject_spatial_out = approxNetwork(reproject_in, True, reverse_name_prefix)
    reproject_out = tf.concat([reproject_spatial_out, inflated_timestamps], 1)


    with tf.name_scope('reproject_loss'):
        reproject_loss = tf.losses.absolute_difference(forward_eval_in, reproject_out)

    #Now do the same thing for reconstruction reprojection loss
    reconstruct_timestamps = reconstruct_forward_in[:, 3]
    inflated_reconstruct_timestamps = tf.reshape(reconstruct_timestamps, [-1, 1])
    reconstruct_reproject_in = tf.concat([reconstruct_forward_out, inflated_reconstruct_timestamps], 1)

    reconstruct_reproject_spatial_out = approxNetwork(reconstruct_reproject_in, True, reverse_name_prefix)
    reconstruct_reproject_out = tf.concat([reconstruct_reproject_spatial_out, inflated_reconstruct_timestamps], 1)


    with tf.name_scope('reconstruct_reproject_loss'):
        reconstruct_reproject_loss = tf.losses.absolute_difference(reconstruct_forward_in, reconstruct_reproject_out)


    #Now define the backwards reprojection error, which is the
    #difference between applying the composite backward -> forward
    #and the identity
    rev_timestamps = reverse_eval_in[:, 3]
    rev_inflated_timestamps = tf.reshape(rev_timestamps, [-1, 1])
    rev_reproject_in = tf.concat([reverse_eval_out, rev_inflated_timestamps], 1)

    rev_reproject_spatial_out = approxNetwork(rev_reproject_in, True, forward_name_prefix)
    rev_reproject_out = tf.concat([rev_reproject_spatial_out, rev_inflated_timestamps], 1)

    with tf.name_scope('rev_reproject_loss'):
        rev_reproject_loss = tf.losses.absolute_difference(reverse_eval_in, rev_reproject_out)

    #Now, during the reconstruction step, the loss is a weighted sum of the
    #reprojection loss, the reverse reprojection loss, and the reconstruction loss
    with tf.name_scope('total_reconstruct_loss'):
        total_reconstruct_loss = reconstruct_loss * reconstruct_weight + reconstruct_reproject_loss * reconstruct_reproject_weight

    #Fantastic. Now, define the isometricity loss term, which
    #rewards undistorted distances between pairs from the template manifold
    #in the reverse pass
    template_a_in, template_b_in, template_sq_dist_in, template_weight_in = dist_iterator.get_next()
    
    num_pairs = DIST_BATCH_SIZE 

    #Okay, great, but we need time stamps! Pull those as a sample from the earlier
    #timestamps variable
    timestamps_flat = tf.reshape(timestamps, [-1])
    shuffled_timestamps = tf.random_shuffle(timestamps_flat)
    shuffled_timestamps_trunc = shuffled_timestamps[0:num_pairs]
    timestamp_column = tf.reshape(shuffled_timestamps_trunc, [-1, 1])

    template_a_timestamped = tf.concat([template_a_in, timestamp_column], 1)
    template_b_timestamped = tf.concat([template_b_in, timestamp_column], 1)

    template_a_reversed = approxNetwork(template_a_timestamped, True, reverse_name_prefix)
    template_b_reversed = approxNetwork(template_b_timestamped, True, reverse_name_prefix)

    template_actual_sq_dist = tf.squared_difference(template_a_reversed, template_b_reversed)
    template_actual_sq_dist = tf.reduce_sum(template_actual_sq_dist, axis=1)

    div_sq_dist = template_actual_sq_dist / template_sq_dist_in

    #Approximation to log(div_sq_dist_mats)^2
    #loss_sq_dist_mats = tf.square(tf.square(1.0 - div_sq_dist_mats)) * 2.0
    #loss_sq_dist_mats = tf.abs(tf.log(tf.maximum(div_sq_dist_mats, 0.01)))
    #Nice approx to e^abs(ln(x))
    loss_sq_dist = tf.maximum(div_sq_dist, 1.0 / (0.6 * div_sq_dist + 0.4)) - 1.0

    #loss_sq_dist_mats = tf.log(tf.exp(div_sq_dist_mats) + tf.exp(1.2 / (div_sq_dist_mats + 0.2))) - 1.69314718056

    #Multiply losses by the weights.
    loss_sq_dist_weighted = loss_sq_dist * template_weight_in

    #Great, now we just care about a comparison with our pre-computed template square distance matrices
    with tf.name_scope('isometricity_loss'):
        #isometricity_loss = tf.losses.absolute_difference(reversed_patch_sq_dist_matrices, template_sq_dist_mats_in)
        #isometricity_loss = tf.losses.mean_squared_error(log_reversed_sq_dist_mats, log_template_sq_dist_mats)
        isometricity_loss = tf.reduce_mean(loss_sq_dist_weighted)

    #SET UP PART METRICS
    
    part_metric_inputs = {}
    part_template_inputs = {}
    part_metric_outputs = {}
    for metricName in partMetricNames:
        part_metric_input = part_iterators[metricName].get_next()[0]

        part_forward_eval_out = approxNetwork(part_metric_input, True, forward_name_prefix)

        part_template_input = tf.constant(templatePartCoverDict[metricName])

        part_dist_mat = sqDistanceMatrix(part_forward_eval_out, part_template_input)

        part_min_dists = minSurrogate(part_dist_mat)

        part_metric_outputs[metricName] = tf.reduce_mean(part_min_dists)
        part_metric_inputs[metricName] = part_metric_input
        part_template_inputs[metricName] = part_template_input

    #Define the part loss
    with tf.name_scope('part_loss'):
        part_loss = 0
        for metricName in partMetricNames:
            part_loss += part_metric_outputs[metricName]

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(init_training_rate, global_step, decay_steps=decay_steps, decay_rate=decay_rate)

    #Okay, great, only three more things to go! But for them, we'll need a session...

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        #Set up all of the datasets we defined earlier
        dist_iterator_init(sess)
        marker_iterator_init(sess)
        point_iterator_init(sess)

        for partMetricName in partMetricNames:
            part_iterator_inits[partMetricName](sess)

        reverse_body_iterator_init(sess)
        reverse_timestamp_iterator_init(sess)
        reconstruct_init(sess)


        #MANIFOLD AND COLOR DIFFERENCE LOSS

        #Great, now compose a loss term which expresses the sum of distances of forward evaluated points to the template manifold

        with tf.name_scope('total_loss'):
            total_loss = part_weight * part_loss + isometricity_weight * isometricity_loss + manifold_weight * manifold_loss + reproject_weight * reproject_loss + rev_reproject_weight * rev_reproject_loss + marker_weight * marker_loss

        #Define optimizer
        forward_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, forward_name_prefix)
        reverse_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, reverse_name_prefix)

        opt_vars = []
        for opt_var in forward_opt_vars:
            opt_vars.append(opt_var)
        for opt_var in reverse_opt_vars:
            opt_vars.append(opt_var)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=momentum_multiplier, 
                                                   epsilon=1e-8).minimize(total_loss, global_step, var_list=opt_vars)
        with tf.name_scope('reconstruct_optimizer'):
            reconstruct_train_step = tf.train.RMSPropOptimizer(reconstruct_training_rate, decay=0.0005, momentum=momentum_multiplier,
                                                   epsilon=1e-8).minimize(total_reconstruct_loss, var_list=opt_vars)

        #Okay, great. Now we need to initialize the variables associated with the optimizer
        adam_opt_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adam_optimizer")
        adam_opt_vars_init = tf.variables_initializer(adam_opt_vars_list)
        sess.run(adam_opt_vars_init)

        reconstruct_opt_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="reconstruct_optimizer")
        reconstruct_opt_vars_init = tf.variables_initializer(reconstruct_opt_vars_list)
        sess.run(reconstruct_opt_vars_init)


        forward_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Forward")
        forward_vars_init = tf.variables_initializer(forward_vars_list)
        sess.run(forward_vars_init)
        
        reverse_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Reverse")
        reverse_vars_init = tf.variables_initializer(reverse_vars_list)
        sess.run(reverse_vars_init)

        #First, run the best rigid alignment reconstruction step.
        batchNum = 0
        start = time.time()
        for i in range(reconstruct_training_iters):
            batchNum += 1

            #Great! Now run a training step

            options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            sess.run([reconstruct_train_step], options=options)

            #TODO: Visualization needed?

            if (i % VIEW_LOSS_AFTER == 0):
                train_loss = total_reconstruct_loss.eval()
                if (math.isnan(train_loss)):
                    #ABORT, and say that we failed!
                    return "Failed due to NaN in Reconstruct step. Retrying."
                loss_components = sess.run([reconstruct_loss, reproject_loss])
                print("Reconstruct Batches per second: ", batchNum / (time.time() - start))
                print("Step %d, training loss %g" % (i, train_loss))
                print("Loss components:")
                print("Reconstruct loss         %g (in)" % (((loss_components[0] / 7.0) * 10.0) / 25.4))
                print("Reprojection loss        %g (in)" % (((loss_components[1] / 7.0) * 10.0) / 25.4))

        batchNum = 0
        start = time.time()
        for i in range(training_iters):
            batchNum += 1

            #That oughtta do it! Run a training step!
            if (i == -1):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run([train_step], options=options, run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_01.json', 'w') as f:
                    f.write(chrome_trace)
            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                sess.run([train_step], options=options)

            if (i % VIEW_SNAPSHOT_AFTER == 0 and i != 0):
                #Display images representing the original
                #point cloud in orthographic projection
                #and the same cloud, but colored according
                #to where each point is identified in the template.
                #This picks a frame at random from the input
                ind = random.randrange(len(timestampsList))
                timestamp = timestampsList[ind]
                pointList = pointsList[ind]
                #Color metric was done in millimeters
                pointList = pointList
                origColorList = colorsList[ind]
                num_points = pointList.shape[0]
                aug_pointList = np.hstack((pointList, np.ones((num_points, 1)) * timestamp))
                forward_out = test_forward_eval_out.eval(feed_dict={test_forward_eval_in : aug_pointList})

                #Now get the actual closest colors from nearest-neighbor
                _, actualInds = bodyKdTree.query(forward_out)
                actualColors = np.array(coloredBody.colors)[actualInds, 0:3]
                
                actualClosestPoints = bodyPointArray[actualInds]


                coordinateColors = []
                #Finally, construct the false-color standard body map
                for i in range(actualClosestPoints.shape[0]):
                    point = actualClosestPoints[i] * 10.0
                    r, g, b, a = StandardBody.xyzToRGBA(point)
                    color = np.array([r, g, b]) 
                    coordinateColors.append(color)
                coordinateColors = np.array(coordinateColors)
                

                VizUtils.displayOrtho("Original", pointList, origColorList)
                #displayOrtho("TransformedColorApprox", pointList, transColorList)
                VizUtils.displayOrtho("TransformedActual", pointList, actualColors)
                VizUtils.displayOrtho("TransformedCoord", pointList, coordinateColors)



            if (i % VIEW_LOSS_AFTER == 0):
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                train_loss = total_loss.eval()
                loss_components = sess.run([part_loss, isometricity_loss, manifold_loss, reproject_loss, rev_reproject_loss, marker_loss], options=options)
                print("Batches per second: ", batchNum / (time.time() - start))
                print("Step %d, training loss %g" % (i, train_loss))
                print("Loss components:")
                print("Part loss         %g (in)" % (((loss_components[0] / 7.0) * 10.0) / 25.4))
                print("Isometricity loss %g" % loss_components[1])
                print("Manifold loss     %g (in)" % ((loss_components[2] * 10.0) / 25.4))
                #print("Color diff loss   %g" % math.sqrt(loss_components[3]))
                print("Reprojection loss %g (in)" % ((loss_components[3] * 10.0) / 25.4))
                print("Projection loss   %g (in)" % ((loss_components[4] * 10.0) / 25.4))
                print("Marker     loss   %g (in)" % ((loss_components[5] * 10.0) / 25.4))

        #Great. Training iterations finished here! Now time to write what we learned out
        #to a file
        for i in range(len(origTimestampsList)):
            origTimestamp = origTimestampsList[i]
            normalizedTimestamp = timestampsList[i]
            #Run the standard body array through the reverse transform, with timestamps
            #set to the given normalized timestamp
            templatePointsArray = np.copy(StandardBody.pointArray) / 10.0
            timestampColumn = np.ones((templatePointsArray.shape[0], 1), dtype=np.float32) * normalizedTimestamp
            templatePointsFourDee = np.hstack((templatePointsArray, timestampColumn))
            threeDeeOut = test_reverse_eval_out.eval(feed_dict={test_reverse_eval_in : templatePointsFourDee})
            threeDeeRescaled = threeDeeOut * 10.0
            #Add it to the label writer
            autoLabelWriter.add(origTimestamp, threeDeeRescaled)
        #Great, once we're done, write everything to the label file
        autoLabelWriter.writeToFile()
        #WE DID IT!
        return None
            


#Okay, great. Now we just gotta feed the beast.

#Given a cloud manager, return a tuple (timestampList, pointsList, colorsList,
#partPointsDict) in the same order as is expected by temporalRegister
#Print progress with respect to converting and filtering point clouds
def loadFromCloudManager(cloud_mgr):
    origTimestampsList = []
    timestampsList = []
    pointsList = []
    colorsList = []
    markerPosList = np.zeros((0, 4))
    templatePosList = np.zeros((0, 3))
    initialRigidPointsList = []
    partPointsDict = {}
    for partMetricName in partMetricNames:
        partPointsDict[partMetricName] = np.zeros((0, 4))

    #Ficticious timestamps (those that are fed to the fancy ML algorithms as floats)
    #are derived by subtracting the first timestamp (minus 1) and dividing by 1000
    #the "minus 1" is used to avoid screwy NaN results of zero activation on
    #the first run
    firstTimestamp = cloud_mgr.getTimestamp()
    
    prog = 0

    while True:
        prog += 1
        print("Loading frame ", prog)

        #REMOVE ME
        if (prog > MEGABATCH_SIZE):
            break

        cloud = cloud_mgr.getCloud()
        timestamp = cloud.getTimestamp() - (firstTimestamp - 1)
        timestamp = float(timestamp) * (1.0 / 10.0)

        #Ensure that the points are in metric (centimeters) coordinates!
        
        #If there are no points in the cloud, ignore doing all the stuff below
        #and just scrub another frame forward
        if (cloud.getPoints().size == 0):
            success_scrub = cloud_mgr.scrub(1)
            if (not success_scrub):
                break
            else:
                continue

        points = np.array(cloud.getPoints(), dtype=np.float32) / 10.0
        colors = np.array(cloud.getColors(), dtype=np.float32)
        colors = colors[:, 0:3]

        #Great. Add the timestamp to the timestamp list
        timestampsList.append(timestamp)
        origTimestampsList.append(cloud.getTimestamp())
        pointsList.append(points)
        colorsList.append(colors)

        #Great. Now we just need to filter out each of the colors here, too

        framePartPoints = {}
        framePartKdTrees = {}
        
        for partMetricName in partMetricNames:
            print("loading part" , partMetricName)
            partFilter = rgbColorFilterDict[partMetricName]
            #First, apply the color filter
            partMask = np.logical_not(ColorFilters.pixelSpaceVectorMask(partFilter, colors))
            partPoints = points[partMask]
            #Then apply a largest component filter
            componentMask = RGBPointCloud.largestComponentMask(partPoints, max_edge_dist=5.0)
            partPoints = partPoints[componentMask]

            framePartPoints[partMetricName] = partPoints
            if (partPoints.size != 0):
                framePartKdTrees[partMetricName] = cKDTree(np.copy(partPoints))
            else:
                framePartKdTrees[partMetricName] = cKDTree(np.array([[100000.0, 100000.0, 100000.0]], dtype=np.float32))

            timestampCol = np.ones((partPoints.shape[0], 1), dtype=np.float32) * timestamp
            partPointsFourDee = np.hstack((partPoints, timestampCol))
            #Add the points found there to the list entry in the part dictionary
            partPointsDict[partMetricName] = np.vstack((partPointsDict[partMetricName], partPointsFourDee))

        #Okay, great. Now, find all of the markers on the body, and the places that they
        #associate to on the template body
        cloudCopy = cloud.copy()
        PointTracking.filterCloudToPoints(cloudCopy)
        #Only do anything here if there are any markers at all!
        if (cloudCopy.getPoints().size != 0):
            markerPositions = np.array(cloudCopy.getPoints(), dtype=np.float32) / 10.0
            if (markerPositions.shape[0] > 0):
                timestampCol = np.ones((markerPositions.shape[0], 1), dtype=np.float32) * timestamp
                markerPositions = np.hstack((markerPositions, timestampCol))

                markerPosList = np.vstack((markerPosList, markerPositions))

        #Figure out the initial rigid alignment, and add it to the initial rigid points list
        initialRigidPoints = getRigidAlignment(framePartPoints, points)

        initialRigidPointsList.append(initialRigidPoints)
        
        success_scrub = cloud_mgr.scrub(1)
        if (not success_scrub):
            break
    return (origTimestampsList, timestampsList, pointsList, colorsList, partPointsDict, initialRigidPointsList, markerPosList)

#Main loop: Take in the sequence root, and find all sequences to annotate there.
sequenceRoot = sys.argv[1]
rootDirector = RootDirector(sequenceRoot)

for recordingName in rootDirector.getPeopleRecordingNames():
    recordingPath = os.path.join(rootDirector.getDirectoryPath(), recordingName)
    #Get a MultiCloudManager for that recording
    cloudManager = rootDirector.getMultiCloudManager(recordingName)
    #Build an AutoLabelManager for the recording
    labelManager = AutoLabelManager(recordingPath)
    #Now, we do the following until frames are exhausted
    while True:
        print("---------")
        print("Loading another MEGABATCH for sequence: ", recordingName)
        print("--------")
        #Make sure that the cloudManager is at the right place before we do anything
        moreFrames = labelManager.updateCloudManager(cloudManager)
        if (not moreFrames):
            break
        #Okay, great, get a bunch of junk from the cloud manager
        origTimestampsList, timestampsList, pointsList, colorsList, partPointsDict, initialRigidPointsList, markerPosList = loadFromCloudManager(cloudManager)

        if (len(origTimestampsList) == 0):
            print("Something fishy going on in this sequence -- no timestamps!")
            break
        #Using the original timestamp list, construct a label writer
        labelWriter = labelManager.getLabelWriter(origTimestampsList)
        #Invoke the god function, retrying if we get an error message,
        #and only continuing if the function returns None
        temporalRegisterReturn = "Starting temporal register"
        while (temporalRegisterReturn is not None):
            print(temporalRegisterReturn)
            temporalRegisterReturn = temporalRegister(labelWriter, origTimestampsList, timestampsList, pointsList, colorsList, partPointsDict, initialRigidPointsList, markerPosList)
