#Module used to approximate the minimum-distance function to a given point cloud
#using a neural network trained with tensorflow

import time
import os
import numpy as np
import sys
import random
import math
import ColorFilters
import pickle

import StandardBody

import scipy as sp
from scipy.spatial import cKDTree
from scipy.stats import norm
import tensorflow as tf

num_middle_layers = 6
middle_layer_width = 200

train_step_size = 1e-3
#ACTIV = tf.nn.elu
ACTIV = tf.nn.leaky_relu
#With this chance, use a collection of points
#which have distance zero as the training batch
zero_hammer_prob = 0.05

training_iters = 100000
#training_iters = 100
BATCH_SIZE = 20000

VIEW_AFTER = 100


def fcLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=ACTIV,
            weights_regularizer=None,
            reuse=reuse, scope=scope)

def fcLinLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=None, reuse=reuse, scope=scope)

def approxNetwork(x, reuse, namePrefix, output_dimension=1):
    out = x
    for i in range(num_middle_layers):
        with tf.variable_scope(namePrefix + "FC" + str(i)) as s:
            with tf.name_scope("FC" + str(i)):
                out = fcLayer(out, middle_layer_width, reuse, s)
    with tf.variable_scope(namePrefix + "CompressLinear") as s:
        lin_compress = fcLinLayer(out, output_dimension, reuse, s)
    return lin_compress

def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=True), :]

def approximate(namePrePrefix, pointSubset, pointList):
    kdTree = cKDTree(pointSubset)
    #We'll pick points by picking a random (spherical normal) offset
    #from randomly-chosen points in the given point list
    variance = 100000.0
    mean_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    covar_mat = np.array([[variance, 0, 0], [0, variance, 0], [0, 0, variance]], dtype=np.float32)

    #The name prefix for all variable scopes
    namePrefix = namePrePrefix + "Metric"


    in_points = tf.placeholder(tf.float32, [None, 3], name=(namePrefix + "In"))

    small_points = in_points * 0.001
    
    crossterm_one = small_points[:, 0] * small_points[:, 1]
    crossterm_two = small_points[:, 1] * small_points[:, 2]
    crossterm_three = small_points[:, 0] * small_points[:, 2]
    crossterms = tf.stack([crossterm_one, crossterm_two, crossterm_three], axis=1)


    poly_aug_in_points = tf.concat([small_points, tf.square(small_points), crossterms], axis=1)
    approx_norm_network = approxNetwork(poly_aug_in_points, False, namePrefix=namePrefix)
    approx_norm_out = tf.identity(approx_norm_network, name=(namePrefix + "Out"))

    target_norms = tf.placeholder(tf.float32, [None, 1])

    with tf.name_scope('loss'):
        loss = tf.losses.absolute_difference(approx_norm_out, tf.square(target_norms * .001))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.GradientDescentOptimizer(train_step_size).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    check = tf.add_check_numerics_ops()

    start = time.time()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        batchNum = 0
        start = time.time()
        num_exact = int(float(BATCH_SIZE) * zero_hammer_prob)
        num_fuzzed = BATCH_SIZE - num_exact

        for i in range(training_iters):
            #Pick a random collection of points on the target manifold
            exactPoints = randomRows(pointSubset, num_exact)
            #Pick a random collection of points from the input point list
            fuzzedPoints = randomRows(pointList, num_fuzzed)
            #Compute normally-distributed offsets for them
            offsets = np.random.multivariate_normal(mean_vec, covar_mat, size=num_fuzzed)
            fuzzedPoints = fuzzedPoints + offsets

            allPoints = np.vstack((exactPoints, fuzzedPoints))
           
            #Great, now for each fuzzed point, compute the actual distances to the original point cloud
            actualDistances, _ = kdTree.query(allPoints)
            actualDistances = np.reshape(actualDistances, (BATCH_SIZE, 1))

            #Okay, now run a training step
            batchNum += 1

            sess.run([train_step, check], feed_dict={in_points : allPoints, target_norms : actualDistances})

            if (i % VIEW_AFTER == 0):
                train_loss = loss.eval(feed_dict={in_points : allPoints, target_norms : actualDistances})
                print("Batches per second: ", batchNum / (time.time() - start))
                train_loss = math.sqrt(train_loss) * 1000.0
                print("Step %d, training loss %g mm" % (i, train_loss))
        saver.save(sess, "./" + namePrefix + "/" + namePrefix)

#Dictionary from names to color filters
colorFilterDict = {"GreenLeg" : ColorFilters.maskGreenLeg,
                   "YellowArm" : ColorFilters.maskYellowArm,
                   "RedArm" : ColorFilters.maskRedArm,
                   "RedHand" : ColorFilters.maskRedHand,
                   "YellowHand" : ColorFilters.maskYellowHand,
                   "WhiteLeg" : ColorFilters.maskWhiteLegInTemplate,
                   "Torso" : ColorFilters.maskTorso}
partName = sys.argv[1]
colorFilter = colorFilterDict[partName]

#Load the colored body template RGB Point cloud

coloredTemplateFile = "ColoredTemplate.pickle"

coloredBody = pickle.load(open(coloredTemplateFile, "rb"))
coloredBody.indices = np.zeros((np.asarray(coloredBody.points).shape[0], 2))

#From the colored body, apply the color filter and a statistical filter
coloredBody.applyColorFilter(colorFilter, negated=True)
coloredBody.applyLargestComponentFilter()

pointSubset = np.asarray(coloredBody.getPoints())

approximate(partName, pointSubset, StandardBody.pointArray)

