#Module used to approximate the nearest-neighbor color function
#for points close to the standard template body
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

num_middle_layers = 4
middle_layer_width = 100

train_step_size = 1e-3
ACTIV = tf.nn.elu

training_iters = 100000
#training_iters = 100
BATCH_SIZE = 10000

VIEW_AFTER = 100

def fcLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=ACTIV,
            weights_regularizer=None,
            reuse=reuse, scope=scope)

def fcLinLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=None, reuse=reuse, scope=scope)

def approxNetwork(x, reuse, namePrefix, output_dimension=3):
    with tf.variable_scope(namePrefix + "Affine") as s:
        out = fcLinLayer(x, 3, reuse, s)
    for i in range(num_middle_layers):
        with tf.variable_scope(namePrefix + "FC" + str(i)) as s:
            with tf.name_scope("FC" + str(i)):
                out = fcLayer(out, middle_layer_width, reuse, s)
    with tf.variable_scope(namePrefix + "CompressLinear") as s:
        lin_compress = fcLinLayer(out, output_dimension, reuse, s)
    with tf.variable_scope(namePrefix + "FinalLinear") as s:
        return fcLinLayer(lin_compress, output_dimension, reuse, s)

def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=False), :]

def approximate(namePrePrefix, pointList, colorList):
    colorList = np.array(colorList, dtype=np.float32)
    colorList = colorList[:, 0:3]
    kdTree = cKDTree(pointList)
    #We'll pick points by picking a random (spherical normal) offset
    #from randomly-chosen points in the given point list
    variance = 100000.0
    mean_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    covar_mat = np.array([[variance, 0, 0], [0, variance, 0], [0, 0, variance]], dtype=np.float32)

    #The name prefix for all variable scopes
    namePrefix = namePrePrefix + "Metric"


    in_points = tf.placeholder(tf.float32, [None, 3], name=(namePrefix + "In"))
    approx_color_network = approxNetwork(in_points, False, namePrefix=namePrefix)
    approx_color_out = tf.identity(approx_color_network, name=(namePrefix + "Out"))

    target_colors = tf.placeholder(tf.float32, [None, 3])

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(approx_color_out, target_colors)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(train_step_size).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    start = time.time()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        batchNum = 0
        start = time.time()

        for i in range(training_iters):
            #Pick a random collection of points from the input point list
            randomPoints = randomRows(pointList, BATCH_SIZE)
            #Compute normally-distributed offsets for them
            offsets = np.random.multivariate_normal(mean_vec, covar_mat, size=BATCH_SIZE)
            fuzzedPoints = randomPoints + offsets

            #Great, now for each fuzzed point, compute the nearest-neighbor colors in the point cloud
            _, actualInds = kdTree.query(fuzzedPoints)
            actualColors = colorList[actualInds]

            #Okay, now run a training step
            batchNum += 1

            sess.run([train_step], feed_dict={in_points : fuzzedPoints, target_colors : actualColors})

            if (i % VIEW_AFTER == 0):
                train_loss = loss.eval(feed_dict={in_points : fuzzedPoints, target_colors : actualColors})
                print("Batches per second: ", batchNum / (time.time() - start))
                print("Step %d, training loss %g" % (i, train_loss))
        saver.save(sess, "./" + namePrefix + "/" + namePrefix)

#Load the colored body template RGB Point cloud

coloredTemplateFile = "ColoredTemplate.pickle"

coloredBody = pickle.load(open(coloredTemplateFile, "rb"))

pointList = np.array(coloredBody.getPoints())

colorList = coloredBody.getColors()

partName = "ColorApprox"

approximate(partName, pointList, colorList)

