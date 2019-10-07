from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import os
import numpy as np

import sys
import random
import math

#from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('WX')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import scipy as sp
from scipy.stats import norm
import tensorflow as tf

#Network parameters
num_middle_layers = 4
middle_layer_width = 20

#how much to prefer isometric transforms
#iso_reg = 1000.0
iso_reg = 0
iso_reg = 1000.0
#iso_reg = 10.0

#How much to care about landmarks
landmark_strength = 100.0
landmark_strength = 1000.0

fitting_totality = 0.1
fitting_totality = 0.05
#fitting_totality = 0.0

fitting_accuracy = 1.0

#VISUALIZATION PARAMS
#How many training steps to view after
VIEW_AFTER = 100



#TRAINING PAREMETERS

num_reconstruct_training_iters = 6000
reconstruct_train_step_size = 1e-2

num_training_iters = 2000
batch_size = 100

l1reg = 1.0
l2reg = 1.0

train_step_size = 1e-4

DIMENSION=3

def visualize(iteration, error, X, Y, Xlandmarks, Ylandmarks, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:,0] ,  Y[:,1], Y[:, 2], color='blue', label='Source')
    #ax.scatter(Xlandmarks[:,0] ,  Xlandmarks[:,1], color='green', label='Source Landmarks')
    #ax.scatter(Ylandmarks[:,0] ,  Ylandmarks[:,1], color='yellow', label='Target Landmarks')
    #plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='large')
    plt.draw()
    plt.pause(0.001)
	
ACTIV=tf.nn.elu

def fcLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=ACTIV,
            weights_regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=l1reg, scale_l2=l2reg),
            reuse=reuse, scope=scope)

def fcLinLayer(inputs, num_outputs, reuse, scope):
    return tf.contrib.layers.fully_connected(inputs, num_outputs, 
            activation_fn=None, reuse=reuse, scope=scope)

#Network which encodes a simple affine transform
def affineNetwork(x, reuse, output_dimension=DIMENSION):
    with tf.variable_scope("AffineNetwork") as s:
        return fcLinLayer(x, output_dimension, reuse, s)

#Neural network for deformable registration
def regNetwork(x, reuse, output_dimension=DIMENSION):
    with tf.variable_scope("Affine") as s:
        out = fcLinLayer(x, DIMENSION, reuse, s)
    for i in range(num_middle_layers):
        with tf.variable_scope("FC" + str(i)) as s:
            with tf.name_scope("FC" + str(i)):
                out = fcLayer(out, middle_layer_width, reuse, s)
    with tf.variable_scope("CompressLinear") as s:
        lin_compress = fcLinLayer(out, output_dimension, reuse, s)
    with tf.variable_scope("FinalLinear") as s:
        return fcLinLayer(lin_compress + x, output_dimension, reuse, s)

#Given an m1xdim and a m2xdim tensor,
#compute the m1xm2 tensor of pairwise euclidean distances
def sqDistanceMatrix(A, B):
    p1 = tf.matmul(
	tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1),
	tf.ones(shape=(1, tf.shape(A)[0]))
    )
    p2 = tf.transpose(tf.matmul(
	tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]),
	tf.ones(shape=(tf.shape(B)[0], 1)),
	transpose_b=True
    ))

    return tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)

def minSurrogate(X):
    K = 3
    eps = 1.0
    bottomK, inds = tf.nn.top_k(-X, k=K)
    bottomK = bottomK * -1.0
    #bottomK = bottomK * (-1.0 / float(K))
    bottomK = tf.sqrt(bottomK + eps)
    bottomK = bottomK / float(K)
    return tf.square(tf.reduce_sum(bottomK, axis=0))


def landmarkError(moved_landmarks, target_landmarks):
    return tf.losses.mean_squared_error(moved_landmarks, target_landmarks)


#Compute how non-isometric the transform is
def non_isometricity(orig, transformed, target):
    origSqDist = sqDistanceMatrix(orig, orig)
    sqDist = sqDistanceMatrix(transformed, transformed)
    #Correct for zero self-distances by adding identity to boh
    origSqDist += tf.eye(tf.shape(orig)[0])
    sqDist += tf.eye(tf.shape(transformed)[0])

    eps = .05

    origLogDists = tf.log(origSqDist + eps)
    logDists = tf.log(sqDist + eps)
    result = tf.losses.mean_squared_error(origLogDists, logDists)
    return result

def sqDiffLoss(transformed, target):
    return tf.losses.mean_squared_error(transformed, target)

#Given the original manifold, the manifold transformed by the network
#and the target manifold, compute dissimilarity
#TODO: Add landmarks!
def lossFunc(orig, transformed, target, moved_landmarks, target_landmarks):
    #Compute loss as the sum of squares of minimum distances
    #between the transformed and target
    sq_dist_mat = sqDistanceMatrix(transformed, target)
    #Min distances to the moving points
    min_sq_dists_transpose = minSurrogate(sq_dist_mat)
    #Min distances to the fixed points
    min_sq_dists = minSurrogate(tf.transpose(sq_dist_mat))

    avg_min_sq_dist_transpose = tf.reduce_sum(min_sq_dists_transpose) / tf.cast(tf.shape(orig)[0], tf.float32)
    avg_min_sq_dist = tf.reduce_sum(min_sq_dists) / tf.cast(tf.shape(target)[0], tf.float32)

    #min_dists = tf.reduce_min(dist_mat, axis=0)
    #min_dists = approxReduceMin(dist_mat)
    #return tf.reduce_sum(tf.square(min_dists))
    return (fitting_accuracy * avg_min_sq_dist + 
            fitting_totality * avg_min_sq_dist_transpose +
            iso_reg * non_isometricity(orig, transformed, target) +
            landmark_strength * landmarkError(moved_landmarks, target_landmarks))

def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=False), :]

#Given the dataset to transform, an initial affine transformed version (could just
#be X), a target set of points, and collections of landmarks on each, do the
#registration
def register(X, Y, X_affine, Xlandmarks, Ylandmarks):
    tf.reset_default_graph()
    fig = plt.figure()
    #fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(111, projection='3d')

    SAMP_POINTS = 300


    in_moving_manifold = tf.placeholder(tf.float32, [None, DIMENSION])
    out_moving_manifold = regNetwork(in_moving_manifold, False)
    target_manifold = tf.placeholder(tf.float32, [None, DIMENSION])

    moving_landmarks = tf.placeholder(tf.float32, [None, DIMENSION])
    moved_landmarks = regNetwork(moving_landmarks, True)
    target_landmarks = tf.placeholder(tf.float32, [None, DIMENSION])

    with tf.name_scope('affinereconstructloss'):
        reconstruct_loss = sqDiffLoss(out_moving_manifold, target_manifold)

    check = tf.add_check_numerics_ops()
    with tf.name_scope('loss'):
        loss = lossFunc(in_moving_manifold, out_moving_manifold, target_manifold,
                        moved_landmarks, target_landmarks)
    with tf.name_scope('reconstruct_adam_optimizer'):
        reconstruct_train_step = tf.train.AdamOptimizer(reconstruct_train_step_size).minimize(reconstruct_loss)

    with tf.name_scope('deformable_adam_optimizer'):
        train_step = tf.train.AdamOptimizer(train_step_size).minimize(loss)

    start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        SAMP_POINTS=500

        #Then, initialize the deformable reg network to match the predictions
        #of the best affine registration
        batchNum = 0
        start = time.time()
        for i in range(num_reconstruct_training_iters):
            rowsPicked = np.random.choice(X.shape[0], SAMP_POINTS, replace=False)
            net_input = X[rowsPicked, :]
            net_target = X_affine[rowsPicked, :]
            net_input_landmarks = Xlandmarks
            net_target_landmarks = Ylandmarks
            batchNum += 1
            print("Reconstruct batches per second: ", batchNum / (time.time() - start))

            if i % VIEW_AFTER == 0:
                train_loss = reconstruct_loss.eval(feed_dict={in_moving_manifold : net_input, target_manifold : net_target })
                print("Step %d, training loss %g" % (i, train_loss))
                current_moving = out_moving_manifold.eval(feed_dict={in_moving_manifold : net_input, target_manifold : net_target})
                visualize(batchNum, train_loss, net_target, current_moving, np.array([[0,0]]), np.array([[0,0]]),
                          ax)
            sess.run([check, reconstruct_train_step], feed_dict={in_moving_manifold : net_input,
                target_manifold : net_target,
                moving_landmarks : net_input_landmarks,
                target_landmarks : net_target_landmarks})

        SAMP_POINTS=1000

        #Now, using the initialized deformable reg network,
        #fit the best deformable reg
        batchNum = 0
        start = time.time()
        for i in range(num_training_iters):
            net_input = randomRows(X, SAMP_POINTS)
            net_target = randomRows(Y, SAMP_POINTS)
            net_input_landmarks = Xlandmarks
            net_target_landmarks = Ylandmarks
            batchNum += 1
            print("Deformable batches per second: ", batchNum / (time.time() - start))
            
            if i % VIEW_AFTER == 0:
                train_loss = loss.eval(feed_dict={in_moving_manifold : net_input,
                    target_manifold : net_target,
                    moving_landmarks : net_input_landmarks,
                    target_landmarks : net_target_landmarks})
                print("Step %d, training loss %g" % (i, train_loss))
                current_moving = out_moving_manifold.eval(feed_dict=
                                {in_moving_manifold : net_input})
                current_moved_landmarks = moved_landmarks.eval(feed_dict= 
                        {moving_landmarks : net_input_landmarks})
                visualize(batchNum, train_loss, current_moving, net_target, 
                            current_moved_landmarks, net_target_landmarks, ax)
            sess.run([check, train_step], feed_dict={in_moving_manifold : net_input, 
                target_manifold : net_target,
                moving_landmarks : net_input_landmarks,
                target_landmarks : net_target_landmarks})
        #Once all is done, return the entire input, transformed by the warp
        plt.close(fig)
        return out_moving_manifold.eval(feed_dict={in_moving_manifold : X})




#def main(_):
#    register(X, Y, Xlandmarks, Ylandmarks)

#if __name__ == '__main__':
#    tf.app.run(main=main, argv=[sys.argv[0]])
