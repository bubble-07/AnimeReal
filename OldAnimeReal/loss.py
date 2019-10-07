from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import cv2
import numpy as np
import random
import math
import mathutils as mu
import lambda_utils as lu
import neural_ops as nn


import parts as partinfo

from params import *

loss_fn = mu.mse_loss

#Module for loss function definitions

#TODO: Move these two functions to somewhere more descriptive

#Given a tensor for images of power-of-two width and height,
#this returns a list of N tensors where each subsequent image
#tensor is a 2x2 max-pooled version of the previous one,
#and the first tensor is the original image tensor
#TODO: Convert the following two functions to use the neural_op framework
#and then move them there!
def iterated_max_pool_downsample(data, N):
    return lu.iterate(lambda x: tf.contrib.layers.max_pool2d(x, 2, padding='SAME'), data, N - 1)
    
#Similar to iterated_max_pool_downsample, but with average pooling
def iterated_avg_pool_downsample(data, N):
    return lu.iterate(lambda x : tf.contrib.layers.avg_pool2d(x, 2, padding='SAME'), data, N - 1)

#Yields the downsampling loss weighting factors
#in the form [weight_8, weight_16, weight_32, weight_64]
#TODO: Hyperparameter optimization?
#TODO: Does it make sense to modify these while training?
def downsample_weighting_factors():
    return [8, 4, 2, 1]


#TODO: Add the PAFs back in? If you do, use average-pooling
#for those components!
 
#Downsampling loss function -- this takes the mean absolute errors
#between the network's 8x8, 16x16, 32x32 and 64x64 outputs
#and iteratively max-pooled versions of the "ground truth" 256x256
#heatmaps
def downsampling_mabs_loss(net_out, expected):
    #Iteratively max-pool the expected
    expected_out = iterated_max_pool_downsample(expected, 7)[2:-1]

    #Compute pairwise mabs losses
    losses = map(lambda pair : loss_fn(*pair), zip(net_out, expected_out))

    #Multiply by downsampling weighting factors and sum
    weighted_losses = map(lambda pair : tf.multiply(*pair), zip(losses, downsample_weighting_factors()))
    return reduce(tf.add, weighted_losses, 0.0)


