from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

#Helper function to compute bell curves of a given height and spread
#These are just vertically-scaled versions of the corresponding normal distribution
def bellCurve(x, mu, sigma, height):
    return tf.exp(-(((x - mu) ** 2) / (2 * (sigma ** 2)))) * height

def avg(x, y):
    return (x + y) / 2.0

#Mean squared error loss
def mse_loss(x, y):
    return tf.losses.mean_squared_error(x, y)

 
#Mean absolute error loss
def mabs_loss(x, y):
    return tf.losses.mean_squared_error(y, x, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
