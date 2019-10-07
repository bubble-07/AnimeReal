import tensorflow as tf
import numpy as np


#Computes a projective transform as expected by tf.contrib.image.transform
#which represents the composite of multiplication of the x-axis by flip, followed by
#rotation by theta about a 480x480 image's center
#followed by scaling by factor gamma
#Note: unlike the documentation of tf.contrib.image.transform, the parameters
#passed to this function should be for the FORWARD direction (input to output) of the transformation
#This function will internally translate between those two representations to yield
#an affine transform matrix in the REVERSE direction
def get_affine_transform(flip, theta, scaling_fac, x_translate=0, y_translate=0, x_center=240, y_center=240):
    g = 1.0 / scaling_fac
    c = tf.cos(-theta)
    s = tf.sin(-theta)
    f = flip
    #See https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform for what these mean
    a_zero = g * c * f
    a_one = -g * s * f
    a_two = x_center - x_center * g * c * f + y_center * g * f * s + x_translate

    b_zero = g * s
    b_one = g * c
    b_two = y_center - x_center * g * s - y_center * g * c + y_translate

    return [a_zero, a_one, a_two, b_zero, b_one, b_two, 0, 0]

def get_aspect_ratio_transform(x_fac, y_fac, x_center=240, y_center=240):
    a_zero = x_fac
    a_one = 0.0
    a_two = x_center - x_center * x_fac

    b_zero = 0.0
    b_one = y_fac
    b_two = y_center - y_center * y_fac

    return [a_zero, a_one, a_two, b_zero, b_one, b_two, 0, 0]
