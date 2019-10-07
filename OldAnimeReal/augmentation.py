from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import random
import math
import mathutils as mu
import tensorflow as tf


import parts as partinfo

from params import *

#File for the data augmentation procedures

def extract_annotation_channels(annotation_tensor):
    swapped = tf.transpose(annotation_tensor, perm=[2, 0, 1])
    xs = swapped[0, :, :]
    ys = swapped[1, :, :]
    zs = swapped[2, :, :]
    cs = swapped[3, :, :]
    return (xs, ys, zs, cs)

def pack_annotation_channels(xs, ys, zs, cs):
    #Stack 'em all up again and return
    return tf.stack([xs, ys, zs, cs], axis=2)

#Flips all annotations about the x-axis, and interchanges the
#heatmaps accordingly
def flip_annotations(annotation_tensor):
    #Recall that the format of the annotation tensor is (num_bodies, num_parts, 4)
    #Transpose to the order (num_parts, 4, num_bodies)
    parts_first = tf.transpose(annotation_tensor, perm=[1, 2, 0])
    #Okay, now extract each of the slices
    slices = [None] * partinfo.num_parts
    for i in range(partinfo.num_parts):
        #Find out what the (4, num_bodies) annotations for the part were
        old_part_annos = parts_first[i, :, :]
        #Extract components of those and flip the value of x
        xs = -old_part_annos[0, :]
        ys = old_part_annos[1, :]
        zs = old_part_annos[2, :]
        cs = old_part_annos[3, :]
        #Pack them back together to get something of size (4, num_bodies)
        new_part_annos = tf.stack([xs, ys, zs, cs], axis=0)
        #Flipped annotation goes with the flipped part
        slices[partinfo.mirror_part[i]] = new_part_annos

    #Stitch it back together, still in the (num_parts, 4, num_bodies) format
    parts_flipped = tf.stack(slices)
    #Return in in the (num_bodies, num_parts, 4) format
    return tf.transpose(parts_flipped, perm=[2, 0, 1])
    
    

#Given a scale factor of the original image, 
#this adjusts the z-values in the annotation tensor appropriately
def scale_annotations(annotation_tensor, scalefac):
    xs, ys, zs, cs = extract_annotation_channels(annotation_tensor)

    #TODO: Incorporate influence of t, if it becomes important to.
    new_zs = zs * (1.0 / scalefac)

    return pack_annotation_channels(xs, ys, zs, cs)
    

def rotate_annotations(annotation_tensor, theta):
    xs, ys, zs, cs = extract_annotation_channels(annotation_tensor)

    #Compute the cosine and sine of theta
    cos = tf.cos(theta)
    sin = tf.sin(theta)

    #Now, apply the standard rotation transform
    new_xs = cos * xs - sin * ys
    new_ys = sin * xs + cos * ys

    return pack_annotation_channels(new_xs, new_ys, zs, cs)

#Computes a projective transform as expected by tf.contrib.image.transform
#which represents the composite of multiplication of the x-axis by flip, followed by
#rotation by theta about a 480x480 image's center
#followed by scaling by factor gamma
#Note: unlike the documentation of tf.contrib.image.transform, the parameters
#passed to this function should be for the FORWARD direction (input to output) of the transformation
#This function will internally translate between those two representations to yield
#an affine transform matrix in the REVERSE direction
def get_affine_transform(flip, theta, scaling_fac):
    g = 1.0 / scaling_fac
    c = tf.cos(-theta)
    s = tf.sin(-theta)
    f = flip
    #See https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform for what these mean
    a_zero = g * c * f
    a_one = -g * s * f
    a_two = 240 - 240 * g * c * f + 240 * g * f * s

    b_zero = g * s
    b_one = g * c
    b_two = 240 - 240 * g * s - 240 * g * c

    return [a_zero, a_one, a_two, b_zero, b_one, b_two, 0, 0]



#Given a 480x480 image, perform data augmentation to yield a new
#480x480 image and corresponding annotation tensor.
#Only a centrally-cropped 256x256 region will be ever seen by the network
def cache_to_augment(img_file, annotation_tensor):
    #TODO: Determine empirical distributions for scale and rotation parameters
    #on a mobile phone! That may help somewhat with learning the target concept!
    



    if augmentation_enabled:
        rotation_value = tf.random_uniform([], minval=-max_radian_tilt, maxval=max_radian_tilt)
        scale_value = tf.random_uniform([], minval=min_scale, maxval=max_scale)
        flip_boolean = tf.random_uniform([], minval=-1.0, maxval=1.0) > 0.0

        #Perform rotations/scales/flips on the annotation tensor
        flipped_annos = tf.cond(flip_boolean, lambda: flip_annotations(annotation_tensor), lambda: annotation_tensor)
        scaled_annos = scale_annotations(flipped_annos, scale_value)
        rotated_annos = rotate_annotations(scaled_annos, rotation_value)

        #Flip as a multiplier (-1 for flip, 1 for identity)
        flip_mult = tf.cond(flip_boolean, lambda: -1.0, lambda: 1.0)

        affine_xform = get_affine_transform(flip_mult, rotation_value, scale_value)

        #Perform rotations/scales/flips on the image tensors
        xformed_img = tf.contrib.image.transform(img_file, affine_xform, interpolation='BILINEAR')

        #Great, now apply a random brightness shift on the images
        bright_adjusted_img = tf.image.random_brightness(xformed_img, max_brightness_shift)

        #Apply a random contrast shift
        contrast_adjusted_img = tf.image.random_contrast(bright_adjusted_img, min_contrast_shift, max_contrast_shift)

        return (xformed_img, rotated_annos)
    else:
        return (img_file, annotation_tensor)


