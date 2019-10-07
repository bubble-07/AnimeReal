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

import parts as partinfo

from params import *

#This file is responsible for heatmap generation from augmented cache image,
#keypoint tensor pairs. To do that, we actually call out to a tensorflow
#custom op implemented in the custom_ops directory called heatmap_gen.so

heatmap_gen_module = tf.load_op_library('./custom_ops/heatmap_gen.so')

#Centrally crop a source image to 256x256
def central_crop_image(img_file):
    return tf.image.crop_to_bounding_box(img_file, 112, 112, 256, 256)


#Given a (just-augmented) 480x480x3 image, generate heatmaps and PAFs from
#the annotation tensor, and return (source image, heatmaps/PAFs)
def augment_to_labeled_example(img_file, annotation_tensor):
    return tf.cond(tf.size(annotation_tensor) > 0, lambda: augment_to_labeled_example_helper(img_file, annotation_tensor),
                   lambda: (central_crop_image(img_file), tf.zeros([256, 256, partinfo.num_parts])))

#NOTE: This version is technically unsafe, because it doesn't handle the case
#where the annotation_tensor has zero bodies in it (first dimension zero).
#This is handled above
def augment_to_labeled_example_helper(img_file, annotation_tensor):  

    s, t = scale_factors
    part_scales = part_scaling_factors

    #Should be of dimensions (256, 256, num_parts)
    #To compute it, call out to the heatmap generation module
    heatmaps = heatmap_gen_module.heatmap_gen(annotation_tensor, part_scales, s, t)

    tf.assert_rank(heatmaps, 3)
    tf.assert_equal(tf.shape(heatmaps)[2], partinfo.num_parts, message='heatmaps')

    cropped_img = central_crop_image(img_file)

    return (cropped_img, heatmaps)
    

