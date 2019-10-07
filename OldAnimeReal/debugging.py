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
import scipy as sp

import parts as partinfo

from params import *


#This module contains routines for visually
#debugging various stages in the training/testing pipeline

#Given tensors for the actual 64x64 field map output of the network
#and the expected 64x64 field map output of the network as NUMPY tensors (must come from tf.eval)
#this visually displays matched pairs for each heatmap with the actual and expected
#as different color channels of the same image
#The network's output is scaled so that its max value achieves the maximum intensity
def actual_expected_visualization(set_name, net_input, actual, expected):
    #Extract the first example from each. Double-index here,
    #because the actual from the neural net contains downsampled copies
    actual_sample = actual[0][0]
    expected_sample = expected[0]

    #Display one image from the input batch and its output from the net
    sample_heatmaps = np.transpose(actual_sample, axes=[2, 0, 1])
    sample_desired_maps = np.transpose(expected_sample, axes=[2, 0, 1])

    C = []
    for i in range(partinfo.num_field_maps):
        max_heat = np.max(sample_heatmaps[i])
        print("Max Heat for " + str(i), max_heat)

        desired_heat = sample_desired_maps[i] / MAX_INTENSITY

        net_heat = sample_heatmaps[i] / max_heat
        net_heat_upscaled = sp.ndimage.zoom(net_heat, 4.0)
        zeros = np.zeros_like(desired_heat)

        comparison = np.stack((net_heat_upscaled, zeros, desired_heat), axis=2)
        C.append(comparison)

    #Now, with all those comparisons, stitch together one big image outta all of them
    image = np.vstack((np.hstack((C[0], C[1], C[2], C[3], C[4])),
                       np.hstack((C[5], C[6], C[7], C[8], C[9])),
                       np.hstack((C[10], C[11], C[12], C[13], C[14]))))

    cv2.namedWindow(set_name + " Comparison")
    cv2.imshow(set_name + " Comparison", image)

    cv2.namedWindow(set_name + " Net Input")
    cv2.imshow(set_name + " Net Input", net_input[0] / 256.0)
    cv2.waitKey(20)


