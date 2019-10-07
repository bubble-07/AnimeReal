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
import StandardBody

#from params import *

def disp_actual_expected_img_mask(set_name, namePrefix, in_img, in_actual):

    random_ind = random.randrange(in_actual[0].shape[0])
    actual_mask = in_actual[0][random_ind, :, :, 3]
    actual_template = in_actual[0][random_ind, :, :, 0:3]
    #Also display the openimages image, the output mask, and the output coordinates
    img_sample = in_img[random_ind]

    img_colored = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(set_name + " " + namePrefix + "Image")
    cv2.imshow(set_name + " " + namePrefix + "Image", img_colored)
    cv2.waitKey(20)

    actual_mask = np.clip(actual_mask, 0.0, 1.0)
    actual_mask = cv2.resize(actual_mask, (256, 256))

    cv2.namedWindow(set_name + " " + namePrefix + "Mask")
    cv2.imshow(set_name + " " + namePrefix + "Mask", actual_mask)
    cv2.waitKey(20)

    flat_actual_sample = np.reshape(actual_template, [-1, 3]) * 1000.0
    _, actualInds = StandardBody.standardKdTree.query(flat_actual_sample)
    actual_colored_flat = StandardBody.standardColors[actualInds]
    actual_colored = np.reshape(actual_colored_flat[:, 0:3], [64, 64, 3])
    actual_colored = actual_colored.astype(np.uint8)
    actual_colored = cv2.resize(actual_colored, (256, 256))

    actual_colored = cv2.cvtColor(actual_colored, cv2.COLOR_BGR2RGB)

    cv2.namedWindow(set_name + " " + namePrefix + "Template")
    cv2.imshow(set_name + " " + namePrefix + "Template", actual_colored)
    cv2.waitKey(20)
    return random_ind

def visualize_all_coco_training_positions(set_name, coco_positions):
    for i in range(coco_positions.shape[0]):
        visualize_coco_training_position(set_name + str(i), coco_positions, i)
    cv2.waitKey(8000)
    
def visualize_coco_training_position(set_name, coco_positions, coco_ind, delay=20):
    coco_position_samp = coco_positions[coco_ind] * 1000.0
    flat_actual_sample = np.reshape(coco_position_samp, [-1, 3])
    _, actualInds = StandardBody.standardKdTree.query(flat_actual_sample)
    actual_colored_flat = StandardBody.standardColors[actualInds]
    actual_colored = np.reshape(actual_colored_flat[:, 0:3], [256, 256, 3])
    actual_colored = actual_colored.astype(np.uint8)
    actual_colored = cv2.resize(actual_colored, (256, 256))

    actual_colored = cv2.cvtColor(actual_colored, cv2.COLOR_BGR2RGB)

    cv2.namedWindow(set_name + " CocoKeypoints")
    cv2.imshow(set_name + " CocoKeypoints", actual_colored)
    cv2.waitKey(delay)
   

def rgb_trainer_visualization_without_openimages(set_name, img, actual, expected, coco_img, coco_actual, coco_positions):

    actual_expected_dense_rgb_visualization(set_name, img, actual, expected)


    coco_ind = disp_actual_expected_img_mask(set_name, "Coco", coco_img, coco_actual)

    visualize_coco_training_position(set_name, coco_positions, coco_ind)

def rgb_trainer_visualization(set_name, img, actual, expected, openimages_img, openimages_actual, coco_img, coco_actual, coco_positions):

    disp_actual_expected_img_mask(set_name, "Open", openimages_img, openimages_actual)
    rgb_trainer_visualization_without_openimages(set_name, img, actual, expected, coco_img, coco_actual, coco_positions)
  

#Visualization of the original image, the actual returned 64x64 heatmap, and the expected heatmap
def actual_expected_dense_rgb_visualization(set_name, img, actual, expected):
    random_ind = random.randrange(actual[0].shape[0])

    actual_mask = actual[0][random_ind, :, :, 3]
    actual_sample = actual[0][random_ind, :, :, 0:3]
    expected_sample = expected[random_ind]
    img_sample = img[random_ind]

    actual_mask = np.clip(actual_mask, 0.0, 1.0)
    actual_mask = cv2.resize(actual_mask, (512, 512))

    actual_sample = cv2.resize(actual_sample, (512, 512))
    img_sample = sp.misc.imresize(img_sample, (512, 512))
    actual_expected_dense_visualization_helper(set_name, actual_sample, expected_sample) 

    img_colored = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(set_name + " Image")
    cv2.imshow(set_name + " Image", img_colored)
    cv2.waitKey(20)

    cv2.namedWindow(set_name + " Mask")
    cv2.imshow(set_name + " Mask", actual_mask)
    cv2.waitKey(20)

def actual_expected_dense_visualization(set_name, actual, expected):
    #Extract the first example from the batch
    actual_sample = actual[0][0]
    expected_sample = expected[0]
    return actual_expected_dense_visualization_helper(set_name, actual_sample, expected_sample)

def actual_expected_dense_visualization_helper(set_name, actual_sample, expected_sample):



    flat_actual_sample = np.reshape(actual_sample, [-1, 3]) * 1000.0
    flat_expected_sample = np.reshape(expected_sample, [-1, 3]) * 1000.0

    #Great, now from the expected sample, find all indices where the value
    #stored is identically zero, and make a mask out of it
    eps = 0.0001
    flat_expected_background_mask = np.linalg.norm(flat_expected_sample, axis=1) < eps

    #Make that into a mask of size (512*512) by padding the top and bottom
    padding_len = int(((512 * 512) - (424 * 512)) / 2)
    mask_len = int(424 * 512)

    flat_actual_background_mask = np.ones((flat_actual_sample.shape[0],)).astype(np.bool_) 
    flat_actual_background_mask[padding_len:(padding_len + mask_len)] = flat_expected_background_mask

    #Take the actual and expected, and turn them into false-color images based on the standard body
    _, actualInds = StandardBody.standardKdTree.query(flat_actual_sample)
    _, expectedInds = StandardBody.standardKdTree.query(flat_expected_sample)

    actual_colored_flat = StandardBody.standardColors[actualInds]
    expected_colored_flat = StandardBody.standardColors[expectedInds]

    #Take the mask for non-body, and apply it to expected, yielding zeroes there
    expected_colored_flat[flat_expected_background_mask, :] = 0

    #Do the same for actual
    actual_colored_flat[flat_actual_background_mask, :] = 0

    actual_colored = np.reshape(actual_colored_flat[:, 0:3], [512, 512, 3])
    expected_colored = np.reshape(expected_colored_flat[:, 0:3], [424, 512, 3])

    actual_colored = actual_colored.astype(np.uint8)
    expected_colored = expected_colored.astype(np.uint8)

    actual_colored = cv2.cvtColor(actual_colored, cv2.COLOR_BGR2RGB)
    expected_colored = cv2.cvtColor(expected_colored, cv2.COLOR_BGR2RGB)

    cv2.namedWindow(set_name + " Actual")
    cv2.imshow(set_name + " Actual", actual_colored)
    cv2.waitKey(20)

    cv2.namedWindow(set_name + " Expected")
    cv2.imshow(set_name + " Expected", expected_colored)
    cv2.waitKey(20)



#This module contains routines for visually
#debugging various stages in the training/testing pipeline

'''
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
'''
#TODO: Rehabilitate


