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
from OpenImagesCommon import *


from DepthTrainerParams import *

loss_fn = mu.mse_loss

mask_loss_fn = mu.mse_invert_mask_loss

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

def downsample_nearest_neighbor(data):
    H, W, C = nn.height_width_channels(data)
    halfH = int(H / 2)
    halfW = int(W / 2)
    return tf.image.resize_images(data, [halfH, halfW], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


#Similar to the two above, but a downsampling using nearest-neighbor
def iterated_nearest_neighbor_downsample(data, N):
    return lu.iterate(lambda x : downsample_nearest_neighbor(x), data, N - 1)

#Yields the downsampling loss weighting factors
#in the form [weight_8, weight_16, weight_32, weight_64]
#TODO: Hyperparameter optimization?
#TODO: Does it make sense to modify these while training?
def downsample_weighting_factors():
    return [8, 4, 2, 1]

def deep_downsample_weighting_factors():
    return [64, 32, 16, 8, 4, 2, 1]


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

#Given a bounding box in template space (see OpenImagesCommon for format example)
#and the template position map outputs of the neural net, compute at every position
#the squared distance each point falls outside the bounding box, or if it falls inside,
#zero
def bbox_sq_dist(box, net_out_templates):
    template_x = net_out_templates[:, :, :, 0]
    template_y = net_out_templates[:, :, :, 1]

    x_min, x_max, y_min, y_max = box 

    below_x_min = tf.maximum(0.0, x_min - template_x)
    above_x_max = tf.maximum(0.0, template_x - x_max)
    below_y_min = tf.maximum(0.0, y_min - template_y)
    above_y_max = tf.maximum(0.0, template_y - y_max)

    x_dist = tf.maximum(below_x_min, above_x_max)
    y_dist = tf.maximum(below_y_min, above_y_max)
    return tf.square(x_dist) + tf.square(y_dist)

#Loss on the openimages dataset from part bounding boxes
def openimages_part_loss(x, expected_part_maps):
    eye_dist = bbox_sq_dist(eyes_bbox, x)
    beard_dist = bbox_sq_dist(beard_bbox, x)
    mouth_dist = bbox_sq_dist(mouth_bbox, x)
    foot_dist = bbox_sq_dist(foot_bbox, x)
    leg_dist = bbox_sq_dist(leg_bbox, x)
    ear_dist = bbox_sq_dist(ears_bbox, x)
    hair_dist = bbox_sq_dist(hair_bbox, x)
    head_dist = bbox_sq_dist(head_bbox, x)
    face_dist = bbox_sq_dist(face_bbox, x)
    arm_dist = tf.minimum(bbox_sq_dist(l_arm_bbox, x), bbox_sq_dist(r_arm_bbox, x))
    nose_dist = bbox_sq_dist(nose_bbox, x)
    hand_dist = tf.minimum(bbox_sq_dist(l_hand_bbox, x), bbox_sq_dist(r_hand_bbox, x))
    dist_mats = [None, eye_dist, beard_dist, mouth_dist, foot_dist, leg_dist, ear_dist,
                 hair_dist, head_dist, face_dist, arm_dist, nose_dist, hand_dist]
    result = 0.0
    for i in range(1, 13):
        dist_mat = dist_mats[i]
        expected_mask = tf.cast(expected_part_maps[:, i, :, :], tf.float32)
        #Okay, great. Now multiply the two together to only count distances which
        #are where we expect parts
        multiplied = dist_mat * expected_mask
        result = result + tf.reduce_mean(multiplied)
    return result


#Loss used for dataset elements coming from the OpenImages dataset
def openimages_loss(net_out, expected_part_maps, covering_frac=0.25, mask_weight=90000.0):
    #We only care about the 64x64 outputs here
    net_out = net_out[0]
    net_out = tf.image.resize_image_with_pad(net_out, 256, 256)

    net_out_templates = net_out[:, :, :, 0:3]
    net_out_masks = net_out[:, :, :, 3]

    expected_masks = expected_part_maps[:, 0, :, :]
    expected_masks = tf.cast(expected_masks, tf.float32)
    expected_weights = tf.math.reduce_mean(expected_masks, axis=[1, 2])
    tot_expected_weight = tf.math.reduce_mean(expected_masks)
    tot_not_expected_weight = 1.0 - tot_expected_weight
    not_expected_masks = 1.0 - expected_masks
    
    #Compare the network's output masks with the expected masks, with a twist:
    #we intuitively want the weight of net_out_masks within expected_masks
    #to be greater than or equal to one-fourth of the total weight of expected_masks
    #(case of person with outstretched arms, bounding box), but want to penalize
    #for any activation within the region lower than this. At the same time,
    #we want to penalize for activation anywhere outside the bounding boxes
    not_expected_component = not_expected_masks * net_out_masks
    expected_component = expected_masks * net_out_masks
    expected_component_weights = tf.reduce_mean(expected_component, axis=[1, 2])

    not_expected_loss = tf.reduce_mean(tf.square(not_expected_component)) * tot_not_expected_weight
    
    #We just want to ensure the following collection of fracs is greater than covering_frac, and less than 1.0
    covered_fracs = expected_component_weights / expected_weights
    covered_fracs_min_sq_diffs = tf.square(covered_fracs - covering_frac)
    covered_fracs_max_sq_diffs = tf.square(covered_fracs - 1.0)

    lt_frac_contrib = tf.where(covered_fracs >= covering_frac, tf.zeros([batch_size]), covered_fracs_min_sq_diffs)
    gt_frac_contrib = tf.where(covered_fracs <= 1, tf.zeros([batch_size]), covered_fracs_max_sq_diffs)

    expected_loss = tot_expected_weight * tf.reduce_mean(lt_frac_contrib + gt_frac_contrib)

    first_mask_loss = (expected_loss + not_expected_loss) * mask_weight
    
    part_loss = openimages_part_loss(net_out_templates, expected_part_maps)

    return first_mask_loss + part_loss


    #TODO: Implement losses for individual parts, too

#Loss for the coco keypoints dataset
#each of seg_masks, keypoint_positions, keypoint_weights are 256x256
def coco_loss(net_out, seg_masks, keypoint_positions, keypoint_weights, mask_weight=0.09):
    net_out_masks = tf.reshape(net_out[0][:, :, :, 3], [-1, 64, 64, 1])
    net_out_templates = tf.reshape(net_out[0][:, :, :, 0:3], [-1, 64, 64, 3])
    #Okay, now what we do is we upscale both to 256x256
    net_out_masks = tf.image.resize_image_with_pad(net_out_masks, 256, 256)
    net_out_masks = tf.reshape(net_out_masks, [-1, 256, 256, 1])
    net_out_templates = tf.image.resize_image_with_pad(net_out_templates, 256, 256)

    #Okay, great. Now let's take what we got and compare it directly
    #to the keypoint maps
    mse_diffs = tf.losses.mean_squared_error(net_out_templates, keypoint_positions, weights=keypoint_weights)

    mask_diff = tf.losses.mean_squared_error(seg_masks, net_out_masks)

    result = mse_diffs + mask_weight * mask_diff
    return result

#The dealio for loss on the RGB model is that we get our
#net out as 64x64, with an extra feature map dedicated to
#predictions of the image masks. Hence, we split the last feature map
#of net_out off and pass it down through to the downsampling masked mabs loss
def downsampling_masked_mabs_loss_rgb(net_out, expected, mask, net_out_w=64, mask_weight=.09):
    net_out_template = [] 
    for net_out_size in net_out:
        net_out_template.append(net_out_size[:, :, :, 0:3])

    net_out_masks = tf.reshape(net_out[0][:, :, :, 3], [-1, net_out_w, net_out_w, 1])

    #Compute the template coordinate component of the loss
    template_component = downsampling_masked_mabs_loss(net_out_template, expected, mask, target_size=64, num_downsamples=5)
    
    #Compute the mask component of the loss
    mask_expected = tf.cast(mask, tf.float32)
    mask_expected = tf.image.resize_image_with_pad(mask_expected, net_out_w, net_out_w)
    mask_expected = mask_expected

    mask_loss = tf.losses.mean_squared_error(mask_expected, net_out_masks)
    mask_component = mask_loss * mask_weight

    return template_component + mask_component


#Same deal as above, but with average pooling
#and a boolean mask defining the extent to compute losses over
def downsampling_masked_mabs_loss(net_out, expected, mask, target_size=512, num_downsamples=8):

    expected = tf.image.resize_image_with_pad(expected, target_size, target_size)

    #For the expected, do an iterated nearest-neighbor downsample
    expected_out = iterated_nearest_neighbor_downsample(expected, num_downsamples)

    #Being __more conservative__ about downsampled masks
    expected_init_mask = tf.cast(tf.cast(mask, tf.int32), tf.float32)
    expected_init_mask = tf.image.resize_image_with_pad(expected_init_mask, target_size, target_size)

    inverted_expected_init_mask = 1.0 - expected_init_mask
    inverted_masks_out = iterated_max_pool_downsample(inverted_expected_init_mask, num_downsamples)

    losses = map(lambda triple : mask_loss_fn(*triple), zip(net_out, expected_out, inverted_masks_out))

    #Multiply by downsampling weighting factors and sum
    weighted_losses = map(lambda pair : tf.multiply(*pair), zip(losses, deep_downsample_weighting_factors()))
    result = reduce(tf.add, weighted_losses, 0.0)

    result = result / 64.0

    return result
