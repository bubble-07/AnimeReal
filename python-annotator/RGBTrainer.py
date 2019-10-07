#This is it, the big deal -- takes in a collection of 412x512 RGB images
#masked with green for transparent areas around people which is annotated
#with template index position masks. Also takes in a root directory for
#background images to composite together with these given foregrounds,
#and trains a model which takes things like these composite images
#downscaled to 256x256
#as input and yields a stacked 64x64 output with the first three feature maps
#representing template body position guesses, and the fourth representing the
#best full-body segmentation mask guess (floating-point, probabilistic rep)

#TODO: Much of this is similar to AnnotationSuitDepthTrainer -- fix things up here

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import radical_densenet as rad_dense
from glob import glob
import shutil
import depth_convolution
import loss as losses
import augmentation
import cv2
import time
import os
import sys
import math
import numpy as np
import debugging

import matplotlib.pyplot as plt
import random

from tensorflow.python.client import timeline
import neural_architecture as na

import neural_structuring as ns

import neural_ops as nn
import mathutils as mu

from random import shuffle

import tensorflow as tf
import CocoCommon

from RGBTrainerParams import *
import StandardBody

#Set up training loss plot
#TODO: Abstract this stuff away, in its own file!
train_loss_xdata = []
train_loss_ydata = []
valid_loss_xdata = []
valid_loss_ydata = []
plt.show()
axes = plt.gca()
axes_xlimit = 10
axes.set_xlim(0, axes_xlimit)
axes.set_ylim(0, 400.0)
train_line, = axes.plot(train_loss_xdata, train_loss_ydata, 'r-', label='training loss')
valid_line, = axes.plot(valid_loss_xdata, valid_loss_ydata, 'g-', label='validation loss')

#Okay, great, now we just need to define a "build_dataset_from_dir" function

feature_description = {
    'rgbImage': tf.FixedLenFeature([], tf.string, default_value=''),
    'templateIndexImage': tf.FixedLenFeature([], tf.string, default_value='')
}

background_feature_description = {
    'rgbImage': tf.FixedLenFeature([], tf.string, default_value='')
}

openimages_feature_description = {
    'rgbImage': tf.FixedLenFeature([], tf.string, default_value=''),
    'annotationImage': tf.FixedLenFeature([], tf.string, default_value='')
}

coco_feature_description = {
    'rgbImage': tf.FixedLenFeature([], tf.string, default_value=''),
    'annotationImage': tf.FixedLenFeature([], tf.string, default_value=''),
    'entityArray': tf.FixedLenFeature([], tf.string, default_value='')
}

num_standardBody_points = StandardBody.pointArray.shape[0]

standardBodyArray = np.copy(StandardBody.pointArray)
standardBodyArray = np.vstack((standardBodyArray, np.array([[0.0, 0.0, 0.0]], dtype=np.float32)))

standardBodyConst = tf.constant(standardBodyArray)

coco_keypoint_positions = np.array(CocoCommon.coco_keypoint_positions, dtype=np.float32)
coco_keypoint_weights = np.array(CocoCommon.coco_keypoint_weights, dtype=np.float32)

def parse_coco_example(proto):
    parsed_dict = tf.parse_single_example(proto, coco_feature_description)

    annotation_image_bytes = parsed_dict['annotationImage']
    annotation_image_flat = tf.decode_raw(annotation_image_bytes, tf.uint8)

    entity_array_bytes = parsed_dict['entityArray']
    entity_array_flat = tf.decode_raw(entity_array_bytes, tf.float32)

    seg_mask = tf.reshape(annotation_image_flat, [640, 640])

    #Okay, now deal with the entity array by reshaping it so that it's all
    #in the shape [num_entities, num_keypoints, 3], where the last dimension
    #is x frac, y frac, confidence
    num_coco_keypoints = CocoCommon.num_coco_keypoints

    entity_array = tf.reshape(entity_array_flat, [-1, num_coco_keypoints, 3])

    rgb_image_bytes = parsed_dict['rgbImage']
    rgb_image_flat = tf.decode_raw(rgb_image_bytes, tf.uint8)

    rgb_image = tf.reshape(rgb_image_flat, [640, 640, 3])

    return (rgb_image, seg_mask, entity_array)

def infinite(x):
    return (math.isinf(x) or math.isnan(x))

#Given a bounding box (in normalized image coordinates) which focuses on
#a collection of keypoints, derive a new bounding box which is that bounding box,
#but altered a little for both data augmentation and to make sure that we're not
#requesting too much upsampling, and also to ensure that the aspect ratio distortion
#isn't too terribly bad, while still __mostly__ keeping the subject in the chosen
#crop frame
def random_adjust_box(focused_box):
    x_min, x_max, y_min, y_max = focused_box
    if infinite(x_min) or infinite(y_min) or infinite(y_max) or infinite(x_max):
        return [0.0, 1.0, 0.0, 1.0]

    focus_x = (x_min + x_max) / 2.0
    focus_y = (y_min + y_max) / 2.0
    x_spread = x_max - x_min
    y_spread = y_max - y_min

    orig_x_spread = x_spread
    orig_y_spread = y_spread

    #First, perform all necessary adjustments on the focus and the spread
    #Then, after adjustments, we'll restrict the box to ensure that it's valid
    #(making sure that it lies entirely within the frame) via translation
    if (x_spread < COCO_MIN_X_SPREAD):
        x_spread = COCO_MIN_X_SPREAD
    if (y_spread < COCO_MIN_Y_SPREAD):
        y_spread = COCO_MIN_Y_SPREAD

    #Adjust spreads by expansion so that aspect ratio constraints are respected
    #(for now) -- later in the enlargment/shrink step, we may unintentionally undo this,
    #but it should sorta-kinda work out on average

    empirical_aspect_ratio = x_spread / y_spread
    if (empirical_aspect_ratio < COCO_MIN_ASPECT_RATIO):
        x_spread = y_spread * COCO_MIN_ASPECT_RATIO
    if (empirical_aspect_ratio > COCO_MAX_ASPECT_RATIO):
        y_spread = x_spread / COCO_MAX_ASPECT_RATIO

    #Okay, great, now perform magnification adjustments
    x_mag = random.uniform(COCO_MIN_X_MAG, COCO_MAX_X_MAG)
    y_mag = random.uniform(COCO_MIN_Y_MAG, COCO_MAX_Y_MAG)

    x_spread *= x_mag
    y_spread *= y_mag
 
    #Correct the aspect ratio again to fall within acceptable limits
    empirical_aspect_ratio = x_spread / y_spread
    if (empirical_aspect_ratio < COCO_MIN_ASPECT_RATIO):
        x_spread = y_spread * COCO_MIN_ASPECT_RATIO
    if (empirical_aspect_ratio > COCO_MAX_ASPECT_RATIO):
        y_spread = x_spread / COCO_MAX_ASPECT_RATIO

    #Clip spreads so they're not too wide
    if (x_spread > 1.0):
        x_spread = 1.0
    if (y_spread > 1.0):
        y_spread = 1.0

    #Great, now shift the focus by translation
    x_shift_frac = random.uniform(-COCO_MAX_X_SPREAD_SHIFT_FRAC, COCO_MAX_X_SPREAD_SHIFT_FRAC)
    y_shift_frac = random.uniform(-COCO_MAX_Y_SPREAD_SHIFT_FRAC, COCO_MAX_Y_SPREAD_SHIFT_FRAC)
    x_shift = x_spread * x_shift_frac
    y_shift = y_spread * y_shift_frac

    focus_x += x_shift
    focus_y += y_shift

    half_width = x_spread / 2.0
    half_height = y_spread / 2.0

    #Finally, we need to shift the focus a bit to ensure that the bounding box
    #actually lies entirely within the original image
    if (focus_x + half_width > 1.0):
        focus_x = 1.0 - half_width
    if (focus_x - half_width < 0.0):
        focus_x = half_width
    if (focus_y + half_height > 1.0):
        focus_y = 1.0 - half_height
    if (focus_y - half_height < 0.0):
        focus_y = half_height

    #Derive the new box coordinates
    x_min = focus_x - half_width
    x_max = focus_x + half_width
    y_min = focus_y - half_height
    y_max = focus_y + half_height

    result_box = [x_min, x_max, y_min, y_max]

    return result_box

#Given a 640x640 rgb image, a 640x640 segmentation mask, and an array of coco
#entity keypoints, return a 256x256x3 uint8 image, a 256x256 uint8 segmentation mask,
#and a 256x256x3 float32 array of template positions.
#These are all taken as np arrays so that this can be wrapped in py_func
def coco_random_crop(rgb_image, seg_mask, entity_array):
    #First thing's first, take the entity array and dictionary-expand the keypoints
    dict_expansions = []
    bounding_boxes = []
    #With that, also create a collection of bounding boxes
    for i in range(entity_array.shape[0]):
        dict_result, boundingBox = CocoCommon.dict_expand_keypoint_array(entity_array[i])
        bounding_boxes.append(boundingBox)
        dict_expansions.append(dict_result)

    #Pick a random bounding box to put into focus
    focused_ind = random.randrange(len(bounding_boxes))
    focused_box = bounding_boxes[focused_ind]

    adjusted_box = random_adjust_box(focused_box)

    min_x, max_x, min_y, max_y = adjusted_box

    epsilon = 1.0 / (640.0 * 4.0)

    spread_x = max_x - min_x
    spread_y = max_y - min_y
    
    spread_x += epsilon
    spread_y += epsilon

    if (min_x < 0.0 or min_x > 1.0):
        min_x = 0.0
    if (max_x > 1.0 or max_x < 0.0):
        max_x = 1.0
    if (min_y < 0.0 or min_y > 1.0):
        min_y = 0.0
    if (max_y > 1.0 or max_y < 0.0):
        max_y = 1.0


    int_min_x = max(0, int(min_x * 640.0) - 1)
    int_max_x = min(640, int(max_x * 640.0) + 1)
    int_min_y = max(0, int(min_y * 640.0) - 1)
    int_max_y = min(640, int(max_y * 640.0) + 1)

    smol_rgb_crop = rgb_image[int_min_y:int_max_y, int_min_x:int_max_x]
    smol_seg_crop = seg_mask[int_min_y:int_max_y, int_min_x:int_max_x]
    smol_seg_crop = np.reshape(smol_seg_crop, [smol_seg_crop.shape[0], smol_seg_crop.shape[1], 1])

    result_img = cv2.resize(smol_rgb_crop, (256, 256))
    result_seg = cv2.resize(smol_seg_crop, (256, 256))

    #Okay, now that adjusted box will become the (256x256) crop region,
    #so what we need to do is we need to adjust all of the positions of
    #the dictionary points to have coordinates defined relative
    #to that crop region

    adjusted_dict_expansions = []
    for dict_expansion in dict_expansions:
        adjusted_dict_expansion = {}
        for pointName in dict_expansion:
            origPoint = dict_expansion[pointName]
            x = origPoint[0]
            y = origPoint[1]
            v = origPoint[2]
            #If confidence is too low, don't include the point at all
            if (v < 0.1):
                continue

            relative_frac_x = (x - min_x) / spread_x
            relative_frac_y = (y - min_y) / spread_y
            adjusted_x = int(relative_frac_x * 256.0)
            adjusted_y = int(relative_frac_y * 256.0)
            newPoint = np.array([adjusted_x, adjusted_y], dtype=np.int32)
            adjusted_dict_expansion[pointName] = newPoint

        adjusted_dict_expansions.append(adjusted_dict_expansion)

    rect_shape = [256, 256]
    result_template, result_nonzero = CocoCommon.draw_keypoint_array(adjusted_dict_expansions, rect_shape)

    return (result_img, result_seg, result_template, result_nonzero)

def coco_random_crop_tf(rgb_image, seg_mask, entity_array):
    return tf.py_func(coco_random_crop, [rgb_image, seg_mask, entity_array], [tf.uint8, tf.uint8, tf.float32, tf.uint8])

def coco_random_crop_cleanup(rgb_image, seg_mask, result_template, result_nonzero):
    rgb_image = tf.reshape(rgb_image, [batch_size, 256, 256, 3])
    seg_mask = tf.reshape(seg_mask, [batch_size, 256, 256, 1])
    result_template = tf.reshape(result_template, [batch_size, 256, 256, 3])
    result_nonzero = tf.reshape(result_nonzero, [batch_size, 256, 256, 1])
    return (rgb_image, seg_mask, result_template, result_nonzero)

def parse_openimages_example(proto):
    parsed_dict = tf.parse_single_example(proto, openimages_feature_description)
    annotation_image_bytes = parsed_dict['annotationImage']
    annotation_image_flat = tf.decode_raw(annotation_image_bytes, tf.uint16)
    annotation_image_flat = tf.reshape(annotation_image_flat, [1, 256, 256])

    part_numerals = tf.constant([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=tf.uint16) 
    part_numerals = tf.reshape(part_numerals, [13, 1, 1])
    
    part_images = tf.bitwise.bitwise_and(part_numerals, annotation_image_flat)
    part_images = tf.cast(part_images, tf.int32) > 0

    rgb_image_bytes = parsed_dict['rgbImage']
    rgb_image_flat = tf.decode_raw(rgb_image_bytes, tf.uint8)

    rgb_image = tf.reshape(rgb_image_flat, [256, 256, 3])

    return (rgb_image, part_images)



def parse_background_example(proto):
    parsed_dict = tf.parse_single_example(proto, background_feature_description)
    rgb_image_bytes = parsed_dict['rgbImage']
    rgb_image_flat = tf.decode_raw(rgb_image_bytes, tf.uint8)

    rgb_image = tf.reshape(rgb_image_flat, [952, 1430, 3])

    return (rgb_image,)

def augment_background_example(rgb_image):
    #This is responsible for "augmenting" the background image
    #in the sense that we're selecting a 256x256 crop of the background here

    #Pick a bounding box in the image such that each dimension includes
    #at least MIN_FRAC of the total dimension length
    box_width = tf.random_uniform([], minval=SELECT_BACKGROUND_MIN_FRAC, maxval=1.0)
    box_height = tf.random_uniform([], minval=SELECT_BACKGROUND_MIN_FRAC, maxval=1.0)
    box_x1 = tf.random_uniform([], minval=0.0, maxval=1.0 - box_width)
    box_y1 = tf.random_uniform([], minval=0.0, maxval=1.0 - box_height)
    box_x2 = box_x1 + box_width
    box_y2 = box_y1 + box_height

    rgb_image = tf.reshape(rgb_image, [1, 952, 1430, 3])
    boxes = [[box_y1, box_x1, box_y2, box_x2]]
    box_ind = [0]
    
    rgb_image = tf.image.crop_and_resize(rgb_image, boxes, box_ind, [256, 256])
    rgb_image = tf.reshape(rgb_image, [256, 256, 3])

    #Okay, once we've picked exactly what we want out of the background,
    #Do background hue and brightness adjustments
    if (AUGMENT_BACKGROUND_HUE_SHIFT):
        hue_shift = tf.random_normal([], stddev=AUGMENT_BACKGROUND_HUE_STDEV)
        rgb_image = tf.image.adjust_hue(rgb_image, hue_shift)
    
    if (AUGMENT_BACKGROUND_BRIGHTNESS_SHIFT):
        bright_shift = tf.random_uniform([], minval=AUGMENT_MIN_BACKGROUND_BRIGHTNESS_DELTA, maxval=AUGMENT_MAX_BACKGROUND_BRIGHTNESS_DELTA)
        rgb_image = tf.image.adjust_brightness(rgb_image, bright_shift)

    return (rgb_image,)


def parse_example_to_sample_label(proto):
    parsed_dict = tf.parse_single_example(proto, feature_description)
    rgb_image_bytes = parsed_dict['rgbImage']
    template_index_image_bytes = parsed_dict['templateIndexImage']
    #Interpret the depth image bytes as a float32
    rgb_image_flat = tf.decode_raw(rgb_image_bytes, tf.uint8)
    template_index_image_flat = tf.decode_raw(template_index_image_bytes, tf.uint16)
    template_index_image_flat = tf.cast(template_index_image_flat, tf.int32)
    template_mask_flat = template_index_image_flat < num_standardBody_points

    template_image_flat = tf.gather(standardBodyConst, template_index_image_flat)

    template_positions_flat = template_index_image_flat

    #Derive the template image through indexing

    rgb_image = tf.reshape(rgb_image_flat, [424, 512, 3])
    template_image = tf.reshape(template_image_flat, [424, 512, 3])
    template_mask = tf.reshape(template_mask_flat, [424, 512, 1])

    return (rgb_image, template_image, template_mask)

def augment_example(rgb_image, template_image, template_mask):
    #Method which applies all those augmentations which are relevant
    #to changing just the foreground rgb image and corresponding template
    #image and mask

    template_mask = tf.cast(template_mask, tf.float32)
    rgb_image = tf.cast(rgb_image, tf.float32) / 255.0

    #Before doing anything, perform those augmentations which
    #only change the rgb image

    if (AUGMENT_HUE_SHIFT):
        hue_shift = tf.random_normal([], stddev=AUGMENT_HUE_STDEV)
        rgb_image = tf.image.adjust_hue(rgb_image, hue_shift)

    if (AUGMENT_BRIGHTNESS_SHIFT):
        bright_shift = tf.random_uniform([], minval=AUGMENT_MIN_BRIGHTNESS_DELTA, maxval=AUGMENT_MAX_BRIGHTNESS_DELTA)
        rgb_image = tf.image.adjust_brightness(rgb_image, bright_shift)

    #This will make things a bit easier, because when we manipulate the
    #rgb image, we'll usually wind up modifying the mask, too
    image_mask = tf.concat([rgb_image, template_mask], 2)

    #Flips are special, because they also flip the annotation positions
    if (AUGMENT_FLIP):
        flip_boolean = tf.random_uniform([], minval=-1.0, maxval=1.0) > 0.0

        image_mask = tf.cond(flip_boolean, lambda: image_mask, lambda: tf.image.flip_left_right(image_mask))
        template_image = tf.cond(flip_boolean, lambda: template_image, lambda: tf.image.flip_left_right(template_image * np.array([-1.0, 1.0, 1.0])))

    #Okay, now we've dealt with flips.
    #Deal with rotations and translations
    translate_x = 0.0
    translate_y = 0.0

    rotate = 0.0

    x_center = 256
    y_center = 212

    x_fac = 1.0
    y_fac = 1.0

    if (AUGMENT_TRANSLATE):
        translate_x = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_TRANSLATE_X_PIX
        translate_y = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_TRANSLATE_Y_PIX
    if (AUGMENT_ROTATE):
        rotate = tf.random_normal([], stddev=AUGMENT_ROTATE_ANGLE_STDEV)
    if (AUGMENT_ASPECT_RATIO):
        x_fac = tf.random_uniform([], minval=AUGMENT_ASPECT_X_MIN, maxval=AUGMENT_ASPECT_X_MAX)
        y_fac = tf.random_uniform([], minval=AUGMENT_ASPECT_Y_MIN, maxval=AUGMENT_ASPECT_Y_MAX)

    aspect_transform = augmentation.get_aspect_ratio_transform(x_fac, y_fac, x_center, y_center)
    rot_scale_xlate_transform = augmentation.get_affine_transform(1.0, rotate, 1.0, translate_x, translate_y, x_center, y_center)
    total_transform = tf.contrib.image.compose_transforms(aspect_transform, rot_scale_xlate_transform)

    #Great, now stack all of the things together, and apply the transform!
    all_together = tf.concat([image_mask, template_image], 2)
    all_transformed = tf.contrib.image.transform(all_together, total_transform)

    #Okay, great. Now, unpack all of that back to the original parameters
    rgb_image = tf.reshape(all_transformed[:, :, 0:3], [424, 512, 3]) * 255.0
    template_mask = tf.reshape(all_transformed[:, :, 3], [424, 512, 1])
    template_image = tf.reshape(all_transformed[:, :, 4:], [424, 512, 3])

    #Final clean-up
    template_mask = template_mask > 0.5
    return (rgb_image, template_image, template_mask)
 
def build_openimages_dataset_from_dir(tfrecordRoot, batch_size):
    tfrecordFiles = [y for x in os.walk(tfrecordRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]
    
    raw_dataset = tf.data.TFRecordDataset(tfrecordFiles)

    raw_dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset_formatted = raw_dataset.map(parse_openimages_example, num_parallel_calls=CPU_PARALLEL_THREADS)

    result_set = dataset_formatted.batch(batch_size, drop_remainder=True)
    result_set = result_set.repeat()

    result_set = result_set.prefetch(2)

    return result_set

  
def build_coco_dataset_from_dir(tfrecordRoot, batch_size):
    tfrecordFiles = [y for x in os.walk(tfrecordRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]
    tfrecordFiles = sorted(tfrecordFiles)
    random.shuffle(tfrecordFiles)
    
    raw_dataset = tf.data.TFRecordDataset(tfrecordFiles)

    raw_dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset_formatted = raw_dataset.map(parse_coco_example, num_parallel_calls=CPU_PARALLEL_THREADS)

    dataset_drawn = dataset_formatted.map(coco_random_crop_tf, num_parallel_calls=CPU_PARALLEL_THREADS)

    result_set = dataset_drawn.batch(batch_size, drop_remainder=True)

    cleaned_set = result_set.map(coco_random_crop_cleanup, num_parallel_calls=CPU_PARALLEL_THREADS)

    result_set = cleaned_set.repeat()

    result_set = result_set.prefetch(2)

    return result_set
  

def build_background_dataset_from_dir(tfrecordRoot, batch_size):
    tfrecordFiles = [y for x in os.walk(tfrecordRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]
    #tfrecordFiles = sorted(tfrecordFiles)
    #random.shuffle(tfrecordFiles)
    
    raw_dataset = tf.data.TFRecordDataset(tfrecordFiles)

    raw_dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset_formatted = raw_dataset.map(parse_background_example, num_parallel_calls=CPU_PARALLEL_THREADS)
    dataset_augmented = dataset_formatted.map(augment_background_example, num_parallel_calls=CPU_PARALLEL_THREADS)

    result_set = dataset_augmented.batch(batch_size, drop_remainder=True)
    result_set = result_set.repeat()

    result_set = result_set.prefetch(2)

    return result_set

#TODO: Maybe merge with annotationSuitDepthTrainer's version?
#whatever the case, you'll need a separate method for backgrounds

def build_dataset_from_dir(tfrecordRoot, batch_size):
    tfrecordFiles = [y for x in os.walk(tfrecordRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]
    tfrecordFiles = sorted(tfrecordFiles)
    random.shuffle(tfrecordFiles)

    raw_dataset = tf.data.TFRecordDataset(tfrecordFiles)

    raw_dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    #Parse from the raw dataset into (depth image, template positions, template mask) format
    dataset_formatted = raw_dataset.map(parse_example_to_sample_label, num_parallel_calls=CPU_PARALLEL_THREADS)
    dataset_augmented = dataset_formatted.map(augment_example, num_parallel_calls=CPU_PARALLEL_THREADS)

    #Okay, great! Now it's batching time!
    result_set = dataset_augmented.batch(batch_size, drop_remainder=True)
    result_set = result_set.repeat()

    result_set = result_set.prefetch(2)

    return result_set

#Given a collection of source images and source masks, and a collection of backdrops,
#yields 256x256 training images 
#TODO: Augmentation should be interesting
def composite_backgrounds(src_imgs, src_masks, backdrops):
    #Resize image to 256x256, which is what we expect for input to the net
    src_imgs = tf.image.resize_image_with_pad(src_imgs, 256, 256)

    #The backdrop is originally 1430x952
    #We need to extract some random scaled crop of it that's 256x256
    #rand_backdrop_crop = tf.image.random_crop(backdrops, [batch_size, 256, 256, 3])
    rand_backdrop_crop = tf.image.resize_image_with_pad(backdrops, 256, 256)

    rand_backdrop_crop = tf.cast(rand_backdrop_crop, tf.float32)

    #Okay, now we need to composite that with the source image wherever
    #the mask isn't true
    masks = tf.cast(src_masks, tf.float32)
    masks = tf.image.resize_image_with_pad(masks, 256, 256)
    masks = masks > 0.5
    masks = tf.reshape(masks, [batch_size, 256, 256])
    masks = tf.stack([masks, masks, masks], axis=-1)

    result_img = tf.where(masks, src_imgs, rand_backdrop_crop)
    return result_img
    
    




def run_tensorflow(quantize, restore):

    #Make everything repeatable here
    np.random.seed(1234)
    random.seed(1234)
    tf.random.set_random_seed(1234)

    '''
    TRAINING_DATASET_DIR = sys.argv[1]
    BACKGROUND_DATASET_DIR = sys.argv[2]
    OPENIMAGES_DATASET_DIR = sys.argv[3]
    COCO_DIR = sys.argv[4]
    VALIDATION_DATASET_DIR = sys.argv[5]
    BACKGROUND_VALIDATION_DATASET_DIR = sys.argv[6]
    OPENIMAGES_VALIDATION_DATASET_DIR = sys.argv[7]
    COCO_VALID_DIR = sys.argv[8]
    '''
    TRAINING_DATASET_DIR = sys.argv[1]
    BACKGROUND_DATASET_DIR = sys.argv[2]
    COCO_DIR = sys.argv[3]
    #VALIDATION_DATASET_DIR = sys.argv[4]
    #BACKGROUND_VALIDATION_DATASET_DIR = sys.argv[5]
    #COCO_VALID_DIR = sys.argv[6]
    VALIDATION_DATASET_DIR = TRAINING_DATASET_DIR
    BACKGROUND_VALIDATION_DATASET_DIR = BACKGROUND_DATASET_DIR
    COCO_VALID_DIR = COCO_DIR

    global axes_xlimit

    print("Prepping everything...")
        #TODO: Split into train/test sets, as well!
    #Should that happen at the data level? (Yeah, probably)

    #DATASET PIPELINE SETUP

    #At present, don't use openimage set, because the bounding boxes don't seem that helpful
    #train_openimages_set = build_openimages_dataset_from_dir(OPENIMAGES_DATASET_DIR, batch_size)
    
    train_coco_set = build_coco_dataset_from_dir(COCO_DIR, batch_size)

    valid_coco_set = build_coco_dataset_from_dir(COCO_VALID_DIR, batch_size)

    #validation_openimages_set = build_openimages_dataset_from_dir(OPENIMAGES_VALIDATION_DATASET_DIR, batch_size)

    train_background_set = build_background_dataset_from_dir(BACKGROUND_DATASET_DIR, batch_size) 

    validation_background_set = build_background_dataset_from_dir(BACKGROUND_VALIDATION_DATASET_DIR, batch_size)

    
    #Create a dataset for training
    train_set = build_dataset_from_dir(TRAINING_DATASET_DIR, batch_size)
    #Prefetching will only happen on the training set
    #TODO: Is this a good thing?
    #train_set = train_set.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

    #Create a dataset for validation
    validation_set = build_dataset_from_dir(VALIDATION_DATASET_DIR, batch_size)
    #TODO: Also add in a evaluation set

    regular_handle = tf.placeholder(tf.string, shape=[])
    regular_iterator = tf.data.Iterator.from_string_handle(regular_handle, train_set.output_types, train_set.output_shapes)
    regular_next_element = regular_iterator.get_next()


    #Okay, great. Now, we train a neural network to generate heatmaps

    
    #INPUT/OUTPUT AND OPTIMIZATION OBJECTIVE

    train_iterator = train_set.make_one_shot_iterator()
    valid_iterator = validation_set.make_one_shot_iterator()


    background_handle = tf.placeholder(tf.string, shape=[])
    background_iterator = tf.data.Iterator.from_string_handle(background_handle, train_background_set.output_types, train_background_set.output_shapes)
    background_next_element = background_iterator.get_next()

    train_background_iterator = train_background_set.make_one_shot_iterator()
    valid_background_iterator = validation_background_set.make_one_shot_iterator()

    #train_openimages_iterator = train_openimages_set.make_one_shot_iterator()
    #valid_openimages_iterator = validation_openimages_set.make_one_shot_iterator()

    coco_handle = tf.placeholder(tf.string, shape=[])
    coco_iterator = tf.data.Iterator.from_string_handle(coco_handle, train_coco_set.output_types, train_coco_set.output_shapes)
    coco_next_element = coco_iterator.get_next()

    train_coco_iterator = train_coco_set.make_one_shot_iterator()
    valid_coco_iterator = valid_coco_set.make_one_shot_iterator()

    #TODO: Validation set for coco, too, or is this overkill?

    #train_openimages_rgb, train_openimages_parts = train_openimages_iterator.get_next()
    #valid_openimages_rgb, valid_openimages_parts = valid_openimages_iterator.get_next()
    to_meters = (1.0 / 1000.0)

    coco_rgb, coco_masks, coco_positions_unaltered, coco_weights = coco_next_element

    coco_positions = coco_positions_unaltered * to_meters


    regular_background = background_next_element[0]

    #Multiplication factor for the inputs, so that they fall within a given normalized range
    #To keep them within a 0 to 6 RELU6 range, we can do it like so:
    input_fac = (6.0 / 255.0)


    #Okay, great, but that's not all. We also need something to take outputs from the network (which
    #will be in [0.0, 6.0] range), and convert them into floating-point confidence (in 0.0 to 1.0)
    #and standard-template-millimeters-space
    def rescale_output(output_tensors):
        result = []
        for output_tensor in output_tensors:
            out_template = output_tensor[:, :, :, 0:3]
            out_mask = output_tensor[:, :, :, 3]

            template_offsets = np.array([StandardBody.xmin, StandardBody.ymin, StandardBody.zmin], dtype=np.float32)
            template_scales = np.array([StandardBody.xspread, StandardBody.yspread, StandardBody.zspread], dtype=np.float32)
            
            template_offsets *= to_meters
            template_scales *= to_meters

            one_sixth = 1.0 / 6.0
            template_scales *= one_sixth
            
            result_template = out_template * template_scales + template_offsets

            result_mask = out_mask * one_sixth

            result_mask = tf.expand_dims(result_mask, axis=-1)

            #Great, now concat them together
            concatted = tf.concat([result_template, result_mask], axis=-1)

            result.append(concatted)
        return result

    #NETWORK PRE-SCALING

    coco_rgb = tf.cast(coco_rgb, tf.float32) * input_fac

    #train_openimages_rgb = tf.cast(train_openimages_rgb, tf.float32) * input_fac

    regular_in, regular_expected, regular_mask = regular_next_element
    regular_expected *= to_meters

    regular_in = composite_backgrounds(regular_in, regular_mask, regular_background)
    regular_in = regular_in * input_fac

    #Training exponential decay set-up
    
    global_step = tf.Variable(0, trainable=False) 
    learning_rate = tf.train.exponential_decay(init_training_rate, global_step, 
                                               decay_steps=decay_steps, decay_rate=decay_rate) 

    #NEURAL OP SETUP
    #Build the Neural_Op for heatmap generation
    #First three feature maps of the output are for template
    #coordinates,
    #and the last feature map is the 0.0->1.0 body mask
    heatmapGen = na.generalizedHeatmapGen(F, L, B, in_channels=3, out_channels=4, convLayerFunc=depth_convolution.dense_tshape_dconv_quant_scale_layer, internalsFunc=na.nonResidHeatmapGenInternals)
    '''
    layers = 40
    def cap_fn(layer_ind, scale_ind):
        if (scale_ind == 0):
            #8x8 feature cascade is allowed to be very dense (up to 16 past layers' whole output)
            return 256
        if (scale_ind == 1):
            #Ditto for 16x16, but less so (up to 8 past layers' whole output)
            return 128
        if (scale_ind == 2):
            #Make 32x32 same as 64x64 size
            return 64
        if (scale_ind == 3):
            #64x64 size
            return 64
        if (scale_ind == 4):
            #128x128 size is dramatically reduced
            return 16

    num_per_conv_outputs = 16

    heatmapGen = rad_dense.radicalDenseNet(layers, cap_fn, num_per_conv_outputs)
    '''

    #Create a bunch of parameters for the heatmap generator network
    heatmapGen_params = heatmapGen.param_generator("RGBheatmap")

    #Evaluation network unchanged for entire duration, lel
    namePrefix = "RGBToCoord"

    if (quantize):
        net_eval_in = tf.placeholder(tf.float32, [1, 256, 256, 3], name=(namePrefix + "In"))
        net_eval_out = heatmapGen.operation(net_eval_in, heatmapGen_params, quantize=True)
        net_eval_out_named = tf.identity(net_eval_out[0], name=(namePrefix + "Out"))

    #Lel, a funny kludge to be able to initialize the RMSProp vars we don't initialize the first time around
    unquantized_variable_set = None

    train_losses = []
    with tf.Session() as sess:
        regular_train_handle = sess.run(train_iterator.string_handle())
        regular_valid_handle = sess.run(valid_iterator.string_handle())

        background_train_handle = sess.run(train_background_iterator.string_handle())
        background_valid_handle = sess.run(valid_background_iterator.string_handle())

        coco_train_handle = sess.run(train_coco_iterator.string_handle())
        coco_valid_handle = sess.run(valid_coco_iterator.string_handle())

        train_feed_dict = {regular_handle : regular_train_handle, 
                           background_handle : background_train_handle,
                           coco_handle : coco_train_handle}
        valid_feed_dict = {regular_handle : regular_valid_handle,
                           background_handle : background_valid_handle,
                           coco_handle : coco_valid_handle}



        saver = tf.train.Saver()
        batchNum = 0
        start = time.time()

        #First, create the graph differently if we're quantizing or if we're not

        #NETWORK SET-UP
            
        heatmapGenOp = lambda x : heatmapGen.operation(x, heatmapGen_params, quantize=quantize)

        coco_out = rescale_output(heatmapGenOp(coco_rgb))

        #Coco factor for the loss
        coco_loss = losses.coco_loss(coco_out, coco_masks, coco_positions, coco_weights)


        #OpenImages factor of the loss
        #train_openimages_out = rescale_output(heatmapGenOp(train_openimages_rgb))
        #valid_openimages_out = rescale_output(heatmapGenOp(valid_openimages_rgb))

        #train_openimages_loss = losses.openimages_loss(train_openimages_out, train_openimages_parts)
        #valid_openimages_loss = losses.openimages_loss(valid_openimages_out, valid_openimages_parts)


        net_out = rescale_output(heatmapGenOp(regular_in))

        #Is this loss?
        with tf.name_scope('loss'):
            our_dataset_loss = losses.downsampling_masked_mabs_loss_rgb(net_out, regular_expected, regular_mask)
            loss = our_dataset_loss + coco_loss # + train_openimages_loss

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(nn.get_l2_reg(), reg_variables)
        loss_to_min = loss + reg_term


        #Optimization
        with tf.name_scope('optimizer'):
            train_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, 
                                 momentum=momentum_multiplier, epsilon=1e-8).minimize(loss, global_step)
            #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_to_min, global_step)

        opt_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="optimizer")
        opt_vars_init = tf.variables_initializer(opt_vars_list)
        sess.run(opt_vars_init)

        print("Initialization complete. Beginning training.")

        print("Tensors Flowing!")

        if (not quantize):
            #Only initialize this crap here if we're in the first stage (unquantized)
            sess.run(tf.global_variables_initializer())
            batchNum = 0
            start = time.time()
            total_num_iters = num_training_iters
            if (restore):
                floatNamePrefix = "RGBToCoordFloat"
                saver.restore(sess, "./" + floatNamePrefix + "/" + floatNamePrefix)
        else:
            total_num_iters = num_quantization_iters
            batchNum = 0
            start = time.time()
            sess.run(tf.global_variables_initializer())
            floatNamePrefix = "RGBToCoordFloat"
            saver.restore(sess, "./" + floatNamePrefix + "/" + floatNamePrefix)

        print("Adding check op")
        #check_op = tf.add_check_numerics_ops()
        print("Added check op")

        #Okay, train in whatever stage we're in (unquantized/quantized)
        for i in range(total_num_iters):
            loss_val = 0
            
            options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
            run_metadata = tf.RunMetadata()
            #That oughtta do it! Run a training step!
            if (i == 30):
                #Get the bytes in use
                bytes_in_use = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
                print("Max bytes in use", bytes_in_use)

                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_val = sess.run([train_step, loss], feed_dict=train_feed_dict,
                                       options=options, run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_01.json', 'w') as f:
                    f.write(chrome_trace)

            else:
                _, loss_val = sess.run([train_step, loss], feed_dict=train_feed_dict,
                                           options=options, run_metadata=run_metadata)

            #Convert the MSE to RMSE
            loss_val = math.sqrt(loss_val) * 1000.0

            batchNum += 1

            train_losses.append(loss_val)

            if (batchNum % 100 == 0):
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()

                #Compute a quick guesstimate of validation set loss based on 10 averaged runs
                valid_losses = []
                for _ in range(20):
                    loss_val = sess.run(loss, feed_dict=valid_feed_dict)
                    loss_val = math.sqrt(loss_val) * 1000.0
                    valid_losses.append(loss_val)

                valid_loss_avg = sum(valid_losses) / len(valid_losses)
                print("Validation loss: " + str(valid_loss_avg))


                #Input, output and expected sample for the validation set
                valid_coco_rgb_samp, valid_coco_out_samp, valid_coco_positions_samp, valid_img_samp, valid_out_samp, valid_expected_samp = sess.run([coco_rgb, coco_out, coco_positions, regular_in, net_out, regular_expected],
                                                                feed_dict=valid_feed_dict, options=options, run_metadata=run_metadata)

                #valid_openimages_rgb_samp *= 1.0 / (255.0 * input_fac)
                valid_coco_rgb_samp *= 1.0 / (255.0 * input_fac)
                
                debugging.rgb_trainer_visualization_without_openimages("Training", valid_img_samp, valid_out_samp, valid_expected_samp, valid_coco_rgb_samp, valid_coco_out_samp, valid_coco_positions_samp)

                #TODO: Fix up visualization!
                '''
                #Display a visual of the actual/expected heatmaps
                #On the training set
                debugging.actual_expected_visualization("Training", img, out, expected)
                #On the validation set
                debugging.actual_expected_visualization("Validation", valid_img,
                                                        valid_out,
                                                        valid_expected)
                '''

            

                #Extract the average of the last hundred loss values
                avg_hundred_loss = sum(train_losses) / len(train_losses)
                train_losses = []

                #Double the x-dimension of the plot whenever we exceed the plot bounds
                if (len(train_loss_xdata) > axes_xlimit): 
                    axes_xlimit *= 2
                    axes.set_xlim(0, axes_xlimit)

                #Update the training and validation loss plots
                train_loss_xdata.append(len(train_loss_xdata))
                train_loss_ydata.append(avg_hundred_loss)
                train_line.set_xdata(train_loss_xdata)
                train_line.set_ydata(train_loss_ydata)

                valid_loss_xdata.append(len(valid_loss_xdata))
                valid_loss_ydata.append(valid_loss_avg)
                valid_line.set_xdata(valid_loss_xdata)
                valid_line.set_ydata(valid_loss_ydata)

                plt.draw()
                plt.pause(1e-17)

            print("\n Batch#: ", batchNum, " quantized? ", quantize)
            print("Loss: ", loss_val)
            print("Batches per second: ", batchNum / (time.time() - start))

            #Every 6000 batches, save a checkpoint
            if ((batchNum % 6000 == 5998) and (not quantize)):
                floatNamePrefix = "RGBToCoordFloat"
                saver.save(sess, "./" + floatNamePrefix + "/" + floatNamePrefix)
        #Save graph out
        if (not quantize):
            floatNamePrefix = "RGBToCoordFloat"
            saver.save(sess, "./" + floatNamePrefix + "/" + floatNamePrefix)
        if (quantize):
            saver.save(sess, "./" + namePrefix + "/" + namePrefix)
            #First delete the old saved model
            savedModelDir = "./" + namePrefix + "SavedModel/SavedModel"
            shutil.rmtree(savedModelDir)
            #Save SavedModel out
            tf.saved_model.simple_save(sess, "./" + namePrefix + "SavedModel/SavedModel", 
                                       inputs={namePrefix + "In" : net_eval_in}, outputs={namePrefix + "Out" : net_eval_out_named})
     
run_tensorflow(False, True) #True)
