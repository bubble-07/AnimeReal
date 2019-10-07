from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import panutils
import os
import camera
import cv2
import json
import numpy as np
import random
import math
import mathutils as mu


import parts as partinfo

from params import *

#This module is responsible for loading from the handle format
#to the cache data format


#Python function (to be wrapped as a tensorflow op)
#which, when given the contents of an annotation file and the camera
#parameters, returns an annotation list as expected
#from the cache format
#As a note, the cache format's annotations have a translated coordinate system
#so that the very center of the image is (0,0) in the x-y plane
def transform_annotation(anno_file, cam_parameters):
    #This stuff right here is mostly pulled from
#https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/python/example.ipynb
    #When you switch to your own dataset,
    #this stuff has to go
    cam = camera.Camera.from_flat_rep(cam_parameters)
    jsonfile = json.loads(anno_file)
    bodies = []
    for body in jsonfile['bodies']:
        skel = np.array(body['joints15']).reshape((-1,4)).transpose()
        #pts contains all projected points of the skeletons
        #in a 3xnum_points numpy array
        pts = panutils.projectPoints(skel[0:3,:], cam.K, cam.R, cam.t,
                                    cam.distCoef)

        #Translate to center x and y coordinates
        pts[0] -= 320
        pts[1] -= 240

        #Extract confidence levels for each point, too
        conf = skel[3, :]
        #Great, now construct a num_pointsx4 annotation list
        annotation_list = np.vstack([pts, conf]).transpose().astype(np.float32)
        bodies.append(annotation_list)
    if (len(bodies) == 0):
        #There's nobody in the frame!
        return np.empty(shape=(0, 15, 4), dtype=np.float32)
    return np.stack(bodies)

#Converts from a single dataset entry in the
#handle format to a single dataset entry in the cache format
def handle_to_cache(img_filepath, anno_filepath, cam_parameters):
    img_file = tf.read_file(img_filepath)
    #Now, we need to load and centrally crop the image to 480x480
    img = tf.image.decode_and_crop_jpeg(img_file, [0, 80, 480, 480], channels=3)
    img = tf.to_float(img)

    #Now, go ahead and load the annotation file, too
    anno_file = tf.read_file(anno_filepath)

    #For now, reading the annotation file and transforming the
    #data happens straight through a Python function.
    #Since we're currently dealing with a dataset which fits
    #entirely into memory, and we cache it later on,
    #there is no performance degradation aside from the
    #program's startup due to Python's GIL
    #Everything in the pipeline
    #past the cache format is done in Tensorflow,
    #so we get nice C-backed multithreading for the data augmentation
    #and GPU data pre-fetching for faster training

    #Nevertheless,
    #TODO: Convert all of the handle -> cache pipeline into
    #something which directly leverages only Tensorflow!
    #(This may require modifying the dataset to use some TFRecord
    #files or similar)
    #This will also be very practical when the dataset grows too large
    #to fit into memory anymore and/or requires networked storage,
    #since Tensorflow has good facilities for handling that

    #Obtain the transformed annotations
    xformed_annos = tf.py_func(transform_annotation, [anno_file, cam_parameters], tf.float32,
               stateful=False, name='anno_xform')


    return img, xformed_annos

