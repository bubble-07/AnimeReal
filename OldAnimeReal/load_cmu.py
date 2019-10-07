from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from camera import Camera

import json
import tensorflow as tf

#Module responsible for loading file handles from CMU-like datasets

#From a filename of the form "body3DScene_{frame number}.json" as in the CMU dataset,
#this gets the frame number
def get_frame_number_from_json_filename(fname):
    return fname[12:-5]

def get_frame_number_from_jpg_filename(fname):
    return fname[6:-4]

#Given a directory for a CMU-like dataset, this yields a list of tuples
#in the file handle format
#TODO: Print progress or something?
def load_handles(DATASET_DIR):
    #First, establish a list of paths to each sequence in the dataset directory
    sequenceNames = os.listdir(DATASET_DIR)

    #Parallel list of paths to each sequence
    sequencePaths = map(lambda y: os.path.join(DATASET_DIR, y), os.listdir(DATASET_DIR))

    #Then, for each sequence, 
    #look into the "vgaImgs" directory to figure out which cameras we'll care about
    cameraNames = map(lambda y: os.listdir(os.path.join(y, "vgaImgs")), sequencePaths)

    cameras = []

    #Most of the data-reading code shamelessly taken from
    #https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/python/example.ipynb
    #Remove once you have your own dataset
    #With the cameras names in hand, load the calibration_{sequence}.json and find
    #camera parameters for each
    for sequenceName, cameraList in zip(sequenceNames, cameraNames):
        with open(os.path.join(DATASET_DIR, sequenceName + '/calibration_{0}.json'.format(sequenceName))) as calib_file:
            calib = json.load(calib_file)
        #Dictionary of all cameras for the sequence, mapping from their names to their other properties
        seq_cams = {cam['name']:cam for cam in calib['cameras']}
        #Filter out to only those cams for which we have VGA images, and for each one, find their parameters
        cam_props = [seq_cams[cam] for cam in cameraList]
        seq_cam_objs = map(lambda cam: Camera([np.array(cam['K']), np.array(cam['distCoef']), 
                                                 np.array(cam['R']), np.array(cam['t'])])
                            , cam_props)
        cameras.append(seq_cam_objs)

    #This list of filepath tuples is in the format expected by load_cache.handle_to_cache
    filepath_tuples = []

    #Okay, great, now that we have the sequences and their cameras all listed out, we need to come up with
    #image_filepath, annotation_filepath, cam_parameter triples. To do this, we'll loop over the sequences,
    #cameras within each sequence, and timepoints within each sequence.
    for sequenceName, sequencePath, cameraNames, cameraList in zip(sequenceNames, sequencePaths, cameraNames, cameras):
        jsonDir = os.path.join(sequencePath, "vgaPose3d_stage1/")
        jsonFiles = os.listdir(jsonDir)

        #Find the collection of all frame numbers for which there is keypoint data
        keypoint_frames = map(get_frame_number_from_json_filename, jsonFiles)

        for cameraName, camera in zip(cameraNames, cameraList):
            imgDir = os.path.join(sequencePath, "vgaImgs/", cameraName)
            imgFiles = os.listdir(imgDir)

            #Now, find the collection of all frame numbers for which there are images
            image_frames = map(get_frame_number_from_jpg_filename, imgFiles)

            #Valid frames are those for which there's an image and keypoint data
            valid_frames = set(keypoint_frames).intersection(set(image_frames))

            for frameName in valid_frames:
                img_filepath = os.path.join(imgDir, cameraName + "_" + frameName + ".jpg")
                anno_filepath = os.path.join(jsonDir, "body3DScene_" + frameName + ".json")
                cam_parameters = camera.to_flat_rep()
                filepath_tuples.append((img_filepath, anno_filepath, cam_parameters))
                
    return filepath_tuples


