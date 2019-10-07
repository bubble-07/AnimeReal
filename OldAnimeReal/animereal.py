from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import load_cache
import loss as losses
import cv2
import time
import os
import numpy as np
import load_cmu
import debugging

import augmentation as aug
import heatmap_gen

import camera
import parts

import matplotlib.pyplot as plt
import random


from tensorflow.python.client import timeline
import neural_architecture as na

import neural_structuring as ns

import neural_ops as nn
import mathutils as mu

import SharedArray as sa

import xml.etree.ElementTree as ET

import gc
import argparse
import sys
import tempfile
import random
import math

import panutils

from multiprocessing import Process
from multiprocessing import Queue

from random import shuffle

import json
import scipy as sp
from scipy.stats import norm
import tensorflow as tf

from params import *

#Set up training loss plot
#TODO: Move to its own file
train_loss_xdata = []
train_loss_ydata = []
valid_loss_xdata = []
valid_loss_ydata = []
plt.show()
axes = plt.gca()
axes_xlimit = 10
axes.set_xlim(0, axes_xlimit)
axes.set_ylim(0, 0.25)
train_line, = axes.plot(train_loss_xdata, train_loss_ydata, 'r-', label='training loss')
valid_line, = axes.plot(valid_loss_xdata, valid_loss_ydata, 'g-', label='validation loss')


#This file is responsible for handling the setup,
#training, and evaluation of the pose detection CNN

#DATA PIPELINE
#The data pipeline is structured as a sequence of Tensorflow dataset
#transformations from an initial collection of files
#to the final actual/expected tensor pairs

#We assume all training set images are 640x480 landscape VGA jpegs

#Handle format:
#(img_filepaths, anno_filepaths, cam_param_lists) 
#(string tensor, string tensor,  uint64 tensor)
#A handle consists of just the filepaths to the captured image,
#the annotations, and the 4-element camera distortion parameter
#list

#Cache format:
#(480x480x3 images, annotation lists)
#([image format] tensor, annotation tensors)
#An annotation tensor is an ordered 2d tensor of
#(x, y, z, c) for each keypoint (in the keypoint ordering)
#where x and y are x/y coordinates in image space,
#z is the z-distance in real centimeters,
#and c is the confidence of that point's detection (as
#a property of the dataset's collection methodology)
#With one annotation tensor per person in an image,
#these tensors are packed into a 3d tensor
#In summary, an annotation list is of dims (num_bodies, num_points, 4)

#Network supervised example format:
#(256x256x3 images, heatmaps + PAFs)
#These are augmented, resized copies of the data from
#the cache format, but with heatmaps + PAFs generated
#from the cache format's annotation lists

#Future extension to new datasets:
#Inevitably, the underlying dataset for training will change
#The format above will ensure that when this happens,
#the only thing which'll need to change is the Handle -> Cache
#part of the pipeline

#For the implementations of each part of the pipeline,
#see load_cache.py, augmentation.py, and heatmap_gen.py


#Converts from a single dataset entry in the cache
#format to a single dataset entry in the supervised example format
#TODO: Add data augmentation back in, plz
def cache_to_labeled_example(img_file, annotation_tensor):
    #This step is broken down into two separate tasks:
    #1. Augmentation: Given the 480x480 initial image and the corresponding annotation tensor,
    #   perform simultaneous transformations (brightness, contrast, rotation, scale, etc)
    #   to both the image and the annotation tensor for data augmentation.
    #2. Heatmap/PAF generation:
    #   From the augmented image from the previous step
    #   and the (transformed) annotation tensor, generate heatmaps/PAFs 
    #   from a centrally-cropped 256x256 region
    aug_img, aug_annos = aug.cache_to_augment(img_file, annotation_tensor)
    return heatmap_gen.augment_to_labeled_example(aug_img, aug_annos)
#TRAINING, TESTING

def build_dataset_from_dir(data_dir, batch_size):
    #Load in a list of tuples for the handle representation of the datset
    handles = load_cmu.load_handles(data_dir)    
    random.shuffle(handles)


    #Now, transform the handle rep into some parallel tensors
    img_filepaths, anno_filepaths, cam_paramlists = zip(*handles)
    img_filepaths = np.array(img_filepaths)
    anno_filepaths = np.array(anno_filepaths)
    cam_paramlists = np.array(cam_paramlists)

    #Construct the handle dataset
    handle_dataset = tf.data.Dataset.from_tensor_slices((img_filepaths,
                                   anno_filepaths, cam_paramlists))

    #Do a local shuffle on handles
    handle_dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    #Pipeline from the handle dataset to the cache format
    cache_dataset = handle_dataset.map(load_cache.handle_to_cache, 
                       num_parallel_calls=CPU_PARALLEL_THREADS)

    
    cache_dataset = cache_dataset #TODO: Add back in .cache()?

    #Now that we have something in the cache format, we should
    #switch it over to the labeled example format
    labeled_examples = cache_dataset.map(cache_to_labeled_example,
                           num_parallel_calls=CPU_PARALLEL_THREADS)

    #TODO: Should we use parallel interleave to also speed up deserialization of the jsons earlier?
    #TODO: Somehow fuse map and batch here?
    #TODO: Shuffle_and_repeat?

    #Finally, apply batch, repeat and prefetch
    result_set = labeled_examples.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    result_set = result_set.repeat() #.prefetch(10)

    return result_set



def main(_):

    global axes_xlimit

    print("Loading in dataset file handles and camera parameters")
        #TODO: Split into train/test sets, as well!
    #Should that happen at the data level? (Yeah, probably)

    #DATASET PIPELINE SETUP
    
    #Create a dataset for training
    train_set = build_dataset_from_dir(TRAINING_DATASET_DIR, batch_size)
    #Prefetching will only happen on the training set
    train_set = train_set.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

    #Create a dataset for validation
    validation_set = build_dataset_from_dir(VALIDATION_DATASET_DIR, batch_size)
    #TODO: Also add in a evaluation set


    #Okay, great. Now, we train a neural network to generate heatmaps

    
    #INPUT/OUTPUT AND OPTIMIZATION OBJECTIVE

    #TODO: The way this currently works is dumb
    #but there's a bug in tensorflow 1.8.0 which means that you can't
    #prefetch_to_device with reinitializable tensors.
    #Remove this "cond" silliness when that's fixed
    train_iterator = train_set.make_one_shot_iterator()
    valid_iterator = validation_set.make_one_shot_iterator()

    #Build the Neural_Op for heatmap generation
    heatmapGen = na.heatmapGen(F, L, B)

    #Create a bunch of parameters for the heatmap generator network
    heatmapGen_params = heatmapGen.param_generator("heatmap")

    #Default in/expected/out for the network __while training__
    net_in, net_expected = train_iterator.get_next()

    net_out = heatmapGen.operation(net_in, heatmapGen_params)

    #Also build a pipeline for validation inputs/outputs/expected
    valid_net_in, valid_net_expected = valid_iterator.get_next()
    valid_net_out = heatmapGen.operation(valid_net_in, heatmapGen_params)
    valid_net_loss = losses.downsampling_mabs_loss(valid_net_out, valid_net_expected)


    global_step = tf.Variable(0, trainable=False) 

    #Is this loss?
    with tf.name_scope('loss'):
        loss = losses.downsampling_mabs_loss(net_out, net_expected)

    learning_rate = tf.train.exponential_decay(init_training_rate, global_step, 
                                               decay_steps=decay_steps, decay_rate=decay_rate) 

    #Optimization
    with tf.name_scope('optimizer'):
        train_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, 
                             momentum=momentum_multiplier, epsilon=1e-8).minimize(loss, global_step)
#        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    print("Initialization complete. Beginning training.")

    graph_location = tempfile.mkdtemp()


    #Save a copy of the computation graph out
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    gc.collect()

    print("Tensors Flowing!")
    start = time.time()

    #TODO: Remove! We have global_step now
    batchNum = 0

    train_losses = []
    with tf.Session() as sess:

        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())


        for i in range(num_training_iters):
            _, loss_val = sess.run([train_step, loss],
                                   options=options, run_metadata=run_metadata)

            #Convert the MSE to RMSE
            loss_val = math.sqrt(loss_val)

            #Care about the runtime trace only on batch 190, because of the warm-up
            if (batchNum == 190 and False):
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(chrome_trace)
                
            batchNum += 1

            train_losses.append(loss_val)

            if (batchNum % 100 == 0):
                #Input, output, and expected sample for training set
                img, out, expected = sess.run([net_in, net_out, net_expected],
                                          options=options, run_metadata=run_metadata)


                #Compute a quick guesstimate of validation set loss based on 10 averaged runs
                valid_losses = []
                for _ in range(20):
                    loss_val = sess.run(valid_net_loss)
                    valid_losses.append(math.sqrt(loss_val))

                valid_loss_avg = sum(valid_losses) / len(valid_losses)
                print("Validation loss: " + str(valid_loss_avg))

                #Input, output and expected sample for the validation set
                valid_img, valid_out, valid_expected = sess.run([valid_net_in, valid_net_out, valid_net_expected],
                                                                options=options, run_metadata=run_metadata)

                #Display a visual of the actual/expected heatmaps
                #On the training set
                debugging.actual_expected_visualization("Training", img, out, expected)
                #On the validation set
                debugging.actual_expected_visualization("Validation", valid_img,
                                                        valid_out,
                                                        valid_expected)

            

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

            print("\n Batch#: ", batchNum)
            print("Loss: ", loss_val)
            print("Batches per second: ", batchNum / (time.time() - start))

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

