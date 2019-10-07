#A utility responsible for training and saving out a tensorflow
#model which takes depth images (of size 424x512, float32)
#and returns a (424x512x3, float32) tensor of 3d template positions

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
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

from DepthTrainerParams import *
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
    'depthImage': tf.FixedLenFeature([], tf.string, default_value=''),
    'templateIndexImage': tf.FixedLenFeature([], tf.string, default_value='')
}

num_standardBody_points = StandardBody.pointArray.shape[0]

standardBodyArray = np.copy(StandardBody.pointArray)
standardBodyArray = np.vstack((standardBodyArray, np.array([[0.0, 0.0, 0.0]], dtype=np.float32)))

standardBodyConst = tf.constant(standardBodyArray)

def parse_example_to_sample_label(proto):
    parsed_dict = tf.parse_single_example(proto, feature_description)
    depth_image_bytes = parsed_dict['depthImage']
    template_index_image_bytes = parsed_dict['templateIndexImage']
    #Interpret the depth image bytes as a float32
    depth_image_flat = tf.decode_raw(depth_image_bytes, tf.float32)
    template_index_image_flat = tf.decode_raw(template_index_image_bytes, tf.uint16)
    template_index_image_flat = tf.cast(template_index_image_flat, tf.int32)
    template_mask_flat = template_index_image_flat < num_standardBody_points

    template_image_flat = tf.gather(standardBodyConst, template_index_image_flat)

    template_positions_flat = template_index_image_flat

    #Derive the template image through indexing

    depth_image = tf.reshape(depth_image_flat, [424, 512, 1])
    template_image = tf.reshape(template_image_flat, [424, 512, 3])
    template_mask = tf.reshape(template_mask_flat, [424, 512, 1])

    return (depth_image, template_image, template_mask)

index_arrays = np.indices((424, 512)).astype(np.float32)
y_sweep_array = index_arrays[0]
x_sweep_array = index_arrays[1]

y_sweep_array = np.reshape((y_sweep_array - 212.0) / 212.0, (424, 512, 1))
x_sweep_array = np.reshape((x_sweep_array - 256.0) / 256.0, (424, 512, 1))


def augment_example(depth_image, template_image, template_mask):

    #Before doing anything else, apply distortions that only happen to the depth image
    #These include speckle noise, point omission noise, and small-angle (approx linear)
    #depth adjustments

    #First, do small-angle adjustments
    if (AUGMENT_SMALL_ANGLE):
        #TODO: Should we use a radially uniform distribution instead?
        x_f = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_SMALL_ANGLE_MAX_MM
        y_f = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_SMALL_ANGLE_MAX_MM

        depth_image += x_f * x_sweep_array
        depth_image += y_f * y_sweep_array
    if (AUGMENT_SMALL_SCALE):
        delta_z = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_SMALL_SCALE_MAX_DELTA_Z
        depth_image += delta_z
        depth_image = tf.maximum(0.0, depth_image)
    if (AUGMENT_GAUSS_NOISE):
        noise = tf.random_normal([424, 512, 1], stddev=AUGMENT_GAUSS_NOISE_STDEV)
        depth_image += noise
    if (AUGMENT_OMIT_NOISE):
        roll = tf.random_uniform([424, 512, 1], minval=0.0, maxval=1.0) < AUGMENT_OMIT_PROB
        mask = 1.0 - tf.cast(roll, tf.float32)
        depth_image = depth_image * mask

    #This will make things a bit easier, because when we manipulate the
    #depth image, we'll usually wind up modifying the mask, too
    template_mask = tf.cast(template_mask, tf.float32)
    image_mask = tf.concat([depth_image, template_mask], 2)

    #First thing's first -- determine whether or not to mirror the image
    #and adjust template positions accordingly
    if (AUGMENT_FLIP):
        flip_boolean = tf.random_uniform([], minval=-1.0, maxval=1.0) > 0.0

        image_mask = tf.cond(flip_boolean, lambda: image_mask, lambda: tf.image.flip_left_right(image_mask))
        template_image = tf.cond(flip_boolean, lambda: template_image, lambda: tf.image.flip_left_right(template_image * np.array([-1.0, 1.0, 1.0])))
    #Okay, now we've dealt with flips. 
    #Okay, now deal with rotations, uniform scaling,
    #translations, and non-uniform scaling
    translate_x = 0.0
    translate_y = 0.0

    rotate = 0.0
    uniform_scale = 1.0

    x_fac = 1.0
    y_fac = 1.0

    x_center = 256
    y_center = 212

    if (AUGMENT_TRANSLATE):
        translate_x = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_TRANSLATE_X_PIX
        translate_y = tf.random_uniform([], minval=-1.0, maxval=1.0) * AUGMENT_TRANSLATE_Y_PIX
    if (AUGMENT_ROTATE):
        rotate = tf.random_normal([], stddev=AUGMENT_ROTATE_ANGLE_STDEV)
    if (AUGMENT_UNIFORM_SCALE):
        uniform_scale = tf.random_uniform([], minval=AUGMENT_MIN_UNIFORM_SCALE, maxval=AUGMENT_MAX_UNIFORM_SCALE)
    if (AUGMENT_ASPECT_RATIO):
        x_fac = tf.random_uniform([], minval=AUGMENT_ASPECT_X_MIN, maxval=AUGMENT_ASPECT_X_MAX)
        y_fac = tf.random_uniform([], minval=AUGMENT_ASPECT_Y_MIN, maxval=AUGMENT_ASPECT_Y_MAX)

    aspect_transform = augmentation.get_aspect_ratio_transform(x_fac, y_fac, x_center, y_center)
    rot_scale_xlate_transform = augmentation.get_affine_transform(1.0, rotate, uniform_scale, translate_x, translate_y, x_center, y_center)
    total_transform = tf.contrib.image.compose_transforms(aspect_transform, rot_scale_xlate_transform)

    #Great, now stack all of the things together, and apply the transform!
    all_together = tf.concat([image_mask, template_image], 2)
    all_transformed = tf.contrib.image.transform(all_together, total_transform)

    #Okay, great. Now, unpack all of that back to the original parameters
    depth_image = tf.reshape(all_transformed[:, :, 0], [424, 512, 1])
    template_mask = tf.reshape(all_transformed[:, :, 1], [424, 512, 1])
    template_image = tf.reshape(all_transformed[:, :, 2:], [424, 512, 3])

    #Okay, now final clean-up.
    #Apply the uniform scaling to the depth image
    depth_image *= uniform_scale
    #Make the template mask binary again
    template_mask = template_mask > 0.5
    
    return (depth_image, template_image, template_mask)
    


def build_dataset_from_dir(tfrecordRoot, batch_size):
    tfrecordFiles = [y for x in os.walk(tfrecordRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]
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
    
def main():

    TRAINING_DATASET_DIR = sys.argv[1]
    VALIDATION_DATASET_DIR = sys.argv[2]

    global axes_xlimit

    print("Prepping everything...")
        #TODO: Split into train/test sets, as well!
    #Should that happen at the data level? (Yeah, probably)

    #DATASET PIPELINE SETUP
    
    #Create a dataset for training
    train_set = build_dataset_from_dir(TRAINING_DATASET_DIR, batch_size)
    #Prefetching will only happen on the training set
    #TODO: Is this a good thing?
    #train_set = train_set.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

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
    heatmapGen = na.depthTemplateMapGen(F, L, B)

    #Create a bunch of parameters for the heatmap generator network
    heatmapGen_params = heatmapGen.param_generator("heatmap")

    #Default in/expected/out for the network __while training__
    train_in, train_expected, train_mask = train_iterator.get_next()

    train_in = tf.image.resize_image_with_pad(train_in, 512, 512)
    train_in = train_in / 10.0

    net_out = map(lambda x : 10.0 * x, heatmapGen.operation(train_in, heatmapGen_params))

    namePrefix = "DepthToCoord"

    net_eval_in = tf.placeholder(tf.float32, [batch_size, 512, 512, 1], name=(namePrefix + "In"))
    net_eval_in = net_eval_in / 10.0
    net_eval_out = map(lambda x : 10.0 * x, heatmapGen.operation(net_eval_in, heatmapGen_params))
    net_eval_out_named = tf.identity(net_eval_out[0], name=(namePrefix + "Out"))

    #Also build a pipeline for validation inputs/outputs/expected
    valid_in, valid_expected, valid_mask = valid_iterator.get_next()

    valid_in = tf.image.resize_image_with_pad(valid_in, 512, 512)
    valid_in = valid_in / 10.0

    valid_net_out = map(lambda x: x * 10.0, heatmapGen.operation(valid_in, heatmapGen_params))
    valid_net_loss = losses.downsampling_masked_mabs_loss(valid_net_out, valid_expected, valid_mask)


    global_step = tf.Variable(0, trainable=False) 

    #Is this loss?
    with tf.name_scope('loss'):
        loss = losses.downsampling_masked_mabs_loss(net_out, train_expected, train_mask)

    learning_rate = tf.train.exponential_decay(init_training_rate, global_step, 
                                               decay_steps=decay_steps, decay_rate=decay_rate) 

    #Optimization
    with tf.name_scope('optimizer'):
        train_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, 
                             momentum=momentum_multiplier, epsilon=1e-8).minimize(loss, global_step)
#        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    print("Initialization complete. Beginning training.")

    print("Tensors Flowing!")

    #TODO: Remove! We have global_step now
    batchNum = 0

    train_losses = []
    with tf.Session() as sess:

        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        start = time.time()

        for i in range(num_training_iters):
            loss_val = 0
            if (i == 300):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_val = sess.run([train_step, loss], options=options, run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_01.json', 'w') as f:
                    f.write(chrome_trace)

            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()
                _, loss_val = sess.run([train_step, loss],
                                       options=options, run_metadata=run_metadata)

            #Convert the MSE to RMSE
            loss_val = math.sqrt(loss_val)

            batchNum += 1

            train_losses.append(loss_val)

            if (batchNum % 100 == 0):
                options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                run_metadata = tf.RunMetadata()
                #Input, output, and expected sample for training set
                img, out, expected = sess.run([train_in, net_out, train_expected],
                                          options=options, run_metadata=run_metadata)


                #Compute a quick guesstimate of validation set loss based on 10 averaged runs
                valid_losses = []
                for _ in range(20):
                    loss_val = sess.run(valid_net_loss)
                    valid_losses.append(math.sqrt(loss_val))

                valid_loss_avg = sum(valid_losses) / len(valid_losses)
                print("Validation loss: " + str(valid_loss_avg))

                #Input, output and expected sample for the validation set
                valid_img_samp, valid_out_samp, valid_expected_samp = sess.run([valid_in, valid_net_out, valid_expected],
                                                                options=options, run_metadata=run_metadata)

                
                debugging.actual_expected_dense_visualization("Training", valid_out_samp, valid_expected_samp)

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

            print("\n Batch#: ", batchNum)
            print("Loss: ", loss_val)
            print("Batches per second: ", batchNum / (time.time() - start))
        saver.save(sess, "./" + namePrefix + "/" + namePrefix)
 
main()
