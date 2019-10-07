from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import neural_ops as nn
#import parts

#This operation splits off the first num number of features, and returns the num_len_part
def split_off(x, num):
    S = x.get_shape()
    H, W, C = nn.height_width_channels(x)
    extra = C - num
    num_len_part, remainder = tf.split(x, [num, extra], axis=len(S)-1)
    return num_len_part

def split_off_half(x):
    first, second = tf.split(x, 2, axis=3)
    return first

#FIXME: Reinstate? Maybe useless, lell
'''
def split_field_maps_off(x):
    return split_off(x, parts.num_field_maps)

def out_conv1x1(x):
    return nn.lin_conv1x1(x, parts.num_field_maps)
'''

def concat(L):
    S = L[0].get_shape()
    return tf.concat(L, len(S) - 1)

#Separates a tensor into a list of single-feature-map tensors
def unconcat(T):
    feat_maps = T.get_shape()[-1]
    result = tf.split(T, num_or_size_splits=feat_maps, axis=-1)
    return list(result)
    
