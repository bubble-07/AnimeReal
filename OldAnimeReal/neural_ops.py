from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structuring as ns
import tensorflow as tf

activ = tf.nn.leaky_relu

reg = 0.0000

#This file contains definitions for basic neural network operations,
#but with a different convention from the native Tensorflow ops
#Here, every network primitive comes from a function which takes no
#arguments, but returns the pair of a
#-Function taking (in, params) pairs and returning an output
#-Function taking a name prefix for variables and returning params
#Where here, params can be anything that captures the state of all
#tf.Variables needed to evaluate the network primitive

class Neural_Op:
    def __init__(self, operation, param_generator):
        self.operation = operation
        self.param_generator = param_generator
    #Create a neural op whose underlying operation does this one's operation
    #and then does the other one (composition). The parameters here become
    #a two
    def then(self, other_op):
        def new_param_generator(name):
            return [self.param_generator(name + "L"), other_op.param_generator(name + "R")]
        def new_op(x, params):
            my_params, other_params = params
            return other_op.operation(self.operation(x, my_params), other_params)
        return Neural_Op(new_op, new_param_generator)

    #Just like "then", but passing the output directly to a TF function without any special naming
    def then_apply(self, fn):
        def new_op(x, params):
            return fn(self.operation(x, params))
        return Neural_Op(new_op, self.param_generator)

    #Changes the output to return a singleton list
    def to_singleton_list(self):
        return self.then(to_neural_op(lambda x: [x]))

    #Yields a new neural op which applies the op by mapping over _lists_ of inputs
    #The parameters are shared by each invocation of the op
    def map_on_lists(self):
        return Neural_Op(lambda lst, params: map(lambda x: self.operation(x, params), lst),
                         self.param_generator)

    #Adds extra identifying information to the names of parameters for this op
    def add_identification(self, prefix):
        return Neural_Op(self.operation, lambda suffix: self.param_generator(prefix + "_" + suffix))

def to_neural_op(fn):
    return Neural_Op(lambda x, params: fn(x), no_params)

#Convenience function for Neural_Ops which have no parameters
def no_params(name):
    return []

def identity():
    return to_neural_op(lambda x : x)

#An op which extracts a particular index from a list-valued input
def get_index(ind):
    return Neural_Op(lambda lst, params: lst[ind], no_params)

#An op which takes a list-valued input and sums all of the entries
def reduce_sum():
    return Neural_Op(lambda lst, params: sum(lst), no_params)

#Given a list of Neural_Ops [o1, o2, o3, o4, ...], this creates a single neural op
#which takes in parallel lists of inputs and parameters to each neural op and returns a list of results
def parallel(*ops):
    def new_op(xs, params):
        result = []
        for x, op, param in zip(xs, ops, params):
            result.append(op.operation(x, param))
        return result
    def new_param_generator(name):
        result = []
        number = 0
        for op in ops:
            number += 1
            result.append(op.param_generator(name + "A" + str(number) + "Z"))
        return result
    return Neural_Op(new_op, new_param_generator)

#Like parallel, but where each op is given by op_generator, with N total parallel ops
def replicate_over_list(op_generator, N):
    ops = [op_generator() for _ in range(N)]
    return parallel(*ops)

#Function which, when given a no-argument function returning a neural op
#and a limit N, yields a neural op which 
#computes the Nth iterate of copies of that neural op (parameters NOT shared between copies!)
def iterate_op(op_generator, N):
    ops = []
    for i in range(N):
        ops.append(op_generator())
    def new_op(x, params):
        temp = x
        for op, param in zip(ops, params):
            temp = op.operation(temp, param)
        return temp
    def new_param_generator(name):
        layerNum = 0
        result = []
        for op in ops:
            layerNum += 1
            result.append(op.param_generator(name + "_layer" + str(layerNum)))
        return result
    return Neural_Op(new_op, new_param_generator)
    
#Neural op which splits a list in two at the given index
def unsplice(ind):
    return to_neural_op(lambda L: [L[:ind], L[ind:]])

#Neural op which concatenates the lists in a list of lists back together
def splice():
    return to_neural_op(lambda L: [item for sublist in L for item in sublist])

#Neural op which appends an element to a list
def append():
    return to_neural_op(lambda L: L[0] + [L[1]])

#Neural op which generates a specified number of copies of its input
def repeat(N):
    return to_neural_op(lambda x: [x] * N)

#Neural op which takes a list-valued input and concatenates all of the feature maps in it
def stack_features():
    return to_neural_op(ns.concat)

#Neural op which takes an input x and a list of ops [o1, o2, o3, o4, ...] and
#Returns a list [o1(x), o2(x), o3(x), ...]
def split(*ops):
    return repeat(len(ops)).then(parallel(*ops)) 

#Batch normalization op
#TODO: Do we need to keep track of parameters here?
def batch_norm():
    return to_neural_op(tf.contrib.layers.batch_norm)
                    

#Default 2x upscaling and downscaling ops
def upscale():
    return upscale2x()

def downscale():
    return max_pool2x2()

#cxc convolution operator straight up (no bias term)
def lin_conv(c, in_chan, out_chan, stride=1):
    return Neural_Op(lambda x, W: lin_conv_helper(x, W, stride),
                     lambda name: get_W_conv(name + "-convC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "I",
                                            c, in_chan, out_chan))

#TODO: Refactor to be just a lin conv + bias!
#cxc convolution operator with the given number of in_channels and out_channels
#without passing through any activation function
def affine_conv(c, in_chan, out_chan, stride=1):
    return Neural_Op(lambda x, params: affine_conv_helper(x, params, stride),
           lambda name: get_W_b_conv(name + "-convC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "I", 
                                    c, in_chan, out_chan))
#cxc convolution operator with the given number of in_channels and out_channels
#followed up by our choice of activation function
def conv(c, in_chan, out_chan, stride=1):
    return affine_conv(c, in_chan, out_chan, stride).then_apply(activ)

#TODO: Test how your network performs with these!
#Linear separable convolution with the given kernel size (cxc), input channels, output channels,
#channel multiplier (default: 1) and spatial stride (default: 1)
def affine_sconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    return Neural_Op(lambda x, params: affine_sconv_helper(x, params, stride),
            lambda name: get_W_b_sconv(name + "-sconvC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "c" + str(chan_mult) + "I",
                                       c, in_chan, out_chan, chan_mult))

def sconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    return affine_sconv(c, in_chan, out_chan, chan_mult, stride).then_apply(activ)


def avg_pool2x2():
    return to_neural_op(lambda x: tf.contrib.layers.avg_pool2d(x, 2, padding='SAME'))
def max_pool2x2():
    return to_neural_op(lambda x: tf.contrib.layers.max_pool2d(x, 2, padding='SAME'))

def upscale2x():
    return to_neural_op(upscale2xhelper)

def upscale2xhelper(x):
    H, W, C = height_width_channels(x)
    return tf.image.resize_images(x, [H * 2, W * 2])

def upscale4x(x):
    return upscale2x().then(upscale2x())
def weight_initializer():
    return tf.contrib.layers.xavier_initializer()

def bias_initializer():
    return tf.zeros_initializer()

def get_W(name, shape):
    return tf.get_variable(name, shape, initializer=weight_initializer())
def get_b(name, shape):
    return tf.get_variable(name, shape, initializer=bias_initializer())

def get_W_conv(name, c, in_chan, out_chan):
    return get_W(name + "_W", [c, c, in_chan, out_chan])

def get_b_conv(name, out_chan):
    return get_b(name + "_b", [out_chan])

def get_W_b_conv(name, c, in_chan, out_chan):
    return [get_W_conv(name, c, in_chan, out_chan), get_b_conv(name, out_chan)]

def get_W_b_sconv(name, c, in_chan, out_chan, chan_mult):
    return [get_W(name + "_W_d", [c, c, in_chan, chan_mult]), get_W(name + "_W_p", [1, 1, in_chan * chan_mult, out_chan]),
            get_b_conv(name, out_chan)]

#Adds a per-feature-map bias
def add_bias(F):
    return Neural_Op(lambda x, b: x + b, lambda name: get_b_conv(name, F))

def lin_conv_helper(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def affine_conv_helper(x, params, stride=1):
    W, b = params
    return lin_conv_helper(x, W, stride) + b

def affine_sconv_helper(x, params, stride=1):
    W_d, W_p, b = params
    return tf.nn.separable_conv2d(x, W_d, W_p, strides=[1, stride, stride, 1], padding='SAME') + b

def batch_height_width_channels(x):
    S = x.get_shape().as_list()
    H = S[-3]
    W = S[-2]
    C = S[-1]
    B = S[-4]
    return (B, H, W, C)

def height_width_channels(x):
    S = x.get_shape().as_list()
    H = S[-3]
    W = S[-2]
    C = S[-1]
    return (H, W, C)
