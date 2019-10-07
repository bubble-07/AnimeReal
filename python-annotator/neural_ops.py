from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_structuring as ns
import tensorflow as tf

#Now gonna use relu, lel
activ = tf.nn.relu
#activ = tf.nn.leaky_relu
#activ = tf.nn.relu6

reg = 0.0001 #0.0001

regularizer = tf.contrib.layers.l2_regularizer(scale=reg)

def get_l2_reg():
    return regularizer

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
        def new_op(x, params, quantize=False):
            my_params, other_params = params
            return other_op.operation(self.operation(x, my_params, quantize), other_params, quantize)
        return Neural_Op(new_op, new_param_generator)

    #Just like "then", but passing the output directly to a TF function without any special naming
    def then_apply(self, fn):
        def new_op(x, params, quantize=False):
            return fn(self.operation(x, params, quantize))
        return Neural_Op(new_op, self.param_generator)

    #Changes the output to return a singleton list
    def to_singleton_list(self):
        return self.then(to_neural_op(lambda x: [x]))

    #Yields a new neural op which applies the op by mapping over _lists_ of inputs
    #The parameters are shared by each invocation of the op
    def map_on_lists(self):
        def result_op(lst, params, quantize=False):
            return map(lambda x: self.operation(x, params, quantize), lst)
        return Neural_Op(result_op, self.param_generator)

    #Adds extra identifying information to the names of parameters for this op
    def add_identification(self, prefix):
        return Neural_Op(self.operation, lambda suffix: self.param_generator(prefix + "_" + suffix))

#Returns a neural op which concatenates along the depth dimension
def concat_depth():
    return to_neural_op(lambda L : tf.concat(L, axis=-1))

def get_initialized_scalar(name, init_val):
    initializer = tf.initializers.constant(init_val, dtype=tf.float32)
    result = tf.get_variable(name, [], initializer=initializer)
    return result

def get_unif_init_vector(name, init_val, num_chan):
    initializer = tf.initializers.constant(init_val, dtype=tf.float32)
    result = tf.get_variable(name, [num_chan], initializer=initializer)
    return result

def quantize_range_channels(min_val, max_val, num_chan):
    def result_op(x, params, quantize=False):
        min_var, max_var = params
        if (quantize):
            return tf.quantization.fake_quant_with_min_max_vars_per_channel(x, min_var, max_var, narrow_range=True)
        else:
            return x
    def result_params(name):
        min_var = get_unif_init_vector(name + "quantize_chan_range_min", min_val, num_chan)
        max_var = get_unif_init_vector(name + "quantize_chan_range_max", max_val, num_chan)
        return [min_var, max_var]
    return Neural_Op(result_op, result_params)

def quantize_resid_channels(num_chan):
    return quantize_range(0.0, 12.0)
    #return quantize_range_channels(0.0, 12.0, num_chan)

def quantize_channels(num_chan):
    return quantize_range(0.0, 6.0)
    #TODO: Cryogenically unfreeze this once per channel quantization has a tflite implementation
    #at present, tflite's support for quantization is dismal
    #return quantize_range_channels(0.0, 6.0, num_chan)


def quantize_fixed_range(min_val, max_val):
    def result_op(x, params, quantize=False):
        if (quantize):
            return tf.quantization.fake_quant_with_min_max_args(x, min_val, max_val, narrow_range=True)
        else:
            return x
    return Neural_Op(result_op, no_params)
    

#Returns a neural op which, if "quantize=True" in the operation,
#it will call tf.quantization.fake_quant_with_min_max_vars
#with min_val and max_val initialized to the given values
def quantize_range(min_val, max_val):
    def result_op(x, params, quantize=False):
        min_var, max_var = params
        if (quantize):
            return tf.quantization.fake_quant_with_min_max_vars(x, min_var, max_var, narrow_range=True)
        else:
            return x
    def result_params(name):
        min_var = get_initialized_scalar(name + "quantize_range_min", min_val)
        max_var = get_initialized_scalar(name + "quantize_range_max", max_val)
        return [min_var, max_var]
    return Neural_Op(result_op, result_params)

def activ_fn_relu6():
    def result_op(x, params, quantize=False):
        if (not quantize):
            eps_max = 0.01
            eps = tf.random.uniform([], minval=0.0, maxval=eps_max, dtype=tf.float32)
            return tf.maximum(eps * x, tf.minimum(6.0 * (1.0 - eps) + eps * x, x))
        else:
            return tf.nn.relu6(x)
    return Neural_Op(result_op, no_params)

def activ_fn():
    def result_op(x, params, quantize=False):
        if (not quantize):
            eps_max = 0.01
            eps = tf.random.uniform([], minval=0.0, maxval=eps_max, dtype=tf.float32)
            return tf.nn.leaky_relu(x, alpha=eps)
        else:
            #otherwise, just use relu
            return activ(x)

    return Neural_Op(result_op, no_params)

def to_neural_op(fn):
    def result_op(x, params, quantize=False):
        return fn(x)
    return Neural_Op(result_op, no_params)

#Convenience function for Neural_Ops which have no parameters
def no_params(name):
    return []

def last_element_singleton_helper(L):
    if (len(L) == 0):
        return L
    return [L[-1]]

def last_element_singleton():
    return to_neural_op(last_element_singleton_helper)

def identity():
    return to_neural_op(lambda x : x)

#An op which extracts a particular index from a list-valued input
def get_index(ind):
    return to_neural_op(lambda lst : lst[ind])

#An op which takes a list-valued input and sums all of the entries
def reduce_sum():
    return to_neural_op(lambda lst : sum(lst))

#Given a list of Neural_Ops [o1, o2, o3, o4, ...], this creates a single neural op
#which takes in parallel lists of inputs and parameters to each neural op and returns a list of results
def parallel(*ops):
    def new_op(xs, params, quantize=False):
        result = []
        for x, op, param in zip(xs, ops, params):
            result.append(op.operation(x, param, quantize))
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
    def new_op(x, params, quantize=False):
        temp = x
        for op, param in zip(ops, params):
            temp = op.operation(temp, param, quantize)
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

def superspliceHelper(L):
    if (len(L) == 0):
        return L
    if (len(L) == 1):
        if (isinstance(L[0], list)):
            return superspliceHelper(L[0])
        else:
            return L
    else:
        head = L[0]
        tail = L[1:] 
        headSplice = None
        if (isinstance(L[0], list)):
            headSplice = superspliceHelper(L[0])
        else:
            headSplice = [head]
        tailSplice = superspliceHelper(tail)
        return headSplice + tailSplice
    

#Neural op which recursively splices a nested structure of lists into a single list
def supersplice():
    return to_neural_op(superspliceHelper)

#Neural op which appends an element to a list
def append():
    return to_neural_op(lambda L: L[0] + [L[1]])

#Neural op which generates a specified number of copies of its input
def repeat(N):
    return to_neural_op(lambda x: [x] * N)

#Neural op which takes a list-valued input and concatenates all of the feature maps in it
def stack_features():
    return to_neural_op(ns.concat)

def unstack_features():
    return to_neural_op(ns.unconcat)

#Neural op which takes an input x and a list of ops [o1, o2, o3, o4, ...] and
#Returns a list [o1(x), o2(x), o3(x), ...]
def split(*ops):
    return repeat(len(ops)).then(parallel(*ops)) 

#Batch normalization op
#TODO: Do we need to keep track of parameters here?
#TODO: Should we quantize this?
def batch_norm():
    return to_neural_op(tf.contrib.layers.batch_norm)

#Something that masquerades as a "batch norm" layer, but is more accurately
#described as just a channel-wise shift/rescale op
def quasi_batch_norm(num_chan):
    def result_op(x, params, quantize=False):
        mean, variance, offset, scale = params
        scale = quantize_weight_array(scale, quantize)
        variance = quantize_weight_array(variance, quantize)
        #TODO: This seems kinda borked...

        epsilon = 0.01
        return tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon)

    def param_gen(name): 
        mean = get_b(name + "-qbatchNorm_mean_", [num_chan])
        variance = get_W(name + "-qbatchNorm_variance_", [num_chan])
        scale = get_W(name + "-qbatchNorm_scale_", [num_chan])
        offset = get_b(name + "-qbatchNorm_offset_", [num_chan])
        return [mean, variance, offset, scale]
    return Neural_Op(result_op, param_gen).then(activ_fn()).then(quantize_channels(num_chan)) #Activation function applied after so that the range is standardized

                    

#Default 2x upscaling and downscaling ops
def upscale():
    return upscale2x()

def downscale(F):
    return dconv(3, F, F, chan_mult=1, stride=2)
    #return max_pool2x2()

#cxc convolution operator straight up (no bias term)
def lin_conv(c, in_chan, out_chan, stride=1):
    def result_op(x, W, quantize=False):
        return lin_conv_helper(x, W, stride, quantize)
    return Neural_Op(result_op,
                     lambda name: get_W_conv(name + "-convC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "I",
                                            c, in_chan, out_chan))

#TODO: Refactor to be just a lin conv + bias!
#cxc convolution operator with the given number of in_channels and out_channels
#without passing through any activation function
def affine_conv(c, in_chan, out_chan, stride=1):
    def result_op(x, params, quantize=False):
        return affine_conv_helper(x, params, stride, quantize)
    return Neural_Op(result_op,
           lambda name: get_W_b_conv(name + "-convC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "I", 
                                    c, in_chan, out_chan))
#cxc convolution operator with the given number of in_channels and out_channels
#followed up by our choice of activation function
def conv(c, in_chan, out_chan, stride=1, fixed_range=False, activation=activ_fn()):
    if (not fixed_range):
        return affine_conv(c, in_chan, out_chan, stride).then(activation).then(quantize_channels(out_chan))
    else:
        return affine_conv(c, in_chan, out_chan, stride).then(activation).then(quantize_fixed_range(0.0, 6.0))

#TODO: Test how your network performs with these!
#Linear separable convolution with the given kernel size (cxc), input channels, output channels,
#channel multiplier (default: 1) and spatial stride (default: 1)
def affine_sconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    def result_op(x, params, quantize=False):
        return affine_sconv_helper(x, params, stride, quantize)
    return Neural_Op(result_op,
            lambda name: get_W_b_sconv(name + "-sconvC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "c" + str(chan_mult) + "I",
                                       c, in_chan, out_chan, chan_mult))

def affine_dconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    def result_op(x, params, quantize=False):
        return affine_dconv_helper(x, params, stride, quantize)
    return Neural_Op(result_op,
            lambda name: get_W_b_dconv(name + "-sconvC" + str(c) + "c" + str(in_chan) + "c" + str(out_chan) + "c" + str(chan_mult) + "I",
                                       c, in_chan, out_chan, chan_mult))


def sconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    return affine_sconv(c, in_chan, out_chan, chan_mult, stride).then(activ_fn()).then(quantize_channels(out_chan))

def dconv(c, in_chan, out_chan, chan_mult=1, stride=1):
    return affine_dconv(c, in_chan, out_chan, chan_mult, stride).then(activ_fn()).then(quantize_channels(out_chan))

def avg_pool2x2():
    return to_neural_op(lambda x: tf.contrib.layers.avg_pool2d(x, 2, padding='SAME'))
def max_pool2x2():
    return to_neural_op(lambda x: tf.contrib.layers.max_pool2d(x, 2, padding='SAME'))

def upscale2x():
    return to_neural_op(upscale2xhelper)

def downscale2xhelper(x):
    return tf.contrib.layers.avg_pool2d(x, 2, padding='SAME')

def upscale2xhelper(x):
    H, W, C = height_width_channels(x)
    return tf.image.resize_images(x, [H * 2, W * 2])

def upscale4x(x):
    return upscale2x().then(upscale2x())
def weight_initializer():
    return tf.contrib.layers.xavier_initializer()

def bias_initializer():
    return tf.zeros_initializer()

#Random vector with the given number of entries, slightly regularized to taste
def get_v(name, num_entries, stddev=0.5):
    return tf.get_variable(name, [num_entries], initializer=tf.random_normal_initializer(0.0, stddev),
                           regularizer=regularizer)

#Random vectors with the given number of entries, slightly regularized to taste
def get_vs(name, num_rows, num_entries, stddev=0.5):
    return tf.get_variable(name, [num_rows, num_entries], initializer=tf.random_normal_initializer(0.0, stddev),
                           regularizer=regularizer)

def get_W(name, shape):
    return tf.get_variable(name, shape, initializer=weight_initializer(), regularizer=regularizer)
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

#dconv is just a depthwise convolution
def get_W_b_dconv(name, c, in_chan, out_chan, chan_mult):
    return [get_W(name + "_W_d", [c, c, in_chan, chan_mult]),
            get_b_conv(name, out_chan)]

#Adds a per-feature-map bias
def add_bias(F):
    return Neural_Op(lambda x, b, quantize: x + b, lambda name: get_b_conv(name, F))

def lin_conv_helper(x, W, stride=1, quantize=False):
    W = quantize_weight_array(W, quantize)
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

#If the "quantize" flag is true,
#this quantizes the weight array
def quantize_weight_array(W, quantize=False):
    if (quantize):
        minVal = tf.reduce_min(W)
        maxVal = tf.reduce_max(W)
        return tf.quantization.fake_quant_with_min_max_vars(W, minVal, maxVal, narrow_range=True)
    else:
        return W

def affine_conv_helper(x, params, stride=1, quantize=False):
    W, b = params
    return lin_conv_helper(x, W, stride, quantize) + b

def affine_sconv_helper(x, params, stride=1, quantize=False):
    W_d, W_p, b = params
    W_d = quantize_weight_array(W_d, quantize)
    W_p = quantize_weight_array(W_p, quantize)
    return tf.nn.separable_conv2d(x, W_d, W_p, strides=[1, stride, stride, 1], padding='SAME') + b

def affine_dconv_helper(x, params, stride=1, quantize=False):
    W_d, b = params
    W_d = quantize_weight_array(W_d, quantize)
    return tf.nn.depthwise_conv2d(x, W_d, strides=[1, stride, stride, 1], padding='SAME') + b

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
