import tensorflow as tf
import neural_ops as nn
import neural_structuring as ns
from neural_ops import *

#Definitions related to _depth convolutions_,
#a novel network architecture primitive.
#These extend regular 2d convolutions
#to a third spatial dimension, which represents
#varying power-of-two scales

#TRAPEZOIDS:
#These are the replicated network structure
#along the scale dimension in each layer.
#They take 1x, 2x, and 4x scaled activation maps
#and return a 2x scaled activation map

#A trapezoid which operates by independently applying
#convolutions to each of its 1x, 2x, 4x inputs,
#up/down sampling the resulting 1x and 4x inputs, concatenating their features,
#and then running a plain-old 3x3 convolution on this concatenated stack
#to produce an output
#Note that the number of input features must match the number of output
#features, as there's an identity pass-through (a la residual nets)
def default_trapezoid(F, one_F, two_F, four_F, one_c=1, two_c=1, four_c=1, out_c=3):
    return split(get_index(1), #Extract the copy of the two-x input to add on as residual later
            parallel(conv(one_c, F, one_F).then(upscale()), #For the one-x input, 1x1 conv then upscale
                     conv(two_c, F, two_F),                 #For the two-x input, 1x1 conv
                     conv(four_c, F, four_F).then(downscale()) #For the four-x input, 1x1 conv and downscale
            ).then(stack_features()).then(conv(out_c, one_F + two_F + four_F, F)) #Concat features and 3x3 conv for out
    ).then(reduce_sum()).add_identification("trapezoid") #Add on two-x to the output (in ResNet style) and identify

#A trapezoid which operates by 1x1 linear transforms and sums
def sum_trapezoid(F, one_c=1, two_c=1, four_c=1, out_c=3):
    return split(get_index(1), #Extract the copy of the two-x input to add on as residual later
            parallel(lin_conv(one_c, F, F).then(upscale()), #For the one-x input, lin conv then upscale
                     lin_conv(two_c, F, F),                  #For the two-x input, just lin conv
                     lin_conv(four_c, F, F).then(downscale()) #For the four-x input, lin conv then downscale
            ).then(reduce_sum()).then(sconv(out_c, F, F, chan_mult=4)) #Add all of those together, then run straight through a traditional 3x3 convolution
    ).then(reduce_sum()).add_identification("trapezoid") #Add on two-x to the output (in ResNet style), and identify

#A denseNet sum trapezoid (like a sum trapezoid, but instead of taking single activation maps from 
#the previous layer, this takes a collection of activation maps from all layers in a block)
#Note that this does not incorporate residual connections within the trapezoid --
#In theory, DenseNets don't need them, except possibly at the block level
#P is the number of layers within the block which precede the layer of this trapezoid
def dense_sum_trapezoid(F, P, out_c=3):
    return parallel(identity(), identity(), downscale().map_on_lists()).then( #Very first thing, downsize the 4x to 2x
           replicate_over_list(lambda: replicate_over_list(lambda: lin_conv(1, F, F), P) ,3)).then( #Then, 1x1 lin convs
           reduce_sum().map_on_lists()).then( #Add together the results to get one 1x map, one 2x map, and one 2x map
           replicate_over_list(lambda: add_bias(F).then(activ_fn()), 3)).then( #bias each one, and run through activation
           parallel(upscale(), identity(), identity())).then( #Finally upscale that 1x map, so all maps are 2x
           replicate_over_list(lambda: lin_conv(1, F, F), 3)).then( #Apply lin 1x1 convs to each of them
           reduce_sum()).then( #Add all of them together to get something new at 2x
           conv(out_c, F, F)).add_identification("trapezoid") #Run all that straight through a convolution

 
#A denseNet sum trapezoid (like a sum trapezoid, but instead of taking single activation maps from 
#the previous layer, this takes a collection of activation maps from all layers in a block)
#Note that this does not incorporate residual connections within the trapezoid --
#In theory, DenseNets don't need them, except possibly at the block level
#P is the number of layers within the block which precede the layer of this trapezoid
def dense_sconv_trapezoid(F, P, out_c=3):
    return parallel(identity(), identity(), downscale().map_on_lists()).then( #Very first thing, downsize the 4x to 2x
           replicate_over_list(lambda: replicate_over_list(lambda: lin_conv(1, F, F), P) ,3)).then( #Then, 1x1 lin convs
           reduce_sum().map_on_lists()).then( #Add together the results to get one 1x map, one 2x map, and one 2x map
           replicate_over_list(lambda: add_bias(F).then(activ_fn()), 3)).then( #bias each one, and run through activation
           parallel(upscale(), identity(), identity())).then( #Finally upscale that 1x map, so all maps are 2x
           replicate_over_list(lambda: lin_conv(1, F, F), 3)).then( #Apply lin 1x1 convs to each of them
           reduce_sum()).then( #Add all of them together to get something new at 2x
           sconv(out_c, F, F)).add_identification("trapezoid") #Run all that straight through a convolution

  
#A denseNet sum trapezoid (like a sum trapezoid, but instead of taking single activation maps from 
#the previous layer, this takes a collection of activation maps from all layers in a block)
#Note that this does not incorporate residual connections within the trapezoid --
#In theory, DenseNets don't need them, except possibly at the block level
#P is the number of layers within the block which precede the layer of this trapezoid
def dense_dconv_trapezoid(F, P, out_c=3):
    return parallel(identity(), identity(), downscale().map_on_lists()).then( #Very first thing, downsize the 4x to 2x
           replicate_over_list(lambda: replicate_over_list(lambda: lin_conv(1, F, F), P) ,3)).then( #Then, 1x1 lin convs
           reduce_sum().map_on_lists()).then( #Add together the results to get one 1x map, one 2x map, and one 2x map
           replicate_over_list(lambda: add_bias(F).then(activ_fn()), 3)).then( #bias each one, and run through activation
           parallel(upscale(), identity(), identity())).then( #Finally upscale that 1x map, so all maps are 2x
           replicate_over_list(lambda: lin_conv(1, F, F), 3)).then( #Apply lin 1x1 convs to each of them
           reduce_sum()).then( #Add all of them together to get something new at 2x
           dconv(out_c, F, F)).add_identification("trapezoid") #Run all that straight through a convolution

#Same dealio as before, but quantizable 
#by replacing sum operations
def dense_dconv_quant_trapezoid(F, P, out_c=3):
    return parallel(identity(), identity(), downscale(F).map_on_lists()).then( #Very first thing, downsize the 4x to 2x
           replicate_over_list(lambda : concat_depth(), 3)).then(#Then, concatenate all feature maps within each scale
           parallel(quantize_resid_channels(F), quantize_resid_channels(P * F), quantize_resid_channels(F))).then(
           parallel(conv(1, F, F), conv(1, P * F, F), conv(1, F, F))).then( #TODO: Make this work for things other than t-shape. Apply 1x1 convs to each size
           parallel(upscale(), identity(), identity())).then( #Finally upscale that 1x map, so all maps are 2x
           concat_depth()).then( #Concatenate all along the last dimension
           quantize_channels(3 * F)).then( #Quantize the concatted collection per-channel
           conv(1, 3 * F, F)).then( #Apply 1x1 convs to the stacked collection of maps to get something new at 2x
           conv(out_c, F, F)).add_identification("trapezoid") #Run all that straight through a convolution


#Same dealio as before, but the dense trapezoid is "t-shaped" in the sense that
#we have dense connections _only_ for the 2x part. The 1x and 4x parts are just
#based on the most recent of each there
def dense_tshape_dconv_trapezoid(F, P, out_c=3):
    return parallel(last_element_singleton(), identity(), last_element_singleton()).then(
           dense_dconv_trapezoid(F, P, out_c=out_c))

def dense_tshape_dconv_quant_trapezoid(F, P, out_c=3):
    return parallel(last_element_singleton(), identity(), last_element_singleton()).then(
           dense_dconv_quant_trapezoid(F, P, out_c=out_c))

#Given a list of activation maps (1x, 2x, 4x, ...), prepend a tensor of zeros at 1/2x
def zero_pad_half(x, quantize=False):
    B, H_one, W_one, F = nn.batch_height_width_channels(x[0])
    half_size_zeros = tf.zeros([B, H_one / 2, W_one / 2, F])
    if (quantize):
        half_size_zeros = tf.quantization.fake_quant_with_min_max_args(half_size_zeros, -0.01, 0.01, narrow_range=True)
    return [half_size_zeros] + x

#Given a list of activation maps (1x, 2x, 4x, ...), append a tensor of zeros at max-times-two size
def zero_pad_two_max(x, quantize=False):
    B, H_max, W_max, F = nn.batch_height_width_channels(x[-1])
    two_max_size_zeros = tf.zeros([B, H_max * 2, W_max * 2, F])
    if (quantize):
        two_max_size_zeros = tf.quantization.fake_quant_with_min_max_args(two_max_size_zeros, -0.01, 0.01, narrow_range=True)
    return x + [two_max_size_zeros]

#Given a list of activation maps, this zero-pads at both 1/2x and 2 times the max size
def zero_pad_edges_helper(x, params, quantize=False):
    return zero_pad_half(zero_pad_two_max(x, quantize), quantize)

def zero_pad_edges():
    return Neural_Op(zero_pad_edges_helper, no_params)

#Given a list of activation maps [a0, a1, a2, a3, ...], this returns a list of consecutive triples
#[[a0, a1, a2], [a1, a2, a3], ...]
def consecutive_triples(xs):
    result = []
    for i in range(len(xs) - 2):
        result.append([xs[i], xs[i + 1], xs[i + 2]])
    return result

#Given a lists of lists, "transpose" it
def transpose(xs):
    return map(list, zip(*xs))


#Defines one scale-convolutional layer as a neural op
#which takes as input a list of activation maps from the previous layer in increasing order
#of scale (1x, 2x, 4x, etc.)
#Returns a list of activation maps which is the same size as the input by applying
#the default trapezoid at all possible positions, zero-padding at the largest-scale
#and smallest-scale output positions
def scale_conv_layer(trapezoid):
    return zero_pad_edges().then_apply(consecutive_triples).then(
        trapezoid.map_on_lists()
    ).add_identification("scaleconv")

#Defines one dense scale-convolutional layer as a neural op
#which takes as input a list of lists of activation maps from the previous layers,
#where the outermost list is indexed by layer, and inner lists are indexed by scale
#Returns a list of activation maps, one per scale
def dense_scale_conv_layer(dense_trapezoid):
    #First, zero pad in the fake 1/2 scale and 2xmax scales on each layer
    #After doing that, generate consecutive triples for each desired output scale
    #We now have something of dims (layers, scales, 3)
    return zero_pad_edges().then_apply(consecutive_triples).map_on_lists().then_apply(
           transpose).then( #Transpose to get something of the form (scales, layers, 3)
           to_neural_op(transpose).map_on_lists()).then( #Transpose again, getting (scales, 3, layers) 
           dense_trapezoid.map_on_lists()).add_identification("dense_scale_conv") 
           #Now the dense trapezoids will handle that for each scale

#Defines a dense scale layer (NOT convolutional) as a neural op
#This is in the same format as the previous function, but instead, it takes
#a generator for trapezoids
def dense_scale_layer(dense_trapezoid_generator, num_scales):
    return zero_pad_edges().then_apply(consecutive_triples).map_on_lists().then_apply(
           transpose).then( #Transpose to get something of the form (scales, layers, 3)
           to_neural_op(transpose).map_on_lists()).then( #Transpose again, getting (scales, 3, layers) 
           replicate_over_list(dense_trapezoid_generator, num_scales)).add_identification("dense_scale_conv") 
           #Now the dense trapezoids will handle that for each scale

      
#A dense scale-convolutional layer built on a dense sum trapezoid operating on P previous layers
def dense_sum_scale_conv_layer(F, P, out_c=3):
    return dense_scale_conv_layer(dense_sum_trapezoid(F, P, out_c))

def dense_sum_scale_sconv_layer(F, P, out_c=3):
    return dense_scale_conv_layer(dense_sconv_trapezoid(F, P, out_c))

def dense_sum_scale_layer(F, P, num_scales, out_c=3):
    return dense_scale_layer(lambda: dense_sum_trapezoid(F, P, out_c), num_scales)

def dense_sconv_scale_layer(F, P, num_scales, out_c=3):
    return dense_scale_layer(lambda: dense_sconv_trapezoid(F, P, out_c), num_scales)

def dense_dconv_scale_layer(F, P, num_scales, out_c=3):
    return dense_scale_layer(lambda: dense_dconv_trapezoid(F, P, out_c), num_scales)

def dense_tshape_dconv_scale_layer(F, P, num_scales, out_c=3):
    return dense_scale_layer(lambda: dense_tshape_dconv_trapezoid(F, P, out_c), num_scales)

def dense_tshape_dconv_quant_scale_layer(F, P, num_scales, out_c=3):
    return dense_scale_layer(lambda: dense_tshape_dconv_quant_trapezoid(F, P, out_c), num_scales)

#A scale-convolutional layer built on a sum trapezoid
def sum_scale_conv_layer(F, one_c=1, two_c=1, four_c=1, out_c=3):
    return scale_conv_layer(sum_trapezoid(F, one_c, two_c, four_c, out_c))

#A scale-convolutional layer with a 1/2/1 feature map split for one_F, two_F and four_F
#This type of layer tends to do more in the current scale than in other scales
def mid_heavy_scale_conv_layer(F, one_c=1, two_c=1, four_c=1, out_c=3):
    one_F = F / 4
    two_F = F / 2
    four_F = F / 4
    return scale_conv_layer(default_trapezoid(F, one_F, two_F, four_F, one_c, two_c, four_c, out_c))

#A scale-convolutional layer with a 1/1/2 feature map split for one_F, two_F and four_F
#This type of layer tends to do more with the higher-res scale than the other scales
def up_heavy_scale_conv_layer(F, one_c=1, two_c=1, four_c=1, out_c=3):
    one_F = F / 4
    two_F = F / 4
    four_F = F / 2
    return scale_conv_layer(default_trapezoid(F, one_F, two_F, four_F, one_c, two_c, four_c, out_c))

#A scale-convolutional layer with a 2/1/1 feature map split for one_F, two_F and four_F
#This type of layer tends to do more with the lower-res scale than the other scales
def down_heavy_scale_conv_layer(F, one_c=1, two_c=1, four_c=1, out_c=3):
    one_F = F / 2
    two_F = F / 4
    four_F = F / 4
    return scale_conv_layer(default_trapezoid(F, one_F, two_F, four_F, one_c, two_c, four_c, out_c))

