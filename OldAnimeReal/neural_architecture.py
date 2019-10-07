from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import augmentation as aug

import parts
import loss

import neural_architecture as na
from depth_convolution import *

import neural_structuring as ns

import neural_ops as nn
import depth_convolution as scale_conv
from params import *
from neural_ops import *

#"Internals" of the network (past the initial input convolutions, before the output convolutions)
#in scale-convolution format
def heatmapGenInternals(F, L, B):
    #OLD Version:
    #iterate_op(lambda: sum_scale_conv_layer(F), L) #Run through L layers of scale-convolutional network
    #Now we're using DenseNet-like blocks

    op = identity() #At first, we just have a single

    layers_per_block = int(L / B)
    for block_num in range(0, B):
        #Create an op for the internals of the block (as a transformation from
        #a singleton list of activation maps to a new list of activation maps)
        block_op = identity()
        for layer in range(1, layers_per_block + 1):
            #For each layer in the block, we add operations as follows:
            block_op = block_op.then(
                    split(identity(), #Maintain a copy of the list of activation maps, unmodified
                    #dense_sum_scale_conv_layer(F, layer))).then( #On a copy of that list, apply a dense sum scale conv layer
                    dense_sum_scale_layer(F, layer, 4))).then(
                    append()) #Append the newly-computed layer onto the result

        #Now, given the definition of the op for the block, we need to do some things.
        op = op.then(split( #Split off a copy of the input so we can get a ResNet-like structure
                        to_neural_op(lambda x: [x]).then( #First, we need to pack the input to a list
                        block_op).then( #Once we've done that, we can apply the block's op
                        get_index(layers_per_block)) #Discard all activation maps from non-final layers in the block
                     , identity()).then_apply( #On the other branch of the split, we just have the unmodified input
                 transpose).then( #Transpose from dimensions 2, scales to dimensions scales, 2
                    reduce_sum().map_on_lists()) #Add the residual at every scale
                )
    return op
    

#Neural network for heatmap generation as a neural_op
#Parameters: F - the number of feature maps per layer
#L - the number of scale-convolutional layers
#B - the block-size for dense connections between layers (must divide L)
def heatmapGen(F, L, B):
    #For when this op runs...
    """Arguments:
        x: an input tensor with the dimensions (N_examples, img_height, img_width, img_color_channels)
        params: the parameters of the model, as expressed by "gen_params" above
       Returns:
        A tuple of tensors of shape (N_examples, height, width, num_field_maps)
        in decreasing order of size (in width and height), from 64x64 down to 8x8
        which stores the detection heatmaps for each body part
    """
    return batch_norm().then_apply( #First, batch-normalize
          lambda x: loss.iterated_avg_pool_downsample(x, 6) #Then, iteratively 2x downsample from 256x256 to 8x8
          ).then(unsplice(3)).then(parallel( #PREPROCESSING: Separate out the 256x256, 128x128 and 64x64 feature maps
                parallel(
                    conv(3, img_color_channels, feats_128, stride=2).then(
                        conv(3, feats_128, F / 4, stride=2)), #256x256 feature maps get down-conved twice at stride 2
                    conv(3, img_color_channels, F / 4, stride=2), #128x128 feature maps get down-conved once at stride 2 
                    conv(3, img_color_channels, F / 2) #Generate F/2 64x64 feature maps in the same manner, but stride-1
                ).then(stack_features()).then(  #Stack all of those together to get something of size 64x64
                    conv(3, F, F)       #Convolve the stacked maps to hopefully get agreement with those 32x32 and below
                                ).to_singleton_list()
                , replicate_over_list(
                    lambda: conv(3, img_color_channels, F), 3) #For the feature maps 32x32 and below, convolve so we can get F channels
            )).then(splice()).then_apply( #That was a mouthful! Put our new 64x64, 32x32, 16x16, and 8x8 feature maps back together
                lambda L: L[::-1]).then( #And reverse the order (8x8 to 64x64 now). SCALE CONVOLUTIONS ARE NEXT
                heatmapGenInternals(F, L, B) #We can actually do stuff with scale-convolutions!
            ).then(replicate_over_list(
                    lambda: conv(3, F, parts.num_parts), 4)).then_apply( #We need heatmaps, not F feature maps
            lambda L : L[::-1]) #Finally, reverse again so we're in increasing order (8x8 to 64x64 heatmaps)
