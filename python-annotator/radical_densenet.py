from neural_ops import *
import tensorflow as tf
import neural_architecture as na
import neural_structuring as ns
from sparse_vec_similarity import *
#Construction of "radical densenets", which
#are an experimental kind of multi-scale densenet

def concatScales(old_and_next):
    result = []
    old_scales, new_maps = old_and_next

    #First, copy over all of the old items in each scale
    for scale_ind in range(len(new_maps)):
        result.append([])

        old_scale_items = old_scales[scale_ind]
        for item_ind in range(len(old_scale_items)):
            result[scale_ind].append(old_scale_items[item_ind])
    #Copy over the new items
    for scale_ind in range(len(new_maps)):
        result[scale_ind].append(new_maps[scale_ind])

    return result


#Given a list lists of pre-existing feature maps (ordered in increasing order of scale) 
#and a list of new feature maps per scale, return a list of lists of feature maps
#which appends the new maps in the right places, but also downscales/upscales of the new maps
#in the right places as well
def propagateAndConcatScales(old_and_next):
    old_scales, new_maps = old_and_next

    result = concatScales(old_and_next)

    #Now, copy over upscales of new maps
    for scale_ind in range(len(new_maps) - 1):
        orig = new_maps[scale_ind]
        upscaled = upscale2xhelper(orig)

        upscale_ind = scale_ind + 1

        result[upscale_ind].append(upscaled)

    #Now, copy over downscales of new maps
    for downscale_ind in range(len(new_maps) - 1):
        scale_ind = downscale_ind + 1
        orig = new_maps[scale_ind]
        downscaled = downscale2xhelper(orig)
        result[downscale_ind].append(downscaled)

    return result


#Tensorflow function with a custom gradient which
#takes a list (feature_selector_1, feature_map_1, ... _n)
#and returns a stack of K feature maps
#(given each feature_selector has K elements)
#built from the n summands
#This is done in a more memory-efficient way
#than the dumb multiply-and-sum way
@tf.custom_gradient
def feature_select(*args):

    xs = args[1:]
    #Define the forward operation

    #First, reconstruct the actual arguments
    feature_selector_mat = args[0]
    #recall, feature_selector_mat is num_features x num_prev_layers

    transpose_feature_selector_mat = tf.transpose(feature_selector_mat)
    #this is num_prev_layers * num_features

    #Get a bool vector of layers with nonzero feature selection vectors
    #condition = tf.logical_not(tf.equal(transpose_feature_selector_mat, 0.0))
    #transpose_feature_selector_nonzero_mat = tf.reduce_any(condition, axis=-1)

    feature_selectors = []
    #feature_selectors_nonzero = []
    for i in range(len(xs)):
        feature_selectors.append(transpose_feature_selector_mat[i])
        #feature_selectors_nonzero.append(transpose_feature_selector_nonzero_mat[i])


    summands = []
    for i in range(len(xs)):
        x = xs[i]
        feature_selector = feature_selectors[i]
        summand = tf.reshape(feature_selector, [1, 1, 1, -1]) * x
        summands.append(summand)

    #Define the gradients
    def grad(dy):
        #Note: dy is intuitively the direction that we __want__ the summed output
        #feature maps to go in.
        x_grads = []
        feature_selector_grads = []

        dy_flat = tf.reshape(dy, [-1, tf.shape(dy)[-1]])

        #Compute gradients with respect to the previous feature maps
        for i in range(len(xs)):
            #For the gradient here, gate the dy by
            #the weights that this x actually effects
            #and collapse across features with a sum
            feature_selector = feature_selectors[i]
            x = xs[i]

            #is_nonzero = feature_selectors_nonzero[i]

            def nonzero_branch():
                
                expanded_feature_selector = tf.reshape(feature_selector, [-1, 1])

                x_grad = tf.matmul(dy_flat, expanded_feature_selector, b_is_sparse=False)

                #The above should be the same thing as:
                #x_grad = tf.einsum('ijkl,l->ijk', dy, feature_selector)

                x_grad = tf.reshape(x_grad, tf.shape(x))
                return x_grad
            def zero_branch():
                return tf.zeros_like(x, dtype=tf.float32) 

            x_grad = nonzero_branch()

            #There's another case -- if the feature is not used, then the gradient is zero!
            #x_grad = tf.cond(is_nonzero, nonzero_branch, zero_branch)

            x_grads.append(x_grad)

        #Compute gradients with respect to the weights on feature maps
        for i in range(len(feature_selectors)):
            #Find how much x goes in the direction of dy
            x = xs[i]
            #is_nonzero = feature_selectors_nonzero[i]
            
            def nonzero_branch_two():
                x_flat = tf.reshape(x, [1, -1])
                x_ys_similarities = tf.matmul(x_flat, dy_flat)

                x_ys_similarities = tf.reshape(x_ys_similarities, tf.shape(feature_selectors[i]))

                
                #The above should be the same as...
                #x_ys_similarities = tf.einsum('ijkl,ijk->l', dy, x)
                return x_ys_similarities

            def zero_branch_two():
                return tf.zeros_like(feature_selectors[i], dtype=tf.float32)

            #There's another case: If the feature is not used, it's zero
            #x_ys_similarities = tf.cond(is_nonzero, nonzero_branch_two, zero_branch_two)
            x_ys_similarities = nonzero_branch_two()
            feature_selector_grads.append(x_ys_similarities)
        #Okay, great. Now return grads in the same order as the arguments. 

        #recall, feature_selector_mat was originally num_features x num_prev_layers
        #we need to take the feature_selector_grads and concat along second axis
        f_grad = tf.stack(feature_selector_grads, axis=-1)


        result_grads = []
        result_grads.append(f_grad)
        for i in range(len(feature_selectors)):
            result_grads.append(x_grads[i])
        return result_grads



    return tf.accumulate_n(summands), grad

def dumbFeatureSelect(xs, f, num_prev_layers, num_layers_to_select):
    #Take all of the x'es and stack them into a tensor with many, many feature
    #maps
    stacked_xs = ns.concat(xs)

    #Reshape xs so that it's in shape
    reshaped_xs_stack = tf.reshape(stacked_xs, [-1,  num_prev_layers])

    #Matmul, baybee, to get something shaped (something) x num_layers_to_select
    selected_features = tf.matmul(reshaped_xs_stack, f, transpose_b=True)

    #Reshape so it again has the same leading dimensions as the x's
    reshaped_features = tf.reshape(selected_features, [tf.shape(xs[0])[0], 
                                   tf.shape(xs[0])[1], tf.shape(xs[0])[2], num_layers_to_select])

    return reshaped_features


#given a list of previous feature maps,
#select num_layers_to_select activations from num_prev_layers,
#and concat them together (based on trainable feature selection vectors)
def trainedFeatureSelect(num_prev_layers, num_layers_to_select):
    def new_op(xs, params, quantize=False):
        print(len(xs))

        vs = params


        #Take the big params vector and feat it to randomized_sparse_mat_similarity
        f = randomized_sparse_mat_similarity(vs, num_prev_layers, num_layers_to_select, dense_rep=True)
        #f now contains a num_layers_to_select x num_prev_layers matrix of weights for selection from xs

        dumb = True #If true, use just a big stacking operation (not memory efficient), otherwise, do things smert
        if (not dumb):
            arg_tuple = []
            arg_tuple.append(f)
            for i in range(num_prev_layers):
                arg_tuple.append(xs[i])

            #Okay, great. Now every previous layer has an associated feature selector
            return feature_select(*arg_tuple)
        else:
            return dumbFeatureSelect(xs, f, num_prev_layers, num_layers_to_select)


    def new_param_generator(name):
        #We have only a single parameter, and that is a num_layers_to_select x num_prev_layers
        #weight matrix
        modName = name + "paramSelector"
        return get_vs(modName, num_layers_to_select, num_prev_layers)
    
    return Neural_Op(new_op, new_param_generator)

#An op which takes a list of single-feature-map activation maps
#(exactly num_prev_layers of those), picks out num_layers_to_select
#layers from that, concats them, and applies a 3x3 convolution yielding
#an activation map with a single output feature map
def radicalScaleFunc(num_prev_layers, num_layers_to_select, num_per_conv_outputs):
    return trainedFeatureSelect(num_prev_layers, num_layers_to_select).then(
                conv(3, num_layers_to_select, num_per_conv_outputs))

#Returns a neural_op which takes a list (ordered in inceasing order of scale) of lists of
#feature maps
def radicalLayerFunc(in_channels, layerNum, cap_fn, num_scales, num_per_conv_outputs=1, endLayers=None):

    #First, compute the number of previous maps in each scale
    num_prev_in_scales = []
    for i in range(num_scales):
        num_prev_in_scales.append(in_channels + layerNum * 3 * num_per_conv_outputs)

    #Scales on the edges of the array are different, because they don't have both upscales and downscales
    num_prev_in_scales[0] = in_channels + layerNum * 2 * num_per_conv_outputs
    num_prev_in_scales[num_scales - 1] = in_channels + layerNum * 2 * num_per_conv_outputs

    if (endLayers is not None):
        for i in range(num_scales):
            num_prev_in_scales[i] += endLayers * num_per_conv_outputs


    scale_caps = []
    #Compute the caps on the layer scales
    for i in range(num_scales):
        scale_caps.append(cap_fn(layerNum, i))

    ops = []
    #Okay, great. Now, construct the resulting function by case inspection for each
    for i in range(num_scales):
        if (num_prev_in_scales[i] < scale_caps[i]):
            #Defer to the implementation where we concat all previous features together
            #and then conv it.
            op = stack_features().then(conv(3, num_prev_in_scales[i], num_per_conv_outputs))
        else:
            #This is the case where we need to do sparse selection of features
            op = radicalScaleFunc(num_prev_in_scales[i], scale_caps[i], num_per_conv_outputs)

        ops.append(op)
    return parallel(*ops)


#Alternative DenseNet Internals utilizing TensorArrays
#TODO: This is a god function, but there doesn't seem to be much of an easy and logical way to
#unravel it, at least with Tensorflow's restrictions that we're trying to work around with this...
def radicalDenseNetInternalsLoop(L, scale_caps, in_channels=3, out_channels=4):
    #First thing's first -- the input we got is a list of tensors with in_channels
    #feature maps, one per scale. We need to change that into a list of the individual
    #feature maps

    #Compute max number of previous layers in each scale
    num_layers_in_scales = []

    for i in range(num_scales):
        num_layers_in_scales.append(in_channels + L * 3)
    num_layers_in_scales[0] = in_channels + layerNum * 2
    num_layers_in_scales[num_scales - 1] = in_channels + layerNum * 2

    def operation(x, params, quantize=False):
        #Now, params contains a lot of stuff.
        #It contains convolution weights for each layer,
        #bias vectors for each layer,
        #and feature selectors for each layer
        conv_weights, bias_vectors, feature_selectors = params
        

        #Condition of the outer while loop
        def cond(*args):
            #For the condition of the while loop, we'll deal with arguments in
            #layerInds..., prevActivations..., (repeating, for each scale)
            #layerInds are the current layer indices for each scale
            #and prevActivations are TensorArrays of the previous activations for each scale
            layerInds = []
            for i in range(len(args)):
                if (i % 2 == 0):
                    layerInds.append(args[i])

            #Okay, great. Now we ensure that all of the layerInds are below their maximums
            result = None
            for i in range(len(layerInds)):
                layerInd = layerInds[i]
                if (result is None):
                    result = layerInd < num_layers_in_scales[i]
                else:
                    result = tf.logical_and(result, layerInd < num_layers_in_scales[i])
            return result
        #Body of the outer while loop
        def body(*args):
            layerInds = []
            prevActivations = []
            for i in range(len(args)):
                if (i % 2 == 0):
                    layerInds.append(args[i])
                else:
                    prevActivations.append(args[i])
            #Okay, so in the while loop, we need to do some wack crap.
            #For each scale, we need to compute the layer's new output
            #from all of the previous activation maps
            activations = []
            for i in range(len(layerInds)):
                scale_cap = scale_caps[i]
                activ = computeRadicalLayerLoop(layerInds[i], prevActivations[i], scale_cap)
                activations.append(activ)

            #Okay, great. Now, with those new activation maps, write up/downscaled
            #maps into proximate scales
            for i in range(len(layerInds)):
                #Write unmodified activations
                prevActivations[i] = prevActivations[i].write(layerInds[i], activations[i])
                layerInds[i] = layerInds[i] + 1

            for i in range(len(layerInds) - 1):
                #Upscaling time
                upscaled = upscale2xhelper(activations[i])
                prevActivations[i+1] = prevActivations[i+1].write(layerInds[i+1], upscaled)
                layerInds[i+1] = layerInds[i+1] + 1

            for low_i in range(len(layerInds - 1)):
                #Downscaling time
                i = low_i + 1
                downscaled = downscale2xhelper(activations[i])
                prevActivations[low_i] = prevActivations[low_i].write(layerInds[low_i], downscaled)
                layerInds[low_i] = layerInds[low_i] + 1

            #Okay, great. Now layerInds should all point to indices which are writeable in the TensorArrays
            #(don't overwrite previous tensors)
            #and they should contain the results we want. Go ahead and return the modified versions of the
            #arguments
            result = []
            for i in range(len(layerInds)):
                result.append(layerInds[i], prevActivations[i])
            return result
        def loop_vars():
            result = []
            #Initialize tensorarray objects for each scale
            for i in range(num_scales):
                result.append(tf.constant(0))
                result.append(tf.TensorArray(tf.float32, size=num_layers_in_scales[i]))
            return result
    def params_generator(name):
        #We need to return three lists,
        #the first being the convolutional weights per each scale and layer,
        #the second being the biases per each scale and layer
        #and the third being the feature selection vectors per each scale and layer
        return None
    return None

#The above is the partially-completed code for an implementation of this with while-loops.
#It was ultimately abandoned because it was realized that while you _could_ do this in tensorflow,
#it would probably not be particularly efficient to do so (even if it did recover linear memory-efficiency,
#training would be slowed by the nested while loops), so a better framework would be better.
#besides, many of the memory constraints can be mitigated here by setting k features to be outputted from
#each layer instead of just one. In other words: the above is for another experimental project, maybe
#in PyTorch, but not here!





#def computeRadicalLayerLoop(layerInd, prevActivations, num_features):


def radicalDenseNetInternals(L, cap_fn, num_per_conv_outputs=1, in_channels=3, out_channels=4, shallow_out=True):
    #First thing's first -- the input we got is a list of tensors with in_channels
    #feature maps, one per scale. We need to change that into a list (ordered by scale)
    #of the individual feature maps

    num_scales = 4

    op = unstack_features().map_on_lists() #Take the input features maps, and add them to the feature cascade

    #Op for the dense block
    block_op = identity()

    for layerNum in range(L):

        block_op = block_op.then(
                split(identity(), #First, maintain a copy of the list of (scales) lists of previous activation maps, unmodified
                radicalLayerFunc(in_channels, layerNum, cap_fn, num_scales, num_per_conv_outputs)).then_apply( #On the copy of that list, generate the new feature maps for this layer
                propagateAndConcatScales #From the previously-maintained list of scales' prev activ maps, and the new activ maps to add for each scale,
                )) #append to each element in the list not only the respective scale activ map, but also proximate 2x up/downscaled maps

    #Shallow out actually sets the output to be a 1x1 conv of _all_ concatted features in the net
    if (not shallow_out):
        #Okay, great. For the last four layers, we need to not propagateAndConcatScales
        out_layers = out_channels / num_per_conv_outputs
        for endLayerNum in range(out_layers):
            layerNum = L
            block_op = block_op.then(
                    split(identity(), #First, maintain a copy of the list of (scales) lists of previous activation maps, unmodified
                    radicalLayerFunc(in_channels, layerNum, cap_fn, num_scales, num_per_conv_outputs, endLayers=endLayerNum)).then_apply( #On the copy of that list, generate the new feature maps for this layer
                    concatScales)) #Just straightforward concat

        #Okay, great, now our network so far is...
        result = op.then(block_op)
        #At the end, we need to extract and stack exactly four features together to spit out as output
        return result.then(extractLastK(out_layers, num_scales))


    if (shallow_out):
        #Derive the number of computed maps for each scale
        num_prev_in_scales = []
        for i in range(num_scales):
            num_prev_in_scales.append(in_channels + L * 3 * num_per_conv_outputs)

        #Scales on the edges of the array are different, because they don't have both upscales and downscales
        num_prev_in_scales[0] = in_channels + L * 2 * num_per_conv_outputs
        num_prev_in_scales[num_scales - 1] = in_channels + L * 2 * num_per_conv_outputs

        block_op = block_op.then(concatAndConvAll(out_channels, num_scales, num_prev_in_scales))
        result = op.then(block_op)
        return result
                

def concatAndConvAll(out_channels, num_scales, num_prev_in_scales):
    ops = []
    for i in range(num_scales):
        op = identity().then(stack_features()
                ).then(conv(1, num_prev_in_scales[i], out_channels, fixed_range=True, activation=activ_fn_relu6()))
        ops.append(op)
    return parallel(*ops)


def extractLastK(out_channels, num_scales):
    ops = []
    for i in range(num_scales):
        op = identity().then_apply(lambda L: L[-out_channels:]).then(stack_features())
        ops.append(op)
    return parallel(*ops)



#L is the number of layers,
#cap_fn is the function taking the current layer number and the scale index
#to the max number of features to consider at that layer and scale
def radicalDenseNet(L, cap_fn, num_per_conv_outputs=1, in_channels=3, out_channels=4, shallow_out=True):

    #TODO: We need to plumb things so that the output activation map is a RELU6
    #TODO: We also need to support quantization
    return identity().then(
        na.iteratedDownsample() #First, iteratively 2x downsample from 256x256 to 8x8
        ).then_apply(
            lambda L: L[::-1]).then_apply( #And reverse the orders so that we're in the format expected
        lambda L: L[0:5]).then( #Restrict interest to only 128x128-size maps and below #TODO: More input maps, then!
            radicalDenseNetInternals(L, cap_fn, num_per_conv_outputs, in_channels, out_channels, shallow_out)
        ).then_apply(
            lambda L: L[0:4] #For compatibility with the old net, for now #TODO: Remove me!
            ).then_apply(
            lambda L: L[::-1]) #Reverse again
