#Smol little module which computes a tensorflow function
#which takes as input a vector in N-dimensional space,
#and returns an interpolation between the two closest (signed) basis vectors
#in such a way that the function as a whole is continuous

import tensorflow as tf

#Same deal as below, but we also pick a random basis vector
#, weight it with a weight in (-epsilon, epsilon), 
#compute the projection of x onto that vector, and add the projection in the direction
#of the randomly-chosen basis vector. The idea here is that we maintain sparsity
#while providing a more useful gradient than sparse_vec_similarity
def randomized_sparse_vec_similarity(x, N, epsilon=0.01, dense_rep=True):
    ident = tf.eye(N)
    random_ind = tf.random.uniform([], 0, N, dtype=tf.int32)
    direction = tf.gather(ident, random_ind, axis=0)

    weight = tf.random.uniform([], 0.0, epsilon, dtype=tf.float32)

    weighted_direction = weight * direction
    
    projection = tf.tensordot(x, weighted_direction, 1)

    if (dense_rep):
        sparse_sim = sparse_vec_similarity(x, N, True)

        contrib = projection * direction

        return sparse_sim + contrib
    else:
        ws, inds = sparse_vec_similarity(x, N, False)

        ws.append(projection)
        inds.append(random_ind)
        return (ws, inds)

def gather_col_indices(A, I):
    return tf.gather_nd(A,
        tf.transpose(tf.stack([tf.to_int64(tf.range(A.get_shape()[0])), I])))

def randomized_sparse_mat_similarity(xs, N, num_features, epsilon=0.01, dense_rep=True):
    random_inds = tf.random.uniform([num_features], 0, N, dtype=tf.int32)
    if (dense_rep):
        #Pick num_features random weights in 0, epsilon
        weights = tf.random.uniform([num_features, 1], 0, epsilon, dtype=tf.float32)

        basis_vectors = tf.one_hot(random_inds, depth=N, dtype=tf.float32, on_value=1.0, off_value=0.0)
        xs_projections = xs * basis_vectors
        #Shape num features x N

        weighted_xs_projections = weights * xs_projections

        sparse_sim = sparse_mat_similarity(xs, N, num_features, dense_rep=True)
        #shape num features x N

        return sparse_sim + weighted_xs_projections
    else:
        weights = tf.random.uniform([num_features], 0, epsilon, dtype=tf.float32)
        new_weights = gather_col_indices(xs, random_inds) * weights

        ws, inds = sparse_mat_similarity(xs, N, num_features, dense_rep=True)

        #concat our new weights and inds onto it
        ws = tf.concat(ws, tf.reshape(new_weights, [num_features, 1]), axis=-1)
        inds = tf.concat(inds, tf.reshape(random_inds, [num_features, 1]), axis=-1)

        return ws, inds


#Same as below, but on matrices of x'es, together with some optimizations
def sparse_mat_similarity(xs, N, num_features, dense_rep=True):
    #Okay, so now we have a matrix of x'es, assumed to be num_features x N
    xs_dot_with_signed_basis = tf.concat([xs, -xs], axis=-1)

    #Okay, great, now find largest dot products per feature, and their indices
    largest_dots, largest_indices = tf.math.top_k(xs_dot_with_signed_basis, k=3, sorted=True)
    #The above are now both num_features x 3
    
    #Using largest_indices as above, throw the last dimension out to get a num_features x 2
    #integer tensor
    used_indices = largest_indices[:, 0:2]

    #Construct a matrix of size num_features x 2 containing columns [w_ones, w_twos]
    #Constant matrix to multiply by to get that
    compute_op = tf.constant([[1, 0], [0, 1], [-1, -1]], dtype=tf.float32)

    ws = tf.matmul(largest_dots, compute_op)

    if (dense_rep):
        #Return results in the dense representation

        #To do this, we'll compute a num_features x 2 x N vector of vector lookups in the signed basis
        ident = tf.eye(N)
        signed_basis = tf.concat([ident, -ident], 0)

        basis_lookups = tf.gather(signed_basis, used_indices)

        #Expand ws to have a unit dimension as the last
        ws = tf.expand_dims(ws, axis=-1)

        weighted_lookups = ws * basis_lookups
        #The above is num_features x 2 x N.
        #Sum the inner dimension
        return tf.reduce_sum(weighted_lookups, axis=1)


    else:
        mod_indices = tf.mod(used_indices, N)

        mod_ws_flips = (tf.cast(used_indices < N, dtype=tf.float32) * 2.0) - 1.0

        mod_ws = tf.multiply(mod_ws_flips, mod_ws)
        
        return (mod_ws, mod_indices)


def sparse_vec_similarity(x, N, dense_rep=True):
    #Okay, so first, let's explicitly list out the (signed) basis vectors
    ident = tf.eye(N)
    signed_basis = tf.concat([ident, -ident], 0)

    #Compute dot products of the vector x with all signed basis vectors
    x_dot_with_signed_basis = tf.concat([x, -x], axis=0)

    #Okay, great. Now, we need to find the largest dot products, and their indices

    largest_dots, largest_indices = tf.math.top_k(x_dot_with_signed_basis, k=3, sorted=True)

    ind_one = largest_indices[0]
    ind_two = largest_indices[1]

    d_one = largest_dots[0]
    d_two = largest_dots[1]
    d_three = largest_dots[2]

    w_one = d_one - d_three
    w_two = d_two - d_three

    if (dense_rep):
        #Okay, now that we have the weights to give to the (signed) basis vectors, we just need
        #to extract them and add them together
        v_one = tf.gather(signed_basis, ind_one, axis=0)
        v_two = tf.gather(signed_basis, ind_two, axis=0)

        return v_one * w_one + v_two * w_two
    else:
        #In the sparse representation, we need to convert indices which are greater
        #than the threshold into 
        adj_ind_one = tf.mod(ind_one, N)
        adj_ind_two = tf.mod(ind_two, N)
        adj_w_one = tf.where(ind_one >= N, -1.0, 1.0) * w_one
        adj_w_two = tf.where(ind_two >= N, -1.0, 1.0) * w_two
        return ([adj_w_one, adj_w_two], [adj_ind_one, adj_ind_two])
