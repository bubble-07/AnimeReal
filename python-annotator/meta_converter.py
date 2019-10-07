import sys
import os
import tensorflow as tf

in_dir = sys.argv[1]
out_dir = sys.argv[2]

meta_path = os.path.join(in_dir, "RGBToCoord.meta")

output_node_names = ['RGBToCoordOut']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(in_dir))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open(os.path.join(out_dir, 'output_graph.pb'), 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
