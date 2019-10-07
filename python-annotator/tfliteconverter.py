import tensorflow as tf
import sys

dest_model_file = "./RGBToCoordTFLite/pose_annotator_slim.tflite"
graph_dump_folder = "./RGBGraphViz"
graph_src_dir = "./RGBToCoordSavedModel/SavedModel"

converter = tf.contrib.lite.TFLiteConverter.from_saved_model(graph_src_dir, 
            input_arrays=["RGBToCoordIn"], 
            input_shapes={"RGBToCoordIn" : [1, 256, 256, 3]}, 
            output_arrays=["RGBToCoordOut"])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.inference_input_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"RGBToCoordIn" : (0.0, 42.5)}
converter.change_concat_input_ranges = False
converter.dump_graphviz_dir = graph_dump_folder
tflite_quantized_model = converter.convert()
open(dest_model_file, "wb").write(tflite_quantized_model)
