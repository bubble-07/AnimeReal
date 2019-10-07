#Coincides with RGBTrainingTFWriter, but think of this more as a collection
#of utility methods to build stuff out of rather than a manager or something
#TODO: Merge with DepthTrainingTFReader

import os
import tensorflow as tf
import numpy as np

def rgbAnnotationPairsFromTFRecord(filename):
    result = []
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        line = tf.train.Example()
        line.ParseFromString(string_record)
        #Extract the depth image and template index image bytes
        rgbImageBytes = line.features.feature['rgbImage'].bytes_list.value[0]
        annotationImageBytes = line.features.feature['annotationImage'].bytes_list.value[0]


        rgbImage = np.frombuffer(rgbImageBytes, dtype=np.uint8)
        rgbImage = np.reshape(rgbImage, (256, 256, 3))

        annotationImage = np.frombuffer(annotationImageBytes, dtype=np.uint16)

        annotationImage = np.reshape(annotationImage, (256, 256))

        result.append((rgbImage, annotationImage))
    return result
