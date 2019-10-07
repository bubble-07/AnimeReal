#Coincides with RGBTrainingTFWriter, but think of this more as a collection
#of utility methods to build stuff out of rather than a manager or something
#TODO: Merge with DepthTrainingTFReader

import os
import tensorflow as tf
import numpy as np

def rgbFrameLabelPairsFromTFRecord(filename):
    result = []
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        line = tf.train.Example()
        line.ParseFromString(string_record)
        #Extract the depth image and template index image bytes
        rgbImageBytes = line.features.feature['rgbImage'].bytes_list.value[0]
        templateIndexImageBytes = line.features.feature['templateIndexImage'].bytes_list.value[0]


        rgbImage = np.frombuffer(rgbImageBytes, dtype=np.uint8)
        rgbImage = np.reshape(rgbImage, (424, 512, 3))

        templateIndexImage = np.frombuffer(templateIndexImageBytes, dtype=np.uint16)
        templateIndexImage = np.reshape(templateIndexImage, (424, 512))

        result.append((rgbImage, templateIndexImage))
    return result
