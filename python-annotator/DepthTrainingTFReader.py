#Coincides with DepthTrainingTFWriter, but think of this more as a collection
#of utility methods to build stuff out of rather than a manager or something

import os
import tensorflow as tf
import numpy as np

def depthFrameLabelPairsFromTFRecord(filename):
    result = []
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        line = tf.train.Example()
        line.ParseFromString(string_record)
        #Extract the depth image and template index image bytes
        depthImageBytes = line.features.feature['depthImage'].bytes_list.value[0]
        templateIndexImageBytes = line.features.feature['templateIndexImage'].bytes_list.value[0]


        depthImage = np.frombuffer(depthImageBytes, dtype=np.float32)
        depthImage = np.reshape(depthImage, (424, 512))

        templateIndexImage = np.frombuffer(templateIndexImageBytes, dtype=np.uint16)
        templateIndexImage = np.reshape(templateIndexImage, (424, 512))

        result.append((depthImage, templateIndexImage))
    return result
