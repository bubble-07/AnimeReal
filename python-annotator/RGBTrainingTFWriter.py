#A class responsible for writing out a collection of .TFRecord
#files associated with a particular sequence for RGB image -> template position training
#TODO: Merge with DepthTrainingTFWriter? Just copypasta for now

import os
import tensorflow as tf

NUM_EXAMPLES_PER_FILE = 100

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(rgbImage, templateIndex):
    rgbImageBytes = rgbImage.tobytes()
    templateIndexBytes = templateIndex.tobytes()
    feature = {
                'rgbImage': _bytes_feature(rgbImageBytes),
                'templateIndexImage': _bytes_feature(templateIndexBytes)
    }
    feature_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return feature_proto.SerializeToString()

class RGBTrainingTFWriter():
    def __init__(self, destDirectory):
        self.destDirectory = destDirectory
        self.fileNumber = 0
        self.fileIndex = 0
        self.rgbImageBuffer = []
        self.templateIndexBuffer = []

    def getNumElemsBuffered(self):
        return self.fileIndex


    def writeFileAndResetState(self):
        destFileName = str(self.fileNumber) + ".tfrecord"
        destFilePath = os.path.join(self.destDirectory, destFileName)

        with tf.python_io.TFRecordWriter(destFilePath) as writer:
            for rgbImage, templateIndexImage in zip(self.rgbImageBuffer, self.templateIndexBuffer):
                example = serialize_example(rgbImage, templateIndexImage)
                writer.write(example)


        self.fileNumber += 1
        self.fileIndex = 0
        self.rgbImageBuffer = []
        self.templateIndexBuffer = []

    #Add a given rgb image (np uint8 array) and template index array (np uint16 array)
    #to the depth training TFRecord
    def add(self, rgbImage, templateIndexImage):
        self.rgbImageBuffer.append(rgbImage)
        self.templateIndexBuffer.append(templateIndexImage)

        self.fileIndex += 1
        if self.fileIndex >= NUM_EXAMPLES_PER_FILE:
            self.writeFileAndResetState()

    def flush(self):
        self.writeFileAndResetState()
