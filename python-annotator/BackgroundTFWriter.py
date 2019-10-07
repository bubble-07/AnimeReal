#A class responsible for writing out a collection of .TFRecord
#files associated with background images
#Each background image is designated to be of size
#2145x1428
#TODO: Merge with DepthTrainingTFWriter? Just copypasta for now

import os
import tensorflow as tf

NUM_EXAMPLES_PER_FILE = 100

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(rgbImage):
    rgbImageBytes = rgbImage.tobytes()
    feature = {
                'rgbImage': _bytes_feature(rgbImageBytes)
    }
    feature_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return feature_proto.SerializeToString()

class BackgroundTFWriter():
    def __init__(self, destDirectory):
        self.destDirectory = destDirectory
        self.fileNumber = 0
        self.fileIndex = 0
        self.rgbImageBuffer = []

    def getNumElemsBuffered(self):
        return self.fileIndex


    def writeFileAndResetState(self):
        destFileName = str(self.fileNumber) + ".tfrecord"
        destFilePath = os.path.join(self.destDirectory, destFileName)

        with tf.python_io.TFRecordWriter(destFilePath) as writer:
            for rgbImage in self.rgbImageBuffer:
                example = serialize_example(rgbImage)
                writer.write(example)


        self.fileNumber += 1
        self.fileIndex = 0
        self.rgbImageBuffer = []

    #Add a given rgb image (np uint8 array)
    #to the depth training TFRecord
    def add(self, rgbImage):
        self.rgbImageBuffer.append(rgbImage)

        self.fileIndex += 1
        if self.fileIndex >= NUM_EXAMPLES_PER_FILE:
            self.writeFileAndResetState()

    def flush(self):
        self.writeFileAndResetState()
