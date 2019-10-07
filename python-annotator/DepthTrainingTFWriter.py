#A class responsible for writing out a collection of .TFRecord
#files associated with a particular sequence for depth image -> template position training
import os
import tensorflow as tf

NUM_EXAMPLES_PER_FILE = 100

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(depthImage, templateIndex):
    depthImageBytes = depthImage.tobytes()
    templateIndexBytes = templateIndex.tobytes()
    feature = {
                'depthImage': _bytes_feature(depthImageBytes),
                'templateIndexImage': _bytes_feature(templateIndexBytes)
    }
    feature_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return feature_proto.SerializeToString()

class DepthTrainingTFWriter():
    def __init__(self, destDirectory):
        self.destDirectory = destDirectory
        self.fileNumber = 0
        self.fileIndex = 0
        self.depthImageBuffer = []
        self.templateIndexBuffer = []

    def getNumElemsBuffered(self):
        return self.fileIndex


    def writeFileAndResetState(self):
        destFileName = str(self.fileNumber) + ".tfrecord"
        destFilePath = os.path.join(self.destDirectory, destFileName)

        with tf.python_io.TFRecordWriter(destFilePath) as writer:
            for depthImage, templateIndexImage in zip(self.depthImageBuffer, self.templateIndexBuffer):
                example = serialize_example(depthImage, templateIndexImage)
                writer.write(example)


        self.fileNumber += 1
        self.fileIndex = 0
        self.depthImageBuffer = []
        self.templateIndexBuffer = []

    #Add a given depth image (np float32 array) and template index array (np uint16 array)
    #to the depth training TFRecord
    def add(self, depthImage, templateIndexImage):
        self.depthImageBuffer.append(depthImage)
        self.templateIndexBuffer.append(templateIndexImage)

        self.fileIndex += 1
        if self.fileIndex >= NUM_EXAMPLES_PER_FILE:
            self.writeFileAndResetState()

    def flush(self):
        self.writeFileAndResetState()
